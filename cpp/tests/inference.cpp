#include "inference.hpp"

#include "instrument_groups.hpp"
#include "open_note_tracker.hpp"
#include "vocabulary.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

namespace msl::test
{

namespace
{

    std::int32_t argmaxOf(const std::vector<float>& inLogits)
    {
        return static_cast<std::int32_t>(
            std::distance(inLogits.begin(), std::max_element(inLogits.begin(), inLogits.end())));
    }

    /** `inTokens` with a trailing EOS dropped, which is how the note-level dumps store them. */
    std::vector<std::int32_t> withoutEos(const std::vector<std::int32_t>& inTokens)
    {
        const std::int32_t eos = Reference::fp32().eos_id();

        if (!inTokens.empty() && inTokens.back() == eos) {
            return {inTokens.begin(), inTokens.end() - 1};
        }

        return inTokens;
    }

    /**
     * Prefill with the initial token and `inPrompt`, then feed `inWant` one token
     * at a time, recording the argmax each position would have picked, and
     * finally the position that should produce EOS. That is the same sequence
     * of forward passes greedy decoding runs when it agrees with the reference.
     *
     * @param inWant The chunk's reference tokens, prompt included, EOS excluded.
     */
    std::unique_ptr<ForcedChunk> runForced(std::span<const float> inConditioning,
                                           const std::vector<std::int32_t>& inPrompt,
                                           const std::vector<std::int32_t>& inWant)
    {
        const Reference& ref = Reference::fp32();
        Model& model = shared_model();

        auto out = std::make_unique<ForcedChunk>();
        out->prompt = inPrompt;

        model.reset();
        std::vector<std::int32_t> prefill_tokens {ref.initial_token_id()};
        prefill_tokens.insert(prefill_tokens.end(), inPrompt.begin(), inPrompt.end());

        std::vector<float> logits = model.prefill(inConditioning, ref.mel_frames(), prefill_tokens);

        const auto record = [&](std::int32_t inToken) {
            ForcedStep step;
            step.argmax = argmaxOf(logits);
            step.want = inToken;
            step.margin = logits[static_cast<std::size_t>(step.argmax)] - logits[static_cast<std::size_t>(inToken)];
            out->steps.push_back(step);
        };

        const auto full_steps = static_cast<std::size_t>(ref.max_decode_steps());

        for (std::size_t i = inPrompt.size(); i < inWant.size(); ++i) {
            record(inWant[i]);
            logits = model.decode(inWant[i]);

            if (out->decode_logits.size() < full_steps) {
                out->decode_logits.push_back(logits);
                out->n_past_after_decode.push_back(model.nPast());
            }
        }

        record(ref.eos_id());
        return out;
    }

    using ForcedCache = std::map<std::pair<std::string, int>, std::unique_ptr<ForcedChunk>>;

    const ForcedChunk&
        cachedForced(const std::string& inKey, int inChunk, const std::function<std::unique_ptr<ForcedChunk>()>& inMake)
    {
        static PerConfig<ForcedCache> cache;
        ForcedCache& runs = cache.get([] { return std::make_unique<ForcedCache>(); });
        std::unique_ptr<ForcedChunk>& slot = runs[{inKey, inChunk}];

        if (!slot) {
            slot = inMake();
        }

        return *slot;
    }

    std::vector<float> chunkConditioning(int inChunk)
    {
        const Reference& ref = Reference::fp32();
        return shared_model().encodeConditioning(
            ref.tensor(msl::format("in.chunk{}.spectrum", inChunk)), ref.mel_frames(), ref.segment_samples());
    }

    /** The prologue prelude forcing builds at `inChunk`'s boundary from the reference's earlier chunks. */
    std::vector<std::int32_t> forcedPrologue(const VariantReference& inVariant, int inChunk)
    {
        if (!inVariant.prelude_forcing || inChunk == 0) {
            return {};
        }

        OpenNoteTracker tracker;
        const std::vector<ChunkBoundary> boundaries = inVariant.boundaries();
        const auto chunk = static_cast<std::size_t>(inChunk);

        for (std::size_t c = 0; c < chunk; ++c) {
            tracker.feed(boundaries[c]);

            for (const std::int32_t token: inVariant.tokens[c]) {
                tracker.feed(token);
            }
        }

        tracker.feed(boundaries[chunk]);
        return Vocabulary::tieSectionTokenIds(tracker.openKeys());
    }

} // namespace

std::vector<std::size_t> ForcedChunk::disagreements() const
{
    std::vector<std::size_t> out;

    for (std::size_t i = 0; i < steps.size(); ++i) {
        if (steps[i].argmax != steps[i].want) {
            out.push_back(i);
        }
    }

    return out;
}

const ForcedChunk& plainChunk(int inChunk)
{
    return cachedForced("", inChunk, [inChunk] {
        const Reference& ref = Reference::fp32();
        const std::vector<float> conditioning = inChunk == 0 ? ref.tensor("cond.embed") : chunkConditioning(inChunk);

        return runForced(conditioning, {}, withoutEos(ref.tokens(inChunk)));
    });
}

const ForcedChunk& variantChunk(const std::string& inVariant, int inChunk)
{
    const VariantReference& variant = requireReferences().variant(inVariant);
    const auto chunk = static_cast<std::size_t>(inChunk);
    REQUIRE(chunk < variant.tokens.size());

    std::vector<std::int32_t> prompt = forcedPrologue(variant, inChunk);

    if (variant.instruments.empty() && prompt.empty()
        && variant.tokens[chunk] == withoutEos(Reference::fp32().tokens(inChunk))) {
        return plainChunk(inChunk);
    }

    return cachedForced(inVariant, inChunk, [&] {
        Model& model = shared_model();
        const std::vector<InstrumentGroup> groups = groupsFor(variant);
        const ScopedSelection selection(model, groups);

        return runForced(chunkConditioning(inChunk), prompt, variant.tokens[chunk]);
    });
}

const TracedPrefill& tracedPrefill(const std::string& inVariant)
{
    using Cache = std::map<std::string, std::unique_ptr<TracedPrefill>>;
    static PerConfig<Cache> cache;
    std::unique_ptr<TracedPrefill>& slot = cache.get([] { return std::make_unique<Cache>(); })[inVariant];

    if (slot) {
        return *slot;
    }

    const Reference& ref = Reference::fp32();
    Model& model = shared_model();
    const std::array<std::int32_t, 1> prompt {ref.initial_token_id()};
    auto out = std::make_unique<TracedPrefill>();

    if (inVariant.empty()) {
        model.reset();
        out->logits = model.prefill(ref.tensor("cond.embed"), ref.mel_frames(), prompt, &out->trace);
    }

    else {
        const std::vector<InstrumentGroup> groups = groupsFor(requireReferences().variant(inVariant));
        const ScopedSelection selection(model, groups);
        const std::vector<float> conditioning = chunkConditioning(0);

        model.reset();
        out->logits = model.prefill(conditioning, ref.mel_frames(), prompt, &out->trace);
    }

    out->n_past = model.nPast();
    slot = std::move(out);
    return *slot;
}

const std::vector<float>& fixtureSignal()
{
    static PerConfig<std::vector<float>> signal;
    return signal.get([] {
        const Reference& ref = Reference::fp32();
        auto out = std::make_unique<std::vector<float>>();

        for (int chunk = 0; chunk < ref.num_chunks(); ++chunk) {
            const std::vector<float>& wav = ref.tensor(msl::format("in.chunk{}.wav", chunk));
            out->insert(out->end(), wav.begin(), wav.end());
        }

        return out;
    });
}

Transcriber& sharedTranscriber()
{
    static PerConfig<Transcriber> instance;
    return instance.get([] {
        LoadOptions options;
        options.use_gpu = gpu_enabled();
        std::expected<Transcriber, Error> loaded = Transcriber::load(Reference::fp32().weights_path(), options);

        if (!loaded.has_value()) {
            throw std::runtime_error(msl::format("Transcriber::load failed: {}", describe(loaded.error())));
        }

        return std::make_unique<Transcriber>(std::move(*loaded));
    });
}

const Transcription& transcription(const std::string& inVariant)
{
    using Cache = std::map<std::string, std::unique_ptr<Transcription>>;
    static PerConfig<Cache> cache;
    std::unique_ptr<Transcription>& slot = cache.get([] { return std::make_unique<Cache>(); })[inVariant];

    if (slot) {
        return *slot;
    }

    const VariantReference& variant = requireReferences().variant(inVariant);
    auto out = std::make_unique<Transcription>();
    out->options.prelude_forcing = variant.prelude_forcing;
    out->options.instruments = groupsFor(variant);

    out->notes =
        sharedTranscriber().transcribe(fixtureSignal(), out->options, [&](const TranscriptionUpdate& inUpdate) {
            RecordedUpdate update;
            update.new_notes.assign(inUpdate.new_notes.begin(), inUpdate.new_notes.end());
            update.finalized_through = inUpdate.finalized_through;
            update.progress = inUpdate.progress;
            out->updates.push_back(std::move(update));
            return true;
        });

    slot = std::move(out);
    return *slot;
}

std::vector<InstrumentGroup> groupsFor(const VariantReference& inVariant)
{
    std::vector<InstrumentGroup> groups;

    for (const std::string& name: inVariant.instruments) {
        const std::optional<InstrumentGroup> group = InstrumentGroups::groupForName(name);
        INFO("instrument " << name);
        REQUIRE(group.has_value());
        groups.push_back(*group);
    }

    return groups;
}

double logitNoiseFloor()
{
    return Reference::fp32().tolerance("dec.step1.logits").atol;
}

int requireOnlyNearTies(const ForcedChunk& inRun)
{
    const double noise_floor = logitNoiseFloor();
    const std::vector<std::size_t> disagreements = inRun.disagreements();

    for (const std::size_t index: disagreements) {
        const ForcedStep& step = inRun.steps[index];
        INFO("step " << inRun.prompt.size() + index << ": got " << step.argmax << ", reference " << step.want
                     << ", margin " << step.margin << " against a noise floor of " << noise_floor);
        REQUIRE(step.margin <= noise_floor);
    }

    return static_cast<int>(disagreements.size());
}

ScopedSelection::ScopedSelection(Model& inModel, std::span<const InstrumentGroup> inGroups)
    : mModel(inModel)
{
    if (!inGroups.empty()) {
        mModel.setInstrumentRows(InstrumentGroups::conditioningRows(inGroups));
        mModel.setForbiddenTokens(InstrumentGroups::forbiddenTokenIds(inGroups));
    }
}

ScopedSelection::~ScopedSelection()
{
    mModel.setInstrumentRows({});
    mModel.setForbiddenTokens({});
}

} // namespace msl::test
