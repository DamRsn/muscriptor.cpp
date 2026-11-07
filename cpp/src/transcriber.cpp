#include "muscriptor/transcriber.hpp"

#include "muscriptor/model.hpp"

#include "format.hpp"
#include "instrument_groups.hpp"
#include "log.hpp"
#include "note_assembler.hpp"
#include "open_note_tracker.hpp"
#include "resume_point.hpp"
#include "silence_gate.hpp"
#include "vocabulary.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iterator>
#include <new>
#include <optional>
#include <string>
#include <string_view>

namespace msl
{

namespace
{

    /**
     * KV positions to allocate.
     *
     * The prefix (mel frames, one dataset row, one row per selected instrument),
     * the initial token, and a chunk's worth of generation. Computed rather than
     * written down so it cannot go stale against the front-end constants.
     */
    constexpr int requiredContext(int inMelFrames)
    {
        return inMelFrames + 1 + InstrumentGroups::MAX_SELECTABLE + 1 + Transcriber::MAX_TOKENS_PER_CHUNK;
    }

    /** Mel frames one chunk produces: centre padding yields one more than it covers. */
    constexpr int melFramesPerChunk(int inHopLength)
    {
        return 1 + Transcriber::SEGMENT_SAMPLES / inHopLength;
    }

    /** The KV positions `load` allocates. The hparams are checked against these constants after loading. */
    constexpr int CHUNK_CONTEXT = requiredContext(melFramesPerChunk(Transcriber::SAMPLE_RATE / Vocabulary::FRAME_RATE));

    /** The selection, with duplicates removed but the caller's order kept. */
    std::vector<InstrumentGroup> uniqueInstruments(std::span<const InstrumentGroup> inGroups)
    {
        std::vector<InstrumentGroup> unique;

        for (const InstrumentGroup group: inGroups) {
            if (std::find(unique.begin(), unique.end(), group) == unique.end()) {
                unique.push_back(group);
            }
        }

        return unique;
    }

    constexpr int FRAME_SAMPLES = Transcriber::SAMPLE_RATE / Vocabulary::FRAME_RATE;
    constexpr int SEGMENT_FRAMES = Transcriber::SEGMENT_SAMPLES / FRAME_SAMPLES;

    /** @return The lead-in in whole frames; 0 when it is off. */
    int leadInFrames(const LeadInOptions& inLeadIn)
    {
        if (!inLeadIn.enabled || !(inLeadIn.duration > 0.0)) {
            return 0;
        }

        return static_cast<int>(
            std::llround(std::min(inLeadIn.duration, Transcriber::SEGMENT_DURATION) * Vocabulary::FRAME_RATE));
    }

    CorrectionSettings correctionSettings(const TranscribeOptions& inOptions)
    {
        CorrectionSettings settings;
        settings.lead_in_frames = leadInFrames(inOptions.lead_in);

        if (inOptions.silence_gate.enabled) {
            settings.silence_gate = true;
            settings.gate_threshold = inOptions.silence_gate.threshold_dbfs;
            settings.gate_min_duration = inOptions.silence_gate.min_duration;
            settings.gate_release = inOptions.silence_gate.release;
        }

        return settings;
    }

    /** Throws `Error::InvalidArgument` for a lead-in or gate setting out of range. */
    void validateCorrections(const TranscribeOptions& inOptions)
    {
        const LeadInOptions& lead_in = inOptions.lead_in;

        if (lead_in.enabled && !(lead_in.duration >= 0.0 && lead_in.duration <= Transcriber::SEGMENT_DURATION)) {
            throw Exception(
                Error::InvalidArgument,
                msl::format("lead-in of {} s is outside 0 to {} s", lead_in.duration, Transcriber::SEGMENT_DURATION));
        }

        const SilenceGateOptions& gate = inOptions.silence_gate;

        if (gate.enabled
            && !(std::isfinite(gate.threshold_dbfs) && std::isfinite(gate.min_duration) && gate.min_duration >= 0.0
                 && std::isfinite(gate.release) && gate.release >= 0.0)) {
            throw Exception(Error::InvalidArgument,
                            "silence gate needs a finite threshold and a finite, non-negative duration and release");
        }
    }

    /** Installs a cancel predicate on a model for one call, and removes it on every exit. */
    class ScopedCancel
    {
    public:
        ScopedCancel(Model& ioModel, const CancelPredicate& inShouldCancel)
            : mModel(ioModel)
        {
            mModel.setShouldCancel(inShouldCancel);
        }

        ~ScopedCancel() { mModel.setShouldCancel({}); }

        ScopedCancel(const ScopedCancel&) = delete;
        ScopedCancel& operator=(const ScopedCancel&) = delete;

    private:
        Model& mModel;
    };

} // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

class Transcriber::Impl
{
public:
    explicit Impl(Model&& inModel)
        : mModel(std::move(inModel))
        , mMelFrames(melFramesPerChunk(mModel.hparams().hop_length))
    {
    }

    std::expected<std::vector<Note>, Error> transcribe(std::span<const float> inSamples,
                                                       const TranscribeOptions& inOptions,
                                                       const NoteCallback& inCallback);

    const Device& device() const { return mModel.device(); }

private:
    /**
     * Configure the model from `inOptions`. Throws `Error::InvalidArgument` on
     * a bad selection, having touched nothing -- the model is left as the
     * previous call configured it, and the next call reconfigures it from
     * scratch anyway.
     */
    void _configure(const TranscribeOptions& inOptions);

    /**
     * Run the model on chunk `inChunk`, after its boundary has been fed, and
     * feed what it decodes.
     */
    void _decodeChunk(std::span<const float> inSamples, const TranscribeOptions& inOptions, int inChunk);

    /** One chunk of the lead-in followed by the signal, zero-padded to a full segment. */
    void _fillChunk(std::span<const float> inSamples, int inIndex);

    /** @return Where chunk `inChunk` starts, in seconds into the signal: the lead-in puts it that much earlier. */
    double _seekTime(int inChunk) const;

    /** @return Decoded notes gated, clamped to the start of the signal, and sorted. */
    std::vector<Note> _release(const std::vector<Note>& inNotes) const;

    /**
     * @return `finalized_through` after the chunk ending at `inTime`: that, or
     *         less if the gate may still cut a note not reported yet.
     */
    double _finalizedThrough(double inTime, int inChunk) const;

    /**
     * Rebuild the decode state `inResumePoint` describes, after `_configure`.
     * @return The chunk to continue from. Throws `Error::InvalidResumePoint`
     *         if the point does not fit this call.
     */
    int _restore(std::string_view inResumePoint,
                 std::size_t inNSamples,
                 const TranscribeOptions& inOptions,
                 int inNChunks);

    /** The resume point for right after chunk `inDecoded - 1`. */
    std::string _resumePoint(std::size_t inNSamples, const TranscribeOptions& inOptions, int inDecoded) const;

    Model mModel;
    int mMelFrames;

    OpenNoteTracker mTracker;
    NoteAssembler mAssembler;
    std::vector<float> mChunk;
    std::vector<InstrumentGroup> mInstruments;

    CorrectionSettings mCorrections;
    SilenceGate mGate;
};

void Transcriber::Impl::_configure(const TranscribeOptions& inOptions)
{
    validateCorrections(inOptions);

    mInstruments = uniqueInstruments(inOptions.instruments);

    for (const InstrumentGroup group: mInstruments) {
        if (instrumentName(group).empty()) {
            throw Exception(Error::InvalidArgument,
                            msl::format("instrument group {} is not a named group", static_cast<int>(group)));
        }
    }

    mCorrections = correctionSettings(inOptions);
    mModel.setNumThreads(inOptions.n_threads);
    mModel.setInstrumentRows(InstrumentGroups::conditioningRows(mInstruments));

    // Guarded rather than called unconditionally: the reference forbids every
    // program and every drum for an empty selection, so passing one straight
    // through would turn "no filter" into "no instruments".
    if (mInstruments.empty()) {
        mModel.setForbiddenTokens({});
    } else {
        mModel.setForbiddenTokens(InstrumentGroups::forbiddenTokenIds(mInstruments));
    }
}

int Transcriber::Impl::_restore(std::string_view inResumePoint,
                                std::size_t inNSamples,
                                const TranscribeOptions& inOptions,
                                int inNChunks)
{
    std::optional<ResumeState> state = parseResumeState(inResumePoint);

    if (!state) {
        throw Exception(Error::InvalidResumePoint, "malformed resume point");
    }

    if (state->n_samples != inNSamples || state->prelude_forcing != inOptions.prelude_forcing
        || state->instruments != mInstruments || state->corrections != mCorrections || state->decoded > inNChunks) {
        throw Exception(Error::InvalidResumePoint,
                        "resume point made for another signal, selection, prelude, lead-in or silence gate setting");
    }

    const int last = state->decoded - 1;

    // Replayed through the same actions decoding produced, so the assembler
    // ends up holding exactly what it held then.
    for (const TrackedNote& tracked: state->withheld) {
        const NoteKey& key = tracked.key;

        if (tracked.drum_hit) {
            mAssembler.apply(std::vector {NoteAction {NoteActionKind::DrumHit, 0, key.pitch, tracked.note.onset}},
                             last);
        } else {
            mAssembler.apply(
                std::vector {NoteAction {NoteActionKind::Start, key.program, key.pitch, tracked.note.onset},
                             NoteAction {NoteActionKind::End, key.program, key.pitch, tracked.note.offset}},
                last);
        }
    }

    for (const OpenNoteTracker::OpenNote& note: state->open) {
        mAssembler.apply(std::vector {NoteAction {NoteActionKind::Start, note.key.program, note.key.pitch, note.onset}},
                         last);
    }

    mTracker.restore(std::move(state->open), _seekTime(last), state->in_prologue);

    return state->decoded;
}

std::string
    Transcriber::Impl::_resumePoint(std::size_t inNSamples, const TranscribeOptions& inOptions, int inDecoded) const
{
    ResumeState state;
    state.n_samples = inNSamples;
    state.prelude_forcing = inOptions.prelude_forcing;
    state.instruments = mInstruments;
    state.corrections = mCorrections;
    state.decoded = inDecoded;
    state.in_prologue = mTracker.inPrologue();
    state.open = mTracker.openNotes();

    // Earlier chunks' notes have all been reported, and nothing later depends on them.
    std::copy_if(mAssembler.closedNotes().begin(),
                 mAssembler.closedNotes().end(),
                 std::back_inserter(state.withheld),
                 [inDecoded](const TrackedNote& n) { return n.chunk_index == inDecoded - 1; });

    return serializeResumeState(state);
}

void Transcriber::Impl::_fillChunk(std::span<const float> inSamples, int inIndex)
{
    mChunk.assign(static_cast<std::size_t>(SEGMENT_SAMPLES), 0.0f);

    // Positions in the lead-in followed by the signal.
    const std::size_t lead_in = static_cast<std::size_t>(mCorrections.lead_in_frames) * FRAME_SAMPLES;
    const std::size_t chunk_start = static_cast<std::size_t>(inIndex) * SEGMENT_SAMPLES;
    const std::size_t chunk_end = chunk_start + SEGMENT_SAMPLES;
    const std::size_t from = std::max(chunk_start, lead_in);
    const std::size_t to = std::min(chunk_end, lead_in + inSamples.size());

    if (from < to) {
        std::copy(inSamples.begin() + static_cast<std::ptrdiff_t>(from - lead_in),
                  inSamples.begin() + static_cast<std::ptrdiff_t>(to - lead_in),
                  mChunk.begin() + static_cast<std::ptrdiff_t>(from - chunk_start));
    }
}

double Transcriber::Impl::_seekTime(int inChunk) const
{
    return static_cast<double>(inChunk * SEGMENT_FRAMES - mCorrections.lead_in_frames) / Vocabulary::FRAME_RATE;
}

std::vector<Note> Transcriber::Impl::_release(const std::vector<Note>& inNotes) const
{
    std::vector<Note> released;

    for (const Note& note: inNotes) {
        std::optional<Note> gated = mGate.apply(note);

        if (!gated || gated->onset < -ONSET_LEAD) {
            continue;
        }

        if (gated->onset < 0.0) {
            gated->onset = 0.0;
            gated->offset = std::max(gated->offset, MINIMUM_NOTE_DURATION_SECONDS);
        }

        released.push_back(*gated);
    }

    sortNotes(released);
    return released;
}

double Transcriber::Impl::_finalizedThrough(double inTime, int inChunk) const
{
    double horizon = inTime;

    for (const OpenNoteTracker::OpenNote& open: mTracker.openNotes()) {
        horizon = std::min(horizon, mGate.cutPoint(open.onset));
    }

    for (const TrackedNote& tracked: mAssembler.closedNotes()) {
        if (tracked.chunk_index == inChunk) {
            horizon = std::min(horizon, mGate.cutPoint(tracked.note.onset));
        }
    }

    return std::max(horizon, 0.0);
}

std::expected<std::vector<Note>, Error> Transcriber::Impl::transcribe(std::span<const float> inSamples,
                                                                      const TranscribeOptions& inOptions,
                                                                      const NoteCallback& inCallback)
{
    // At the top, not the bottom: a previous call that was cancelled or threw
    // must not leave open notes for this one to inherit.
    mTracker.reset();
    mAssembler.reset();

    _configure(inOptions);

    mGate = inOptions.silence_gate.enabled ? SilenceGate(inSamples, inOptions.silence_gate) : SilenceGate();

    const int n_chunks = Transcriber::chunkCount(inSamples.size(), inOptions.lead_in);
    int first_chunk = 0;

    if (!inOptions.resume_from.empty()) {
        first_chunk = _restore(inOptions.resume_from, inSamples.size(), inOptions, n_chunks);
    }

    const ScopedCancel scoped_cancel(mModel, inOptions.should_cancel);

    for (int chunk = first_chunk; chunk < n_chunks; ++chunk) {
        if (inOptions.should_cancel && inOptions.should_cancel()) {
            return std::unexpected(Error::Cancelled);
        }

        ChunkBoundary boundary;
        boundary.seek_time = _seekTime(chunk);

        if (chunk + 1 < n_chunks) {
            boundary.next_seek_time = _seekTime(chunk + 1);
        }

        // Feed the boundary before reading open keys: it settles a previous
        // chunk that ended mid-prologue, and only then is openKeys() the
        // decoder's own view.
        mAssembler.apply(mTracker.feed(boundary), chunk);

        if (mGate.covers(boundary.seek_time, boundary.seek_time + SEGMENT_DURATION)) {
            // Decoded as an empty tie section, which closes every open note,
            // instead of letting the model carry them on through the silence.
            mAssembler.apply(mTracker.feed(Vocabulary::TIE_FIRST), chunk);
        } else {
            _decodeChunk(inSamples, inOptions, chunk);
        }

        if (inCallback) {
            // One chunk late: a note is only final once the following chunk has
            // decoded, so releasing it now would mean correcting it later.
            const std::vector<Note> ready = chunk > 0 ? _release(mAssembler.closedIn(chunk - 1)) : std::vector<Note> {};

            TranscriptionUpdate update;
            update.new_notes = ready;
            // Chunks 0 through chunk-1 have been released, and each of them
            // closed only notes ending inside its own window.
            update.finalized_through = _finalizedThrough(boundary.seek_time, chunk);
            update.progress = static_cast<float>(chunk + 1) / static_cast<float>(n_chunks);
            update.resume_point = _resumePoint(inSamples.size(), inOptions, chunk + 1);

            if (!inCallback(update)) {
                return std::unexpected(Error::Cancelled);
            }
        }
    }

    if (n_chunks > 0) {
        mAssembler.apply(mTracker.finish(), n_chunks - 1);

        if (inCallback) {
            // The withheld tail. Notes finish() closes were open throughout, so
            // nothing can have opened on their channel to trim an earlier chunk's --
            // this call cannot invalidate anything already reported.
            const std::vector<Note> tail = _release(mAssembler.closedIn(n_chunks - 1));

            TranscriptionUpdate update;
            update.new_notes = tail;
            // An under-claim: the last chunk has no window past it, so a note
            // there may end fractionally later. Claiming less than is known
            // costs a host nothing; claiming more would cost it a note.
            update.finalized_through = std::max(_seekTime(n_chunks), 0.0);
            update.progress = 1.0f;

            if (!inCallback(update)) {
                return std::unexpected(Error::Cancelled);
            }
        }
    }

    return _release(mAssembler.finalize());
}

void Transcriber::Impl::_decodeChunk(std::span<const float> inSamples, const TranscribeOptions& inOptions, int inChunk)
{
    _fillChunk(inSamples, inChunk);

    // The tail is zero-padded and the padding is *not* masked away: the
    // conditioner is handed a full segment, so the model sees the trailing
    // silence as audio. That is what the reference does.
    const std::vector<float> conditioning = mModel.encodeAudio(mChunk);

    std::vector<std::int32_t> prompt;

    if (inChunk > 0 && inOptions.prelude_forcing) {
        prompt = Vocabulary::tieSectionTokenIds(mTracker.openKeys());
    }

    // Clamped rather than allowed to overflow. A pathological prologue eats
    // into the generation budget, which is the right trade: the reference
    // treats running out without an EOS as a warning, not a failure.
    const int prefix = mMelFrames + 1 + static_cast<int>(mModel.instrumentRows().size());
    const int budget = mModel.contextSize() - prefix - 1;

    if (budget <= static_cast<int>(prompt.size())) {
        throw Exception(
            Error::ContextOverflow,
            msl::format("chunk {}: a {}-token prompt leaves no room after a {}-row prefix in a {}-position context",
                        inChunk,
                        prompt.size(),
                        prefix,
                        mModel.contextSize()));
    }

    const int max_tokens = std::min(Transcriber::MAX_TOKENS_PER_CHUNK, budget);

    const std::vector<std::int32_t> tokens =
        mModel.generate(conditioning, mMelFrames, max_tokens, Vocabulary::EOS_ID, prompt);

    for (const std::int32_t token: tokens) {
        if (token == Vocabulary::EOS_ID) {
            break;
        }

        mAssembler.apply(mTracker.feed(token), inChunk);
    }
}

// ---------------------------------------------------------------------------
// Transcriber
// ---------------------------------------------------------------------------

namespace
{

    /**
     * @return The error a caught exception carries, after logging what it says:
     *         the returned code alone does not say why `inCall` failed.
     */
    Error caught(std::string_view inCall, const std::exception_ptr& inException)
    {
        Error error = Error::Internal;
        std::string reason;

        try {
            std::rethrow_exception(inException);
        } catch (const Exception& e) {
            error = e.error();
            reason = e.what();
        } catch (const std::bad_alloc& e) {
            error = Error::OutOfMemory;
            reason = e.what();
        } catch (const std::exception& e) {
            reason = e.what();
        } catch (...) {
            reason = "unknown exception";
        }

        if (error != Error::Cancelled) {
            logMessage(LogLevel::Error, msl::format("{} failed ({}): {}", inCall, describe(error), reason));
        }

        return error;
    }

} // namespace

Transcriber::Transcriber() = default;
Transcriber::~Transcriber() = default;
Transcriber::Transcriber(Transcriber&&) noexcept = default;
Transcriber& Transcriber::operator=(Transcriber&&) noexcept = default;

int Transcriber::chunkCount(std::size_t inNSamples, const LeadInOptions& inLeadIn)
{
    if (inNSamples == 0) {
        return 0;
    }

    const std::size_t padded = inNSamples + static_cast<std::size_t>(leadInFrames(inLeadIn)) * FRAME_SAMPLES;
    return static_cast<int>((padded + SEGMENT_SAMPLES - 1) / SEGMENT_SAMPLES);
}

std::expected<Transcriber, Error> Transcriber::load(const std::filesystem::path& inGgufPath, LoadOptions inOptions)
{
    try {
        ModelOptions options;
        options.device = inOptions.device;
        options.should_cancel = std::move(inOptions.should_cancel);
        options.on_progress = std::move(inOptions.on_progress);
        options.n_ctx = CHUNK_CONTEXT;

        Model model = Model::load(inGgufPath, options);
        const Hparams& hp = model.hparams();

        // This library is 16 kHz and 10 ms only: both are baked into the
        // segment length and into the decode state machine's tick arithmetic.
        if (hp.sample_rate != SAMPLE_RATE || hp.frame_rate != Vocabulary::FRAME_RATE
            || hp.hop_length != SAMPLE_RATE / Vocabulary::FRAME_RATE) {
            throw Exception(
                Error::UnsupportedArch,
                msl::format("audio framing {} Hz, {} frames/s, hop {}", hp.sample_rate, hp.frame_rate, hp.hop_length));
        }

        Transcriber transcriber;
        transcriber.mImpl = std::make_unique<Impl>(std::move(model));
        return transcriber;
    } catch (...) {
        return std::unexpected(caught("load", std::current_exception()));
    }
}

std::expected<std::vector<std::optional<DeviceSupport>>, Error>
    Transcriber::checkDevices(const std::filesystem::path& inGgufPath, std::optional<std::size_t> inDevice)
{
    try {
        ModelOptions options;
        options.device = inDevice;
        options.n_ctx = CHUNK_CONTEXT;
        return Model::checkDevices(inGgufPath, options);
    } catch (...) {
        return std::unexpected(caught("checkDevices", std::current_exception()));
    }
}

const Device& Transcriber::device() const
{
    return mImpl->device();
}

std::expected<std::vector<Note>, Error> Transcriber::transcribe(std::span<const float> inSamples,
                                                                const TranscribeOptions& inOptions,
                                                                const NoteCallback& inCallback)
{
    try {
        return mImpl->transcribe(inSamples, inOptions, inCallback);
    } catch (...) {
        return std::unexpected(caught("transcribe", std::current_exception()));
    }
}

} // namespace msl
