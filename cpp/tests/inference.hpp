#pragma once

// The expensive part of the suite, run once per configuration and shared.
//
// Every model pass the tests compare against is computed lazily, the first time
// a test case asks for it, and kept until the configuration changes. Test cases
// assert over the recorded result instead of re-running the transformer, so a
// chunk that several tests check is decoded once.

#include "reference.hpp"
#include "vectors.hpp"

#include "muscriptor/muscriptor.hpp"
#include "trace.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace msl::test
{

/** One decode position, teacher-forced: what the port picked and what the reference emitted. */
struct ForcedStep {
    std::int32_t argmax = 0;
    std::int32_t want = 0;
    // logits[argmax] - logits[want]; zero whenever the two agree.
    double margin = 0.0;
};

/**
 * One chunk decoded on the reference's own tokens: the prefill, then one decode
 * per reference token. Feeding the reference's token rather than the port's
 * argmax isolates each step, so a disagreement shows up at the step it starts
 * instead of cascading. When every step agrees, greedy decoding feeds exactly
 * these tokens and so reproduces the reference stream.
 */
struct ForcedChunk {
    std::vector<std::int32_t> prompt;

    // One per generated position; the last one's `want` is EOS.
    std::vector<ForcedStep> steps;

    // The first Reference::max_decode_steps() decodes in full: entry k holds the
    // logits and the position count after decode step k + 1.
    std::vector<std::vector<float>> decode_logits;
    std::vector<int> n_past_after_decode;

    /** @return Indices into `steps` where the port's argmax is not the reference's token. */
    std::vector<std::size_t> disagreements() const;
};

/**
 * Unconditional, unforced chunk `inChunk`. Chunk 0 is fed the dumped
 * conditioning embedding, so its logits carry no front-end error and compare
 * against the per-step tolerances directly; later chunks are conditioned from
 * their dumped spectra.
 */
const ForcedChunk& plainChunk(int inChunk);

/**
 * Chunk `inChunk` of a note-level variant: its instrument selection installed
 * and, when forcing is on, the prologue the port's own tracker builds from the
 * reference's earlier chunks. A chunk with no selection and no prologue is the
 * plain chunk and shares its run.
 */
const ForcedChunk& variantChunk(const std::string& inVariant, int inChunk);

/** A single traced prefill of chunk 0. */
struct TracedPrefill {
    Trace trace;
    std::vector<float> logits;
    int n_past = 0;
};

/**
 * Chunk 0's prefill with every stage captured. Empty `inVariant` is the
 * unconditional path fed the dumped conditioning embedding; a variant name
 * installs that variant's selection and conditions from the dumped spectrum.
 */
const TracedPrefill& tracedPrefill(const std::string& inVariant = {});

/** The fixture's chunks concatenated back into one signal. */
const std::vector<float>& fixtureSignal();

/** The public API, loaded once per configuration. */
Transcriber& sharedTranscriber();

/** One callback invocation, copied out of the call. */
struct RecordedUpdate {
    std::vector<Note> new_notes;
    double finalized_through = 0.0;
    float progress = 0.0f;
};

struct Transcription {
    TranscribeOptions options;
    std::expected<std::vector<Note>, Error> notes;
    std::vector<RecordedUpdate> updates;
};

/**
 * The whole fixture through `Transcriber::transcribe`, configured the way the
 * named note-level variant is -- its prelude forcing and its instrument
 * selection -- with every callback update recorded.
 */
const Transcription& transcription(const std::string& inVariant);

/** @return A variant's instrument names as groups. */
std::vector<InstrumentGroup> groupsFor(const VariantReference& inVariant);

/**
 * The logit noise floor the dumper measured: how far apart the reference's own
 * fp32 and fp16 runs put a decode step's logits.
 */
double logitNoiseFloor();

/**
 * Classifies every disagreement in `inRun` against the noise floor. A top-two
 * margin inside it is a choice the checkpoint does not determine and is counted;
 * anything outside it fails the test.
 *
 * @return How many near-ties `inRun` holds.
 */
int requireOnlyNearTies(const ForcedChunk& inRun);

/**
 * Applies an instrument selection to the shared model and puts the
 * unconditional path back afterwards. The selection is transcription-scoped
 * state on a model the whole suite shares, so leaking it would be a confusing
 * failure a long way from its cause.
 */
class ScopedSelection
{
public:
    ScopedSelection(Model& inModel, std::span<const InstrumentGroup> inGroups);
    ~ScopedSelection();

    ScopedSelection(const ScopedSelection&) = delete;
    ScopedSelection& operator=(const ScopedSelection&) = delete;

private:
    Model& mModel;
};

} // namespace msl::test
