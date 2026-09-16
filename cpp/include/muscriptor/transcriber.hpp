#pragma once

#include "muscriptor/error.hpp"
#include "muscriptor/note.hpp"

#include <cstddef>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace msl
{

struct TranscribeOptions {
    /**
     * Instruments to transcribe. Empty selects the unconditional path.
     *
     * A non-empty selection adds the `instrument_group` conditioning prefix and
     * masks every other instrument's tokens, so no other instrument can appear.
     * It changes decoding, so it cannot be applied to a finished result.
     */
    std::vector<InstrumentGroup> instruments;

    /**
     * Teacher-force still-sounding notes across chunk boundaries, as the
     * reference does by default. Chunks are decoded strictly in order.
     */
    bool prelude_forcing = true;

    /**
     * CPU threads; 0 selects the performance-core count. Ignored on a GPU
     * backend. Inside a DAW, cap it to leave cores for the host's audio threads.
     */
    int n_threads = 0;
};

/** Options fixed at load: they decide where the weights and KV cache live. */
struct LoadOptions {
    /**
     * Run on the GPU when the build and the machine both have one, otherwise on
     * the CPU; `Transcriber::backendName` reports which. CPU and GPU results
     * are not bit-identical.
     */
    bool use_gpu = true;
};

/**
 * What one chunk's completion adds to the transcription.
 *
 * Reported once per chunk and once more at the end, synchronously on the thread
 * that called `transcribe`. Nothing here outlives the call.
 */
struct TranscriptionUpdate {
    /**
     * Notes finalised by this update; may be empty.
     *
     * Released one chunk late -- the call after chunk *k* carries what closed
     * during chunk *k-1* -- since overlap trimming can still shorten a note
     * against one starting in the next chunk. Each note is reported once and
     * never changes; concatenating every call reproduces `transcribe`'s result.
     */
    std::span<const Note> new_notes;

    /**
     * Seconds. Every note whose offset is below this has been reported; nothing
     * is known about the signal beyond it. After chunk *k* this is
     * `k * SEGMENT_DURATION`.
     *
     * Exception: a note the model never closes is closed at `onset + 10 ms`
     * when the signal ends, possibly far behind this line, and arrives in the
     * final call.
     */
    double finalized_through = 0.0;

    /**
     * Fraction of the signal consumed, 0 to 1, non-decreasing. The last two
     * calls both report 1; `transcribe` returning is the end signal.
     */
    float progress = 0.0f;
};

/**
 * @param inUpdate What this chunk added; valid only for the duration of the call.
 * @return false to cancel; `transcribe` then returns `Error::Cancelled`.
 */
using NoteCallback = std::function<bool(const TranscriptionUpdate& inUpdate)>;

/**
 * The `muscriptor.format_version` this build loads; any other value fails with
 * `Error::UnsupportedCheckpointVersion`. See docs/API.md.
 */
inline constexpr int CHECKPOINT_FORMAT_VERSION = 1;

/**
 * Audio in, notes out: chunking, the mel front-end, the transformer, greedy
 * decoding, the MT3 decode state machine, prelude forcing and note assembly.
 */
class Transcriber
{
public:
    /** The only sample rate this model accepts. */
    static constexpr int SAMPLE_RATE = 16000;

    /** Audio window the model consumes at a time, in samples and in seconds. */
    static constexpr int SEGMENT_SAMPLES = 80000;
    static constexpr double SEGMENT_DURATION = 5.0;

    /** Upper bound on tokens per chunk, matching the reference's max_gen_len. */
    static constexpr int MAX_TOKENS_PER_CHUNK = 2000;

    static std::expected<Transcriber, Error> load(const std::filesystem::path& inGgufPath, LoadOptions inOptions = {});

    /**
     * @return Backend in use: `"CPU"`, `"Metal"` or `"Vulkan"`. These names are
     *         stable, unlike ggml's device names.
     */
    const char* backendName() const;

    ~Transcriber();
    Transcriber(Transcriber&&) noexcept;
    Transcriber& operator=(Transcriber&&) noexcept;
    Transcriber(const Transcriber&) = delete;
    Transcriber& operator=(const Transcriber&) = delete;

    /**
     * Transcribe a whole signal. Blocking, seconds to minutes; never call it
     * from an audio thread.
     *
     * One call at a time per instance: it mutates the KV cache and is not
     * synchronised.
     *
     * @param inSamples 16 kHz mono float32, the entire signal. Split into 5 s
     *        chunks; a short tail is zero-padded. An empty signal yields no notes.
     * @param inOptions Decode-time configuration.
     * @param inCallback Optional per-chunk progress and cancellation.
     * @return Every note, sorted by (onset, is_drum, program, pitch, offset).
     */
    std::expected<std::vector<Note>, Error> transcribe(std::span<const float> inSamples,
                                                       const TranscribeOptions& inOptions = {},
                                                       const NoteCallback& inCallback = {});

    /** @return Number of 5 s chunks `inNSamples` will be split into. */
    static int chunkCount(std::size_t inNSamples);

private:
    Transcriber();

    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace msl
