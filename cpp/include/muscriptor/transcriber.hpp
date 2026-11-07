#pragma once

#include "muscriptor/device.hpp"
#include "muscriptor/error.hpp"
#include "muscriptor/note.hpp"

#include <cstddef>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace msl
{

/**
 * Polled while `load` or `transcribe` runs, always on the thread that called
 * it; return true to stop, and the call returns `Error::Cancelled`. It is
 * called often, so keep it to something like an atomic load.
 */
using CancelPredicate = std::function<bool()>;

/**
 * Silence decoded ahead of the signal and removed from the result. The model
 * reads sound in the first frames of a chunk as already sounding, so without
 * it notes that start the signal are lost. See docs/API.md.
 */
struct LeadInOptions {
    bool enabled = true;

    // Seconds, 0 to SEGMENT_DURATION; rounded to the 10 ms frame grid.
    double duration = 0.2;
};

/**
 * Drops what the model produces where the input is silent, and skips decoding
 * chunks that are silent throughout. See docs/API.md.
 */
struct SilenceGateOptions {
    bool enabled = true;

    // A 10 ms frame whose RMS is below this, in dB relative to full scale, is silent.
    float threshold_dbfs = -60.0f;

    // Seconds. A quieter run shorter than this is not silence; one touching
    // either end of the signal always is.
    double min_duration = 0.3;

    // Seconds after the input goes silent during which a note may still ring
    // and start.
    double release = 0.1;
};

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

    /** Turn off, with `silence_gate`, for the reference's exact output. */
    LeadInOptions lead_in;

    SilenceGateOptions silence_gate;

    /**
     * CPU threads; 0 selects the performance-core count. Ignored on a GPU
     * backend. Inside a DAW, cap it to leave cores for the host's audio threads.
     */
    int n_threads = 0;

    /**
     * Polled before each chunk, before each decode step, and on the CPU backend
     * at every node of a graph. Empty: never cancelled. See docs/API.md for the
     * longest stretch between two polls on each backend.
     */
    CancelPredicate should_cancel;

    /**
     * A `TranscriptionUpdate::resume_point` from an earlier call on the same
     * signal with the same options. Empty: start from the beginning. Otherwise
     * decoding continues after that update, and only the notes it had not
     * reported yet are returned. See docs/API.md, "Resuming".
     */
    std::string resume_from;
};

/** Options fixed at load: they decide where the weights and KV cache live. */
struct LoadOptions {
    /**
     * Index into `availableDevices()`, or empty for Auto: the GPU
     * `autoDevice(devices, support)` names, falling back to the CPU. An
     * explicit device that cannot run the checkpoint or fails to initialise
     * makes `load` return `Error::DeviceUnavailable`. `Transcriber::device`
     * reports where the model runs. CPU and GPU results are not bit-identical.
     */
    std::optional<std::size_t> device;

    /**
     * Polled after backend initialisation, after each weight tensor, and once
     * more before `load` returns. Empty: never cancelled. Backend
     * initialisation itself cannot be interrupted; see docs/API.md.
     */
    CancelPredicate should_cancel;

    /**
     * Fraction of the weight bytes uploaded, 0 to 1, non-decreasing, on the
     * loading thread. The first call, with 0, comes after backend
     * initialisation; the last, with 1, is followed by a short setup before
     * `load` returns.
     */
    std::function<void(float inProgress)> on_progress;
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
     * is known about the signal beyond it. After chunk *k* this is at most
     * `k * SEGMENT_DURATION` minus the lead-in, and less while the silence gate
     * may still cut a note that has not been reported.
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

    /**
     * Pass as `TranscribeOptions::resume_from` to continue right after this
     * update, in this process or a later one. Empty in the final call.
     */
    std::string resume_point;
};

/**
 * @param inUpdate What this chunk added; valid only for the duration of the call.
 * @return false to cancel; `transcribe` then returns `Error::Cancelled`. Only
 *         checked once per chunk: `TranscribeOptions::should_cancel` is the
 *         prompt way.
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
     * Which devices can run a checkpoint, from its metadata alone: no weight is
     * read. Checks the devices `load` with this `inDevice` would check: that
     * device, or for Auto the GPUs in Auto order up to the first one that is
     * supported or could not be created, and the CPU when every GPU is
     * unsupported. On Vulkan, checking a GPU
     * sets up ggml's device for it, as a load would. Thread-safe.
     *
     * @param inDevice As `LoadOptions::device`.
     * @return One entry per device of `availableDevices()`, in the same order,
     *         empty for a device not checked. Fails
     *         on a missing or unreadable checkpoint, an unsupported CPU and an
     *         index out of range, with the errors `load` returns; the other
     *         reasons `load` can reject a checkpoint are not checked.
     */
    static std::expected<std::vector<std::optional<DeviceSupport>>, Error>
        checkDevices(const std::filesystem::path& inGgufPath, std::optional<std::size_t> inDevice = std::nullopt);

    /** @return The device the model runs on, as listed by `availableDevices()`. */
    const Device& device() const;

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
     * @param inSamples 16 kHz mono float32, the entire signal. Split, after the
     *        lead-in, into 5 s chunks; a short tail is zero-padded. An empty
     *        signal yields no notes.
     * @param inOptions Decode-time configuration.
     * @param inCallback Optional per-chunk progress and cancellation.
     * @return Every note, sorted by (onset, is_drum, program, pitch, offset).
     */
    std::expected<std::vector<Note>, Error> transcribe(std::span<const float> inSamples,
                                                       const TranscribeOptions& inOptions,
                                                       const NoteCallback& inCallback = {});

    /**
     * @return Number of 5 s chunks `transcribe` splits `inNSamples` into with
     *         `inOptions`, lead-in included. Meaningless for options it rejects.
     */
    static int chunkCount(std::size_t inNSamples, const TranscribeOptions& inOptions);

private:
    Transcriber();

    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace msl
