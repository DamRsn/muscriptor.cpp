# API

Include `muscriptor/muscriptor.hpp`. Everything is in namespace `msl`, and the
headers under [`cpp/include/muscriptor/`](../cpp/include/muscriptor) are the
reference; this page describes how the pieces behave together.

## Scope

The library covers chunking, the STFT and mel front-end, the transformer,
greedy decoding, the MT3 decode state machine, prelude forcing, instrument
selection and note assembly.

The caller handles audio decoding, downmixing, resampling to 16 kHz, MIDI
writing and threading.

## Loading

```c++
std::expected<msl::Transcriber, msl::Error> t =
    msl::Transcriber::load("muscriptor-medium-f16.gguf");
```

CPU and GPU results are not bit-identical, since the backends accumulate in
different orders. To reproduce an earlier run exactly, use the same backend.

Loading reads the whole weights file. GPU backend initialisation can take much
longer the first time it runs than afterwards
([`PERFORMANCE.md`](PERFORMANCE.md#loading)). Two optional callbacks, both
called on the loading thread, cover it:

- **`should_cancel`** is polled after backend initialisation, after each weight
  tensor, and once more before `load` returns. When it returns `true`, `load`
  returns `Error::Cancelled`. Backend initialisation itself cannot be
  interrupted.
- **`on_progress`** reports the fraction of weight bytes uploaded, from 0 to 1,
  never decreasing. The first call, with 0, comes once the backend is up; until
  then there is nothing to measure progress against.

### Devices

`LoadOptions::device` picks where the model runs:

- **Empty, the default: Auto.** The first discrete GPU, otherwise the first
  integrated GPU, and the CPU when there is no GPU or it fails to initialise.
  `autoDevice(devices)` says which one that is before loading.
- **An index into `availableDevices()`.** A requirement, not a request: if
  that device fails to initialise, or the index is out of range, `load`
  returns `Error::DeviceUnavailable` rather than running somewhere else.

`availableDevices()` lists the GPUs in the backend's own order, then the CPU,
which is always last, so the list is never empty. Each `Device` has a `name`
(e.g. `"NVIDIA GeForce RTX 4070"`, `"Apple M1 Pro"`, `"CPU"`), a `backend`
(`"Metal"`, `"Vulkan"` or `"CPU"`, stable enough for a host to branch on), an
`integrated` flag, and `memory_total` in bytes (0 when unknown).

- Names are not unique: two identical cards share one. Indices are stable only
  within a process; a host that stores a choice should store the name.
- The list is built on the first call and is fixed for the life of the
  process: ggml enumerates GPUs once, so a GPU added or removed later shows up
  after a restart.
- The first call initialises the GPU backends and can be slow
  ([`PERFORMANCE.md`](PERFORMANCE.md#loading)). Make it off a UI thread.
  It is thread-safe.
- Metal exposes one device, the system default GPU. Vulkan lists the discrete
  and integrated GPUs that ggml supports, and marks integrated ones.

`Transcriber::device()` returns the entry the model actually runs on.

## Transcribing

```c++
std::expected<std::vector<msl::Note>, msl::Error> notes =
    t->transcribe(samples, options, callback);
```

- `samples` is the whole signal, 16 kHz mono float32. It is split into 5 s
  chunks (`Transcriber::SEGMENT_SAMPLES`), and the last chunk is zero-padded.
  An empty signal returns no notes.
- The call blocks for seconds to minutes. Never call it from an audio thread.
- One call at a time per instance: it mutates the KV cache and is not
  synchronised. For concurrent transcription, use one `Transcriber` per thread.
  Each holds its own weights.
- The result is sorted by `(onset, is_drum, program, pitch, offset)`.
- A chunk that never produces EOS is not an error. Its decoding stops after
  `MAX_TOKENS_PER_CHUNK` tokens, counting its forced prompt, or when the KV
  cache is full, and the tokens decoded so far are kept.

### `TranscribeOptions`

| Field | Default | Effect |
|---|---|---|
| `instruments` | empty (all) | Restricts transcription to these groups. It adds a conditioning prefix and masks every other instrument's tokens, so no other instrument can appear. Set it before transcribing; it cannot be applied to a finished result. |
| `prelude_forcing` | `true` | Feeds each chunk's still-sounding notes into the next chunk as a forced prompt ([`TOKENIZER.md`](TOKENIZER.md) § 4). |
| `resume_from` | empty | Continues an earlier call from one of its `resume_point`s instead of starting over ([Resuming](#resuming)). |
| `n_threads` | `0` | CPU threads. `0` selects the number of performance cores on macOS, and of logical CPUs elsewhere. On a GPU backend, only the conditioning front-end runs on the CPU and uses them. |

## Notes

```c++
struct Note {
    double onset;    // seconds from the start of the signal
    double offset;
    int pitch;       // MIDI note number; the GM percussion note for drums
    int program;     // decoded MIDI program, or DRUM_PROGRAM (128)
    bool is_drum;
};
```

- Onsets are on the model's 10 ms grid. Offsets can fall between grid points
  after overlap trimming.
- Drum-token hits last 10 ms: their `offset` is `onset + 10 ms`. A decoded
  program 96 is reported as a drum and keeps its decoded offset.
- There is no velocity: the model does not predict one
  ([`MODEL.md`](MODEL.md#outputs)).

`program` is the raw decoded program. `InstrumentGroup` sits on top of it and
has 35 values, the 34 named groups plus `Drums`. Its enumerators are the
reference's group ids, so they are not contiguous; `allInstrumentGroups()`
lists them.

| Function | Returns |
|---|---|
| `instrumentGroupFor(program)` | The named group, or `nullopt` for a program in one of the unnamed groups |
| `programFor(group)` | The program the model emits for that group |
| `instrumentName(group)` | The group's name, e.g. `"electric_bass"` |
| `instrumentLabel(program)` | The group name, or `"program_<n>"` |

Program 96 maps to `Drums`. This matches the reference
([`TOKENIZER.md`](TOKENIZER.md) § 5).

## Progress and streaming

The optional callback is called synchronously, on the thread that called
`transcribe`, once after each chunk and once more at the end. An empty signal
gets no calls. `new_notes` is only valid during the call. Returning `false`
stops the transcription: `transcribe` then returns `Error::Cancelled`. That
only happens between chunks; `TranscribeOptions::should_cancel` stops sooner.

```c++
struct TranscriptionUpdate {
    std::span<const Note> new_notes;
    double finalized_through;
    float progress;
    std::string resume_point;
};
```

- **`new_notes`** holds notes that closed one chunk earlier: the call after
  chunk *k* carries the notes that closed during chunk *k−1*. A note is
  reported exactly once and never changes afterwards. Joining every call's
  `new_notes` gives the same list `transcribe` returns.
- **`finalized_through`**, in seconds. After chunk *k* it is `k × 5 s`. Every
  note ending before it has been reported, and nothing is known yet about later
  audio. A host playing a transcription while it is still running can play up
  to this point.

  One exception: a note the model never closes is closed at `onset + 10 ms`
  when the signal ends. It can end well before the current line, and only
  arrives in the final call.
- **`progress`** runs from 0 to 1 and never decreases. The last two calls both
  report 1; `transcribe` returning is the end signal.
- **`resume_point`** lets a later call continue from right after this update
  ([Resuming](#resuming)). Empty in the final call.

## Resuming

A cancelled transcription can be continued later, in the same process or
another one. Keep the notes reported so far and the latest `resume_point`, then
call `transcribe` again with the same signal and options and
`TranscribeOptions::resume_from` set to that point:

- Decoding continues at the next chunk. The callbacks carry on exactly where
  the first call left off: the same `new_notes`, `finalized_through`,
  `progress` and `resume_point` an uninterrupted run would have reported.
- The call returns only the notes not reported before the resume point. Those
  plus the ones reported before are every `new_notes` of an uninterrupted run
  joined, which is that run's result.
- A point that is malformed, or that was made for another signal length,
  instrument selection or `prelude_forcing`, returns
  `Error::InvalidResumePoint` before anything is decoded.
- The point does not record the checkpoint or the device. Resuming with another
  checkpoint mixes two models' output. Resuming on another backend works, and
  differs the way CPU and GPU results always do.

The point holds only what the next chunk needs: the notes still sounding, and
the notes that closed in the last decoded chunk, which are reported one chunk
late. It is plain text, one record per line, with times in 10 ms frames so no
floating-point value is written. Annotated (the annotations are not part of
it):

```
muscriptor-resume 1
samples 240000          signal length the point was made for
prelude 1               prelude_forcing
instruments 2 33 35     count, then the deduplicated selection, in order
decoded 2               chunks done; decoding resumes at chunk 2
prologue 0              1 if the last chunk never reached its tie token
open 2                  notes still sounding, in the order they opened
29 48 992               program pitch onset
33 38 992
withheld 3              notes closed in the last chunk, in close order
n 29 46 491 515         note: program pitch onset offset
d 38 515                drum hit: pitch onset (it always lasts 10 ms)
n 33 34 491 515
```

Treat it as opaque: the format belongs to the library and changes with its
version number.

## Cancellation

`LoadOptions::should_cancel` and `TranscribeOptions::should_cancel` take a
`CancelPredicate`, a `std::function<bool()>`. The library polls it on the
thread that called `load` or `transcribe`, and stops with `Error::Cancelled`
once it returns `true`. It is polled often, so it should be cheap, typically
an atomic flag set from another thread.

A cancel takes effect at the next poll. How far apart polls are:

| Phase | Polled | Longest stretch without a poll |
|---|---|---|
| Backend initialisation | Once, after it | All of it |
| Reading the file header, allocating the weights | Once, after it | All of it |
| Weight upload | After each tensor | One tensor read and upload |
| Setup after the upload | Once, after it | All of it |
| Transcription, CPU | Before each chunk and decode step, and at every graph node | One graph node |
| Transcription, Metal or Vulkan | Before each chunk and decode step | One prefill |

On a GPU the prefill is a single graph that ggml cannot interrupt.
[`PERFORMANCE.md`](PERFORMANCE.md) has the prefill and decode-step timings.

A `Transcriber` stays usable after a cancelled `transcribe`: the next call
starts from a clean state.

## Errors

`Transcriber` never throws; failures come back as `msl::Error` values.
`describe(error)` gives a short, stable string for logs.

| Error | Meaning |
|---|---|
| `FileNotFound` | No file at the path |
| `InvalidCheckpoint` | Not a readable GGUF, or missing tensors or metadata |
| `UnsupportedArch` | Audio framing other than 16 kHz / 100 Hz / hop 160, or inconsistent attention geometry |
| `UnsupportedCheckpointVersion` | A GGUF with a different `muscriptor.format_version`, or none, as in a GGUF of another model |
| `OutOfMemory` | Allocation failed, or no backend could be initialised |
| `DeviceUnavailable` | `LoadOptions::device` is out of range, or that device failed to initialise |
| `ContextOverflow` | A chunk's prefix plus its forced prompt does not fit in the KV cache |
| `Cancelled` | A `should_cancel` predicate returned `true`, or the note callback returned `false` |
| `InvalidArgument` | Unusable `TranscribeOptions`, e.g. an instrument outside the named groups |
| `InvalidResumePoint` | `TranscribeOptions::resume_from` is malformed, or does not match this call's signal and options |
| `Internal` | A bug in the library |

## Checkpoint format version

Each converted GGUF stores `muscriptor.format_version`. `load` accepts only
`CHECKPOINT_FORMAT_VERSION` and returns `UnsupportedCheckpointVersion` for any
other value. The version tracks what the loader expects: which metadata keys
and tensors a file contains, and how they are read. It is not a release number,
and it does not change when the weights do. Published checkpoints live under a
directory named after it (`v1/`).

## Logging

The library prints nothing. ggml's log output is routed through a sink that is
silent by default:

```c++
msl::setLogCallback([](msl::LogLevel level, std::string_view text) { myLog(level, text); },
                    msl::LogLevel::Debug);
```

The sink is process-wide, as ggml's log hook is, and the last call wins. ggml's
Vulkan backend writes some errors and warnings straight to `std::cerr`, and
those bypass the sink.

## Running inside a host application

- Run `transcribe` on a worker thread.
- Inside a DAW, limit `n_threads`. ggml's thread pool will otherwise occupy
  every performance core and can starve the host's audio threads. GPU backends
  avoid this.
- Resampling quality matters. The mel front-end is sensitive enough that a
  different resampling filter can move onsets, so check the full path end to
  end.
- Memory is the weights plus a float32 KV cache
  ([`PERFORMANCE.md`](PERFORMANCE.md#memory)).
- On Windows, do not call ggml's global backend registry (`ggml_backend_dev_*`,
  `ggml_backend_reg_*`) from the host. On a machine without Vulkan it faults
  ([`BUILDING.md`](BUILDING.md#vulkan)).

## `msl::Model`

`muscriptor/model.hpp` exposes the layer below `Transcriber`, which stops at
token ids. The test ladder drives it one stage at a time. Its main methods:

```
Model::load(gguf, options)                           → Model
  .hparams() / .stft() / .positionEmbeddings() / .nPast() / .contextSize()
  .encodeAudio(samples)                              → conditioning embedding
  .encodeConditioning(spectrum, n_frames, n_samples) → conditioning embedding
  .setInstrumentRows(rows) / .setForbiddenTokens(ids) / .setNumThreads(n)
  .setShouldCancel(predicate)
  .reset() / .prefill(conditioning, n_frames, tokens) → logits
  .decode(token)                                     → logits
  .generate(conditioning, n_frames, max_tokens, eos_id[, prompt]) → token ids
```

- `Model` reports errors by throwing `msl::Exception`, which carries an
  `Error`.
- It is not safe to call from several threads, even through `const` methods:
  every evaluation builds its graph in one shared scratch buffer.
- `generate` with a prompt returns the prompt followed by the generated tokens.
- `reset()` leaves the instrument rows and the forbidden-token mask in place;
  they apply to the whole transcription.
- `setShouldCancel` installs a predicate polled before each decode step of
  `generate`, and at every graph node on the CPU. A cancelled evaluation throws
  `Error::Cancelled` and leaves the KV cache partly written; `generate` resets
  it. `ModelOptions::should_cancel` and `on_progress` apply to `load` only.
