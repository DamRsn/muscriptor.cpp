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
    msl::Transcriber::load("muscriptor-medium-f16.gguf", {.use_gpu = true});
```

`LoadOptions::use_gpu` (default `true`) runs on Metal or Vulkan when the build
has a GPU backend and the machine has a usable device, and on the CPU
otherwise. `backendName()` returns `"CPU"`, `"Metal"` or `"Vulkan"`, and a host
can branch on these names.

CPU and GPU results are not bit-identical, since the backends accumulate in
different orders. To reproduce an earlier run exactly, use the same backend.

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

## Progress, streaming and cancellation

The optional callback is called synchronously, on the thread that called
`transcribe`, once after each chunk and once more at the end. An empty signal
gets no calls. `new_notes` is only valid during the call. Return `false` to
stop: `transcribe` then returns
`Error::Cancelled`.

```c++
struct TranscriptionUpdate {
    std::span<const Note> new_notes;
    double finalized_through;
    float progress;
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
| `ContextOverflow` | A chunk's prefix plus its forced prompt does not fit in the KV cache |
| `Cancelled` | The callback returned `false` |
| `InvalidArgument` | Unusable `TranscribeOptions`, e.g. an instrument outside the named groups |
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
