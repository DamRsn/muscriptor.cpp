# Testing

The tests compare the port against the PyTorch reference implementation stage
by stage, from the STFT to the note list. They run against *reference dumps*:
tensors, tokens and notes recorded from the reference on the audio fixture.

## Setup

```bash
uv sync                                   # gguf and numpy
uv run msl-fetch --size medium            # F16 weights → testdata/weights/

uv sync --group convert --group reference # PyTorch and the pinned reference
uv run msl-dump-refs --size medium        # reference dump → testdata/refs/medium/

cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build cpp/build
./cpp/build/tests/muscriptor_tests
```

The `[pure]` tests need neither weights nor a dump:

```bash
./cpp/build/tests/muscriptor_tests "[pure]"
```

### Test data

| Path | Tracked | Contents | Produced by |
|---|---|---|---|
| `testdata/audio/fixture_3chunks_16k.wav` | yes | 15 s, 16 kHz mono float32: three chunks ([attribution](../testdata/audio/README.md)) | `msl-audio-fixture --source <mp3>` |
| `testdata/vectors/tables.json` | yes | Vocabulary and instrument group tables | `msl-tables` (`--emit-cpp` also writes `cpp/src/instrument_groups.inc`) |
| `testdata/vectors/note_vectors.json` | yes | Hand-written token sequences, decoded by the reference | `msl-note-vectors` |
| `testdata/weights/` | no | Converted GGUF checkpoints | `msl-fetch`, or `msl-convert` |
| `testdata/refs/<size>/` | no | `refs_fp32.gguf`, `refs_fp16.gguf`, `manifest.json`, `notes.json` | `msl-dump-refs --size <size>` |

`msl-fetch` downloads published F16 checkpoints and checks them against
`SHA256SUMS`. `msl-convert --size <size> --weight-dtype f16|f32` converts from
the upstream safetensors release instead; this needs a Hugging Face account
that has accepted the upstream licence.

### Reference dumps

`msl-dump-refs` runs the reference twice, once with fp32 weights and once with
the model in fp16, and records:

- every intermediate tensor of chunk 0, plus the first decode steps
- the greedy token stream of each chunk, with prelude forcing off and no
  instrument selection
- note-level *variants* in `notes.json`:

  | Variant | Settings |
  |---|---|
  | `plain` | no prelude forcing, no instrument selection |
  | `prelude` | prelude forcing on (the default for `Transcriber`) |
  | `bass` | `electric_bass` selected |
  | `band` | `distorted_electric_guitar`, `synth_lead`, `electric_bass`, `drums`, `voice` |

`--variants plain` or `--variants none` limits the variants, which speeds up
dumping. Every size dumps into its own directory.

## Tolerances

The tests compare against the fp32 reference. For each tensor, the dumper
measures how far the reference's own fp16 run is from its fp32 run, and sets:

- `atol` = 4 × that max-abs difference, but at least 1e-5 of the tensor's
  scale
- `cosine_min` = 1 − 4 × that cosine deficit, but with a deficit of at least
  1e-7

`manifest.json` stores these values. Don't loosen a tolerance to make a test
pass; see [Diagnosing a failure](#diagnosing-a-failure).

Two stages use a different bound, because this measurement says nothing about
them:

- **STFT from the waveform.** Every later stage's tolerance was measured on the
  dumped spectrum. An fp32 STFT, PyTorch's included, moves log-mel by more than
  the `cond.logmel` tolerance. So the STFT test compares the spectrum itself,
  at 1e-5 of its scale, and then checks that decoding from the waveform gives
  the reference tokens.
- **Position table.** Both reference precisions compute it in fp32, so they
  agree exactly. The phase reaches about 500 rad, where one fp32 ulp is already
  3e-5, so the table is checked against an fp64 evaluation instead.

## The ladder

Tests run in the order the model computes, so the first failure points to the
first stage that went wrong.

| File | Tags | Checks |
|---|---|---|
| `test_gguf.cpp` | `[gguf]` | Loader, hyperparameters matching the reference, sinusoidal positions |
| `test_backend.cpp` | `[backend]` | Backend selection, graph node headroom, thread count |
| `test_stft.cpp` | `[stft]` | STFT magnitudes, window, DC/Nyquist handling; waveform to tokens for chunk 0 |
| `test_conditioning.cpp` | `[conditioning]` | Mel filterbank, log-mel, projection, frame mask, class embeddings |
| `test_prefill.cpp` | `[prefill]` | Token embedding, prefix order, positions, layer 0 internals, every layer's output, output norm, logits |
| `test_decode.cpp` | `[decode]` | Decode steps: logits, argmax, `n_past` |
| `test_generate.cpp` | `[generate]` | Greedy decoding matching the reference tokens to EOS: chunk 0, then every chunk `[slow]` |
| `test_vocabulary.cpp`, `test_instrument_groups.cpp` | `[pure]` | Token vocabulary and instrument tables |
| `test_tracker.cpp`, `test_note_assembly.cpp` | `[pure]` | Decode state machine and note assembly, on the hand-written vectors |
| `test_tokens_to_notes.cpp` | `[tokens-to-notes]` | The reference's token streams replayed to notes, for every variant |
| `test_prelude.cpp` | `[prelude]` | Forced prompts; prelude-forced decoding step by step `[slow]` |
| `test_instrument_conditioning.cpp` | `[instruments]` | Conditioning rows, prefix length and order, forbidden-token mask; conditioned decoding `[slow]` |
| `test_transcriber.cpp` | `[transcriber]` | The public API: errors, cancellation, streaming, and note lists matching the reference `[slow]` |

`[pure]` tests need no weights. `[tokens-to-notes]` tests need only
`notes.json`. Tests also carry narrower tags such as `[layer0]` and `[tracker]`;
`[instruments]`, `[prelude]` and `[generate]` span several files.

Step-by-step decoding tests feed the reference's tokens into the model and
check that its argmax picks the same token at every position. If every step
agrees, free-running greedy decoding reproduces the stream exactly.

Decoding runs once per configuration and is cached; the tests read the cached
results (`cpp/tests/inference.hpp`).

## Run options

```bash
./cpp/build/tests/muscriptor_tests [--size all|small|medium|large] [--device all|cpu|gpu]
                                   [--weight-dtype f16|f32] [--full] [--parity-report file]
                                   [Catch2 test specs and options]
```

| Option | Default | Effect |
|---|---|---|
| `--size` | `medium` | Checkpoint: `testdata/weights/muscriptor-<size>-<dtype>.gguf` and `testdata/refs/<size>/` |
| `--device` | `gpu`, or `cpu` with `--weight-dtype f32` | Backend. Without a GPU backend in the build, `gpu` falls back to the CPU |
| `--weight-dtype` | `f16` | Which converted weights to load |
| `--full` | off | Also run the `[slow]` tests. Naming any test or tag implies it |

The defaults are the CMake variables `MUSCRIPTOR_TEST_SIZE`,
`MUSCRIPTOR_TEST_DEVICE` and `MUSCRIPTOR_TEST_WEIGHT_DTYPE`.

`--size all` and `--device all` run one Catch2 session per combination and
print a summary. The exit code is the worst combination's, or 3 if nothing
ran. `gpu` with `f32` is always skipped, since both GPU backends round some F32
matmul inputs to half precision
([`PERFORMANCE.md`](PERFORMANCE.md#backend-notes)). When they come from `all`,
these are skipped too:

- `large` on the CPU.
- A size without weights or a dump. A size named explicitly runs, and its tests
  fail with the command that generates the missing data.

`muscriptor_bench` takes the same `--size`, `--device` and `--weight-dtype`
options ([`PERFORMANCE.md`](PERFORMANCE.md)).

### Tiers

Measured on an M1 Pro, F16, one run each:

| Configuration | Default tier | `--full` |
|---|---|---|
| `small`, Metal | 4 s | — |
| `medium`, Metal | 9 s | 70 s |
| `medium`, CPU | 28 s | 160 s |
| `large`, Metal | — | 3 min 44 s |

The default tier skips the `[slow]` tests. It still covers every numerical
stage, chunk 0 decoded to EOS, the decode state machine on real token streams,
and the fast API tests.

Never run two test processes at once.

## Diagnosing a failure

**`--weight-dtype f32`** (CPU only) runs the same tests on an all-F32
conversion. It separates a graph bug from a precision one: if a test still
fails at f32, the graph is wrong. Create the weights with
`uv run msl-convert --size <size> --weight-dtype f32`.

**`--parity-report <file>`** writes one JSON line per compared tensor: its
`max_abs` and cosine deficit, each also as a ratio of its tolerance, where 1.0
is the limit. `msl-parity-report` summarises one or more reports side by side:

```bash
./cpp/build/tests/muscriptor_tests --parity-report parity.jsonl
./cpp/build-fast-cpu/tests/muscriptor_tests --device cpu --parity-report fast.jsonl
uv run msl-parity-report fast.jsonl parity.jsonl
```

## Near-ties at F16

Greedy decoding picks the largest logit. When the top two logits are closer
than the reference's own fp32-to-fp16 gap on the logits (`dec.step1.logits` in
the manifest), which one wins depends on rounding rather than on the model.

The step-by-step decoding tests treat a disagreement as a *near-tie* when the
margin is inside that gap. They count it and print a warning, and fail on any
disagreement outside it. When a near-tie occurs, the transcriber tests compare
notes only up to the start of the chunk where the first one happened.

On the current fixture, the fp32 and fp16 reference token streams match for
every variant and every size, and `medium` at F16 shows no near-ties on either
the CPU or Metal.
