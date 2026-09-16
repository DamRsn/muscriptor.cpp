# muscriptor.cpp

A C++/[ggml](https://github.com/ggml-org/ggml) port of
[MuScriptor](https://github.com/muscriptor/muscriptor), Kyutai and Mirelo's
multi-instrument music transcription model, built as a static library.
Originally developed for [NeuralNote](https://github.com/DamRsn/NeuralNote).

It takes 16 kHz mono float32 audio and returns note events: onset, offset,
pitch and instrument. It runs on the CPU, Metal or Vulkan, with no Python at
run time. Decoding audio files, resampling to 16 kHz and writing MIDI are left
to the caller.

The port is checked stage by stage against the PyTorch reference implementation
and reproduces its token streams and note lists.

## Platforms

- **Tested:** macOS on Apple Silicon (CPU and Metal), Windows (CPU and Vulkan).
- **Untested:** macOS on Intel. It builds with the `macos-x86_64` preset and
  should work.
- **Linux:** support is coming.

## Quick start

Build requirements: CMake 3.24+, Ninja, a C++23 compiler, and network access,
since configuring fetches the pinned ggml.

```bash
cmake -S cpp -B cpp/build -G Ninja -DMUSCRIPTOR_BUILD_TESTS=OFF
cmake --build cpp/build
```

This builds the library and `muscriptor_bench`. Most tests need weights and a
reference dump ([`docs/TESTING.md`](docs/TESTING.md)); the `[pure]` tier needs
neither, and is the quickest check that a clone is sound:

```bash
cmake -S cpp -B cpp/build -G Ninja && cmake --build cpp/build
./cpp/build/tests/muscriptor_tests "[pure]"
```

Weights are a separate download. Converted GGUF checkpoints are published at
[`DamRsn/muscriptor-gguf`](https://huggingface.co/DamRsn/muscriptor-gguf).
With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run msl-fetch --size medium      # testdata/weights/muscriptor-medium-f16.gguf
```

The Python tooling does not run on Intel Macs. There, download
`v1/muscriptor-medium-f16.gguf` from the Hugging Face repo into
`testdata/weights/` instead.

Transcribe the bundled 15 s test fixture, or your own audio:

```bash
./cpp/build/bench/muscriptor_bench --transcribe
ffmpeg -i song.mp3 -ac 1 -ar 16000 -c:a pcm_f32le song.wav   # 16 kHz mono float32
./cpp/build/bench/muscriptor_bench --transcribe --audio song.wav
```

The benchmark prints the backend, the real-time factor and the number of notes.
In your own code:

```c++
#include "muscriptor/muscriptor.hpp"

auto transcriber = msl::Transcriber::load("testdata/weights/muscriptor-medium-f16.gguf");

if (!transcriber) {
    return report(msl::describe(transcriber.error()));
}

// 16 kHz mono float32, the whole signal.
auto notes = transcriber->transcribe(samples);
```

## Using it from CMake

```cmake
add_subdirectory(third_party/muscriptor.cpp/cpp muscriptor)
target_link_libraries(MyApp PRIVATE muscriptor_ggml)
```

Everything links statically: the library, ggml and pffft. Consumers inherit
C++23 and, on Apple, one architecture per build tree and a macOS 11.0
deployment target unless the project sets its own. Tests and the benchmark are
not built. [`docs/BUILDING.md`](docs/BUILDING.md) has the details.

## Docs

| Doc | Contents |
|---|---|
| [`docs/API.md`](docs/API.md) | The public API: transcription, streaming updates, instruments, errors, logging |
| [`docs/MODEL.md`](docs/MODEL.md) | The model: sizes, inputs and outputs, pipeline, tensor map, GGUF metadata |
| [`docs/TOKENIZER.md`](docs/TOKENIZER.md) | MT3 vocabulary, decode state machine, prelude forcing, instrument groups |
| [`docs/BUILDING.md`](docs/BUILDING.md) | Build options, presets, platform notes |
| [`docs/TESTING.md`](docs/TESTING.md) | Reference dumps, the test ladder, run options |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | Throughput, memory, threads |

## Layout

| Path | Contents |
|---|---|
| `cpp/include`, `cpp/src` | The library |
| `cpp/tests`, `cpp/bench` | Catch2 test ladder and benchmark |
| `cpp/third_party/pffft` | Vendored FFT |
| `msl/` | Python tooling: weight fetch, conversion, reference dumps |
| `testdata/audio`, `testdata/vectors` | Committed test fixtures |

## Contributing

- Never widen a test tolerance to make a test pass; see
  [`docs/TESTING.md`](docs/TESTING.md#tolerances).
- `cpp/src/instrument_groups.inc` and `testdata/vectors/` are generated
  (`msl-tables`, `msl-note-vectors`). Don't edit them by hand.
- `cpp/third_party/` is vendored: don't modify or reformat it.
- Format C++ with the repo's `.clang-format`:

  ```bash
  find cpp -path cpp/third_party -prune -o -path 'cpp/build*' -prune -o \
       -regex '.*\.\(h\|hpp\|c\|cpp\)$' -print0 | xargs -0 clang-format -i
  ```

## Licences

- **Code**: MIT, see [`LICENSE`](LICENSE). It includes MuScriptor's own MIT
  notice.
- **Model weights**: CC BY-NC 4.0, from
  [MuScriptor](https://huggingface.co/MuScriptor). Not included in this
  repository.
- **Test audio**: an excerpt of "You Drive Me Insane" by Jon Worthy and the
  Bends, CC BY 4.0. See [`testdata/audio/README.md`](testdata/audio/README.md).
- **pffft**: BSD-style, see `cpp/third_party/pffft`.

## Credits

[MuScriptor](https://github.com/muscriptor/muscriptor) by Kyutai and Mirelo
([paper](https://arxiv.org/abs/2607.08168v1)), [ggml](https://github.com/ggml-org/ggml),
and [pffft](https://bitbucket.org/jpommier/pffft).

This repository was developed by Damien Ronssin with AI assistance.
