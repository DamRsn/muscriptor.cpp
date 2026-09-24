# Performance

## Throughput

Numbers from `muscriptor_bench` on an Apple M1 Pro (8 performance + 2
efficiency cores), using the default portable build. Each configuration was run
once on an idle machine. All runs transcribe the 15 s test fixture with the
default options, including prelude forcing.

Real-time factor (above 1 is faster than real time):

| Size | CPU | Metal |
|---|---|---|
| `small` | 1.94x | 3.48x |
| `medium` | 0.69x | 1.56x |
| `large` | — | 0.48x |

Per phase, `medium`, first chunk, 400 decode steps:

| | CPU | Metal |
|---|---|---|
| Prefill | 1.71 s | 0.16 s |
| Decode, per step (mean / p50) | 13.1 / 11.9 ms | 7.3 / 7.3 ms |
| Conditioning front-end | 6 ms | 6 ms |

Both backends produce the same token digest over those 400 steps.

To reproduce:

```bash
./cpp/build/bench/muscriptor_bench --transcribe             # whole signal
./cpp/build/bench/muscriptor_bench --steps 400              # per phase
./cpp/build/bench/muscriptor_bench --size small --device cpu --threads 4
./cpp/build/bench/muscriptor_bench --load-profile               # load breakdown
./cpp/build/bench/muscriptor_bench --transcribe --cancel-after 4000
```

In per-phase mode, the benchmark prints a digest of the tokens it decoded. If a
change leaves the digest unchanged, it produces the same output.

`uv run msl-bench-torch` runs the same benchmark on the PyTorch reference and
prints the same digest. Its flags mirror the benchmark's, except that `--model`
takes a size or a safetensors path, `--device` takes `mps`, `cpu` or `cuda`, and
`--dtype` sets the precision.
`uv run msl-bench-matrix` runs both for every size and device, one process per
measurement.

For stable numbers, run one configuration per process on an idle machine.
Never run two benchmarks or test runs at the same time: ggml's threads wait on
each other between ops, and competing for cores slows both runs down severalfold.

## Loading

`muscriptor_bench --load-profile` times `Transcriber::load` through its public
callbacks. **Init** runs up to the first `on_progress` call: backend
initialisation, reading the GGUF metadata, and allocating the weight buffer.
**Upload** runs from the first `on_progress` call to the last, reading and
uploading every tensor. **Setup** covers the rest: the KV cache, the CPU copy of
the conditioning weights, and the position table.

M1 Pro, weights file already in the page cache, two runs per row, in ms:

| Size | Backend | Init | Upload | Setup | Total |
|---|---|---|---|---|---|
| `small` | CPU | 0.6–0.7 | 66–69 | 28 | 94–97 |
| `small` | Metal | 53–57 | 60–61 | 31 | 144–149 |
| `medium` | CPU | 0.9–1.1 | 188–199 | 50–51 | 239–250 |
| `medium` | Metal | 63–66 | 174–177 | 57–60 | 297–299 |
| `large` | Metal | 135–146 | 747–769 | 135–178 | 1017–1093 |

The first Metal initialisation after a new build took about 20 s: ggml-metal
compiles its embedded kernel source, and later runs are fast.

## Cancellation

A cancel waits for the next poll of `should_cancel` ([`API.md`](API.md#cancellation)).
`--load-profile` and `--transcribe` print how far apart the polls were, and
`--cancel-after MS` cancels at that time and prints how long the call then took
to return. After a cancelled `--transcribe`, the benchmark transcribes again
with the same instance and prints the note digest, which should match an
uncancelled run.

M1 Pro, `medium`, 15 s fixture:

| | CPU | Metal |
|---|---|---|
| Load: longest gap between polls after init | 50–51 ms | 56–60 ms |
| Transcription: longest gap between polls | 151 ms | 165–176 ms |

On Metal the longest gap is a prefill, which is one graph. During load it is
the setup after the last tensor, and at `large` on Metal that reached 178 ms.

## Threads

On macOS, `n_threads = 0` uses the performance cores only: ggml synchronises
all its threads at every graph node, so a slower efficiency core would hold back
every op. Elsewhere it uses every logical CPU. The thread count doesn't change
results. On a GPU backend it only applies to the conditioning front-end, which
runs on the CPU.

## Memory

- **Weights.** The F16 GGUF, loaded onto the chosen backend:

  | Size | F16 file |
  |---|---|
  | `small` | 209 MB |
  | `medium` | 618 MB |
  | `large` | 2.74 GB |

- **KV cache.** F32. `Transcriber` allocates `n_ctx = 2538` positions per
  layer: 501 mel frames, the dataset row, up to 35 instrument rows, the initial
  token and 2000 tokens. For each layer, K is `[dim, n_ctx]` and V is
  `[n_ctx, head_dim, n_head]`: in total `2 × n_layer × dim × n_ctx × 4` bytes,
  about 218 MB, 499 MB and 1.5 GB for `small`, `medium` and `large`.
- **Conditioning front-end.** On a GPU backend, it keeps a CPU copy of the mel
  filterbank and projection weights.

## Backend notes

- **Metal.** The matmul kernel for F32 inputs rounds them to half precision.
  This is why the conditioning front-end always runs on the CPU, and why F32
  weights are not tested on the GPU.
- **Vulkan.** Every matmul is marked `GGML_PREC_F32`. This makes the F16
  matrix-matrix kernels, used in prefill, accumulate in fp32. Single-token
  decode steps use vector kernels, which always accumulate in fp32.

  Separately, ggml-vulkan's matrix-matrix path converts non-contiguous inputs,
  such as the attention views in prefill, to F16 whatever the mark says. So
  Vulkan and CPU results can differ slightly at F16.
