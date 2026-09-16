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
