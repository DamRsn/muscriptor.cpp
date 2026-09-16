# Building

Requirements: CMake 3.24+, Ninja, a C++23 compiler, and network access, since
ggml (pinned by commit in `cpp/CMakeLists.txt`) is fetched at configure time.
Catch2 and nlohmann_json are fetched only when tests are built.

```bash
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build cpp/build
```

To build against a local ggml checkout instead, add
`-DFETCHCONTENT_SOURCE_DIR_GGML=/path/to/ggml`.

## Options

| Option | Default | Effect |
|---|---|---|
| `MUSCRIPTOR_GGML_FP32_ACCUM` | `ON` | On ARM, `ggml-cpu` sums F16 matmuls in fp32 instead of fp16. See below. |
| `MUSCRIPTOR_NATIVE` | `OFF` | Tunes the build for the machine it is compiled on (ggml's `GGML_NATIVE`). For local benchmarking only: the binary may crash on other CPUs. |
| `MUSCRIPTOR_METAL` | `ON` on Apple | Builds the Metal backend, with its shaders embedded in the binary. |
| `MUSCRIPTOR_VULKAN` | `ON` elsewhere, when the Vulkan SDK is found | Builds the Vulkan backend. |
| `MUSCRIPTOR_BUILD_TESTS` | `ON` when top-level | Builds `muscriptor_tests`. `-DBUILD_TESTING=OFF` also turns it off. |
| `MUSCRIPTOR_BUILD_BENCH` | `ON` when top-level | Builds `muscriptor_bench`. |
| `MUSCRIPTOR_TEST_SIZE`, `_DEVICE`, `_WEIGHT_DTYPE` | `medium`, `gpu`, `f16` | Default run options for the tests and benchmark ([`TESTING.md`](TESTING.md#run-options)). |

Building a GPU backend does not force its use: at run time the library uses the
GPU when `use_gpu` is set and a device is available, and the CPU otherwise.

The build always sets:

- the library, ggml and pffft as static archives; ggml's are checked after the
  fetch
- OpenMP off
- Accelerate and BLAS off, since `Model` runs a single backend with no
  scheduler, which is the only way ggml reaches its BLAS backend
- C++23 as a public compile feature

### `MUSCRIPTOR_GGML_FP32_ACCUM`

On ARM with fp16 vector arithmetic, ggml's F16 matmul kernels sum each row in
fp16 (`GGML_F16_VEC` maps to `float16x8_t`). The flag undefines
`__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` for the `ggml-cpu` target. Those
kernels then use their fp32 fallbacks: F16 weights, fp32 sums.

The CPU test tolerances assume this build. It costs CPU throughput and has no
effect on Metal or Vulkan. The `fast-cpu` preset turns it off, so the two can
be compared with `--parity-report` ([`TESTING.md`](TESTING.md#diagnosing-a-failure)).

The `-U` has to come after ggml's own `-mcpu=...+fp16`. With it applied,
`ggml_cpu_has_fp16_va()` returns 0. A ggml update can reorder the flags, so
check this and re-run the tests on CPU and GPU after changing the pinned
commit.

## Presets

| Preset | Build directory | Settings |
|---|---|---|
| `parity` | `cpp/build` | The defaults |
| `fast-cpu` | `cpp/build-fast-cpu` | `MUSCRIPTOR_GGML_FP32_ACCUM=OFF` |
| `macos-arm64` | `cpp/build-macos-arm64` | `CMAKE_OSX_ARCHITECTURES=arm64` |
| `macos-x86_64` | `cpp/build-macos-x86_64` | `CMAKE_OSX_ARCHITECTURES=x86_64` |

The presets need CMake 3.25 or later, and are run from `cpp/`:

```bash
cd cpp
cmake --preset parity && cmake --build --preset parity
```

## macOS

- **Deployment target.** `CMAKE_OSX_DEPLOYMENT_TARGET` defaults to 11.0 when
  the consuming project hasn't set it. It is a cache entry, so it applies to
  the whole consuming project.
- **One architecture per build tree.** Configuring with more than one
  architecture in `CMAKE_OSX_ARCHITECTURES` fails, because ggml picks its CPU
  kernels once per tree and would silently fall back to generic scalar code.
  For a universal binary, build `macos-arm64` and `macos-x86_64` separately
  and combine the results.

## x86 baseline

x86 builds target Ivy Bridge: SSE4.2, AVX and F16C on; AVX2, FMA and BMI2 off.
These are pinned, and configuring fails if AVX2, FMA or BMI2 reach ggml's
compile flags; ggml would otherwise enable all six when native tuning is off.
The same set is used on every OS.

## Vulkan

- **Build time.** Needs the Vulkan SDK: its headers, and `glslc`, which
  compiles the shaders into the binary. Nothing from the SDK is shipped.
- **Windows compiler.** ggml builds its shader generator as a nested CMake
  project that sees only the environment. Outside a Visual Studio developer
  prompt, set `CC`, `CXX` and `RC`, e.g.:

  ```bash
  CC=clang CXX=clang++ RC=llvm-rc cmake -S cpp -B cpp/build -G Ninja
  ```
- **Run time.** The only dependency is the system loader, `vulkan-1.dll`.
  - With MSVC and clang-cl it is delay-loaded, through INTERFACE link options
    that reach the consuming binary. The library checks it can load the DLL
    before its first Vulkan call, and falls back to the CPU if not.
  - MinGW links the loader directly, so a MinGW binary does not load at all
    without `vulkan-1.dll`.
- **The ggml registry.** ggml's global backend registry initialises Vulkan the
  first time it is used. On Windows, without `vulkan-1.dll`, that raises a
  delay-load exception that its C++ `catch` does not handle. The library never
  touches the registry, and a host on Windows must not either.
- **Debugging.** A `GGML_VULKAN_CHECK_RESULTS=ON` build of ggml compares every
  Vulkan op against the CPU and reports the first one that differs.
