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
| `MUSCRIPTOR_METAL` | `ON` on Apple | Builds the Metal backend. |
| `MUSCRIPTOR_METAL_PRECOMPILED` | `ON` with Metal | Compiles the Metal shaders into `.metallib` files at build time, which the executable ships. `OFF` embeds the shader source and compiles it at run time ([Metal](#metal)). |
| `MUSCRIPTOR_VULKAN` | `ON` elsewhere, when the Vulkan SDK is found | Builds the Vulkan backend. |
| `MUSCRIPTOR_BUILD_TESTS` | `ON` when top-level | Builds `muscriptor_tests`. `-DBUILD_TESTING=OFF` also turns it off. |
| `MUSCRIPTOR_BUILD_BENCH` | `ON` when top-level | Builds `muscriptor_bench`. |
| `MUSCRIPTOR_TEST_SIZE`, `_DEVICE`, `_WEIGHT_DTYPE` | `medium`, `gpu`, `f16` | Default run options for the tests and benchmark ([`TESTING.md`](TESTING.md#run-options)). |

Building a GPU backend does not force its use: at run time the caller picks a
device, or leaves it to Auto, which uses the CPU when no GPU is available
([`API.md`](API.md#devices)).

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

## Metal

The shaders reach the GPU in one of two ways, chosen by
`MUSCRIPTOR_METAL_PRECOMPILED`.

- **`ON`, the default: precompiled.** The build compiles them into
  `default.metallib` and, with the macOS 26 SDK or later,
  `ggml-tensor.metallib`. The executable must ship both.
- **`OFF`: compiled at run time.** The shader source is embedded in the binary
  and compiled when Metal is first initialised. macOS caches the result per
  host application. On a cache miss (the first run in each application, and
  again after an OS update or a ggml update) this takes about 20 s. It happens
  in the process's first device enumeration, `availableDevices()` or the first
  load, and neither can be cancelled during it.

M1 Pro, `medium`, on the 15 s fixture, one run each. "Cold" is a fresh
application bundle id, so macOS has nothing cached. Devices and Load are from
`muscriptor_bench --load-profile`, Transcribe from `--transcribe`, in separate
runs:

| | Devices, cold | Devices, warm | Load, cold | Load, warm | Transcribe, cold | Transcribe, warm |
|---|---|---|---|---|---|---|
| Precompiled | 88 ms | 31 ms | 239 ms | 242 ms | 9.89 s | 9.67 s |
| Compiled at run time | 19 886 ms | 42 ms | 243 ms | 244 ms | 9.91 s | 9.64 s |

Both runs produced the same number of notes. Either way, the GPU-specific code
is generated on the user's machine, the first time each kernel runs.

**Building.** Needs Xcode's Metal toolchain; configuring fails without it:

```bash
xcodebuild -downloadComponent MetalToolchain
```

`default.metallib` is built for `CMAKE_OSX_DEPLOYMENT_TARGET`. The tensor API
kernels in `ggml-tensor.metallib` need macOS 26 and run only on M5-class GPUs.
With an older SDK the file is not built, configuring warns, and those GPUs use
ggml's other kernels. The files come to about 15 MB, and the build compiles
the flash-attention kernels for about 18 s.

**Shipping.** ggml looks for `default.metallib` in the `Resources` of the bundle
that contains its code, then next to the running executable. After
`add_subdirectory`, call:

```cmake
muscriptor_add_metal_library(<target>)
```

for each executable, app or plugin target that links the library. A bundle
target gets the files in `Contents/Resources`; any other executable gets them
copied beside it. It does nothing when `MUSCRIPTOR_METAL_PRECOMPILED` is off.

**A missing `default.metallib` is not reported as such.** `availableDevices()`
still lists the Metal device, but it fails to initialise: a load that names it
returns `Error::DeviceUnavailable`, and Auto runs on the CPU, several times
slower. Check that the files are in the shipped bundle, or that
`device().backend` is `"Metal"` after an Auto load. The test suite checks it.

**ggml's Metal class.** With `MUSCRIPTOR_METAL_PRECOMPILED` on, the build
renames ggml's `GGMLMetalClass`, the Objective-C class whose bundle ggml
searches. Class names are shared across a process, so another copy of ggml in
the same host could otherwise point the search at its own bundle.

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
- **clang-cl iterator debugging.** With clang-cl, ggml compiles its Vulkan
  backend with `_ITERATOR_DEBUG_LEVEL=0`. Everything linked with it must use the
  same level, or a Debug link fails on `/failifmismatch`. The library builds
  with it and passes it to consumers through `muscriptor_ggml`. A consumer's
  own C++ static libraries that don't link `muscriptor_ggml` need it too.
- **Run time.** The only dependency is the system loader, `vulkan-1.dll` on
  Windows and `libvulkan.so.1` on Linux.
  - With MSVC and clang-cl it is delay-loaded, through INTERFACE link options
    that reach the consuming binary. The library checks it can load the DLL
    before its first Vulkan call, device enumeration included, and lists no
    Vulkan devices if it cannot.
  - MinGW links the loader directly, so a MinGW binary does not load at all
    without `vulkan-1.dll`.
  - On Linux the binary does not link the loader. `ggml-vulkan`'s link to it is
    dropped, and the few Vulkan functions ggml calls directly are defined in
    `cpp/src/vulkan_loader_linux.cpp`, which opens `libvulkan.so.1` at run
    time. Without the loader, or without a Vulkan driver, the library lists no
    Vulkan devices. If a ggml update calls another Vulkan function directly,
    executables (the tests, the benchmark) fail to link until it is added there.
    A shared library only fails that way when linked with `-Wl,-z,defs`.
- **The ggml registry.** ggml's global backend registry initialises Vulkan the
  first time it is used. On Windows, without `vulkan-1.dll`, that raises a
  delay-load exception that its C++ `catch` does not handle. The library never
  touches the registry, and a host on Windows must not either:
  `availableDevices()` enumerates through the Vulkan backend's own registry,
  behind the same loader check.
- **Device list.** ggml enumerates Vulkan devices once per process, so a GPU
  that appears or disappears later is seen only after a restart.
  `GGML_VK_VISIBLE_DEVICES` (comma-separated physical device indices) narrows
  the list, which is handy for testing a device that has gone away.
- **Debugging.** A `GGML_VULKAN_CHECK_RESULTS=ON` build of ggml compares every
  Vulkan op against the CPU and reports the first one that differs.
