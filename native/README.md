# Optional vendor-native GPU filters

These are **experimental native filter implementations**, independently built
with each vendor's SDK. They are *not* implementations of Bato's context model,
entropy coder or full GPU compressor. All paths retain byte-for-byte lossless
results and Bato's single strongest compression mode.

| Backend | Native API | Plugin build | Cargo feature | CLI |
| --- | --- | --- | --- | --- |
| NVIDIA | CUDA Driver API + embedded PTX | Included in Rust | `cuda` (default) | `--compute cuda` |
| AMD | ROCm/HIP kernels | `scripts/build_native_backends.sh hip` | `hip` | `--compute hip` |
| Intel | oneAPI SYCL **Level Zero** backend | `scripts/build_native_backends.sh level-zero` | `level-zero` | `--compute level-zero` |
| Apple M-series | Direct Metal API and MSL | `scripts/build_native_backends.sh metal` | `metal` | `--compute metal` |
| Cross-platform fallback | Vulkan, Metal or D3D12 through wgpu | Included in Rust | `wgpu` (default) | `--compute wgpu` |
| CPU | Rust integer filters | Included | no GPU feature | `--compute cpu` |

## Build

All GPU implementations are feature-isolated. The Rust code lives under
`src/core/gpu/backend/{cuda,wgpu}/` and
`src/core/gpu/backend/{hip,level_zero,metal}.rs`; vendor-specific compiled
kernels live under `native/{cuda,hip,level_zero,metal}/`.
Device scheduling is common to all backends.

`cargo build --release` now selects **all five GPU features**
(`cuda,hip,level-zero,metal,wgpu`) plus CPU and uses `--compute auto` by
default. The `gpu` umbrella feature remains an alias for CUDA + wgpu.
Features are independently selectable for developer/QA testing; end users
use the same executable for every GPU brand on their OS/architecture.
Build a backend without the others by passing `--no-default-features`.
For cross-vendor fallback on the same binary, list explicit features such as
`--no-default-features --features hip,wgpu`.

Normal Bato builds do not need a vendor-native GPU SDK. On a matching GPU
development host with the SDK and driver installed:

```bash
# AMD / Linux (ROCm HIP SDK)
bash scripts/build_native_backends.sh hip
BATO_HIP_LIBRARY="$PWD/target/native/libbato_hip.so" \
  cargo run --release --no-default-features --features hip -- --compute hip input.bin

# Intel / Linux (oneAPI DPC++ with Level Zero GPU driver)
bash scripts/build_native_backends.sh level-zero
BATO_LEVEL_ZERO_LIBRARY="$PWD/target/native/libbato_level_zero.so" \
  cargo run --release --no-default-features --features level-zero -- --compute level-zero input.bin

# Apple M-series / macOS (Xcode command line tools)
bash scripts/build_native_backends.sh metal
BATO_METAL_LIBRARY="$PWD/target/native/libbato_metal.dylib" \
  cargo run --release --no-default-features --features metal -- --compute metal input.bin

# Existing NVIDIA implementation (no NVIDIA Toolkit required at build time)
cargo run --release -- --compute cuda input.bin

# Cross-platform GPU fallback
cargo run --release -- --compute auto input.bin

# No GPU dependencies (CPU-only)
cargo build --release --no-default-features
```

Enable multiple native features in one Bato build with
`--no-default-features --features hip,level-zero,metal,wgpu`. Runtime libraries are loaded only for
features compiled in; missing libraries or unsupported devices do not prevent
a normal `--compute auto` run from using the existing CUDA, wgpu or CPU path.
An explicit native `--compute hip`, `level-zero` or `metal` request reports
the native error rather than pretending that native work ran.

The compatibility `gpu` feature enables both CUDA and wgpu; neither is
required for native-only builds. Runtime fallback uses only the backends
actually compiled into the binary. Native toolchain support
is currently intended for Linux HIP/Intel oneAPI and macOS Metal; Windows GPU
users should use CUDA or wgpu until vendor-specific native builds are tested.

## One executable for all GPU brands

Bato ships one **platform-specific executable** (`bato` or `bato.exe`) for
all supported GPU manufacturers and the CPU. A Linux executable does not run
on Windows or macOS, and x64/ARM64 remain separate CPU architectures.

All five Rust backend interfaces are compiled by default. CUDA's PTX and wgpu
shaders are embedded in Rust already. Native SDK libraries for AMD HIP,
Intel Level Zero, and Apple Metal can be **embedded as bytes in the same
executable**: Rust extracts them into a private temporary directory for
dynamic loading and removes them after the native module unloads. No separate
GPU plugin download or installation step is needed for such a release.

The system still needs the device's compatible GPU driver. Vendor SDK
runtime dependencies (e.g. ROCm or Intel oneAPI components) may also be
needed by native libraries; if unavailable, Bato auto-falls back to wgpu
(Vulkan/D3D12/Metal) or CPU. This is not a claim that the GPU driver or the
full vendor runtime is embedded.

### Linux all-native release (SDK-equipped build machine)

```bash
bash scripts/build_native_backends.sh hip
bash scripts/build_native_backends.sh level-zero
BATO_BUNDLE_HIP_LIBRARY="$PWD/target/native/libbato_hip.so" \
BATO_BUNDLE_LEVEL_ZERO_LIBRARY="$PWD/target/native/libbato_level_zero.so" \
BATO_REQUIRE_BUNDLED_NATIVE=1 cargo build --release --locked --bin bato
```

### Apple Silicon / macOS release

```bash
bash scripts/build_native_backends.sh metal
BATO_BUNDLE_METAL_LIBRARY="$PWD/target/native/libbato_metal.dylib" \
BATO_REQUIRE_BUNDLED_NATIVE=1 cargo build --release --locked --bin bato
```

The strict `BATO_REQUIRE_BUNDLED_NATIVE=1` flag rejects builds missing
the platform's required native plugins. On Linux and Windows it requires
HIP and Level Zero; on macOS it requires direct Metal. For Windows,
native vendor SDK plugins must first be built with a compatible Windows
toolchain; the supplied Linux build script does not produce DLLs.

A normal `cargo build --release --locked` works without vendor SDKs:
CUDA, wgpu, all native plugin loaders, and CPU are included. Windows/Linux
GitHub-hosted universal artifacts currently do **not** embed HIP/Level Zero,
because those hosted runners lack the required SDKs; AMD and Intel run through
wgpu instead. macOS universal artifacts **do** build and embed Metal through
Xcode. Release packaging will not falsely label an SDK-free binary
"all-native".

The `bato --help` output lists which GPU backends were compiled and which
native vendor libraries were actually bundled. Runtime `--stats` shows
which devices really executed filters.

### Universal benchmark

```bash
python3 scripts/benchmark_suite.py --only-bato --bato-compute auto --bato-bin target/release/bato
python3 scripts/benchmark_bato_table.py --branch-bin target/release/bato
```

Both benchmarks use `auto` by default for current builds. The table
also measures a CPU baseline and reports CUDA/wgpu/native launch counts.
GPU discovery does not guarantee a speedup, particularly for small segments.

## Safety and performance

Native plugins use the versioned header
[`native/include/bato_gpu_backend.h`](../native/include/bato_gpu_backend.h).
This C ABI is deliberately separate from Bato's on-disk format. The native
library implements five operations: delta distances 1/2/4 and 2/4-byte shuffle,
both forward and inverse. The inverse delta uses a three-pass, tile-local
prefix sum plus carry propagation.

**Every native output is verified byte-for-byte against Bato's CPU filter**
before it can enter an archive or be returned from extraction. This protects
correctness on untested hardware but makes early native runs more expensive.
Native libraries, drivers, and GPU hardware are **not available in the
development session that authored these modules**. Source was written and
committed, but vendor-SDK compilation, GPU parity testing, and performance
validation must run on actual AMD, Intel, and Apple machines.

Bato's existing `--compute auto` path can select configured native devices,
but falls back to CUDA/wgpu/CPU for absent or failed native plugins. Benchmark
with the canonical repository scripts before treating any native path as a
performance win. Do not enable extra vendor features in release builds until
their parity and decode-speed measurements meet `AGENTS.md` requirements.

## CUDA kernel source and PTX

The native NVIDIA CUDA source and embedded PTX are now stored together in
`native/cuda/kernels.cu` and `native/cuda/kernels.ptx`. The Rust Driver
API wrapper is in `src/core/gpu/backend/cuda/`. The exact existing PTX and
its source SHA-256 header were retained, so the archive codec does not need
an NVIDIA Toolkit at build time. When changing the CUDA source, run:

```bash
bash scripts/regenerate_cuda_ptx.sh
cargo test --no-default-features --features cuda ptx_is_generated_from_the_current_source
```

The CUDA source-hash parity test checks that the embedded PTX corresponds
to the current CUDA source.
