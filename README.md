<div align="center">

# Bato

### ⚠️ DO NOT USE — UNDER HEAVY DEVELOPMENT ⚠️

**Bato is in active, heavy development and is NOT fully tested.**<br>
**The archive format and wire protocols change daily with breaking changes and NO backward compatibility.**<br>
**DO NOT USE FOR CRITICAL DATA OR PRODUCTION STORAGE.**

---

</div>

Bato is an original, lossless, byte-oriented context-mixing compressor and archiver written in pure Rust.

It operates under one strict, non-negotiable policy: **compressed size > decompression time > compression time**. It has only **one compression mode, always the strongest**—no fast/balanced presets, compromise modes, or lossy shortcuts. Files, stdin streams, and directory trees all use it.

## Key Properties

- **Universal Byte Archiver:** Compresses arbitrary byte contents (executables, invalid UTF-8, media, pre-compressed streams, or unknown formats). Decisions depend purely on byte statistics and entropy models—never on file extensions, MIME types, or names.
- **Context-Mixing Engine:** An adaptive ensemble model featuring order 0–7 hashed context maps, a 4-way associative match model, record-stride and row experts, secondary symbol estimators (ISSE), and chained adaptive probability maps (APM).
- **Bit-Identical Determinism:** Pure integer arithmetic everywhere (no floating point, clocks, RNG, or scheduling-dependent float operations). Output is bit-identical across all CPU architectures and thread counts.
- **Built-in Reversible Transforms:** Deflate stream discovery and recompression, x86 and x86-64 branch and RIP-relative address translation, byte delta, and 2D byte planes, all gated by exact byte-cost selection with raw fallback.
- **Integrity & Safety:** CRC32C chunk checksums, SHA-256 identities, directory traversal defense, staged atomic extraction, and bounded memory allocation.

---

## Build

Requires a recent Rust toolchain (Rust 1.89+).

```bash
cargo build --release --locked                           # Existing GPU-capable build
cargo build --release --locked --no-default-features     # Pure CPU build, no GPU dependencies
```

The `gpu` Cargo feature is enabled by default and contains the existing
NVIDIA CUDA and cross-platform wgpu filter implementations. The CPU-only
build retains the same archive format and deterministic CPU transforms.
In a CPU-only build, `--compute auto` and `--compute cpu` use the CPU;
explicit `--compute cuda` or `--compute wgpu` reports an unavailable
backend rather than silently ignoring the request. Dedicated native AMD
HIP, Intel Level Zero and direct Metal backends are **not implemented** yet.


The compiled binary will be placed at `target/release/bato`.

---

## Quick Start

### Single File Compression

```bash
bato input.dat              # Compresses to input.dat.bato
bato -d input.dat.bato      # Decompresses to input.dat
bato -t input.dat.bato      # Verifies archive integrity without writing
bato -i input.dat.bato      # Displays format headers and segment details
```

### Directory Archives

```bash
bato archive my_project/                 # Creates my_project.bato
bato extract my_project.bato -o out/     # Restores directory tree to out/
bato list my_project.bato                # Lists archive catalogue
```

---

## Archive Capabilities

Bato includes built-in features for robust storage and operations on directory archives:

| Feature | Example Command | Description |
| :--- | :--- | :--- |
| **Selective Extraction** | `bato extract proj.bato -o out/ docs/ src/main.rs` | Extracts only the specified relative paths without materializing unwanted files. |
| **Full Encryption** | `bato archive proj/ --password-file key.txt -o full.bato` | Authenticated encryption (age/ChaCha20-Poly1305) protecting data and filenames. |
| **Selective Protection** | `bato archive proj/ --password-file key.txt --protect secrets/` | Keeps general catalogue public while encrypting sensitive subtrees. |
| **Volume Splitting** | `bato archive proj/ --split-size 100M -o split.bato` | Splits archives into fixed-size volume chunks. |
| **Parity & Repair** | `bato archive proj/ --split-size 100M --parity` | Adds Reed-Solomon parity volumes to survive lost or corrupted parts (`repair-split`). |
| **Checkpoint & Resume** | `bato archive proj/ --checkpoint job_dir --pause-file hold` | Allows pausing and resuming large compression or decompression jobs (`--resume`). |
| **Recovery Index** | `bato recovery-index proj.bato -o proj.rec` | Creates an external salvage index to reconstruct damaged archives (`bato recover`). |
| **Archive Diff** | `bato diff a.bato b.bato` or `bato diff a.bato ./dir` | Compares archive contents against another archive or local tree. |

---

## Resource Controls

By default, Bato utilizes all available CPU cores (`--threads 0`) and manages up to 90% of usable RAM (`--memory-budget auto`).

- `--threads N`: Restrict worker thread count.
- `--memory-budget <auto|BYTES>`: Set memory budget (e.g. `2G`, `512M`).
- `--segment-size BYTES`: Set segment geometry (default: 64 MiB).
- `--timing`: Display per-stage time and bit attribution breakdowns.
- `--compute auto|cpu|cuda|hip|level-zero|metal|wgpu` selects the experimental byte-filter backend.
  CPU is the default. Auto tries GPU filters on 8–64 MiB segments and uses the
  CPU when no device is available or a device is busy. Select `cuda` or `wgpu`
  to test eligible filters explicitly and surface device errors.
- CUDA uses the installed NVIDIA driver and the embedded `sm_75` PTX. wgpu
  uses Vulkan on Linux, Direct3D 12 or Vulkan on Windows, and Metal on Apple platforms.
  Auto can use multiple detected devices concurrently when archive workers
  have independent segments. Matching CUDA and wgpu PCI IDs share one slot,
  with CUDA preferred and wgpu available as a fallback. Ambiguous NVIDIA
  adapters are considered for fallback after the CUDA devices are unavailable.
- The device backends apply the existing delta and byte-shuffle transforms.
  The context model, entropy coder and checksums run on the CPU. GPU filters
  preserve the archive format and are designed to produce the same bytes.
- Experimental native AMD HIP, Intel oneAPI Level Zero, and direct Apple Metal
  filter backends are available as **optional Cargo features and vendor SDK
  plugins**. See [native GPU build instructions](native/README.md).
  Native plugin output is checked against the CPU reference on every call.
- `--compute auto` retains CUDA, wgpu and CPU fallback when a native plugin
  is missing, unsupported, or fails validation.
- `--stats` reports successful CUDA and wgpu launches, processed filter bytes,
  and the devices that completed work. Zero launches indicate CPU processing
  for that phase.

Compare both paths after a release build with
`python3 scripts/benchmark_bato_table.py --branch-bin target/release/bato`.
An asterisk marks a phase that used its GPU backend in every measured trial.
Use `--show-usage` to print average CPU use and GPU launch counts.

---

## Benchmarks & Performance (Format 1.22)

All figures below reflect **Bato 0.8.0 (Format 1.22)**, measured on a 4-core, 15 GiB Linux host under default settings. Output is verified byte-for-byte lossless.

### Canterbury Corpus (11 Files, 2,810,784 Raw Bytes)

| Compressor | Compressed Size | vs Bato 1.22 | Pack Time | Unpack Time |
| :--- | :---: | :---: | :---: | :---: |
| `gzip -9` | 730,732 B | +117.6% (Bato is 54.0% smaller) | 0.4s | 0.1s |
| `zip -9` | 732,269 B | +118.0% (Bato is 54.1% smaller) | 0.4s | 0.1s |
| `rar -m5` | 512,223 B | +52.5% (Bato is 34.4% smaller) | 0.9s | 0.2s |
| `7z -mx9` | 494,801 B | +47.3% (Bato is 32.1% smaller) | 1.8s | 0.2s |
| `xz -9e` | 493,080 B | +46.8% (Bato is 31.9% smaller) | 2.1s | 0.2s |
| **Bato 1.22** | **335,885 B** | **Baseline** | **33.99s** | **27.40s** |
| `paq8px -9L` (v217) | **300,270 B** | **-10.6%** (paq8px is smaller) | ~280s | ~280s |

### Real-World Release Trees

Packed as folders at strongest competitor settings (`xz -9e`, `7z -mx9`, `zstd --ultra -22`, `rar -m5`):

| Package / Tree | Unpacked Size (Files) | xz -9e | 7z -mx9 | zstd -22 | **Bato 1.22** | Bato vs Best Other |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Antigravity IDE 2.6.0 | 595.5 MB (15,898) | 124.4 MB | 115.5 MB | 129.9 MB | **105.9 MB** | **-8.3%** |
| Node.js v22.23.4 | 196.5 MB (4,750) | 31.0 MB | 29.4 MB | 33.5 MB | **25.2 MB** | **-14.3%** |
| Go 1.27.3 | 244.1 MB (15,657) | 39.7 MB | 38.6 MB | 42.4 MB | **33.8 MB** | **-12.4%** |

### Honest Trade-offs

- **Size (Primary Goal):** Bato produces significantly smaller archives than standard dictionary/LZMA archivers (`7z`, `xz`, `rar`, `zstd`, `zip`), saving 8% to 54% in compressed bytes across text, code, executables, and containerized formats. On raw single-stream text, specialized multi-gigabyte models like `paq8px` achieve ~10% better ratios; however, Bato outperforms `paq8px` on mixed trees and containers (such as `.tar.gz` and Office documents) thanks to native deflate recompression.
- **Decompression Speed (Open Gap):** Because context mixing must update its neural mixers and adaptive state sequentially for every decoded bit, decompression speed is roughly comparable to compression speed. While SIMD AVX2 kernels and memory cache optimizations in Format 1.22 improved throughput by over 30%, Bato remains substantially slower than dictionary compressors whose decompression consists of simple LZ77 byte copies.

---

## Documentation

- [Format Specification (`docs/format.md`)](docs/format.md): Wire format, segment and block headers, context models, and arithmetic coding.
- [Validation & Benchmark Log (`docs/validation.md`)](docs/validation.md): Detailed logs of all versions, experimental ablations, and competitor comparisons.
- [Benchmark Methodology (`docs/benchmark-iterations.md`)](docs/benchmark-iterations.md): Reproduction steps, frozen workload manifests, and measurement procedures.

## License

Dual-licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
