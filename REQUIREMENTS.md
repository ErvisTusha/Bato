# Bato Requirements & Dependencies

This document outlines all requirements and dependencies for developing, building, testing, and benchmarking the Bato compression engine.

---

## 1. Rust Compiler & Toolchain (Bato Engine)

Bato's compression engine is implemented in pure Rust (2021 edition) with no external compressor dependencies.

| Component | Minimum Version | Purpose |
| :--- | :--- | :--- |
| **rustc** | `1.89+` (see `rust-version` in `Cargo.toml`) | Rust compiler |
| **cargo** | `1.89+` | Build manager & package manager |
| **rustup** | Latest | Toolchain installer & target manager |
| **musl target** *(optional)* | `x86_64-unknown-linux-musl` | Static, portable CPU-only builds |
| **musl-tools** *(optional)* | Latest | C toolchain for musl target linking |

### Installation

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Optional: Add musl target for portable CPU builds
rustup target add x86_64-unknown-linux-musl
sudo apt-get install -y musl-tools
```

---

## 2. Python Environment (Scripts & Benchmarks)

Bato's benchmarking, corpus management, and verification scripts live in `scripts/`. They are strictly designed to run on the standard library with zero mandatory third-party packages.

| Component | Minimum Version | Purpose |
| :--- | :--- | :--- |
| **python3** | `3.8+` (tested with `3.14`) | Runs benchmark runner, corpus downloader, and verification |
| **Standard Library** | Built-in | `argparse`, `json`, `sqlite3`, `pathlib`, `struct`, `urllib`, `tarfile`, `zipfile`, `unittest`, `dataclasses` |
| **pytest / ruff** *(optional)* | Latest | Optional testing and linting tools |

---

## 3. External Archiver Competitors (Benchmark Suite)

Bato's evaluation suite compares Bato against industry-standard compressors under identical execution profiles (`fast`, `default`, `high` / `extreme`).

### Standard Package Competitors

These can be installed directly through system package managers:

| Codec | Binary / Command | Debian / Ubuntu Package | Notes |
| :--- | :--- | :--- | :--- |
| **7-Zip** | `7zz` or `7z` | `p7zip-full` | High profile uses `-mx=9` |
| **ZIP** | `zip` & `unzip` | `zip`, `unzip` | Native info-zip (Python fallback if absent) |
| **Gzip** | `gzip` | `gzip` | Standard DEFLATE `-9` |
| **Bzip2** | `bzip2` | `bzip2` | Burrows-Wheeler `-9` (Python fallback if absent) |
| **XZ** | `xz` | `xz-utils` | LZMA2 `--extreme -9` |
| **Zstandard** | `zstd` | `zstd` | `--ultra -22` |
| **LZ4** | `lz4` | `lz4` | `-12` |
| **Tar** | `tar` | `tar` | Used for common-tar comparison tracks |
| **ZPAQ** | `zpaq` | `zpaq` | Journaling archiver `-method 5` |

### Specialized Competitors

| Codec | Binary | Source / Notes |
| :--- | :--- | :--- |
| **RAR** | `rar` | Official CLI from [RARLAB](https://www.rarlab.com/download.htm) (`unrar` alone cannot compress) |
| **KGB** | `kgb` | KGB Archiver v1.0 CLI (PAQ-based) |
| **PAQ8L** | `paq8l` | Matt Mahoney PAQ8L executable placed on `PATH` |
| **PAQ8PX** | `paq8px` | Jan Ondrus PAQ8PX executable placed on `PATH` |

---

## 4. Verification

You can verify which tools are present and ready on your system at any time by running:

```bash
# Check status of all benchmark competitors
python3 scripts/benchmark_suite.py --check-tools --allow-unavailable

# Run Bato's Rust and Python test suite
cargo test
python3 -m unittest discover -s scripts/tests -p 'test_*.py'
```

---

## 5. Quick Setup for Debian / Ubuntu

```bash
# 1. System packages & standard archivers
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    musl-tools \
    python3 \
    p7zip-full \
    zip \
    unzip \
    gzip \
    bzip2 \
    xz-utils \
    zstd \
    lz4 \
    tar \
    zpaq \
    kgb

# 2. Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add x86_64-unknown-linux-musl
```


### Optional extreme neural/context-mixing competitors

- **CMIX**: build [byronknoll/cmix](https://github.com/byronknoll/cmix), executable `cmix` on PATH; at least 32 GiB RAM recommended. Single-file input and no common `-9` quality switch.
- **NNCP**: obtain [Bellard NNCP v3.3](https://bellard.org/nncp/), executable `nncp` on PATH; the benchmark uses CPU and `--profile default` unless explicitly configured. For supported NVIDIA GPUs use `--nncp-cuda`; the `enwik8/enwik9` profiles are text-specific.
- **PAQ8PX and ZPAQ** were already supported; PAQ8PX high can exceed 32 GiB. All four must pass independent capability round-trip checks before results are accepted.

`python3 scripts/benchmark_suite.py --check-tools --codecs cmix,nncp,paq8px,zpaq --tool-check-timeout 600 --allow-unavailable`

Use `--memory-budget` for the process address-space cap, `--timeout` for each measured phase, and `--nncp-profile` to explicitly freeze NNCP model choice. Missing GPU, peak process RAM, or timing data remain unknown, not performance wins.
