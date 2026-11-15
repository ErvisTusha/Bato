# Bato: improve the existing GPU branch (`claude/focused-faraday-t2c712`)

## Context

The branch adds CUDA, wgpu, HIP, Level Zero and Metal backends to Bato. A read-only
audit of the remote tip (`d533d1a`, format 1.22) found:

- **GPU work is five byte filters only** (delta 1/2/4, shuffle 2/4). All context
  mixing (CM), which dominates encode and decode, is CPU. GPU share of encode is
  estimated well under 1%; this is not yet measured.
- **`auto` rarely reaches the GPU**: segments under 8 MiB stay on CPU
  (`gpu/operations/filters.rs:8`) and inputs up to 64 MiB are cut into 4 streams
  (`types.rs:405`), so only inputs of about 28 MiB and up dispatch.
- **CUDA and wgpu results are unverified** (`gpu/scheduler.rs:216,251`); native
  plugins are verified by a full CPU pass, so they can never beat CPU.
- **Waste**: the raw segment is uploaded up to 6 times per encode (`fast.rs:85`,
  `segment.rs:453`), one a recomputation; buffers (and Metal pipelines, Level Zero
  queues, CUDA contexts on slot hops) are recreated per call.
- **About 800 lines are not archive code**: `gpu/matvec.rs`, `gpu/streams.rs`,
  their shaders and CUDA kernels, used only by `examples/gpu_probe.rs`.
- **Largest measured gap is size**: Canterbury 335,885 B vs paq8px `-9L`
  300,270 B (Bato 10.6% larger; worst on `kennedy.xls`, `ptt5`).
- **Decode**: every segment rebuilds a Predictor and retrains the 104 KB primer
  (`segment.rs:817`); primer training is 34% of unpack time on the 24-file set.
- **Priority conflict on the branch**: commit `593e3ce` (auto 4 streams for
  inputs of 2 MiB or more) trades about +3.6% size for 3x decode on its own
  test file. AGENTS.md rule 1 forbids size loss for speed.

Consequence: the real gains are in the CPU engine and in what the GPU is asked to
do, not in speeding up the existing filters. The plan measures first and keeps
only measured wins; GPU code that cannot show a benefit is removed, not polished.

## Status (pushed to the branch)

- `4dd46e6`: `scripts/benchmark_bato_table.py` stores every result in SQLite
  (`scripts/bato_tools/table_store.py`, default
  `~/data-research/bato-table.sqlite3`, `--database`). A verified result for
  the same binary hash, input, compute mode, limits, geometry and hardware is
  reused, so unchanged `main` runs once per input. Reused cells show `=` and
  their date; `--refresh` re-measures; failures are stored, never reused.
  Also `--input PATH` and `--gpu-compute MODE` (third column as `auto`,
  `cuda` or `wgpu`).
- `scripts/ablate.py` works again after the predictor split; all 162 script
  tests, `cargo fmt --check`, clippy and `cargo test --release` pass on a
  4-CPU host without a GPU (GPU tests skip there).
- `scripts/benchmark_suite.py` is never run as a command; the table script
  imports it as its measurement and verification library.
- No benchmark evidence exists yet: the cloud host had no GPU and could not
  fetch the corpus. Everything below is still to do, starting at Phase 1.
- Owner decision to confirm before Phase 3 step 1: commit `593e3ce` records
  the 4-stream split as a size-for-speed trade "by owner decision". Do not
  revert it without the owner's word.
- Findings that change steps below: a decode-side primed-model cache
  (Phase 4 step 1) only helps when several segments share one length and
  decode one after another, so expect little from it; deleting the
  matvec/streams CUDA kernels needs `nvcc` to regenerate
  `native/cuda/kernels.ptx`.

## Where this runs

In a Claude Code session on `et` (GTX 1650 4 GB, driver 595.99; Intel iGPU to be
confirmed with `lspci`/`vulkaninfo`). The cloud container has no GPU or Vulkan
driver. Start by `git pull --ff-only` on the branch (local was 7 commits behind).
Never rebase onto or merge `main` (no common history); never force-push.

Untested and to be reported as such: AMD (HIP), Apple (Metal), Level Zero unless
the Intel SDK plugin is built, and any multi-NVIDIA setup.

## Phase 1: audit baseline and measurement (no behaviour change)

1. Build: branch tip (default features), branch `--no-default-features`, and
   `origin/main` in a separate worktree as reference binaries. Record SHAs and
   binary hashes in the ignored checkpoint.
2. Corpus: the table script's corpus on `et`
   (`/home/et/data-research/corpus/real`), fixed `--seed`. Add `--input PATH`
   (repeatable) to the table script so named inputs can be benchmarked
   besides the random sample: Canterbury and Calgary files (via
   `scripts/download_corpus.py`) and one large input of at least 64 MiB, the
   only case where GPU code is reachable; label it separately.
3. Telemetry (first code change): extend the existing `--stats` counters in
   `src/core/types.rs:118-142` with host-to-device and device-to-host bytes,
   transfer and kernel wall time, and allocation count, with one meaning of
   "dispatch" (kernel launch) across backends. VRAM via `nvidia-smi` sampling in
   the table script; "unknown" elsewhere.
4. **Only benchmark tool (owner instruction):
   `scripts/benchmark_bato_table.py`. Do not run `scripts/benchmark_suite.py`
   or `compare_benchmarks.py` as commands.** (The table script imports
   `benchmark_suite` as a library for measurement and verification; that
   stays, nothing is duplicated.) It builds `origin/main`, rebuilds the
   local branch binary, and prints Main CPU / Branch CPU / Branch auto (all
   GPU + CPU) per input with real CUDA, native and wgpu dispatch marks. Its
   default corpus is `/home/et/data-research/corpus/real` on `et` (5 sampled
   files + 2 directories). Baseline run:
   `python3 scripts/benchmark_bato_table.py --seed <S> --warmups 1 --repetitions 5 --show-usage`
   and keep the printed seed fixed for every later run so each accepted
   change is compared on identical inputs. For changes that only alter the
   branch, the "before" is the previous branch commit, run with
   `--main-revision <previous SHA>`. Extend this script (not a new tool) with
   the new transfer/launch/VRAM columns from step 3 and, if missing, explicit
   `cuda`-only and `wgpu`-only columns.
5. **SQLite result store in the table script (owner instruction), done before
   the first baseline run so `main` is measured once per file.**
   - New module `scripts/bato_tools/table_store.py` (plain `sqlite3`), default
     path `~/data-research/bato-table.sqlite3` (next to `DEFAULT_DATABASE` in
     `scripts/bato_tools/database.py`; ignored, outside the repo), flag
     `--database PATH`.
   - One row per measured trial. Cache key: binary SHA-256 (already in
     `build.json` via `cached_build`), input content identity (file SHA-256,
     or the directory inventory hash from `bato_tools.workloads.identify`),
     compute mode (`cpu`/`auto`/`cuda`/`wgpu`), threads, memory budget,
     segment and chunk size, warmups, and a hardware identity (CPU model,
     core count, RAM, GPU list).
   - Stored values: git revision, timestamp, archive bytes, pack/unpack wall
     and CPU time, peak RAM, VRAM, dispatch/transfer counters, verification
     status.
   - Lookup: before running a variant, if the key has at least
     `--repetitions` verified trials, reuse them and skip the run. This
     covers `main` and also any unchanged branch binary. Failed or unverified
     trials are stored but never satisfy a lookup.
   - Honest reporting (AGENTS.md): reused cells are marked in the table (e.g.
     `=`) with a footer giving the stored run date; never presented as fresh.
     `--refresh` forces a re-measure. Keying on the binary hash means a new
     `main` commit that changes the binary is measured again automatically.
   - Before/after for a branch change: the "before" is the previous branch
     binary's stored rows; add `--compare-revision SHA` to print it as an
     extra column from the store without re-running.
   - Unit tests in `scripts/tests/` for hit, miss, hardware/settings mismatch,
     failed-trial exclusion and `--refresh`.
   Variants per input: `main` cpu; branch `cpu`, `cuda`, `wgpu` (NVIDIA, then
   Intel), `auto` (both GPUs). `cmp` every archive across backends.
6. Fix stale `docs/benchmark-iterations.md:91` ("Bato has no GPU path").

Deliverable: a baseline table (size, pack/unpack wall and CPU, RAM, VRAM,
launches, transfer bytes) per configuration.

## Phase 2: GPU path, keep only what measures faster

Order matters; each step is benchmarked against the previous one.

1. **Correctness first**: verify CUDA and wgpu output. On decode the chunk CRC
   already catches errors, so add a CPU retry instead of a hard error
   (`segment.rs:606-608`). On encode, check the forward result by a cheap
   integer checksum computed on both sides, falling back to CPU on mismatch.
2. **One upload per segment**: a single batched call that uploads the segment
   once, runs all 5 forward filters into one device buffer, and reads back once;
   reuse the winner instead of recomputing (`fast.rs:85`, `segment.rs:453`).
   Run the CPU candidates (LZ, Huffman, numeric on raw bytes) while the GPU
   works (async submit, join before use).
3. **Reuse**: per-slot persistent buffers, cached CUDA function handles and
   contexts (no reopen on slot hop, `scheduler.rs:335`), wgpu buffer and
   bind-group pool; cache Metal pipelines and the Level Zero queue.
4. **Scheduling**: match native devices against wgpu slots by PCI ID to stop
   double counting; pick slots by measured throughput rather than `w % n`.
5. **Candidate search on GPU (size)**: on-device order-0/order-1 byte
   histograms for many more filter parameters (delta and shuffle widths 1..16,
   record strides) returning only scores; the best few then get the existing
   exact CPU trial. CPU fallback evaluates the same candidate set so output is
   identical; kept only if archives get smaller on the fixed input set.
6. **Long-distance matches**: GPU fingerprinting of fixed blocks to find
   repeats beyond the matcher's 2^22-entry index, fed to the CM match model
   and to partial dedup in archives. Same rule: CPU reference implementation
   first, identical bytes, kept only for a measured size gain.
7. **Decision gate**: if after steps 1-4 no backend beats `--compute cpu` on
   the large workload (PCIe transfer costs about a CPU filter pass), remove the
   filter offload from the default path and keep GPU use only where step 5 or 6
   earns size. Delete `gpu/matvec.rs`, `gpu/streams.rs`, their shaders/kernels
   and the probe example regardless, plus the duplicated `Direction` enum and
   pass-through files.

## Phase 3: archive size (largest gap, highest priority)

One change per experiment, format version bump each time it changes bytes:

1. **Auto-streams** (`types.rs:405`): measure size cost on the fixed input set.
   Per AGENTS.md rule 1, revert the split unless size is equal; recover the
   decode speed instead via Phase 4. In archives, cut only at file boundaries
   if any split survives.
2. Learned blending of the 8-APM chain and the two final mixers instead of
   fixed 1:1 averages (`predictor/calibration.rs:63`, `mixing.rs:145`).
3. Adaptive match-model counter (length bucket x expected bit) and a second
   long-minimum match model (`models/matcher.rs:121-140`).
4. 16-bit slot check in the hash table (`table.rs:138`).
5. Mixer learning-rate annealing and wider weights (`mixer.rs:10-16`).
6. Targeted work on `kennedy.xls` and `ptt5` (record and row models) guided by
   `scripts/ablate.py`.

Do not repeat experiments rejected in `docs/validation.md:737-871` (extra
orders, smaller primer, skipping the plain trial, sampled estimates, etc.).

## Phase 4: CPU and decode speed (never at a size cost)

1. Cache one primed Predictor per process and clone it per segment on decode
   (the encoder already snapshots, `segment.rs:666`); widen `SNAPSHOT_BYTES`.
2. Fuse the three first-layer dot products and training over the shared 48
   inputs (`simd.rs`, `mixing.rs`).
3. Rolling hashes for order contexts (`models/order.rs:8`).
4. Encode: run the `fast.rs:85` candidate loop and the deflate scan in
   parallel; enable the 3-thread pipeline whenever cores are idle
   (`segment.rs:142`).
5. NEON kernels and `prfm` prefetch for aarch64 (compile- and parity-tested in
   CI only; untested for speed without ARM hardware).

Each is a pure refactor for bytes: archives must `cmp` identical to the
previous commit.

## Phase 5: verification and delivery

- Gates from AGENTS.md before every commit: `cargo fmt --check`, `cargo clippy
  --release --all-targets`, `cargo test --release`, `python3 -m unittest
  discover -s scripts/tests`, `scripts/run_tests.py --bato-bin
  target/release/bato --self-contained` (competitors are not installed for the
  default mode), plus `--no-default-features` build and tests.
- Parity: `BATO_TEST_REQUIRE_CUDA=1 cargo test --release` on `et`; a new
  randomized and adversarial test (empty, 1 byte, sizes around word, tile and
  64 MiB limits, all byte values, fixed-seed integer hash data) comparing
  archives from `cpu`, `cuda`, `wgpu`, `auto` byte for byte, and round-tripping
  each.
- Per accepted change: the `benchmark_bato_table.py` table (same seed,
  previous commit vs candidate, `main` reused from the SQLite store), exact
  bytes and CPU deltas, regressions
  listed; row added to `docs/validation.md`; `docs/format.md` updated on format
  bumps (lines 1, 4, 17, 108-115, 308, 336).
- Commit each verified improvement to `claude/focused-faraday-t2c712` with
  `git push -u origin` (fast-forward only, re-fetch first). Rejected
  experiments are not committed. No PR and no merge to `main`.
- Final report: wins, ties, losses per workload and metric for one declared
  configuration, competitor baselines (paq8px etc. if installable on `et`),
  and the explicit list of untested hardware.

## Critical files

`src/core/gpu/{scheduler.rs,device.rs,backend/cuda/filters.rs,backend/wgpu/filters.rs,backend/native.rs,operations/filters.rs}`,
`native/*`, `src/core/cm/{fast.rs,filters.rs,segment.rs,table.rs,mixer.rs,simd.rs}`,
`src/core/cm/predictor/{mixing.rs,calibration.rs,pipeline.rs}`,
`src/core/cm/models/{matcher.rs,order.rs}`, `src/core/types.rs`,
`scripts/benchmark_bato_table.py`, `scripts/bato_tools/table_store.py` (new),
`docs/{format.md,validation.md,benchmark-iterations.md}`.
