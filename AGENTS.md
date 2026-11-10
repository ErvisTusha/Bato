# Working on Bato

## Owner priorities (non-negotiable)
1. **Order: compressed size > decompression time > compression time.** The
   higher goal wins every conflict. No size loss for speed; no slower
   decoding for faster encoding.
2. **One mode, always the strongest.** No fast/balanced/selectable modes,
   profiles, or header fields and options that can hold only one value.
3. **Best size and best decompression time** against every tested
   competitor. Target: smaller than paq8px at its strongest setting;
   decompression as fast as possible, ZIP speed as the aspiration. Speed
   work goes to decoding first and never costs size.
4. **Report honestly:** name the competitor, data and margin of every loss.
   Never claim an unmeasured win.

## Repo map
- `src/main.rs` CLI only · `src/lib.rs` public API · `src/core/types.rs`
  format constants, options, errors.
- `src/core/frame.rs` stream framing · `archive.rs` directory catalogue ·
  `codec.rs` memory admission and worker pool · `safe_fs/` anchored output
  and staged extraction · `memory/` budgets and platform probes.
- `src/core/cm/` model: `predictor.rs` ensemble, `models/` experts
  (order, word, match, record, neighbours, stride, rows), `mixer.rs`, `apm.rs`,
  `isse.rs`, `simd.rs` kernels, `segment.rs` chunk coding, `x86.rs`
  transform.
- `src/core/cm/profile.rs` step timing behind `bato --timing`;
  `scripts/ablate.py` size cost of each model step. Use both before keeping or
  adding a step: its time share against its size gain.
- `docs/format.md` wire format · `docs/benchmark-iterations.md` measurement
  procedure · `scripts/` benchmark tooling and its tests.

## Engine
- Pure Rust; never call another compressor inside Bato (competitors and tar
  only in benchmark tooling).
- No proprietary code or algorithms: nothing from closed-source, patented,
  licence-restricted or reverse-engineered compressors (e.g. RAR, KGB
  internals), and no code whose licence is incompatible with
  `MIT OR Apache-2.0` (e.g. copying GPL paq8px/cmix source). Use only ideas
  from public, unencumbered descriptions (papers, books, public-domain
  specifications), implement them originally, and name the source in the
  module's `Provenance:` header. Dependencies must be permissively licensed.
- Universal byte archiver: accept any regular-file contents (empty, invalid
  UTF-8, compressed, encrypted, unknown formats). Never choose a codec, or
  reject input, by name, extension, MIME type or signature. Decisions follow
  byte patterns and bounded costs; transforms keep the raw fallback and
  count their metadata and CPU in selection.
- Streams and directory archives are told apart by the header flag, never a
  suffix. Default output names keep the complete source name. No silent
  filename conversion; portable path restrictions live in `docs/format.md`.
- Incompressible data is stored raw with bounded overhead; never claim every
  input shrinks.
- Bit-identical output on every platform and thread count: integer only
  (enforced by `cm_sources_are_integer_only`), no RNG, clocks, floats,
  hash-map order or scheduling-dependent results. Pseudo-random values come
  only from fixed-seed integer hashes.
- Keep checksums, bounded memory and output, path, symlink and device-file
  safety, and staged extraction.
- Multicore: independent segments run on all admitted workers. No GPU path
  unless it is measured to improve size or decoding with identical bytes.
- Development mode: no backward compatibility. Break the format freely, bump
  the version in `src/core/types.rs`, update `docs/format.md`, and reject
  unsupported data clearly. Delete legacy decoders and compatibility tests.

## Resources
- Default runs use `--threads 0 --memory-budget auto` (all CPUs, 90% of
  usable RAM, 10% headroom). Benchmarks resolve and record these once;
  `--resume` keeps them. Explicit limits are only for labelled controlled
  tests.
- Competitors get identical CPU and process limits, and every run needs
  enough independent work for its workers. Segment geometry changes are a
  different configuration; report them, never substitute them silently.

## Workflow
1. Read the code and `docs/format.md` first. Find the largest measured gap,
   profile it, and make one focused change.
2. Iterate on the frozen development manifest (Canterbury; Calgary is
   held out; see `docs/benchmark-iterations.md`). Pilot files of at most
   2 MiB each and at most 8 MiB in total, whole files only.
3. Pass the gates below, then benchmark the candidate against the previous
   best on identical hardware and settings.
4. Keep a change only for a measured useful gain: compare the exact encoded
   bytes and the extra compression and decompression CPU, and record the
   regressions. A pure refactor must keep archives byte-identical.
5. Commit and push each retained, verified improvement to `main` before the
   next experiment, against the expected current `main` SHA, preserving
   concurrent changes and file modes. Never force-push. Record source and
   binary identities and accepted run IDs in the ignored checkpoint.
   Rejected experiments stay off `main`: no feature flags or options for
   inferior variants.
6. Full evaluation only after the gates pass: frozen and held-out
   manifests, `--mode rigorous --warmups 1 --repetitions 5` (large cases via
   `--large-warmups`/`--large-repetitions`). First compare the manifest's
   paths, file counts and bytes with the intended set; set directory caps
   explicitly; label reduced coverage as partial.
7. If blocked by the environment: checkpoint the source and binary
   identities, commands, results and the next step; say the goal is
   incomplete. Don't loop on setup.

### Gates
```bash
cargo fmt --check && cargo clippy --release --all-targets
cargo test --release
python3 -m unittest discover -s scripts/tests
python3 scripts/run_tests.py --bato-bin target/release/bato --timeout 180
```
Scalar/SIMD parity tests are part of `cargo test`. Correctness fixtures
cover misleading names, all byte values, empty and duplicate files, nested
compressed data and mixed trees, checking exact bytes and structure.
For a refactor, also build the previous `main` as a reference, compress the
development files with both binaries and `cmp` every archive.

## Benchmarking
- Use only `scripts/download_corpus.py`, `scripts/benchmark_suite.py` and
  `scripts/compare_benchmarks.py`; extend them rather than adding parallel
  tools. Don't re-download verified data.
- Freeze dataset identities, hashes, roles and settings before evaluating:
  `--list-workloads > …frozen.json`, then `--frozen-workloads`. Duplicates
  of development data are not independent held-out evidence.
- Competitors (paq8px, paq8l, zpaq, cmix, nncp, KGB, 7z, ZIP, RAR, xz, zstd,
  gzip, LZ4, bzip2 where available): preflight with `--check-tools`, record
  versions, and use the strongest supported profile (`--profile high`;
  paq8px runs `-12L -lstmlayers=5`). Record resource-limited PAQ settings
  explicitly; never substitute a lower level silently. GPU competitors
  (`--nncp-cuda`) are a separate hardware track.
- Run each competitor once per workload, profile, track, hardware and limits
  as a baseline campaign; then reuse it with `compare_benchmarks.py
  --competitor-run` while only Bato changes, reporting its run ID and date.
  Never copy stored observations into new trials or present them as fresh.
- During development run `benchmark_suite.py --codecs bato` only.
- Verify every restored file and directory; delete temporary archives and
  extraction trees but keep their measurements. Unavailable tools, failures,
  timeouts and zero-exit partial archives are failures, not Bato wins
  (`--allow-unavailable` records them as such).
- Record size and ratio, compression and decompression wall and CPU time,
  throughput, process-tree RAM, hardware, failures and uncertainty. VRAM,
  energy and I/O only where instruments are reliable; otherwise "unknown".
- Keep single-worker and all-worker runs (`--bato-modes cpu1,cpu`), native
  and common-tar tracks separate; every setting goes in the run identity.
  The tar track packages exactly the frozen members, revalidates after
  packaging, rejects changed native inventories and keeps hidden files.
  Nothing else may run during timed trials.
- Report wins, ties, losses and uncertain results per workload and metric,
  for one declared Bato configuration; never combine the best metrics from
  different runs.
- Datasets, reports, databases, binaries and checkpoints live in ignored
  directories. Keep live scratch (`--scratch-dir`) outside synced folders;
  mirror evidence with `--checkpoint-dir`.

## Code quality (DRY, KISS, YAGNI)
- One implementation per rule, constant and format detail; the encoder and
  decoder share one model path (`Predictor::probability`/`observe`). Prefer
  plain functions and data to single-use traits and generics. Build nothing
  speculative.
- Delete dead and commented-out code; no `#[allow(dead_code)]` and no unused
  parameters. Test-only code is `#[cfg(test)]`.
- Modules have one responsibility and stay under about 800 lines.
- At most four parameters (otherwise use a struct); no boolean flag
  parameters (use an enum); enums and newtypes instead of bare integers and
  strings.
- Named `const`s with unit and reason: format constants in `types.rs` or the
  owning module, model tuning at the top of its model.
- Nesting at most three levels; return early.
- No mutable globals (only exception: the test heap observer in
  `src/core/diagnostics.rs`). No `dyn Any` control flow. No
  `unwrap`/`expect` on input-dependent library paths.
- Every `unsafe` block has a `// SAFETY:` comment. SIMD and pointer code also
  needs a test against the scalar or safe reference.
- Never discard errors silently (`let _ =`, `.ok()`, `unwrap_or_default()`)
  without a comment on that line saying why.
- Methods that mutate say so in their name or docs. Getters, predicates,
  `Display` and `Drop` do no I/O and no unrelated mutation.
- `src/main.rs` only parses arguments, calls the library and prints.
- Fix violations in code you touch; record larger ones as follow-up work.
