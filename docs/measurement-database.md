# Measurement database

The canonical database defaults to `~/data-research/measurements.sqlite3`. Reports still use `--output-dir`. Both benchmark and comparison commands accept `--database PATH`; pass the same path when resuming or comparing an explicit database.

Initialize the schema without selecting inputs or running compressors:

```bash
python3 scripts/benchmark_suite.py --init-db
```

Initialization at the default location relocates the previous default database using SQLite backup, verifies integrity and all records, and removes the old copy only after verification. An active SQLite writer prevents relocation. Different databases at both locations cause a conflict and are preserved; use `--database` to select one explicitly. Checkpoint databases must have a different path from the canonical database.

The `runs`, `workloads`, `cases`, and `trials` tables remain authoritative. The `competitor_measurements` table is a typed projection, backfilled from historical cases and synchronized transactionally with case writes. It has one row per run, case and competitor, with filename, original and compressed bytes, wall and CPU seconds, peak RSS and sampled process-tree RSS in KiB for both phases, case and run status, verification, settings and provenance. The run status distinguishes an interrupted campaign from an active case. `source=preflight` records unavailable tools without fabricating cases or measurements. Sampled tree RSS is a lower bound. Reused baselines remain attached to their original runs.

SQL views update automatically as benchmark scripts commit measurements:

- `competitor_results`: one row per run, workload, track, input identity and profile, with `zip_bytes`, `gzip_bytes`, `seven_zip_bytes`, `rar_bytes`, `paq8px_bytes`, `cmix_bytes`, and `nncp_bytes`. A size is populated only for a single verified successful case. Failures, settings and individual codec status remain in `codec_results`; unmeasured or unsuccessful sizes are NULL. Runs are never merged to choose a best result.
- `bato_results`: one row per Bato case, including version label, checkout revision, binary hash, status, compression/decompression sizes, wall and CPU seconds, MiB/s, peak RSS in KiB, sampled process-tree peak RSS, workers, resource policy, and hardware. `binary_build_revision` retains the harness's explicit provenance statement: a checkout revision alone does not prove which source produced a binary.
- `measurement_results`: the common case projection, including `case_record` and `run_record` JSON for full details. Summaries come from canonical cases aggregated by the existing harness over measured trials; raw warmup and measured observations remain separate in `trials`. Unsupported resource measurements are NULL. Sampled tree RAM is a lower bound, not an exact peak; scope columns preserve this distinction.

PAQ8PX, CMIX and NNCP are selected size-focused competitors, not an assumed ranking. Codec settings and binaries are recorded with each run. No measurements are fabricated when initializing the database.

```sql
-- Open with: sqlite3 /home/et/data-research/measurements.sqlite3
SELECT run_id, filename, original_bytes, zip_bytes, gzip_bytes,
       seven_zip_bytes, rar_bytes, paq8px_bytes, cmix_bytes, nncp_bytes
FROM competitor_results
ORDER BY run_id, filename;

SELECT run_id, version_label, checkout_revision, binary_sha256, target,
       raw_size, comp_size, ratio, comp_time_s, decomp_time_s,
       comp_cpu_s, decomp_cpu_s, comp_peak_rss_kib, decomp_peak_rss_kib,
       comp_sampled_tree_peak_rss_kib, workers, status
FROM bato_results
ORDER BY run_id, target;
```

`ratio` is original bytes divided by compressed bytes. Use status and verification fields when analyzing Bato results. Different worker configurations, tracks, machines and settings are separate observations and must not be treated as directly comparable without checking their identities.

## Copied-corpus tests

`run_tests.py` defaults to the copied corpus at `data-research/corpus` and the separate project database `data-research/measurements.sqlite3`. The benchmark and comparison tools retain their own documented default database; pass `--database` explicitly to use the project copy. The copied corpus must contain exactly ten regular files for each of the twenty selected extensions.

```bash
python3 scripts/run_tests.py --sample-mode fast --seed 42 --bato-bin target/release/bato --timeout 180
python3 scripts/run_tests.py --sample-mode normal --seed 42
python3 scripts/run_tests.py --sample-mode full --seed 42
```

`fast` chooses one seeded random file from each half of the size-sorted files per extension; `normal` chooses five; `full` chooses all ten. `--min` is an alias for `fast`. A frozen manifest records the selected paths, hashes, mode and seed. The campaign is a partial assessment of the copied corpus, not a substitute for the frozen development and held-out evaluations. Real-file tests verify inputs and round trips. The harness preflights ZIP, gzip, 7z, RAR, PAQ8PX, CMIX and NNCP and measures available tools at their strongest supported profile, with one warmup and five measured repetitions including large cases. Missing or failed tools are recorded as unavailable; failures do not prevent the correctness suite, but a partial or failed campaign exits nonzero.

```sql
SELECT run_id, filename, competitor, original_bytes, compressed_bytes,
       compress_seconds, decompress_seconds, compress_cpu_seconds, decompress_cpu_seconds,
       compress_tree_rss_kib, decompress_tree_rss_kib, status, run_status, verified_lossless
FROM competitor_measurements
ORDER BY run_id, filename, competitor;
```

Use status, verification, source, hardware and settings when comparing measurements. Unknown metrics remain NULL. Exact compatible verified baselines retain their original run and case identity; no trials are copied into a new campaign.
