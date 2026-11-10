# Reproducible Bato improvements

Keep downloaded corpora, frozen manifests, benchmark databases, binaries,
profiles, generated reports, and checkpoints in ignored directories such as
`multi_corpus/` and `artifacts/`. The authoritative tools are
`scripts/download_corpus.py`, `scripts/benchmark_suite.py`, and
`scripts/compare_benchmarks.py`.

The SQLite store uses full-synchronous rollback journaling under the existing
host measurement lock. Before marking a run complete it checks database
integrity, the planned case count, and every accepted fresh case's committed
warmup and measured trial numbers. Preserve a damaged database and journals;
start fresh measurements in a new ignored output directory rather than
silently trusting an inconsistent projection.
The comparison reader independently checks database integrity, accepted trial
numbers, and the case projection against the committed trials. Schema-v4 JSON
reports alone cannot establish a comparison: preserve the canonical database.
If the environment replaces or damages a live database, retain the failure
and use an unsynchronized temporary measurement directory, then back up the
completed database and reports into an ignored directory. Do not repair lost
observations by copying values from a report.

## Freeze before evaluating

Download and validate real data with the downloader. For example:

```bash
python3 scripts/download_corpus.py -o multi_corpus --target compression
mkdir -p artifacts/iteration
python3 scripts/benchmark_suite.py -c multi_corpus --mode rigorous --target canterbury \
  --track both --workload both --list-workloads > artifacts/iteration/development.frozen.json
python3 scripts/benchmark_suite.py -c multi_corpus --mode rigorous --target calgary \
  --track both --workload both --list-workloads > artifacts/iteration/heldout.frozen.json
```

Declare the dataset split, manifest hashes, hardware, binary source revision,
profile, CPU allocation, thread limits, memory limits, group size, seed,
warmup/trial counts, timeout, and uncertainty rule in an ignored checkpoint.
Tune only on development data. Evaluate held-out data after selecting each
candidate; do not use its results to choose parser parameters. Synthetic
fixtures are useful for corruption, bounds, and parity testing, but do not
replace real-data measurements.

For every registered asset, including the large tier, use the actual downloader
with an explicit prepared-data budget, then audit the indexed files:

```bash
python3 scripts/download_corpus.py -o multi_corpus --all --tier extreme \
  --jobs 8 --per-host 2 --extraction-jobs 2 --max-bytes 10000000000
python3 scripts/download_corpus.py -o multi_corpus --audit-existing
python3 scripts/benchmark_suite.py -c multi_corpus --mode rigorous --suite corpus \
  --workload directories --track both --include-duplicates \
  --max-file-bytes 8589934592 --max-directory-bytes 8589934592 \
  --list-workloads > artifacts/iteration/full-collections.frozen.json
```

Whole collection directory workloads cover every indexed byte without
sampling. Verify that their file paths and hashes exactly equal the index;
record this scope distinctly from benchmarking every individual file. Keep
duplicate files in the complete directory trees. Freeze the collection
manifest and the development/held-out split before measurements.

## Measure separately for each configuration

Use `--check-tools` to preflight the actual installed tools and record
versions. Missing RAR, KGB, or any other adapter must remain explicitly missing.
The `high` profile applies 7z `-mx=9`, ZIP `-9`, RAR `-m5`, KGB `-9`,
xz `-9 --extreme --no-adjust`, zstd `--ultra -22`, gzip `-9`, LZ4 `-12`,
and bzip2 `-9`. These are tool settings, not evidence that a competitor's
size is optimal for every input. Fast and default settings remain available
as separate profiles; never compare unlike profiles silently.

Example for one CPU and all available competitors at their highest settings:

```bash
python3 scripts/benchmark_suite.py -c multi_corpus --target canterbury \
  --mode rigorous --track both --workload both \
  --frozen-workloads artifacts/iteration/development.frozen.json \
  --bato-bin artifacts/iteration/baseline --profile high \
  --cpu-limit 1 --threads 1 --bato-modes cpu1 \
  --bato-memory-budget 1073741824 \
  --memory-budget 2147483648 --warmups 1 --repetitions 5 \
  --timeout 120 --allow-unavailable --label baseline-cpu1
```

Save the printed run ID and binary hash. Repeat the exact settings with a
candidate binary. Change `--cpu-limit`, `--threads`, and `--bato-modes`
together for an equal-resource parallel track, e.g. eight allocated CPUs and
`--bato-modes cpu`. Serial competitors remain serial within the same
allocation; do not pretend their CLI supports thread settings it lacks.
Bato has no GPU path. A GPU competitor (`--nncp-cuda`) is a separate
hardware track and is never compared with CPU runs as if equal.

Pass both `--large-warmups 1 --large-repetitions 5` as well as
`--warmups 1 --repetitions 5` for the full suite. Large workloads must not
silently use a single trial. The conservative scratch admission reserves five
times the payload and per-file headers plus 64 MiB; free reproducible build and
download temporaries after pinning binaries if necessary. Do not lower the
admission factor to make a large workload pass. Resource failures and timeouts
remain explicit results, never competitor wins. A Bato-only full-data run
can validate engine changes, but does not replace the full competitor matrix.

Native directory archives and a normalized common-tar input are separate
tracks. External compression and tar packaging occur only in Python benchmark
tooling. Every decompressed file, path, and directory, including empty
directories, is verified before a trial is accepted. Common-tar trials verify
the exact restored tar bytes, then safely extract that tar and compare every
file and directory with the frozen source snapshot. These verification checks
occur outside codec compression/decompression timing. Compression and
decompression wall time, throughput, and isolated workflow CPU time have
separate recorded scopes. Container construction is included in native
workflow timing; common-tar construction is prepared outside encoder timing.

## Compare and retain useful changes

```bash
python3 scripts/compare_benchmarks.py --baseline BASELINE_RUN_ID \
  --candidate CANDIDATE_RUN_ID --competitors bato-v1
```

The output contains per-case comparisons and exclusion reasons. Exact stable
bytes determine size wins. Timed wins or losses require a warmup, at least
five fresh trials, a difference greater than 3%, and disjoint observed trial
ranges. A 1 ms wall-time floor avoids claiming sub-millisecond differences
as wins; CPU time has a 0.1 ms floor. Overlapping ranges are uncertain, not
statistical confidence intervals. Small changes inside the tolerance are
reported as ties. Run-to-run drift and the sequential evaluation order remain
limitations even with this conservative rule.

Use `--competitor-run PEER_RUN_ID --competitors bato-v1` (or one explicit
parallel configuration) to reference existing competitor trials. The command
requires identical hardware/harness identity, CPU allocation, requested threads,
memory policy, profile, seed, timeout, and evaluation mode. It reads each peer's
original committed trials and records both source run IDs. It does not copy
observations into a new run. Failed, cached, short, missing, or incompatible
trials cannot establish wins; at least one warmup and five verified fresh trials
are still required. Only matching input bytes and tracks are compared.

Back up the canonical database and frozen inputs durably before large workloads.
An environment reset can erase both the workspace and temporary database. If
the canonical database is lost, retain session observations as unconfirmed
diagnostics and start a fresh generation; never reconstruct trial records from
logs, summaries, or comparison projections. Re-downloaded Git metadata can
change hashes even at a pinned revision, so re-audit and freeze the new index.

GNU time child peak RSS and the isolated Python worker's peak RSS are
individual process high-water marks. They are not a concurrent process-tree
peak. Process-tree RSS samples are only a lower bound, and sampling is disabled
when namespace PIDs do not match `/proc`. GPU VRAM (GPU competitors only) and
energy stay unknown without reliable instruments. Resource unknowns cannot
establish an efficiency win.

Linux workflow I/O counters are read from `/proc/self/io` before and after the
isolated phase, including waited-for children. Logical read/write bytes and
syscall counts are distinct from storage-layer read bytes, page-dirtying write
bytes, and cancelled write bytes. File-cache state is uncontrolled; these
counters do not establish physical SSD/HDD traffic, latency, or energy. Unknown
physical-device measurements remain unknown. See the kernel's
[I/O accounting documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html).

Bato `--stats` snapshots report admitted workers, coded blocks, estimated
host memory, the memory budget and the managed peak reservation. They are
engine accounting, not process RSS.

After each retained change, checkpoint its source revision, binary hashes,
commands, run IDs, manifest hashes, correctness/build results, comparison
files, regressions, unmeasured metrics, and next experiment. Favorable averages
do not establish the overall goal: require each declared configuration to win
each workload and metric, and report every missing competitor and loss.

## Diagnostics and next experiment

`examples/cm_probe.rs` reports exact encoded sizes for complete frozen
development files. It supplements the authoritative suite and never
establishes a competitor win. The known gaps are summarized in the README
status; profile the largest measured size gap first, declare each candidate before new trials, and measure fresh
development trials before the full frozen and held-out evaluation.

Selected codecs always receive fresh observations; competitor reuse reads original canonical runs via `--competitor-run`. Historical databases are preserved. A tooling change changes the recorded harness hash: old/new size comparisons remain available when inputs and settings match, but timing and competitor-reference comparisons require identical harness and resource conditions. Never rewrite old identities to bypass that check.

## Correctness tests and competitor measurements

Run the CLI correctness gate with the selected Bato binary.

```bash
python3 scripts/run_tests.py --bato-bin target/release/bato --timeout 180
```

Run competitor measurements as a separate command using a frozen corpus
manifest and the strongest supported profile.

```bash
python3 scripts/benchmark_suite.py --corpus /path/to/corpus \
  --frozen-workloads /path/to/frozen.json --codecs bato,zip,7z \
  --bato-bin target/release/bato --profile high --mode rigorous \
  --warmups 1 --repetitions 5
```

## Bato archive operations

Run this track separately from compressor comparisons. It uses a deterministic
synthetic tree, a same-size edit and a separate add payload. The manifest
records the tree and content identities, operation set and controlled 64 KiB
segment / 16 KiB chunk geometry. Preserve old manifests alongside the
database. Generate a new versioned manifest when changing fixture semantics.

```bash
mkdir -p artifacts/archive-ops
python3 scripts/benchmark_suite.py --archive-operations --list-workloads \
  > artifacts/archive-ops/indexed-v1.json
python3 scripts/benchmark_suite.py --archive-operations \
  --frozen-workloads artifacts/archive-ops/indexed-v1.json \
  --bato-bin target/release/bato --bato-revision HEAD \
  --baseline-bato-bin /path/to/older/bato --baseline-bato-revision OLDER_SHA \
  --bato-modes cpu1,cpu --warmups 1 --repetitions 5 \
  --output-dir artifacts/archive-ops --scratch-dir /var/tmp \
  --checkpoint-dir artifacts/archive-ops-checkpoint
python3 scripts/compare_benchmarks.py --archive-operations \
  --output-dir artifacts/archive-ops --baseline BASELINE_RUN --candidate CANDIDATE_RUN
```

Binary hashes identify measured builds. Revision arguments declare their source
commits. Keep the exact binaries or record how they were built. Use
`--archive-operation-format plain` on both manifest generation and campaign
commands to compare older builds that lack indexed archives. Unsupported
operations are recorded as such. Compare indexed and plain runs separately.

The archive-operation tables share `measurements.sqlite3` with the existing
benchmark schema. Trials are committed individually. A campaign left running
in the database is incomplete. Start a new campaign after interruption.
Preparation uses fresh copies of each build's verified starting archive.
The command runs with an isolated temporary directory. The OS cache is
uncontrolled. Scratch usage is unknown because short-lived files cannot be
reliably measured after each command. The JSON comparison includes raw outcome
labels and per-metric median, minimum, and maximum. Overlapping timing ranges
are uncertain. For a later run on the same host, reuse the recorded worker
and managed-memory settings with `--threads` and `--bato-memory-budget`
if automatic resource detection resolves different values.

The default CLI correctness run (also available as `--min`) includes the standard
rules and ten verified corpus files: the two smallest distinct TXT, PDF, JPG,
PNG, and MP4 inputs. It reads `~/data-research/corpus/.corpus.json`; override
with `--corpus PATH`. Selection sorts by byte size then relative path, checks
sizes and SHA-256 before testing, and rejects missing types or changed inputs.
Each real-file check compresses, tests, restores, and compares exact bytes,
printing original/compressed sizes and elapsed seconds. Controlled fixtures
remain for corruption, malformed data, boundary sizes and recovery. Temporary
outputs are removed and corpus files remain read-only. Before any Bato correctness test, the command registers the selected paths,
hashes and original byte sizes in SQLite, then ensures verified competitor
baselines for ZIP, gzip, 7z, RAR, PAQ8PX, CMIX and NNCP using the benchmark
harness at its high profile, one warmup and five measured repetitions. Only
missing compatible cases are measured; reused observations retain their original
run IDs and dates. File bytes, codec identities, settings, hardware and limits
must match, and committed lossless trials are required. Missing tools, failures
or timeouts stop execution before the Bato tests. This gate never installs tools
or silently lowers settings. Original baselines remain in their source runs;
reference-only campaigns do not copy trials or appear as fresh measurements.
`--database PATH` selects the database (default
`~/data-research/measurements.sqlite3`), `--benchmark-output PATH` selects reports
and scratch (default `~/data-research/test-results`), and
`--competitor-timeout SECONDS` sets the measurement phase timeout (default 600).
The Bato round trips remain correctness checks; their diagnostic elapsed times
do not create new Bato benchmark observations.
`--test NAME` runs only the selected existing rules and needs no corpus.
Explicit `--min` cannot be combined with `--test` or `--extended`; an extended
run also includes the ten real files.

Use `scripts/run_tests.py --extended --timeout 900` for the 100,000-file admission
check, 30,000-file edits, and multi-GiB restore cases. Run these separately from timed campaigns.
