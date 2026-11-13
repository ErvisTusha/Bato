# Bato

Bato is an original, lossless, byte-oriented compressor and archiver written in Rust. It has one compression mode, always its strongest: byte-cost selection among context mixing, independent literal/copy commands, encoder-built canonical Huffman tables and run storage. Files, stdin streams and directory trees all use it.

Bato accepts arbitrary file contents, including unknown formats, invalid UTF-8, already compressed or encrypted data and misleading names. Compression decisions follow byte patterns only, never names, extensions or MIME types. Incompressible segments are stored raw with bounded framing overhead, so not every input becomes smaller.

## Build

```bash
cargo build --release --locked
```

## Use

```bash
bato input.dat                      # writes input.dat.bato
bato -d input.dat.bato              # restores input.dat
bato -t input.dat.bato              # verifies without writing
bato -i input.dat.bato              # shows header and blocks
bato archive project                # writes project.bato
bato extract project.bato -o out    # restores the tree
```

Single files and directory trees both use the `.bato` extension; the header records which one a file holds. Decoding reads the geometry from the header, never from command-line settings. Original bytes and supported names are restored exactly. Checksums, output limits and staged extraction are always active.

## Extract selected files or folders

```bash
bato extract project.bato -o out docs/readme.txt assets
bato extract project.bato -o out -- -notes.txt
```

Use exact, archive-relative paths with `/` separators and omit the trailing
slash on folder names. A folder selection includes its descendants, including
empty folders. Bato preserves the directory structure and fails before writing
to the destination if a requested path is missing. With `--public-only`, paths
are matched against the public section. To resume a checkpointed extraction,
pass the same member arguments.

Bato still decodes and verifies the full requested archive or public section
and needs temporary disk space to stage it. Only the selected entries and their
parent directories are published.

## Edit, split, and protect archives

```bash
bato add project.bato notes.txt assets/        # add at archive root
bato remove project.bato notes.txt assets/old  # remove a file or a subtree
bato archive project --split-size 104857600 -o project-split.bato
bato extract project-split.bato -o restored
```

Edits rebuild the solid archive and publish the replacement after a successful
write. Adding a directory merges it by basename. Pass `-f` to replace existing
files during an add. Remove paths are relative to the archive root. For an
argument that starts with a dash, place it after `--`. A split archive is a
directory containing an index and numbered parts. Keep the entire directory
together when moving it. The example caps each part at 100 MiB. Edits keep the
existing split size. Create split or password-protected output outside the
source tree. Split operations need temporary disk space for the assembled
stream, and edits need space for a staged extraction.

```bash
# Protect the complete archive, including filenames
bato archive project --password-file key.txt -o full.bato
bato extract full.bato --password-file key.txt -o restored-full

# Protect only selected paths. Repeat --protect for multiple paths.
bato archive project --password-file key.txt --protect private/records \
  --protect credentials.json -o selective.bato
bato extract selective.bato --public-only -o public-files
bato extract selective.bato --password-file key.txt -o all-files

# Add a protected folder to an existing selectively protected archive
bato add selective.bato confidential --password-file key.txt --protect confidential
```

Keep the UTF-8 password file outside the archived tree with access restricted
to its owner. Bato strips one trailing newline. Each archive uses one password.
Selective archives expose public names and contents and may reveal shared
ancestor names and section sizes. Existing protection survives edits. New
members added to a selectively protected archive are public unless they fall
under a protected path or you select them with `--protect`. Encrypted archives
require the password for editing and full verification. To verify only the
public section, use `bato -t selective.bato --public-only`. Both protection modes
can be combined with split volumes. These features apply to directory archives.

## Pause, resume, and recover

A checkpoint preserves completed segments across a process exit. Give it a
new, separate directory, then repeat the command with `--resume` and the same
input, output, and format settings.

```bash
bato archive project -o project.bato --checkpoint encode-job --pause-file hold
# From another terminal, request a pause
touch hold
# After Bato reports that it paused, remove the marker and resume
rm hold
bato archive project -o project.bato --resume encode-job

bato extract project.bato -o restored --checkpoint decode-job --pause-after-segments 3
bato extract project.bato -o restored --resume decode-job
```

The same flags work with `bato FILE` and `bato -d ARCHIVE`. Bato checks
the input and checkpoint data before continuing. Work beyond the last
committed boundary can be repeated after a pause or interruption. Directory
extraction checkpoints the decoded catalogue and materializes files after
the complete stream verifies. If an interruption during final publication
leaves existing files, inspect them before rerunning with `--resume` and
`-f` to permit replacement. Successful jobs clean up their checkpoints.

Split and password-protected jobs use the same flags. Pass the same password
file on resume. Private checkpoint records are encrypted and authenticated
with a job key protected by the archive passphrase. Preparing inputs,
joining volumes, authenticating or writing the outer encryption envelope,
and publishing output can restart as needed. Checkpointing uses extra
temporary disk space for seekable staging. Plain compression produces the
same archive bytes with or without checkpoints.

```bash
# Reconstruct one missing or damaged volume
bato archive project --split-size 104857600 --parity -o split.bato
bato repair-split split.bato -o repaired.bato

# Prepare the salvage index while the plain archive is still intact
bato recovery-index project.bato -o project.recovery
bato recover project.bato --index project.recovery -o salvaged
```

A parity-enabled split archive adds one parity volume, at most the size of
one data volume, plus SHA-256 hashes in the index. Extraction can reconstruct
one missing or damaged data volume in temporary storage. `repair-split`
writes a separate verified copy and can regenerate damaged parity when the
data volumes are intact.

Keep the salvage index separately from the plain archive. Recovery of a
folder restores complete verified members and reports missing files. For a
single-file stream, recovery writes verified segments named with their
original byte offsets. Recovery requires a new destination. For a damaged
password-protected archive, split parity can restore the original ciphertext
before authentication.

## Archive inspection, indexed extraction, and job status

```bash
bato list project.bato --json
bato diff project.bato project-updated.bato --json
bato diff project.bato ./project
bato health split.bato --json
bato archive project --indexed --password-file password.txt -o indexed.bato
bato extract indexed.bato docs/report.pdf --password-file password.txt -o restored
bato recovery-index project.bato -o project.recovery
bato extract project.bato docs/report.pdf --index project.recovery -o restored --stats
bato archive project -o backup.bato --checkpoint encode-job --progress
bato job-status encode-job
bato archive project --split-size 104857600 --parity-volumes 2 -o split.bato
```

Listing verifies and streams the catalogue without writing an extracted
tree. It reports paths, entry types and original sizes. Use a password for
protected contents or `--public-only` to inspect the public section.

`diff` reports added, removed and changed paths against a second archive or a
live directory. It uses SHA-256 for file contents and detects file/directory
changes. A supplied password applies to both archives when needed.

`health` verifies a regular directory archive. For a split archive it reports
missing or damaged volumes, whether the assembled archive SHA-256 verified,
and whether repair is possible. A degraded archive has lost parity while its
data is still readable. Run `bato -t` to verify the contents of a split archive.

`archive --indexed` embeds the recovery index. With a password it also enables
authenticated random access. Selective extraction reads the encrypted index
and the chunks covering selected members, including any earlier duplicate
data. Use `bato -t` with the password to verify every chunk. The index adds
metadata and creation work. Indexed creation supports plain and whole-password
archives. Splitting, partial protection, and checkpointed creation use the
regular archive format. Indexed member extraction runs without checkpoints.

A separate recovery index still works with an ordinary plain archive.
Plain indexed extraction reads the compressed stream to verify its SHA-256
before decoding the needed segments. Keep separate indexes in trusted storage.

Checkpoint progress reports committed input and output bytes. Job status
shows the saved offsets for public state and indicates when private state
is encrypted.

Multi-volume parity supports 2 to 16 parity volumes and up to 256 data
and parity volumes combined. Each parity volume is as large as the longest
data volume. Bato verifies per-volume hashes and the complete archive hash
before extraction. The existing `--parity` option creates one parity volume.

## Resources

By default Bato uses all CPU capacity available to the process (`--threads 0`) and a managed allowance of 90% of usable RAM (`--memory-budget auto`). Operating-system, container and memory-admission limits remain active. Actual use depends on the work available.

Segments are 64 MiB and chunks 256 KiB. For context-mixed segments the model keeps learning across chunks, and separate segments run in parallel. Independent copy, run, static-table and stored segments decode without adaptive training. Smaller segments (`--segment-size`) expose more parallel work but restart learning more often, which usually makes output larger.

Bato has no default extraction byte cap or CLI runtime deadline.
Other controls: `--threads N`, `--memory-budget auto|BYTES`, `--segment-size BYTES`, `--chunk-size BYTES`. `bato --help` lists them all.

## Format

Bato 0.8 writes format 1.20 (the benchmark results below were measured with 1.18). A stream has a 10-byte header, packed block tags, canonical variable-length sizes, CRC32C checksums and a compact footer. For every 64 MiB segment the encoder compares the complete byte cost of context mixing (with a built-in primer) against stored, repeated-byte, literal/copy, static Huffman and digit-run representations, and keeps the smallest. A context-mixed segment may first pass through one reversible transform chosen from byte statistics alone, never from names or signatures: byte delta, byte-plane shuffle, x86 and x86-64 address rewriting (call/jump operands and RIP-relative displacements, with a reach that grows for executables larger than 16 MiB), or deflate recompression, which replaces valid deflate streams found anywhere in the segment (inside gzip, zip or PNG data, for example) by their inflated bytes and a small recipe. The x86 rewrite also applies on top of deflate recompression. Every transform keeps the raw fallback and counts its metadata in the selection. The optional password envelopes and volume index are versioned separately. See [docs/format.md](docs/format.md).

Digit runs can also be separated from surrounding bytes. The encoder tests
absolute values and two contextual integer predictors, packs small static ANS
tables and copy commands, and selects this representation only when its full
byte cost is smaller. Decoding reconstructs the original digit strings,
including leading zeroes, without neural training. This uses byte patterns,
not a JSON parser or filename detection. Other data retains the existing
candidates, including context mixing.

## Validation and measurement

```bash
cargo test
cargo test --release
python -m unittest discover -s scripts/tests -v
python scripts/run_tests.py --bato-bin target/release/bato --timeout 180
```

Use `scripts/download_corpus.py`, `scripts/benchmark_suite.py` and `scripts/compare_benchmarks.py` for reproducible, verified measurements. Competitors run with the same CPU and process limits.

```bash
python scripts/benchmark_suite.py --corpus /path/to/corpus \
  --frozen-workloads /path/to/frozen.json --codecs bato,zip,7z \
  --bato-bin target/release/bato \
  --mode rigorous --warmups 1 --repetitions 5
```

Freeze inputs and hashes before evaluating a change, verify every restored result, and report size, compression and decompression wall and CPU time, memory, failures and uncertainty together.

The Python CLI suite exercises the selected Bato binary. Nested RAR and
7z archives are frozen byte fixtures. Run historical Bato comparisons
explicitly with the archive-operation benchmark in the
[benchmark guide](docs/benchmark-iterations.md#bato-archive-operations).

For file-size boundary testing around 2 GiB and 4 GiB, run the Python CLI
suite with `BATO_MULTIGIB_STRESS=1` and at least 5 GiB of free temporary
space. The real disk-exhaustion test requires Linux with permission to
create a private mount namespace.

## Benchmark results

These are measured results for Bato 0.8, format 1.18, on a 4-core, 15 GiB Linux machine. Every Bato output was decoded and compared with the original. Sizes are exact; times are single runs with about 5% noise. The complete log, including earlier formats, losses and rejected experiments, is [docs/validation.md](docs/validation.md).

### Real release downloads, packed as folders

Three public release downloads were unpacked, symlinks removed (Bato stores none; Node had 3), and the same tree was packed by every tool at its strongest setting: `gzip -9` and `xz -9e` on a deterministic `tar`, `zstd --ultra -22` on the tar, `7z -mx9` and `rar -m5` natively. Bato used `bato archive` with the defaults. The Antigravity and Go downloads were gzip files and are repacked with `gzip -9`; Node's download was xz, and a gzip size was not measured. Format 1.18 was developed while measuring these trees, so its gains on them are tuned, not independent.

| Tree | Unpacked bytes (files) | gzip -9 | xz -9e | 7z -mx9 | zstd -22 | rar -m5 | **Bato 1.18** | Bato vs best other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Antigravity IDE 2.6.0 (`.tar.gz`) | 595,461,667 (15,898) | 193,826,513 | 124,350,148 | 115,498,890 | 129,944,442 | 139,420,003 | **106,026,019** | -8.20% |
| Node.js v22.23.4 (`.tar.xz`) | 196,453,447 (4,750) | not measured | 31,040,920 | 29,426,604 | 33,513,271 | 35,429,629 | **25,212,353** | -14.32% |
| Go 1.27.3 (`.tar.gz`) | 244,139,206 (15,657) | 69,939,410 | 39,725,940 | 38,584,607 | 42,447,576 | 46,390,543 | **33,782,384** | -12.45% |

Bato is smallest on all three trees, and every restored tree was identical to the original (per-file SHA-256 and directory list).

| Tree | Bato pack / unpack (s) | Bato peak RSS pack / unpack (MiB) | Other tools, compress (s) |
| --- | ---: | ---: | --- |
| Antigravity | 1681.29 / 1606.30 | 2,399 / 1,606 | gzip 42.94; others not timed |
| Node.js | 515.37 / 491.30 | 1,433 / 1,018 | xz 85.7, zstd 76.9, 7z 39.7, rar 13.2 |
| Go | 589.64 / 564.65 | 2,272 / 1,485 | gzip 23.57, xz 152.6, zstd 159.1, 7z 68.3, rar 22.8 |

**Speed is the main loss.** Bato packs these trees about 4 to 39 times slower than the other tools (3.7 times slower than `zstd -22` on Go, 39 times slower than `rar -m5` on Node), and unpacking takes about as long as packing, because context mixing must update its model for every decoded byte. The other tools' decompression times on these trees were not measured. On one 199,855,320-byte executable from the Antigravity tree, 7z unpacked in 2.7 s and Bato in 510.73 s.

### One large executable

The `antigravity-ide` program file (199,855,320 bytes, as one stream):

| Tool | Size (bytes) | Pack / unpack (s) |
| --- | ---: | ---: |
| **Bato 1.18** | **47,536,248** | 526.42 / 510.73 |
| 7z -mx9 | 49,191,016 | 85.6 / 2.7 |
| xz -9e | 57,065,392 | not timed |

Before format 1.18 Bato was 0.73% larger than 7z on this file.

### Other results

* The 103 development files total 1,398,987 bytes with format 1.18, the same as with 1.15.
* Format 1.9 against the strongest competitors on 11 Canterbury files (2,810,784 bytes): Bato 341,001, xz 493,080, 7z 494,801, rar 512,223, gzip 730,732. `paq8px -9L` was smaller at 300,270 (12% below Bato) and won on 10 of the 11 files.
* Earlier experiments on Silesia and enwik8 found `zpaq -m5` about 2 to 4% smaller than Bato. Those corpora have not been rerun with format 1.18.
* Sizes for sealed, fresh and media sets are listed per format version in [docs/validation.md](docs/validation.md).

## Status

Context-mixed blocks require causal model updates during decoding, so decode time grows with the data and is far slower than LZ-family tools. Compressed size is the priority, then decompression time, then compression time. Open gaps: speed against every other tested tool, and size against `paq8px` and `zpaq` on corpora where they were measured smaller.
