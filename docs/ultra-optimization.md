# Bato single-mode size and speed work

Bato has one compression policy: seek the smallest complete lossless archive. This
work does not introduce fast/balanced presets, loosen validation, or change the
1.18 decoder syntax.

## Implemented on this branch

- Cache immutable context-map features once per process and reuse bit depth;
  predictions must remain bit-exact.
- Read adjacent APM interpolation bins through one checked slice and remove a redundant prefetch request for the same cache line.
- Compare LZ and DEFLATE match candidates word-at-a-time with the same first-mismatch semantics;
  select LZ matches using encoded command cost, not just match length.
- Use optimal package-merge for Huffman histograms whose normal tree exceeds 15 bits.
- Choose DEFLATE reconstruction recipes by their exact serialized byte count, without repeated serialization; also test the supported wider hash configuration at the strongest narrow predictor level.
- Recognize stored DEFLATE blocks using the length complement, and selectively
  recognize fixed-Huffman blocks at the beginning of input or after a valid zlib
  header. The existing exact re-encode check still decides acceptance.
- Compare actual framed costs of DEFLATE, x86-transformed and original context mixing, rather than returning as soon as DEFLATE wins.
- When the independent codec finds a winning delta/shuffle filter, test that filter with CM as an additional candidate (not an exhaustive search of all five expensive CM filters).
- Reject the whole x86 selection allocation when fewer than 33 candidate sites could ever qualify.
- Tighten sequential trial cost bounds to discard only provably losing candidates.
- Allow concurrent predictor trials only when the existing worker reservation covers both models; otherwise run the same candidates sequentially.
- For indexed archives freshly produced by Bato, skip materializing the
  decoded catalogue tree and its second stability fingerprint. Independently
  indexed/untrusted archives retain the complete validation path.
- Add regression tests for length-limited Huffman, LZ byte-exact matching,
  DEFLATE stored-block discovery, and indexed output equivalence.

## Guarantees and current limits

Changes in this branch do **not** add a compression preset or bypass any
decoder CRC, SHA-256, or canonical payload checks. Every selected representation
must decode into the same original bytes. Filtered/raw/deflate CM comparisons
are based on exact framed byte cost, but a universal optimum over all possible
compression algorithms is impossible, and exhaustive evaluation of every CM
filter would multiply model time. The current filter shortlisting uses the best
independent transform as a cheap proxy and retains unfiltered CM as a candidate.

The code has not been compiled or benchmarked in this environment. Existing
repository tests need to be run locally or by CI before the draft PR is ready.

## More invasive work intentionally not mixed into this compatibility PR

1. **Primed-model snapshots.** A fixed primer is trained for every modeled
   segment, but taking an equivalent snapshot requires copying the full
   model state (including context tables and match history), accounting for
   its memory, and benchmarking copy-on-write/sparse snapshot options.
   Naively cloning a model can negate the speed benefit and oversubscribe RAM.
2. **Content-defined shared ranges.** Cross-file references require new
   archive syntax, reference validation, bounded random-access reconstruction,
   security analysis, and a format-version migration. Whole-file duplicates
   are already supported. This is an independent codec project.
3. **Emit index while encoding.** The current recovery-index writer decodes
   the archive again to gather per-segment decoded hashes. An in-flight index
   would need encoder-to-index callbacks and equivalent verified segment
   boundaries; the branch only removes the extra temporary extraction.
4. **Source-file fingerprint consolidation.** Initial SHA-256 fingerprints,
   exact duplicate-byte comparisons, and write-time rehashing protect against
   changing files and accidental false deduplication. A single read is only
   safe with immutable snapshots or a different consistency contract.
5. **APM/table geometry and model redesign.** Changing probability behavior
   changes the format model. Any new proposal must be evaluated on held-out
   corpora and have a deliberate version/compatibility policy.
6. **Alternative segment geometry and nested scheduling.** The best segment
   size depends on workload and available memory; this branch gates
   parallel candidate trials without increasing the admission floor. Geometry changes require measured
   speed-versus-ratio comparisons.

## Validation gate

Do not mark this work ready or claim a performance gain until the following
checks pass on the exact branch head:

- `cargo fmt --all -- --check`
- `cargo clippy --release --all-targets --locked -- -D warnings`
- `cargo test --locked` and `cargo test --release --locked`
- `cargo build --release --locked`
- `python3 -m unittest discover -s scripts/tests -p 'test_*.py'`
- `python3 scripts/run_tests.py --bato-bin target/release/bato`
- Repeatable baseline comparisons for compression/decompression throughput,
  archive bytes, peak resident memory, and correctness on independent corpora.

The only acceptable outcome is **lossless data reconstruction with no weakened
corruption detection**, and favorable measured size/speed tradeoffs. Each
candidate should be kept or reverted on the evidence, not assumed beneficial.

## Future format and feature milestones

- Independently decodable LZ/entropy blocks with separate literal and
  length/offset streams. Investigate full cost-based parsing and long-distance
  matching with strict, bounded decoding.
- Cross-file and cross-segment references selected by content-defined anchors,
  with verified earlier-range-only links and controlled memory/disk limits.
- Adaptive region framing and reversible structural transforms for integer,
  image, audio and instruction streams, always tested against original CM.
- Immutable source snapshots for eliminating archive source rereads without
  sacrificing change detection or exact duplicate verification.
- Backward-compatible decoders for every supported format minor revision.
- Indexed archive mounting, incremental snapshots/retention and richer
  filesystem metadata. All need format security and compatibility review.

None of these is claimed as implemented in this compatibility-focused PR.
