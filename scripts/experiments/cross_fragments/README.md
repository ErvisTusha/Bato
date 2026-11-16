# Cross-file fragment experiment

This standalone Rust helper factors exact ranges shared by files before Bato
compresses the remaining bytes of a native directory archive. Recipes refer to
earlier files by their sorted position, independent of Bato segment boundaries.
The benchmark counts the wrapper and all metadata in archive size. Its wall
and CPU measurements include fragment scanning, staging, and reconstruction.
Each run checks the restored inventory and compares every restored file byte.

Use trusted, immutable benchmark trees. The helper accepts up to 8 MiB of
source data and 4,096 files. Its separate experimental wrapper supports
whole-tree restoration. Bato's normal CLI, format, and indexed extraction
are unchanged.

```sh
rustc --edition=2021 -C opt-level=3 fragments.rs -o /tmp/bato-fragment-pilot
python3 benchmark.py --main /path/to/1.22/bato --branch /path/to/branch/bato \
  --helper /tmp/bato-fragment-pilot --input /path/to/native-directory \
  --output /tmp/bato-fragment-pilot.tsv
```

The default comparison includes stock main, stock branch, and the prototype
with 256 KiB compacted segments. The explicit `wide` and `narrow` variants
use 128 and 64 KiB compacted segments. Inputs without eligible references
go through the branch's normal archive path.

The initial five-directory CPU comparison used main `c5db485` and branch
`fcc0683`, with three alternating runs per input on nine logical CPUs.
All 45 round trips passed. Times are sums of per-input medians in seconds.

| Build | Archive bytes | Pack wall | Restore wall | Pack CPU | Restore CPU |
|---|---:|---:|---:|---:|---:|
| Main 1.22 | 150,565 | 18.125 | 13.040 | 39.042 | 12.901 |
| Branch 1.24 | 199,430 | 20.913 | 7.714 | 55.248 | 20.283 |
| Fragment prototype | 150,430 | 17.184 | 8.488 | 31.455 | 9.345 |

The panel contained the real Bato source revisions and the two edited-copy
trees described in [validation](../../../docs/validation.md), the full
`src` trees of parking_lot 0.11.2 and 0.12.5 in one directory, and the
unrelated mixed-data control. The prototype reduced archive bytes and
packing time on the panel. The 256 KiB segments increased restore wall
time compared with the branch.
