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

## Independent residual segments

A follow-up used the four inputs that contain references from the panel above.
All layouts used the same recipes. The stock branch and all three layouts
ran in a rotating order with three iterations per input. All 48 round trips
passed. Times are sums of per-input medians in seconds.

| Layout | Archive bytes | Pack wall | Restore wall | Pack CPU | Restore CPU |
|---|---:|---:|---:|---:|---:|
| Stock branch | 141,329 | 14.341 | 6.241 | 42.984 | 18.281 |
| 256 KiB | 92,329 | 10.380 | 6.261 | 15.597 | 6.169 |
| 128 KiB | 96,029 | 11.215 | 5.765 | 17.498 | 7.050 |
| 64 KiB | 99,812 | 13.407 | 5.013 | 25.404 | 10.282 |

The earlier format 1.22 total for these four inputs was 96,533 bytes.
The 128 KiB layout saved 504 bytes against that baseline. It cut pack and
restore wall time by 21.8% and 7.6% against the contemporaneous stock branch.
The 128 KiB layout moved `parking_lot` from 19,384 to 23,084 bytes and reduced
its restore wall time from 2.533 to 2.232 s compared with the 256 KiB layout.
The other three archives tied the 256 KiB sizes. The executable directory
restored more slowly than the stock branch at 128 KiB (0.670 to 0.825 s).

The unrelated control goes through the stock branch path for every layout.
Adding its 58,101 bytes puts the 128 KiB total at 154,130 bytes, above the
five-input format 1.22 total of 150,565 bytes. The prototype's default
remains 256 KiB. The 128 KiB measurements apply to the fragment-bearing
directories tested here.
