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

## Multi-megabyte files and directories

The 512 KiB and 1 MiB prototype layouts use `--variants 512k 1m`. They
keep the inner CM chunk size at 256 KiB. All 20 CPU-only round trips in
this screen and its follow-up matched the original paths and file bytes.
The branch binary was `fcc0683`, with nine logical CPUs available and a
7.2 GiB memory budget.

Two unchanged standalone inputs were run once per layout. Webster was
a 3 MiB Silesia slice starting at byte 13,597,871. Its automatic geometry
and the explicit 512 KiB geometry produced identical archive bytes.
Times are wall seconds.

| File | Geometry | Archive bytes | Pack | Restore |
|---|---|---:|---:|---:|
| Canterbury kennedy.xls | Auto, 128 KiB | 18,359 | 6.800 | 2.226 |
| Canterbury kennedy.xls | 512 KiB | 14,365 | 18.952 | 5.586 |
| Canterbury kennedy.xls | 1 MiB | 13,275 | 12.014 | 10.419 |
| Webster slice | Auto, 512 KiB | 560,737 | 15.081 | 8.876 |
| Webster slice | Explicit 512 KiB | 560,737 | 16.281 | 10.943 |
| Webster slice | 1 MiB | 532,894 | 23.411 | 13.785 |

The real Bato directory contains the entire `src` tree from each of
`c244a25` and `fcc0683` (2,199,023 original file bytes). One run per layout
gave the following results.

| Bato source directory layout | Archive bytes | Pack wall | Restore wall |
|---|---:|---:|---:|
| Stock branch | 199,229 | 14.195 | 4.388 |
| Prototype, 128 KiB | 179,201 | 9.663 | 4.690 |
| Prototype, 256 KiB | 172,488 | 11.212 | 7.029 |
| Prototype, 512 KiB | 156,153 | 19.505 | 7.804 |
| Prototype, 1 MiB | 150,257 | 27.413 | 13.149 |

The constructed Silesia directory contains four 384 KiB slices and one
edited revision of each (3,148,204 original file bytes). The source
offsets are nci 5,208,487, x-ray 651,003, ooffice 2,800,005, and reymont
406,810. For slice index `i` in that order, replace 127 bytes at
`10947 + 17*i` and 421 bytes at `267449 + 29*i` with the same number of
bytes starting 9,897 later in the original. Insert original bytes 99,803
through 100,421 at `179877 + 13*i`. Original and revised copies were
stored in separate directory trees.

The branch and 256 KiB rows are medians of three runs. The other rows
are single runs. CPU seconds include scanning, staging, and rebuilding.

| Edited Silesia directory layout | Archive bytes | Pack wall | Restore wall | Pack CPU | Restore CPU |
|---|---:|---:|---:|---:|---:|
| Main 1.22, c5db485 | 403,876 | 53.156 | 29.116 | 100.714 | 29.108 |
| Stock branch | 801,461 | 35.378 | 9.210 | 186.141 | 44.224 |
| Prototype, 256 KiB | 403,316 | 17.543 | 4.595 | 73.704 | 22.444 |
| Prototype, 512 KiB | 400,531 | 23.193 | 5.425 | 66.391 | 15.421 |
| Prototype, 1 MiB | 395,470 | 45.248 | 10.417 | 74.598 | 16.601 |

The 256 KiB prototype beat the branch on archive bytes and both wall
times on the edited Silesia tree. The source directory had slower
restores at every tested size, and the larger layouts on the standalone
files also traded wall time for size. These results retain 256 KiB as
the prototype default.
