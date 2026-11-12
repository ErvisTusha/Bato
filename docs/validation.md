# Independent validation log

This file records how Bato was checked against data it was not tuned on, and
how it compares with other compressors. It is a log of what was measured,
including losses. Nothing here is estimated; a number that was not measured
is marked "not measured".

## Method and limits

* **Ad hoc runs.** These are single runs on one 4-core, 15 GiB Linux machine,
  not the rigorous campaign of `scripts/benchmark_suite.py` (no warmups, no
  repetitions). Timing noise is about 5%. Sizes are exact and deterministic.
* **Lossless check.** Every Bato figure marked OK was decoded and compared
  byte for byte with the original (`cmp`).
* **Competitors**, at their strongest setting: `gzip -9`, `zip -9`,
  `rar -m5 -ma5 -md64m`, `7z -mx9`, `xz -9e`, `zstd --ultra -22`, and for some
  files `paq8px -9L` (v217; `-12L` needs more than the 15 GiB available). 7z
  and rar sizes include their archive headers.
* **Data roles.**
  * *Pilot* (11 Canterbury files), *held-out* (31 files: Calgary and
    `data-research` samples) and *media development* (6 QCIF and 3 CIF video
    slices, WAVs, bitmaps, photos as PPM) were used to make design decisions.
    Treat gains on them as optimistic.
  * *Fresh* (56 files: curl and zlib sources, Office documents) was created
    after the first design decisions and consulted for a few later ones.
  * *Sealed* (24 files, 293 MB, downloaded: enwik8, raw video, MP4/WebM,
    WAV, MP3, arXiv PDFs, PNG, ELF and PE executables, a .deb) was never used
    for a decision. It is measured once per format version.
  * *New files* are downloaded after the decisions they test and are used
    once: `text8` (100 MB), `sqlite-amalgamation.zip`, `sqlite-doc.zip`,
    `bash-5.2.tar.gz`, `lena.png`, `sample-1.ppt`, `sample.csv`.
* Files are downloaded from public sources; none is generated, except the
  correctness fixtures of `scripts/run_tests.py`.

## Versions

| Version | Format | Notes |
|---|---|---|
| Bato 0.7.0 | 1.3 | starting point |
| Bato 0.8.0 | 1.9 | built-in primer, nine calibration stages, larger tables, deflate recompression |
| Bato 0.8.0 | 1.10 | lag and row experts for video, audio and image data |
| Bato 0.8.0 | 1.11 | dormant experts skipped, table prefetch, one slow model dropped |
| Bato 0.8.0 | 1.12 | row expert with eight contexts (quantised neighbour differences) |
| Bato 0.8.0 | 1.13 | built-in primer text without the removed model |
| Bato 0.8.0 | 1.14 | calibration stage keyed by one previous byte removed |
| Bato 0.8.0 | 1.15 | large deflate streams |
| Bato 0.8.0 | 1.16 | x86-64 RIP-relative transform; x86 rewrite beside deflate streams |
| Bato 0.8.0 | 1.17 | segment reach for the x86 transforms |
| Bato 0.8.0 | 1.18 | the segment reach qualifies on its own |

## Development, held-out and fresh sets (total compressed bytes)

| Set | Original | 0.7 | 1.9 | 1.10 | **1.11** | gzip | zip | rar | 7z | xz |
|---|---|---|---|---|---|---|---|---|---|---|
| Pilot, 11 files | 2,810,784 | 357,616 | 341,001 | 339,610 | **339,935** (1.12: 339,961) | 730,732 | 732,269 | 512,223 | 494,801 | 493,080 |
| Held-out, 31 files | 1,728,724 | 400,224 | 377,880 | 377,966 | **378,287** (1.12: 378,287) | 585,999 | 590,838 | 542,651 | 513,273 | 509,668 |
| Fresh, 56 files | 609,660 | 303,173 | 186,041 | 185,752 | **185,956** (1.12: 185,838) | 319,336 | 327,426 | 327,585 | 321,819 | 317,492 |

`paq8px -9L` totals: pilot 300,270 (smaller than Bato by 12%), fresh 215,587
(larger than Bato 1.9's 186,041). On the pilot paq8px wins on 10 of 11 files,
by a wide margin on `kennedy.xls` (7,694 against about 13,800) and `ptt5`
(16,907 against about 27,400).

## Sealed set (24 files, 293,278,671 bytes), format 1.9 only

| | Bytes |
|---|---|
| Bato 0.7 | 129,803,554 |
| Bato 1.9 | 127,616,390 |
| xz -9e | 142,009,560 |

Bato 1.9 is 10.1% smaller than xz in total. Per file: enwik8 19,650,762
against xz 24,831,648; `cargo` ELF 9,123,273 against 11,086,624; four
arXiv PDFs 33 to 51% below xz. No gain on MP3, MP4 and WebM (equal to xz,
MP3 0.17% larger). Raw video `akiyo_qcif.y4m`: Bato 2,081,770 against xz
1,915,416, a loss that format 1.10 addresses on comparable video but that was
not re-measured on the sealed file. Versions 1.10 and 1.11 are not measured
on the sealed set.

## Speed and size, five files, all versions (sizes in bytes; times in seconds)

| File | Original | 1.10 size · enc / dec | 1.11 size · enc / dec | xz size · enc / dec |
|---|---|---|---|---|
| alice29 (text, 152 KB) | 152,089 | 35,900 · 5.2 / 4.4 | 35,903 · 4.4 / 2.6 | 48,528 · 0.1 / 0.0 |
| lcet10 (text, 427 KB) | 426,754 | 86,262 · 12.5 / 10.7 | 86,384 · 10.8 / 6.2 | 119,488 · 0.2 / 0.0 |
| ptt5 (fax, 513 KB) | 513,216 | 27,361 · 9.0 / 8.7 | 27,372 · 8.5 / 5.1 | 39,860 · 0.3 / 0.0 |
| ZIP, 13.5 KB | 13,526 | 7,927 · 3.0 / 2.8 | 7,992 · 2.4 / 1.8 | 13,368 · 0.0 / 0.0 |
| video, 4 MB | 4,000,001 | 847,014 · 61.6 / 90.6 | 847,841 · 54.2 / 53.6 | 1,046,704 · 2.0 / 0.1 |

Format 1.11 decodes about 40% faster than 1.10 for +0.01% to +0.14% size on
these files (+0.82% on the 13.5 KB ZIP). Against 1.10 it costs 1.0% on the
media development set and 0.06 to 0.11% on text, code and binaries. Bato
remains 100 to 1,000 times slower than xz.

## New files

`text8` (100,000,000 bytes, SHA-1 `0dc3edebc970dcc96137e7deda4d9995af9d93de`),
format 1.11, round trip OK:

| Compressor | Bytes | Encode |
|---|---|---|
| **Bato 1.11** | **18,976,467** | 730 s (decode 669 s) |
| xz -9e | 23,221,180 | 126 s |
| 7z -mx9 | 23,224,497 | 120 s |
| zstd -22 | 23,337,375 | 120 s |
| rar -m5 | 25,237,526 | 43 s |
| gzip -9 | 33,048,240 | 7.7 s |

Other new files (format 1.10 / 1.11 where measured):

| File | Original | Bato 1.10 | Bato 1.11 | xz | 7z | rar | gzip |
|---|---|---|---|---|---|---|---|
| sample.csv | 321 | 137 | 137 | 232 | 302 | 284 | 201 |
| lena.png | 473,831 | 473,165 | 473,110 | 473,912 | 473,984 | 474,029 | 473,933 |
| sample-1.ppt | 912,384 | 706,934 | 707,165 | 707,468 | 706,895 | 711,711 | 719,616 |
| sqlite-amalgamation.zip | 2,729,858 | 1,280,639 | 1,284,386 | 2,715,760 | 2,716,216 | 2,719,680 | 2,729,716 |
| sqlite-doc.zip | 10,760,211 | 4,539,001 | 4,558,880 | 10,294,848 | 10,290,042 | 10,352,929 | 10,578,842 |
| bash-5.2.tar.gz | 10,950,833 | 5,434,718 | 5,440,967 | 10,923,776 | 10,916,575 | 10,943,059 | 10,943,779 |

Bato encode / decode seconds, format 1.10 then 1.11: sqlite-amalgamation.zip
140 / 218 then 109 / 111; sqlite-doc.zip 478 / 643 then 351 / 341;
bash-5.2.tar.gz 457 / 721 then 387 / 380. All round trips OK.

## Known losses and limits

* Speed: encoding and decoding run at roughly 30 to 130 KB/s, far slower than
  every other tool above.
* PNG: no gain, because the deflate stream is split into IDAT chunks.
* MP3, MP4, WebM and already-compressed formats other than deflate: no gain.
* Office 97 (`sample-1.ppt`): no gain over 7z.
* Structured binaries such as spreadsheets and fax images: `paq8px` is much
  smaller.
* The deflate transform codes a segment with its streams inflated only if the
  result is at most four segment sizes (256 MiB for the default 64 MiB
  segments). Streams that inflate to more are coded as incompressible bytes.
  Archives that inflate to a large size take correspondingly long to code and
  to decode, on one worker per segment (see "Large deflate streams").

## How to repeat

Download the files named above, run `bato -c FILE -o OUT`, `bato -d OUT -o
BACK`, `cmp FILE BACK`, and compare `stat -c%s OUT` with the competitors'
sizes. Update this file after each format version, adding a fresh set that
was not used for any decision.

## Mixed folder, validation round 2 (format 1.11, 2026-11-09)

A new folder of real downloaded files, never used before: GNU `sed` 4.9 and
`tar` 1.35 source trees, four arXiv PDFs (VGG, distillation, VAE, dropout),
the ripgrep 14.1.0 ELF binary and six small images. 2,258 files, 41,678,510
bytes. Compared as a solid archive of the directory (`bato archive`, 7z solid,
rar `-s`, tar piped through the single-stream tools).

| Tool | Bytes | Encode |
|---|---|---|
| **Bato 1.11** | **9,011,381** | 495 s (extract 451 s) |
| 7z -mx9 | 10,021,693 | 14.4 s |
| tar + xz -9e | 10,069,104 | 20.9 s |
| tar + zstd -22 | 10,368,533 | 20.3 s |
| rar -r -s -m5 | 10,526,379 | 4.4 s |
| tar + gzip -9 | 14,544,723 | 2.4 s |
| zip -r -9 | 16,064,039 | 2.1 s |

`bato extract` followed by `diff -r` against the source folder reported the
trees identical. Bato is 10.1% smaller than 7z and 10.5% smaller than xz on
this folder, whose bulk is text sources; the margin is much smaller than on
single deflate archives, and extraction takes about 30 times longer than the
others' compression.

## Photographs, format 1.12 (2026-11-09)

Format 1.11 made photographs larger. Format 1.12 gives the row
expert eight contexts instead of four (upper-right neighbour, quantised
neighbour differences, the previous sample's vertical difference). Four Kodak
photos as PPM (media development set, so optimistic):

| Photo | 1.10 | 1.11 | 1.12 | 1.12 against 1.11 | 1.12 against 1.10 |
|---|---|---|---|---|---|
| kodim03 | 302,840 | 306,144 | 302,372 | -1.2% | -0.2% |
| kodim06 | 430,378 | 458,351 | 453,487 | -1.1% | +5.4% |
| kodim09 | 369,662 | 377,400 | 371,706 | -1.5% | +0.6% |
| kodim12 | 342,500 | 349,490 | 343,838 | -1.6% | +0.4% |

Other sets, 1.12 against 1.11: pilot 339,961 against 339,935 (+26 bytes),
held-out 378,287 (equal), fresh 185,838 against 185,956 (-118), media
development set 11,305,308 against 11,345,533 (-0.35%). `alice29.txt` is
35,903 bytes in both and decodes in 2.7 to 2.8 s in both (measured while other
jobs ran). `kodim03` decodes in 16.0 s against 16.6 s. Round trips (`cmp`) OK,
including compression with a 4 GiB budget and decompression with 1 GiB.
A later commit makes each expert mixer read only the inputs its contexts need;
archives are byte-identical and `kodim09` decodes 2 to 4% faster (1.12:
19.85 and 20.32 s; trimmed: 19.55 and 19.55 s; 1.11: 18.08 and 17.98 s, same
machine), so 1.12 still decodes photos about 10% slower than 1.11.
Photos remain larger than in 1.10, most on `kodim06` (+5.4%). The gates
(`cargo test`, unit tests, `run_tests.py`) passed.

## Validation round 3, format 1.12 (2026-11-09)

Six new real downloads, used for no decision (sizes in bytes; best other
tool in bold; Bato times are one run on a 4-core machine). Every Bato round
trip was checked with `cmp`: all OK.

| File | Original | Bato 1.12 | xz -9e | 7z -mx9 | zstd -22 | rar -m5 | gzip -9 | Bato against best other | Bato enc / dec |
|---|---|---|---|---|---|---|---|---|---|
| `pg2600.txt` (text, CRLF) | 3,359,610 | 691,740 | **935,548** | 935,712 | 943,285 | 994,858 | 1,220,192 | -26.1% | 42 s / 41 s |
| `fd-v10.2.0` musl `.tar.gz` | 1,681,679 | 1,063,226 | 1,665,136 | **1,663,299** | 1,665,226 | 1,667,700 | 1,665,926 | -36.1% | 58 s / 51 s |
| ripgrep 14.1.0 Windows `.zip` | 2,020,417 | 1,667,082 | 2,008,088 | **2,006,879** | 2,009,539 | 2,013,556 | 2,011,765 | -16.9% | 80 s / 77 s |
| `dummy.pdf` | 13,264 | 9,867 | 12,236 | 12,305 | **12,209** | 12,466 | 12,335 | -19.2% | 1 s / 2 s |
| `Example.ogg` (Vorbis) | 104,793 | 102,365 | **102,956** | 103,047 | 102,903 | 103,240 | 102,800 | -0.6% | 4 s / 2 s |
| `PNG_transparency_demonstration_1.png` | 224,566 | 223,516 | 223,636 | 223,658 | 223,133 | 223,570 | **222,950** | **+0.3% (loss)** | 23 s / 0 s |

Bato wins on five files. The large wins come from recompressing deflate data
(tar.gz, zip, PDF) and from modelling text; Vorbis audio is incompressible
for every tool (Bato -0.6%). On the PNG, gzip is 0.25% smaller and zstd
0.17% smaller than Bato: the image data is already deflate-coded in IDAT
chunks, which the transform does not handle (see Known losses).

## Validation round 4, format 1.12 (2026-11-09)

Five more real downloads, used for no decision (bytes; best other tool in
bold; one run each on a 4-core machine; every Bato round trip `cmp` OK).

| File | Original | Bato 1.12 | xz -9e | 7z -mx9 | zstd -22 | rar -m5 | gzip -9 | Bato against best other | Bato enc / dec |
|---|---|---|---|---|---|---|---|---|---|
| `compressed.tracemonkey-pldi-09.pdf` | 1,016,315 | 404,014 | **801,280** | 801,813 | 801,929 | 808,262 | 817,607 | -49.6% | 30 s / 27 s |
| `Free_Test_Data_1MB_PDF.pdf` | 1,052,352 | 682,815 | 883,092 | 882,844 | **881,288** | 888,538 | 894,858 | -22.5% | 45 s / 21 s |
| bat 0.24.0 Windows `.zip` | 3,150,280 | 2,816,069 | 3,138,860 | **3,137,330** | 3,140,177 | 3,145,612 | 3,141,284 | -10.2% | 97 s / 87 s |
| `FullMoon2010.jpg` | 2,143,958 | 2,056,572 | 2,093,200 | 2,093,066 | **2,081,833** | 2,086,697 | 2,083,702 | -1.2% | 32 s / 30 s |
| pandoc 3.1.11 Linux `.tar.gz` (31 MB, 149 MB inflated) | 31,352,376 | 30,915,519 | 30,839,848 | **30,819,415** | 30,888,402 | 30,921,196 | 30,956,667 | **+0.3% (loss)** | 407 s / 402 s |

The pandoc archive is a loss: its deflate stream inflates to about 149 MB,
more than one 64 MiB segment, so the deflate transform is not applied (see
Known losses); Bato then codes the compressed bytes, as it does for any
incompressible data, and ends 0.3% above 7z and 0.25% above xz. Not
measured: `linux-2.6.39.4.tar.xz` (63 MB, already xz-coded; no transform
exists for it), because it would take about 40 minutes for a predictable
result.

Offline estimate for the pandoc loss (not a Bato feature): gunzip the archive
to its 149,483,520-byte tar and compress that. xz -9e gives 18,190,344; Bato
1.12 with the default three 64 MiB segments gives 14,654,209 (753 s on the
4-core machine), against 30,915,519 for the archive as it is now (-52.6%) and
30,819,415 for 7z. A deflate transform that could span segments would
therefore be worth about half the size on archives of this kind, at the cost
of coding the inflated bytes (about 5 times the data) and of a format change.
It is not implemented.
## Compression speed, format 1.12 (2026-11-09)

For inputs up to 2 MiB the encoder codes the segment twice, with and without
the built-in primer, and keeps the smaller result. The two runs are
independent, so they now run on two threads when at least two workers are
admitted (`--threads 1` keeps them sequential). The archives are byte for byte
the same for every thread count; only the time changes. Wall seconds on the
4-core machine, one run each, otherwise idle; sizes in bytes:

| File | Size | Compress before | Compress after | Decompress before | Decompress after |
|---|---|---|---|---|---|
| alice29.txt | 35,903 | 4.58 | 2.79 | 2.75 | 2.73 |
| lcet10.txt | 86,384 | 10.17 | 6.15 | 5.87 | 5.91 |
| ptt5 | 27,411 | 8.75 | 5.09 | 5.20 | 5.05 |
| news_qcif_first4MB.y4m | 842,125 | 56.47 | 56.94 | 54.75 | 55.08 |
| kodim09.ppm | 371,706 | 33.84 | 19.42 | 17.58 | 18.59 |

Compression of the four inputs under 2 MiB is 39 to 43% faster. The 4 MB video
has a single trial and is unchanged; decompression is unchanged (differences
are measurement noise of about 5%). Sums over the sets are the same as before:
pilot 339,961, held-out 378,287, fresh 185,838, media development 11,305,308.

## Validation round 5, format 1.14 (2026-11-09)

Ten new real downloads, used for no decision: the overfit check for the changes
of formats 1.12 to 1.14, which were decided on the development sets. Bytes;
one run each on a 4-core machine; every Bato round trip checked with `cmp`: all
OK. The last column is Bato against the smallest competitor.

| File | Original | Bato 1.14 | xz -9e | 7z -mx9 | zstd -22 | rar -m5 | gzip -9 | Bato against best other | Bato enc / dec |
|---|---|---|---|---|---|---|---|---|---|
| pg1342.txt (text, CRLF) | 772,386 | 162,953 | 223,640 | 223,710 | 226,053 | 236,322 | 274,759 | -27.136% | 6 s / 6 s |
| linux `core.c` (C source) | 302,667 | 54,345 | 74,408 | 74,416 | 75,823 | 78,390 | 87,898 | -26.963% | 3 s / 2 s |
| busybox 1.35 (ELF x86-64) | 1,131,168 | 505,359 | 624,496 | 624,112 | 647,756 | 621,178 | 699,282 | -18.645% | 9 s / 9 s |
| arXiv 1706.03762 (PDF) | 2,215,244 | 489,394 | 1,033,444 | 1,034,012 | 1,041,546 | 1,068,868 | 1,130,149 | -52.644% | 164 s / 165 s |
| demo.docx | 1,311,881 | 861,904 | 1,304,780 | 1,302,696 | 1,301,050 | 1,303,091 | 1,301,176 | -33.753% | 22 s / 20 s |
| gawk-5.3.0.tar.gz | 6,254,035 | 2,566,993 | 6,248,192 | 6,247,334 | 6,245,197 | 6,254,214 | 6,251,080 | -58.897% | 169 s / 167 s |
| hyperfine Windows .zip | 582,727 | 522,399 | 581,332 | 581,506 | 581,207 | 582,510 | 581,006 | -10.087% | 13 s / 12 s |
| testorig.jpg (JPEG) | 5,770 | 5,522 | 5,660 | 5,733 | 5,598 | 5,819 | 5,666 | -1.358% | 1 s / 0 s |
| FLAC (16-bit stereo) | 549,954 | 541,765 | 542,364 | 542,343 | 541,812 | 543,278 | 541,903 | -0.009% | 5 s / 0 s |
| Big Buck Bunny 10 s .mp4 | 991,017 | 988,480 | 988,912 | 988,880 | 988,729 | 990,112 | 988,803 | -0.025% | 8 s / 0 s |

Bato is smallest on all ten files. The margins are large where the data has
structure the others cannot model (source, text, executable, PDF, Office and
deflate archives: 10 to 59%) and small where the data is already compressed by
a method Bato cannot undo (JPEG 1.4%, FLAC and MP4 under 0.05%, which is a tie
within what a different header size could change). Bato takes 100 to 1,000
times longer than the others on the larger files (gawk and the PDF: about
165 s each way).

## Large deflate streams, format 1.15 (2026-11-09)

Until format 1.14 a deflate-transformed segment could not be larger than the
segment size (64 MiB), and inflated data is always larger than what it
replaces, so a full segment of zip, gzip or tar.gz data was never transformed
and was coded as incompressible bytes. Format 1.15 allows four segment sizes;
a short last chunk ends such a segment (a pad byte guarantees one) and the
memory bound counts the larger buffers (the existing per-worker bound already
exceeded what the transformed path needs, so admission is unchanged).

`pandoc-3.1.11-linux-amd64.tar.gz` (31,352,376 bytes, inflating to
149,483,520), default segments, one run, round trip `cmp` OK:

| | Bytes | Encode | Decode | Peak RSS (enc / dec) |
|---|---|---|---|---|
| Bato 1.14 | 30,915,519 | 407 s | 402 s | not measured |
| **Bato 1.15** | **15,440,023** | 1,115 s | 1,048 s | 761 / 485 MiB |
| 7z -mx9 | 30,819,415 | | | |
| xz -9e | 30,839,848 | | | |

The archive is 50.1% smaller than before and 49.9% smaller than 7z, and takes
2.7 times as long to encode and 2.6 times as long to decode: the 149 MB of
inflated bytes are modelled by one worker, about 18 minutes each way on this
machine. An offline estimate (the inflated tar coded as three independent
segments) had given 14,654,209 bytes; the single transformed segment also
carries the recipes. The mechanism was also checked with `gawk-5.3.0.tar.gz`
and 16 MiB segments (round trip OK; 171 s encode, 166 s decode; 430 / 353 MiB).

Sums over the four development sets against 1.14: pilot, held-out and media
identical; fresh +6 bytes (the pad byte of each transformed segment).

## Speed over the whole speed work, format 1.12 against 1.15 (2026-11-09)

The same five files, run one after another on an idle 4-core machine, one run
each (about 5% noise). Format 1.12 is the state before the timing tool and
the changes it led to: model trials on two threads, calibration-row prefetch,
the removal of one calibration stage and the primer text change (formats
1.13 and 1.14). Seconds.

| File | Bytes 1.12 | Bytes 1.15 | Compress 1.12 | Compress 1.15 | Decompress 1.12 | Decompress 1.15 |
|---|---|---|---|---|---|---|
| alice29.txt | 35,903 | 35,928 (+0.07%) | 4.58 | 1.59 (-65%) | 2.75 | 1.62 (-41%) |
| lcet10.txt | 86,384 | 86,384 (+0.00%) | 10.17 | 3.51 (-65%) | 5.87 | 3.35 (-43%) |
| ptt5 | 27,411 | 27,408 (-0.01%) | 8.75 | 4.10 (-53%) | 5.20 | 3.32 (-36%) |
| news_qcif_first4MB.y4m | 842,125 | 841,358 (-0.09%) | 56.47 | 36.30 (-36%) | 54.75 | 35.30 (-36%) |
| kodim09.ppm | 371,706 | 372,120 (+0.11%) | 33.84 | 10.90 (-68%) | 17.58 | 9.88 (-44%) |

Decompression is 36 to 44% faster and compression 36 to 68% faster. The sizes
moved by -0.09% to +0.11% (`kodim09.ppm` and `alice29.txt` grew by 414 and 25
bytes). Compression of the three inputs under 2 MiB gains most from the two
parallel trials; the 4 MB video has a single trial and gains from the faster
model only. Decompression of the same archives at 1 worker has no parallel
part, so its gain is all from the faster model.

## Where the time and the bytes go: `--timing` and `scripts/ablate.py` (2026-11-09)

Two tools answer "which step is slow" and "which step is needed".

* `bato -c FILE -o OUT --timing` (also with `-d`) prints, without changing a
  byte of the archive (a test checks this): the archive bytes by purpose
  (headers, checksums, footer, coded and stored payload); the time of every
  whole-segment step (x86 selection, fast candidates, deflate scan, primer
  training, model coding, inverse transforms, checksums); the time of every
  step inside the model, measured on one byte in 32 and scaled, with the cost
  of the clock subtracted; and the share of bits in which each optional model
  (match, records, four experts) took part.
* `python3 scripts/ablate.py --work DIR FILE...` patches a temporary copy of
  the sources so that one step at a time contributes nothing, builds it, and
  prints the size change over the files; `--combo "apm=1,5;isse=2"` skips
  several stages together. Every ablated archive is checked to decode.

**First finding (timing).** On `alice29.txt` the calibration (APM) stages took
45% of the model time although they are 3% of the instructions: each of six
stages reads a 8.6 MB table at a random row, a cache miss per bit. The row is
known at the start of the bit, so it is now prefetched. Archives are
byte-identical; decoding is 12 to 38% faster (seconds, one run each, idle
machine):

| File | Decode before | Decode after |
|---|---|---|
| alice29.txt | 2.73 | 1.77 |
| lcet10.txt | 5.91 | 3.82 |
| ptt5 | 5.05 | 4.44 |
| news_qcif_first4MB.y4m | 55.08 | 36.70 |
| kodim09.ppm | 18.59 | 11.47 |

Compression of the same files fell from 2.79, 6.15, 5.09, 56.94 and 19.42 s
to 1.67, 4.62, 4.59, 38.39 and 12.01 s. The built-in primer is trained before
every segment and costs 0.64 s of the 1.63 s decode of `alice29.txt` (39%):
a fixed cost per segment.

**Second finding (ablation, calibration and ISSE stages).** Total size over
103 development files (pilot, held-out, fresh, three media slices; 6.9 MB,
1,399,877 bytes coded). Size change when the step is removed:

| Step | Change | Worst file | |
|---|---|---|---|
| APM 0 (partial byte) | -0.018% | +0.051% | neutral |
| **APM 1 (one previous byte)** | **-0.064%** | +0.394% | **removed in 1.14** |
| APM 2 (match state) | +0.047% | +0.909% | kept |
| APM 3 (two previous bytes) | +0.071% | +0.909% | kept |
| APM 4 (three previous bytes) | +0.050% | +0.909% | kept |
| APM 5 (order 4) | -0.023% | +0.909% | see below |
| APM 6 (current word) | +0.046% | +0.909% | kept |
| APM 7 (word pair) | +0.075% | +2.446% | kept |
| APM 8 (expected match byte) | +0.159% | +1.916% | kept |
| ISSE 0 / 1 / 2 | +0.059% / +0.099% / -0.008% | | kept |

Stages interact: removing stages 1 and 5 together gives -0.050%, less than
stage 1 alone (-0.064%); 0, 1 and 5 together +0.015%. A ten-file panel with
three text files suggested that stages 1, 5 and 6 could all go (-0.203%); on
the real sets that removal made text larger (pilot +0.29%, held-out +0.24%)
while helping fresh (-0.29%) and media (-0.27%), so it was rejected.

**Format 1.14** removes only stage 1. Totals against format 1.13:

| Set | 1.13 | 1.14 | Change |
|---|---|---|---|
| Pilot | 339,945 | 340,045 | +0.029% |
| Held-out | 378,303 | 378,328 | +0.007% |
| Fresh | 185,051 | 184,773 | -0.150% |
| Media development | 11,304,500 | 11,291,839 | -0.112% |
| Sum | 12,207,799 | 12,194,985 | -0.105% |

Decoding is 4 to 5% faster (alice29 1.678 to 1.613 s; kodim09 10.25 to
9.70 s). `alice29.txt` grows by 26 bytes (35,902 to 35,928).

**All other steps, full development panel** (the same 103 files, format 1.14,
1,398,981 bytes coded). Size change when the step is removed, and the worst
single file; the time column is the share of the step's group in the model
or whole-segment time (`--timing`).

| Step | Size | Worst file | Group time |
|---|---|---|---|
| lag expert | +3.093% | +23.371% | experts 24.1% |
| built-in primer | +2.512% | +69.597% | primer training 37.5% (small files) |
| deflate recompression | +7.276% | +131.780% | 0.1% |
| sparse expert | +1.002% | +4.528% | experts 24.1% |
| row expert | +0.869% | +3.203% | experts 24.1% |
| first-layer mixers 0 / 1 / 2 | +0.315% / +0.708% / +0.525% | up to +3.474% | 6.3% |
| context order 2 | +0.599% | +3.096% | context maps 15.1% |
| match model input | +0.494% | +4.151% | inputs 3.5% |
| text expert | +0.343% | +4.111% | experts 24.1% |
| second / first final mixer | +0.360% / +0.309% | +2.121% / +1.186% | 5.1% |
| context order 3 / 1 / 0 | +0.298% / +0.240% / +0.200% | up to +2.0% | context maps |
| word contexts 4 / 1 / 2 / 3 | +0.188% / +0.168% / +0.139% / +0.134% | up to +3.0% | context maps |
| second encoding trial without primer | +0.168% | +1.809% | encoding CPU 61.6% |
| bit-neighbourhood input | +0.138% | +5.765% | 6.3% |
| context order 4 / 6 | +0.096% / +0.015% | +1.370% / +5.283% | context maps |
| record-stride model | +0.048% | +2.041% | 3.5% |
| byte-run input | +0.007% | +0.488% | 3.5% |
| x86 transform | +0.000% | +0.000% | 0.1% |
| ISSE 2 | -0.009% | +0.990% | ISSE 3.5% |
| order-0 input | -0.010% | +0.990% | 3.5% |

The panel has few executables, so the x86 transform shows nothing here; a
ten-file panel with an x86 binary gave +0.272% for it. Every step costs
little time on its own (a context map about 1.5% of model time) but the
lag expert, the primer, deflate recompression, the experts and the mixers
each earn far more than they cost. Only ISSE 2 and the order-0 input are
neutral or negative one at a time, and they are not worth removing (see the
rejected experiments).

## Validation round 6, partial: folders from release tarballs (format 1.15, 2026-11-09)

An overfitting check on data never used for design: release tarballs were
downloaded, unpacked, and the folder re-packed by every tool. Six were
planned; **three were measured** (the round was stopped early), so this is
partial evidence. Firefox, Python and VS Code were not measured.

Sources (SHA-256 of the download): Antigravity IDE 2.6.0 linux-x64 `.tar.gz`
(`b060a7c7…f94356`, 195,292,315 bytes), Node.js v22.23.4 linux-x64 `.tar.xz`
(`1805bcd3…37e9ce`, 31,029,996 bytes), Go 1.27.3 linux-amd64 `.tar.gz`
(`65efbc80…379b40`, 70,588,678 bytes). Bato binary: `main` at format 1.15,
SHA-256 `ce2a7815…46af15`; 4 cores, 15 GiB RAM, default threads and memory.
Symlinks were removed from each tree first (Bato cannot store them; Node had
3), and all tools packed that same tree. "Same type" is deterministic `tar`
piped to `gzip -9` (Antigravity, Go) or `xz -9e` (Node). Other tools:
`xz -9e` and `zstd --ultra -22` on the tar, `7z -mx9`, `rar -m5`.

| Tree | Bytes (files) | Same type, max | xz -9e | 7z -mx9 | zstd -22 | rar -m5 | Bato | vs same type | vs best other |
|---|---|---|---|---|---|---|---|---|---|
| Antigravity | 595,461,667 (15,898) | 193,826,513 | 124,350,148 | 115,498,890 | 129,944,442 | 139,420,003 | 108,612,849 | -43.96% | -5.96% |
| Node.js | 196,453,447 (4,750) | 31,040,920 | 31,040,920 | 29,426,604 | 33,513,271 | 35,429,629 | 26,275,756 | -15.35% | -10.71% |
| Go | 244,139,206 (15,657) | 69,939,410 | 39,725,940 | 38,584,607 | 42,447,576 | 46,390,543 | 35,139,108 | -49.76% | -8.93% |

Size: Bato is smallest on all three. Every restored tree was compared with the
packed tree (per-file SHA-256 and directory list): identical.

Time (single runs, wall seconds): Bato loses to every tool by a wide margin.

| Tree | Bato pack / unpack | Bato peak RSS pack / unpack (MiB) | gzip -9 | xz -9e | zstd -22 | 7z -mx9 | rar -m5 |
|---|---|---|---|---|---|---|---|
| Antigravity | 1664.50 / 1608.74 | 2413 / 1609 | 42.94 | not timed | not timed | not timed | not timed |
| Node.js | 511.18 / 499.89 | 1465 / 1019 | n/a | 85.7 | 76.9 | 39.7 | 13.2 |
| Go | 593.29 / 579.08 | 2283 / 1488 | 23.57 | 152.6 | 159.1 | 68.3 | 22.8 |

Caveats: the tar-based tools also store modes, owners and times, which Bato
does not; the competitors' unpack times were not measured (they are far faster
than Bato's); timings are single runs.

## x86-64 addresses and code beside deflate streams, format 1.16 (2026-11-09)

Two changes. Transform 8 also rewrites RIP-relative displacements of x86-64
code to absolute addresses. Transforms 9 and 10 apply the x86 rewrite on top
of the deflate transform: before, a segment holding any deflate stream used
the deflate transform alone and lost the x86 rewrite completely, call/jump
sites included.

Single files (one thread, exact bytes, each restored and compared with `cmp`):

| File | Bytes | 1.15 | 1.16 | Change |
|---|---|---|---|---|
| Go executable (preprofile) | 2,379,482 | 694,014 | 679,098 | -2.15% |
| Go executable (gofmt) | 3,144,728 | 893,292 | 871,341 | -2.46% |
| C executable (bash) | 1,446,024 | 443,903 | 415,685 | -6.36% |
| C executable (git) | 4,066,232 | 1,231,793 | 1,152,207 | -6.46% |

Folders (all workers; restored tree compared with `diff -r`; single-run wall
seconds; the two trees are those of validation round 6):

| Folder | Bytes | 1.15 | 1.16 | Change | Pack / unpack 1.15 | Pack / unpack 1.16 |
|---|---|---|---|---|---|---|
| 4 executables, 3 text files | 11,224,308 | 2,902,185 | 2,766,175 | -4.69% | 84.66 / 80.17 | 85.91 / 80.38 |
| The same plus one gzip file | 11,278,499 | 3,093,125 | 2,765,991 | -10.58% | 87.33 / 82.32 | 85.49 / 81.52 |
| Go 1.27.3 tree | 244,139,206 | 35,139,108 | 33,782,384 | -3.86% | 593.29 / 579.08 | 591.61 / 569.79 |
| Node.js v22.23.4 tree | 196,453,447 | 26,275,756 | 25,408,068 | -3.30% | 511.18 / 499.89 | 505.60 / 494.16 |

The 103 development files total 1,398,987 bytes with both formats and no file
changes size. Time: no measurable change either way (differences are inside
single-run noise). Limits: RIP-relative sites are only rewritten when the
displacement is within 16 MiB, so the 120 MB Node executable gains almost
nothing from transform 8 itself (its gain comes from the call/jump rewrite
beside deflate data); the first version of this change, without transforms 9
and 10, left both trees at exactly their 1.15 sizes. The Antigravity tree was
not re-measured.

## Segment reach for the x86 transforms, format 1.17 (2026-11-10)

In executables larger than 16 MiB most call targets and data lie further away
than the near reach of format 1.16 sees. Transforms 11 to 14 accept any target
inside the rewritten bytes; the encoder uses them for a segment only when they
find more than 17/16 of the near reach's repeated call targets.

| Data | Bytes | 1.16 | 1.17 | Change | Pack / unpack 1.16 | Pack / unpack 1.17 |
|---|---|---|---|---|---|---|
| Node.js v22.23.4 tree | 196,453,447 | 25,408,068 | 25,212,353 | -0.77% | 505.60 / 494.16 | 506.53 / 489.55 |
| Go 1.27.3 tree | 244,139,206 | 33,782,384 | 33,782,384 | 0 | 591.61 / 569.79 | 574.62 / 561.23 |
| Antigravity 2.6.0 tree | 595,461,667 | not measured (1.15: 108,612,849) | 107,810,883 | -0.74% against 1.15 | 1664.50 / 1608.74 (1.15) | 1637.24 / 1616.58 |
| Four executables under 5 MB | 11,036,466 | 3,118,331 | 3,118,331 | 0 | | |
| 103 development files | 6,898,748 | 1,398,987 | 1,398,987 | 0 | | |

All three restored trees are identical to the originals (`diff -r`; the
Antigravity download matched the round 6 SHA-256); single-run
wall seconds, no measurable time change. Using the segment reach for every
segment was measured first and rejected: Node 25,212,353 (-0.77%) but Go
34,043,763 (+0.77%), because the wider interval admits more false sites
among many small executables; on the first 64 MiB of the Node executable
alone it gave 13,183,004 against 13,398,198 (-1.61%). Limit: a target in an
earlier segment is not rewritten by the segment reach.

## Segment reach qualifying on its own, format 1.18 (2026-11-10)

A measured loss led here. On the 199,855,320-byte Antigravity executable
alone, `7z -mx9` gave 49,191,016 bytes and Bato 1.17 49,549,470 (+0.73%);
`xz -9e` gave 57,065,392. Cut into 64 MiB slices Bato won each one (13,930,615
/ 19,200,490 / 16,418,403 against 7z's 16,463,250 / 22,387,465 / 19,055,675),
so 7z's whole-file result came from its x86 filter. Bato 1.17 applied no x86
transform to this file: the rule required the near reach to qualify first,
and on the first slice its call sites repeat 194,564 times as absolute values
against 142,600 as stored (below the 1.5 ratio), while the segment reach finds
387,256 against 148,689. In 1.18 the chosen reach must qualify, not the near
one.

| Data | 1.17 | 1.18 | Change |
|---|---|---|---|
| Executable slice 1 (64 MiB) | 13,930,615 | 13,528,964 | -2.88% |
| Executable slice 2 (64 MiB) | 19,200,490 | 17,588,919 | -8.39% |
| Executable slice 3 (62.6 MiB) | 16,418,403 | 16,418,403 | 0 |
| Antigravity 2.6.0 tree (595,461,667 bytes) | 107,810,883 | 106,026,019 | -1.66% |
| Node.js v22.23.4 tree | 25,212,353 | 25,212,353 | 0 |
| Go 1.27.3 tree | 33,782,384 | 33,782,384 | 0 |
| Four executables under 5 MB | 3,118,331 | 3,118,331 | 0 |
| 103 development files | 1,398,987 | 1,398,987 | 0 |

The whole executable as one stream with 1.18: 47,536,248 bytes, 3.36% below
7z's 49,191,016, restored and compared with `cmp`. Time on this file (single
runs, 4 cores): Bato pack 526.42 s, unpack 510.73 s; 7z pack 85.6 s, unpack
2.7 s. Bato is 6 times slower to pack and 189 times slower to unpack. Antigravity tree with 1.18:
pack 1681.29 s, unpack 1606.30 s (1.17: 1637.24 / 1616.58; single runs),
restored tree identical. Every slice and tree was restored and compared.

## Exact candidate comparison and filter trial, commit 89a3365 (2026-11-12)

The changes listed in `docs/ultra-optimization.md` (format 1.18 unchanged)
against the previous 1.18 build, same machine, single runs. Gates pass: fmt,
clippy, `cargo test --release`, Python unit tests, `scripts/run_tests.py`.

| Data | Before | After | Change | Pack s before / after | Unpack s before / after |
|---|---|---|---|---|---|
| 103 development files (4 at a time) | 1,398,987 | 1,360,250 | -2.77% | 46.8 / 54.2 | 32.4 / 27.9 |
| Folder: 4 executables, 3 text files, 1 gzip (11,278,499 bytes) | 2,765,991 | 2,765,008 | -0.04% | 79.01 / 171.38 | 79.23 / 62.87 |
| Node.js v22.23.4 tree (196,453,447 bytes) | 25,212,353 | 25,216,813 | **+0.02% (4,460 bytes larger)** | 515.37 / 1150.74 | 491.30 / 389.52 |
| Four executables under 5 MB | 3,118,331 | 3,117,859 | -0.02% | not timed | not timed |

Every output was restored and compared (`cmp`, `diff -r`). No development file
grew; the largest gains are `hopper.bmp` 28,418 to 23,706 (-16.6%) and
`sample-3s.wav` 211,340 to 177,579 (-16.0%), from coding the delta or shuffle
filter the independent codec picked with the model as well; Office files gain
0.1% to 0.5%. Unpacking is 14% to 21% faster. Losses: packing code or deflate
data is about 2.2 times slower, because each such segment is modeled once per
candidate; and the Node.js tree is 4,460 bytes larger, cause not yet found.
The Go and Antigravity trees were not re-measured.

## Rejected experiments (kept so they are not repeated)

* **Early exit for incompressible chunks** (abort the model attempt after two
  chunks coded to 99% or more of their length). Identical archives on the
  pilot, held-out, fresh and media sets and 60% faster encoding of a 2.8 MB
  MP4 (38.7 s to 15.3 s), but on a segment of 1 MiB of MP4 followed by
  427 KB of text the output grew from 1,138,718 to 1,258,621 bytes (+10.5%),
  because the text after the video was no longer modeled. Size has priority
  over encoding time, so it was not merged. A safe variant would need a check
  of all remaining chunks, including repeated blocks.
* **Row expert contexts, format 1.12.** Ablation on four Kodak PPMs (total
  1,471,403 bytes): replacing context 4 (quantised gradient and differences)
  with a constant costs +0.26%, context 5 (left, upper-left) +0.20%, context 6
  (average, upper-right difference) +0.17%, context 7 (previous sample's
  vertical difference) +1.40%. All four earn their time. Replacing context 6
  by a second cross-channel context (previous sample against its upper and
  left neighbours) gave 1,468,968 on the photos (-0.17%) but made the media
  development set larger (11,311,195 against 11,305,308), so it was not kept.
* **Removing ISSE expert 2 and the order-0 input** (each -0.01% over the
  103 development files). Together: pilot +0.044%, held-out -0.006%, fresh
  -0.126%, media +0.014%, sum +1,502 bytes. ISSE 2 alone: pilot +0.037%,
  held-out -0.003%, fresh -0.064%, media +0.013%, sum +1,482 bytes. Stage
  effects measured one at a time do not add up; only a removal verified on
  all four sets counts.
* **Sleeping text expert** (the text expert inactive until the last 6 bytes
  were printable ASCII or white space, as the lag and row experts sleep when
  they have no distance). Against 1.14: pilot +0.16%, held-out +0.04%, fresh
  -0.06%, media +0.04%. The expert helps on media files too.
* **Earlier table prefetch** (requesting the first-nibble slots of the order,
  word, text and sparse contexts before the match, lag and row work of the byte
  start). Archives identical; decoding of `alice29.txt` and `kodim09.ppm` within
  the run-to-run noise of about 10% (the same binary measured 1.84 and 1.57 s).
* **Timing sampler aliasing.** The first sampler lapped one byte in 32; the
  row-distance scan runs every 32,768 bytes, so every scan fell on a lapped
  byte and its cost was scaled 32 times (13% of model time against a real
  0.5%). The period is now 61.
* **Resuming a long match after a mispredicted byte** (-0.04% on the media
  development set).
* **Orders 5 and 8 in the main contexts** (+0.1%), **a third expert with orders 9 to 24** (+0.3%),
  **an extra primer of 7.6 KB prose** (-0.2% on the pilot, 0% on the fresh set).
* **Text files before binary files in folder archives, format 1.15.** An
  `xz` proxy suggested -0.48% on the Go tree; real Bato gave 35,568,931 bytes
  against 35,139,108 in path order (+1.22%; restored tree verified). Path
  order keeps related files adjacent, which matters more.
* **Near conditional jumps (`0F 8x rel32`) in the x86 transform.** Larger on
  three of four x86-64 executables (+0.33% to +0.47%), -0.13% on the fourth.
* **Other RIP-relative site tests for transform 8.** Any preceding byte:
  -0.14% and -4.16% on two executables where the chosen opcode set gives
  -2.15% and -6.36%. Only `8B`, `8D`, `89`: -2.01% and -6.00%.
* **Sign-extended 27-bit operands for every x86 site** (instead of the
  segment reach): -1.50% on the first 64 MiB of the Node executable, but
  +0.12% to +0.62% on four executables under 5 MB. With 27 bits only for
  RIP-relative sites: -0.12% on the slice, +0.05%, -0.08%, +0.54%, +0.51% on
  the four.
* **Segment reach without the 16 MiB minimum window:** `bash` 432,755 against
  415,685 bytes (+4.1%); targets in zero-initialised data lie past the file.
* **Huge-page hint for the model tables (`madvise` with `MADV_HUGEPAGE`
  on allocations of 4 MiB or more), format 1.18.** Archives identical. Decoding
  `c_git` (4,066,232 bytes) three times each: 30.39 / 32.21 / 30.57 s without,
  31.95 / 29.67 / 30.22 s with; no gain beyond run-to-run noise. Forcing huge
  pages system-wide (`always`) gave 28.75 / 28.94 / 28.93 s against 29.12 /
  31.22 / 30.38 s, a gain of about 4% that needs a system setting, so it is
  not available to the program itself.
* **Skipping the plain model run once an x86 or deflate candidate has won**
  (on commit 89a3365). The folder of executables stays at 2,765,008 bytes and
  packs in 73.72 s instead of 171.38 s, but `ptt5` grows from 27,408 to 27,584
  bytes (+0.64%): there the plain run beats the transform the statistics chose.
  Size outranks packing time, so the exact comparison stays.

