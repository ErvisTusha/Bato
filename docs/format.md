# Bato format 1.19

A `.bato` stream holds file bytes or a serialized directory tree. Bato 0.8
accepts format 1.19 as its underlying compression stream.

The stream is a 10-byte header, zero or more blocks, and a footer. Fixed-width
integers are little-endian. Variable integers are canonical unsigned LEB128:
seven value bits per byte, low group first, high bit set when another byte
follows; redundant leading zero groups and values exceeding u64 are rejected.

## Header

| Offset | Bytes | Field |
|---|---|---|
| 0 | 4 | `BATO` |
| 4 | 1 | Major `1` |
| 5 | 1 | Minor `19` (the version of this document; other values are rejected) |
| 6 | 1 | Flags: bit 0 directory archive; other bits zero |
| 7 | 1 | Segment log2, 16..30 (default 26) |
| 8 | 1 | CM chunk log2, 12..20, no greater than segment log2 (default 18) |
| 9 | 1 | Model table-cap log2, 0..24 (default 22) |

The version identifies the entire format, model, and transforms. Geometry is
encoded once. Each segment is independent and has at most `2^segment_log2`
original bytes; only the final segment may be shorter.

## Blocks

The first byte packs type in bits 0..2 and the transform in bits 3..5 (low
three bits) and bit 7 (fourth bit). Bit 6 is set
on the CM chunks (types 0 and 1) of a segment whose model skipped the primer
(below); all chunks of a segment agree, and independent types must clear it.
The standalone end marker is the byte `255`; transform value 15 is
undefined and rejected. Original length `n`
follows as a variable integer. Types 1, 3, 4, 5 and 7 then encode payload length
`c` as a variable integer; it must be strictly smaller than `n`. The payload
is followed by four bytes of CRC32C over the original, untransformed bytes.
All lengths, segment boundaries and output limits are checked before payload
allocation. Blocks cannot cross segment boundaries.

| Type | Meaning | Payload |
|---|---|---|
| 0 | Raw CM chunk | Exactly `n` bytes; update history if a later CM coded chunk needs it |
| 1 | Coded CM chunk | `c` bytes of arithmetic code |
| 2 | Independent stored segment | Exactly `n` bytes, no model |
| 3 | Independent literal/copy segment | `c` bytes of commands |
| 4 | Independent static Huffman segment | `c` bytes of table and bits |
| 5 | Huffman-coded literal/copy segment | Token-stream length variable integer, then static table and bits |
| 6 | Repeated-byte segment | One byte repeated `n` times, `n >= 2` |
| 7 | Contextual digit-run segment | Literal skeleton and integer streams |

### Primer

Unless bit 6 of its chunks' first byte is set, before the first coded or raw
chunk of every CM segment both encoder and decoder train a fresh model on a
fixed 104,049-byte primer (SHA-256
`add44b9a887b2d00ac973cf83b35f5f78ed2a42b265cdd22cd463ea63ee3bb20`, stored in
`src/core/cm/primer.bin`). The primer is English prose and code in several
languages written for this purpose; it is never emitted. Training runs every
model exactly as coding would, one byte at a time,
with the primer as the history. The segment's own bytes then continue that
history, so contexts, matches and word statistics span the boundary. Table
and match-index sizes are chosen for primer plus
segment length. When the bit is set, the segment's model starts empty and the history is just
the segment. The encoder tries both for segments up to 16 MiB and keeps the
smaller; larger segments always read the primer. The primer is part of the
format: changing a byte of it requires a new format version.

A segment holds either one independent block (2..7), or CM chunks (0..1).
All nonfinal CM chunks have the geometry's full chunk length. A CM segment
uses a single transform. Its model persists across chunks, including raw
chunks; decoders can skip updates for an all-raw segment or a raw suffix
after its final coded chunk. Models restart at segment boundaries.

The encoder compares complete block bytes, including size fields and CRCs.
It tries the untransformed independent representations and five reversible
filters. The existing x86 selector admits an x86 candidate. Context mixing
remains a candidate with its existing x86 decision. Early termination uses
only emitted coder bytes and mandatory framing as a lower bound, never a
sampled compression estimate. No filename, MIME routing, or quality modes
are involved. Encoder search is not normative; decoding semantics are.

### Literal/copy commands

A command begins with variable integer `v`. If even, `v >> 1` is a nonzero
literal length, followed by that many bytes. If odd, copy length is
`(v >> 1) + 4`, followed by a variable integer distance. Distance must be
nonzero and no greater than the already produced length. Overlapping
copies repeat already decoded bytes. Commands must produce exactly `n`
bytes with no trailing input. Decoder copies expand geometrically without
per-byte prediction or training.

### Static Huffman tables

The table starts with a variable integer symbol count, 2..256, then pairs
of symbol byte and code-length byte in strictly increasing symbol order.
Lengths are 1..15 and must form a complete prefix tree. Codes are canonical:
assign increasing codes by length then symbol. Their bits are emitted
most-significant code bit first into the least-significant available stream
bit. Unused high bits in the final byte are zero; trailing bytes and
nonzero padding are rejected. The original size determines decoded length.
For type 5 the prefixed length determines decoded token bytes, bounded by
`n + 10`, and the token decoder must then produce exactly `n` original bytes.
The decoder builds one lookup table; it does no adaptive training.

### Coded CM payload

The payload is a binary arithmetic code: 32-bit interval, 16-bit probabilities, byte renormalization and the shortest zero-padded final code point. The decoder checks the canonical prefix and the exact tail. Bit probabilities come from the context-mixing model in `src/core/cm`, which is the normative definition for this version:

* hashed order 0–4 and 6 contexts and four word/line contexts, each a nibble bit-history map in a shared cache-line table of `min(4 * (primer + segment length), 2^cap)` buckets rounded up to a power of two, 64 bytes each;
* an order-0 map, a byte-run model, a match model, a record-stride model and a bit-neighbourhood model;
* four auxiliary experts with their own mixers, the first two being: text (orders 5 and 7, word + previous byte, three words) and sparse structure (bytes 2, 3, 4 back and the 2/3 pair);
* a lag expert (`src/core/cm/models/stride.rs`, normative): long match distances vote for a lag between 64 B and 2 MiB at which the data repeats approximately (a video frame, an image plane, fixed-size records); every 4,096 bytes the lag with the best score over the last 2,048 bytes, among the most voted distances and their divisors, is chosen, and four contexts built from the bytes one lag back and their left neighbours feed a third auxiliary expert (constant when no lag is known);
* a row expert (`src/core/cm/models/rows.rs`, normative): every 32,768 bytes, once 16,384 + 31,232 bytes of history exist, the distance in 16..16,384 bytes whose sampled mean difference is lowest, if under 9/16 of the median over all distances, is taken as the row distance, and the distance in 1..4 with the lowest difference as the sample unit; eight contexts built from the left, upper, upper-left and upper-right neighbours, quantised neighbour differences and the previous sample's vertical difference feed a fourth auxiliary expert (constant when no row is known);
* three first-layer mixers, two final mixers whose outputs are averaged (selected by the previous byte's high bits and by the match model's state), three state-conditioned calibrators and eight chained APMs (partial byte; match state; two and three previous bytes; order 4; current word; word pair; expected match byte), each blending 1:1 with the previous stage's output.

All arithmetic is integer, so every platform produces identical bytes.

### Reversible transforms

| ID | Transform |
|---|---|
| 0 | None |
| 1 | x86 call/jump addresses |
| 2 | Byte delta at distance 1 |
| 3 | Byte delta at distance 2 |
| 4 | Byte delta at distance 4 |
| 5 | Byte-plane shuffle of 2-byte words |
| 6 | Byte-plane shuffle of 4-byte words |
| 7 | Inflated deflate streams with recipes (below); CM chunks only |
| 8 | x86 call/jump addresses and x86-64 RIP-relative displacements |
| 9 | Transform 7, then transform 1 over the transformed bytes; CM chunks only |
| 10 | Transform 7, then transform 8 over the transformed bytes; CM chunks only |
| 11 | Transform 1 with the segment reach |
| 12 | Transform 8 with the segment reach |
| 13 | Transform 9 with the segment reach |
| 14 | Transform 10 with the segment reach |

Delta writes `out[i] = input[i] - input[i-distance]` modulo 256, using zero
for positions before the start. Inverse adds restored preceding bytes.
Shuffle groups byte 0 from each full word, then byte 1, and so on; an
incomplete trailing word stays unchanged at the end. The inverse undoes
this permutation. Filters operate within a segment and CRCs are verified
after inversion.

The x86 transforms `1`, `8`, `11` and `12` rewrite relative operands to
absolute addresses before modeling. Every chunk of the segment carries the
transform; a segment mixing transform values is corrupt. Chunk CRCs cover the
original bytes and are checked after the inverse.

Let `len` be the number of bytes rewritten and `v` the signed 32-bit
little-endian value in bytes `i+1..=i+4`. Position `i` (`i ≤ len − 5`) is a
site when its operand is in reach and

* byte `i` is `0xE8` or `0xE9` (all four transforms), or
* for transforms `8` and `12`: `i > 0`, `byte[i] & 0xC7 == 0x05` (a ModRM byte
  with `mod = 00`, `r/m = 101`) and byte `i − 1` is one of these opcode bytes
  (hexadecimal):

  `01 03 09 0B 10 11 13 1B 21 23 28 29 2A 2B 2C 2D 2E 2F 31 33 38 39 3A 3B 51
  54 57 58 59 5C 5E 63 6E 6F 7E 7F 80 81 83 84 85 86 87 88 89 8A 8B 8D AF B6
  B7 BE BF C6 C7 D6 EF FF`

Near reach (transforms `1` and `8`): byte `i + 4` is `0x00` or `0xFF`. At a
site the 24-bit value in bytes `i+1..=i+3` becomes `(value + i) mod 2^24`;
the inverse subtracts `i`.

Segment reach (transforms `11` and `12`): `−i ≤ v < W` with
`W = max(len, 2^24)`. At a site `v` becomes `v + i` when `v + i < W` and
`v − W` otherwise; the inverse restores `v − i` from a value `v ≥ 0` and
`v + W` from a negative one.

The encoder rewrites positions from `len − 5` down to `0`, the decoder from
`0` upwards with the same site test, which is exact for every input.

Selection is the encoder's; the decoder inverts whatever the block headers
declare. The encoder counts, over a set of sites, how often the low 24 bits
of an operand repeat as stored and how often they repeat after adding `i`. A
set qualifies when the absolute form repeats at least 32 times and more than
1.5 times as often as the stored form. The segment reach is chosen when its
call/jump sites give more than 17/16 of the near reach's absolute repeats,
otherwise the near reach. No transform is used unless the call/jump sites of
the chosen reach qualify. The RIP-relative sites of that reach are added when
they qualify.

### Deflate transform (7)

The encoder looks for DEFLATE (RFC 1951) streams at every byte offset of a
segment whose first three bits and counts could begin a dynamic-Huffman
block, without regard to names or signatures. A stream qualifies if it parses
to its end, re-encodes to exactly the same bytes, is at least 256 bytes long,
inflates to at least its own length, and its recipe is at most twice its
length; the streams must total at least 1,024 bytes. The encoder keeps the
transform only if the coded result is smaller than the best alternative. The segment is then
coded as the transformed bytes `L meta body pad`, where `L` is the meta length
as a variable integer, the body is the segment with each qualifying stream
replaced by its inflated bytes and `pad` is zero or one zero byte. The
transformed length may not exceed four times the segment size, because
inflated streams are larger than the bytes they replace. A transformed
segment ends with a chunk shorter than the chunk size, and the padding byte
ensures that the transformed length is never a multiple of the chunk size, so
that this short chunk always exists; a transformed segment that is not the
last of the stream is therefore allowed to end in a short chunk, which
no other segment may. Every chunk of the segment carries transform `7` and CM
chunk types only. Chunk lengths and CRC32C values describe the transformed
bytes; the stream footer's total and CRC32C describe the restored segment.

Transforms `9`, `10`, `13` and `14` are transform `7` followed by the x86
rewrite of transform `1`, `8`, `11` or `12` over the whole transformed byte string `L meta body pad`,
with positions counted from its start. The encoder applies the selection rule
of the x86 transforms to the transformed bytes. Everything said about
transform `7` holds for them, and chunk lengths and CRC32C values describe
the bytes after the x86 rewrite. The decoder verifies the chunks, inverts the
x86 rewrite and then restores the deflate streams.

`meta` is a variable-integer padding count (0 or 1), a variable-integer stream
count, then for each stream, in order,
its offset in the body, its compressed length, its inflated length, its
recipe length and the recipe. The decoder copies the body between streams
and, for each stream, rebuilds the deflate bytes from the inflated bytes and
the recipe; the rebuilt length must equal the compressed length and the
restored segment may not exceed the segment size.

A recipe is: one byte holding an imitated-encoder level `0..8` (bit 7 selects
a 16-bit hash); the block count; for each block a tag (bits 0..1 type 0 stored,
1 fixed, 2 dynamic; bit 2 final) followed by the stored length, or the token
count (and for dynamic blocks the header bit count and the header bits
exactly as in the stream); then the number of mispredicted tokens and for
each its index delta and token (`0 byte` literal, or `1 length distance`).
Tokens are those a zlib-style hash-chain matcher with the level's chain,
good, lazy and nice lengths would emit (greedy for levels 0..2, lazy with
one-step look-ahead above); the matcher indexes every position of a token
except interior positions of greedy matches longer than the level's lazy
length, searches at most 32,506 bytes back, and ignores length-3 matches
farther than 4,096 bytes. A block may declare at most 2^20 tokens. The
normative definition of the matcher is `src/core/cm/deflate/predict.rs`.

## Footer

The standalone byte `255`, canonical variable integer total original length,
and four-byte CRC32C over the complete original output. Trailing bytes are
rejected. Empty streams are 16 bytes. A stored segment adds one tag byte,
one length variable integer and four CRC bytes; it has no payload-size field.

## Directory archives (flag bit 0)

The stream's decoded bytes are a catalogue of entries, each starting with a kind byte:

| kind | layout | meaning |
|---|---|---|
| `1` | path length (u16) · path | directory |
| `2` | path length (u16) · path · size (u64) · contents · CRC32C (u32) | file |
| `3` | path length (u16) · path · size (u64) · id (u32) · CRC32C (u32) | file identical to the earlier file entry number `id`, stored once |
| `0` | — | end of catalogue |

Paths are UTF-8, use `/` separators, and are relative. Absolute paths, `..`, empty components, duplicates and names that would alias on the target filesystem are rejected. Extraction writes into a staging directory, enforces entry, path, metadata and output limits, verifies every checksum, and publishes the tree only after everything is valid.

## Contextual digit-run representation (type 7)

This is a reversible byte transform, without a file-format parser. Maximal
ASCII decimal runs of 1..19 bytes become markers `0,1` in a literal skeleton.
Longer digit runs remain literal. Literal zero bytes are escaped as `0,0`.
The payload starts with variable integers for skeleton length, then one
skeleton representation type (2..6), its payload length, and that payload.
Nested type 7 is rejected. Skeleton output is bounded by twice original size.

The context is the low eight bits of FNV-1a over at most eight immediately
preceding original non-digit bytes, stopping at a digit or stream start.
The seed is 2166136261 and multiplier 16777619. A variable integer context
count (1..256) follows the skeleton. Entries occur in strictly increasing
context-byte order. Each entry has context byte, predictor byte, then three
bounded packets: low seven value bits, high value groups, leading-zero count.
A context's low and zero-count streams have one byte per numeric run. The
high stream is canonical ULEB128 for each value shifted right seven bits.

Predictor 0 sends the absolute unsigned integer. Predictor 1 subtracts the
last value of that context. Predictor 2 subtracts `2*last - previous`. All
operations wrap modulo 2^64. Modes 1/2 map the residual by signed zigzag
`(residual << 1) XOR (0 - (residual >> 63))`; mode 0 has no zigzag. Histories
start at zero. Decoding reverses that mapping, restores the predictor,
updates history, emits the indicated leading zeroes and canonical decimal
digits, and rejects a resulting run longer than 19 bytes. Every packet
must be consumed completely. Contexts are inferred from restored bytes.

A packet is original length, one codec byte, payload length, payload. Codec
0 is exact raw bytes; 1 repeats its one payload byte; 2 uses static rANS;
3 uses literal/copy commands; 4 prefixes the command-stream length and then
rANS-codes those commands. Command-stream output is bounded by packet length
plus ten. Low-stream counts sum to at most ceil(original length / 2), since digit
runs are maximal. High-stream output is bounded by nine bytes per number
and, across contexts, twice the original length plus two bytes. All lengths are
checked before allocation, and outer block/output limits and CRCs still apply.

Static rANS uses scale 12, total frequency 4096, lower state 2^23 and byte
renormalization. Its table is a variable integer symbol count (1..256), then
ascending symbol byte and variable integer frequency-minus-one. Frequencies
must sum exactly to 4096. The table is followed by a four-byte little-endian
state in [2^23,2^31), then renormalization bytes. For slot `state & 4095`,
the inverse update is `frequency * (state >> 12) + slot - cumulative`.
Read bytes while the resulting state is below 2^23. The final state must be
2^23 with no remaining payload. Frequencies are transmitted, so the decoder
does not train a model. Encoder normalization and predictor choices are not
part of the decoding contract. All choices compare complete serialized bytes.

## Archive envelopes and split volumes

Directory archives can be wrapped in separately versioned envelopes. Plain
`BATO` streams use format 1.19. All wrapper
integers are unsigned and little-endian.

- `BAE1` is followed by one binary [age v1](https://c2sp.org/age) passphrase
  payload containing the complete compressed Bato directory stream.
- `BAP1` is followed by a u64 public stream length, the public Bato directory
  stream, and one binary age v1 passphrase payload. The age plaintext contains
  the 32-byte SHA-256 of the compressed public stream, a u16 protection scope
  count, the scopes, and the private Bato directory stream. Each scope is a
  u16 byte length and that many UTF-8 bytes in catalogue path syntax. A scope
  includes the exact path and its descendants. The sections may share ancestor
  directories. Full extraction authenticates the private stream and checks its
  binding to the public stream before publishing files. Public-only extraction
  verifies only the public Bato stream and any split-volume checksum.
- A split archive is a directory with an `index.bato` file and the volume
  directory named by that index. The index contains `BAV1`, a u64 maximum part
  size, a u64 total archive size, the 32-byte SHA-256 of the complete archive, and
  a 12-byte ASCII alphanumeric suffix. Volume files are named `part-0000001`
  onward in `volumes-<suffix>`. Each part is the maximum size except the last
  when there is a remainder. The assembled bytes form a plain or encrypted
  directory archive as described above. Updates publish a new index only after
  the complete replacement generation is written.

Password envelopes use age's scrypt recipient with `N=2^17, r=8, p=1`.
Encryption randomness is independent of Bato's deterministic compressor.

## Checkpoints and recovery

Bato 0.8 keeps the normal `BATO` 1.19 stream. A local `BAJ2` checkpoint
records complete segment boundaries, accumulated CRC32C, and SHA-256
identities for the source and staged output. Private checkpoint records use
independent age X25519 payloads. The job identity is stored in an age
passphrase envelope. These files are local job state and are separate from
the finished archive.

The optional `BAV2` split index uses the same fields as `BAV1`, followed
by one 32-byte SHA-256 per numbered volume and a final SHA-256 for the
`parity` file. The parity file is the bytewise XOR of all numbered volumes,
with the shorter final data volume treated as zero-padded. The parity file
length is the longest data volume length. The combined archive SHA-256
must verify after reconstruction.

A `BAR1` salvage index is generated by fully decoding and verifying an
intact plain stream. It stores the original stream header, encoded and
decoded segment lengths, per-segment CRC32C and SHA-256, and directory
member paths and raw extents. A SHA-256 footer protects the index against
accidental damage. Salvage validates each segment independently and uses
the directory catalogue checks when restoring complete files.

`BAV3` extends the split manifest with a parity count byte after the
generation suffix, followed by the data-volume SHA-256 values and the
parity-volume SHA-256 values. The parity files are named `parity-01` and
onward. They are Reed-Solomon parity over GF(2⁸), with short data volumes
zero-padded for coding. Data and parity use the systematic matrix of
`reed-solomon-erasure` 6. Reads verify the assembled archive hash.

`BAR2` extends `BAR1` with a 32-byte SHA-256 of the complete source archive
immediately after its magic. Selective indexed extraction requires this
binding. Salvage accepts both versions and can operate on a damaged source.

### Embedded indexes

`BAI1` contains a little-endian u64 BAR2 length, the complete BAR2 index, and
the original directory stream to EOF. The BAR2 index binds the stream's
SHA-256 and independently checks decoded segment SHA-256 values. Its size
limit is 64 MiB.

`BAE2` contains a single password-encrypted age v1 file to EOF. Its plaintext
is the same body as BAI1, starting with the length. The age stream
authenticates its header, encrypted chunks and final length. The reader seeks
within the authenticated plaintext to reach selected Bato segments and
verifies their decoded SHA-256 values from the encrypted BAR2 index. A
selection verifies the index, final age chunk and chunks it reads. Full
verification reads the entire stream and checks the BAR2 stream binding.
