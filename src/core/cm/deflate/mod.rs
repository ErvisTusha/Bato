//! Exact DEFLATE parsing and re-encoding: the first stage of recompressing
//! deflate streams found inside other data.
//! Provenance: RFC 1951 (public specification) and the canonical-code
//! decoding it describes; original implementation, no third-party code.
//!
//! `parse` accepts a stream only when `encode` reproduces its bytes exactly,
//! so every rejection (non-zero padding, non-canonical length symbols,
//! incomplete codes, distances before the start) simply leaves the bytes to
//! the generic coder.

pub(in crate::core::cm) mod container;
mod predict;
mod recipe;

const MAX_BITS: usize = 15;
const END_OF_BLOCK: usize = 256;
const LITERAL_SYMBOLS: usize = 288;
const DISTANCE_SYMBOLS: usize = 30;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Token {
    Literal(u8),
    Match { length: u16, distance: u16 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// Stored block of this many bytes.
    Stored(u16),
    Fixed,
    /// Dynamic block; the header bits after BTYPE are kept verbatim so the
    /// encoder's choices (symbol 16/17/18 runs, HCLEN) survive.
    Dynamic {
        header: Vec<u8>,
        header_bits: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Block {
    pub(super) kind: Kind,
    pub(super) last: bool,
    /// Tokens in this block, excluding the end-of-block symbol.
    pub(super) tokens: usize,
}

#[derive(Debug)]
pub(super) struct Parsed {
    pub(super) data: Vec<u8>,
    pub(super) blocks: Vec<Block>,
    pub(super) tokens: Vec<Token>,
    /// Input bytes the stream occupies, including the final partial byte.
    pub(super) consumed: usize,
}

struct BitReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl BitReader<'_> {
    fn bit(&mut self) -> Option<u32> {
        let byte = *self.bytes.get(self.position / 8)?;
        let bit = (byte >> (self.position % 8)) & 1;
        self.position += 1;
        Some(u32::from(bit))
    }

    fn bits(&mut self, count: u8) -> Option<u32> {
        let mut value = 0;
        for shift in 0..count {
            value |= self.bit()? << shift;
        }
        Some(value)
    }

    fn align_zero(&mut self) -> Option<()> {
        while !self.position.is_multiple_of(8) {
            if self.bit()? != 0 {
                return None;
            }
        }
        Some(())
    }
}

/// Canonical Huffman decoding table in the counts/symbols form of RFC 1951.
struct Decoder {
    count: [u16; MAX_BITS + 1],
    symbol: Vec<u16>,
}

impl Decoder {
    /// Builds a decoder from code lengths; `None` unless the code is complete,
    /// or a single one-bit code (which zlib emits for lone distances).
    fn new(lengths: &[u8]) -> Option<Self> {
        let mut count = [0u16; MAX_BITS + 1];
        for &length in lengths {
            count[usize::from(length)] += 1;
        }
        let used = lengths.len() - usize::from(count[0]);
        let mut left = 1i32;
        for &c in &count[1..] {
            left = (left << 1) - i32::from(c);
            if left < 0 {
                return None;
            }
        }
        let single = used == 1 && count[1] == 1;
        if used == 0 || (left != 0 && !single) {
            return None;
        }
        let mut offset = [0usize; MAX_BITS + 2];
        for bits in 1..=MAX_BITS {
            offset[bits + 1] = offset[bits] + usize::from(count[bits]);
        }
        let mut symbol = vec![0u16; used];
        for (value, &length) in lengths.iter().enumerate() {
            if length != 0 {
                symbol[offset[usize::from(length)]] = value as u16;
                offset[usize::from(length)] += 1;
            }
        }
        Some(Self { count, symbol })
    }

    fn decode(&self, reader: &mut BitReader) -> Option<usize> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for bits in 1..=MAX_BITS {
            code |= reader.bit()? as i32;
            let count = i32::from(self.count[bits]);
            if code - count < first {
                return self
                    .symbol
                    .get((index + (code - first)) as usize)
                    .map(|&s| usize::from(s));
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        None
    }
}

fn fixed_lengths() -> ([u8; LITERAL_SYMBOLS], [u8; 32]) {
    let mut literal = [8u8; LITERAL_SYMBOLS];
    literal[144..256].fill(9);
    literal[256..280].fill(7);
    (literal, [5u8; 32])
}

/// Reads a dynamic block header into literal and distance code lengths.
fn read_dynamic_lengths(reader: &mut BitReader) -> Option<(Vec<u8>, Vec<u8>)> {
    let literals = reader.bits(5)? as usize + 257;
    let distances = reader.bits(5)? as usize + 1;
    let code_lengths = reader.bits(4)? as usize + 4;
    if literals > 286 || distances > DISTANCE_SYMBOLS {
        return None;
    }
    let mut meta = [0u8; 19];
    for &slot in CODE_LENGTH_ORDER.iter().take(code_lengths) {
        meta[slot] = reader.bits(3)? as u8;
    }
    let meta = Decoder::new(&meta)?;
    let mut lengths = Vec::with_capacity(literals + distances);
    while lengths.len() < literals + distances {
        let symbol = meta.decode(reader)?;
        let (value, repeat) = match symbol {
            0..=15 => (symbol as u8, 1),
            16 => (*lengths.last()?, 3 + reader.bits(2)? as usize),
            17 => (0, 3 + reader.bits(3)? as usize),
            _ => (0, 11 + reader.bits(7)? as usize),
        };
        if lengths.len() + repeat > literals + distances {
            return None;
        }
        lengths.resize(lengths.len() + repeat, value);
    }
    if lengths[END_OF_BLOCK] == 0 {
        return None;
    }
    let distance = lengths.split_off(literals);
    Some((lengths, distance))
}

fn copy_match(data: &mut Vec<u8>, length: usize, distance: usize) {
    for _ in 0..length {
        let byte = data[data.len() - distance];
        data.push(byte);
    }
}

fn read_tokens(
    reader: &mut BitReader,
    codes: (&Decoder, &Decoder),
    parsed: &mut Parsed,
    max_output: usize,
) -> Option<usize> {
    let mut count = 0;
    loop {
        let symbol = codes.0.decode(reader)?;
        if symbol == END_OF_BLOCK {
            return Some(count);
        }
        if symbol < END_OF_BLOCK {
            parsed.data.push(symbol as u8);
            parsed.tokens.push(Token::Literal(symbol as u8));
        } else {
            let index = symbol - 257;
            let extra = reader.bits(*LENGTH_EXTRA.get(index)?)?;
            let length = usize::from(LENGTH_BASE[index]) + extra as usize;
            // Symbol 284 with extra 31 would also spell 258; only 285 is canonical.
            if length > MAX_MATCH || (index == 27 && length == MAX_MATCH) {
                return None;
            }
            let slot = codes.1.decode(reader)?;
            let extra = reader.bits(*DISTANCE_EXTRA.get(slot)?)?;
            let distance = usize::from(DISTANCE_BASE[slot]) + extra as usize;
            if distance > parsed.data.len() {
                return None;
            }
            copy_match(&mut parsed.data, length, distance);
            parsed.tokens.push(Token::Match {
                length: length as u16,
                distance: distance as u16,
            });
        }
        count += 1;
        if parsed.data.len() > max_output {
            return None;
        }
    }
}

/// Parses one complete deflate stream at the start of `input`, accepting it
/// only if `encode` would reproduce the consumed bytes exactly.
pub(super) fn parse(input: &[u8], max_output: usize) -> Option<Parsed> {
    let mut reader = BitReader {
        bytes: input,
        position: 0,
    };
    let mut parsed = Parsed {
        data: Vec::new(),
        blocks: Vec::new(),
        tokens: Vec::new(),
        consumed: 0,
    };
    loop {
        let last = reader.bit()? == 1;
        let kind_bits = reader.bits(2)?;
        let (kind, tokens) = match kind_bits {
            0 => {
                reader.align_zero()?;
                let length = reader.bits(16)? as u16;
                if reader.bits(16)? as u16 != !length {
                    return None;
                }
                let start = reader.position / 8;
                let end = start.checked_add(usize::from(length))?;
                parsed.data.extend_from_slice(input.get(start..end)?);
                reader.position = end * 8;
                (Kind::Stored(length), 0)
            }
            1 => {
                let (literal, distance) = fixed_lengths();
                let tokens = read_tokens(
                    &mut reader,
                    (&Decoder::new(&literal)?, &Decoder::new(&distance)?),
                    &mut parsed,
                    max_output,
                )?;
                (Kind::Fixed, tokens)
            }
            2 => {
                let start = reader.position;
                let (literal, distance) = read_dynamic_lengths(&mut reader)?;
                let header_bits = reader.position - start;
                let mut header = vec![0u8; header_bits.div_ceil(8)];
                for bit in 0..header_bits {
                    header[bit / 8] |=
                        ((input[(start + bit) / 8] >> ((start + bit) % 8)) & 1) << (bit % 8);
                }
                let tokens = read_tokens(
                    &mut reader,
                    (&Decoder::new(&literal)?, &Decoder::new(&distance)?),
                    &mut parsed,
                    max_output,
                )?;
                (
                    Kind::Dynamic {
                        header,
                        header_bits,
                    },
                    tokens,
                )
            }
            _ => return None,
        };
        if parsed.data.len() > max_output {
            return None;
        }
        parsed.blocks.push(Block { kind, last, tokens });
        if last {
            break;
        }
    }
    reader.align_zero()?;
    parsed.consumed = reader.position / 8;
    // Rejects anything the encoder would spell differently.
    (encode(&parsed.blocks, &parsed.tokens, &parsed.data)? == input[..parsed.consumed])
        .then_some(parsed)
}

struct BitWriter {
    bytes: Vec<u8>,
    position: usize,
}

impl BitWriter {
    fn bit(&mut self, bit: u32) {
        if self.position.is_multiple_of(8) {
            self.bytes.push(0);
        }
        let last = self.bytes.len() - 1;
        self.bytes[last] |= (bit as u8) << (self.position % 8);
        self.position += 1;
    }

    fn bits(&mut self, value: u32, count: u8) {
        for shift in 0..count {
            self.bit((value >> shift) & 1);
        }
    }

    /// Huffman codes are packed most significant bit first.
    fn code(&mut self, code: u32, length: u8) {
        for shift in (0..length).rev() {
            self.bit((code >> shift) & 1);
        }
    }

    fn align(&mut self) {
        self.position = self.bytes.len() * 8;
    }
}

/// Canonical code values for the given lengths.
fn canonical_codes(lengths: &[u8]) -> Vec<u32> {
    let mut count = [0u32; MAX_BITS + 2];
    for &length in lengths {
        count[usize::from(length)] += 1;
    }
    count[0] = 0;
    let mut next = [0u32; MAX_BITS + 2];
    let mut code = 0;
    for bits in 1..=MAX_BITS {
        code = (code + count[bits - 1]) << 1;
        next[bits] = code;
    }
    lengths
        .iter()
        .map(|&length| {
            let value = next[usize::from(length)];
            next[usize::from(length)] += 1;
            value
        })
        .collect()
}

fn bucket(table: &[u16], value: usize) -> usize {
    table.partition_point(|&base| usize::from(base) <= value) - 1
}

fn write_token(
    writer: &mut BitWriter,
    token: Token,
    literal: (&[u8], &[u32]),
    distance: (&[u8], &[u32]),
) -> Option<()> {
    match token {
        Token::Literal(byte) => {
            writer.code(literal.1[usize::from(byte)], literal.0[usize::from(byte)]);
        }
        Token::Match {
            length,
            distance: back,
        } => {
            let length = usize::from(length);
            if !(MIN_MATCH..=MAX_MATCH).contains(&length) {
                return None;
            }
            let index = if length == MAX_MATCH {
                28
            } else {
                bucket(&LENGTH_BASE[..28], length)
            };
            let symbol = 257 + index;
            writer.code(literal.1[symbol], literal.0[symbol]);
            writer.bits(
                (length - usize::from(LENGTH_BASE[index])) as u32,
                LENGTH_EXTRA[index],
            );
            let slot = bucket(&DISTANCE_BASE, usize::from(back));
            writer.code(distance.1[slot], distance.0[slot]);
            writer.bits(
                (usize::from(back) - usize::from(DISTANCE_BASE[slot])) as u32,
                DISTANCE_EXTRA[slot],
            );
        }
    }
    Some(())
}

/// Builds a deflate stream one block at a time, so a caller can generate
/// each block's tokens just before writing them.
pub(super) struct StreamWriter {
    writer: BitWriter,
    data_at: usize,
}

impl StreamWriter {
    pub(super) fn new() -> Self {
        Self {
            writer: BitWriter {
                bytes: Vec::new(),
                position: 0,
            },
            data_at: 0,
        }
    }

    /// Writes `block` with its `tokens`; stored blocks take their bytes from
    /// `data` and need no tokens. `None` if a token has no symbol under the
    /// block's code or the data runs out.
    pub(super) fn write_block(
        &mut self,
        block: &Block,
        tokens: &[Token],
        data: &[u8],
    ) -> Option<()> {
        let writer = &mut self.writer;
        writer.bit(u32::from(block.last));
        let lengths = match &block.kind {
            Kind::Stored(length) => {
                writer.bits(0, 2);
                writer.align();
                writer.bits(u32::from(*length), 16);
                writer.bits(u32::from(!*length), 16);
                let end = self.data_at + usize::from(*length);
                writer.bytes.extend_from_slice(data.get(self.data_at..end)?);
                writer.position = writer.bytes.len() * 8;
                self.data_at = end;
                return Some(());
            }
            Kind::Fixed => {
                writer.bits(1, 2);
                let (literal, distance) = fixed_lengths();
                (literal.to_vec(), distance.to_vec())
            }
            Kind::Dynamic {
                header,
                header_bits,
            } => {
                writer.bits(2, 2);
                let mut reader = BitReader {
                    bytes: header,
                    position: 0,
                };
                let lengths = read_dynamic_lengths(&mut reader)?;
                if reader.position != *header_bits {
                    return None;
                }
                for bit in 0..*header_bits {
                    writer.bit(u32::from((header[bit / 8] >> (bit % 8)) & 1));
                }
                lengths
            }
        };
        let (literal_codes, distance_codes) =
            (canonical_codes(&lengths.0), canonical_codes(&lengths.1));
        for &token in tokens {
            self.data_at += token_length(token);
            write_token(
                writer,
                token,
                (&lengths.0, &literal_codes),
                (&lengths.1, &distance_codes),
            )?;
        }
        writer.code(literal_codes[END_OF_BLOCK], lengths.0[END_OF_BLOCK]);
        Some(())
    }

    /// The finished stream, if its blocks covered exactly `data_len` bytes.
    pub(super) fn finish(self, data_len: usize) -> Option<Vec<u8>> {
        (self.data_at == data_len).then_some(self.writer.bytes)
    }
}

pub(super) fn token_length(token: Token) -> usize {
    match token {
        Token::Literal(_) => 1,
        Token::Match { length, .. } => usize::from(length),
    }
}

/// Writes the stream described by `blocks`, `tokens` and `data`. Returns
/// `None` if a token has no symbol under its block's code.
pub(super) fn encode(blocks: &[Block], tokens: &[Token], data: &[u8]) -> Option<Vec<u8>> {
    let mut stream = StreamWriter::new();
    let mut token_at = 0;
    for block in blocks {
        let count = if matches!(block.kind, Kind::Stored(_)) {
            0
        } else {
            block.tokens
        };
        stream.write_block(block, tokens.get(token_at..token_at + count)?, data)?;
        token_at += count;
    }
    (token_at == tokens.len()).then(|| stream.finish(data.len()))?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> &'static [u8] {
        match name {
            "stored" => include_bytes!("fixtures/stored.deflate"),
            "fixed" => include_bytes!("fixtures/fixed.deflate"),
            "l1" => include_bytes!("fixtures/l1.deflate"),
            "l6" => include_bytes!("fixtures/l6.deflate"),
            _ => include_bytes!("fixtures/l9.deflate"),
        }
    }

    #[test]
    fn real_zlib_streams_parse_and_reencode_exactly() {
        let mix: &[u8] = include_bytes!("fixtures/mix.raw");
        let expected: [(&str, &[u8]); 5] = [
            ("stored", include_bytes!("fixtures/stored.raw")),
            ("fixed", include_bytes!("fixtures/fixed.raw")),
            ("l1", mix),
            ("l6", mix),
            ("l9", mix),
        ];
        for (name, raw) in expected {
            let stream = fixture(name);
            let parsed = parse(stream, 1 << 20).unwrap_or_else(|| panic!("{name} rejected"));
            assert_eq!(parsed.data, raw, "{name}");
            assert_eq!(parsed.consumed, stream.len(), "{name}");
            let rebuilt = encode(&parsed.blocks, &parsed.tokens, &parsed.data).unwrap();
            assert_eq!(rebuilt, stream, "{name}");
        }
    }

    #[test]
    fn block_kinds_match_the_encoder_level() {
        let stored = parse(fixture("stored"), 1 << 20).unwrap();
        assert!(matches!(stored.blocks[0].kind, Kind::Stored(3000)));
        let fixed = parse(fixture("fixed"), 1 << 20).unwrap();
        assert!(matches!(fixed.blocks[0].kind, Kind::Fixed));
        let dynamic = parse(fixture("l6"), 1 << 20).unwrap();
        assert!(dynamic
            .blocks
            .iter()
            .any(|b| matches!(b.kind, Kind::Dynamic { .. })));
        assert!(dynamic.blocks.last().unwrap().last);
    }

    #[test]
    fn truncated_corrupt_and_oversized_streams_are_rejected() {
        let stream = fixture("l6");
        for cut in [0, 1, 10, stream.len() / 2, stream.len() - 1] {
            assert!(parse(&stream[..cut], 1 << 20).is_none(), "cut {cut}");
        }
        assert!(parse(stream, 1000).is_none());
        let mut flipped = stream.to_vec();
        flipped[stream.len() / 2] ^= 0x10;
        if let Some(parsed) = parse(&flipped, 1 << 20) {
            assert_eq!(parsed.consumed, flipped.len().min(parsed.consumed));
            assert_eq!(
                encode(&parsed.blocks, &parsed.tokens, &parsed.data).unwrap(),
                flipped[..parsed.consumed]
            );
        }
        assert!(parse(&[0x07], 16).is_none(), "reserved block type");
        let mut padded = fixture("fixed").to_vec();
        *padded.last_mut().unwrap() |= 0x80;
        assert!(parse(&padded, 1 << 20).is_none(), "non-zero padding");
    }
}
