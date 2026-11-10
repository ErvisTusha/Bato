//! Everything needed, besides the inflated bytes, to rebuild a deflate stream
//! exactly: the imitated encoder, the block layout, the dynamic headers and
//! the tokens the predictor gets wrong.
//! Provenance: original design; token prediction in `predict.rs`.

use super::{
    predict::{Params, Predictor, LEVELS},
    token_length, Block, Kind, Parsed, StreamWriter, Token,
};
use crate::core::{cm::varint, types::BatoError};

/// A token the predictor did not foresee, at its index in the token list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Miss {
    pub(super) index: usize,
    pub(super) token: Token,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Recipe {
    /// Index into `LEVELS`.
    pub(super) level: u8,
    pub(super) wide_hash: bool,
    pub(super) blocks: Vec<Block>,
    pub(super) misses: Vec<Miss>,
}

/// Most tokens a recipe may declare for one block; real encoders emit at
/// most a few tens of thousands, and this bounds the decoder's token buffer.
pub(super) const MAX_BLOCK_TOKENS: usize = 1 << 20;
const WIDE_HASH_BITS: u32 = 16;
const KIND_STORED: u8 = 0;
const KIND_FIXED: u8 = 1;
const KIND_DYNAMIC: u8 = 2;
const LAST_FLAG: u8 = 4;
const LITERAL_FLAG: u8 = 0;
const MATCH_FLAG: u8 = 1;

impl Recipe {
    fn params(&self) -> Option<Params> {
        let level = *LEVELS.get(usize::from(self.level))?;
        Some(if self.wide_hash {
            level.with_hash_bits(WIDE_HASH_BITS)
        } else {
            level
        })
    }
}

/// Misses of one parameter set over a parsed stream.
fn misses_for(parsed: &Parsed, params: Params) -> Vec<Miss> {
    let mut predictor = Predictor::new(&parsed.data, params);
    let mut misses = Vec::new();
    let (mut at, mut index, mut next_token) = (0, 0, 0);
    for block in &parsed.blocks {
        if let Kind::Stored(length) = block.kind {
            predictor.skip(at, usize::from(length));
            at += usize::from(length);
            continue;
        }
        for &token in &parsed.tokens[next_token..next_token + block.tokens] {
            if predictor.predict(at) != token {
                misses.push(Miss { index, token });
            }
            predictor.advance(at, token);
            at += token_length(token);
            index += 1;
        }
        next_token += block.tokens;
    }
    misses
}

fn recipe_for(parsed: &Parsed, level: usize, wide_hash: bool) -> Recipe {
    let params = if wide_hash {
        LEVELS[level].with_hash_bits(WIDE_HASH_BITS)
    } else {
        LEVELS[level]
    };
    Recipe {
        level: level as u8,
        wide_hash,
        blocks: parsed.blocks.clone(),
        misses: misses_for(parsed, params),
    }
}

/// Choose the predictor requiring the fewest *serialized bytes*, not merely
/// the fewest misses. A match token's length and distance cost more than a
/// literal, and large miss-position gaps need longer varints.
pub(super) fn analyze(parsed: &Parsed) -> Recipe {
    let mut best = recipe_for(parsed, 0, false);
    let mut best_bytes = serialized_len(&best);
    for level in 1..LEVELS.len() {
        let candidate = recipe_for(parsed, level, false);
        let bytes = serialized_len(&candidate);
        if bytes < best_bytes {
            best = candidate;
            best_bytes = bytes;
        }
    }
    // The decoder already supports 16-bit hash tables; test the alternate
    // geometry at the winning level instead of leaving this candidate unused.
    // This adds one search, not another public compression mode.
    let wide = recipe_for(parsed, usize::from(best.level), true);
    if serialized_len(&wide) < best_bytes {
        best = wide;
    }
    best
}

/// Rebuilds the deflate stream from the inflated bytes and the recipe,
/// generating the tokens of one block at a time.
pub(super) fn synthesize(data: &[u8], recipe: &Recipe) -> Option<Vec<u8>> {
    let mut predictor = Predictor::new(data, recipe.params()?);
    let mut stream = StreamWriter::new();
    let mut misses = recipe.misses.iter().peekable();
    let (mut at, mut index) = (0, 0);
    let mut tokens = Vec::new();
    for block in &recipe.blocks {
        tokens.clear();
        if let Kind::Stored(length) = block.kind {
            predictor.skip(at, usize::from(length));
            at += usize::from(length);
        } else {
            if block.tokens > MAX_BLOCK_TOKENS {
                return None;
            }
            for _ in 0..block.tokens {
                if at >= data.len() {
                    return None;
                }
                let predicted = predictor.predict(at);
                let token = match misses.next_if(|m| m.index == index) {
                    Some(miss) => miss.token,
                    None => predicted,
                };
                predictor.advance(at, token);
                at += token_length(token);
                index += 1;
                tokens.push(token);
            }
        }
        stream.write_block(block, &tokens, data)?;
    }
    if misses.next().is_some() {
        return None;
    }
    stream.finish(data.len())
}

fn write_token(out: &mut Vec<u8>, token: Token) {
    match token {
        Token::Literal(byte) => out.extend([LITERAL_FLAG, byte]),
        Token::Match { length, distance } => {
            out.push(MATCH_FLAG);
            varint::push(out, usize::from(length));
            varint::push(out, usize::from(distance));
        }
    }
}

/// Exact side-information size without allocating a temporary serialized
/// recipe for each of the nine predictor candidates.
fn serialized_len(recipe: &Recipe) -> usize {
    let mut length = 1 + varint::len(recipe.blocks.len() as u64);
    for block in &recipe.blocks {
        length += 1; // block kind and last-block flag
        match &block.kind {
            Kind::Stored(bytes) => {
                length += varint::len(u64::from(*bytes));
            }
            Kind::Fixed => {
                length += varint::len(block.tokens as u64);
            }
            Kind::Dynamic {
                header,
                header_bits,
            } => {
                length += varint::len(block.tokens as u64)
                    + varint::len(*header_bits as u64)
                    + header.len();
            }
        }
    }
    length += varint::len(recipe.misses.len() as u64);
    let mut previous = 0;
    for miss in &recipe.misses {
        length += varint::len((miss.index - previous) as u64);
        length += match miss.token {
            Token::Literal(_) => 2,
            Token::Match { length, distance } => {
                1 + varint::len(u64::from(length)) + varint::len(u64::from(distance))
            }
        };
        previous = miss.index;
    }
    length
}

/// Plain serialisation, used to size the side information and to store it.
pub(super) fn serialize(recipe: &Recipe) -> Vec<u8> {
    let mut out = vec![recipe.level | u8::from(recipe.wide_hash) << 7];
    varint::push(&mut out, recipe.blocks.len());
    for block in &recipe.blocks {
        let last = if block.last { LAST_FLAG } else { 0 };
        match &block.kind {
            Kind::Stored(length) => {
                out.push(KIND_STORED | last);
                varint::push(&mut out, usize::from(*length));
            }
            Kind::Fixed => {
                out.push(KIND_FIXED | last);
                varint::push(&mut out, block.tokens);
            }
            Kind::Dynamic {
                header,
                header_bits,
            } => {
                out.push(KIND_DYNAMIC | last);
                varint::push(&mut out, block.tokens);
                varint::push(&mut out, *header_bits);
                out.extend_from_slice(header);
            }
        }
    }
    varint::push(&mut out, recipe.misses.len());
    let mut previous = 0;
    for miss in &recipe.misses {
        varint::push(&mut out, miss.index - previous);
        previous = miss.index;
        write_token(&mut out, miss.token);
    }
    out
}

fn read_token(input: &[u8], at: &mut usize) -> Result<Token, BatoError> {
    let flag = *input.get(*at).ok_or_else(corrupt)?;
    *at += 1;
    match flag {
        LITERAL_FLAG => {
            let byte = *input.get(*at).ok_or_else(corrupt)?;
            *at += 1;
            Ok(Token::Literal(byte))
        }
        MATCH_FLAG => {
            let length = u16::try_from(varint::take(input, at)?).map_err(|_| corrupt())?;
            let distance = u16::try_from(varint::take(input, at)?).map_err(|_| corrupt())?;
            Ok(Token::Match { length, distance })
        }
        _ => Err(corrupt()),
    }
}

fn corrupt() -> BatoError {
    BatoError::CorruptStream("invalid deflate recipe".into())
}

/// Inverse of `serialize`; every count is checked against the input left.
pub(super) fn deserialize(input: &[u8]) -> Result<Recipe, BatoError> {
    let first = *input.first().ok_or_else(corrupt)?;
    let mut at = 1;
    let recipe_blocks = varint::take(input, &mut at)?;
    if recipe_blocks > input.len() {
        return Err(corrupt());
    }
    let mut blocks = Vec::with_capacity(recipe_blocks);
    for _ in 0..recipe_blocks {
        let tag = *input.get(at).ok_or_else(corrupt)?;
        at += 1;
        let last = tag & LAST_FLAG != 0;
        let (kind, tokens) = match tag & !LAST_FLAG {
            KIND_STORED => {
                let length = u16::try_from(varint::take(input, &mut at)?).map_err(|_| corrupt())?;
                (Kind::Stored(length), 0)
            }
            KIND_FIXED => (Kind::Fixed, varint::take(input, &mut at)?),
            KIND_DYNAMIC => {
                let tokens = varint::take(input, &mut at)?;
                let header_bits = varint::take(input, &mut at)?;
                let end = at
                    .checked_add(header_bits.div_ceil(8))
                    .ok_or_else(corrupt)?;
                let header = input.get(at..end).ok_or_else(corrupt)?.to_vec();
                at = end;
                (
                    Kind::Dynamic {
                        header,
                        header_bits,
                    },
                    tokens,
                )
            }
            _ => return Err(corrupt()),
        };
        blocks.push(Block { kind, last, tokens });
    }
    let count = varint::take(input, &mut at)?;
    if count > input.len() {
        return Err(corrupt());
    }
    let mut misses = Vec::with_capacity(count);
    let mut index = 0usize;
    for _ in 0..count {
        index = index
            .checked_add(varint::take(input, &mut at)?)
            .ok_or_else(corrupt)?;
        misses.push(Miss {
            index,
            token: read_token(input, &mut at)?,
        });
    }
    if at != input.len() {
        return Err(corrupt());
    }
    Ok(Recipe {
        level: first & 0x7f,
        wide_hash: first & 0x80 != 0,
        blocks,
        misses,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cm::deflate::parse;

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
    fn recipes_rebuild_real_streams_exactly_and_survive_serialisation() {
        for name in ["stored", "fixed", "l1", "l6", "l9"] {
            let stream = fixture(name);
            let parsed = parse(stream, 1 << 20).unwrap();
            let recipe = analyze(&parsed);
            assert_eq!(synthesize(&parsed.data, &recipe).unwrap(), stream, "{name}");
            let bytes = serialize(&recipe);
            assert_eq!(serialized_len(&recipe), bytes.len(), "{name}");
            assert_eq!(deserialize(&bytes).unwrap(), recipe, "{name}");
            println!(
                "{name}: stream {} recipe {} misses {} level {}",
                stream.len(),
                bytes.len(),
                recipe.misses.len(),
                recipe.level + 1
            );
        }
    }

    #[test]
    fn wide_hash_recipes_rebuild_byte_identically() {
        for name in ["fixed", "l1", "l6", "l9"] {
            let stream = fixture(name);
            let parsed = parse(stream, 1 << 20).unwrap();
            let wide = recipe_for(&parsed, 5, true);
            assert!(wide.wide_hash);
            assert_eq!(synthesize(&parsed.data, &wide).unwrap(), stream);
            let serialized = serialize(&wide);
            assert_eq!(serialized_len(&wide), serialized.len());
            assert_eq!(deserialize(&serialized).unwrap(), wide);
        }
    }

    #[test]
    fn corrupt_recipes_are_rejected_or_fail_synthesis() {
        let parsed = parse(fixture("l6"), 1 << 20).unwrap();
        let bytes = serialize(&analyze(&parsed));
        for cut in [0, 1, 2, bytes.len() / 2, bytes.len() - 1] {
            assert!(deserialize(&bytes[..cut]).is_err(), "cut {cut}");
        }
        let mut recipe = analyze(&parsed);
        recipe.level = 99;
        assert!(synthesize(&parsed.data, &recipe).is_none());
        let mut recipe = analyze(&parsed);
        recipe.misses.push(Miss {
            index: usize::MAX,
            token: Token::Literal(0),
        });
        assert!(synthesize(&parsed.data, &recipe).is_none());
    }
}
