//! Segment transform that swaps each deflate stream found in a segment for
//! its inflated bytes plus a recipe that rebuilds the stream bit for bit.
//! Provenance: original design; stream parsing and token prediction are in
//! the sibling modules (RFC 1951 and public LZ77 matcher descriptions).
//!
//! Streams are found by trying to parse at byte offsets whose first bits form
//! a plausible dynamic block header, never by names or signatures. A stream
//! is kept only if it re-encodes exactly and its recipe is small.
//!
//! Transformed layout: a variable integer meta length, the meta, then the
//! body, which is the original segment with every stream replaced by its
//! inflated bytes, then `pad` zero bytes. Meta: `pad` (0 or 1), stream count,
//! then per stream its body offset, compressed length, inflated length and
//! recipe length, then the recipe. The padding keeps the transformed length
//! from being a multiple of the chunk size, so that a short final chunk
//! always marks the end of a transformed segment.

use super::{
    parse,
    recipe::{analyze, deserialize, serialize, synthesize},
};
use crate::core::{cm::varint, memory::try_capacity, types::BatoError};

/// Smallest deflate stream worth recompressing, in compressed bytes.
const MIN_STREAM_BYTES: usize = 256;
/// A recipe may take at most this percentage of the stream it replaces. The
/// recipes are coded with the model, so a recipe far larger than its stream
/// can still pay (7-Zip's streams, 40% raw, halve the output); the segment
/// keeps the transform only when the coded size beats the alternatives.
const MAX_RECIPE_PERCENT: usize = 200;
/// Streams must cover at least this many bytes in total to transform.
const MIN_TOTAL_BYTES: usize = 1024;

struct Found {
    start: usize,
    consumed: usize,
    inflated: Vec<u8>,
    recipe: Vec<u8>,
}

/// A cheap prefilter for valid DEFLATE beginnings. Stored-block length
/// complements are nearly free to check; fixed-Huffman scans are restricted
/// to the segment start or a preceding zlib header to avoid parsing random
/// bytes at roughly one in eight offsets.
fn plausible_start(raw: &[u8], at: usize) -> bool {
    let bytes = &raw[at..];
    let Some(&first) = bytes.first() else {
        return false;
    };
    match (first >> 1) & 3 {
        0 => {
            // Stored blocks ignore the five padding bits in their first
            // byte; do not reject valid nonzero padding.
            if bytes.len() < 5 {
                return false;
            }
            let length = u16::from_le_bytes([bytes[1], bytes[2]]);
            let complement = u16::from_le_bytes([bytes[3], bytes[4]]);
            length ^ complement == u16::MAX
        }
        1 => {
            // Fixed-Huffman blocks may be followed by another block. Scan
            // only the segment start or a plausible preceding zlib header.
            if at == 0 {
                return true;
            }
            if at < 2 {
                return false;
            }
            let cmf = raw[at - 2];
            let flg = raw[at - 1];
            cmf & 15 == 8
                && cmf >> 4 <= 7
                && flg & 0x20 == 0
                && (u16::from(cmf) * 256 + u16::from(flg)) % 31 == 0
        }
        2 => bytes.len() >= 2 && (first >> 3) <= 29 && (bytes[1] & 31) <= 29,
        _ => false,
    }
}

fn corrupt() -> BatoError {
    BatoError::CorruptStream("invalid deflate container".into())
}

fn scan(raw: &[u8], limit: usize) -> Vec<Found> {
    let mut found: Vec<Found> = Vec::new();
    let mut extra = 0usize;
    let mut at = 0;
    while at + MIN_STREAM_BYTES <= raw.len() {
        if !plausible_start(raw, at) {
            at += 1;
            continue;
        }
        let room = limit.saturating_sub(raw.len() + extra) + (raw.len() - at);
        let Some(parsed) = parse(&raw[at..], room) else {
            at += 1;
            continue;
        };
        let recipe = serialize(&analyze(&parsed));
        let worthwhile = parsed.consumed >= MIN_STREAM_BYTES
            && parsed.data.len() >= parsed.consumed
            && recipe.len() * 100 <= parsed.consumed * MAX_RECIPE_PERCENT;
        if !worthwhile {
            at += 1;
            continue;
        }
        extra += parsed.data.len() - parsed.consumed;
        let consumed = parsed.consumed;
        found.push(Found {
            start: at,
            consumed,
            inflated: parsed.data,
            recipe,
        });
        at += consumed;
    }
    found
}

/// Rewrites `raw` with its recompressible deflate streams inflated, or
/// `None` if too little of it is deflate or the result would exceed `limit`.
pub(in crate::core::cm) fn forward(raw: &[u8], limit: usize, chunk: usize) -> Option<Vec<u8>> {
    let found = scan(raw, limit);
    if found.iter().map(|f| f.consumed).sum::<usize>() < MIN_TOTAL_BYTES {
        return None;
    }
    let mut meta = Vec::new();
    varint::push(&mut meta, 0);
    varint::push(&mut meta, found.len());
    let mut body = Vec::new();
    let mut previous = 0;
    for stream in &found {
        body.extend_from_slice(&raw[previous..stream.start]);
        varint::push(&mut meta, body.len());
        varint::push(&mut meta, stream.consumed);
        varint::push(&mut meta, stream.inflated.len());
        varint::push(&mut meta, stream.recipe.len());
        meta.extend_from_slice(&stream.recipe);
        body.extend_from_slice(&stream.inflated);
        previous = stream.start + stream.consumed;
    }
    body.extend_from_slice(&raw[previous..]);
    let mut out = Vec::with_capacity(meta.len() + body.len() + 5);
    varint::push(&mut out, meta.len());
    out.extend_from_slice(&meta);
    out.extend_from_slice(&body);
    if out.len() % chunk == 0 {
        // The pad count is the meta's first byte, so the meta keeps its length.
        let at = out.len() - body.len() - meta.len();
        out[at] = 1;
        out.push(0);
    }
    (out.len() <= limit).then_some(out)
}

struct Entry<'a> {
    offset: usize,
    consumed: usize,
    inflated: usize,
    recipe: &'a [u8],
}

/// The meta's padding byte count and its stream entries.
fn read_entries(meta: &[u8]) -> Result<(usize, Vec<Entry<'_>>), BatoError> {
    let mut at = 0;
    let pad = varint::take(meta, &mut at)?;
    if pad > 1 {
        return Err(corrupt());
    }
    let count = varint::take(meta, &mut at)?;
    if count > meta.len() {
        return Err(corrupt());
    }
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let offset = varint::take(meta, &mut at)?;
        let consumed = varint::take(meta, &mut at)?;
        let inflated = varint::take(meta, &mut at)?;
        let length = varint::take(meta, &mut at)?;
        let end = at.checked_add(length).ok_or_else(corrupt)?;
        let recipe = meta.get(at..end).ok_or_else(corrupt)?;
        at = end;
        entries.push(Entry {
            offset,
            consumed,
            inflated,
            recipe,
        });
    }
    if at != meta.len() {
        return Err(corrupt());
    }
    Ok((pad, entries))
}

/// Restores the original segment from `transformed`, never producing more
/// than `max_output` bytes.
pub(in crate::core::cm) fn inverse(
    transformed: &[u8],
    max_output: usize,
) -> Result<Vec<u8>, BatoError> {
    let mut at = 0;
    let meta_length = varint::take(transformed, &mut at)?;
    let body_start = at.checked_add(meta_length).ok_or_else(corrupt)?;
    let meta = transformed.get(at..body_start).ok_or_else(corrupt)?;
    let (pad, entries) = read_entries(meta)?;
    let body_end = transformed.len().checked_sub(pad).ok_or_else(corrupt)?;
    let body = transformed.get(body_start..body_end).ok_or_else(corrupt)?;
    let inflated_total = entries
        .iter()
        .try_fold(0usize, |sum, e| sum.checked_add(e.inflated))
        .ok_or_else(corrupt)?;
    let consumed_total = entries
        .iter()
        .try_fold(0usize, |sum, e| sum.checked_add(e.consumed))
        .ok_or_else(corrupt)?;
    let size = body
        .len()
        .checked_sub(inflated_total)
        .and_then(|rest| rest.checked_add(consumed_total))
        .filter(|&size| size <= max_output)
        .ok_or_else(corrupt)?;
    let mut out = try_capacity(size)?;
    let mut previous = 0;
    for entry in &entries {
        let end = entry
            .offset
            .checked_add(entry.inflated)
            .ok_or_else(corrupt)?;
        out.extend_from_slice(body.get(previous..entry.offset).ok_or_else(corrupt)?);
        let inflated = body.get(entry.offset..end).ok_or_else(corrupt)?;
        let stream = synthesize(inflated, &deserialize(entry.recipe)?).ok_or_else(corrupt)?;
        if stream.len() != entry.consumed {
            return Err(corrupt());
        }
        out.extend_from_slice(&stream);
        previous = end;
    }
    out.extend_from_slice(&body[previous..]);
    if out.len() != size {
        return Err(corrupt());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn streams() -> (&'static [u8], &'static [u8]) {
        (
            include_bytes!("fixtures/l6.deflate"),
            include_bytes!("fixtures/l9.deflate"),
        )
    }

    fn sample() -> Vec<u8> {
        let (first, second) = streams();
        let mut raw = b"PK header bytes and some text before the first stream\n".repeat(3);
        raw.extend_from_slice(first);
        raw.extend_from_slice(b"\x00\x01 middle section between streams \xff\xfe");
        raw.extend_from_slice(second);
        raw.extend_from_slice(b"trailing descriptor");
        raw
    }

    #[test]
    fn stored_blocks_and_zlib_wrapped_fixed_starts_are_detected_cheaply() {
        let stored = include_bytes!("fixtures/stored.deflate");
        assert!(plausible_start(stored, 0));
        let mut embedded = b"unrelated bytes preceding a stored stream".to_vec();
        embedded.extend_from_slice(stored);
        assert!(plausible_start(&embedded, embedded.len() - stored.len()));
        // A stored block grows by its own DEFLATE header. Recognition does
        // not imply that inflating it is a profitable representation.
        assert!(scan(&embedded, 1 << 22).is_empty());
        assert!(forward(&embedded, 1 << 22, 4096).is_none());

        assert!(plausible_start(&[0xf8, 0x01, 0x00, 0xfe, 0xff], 0));
        assert!(plausible_start(&[0x02, 0x00], 0));
        assert!(plausible_start(&[0x03, 0x00], 0));
        assert!(plausible_start(&[0x78, 0x9c, 0x03, 0x00], 2));
        assert!(!plausible_start(&[0xaa, 0xbb, 0x03, 0x00], 2));
        assert!(!plausible_start(&[0x00, 0x01, 0x00, 0x00, 0x00], 0));
    }

    #[test]
    fn embedded_streams_are_inflated_and_restored_exactly() {
        let raw = sample();
        let transformed = forward(&raw, 1 << 22, 4096).unwrap();
        assert!(transformed.len() > raw.len(), "inflated data is larger");
        assert_eq!(inverse(&transformed, raw.len()).unwrap(), raw);
        assert!(inverse(&transformed, raw.len() - 1).is_err());
    }

    #[test]
    fn transformed_length_is_never_a_multiple_of_the_chunk_size() {
        // A short last chunk marks the end of a transformed segment, so the
        // length must not be a whole number of chunks; the pad byte ensures it.
        for chunk in [16, 17, 64] {
            for filler in 0..chunk {
                let mut raw = sample();
                raw.extend(std::iter::repeat_n(b'.', filler));
                let transformed = forward(&raw, 1 << 22, chunk).unwrap();
                assert_ne!(
                    transformed.len() % chunk,
                    0,
                    "chunk {chunk} filler {filler}"
                );
                assert_eq!(inverse(&transformed, raw.len()).unwrap(), raw);
            }
        }
    }

    #[test]
    fn data_without_streams_or_over_the_limit_is_left_alone() {
        let plain: Vec<u8> = (0..5000u32).map(|i| (i * 7 % 251) as u8).collect();
        assert!(forward(&plain, 1 << 22, 4096).is_none());
        assert!(forward(&sample(), 100, 4096).is_none());
        assert!(forward(&[], 1 << 22, 4096).is_none());
    }

    #[test]
    fn damaged_containers_fail_cleanly_without_panicking() {
        let transformed = forward(&sample(), 1 << 22, 4096).unwrap();
        for cut in [0, 1, 5, 40, transformed.len() / 2, transformed.len() - 1] {
            let _ = inverse(&transformed[..cut], 1 << 22);
        }
        for position in (0..transformed.len()).step_by(37) {
            let mut damaged = transformed.clone();
            damaged[position] ^= 0x5a;
            if let Ok(out) = inverse(&damaged, 1 << 22) {
                // A flip inside inflated bytes still yields a valid stream.
                assert!(out.len() <= 1 << 22);
            }
        }
    }
}
