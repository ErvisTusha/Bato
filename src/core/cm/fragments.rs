//! Sparse, content-selected anchors for exact long byte copies.
//! Provenance: LZ77's original distance/length representation (Ziv and
//! Lempel, 1977) and content-selected fingerprints (Schleimer et al., 2003,
//! Winnowing). Encoding and bounded candidate selection are original.

use super::{lz, varint};
use crate::core::{memory::try_filled, types::BatoError};

const MIN_COPY: usize = 128;
const MIN_INPUT: usize = 8 * 1024;
const ANCHOR: usize = 8;
const MAX_BUCKETS: usize = 1 << 16;

fn fingerprint(bytes: &[u8]) -> u64 {
    let value = u64::from_le_bytes(bytes[..ANCHOR].try_into().unwrap());
    let mixed = (value ^ (value >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    mixed ^ (mixed >> 33)
}

fn literal(tape: &mut Vec<u8>, bytes: &[u8]) {
    if !bytes.is_empty() {
        varint::push(tape, bytes.len() << 1);
        tape.extend_from_slice(bytes);
    }
}

/// Builds the ordinary LZ command grammar using only large exact matches.
/// Only a substantial reduction in bytes warrants an extra model trial.
pub(super) fn commands(raw: &[u8]) -> Result<Option<Vec<u8>>, BatoError> {
    if raw.len() < MIN_INPUT {
        return Ok(None);
    }
    let buckets = (raw.len() / 8).next_power_of_two().min(MAX_BUCKETS);
    let mut anchors = try_filled(buckets, [0u32; 2])?;
    let cap = raw.len().saturating_sub(raw.len() / 3);
    let mut tape = Vec::new();
    let mut end = 0;
    let mut at = 0;
    while at + MIN_COPY <= raw.len() {
        let hash = fingerprint(&raw[at..]);
        if hash & 15 != 0 {
            at += 1;
            continue;
        }
        let bucket = &mut anchors[((hash >> 4) as usize) & (buckets - 1)];
        let mut best = (0, 0);
        for previous in bucket.iter().copied().filter(|&p| p != 0) {
            let previous = previous as usize - 1;
            let length = lz::matched(raw, previous, at);
            if length >= MIN_COPY && length > best.0 {
                best = (length, at - previous);
            }
        }
        *bucket = [(at + 1) as u32, bucket[0]];
        let (length, distance) = best;
        if length < MIN_COPY {
            at += 1;
            continue;
        }
        let mut start = at;
        while start > end.max(distance) && raw[start - 1] == raw[start - 1 - distance] {
            start -= 1;
        }
        if tape.len() + start - end >= cap {
            return Ok(None);
        }
        literal(&mut tape, &raw[end..start]);
        end = at + length;
        varint::push(&mut tape, ((end - start - lz::MIN_MATCH) << 1) | 1);
        varint::push(&mut tape, distance);
        at = end;
    }
    if tape.len() + raw.len() - end >= cap {
        return Ok(None);
    }
    literal(&mut tape, &raw[end..]);
    Ok((tape.len() < cap).then_some(tape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_shifted_fragments_and_uses_exact_copies() {
        let source: Vec<u8> = (0u32..2400)
            .flat_map(|n| n.wrapping_mul(0xabcd_1239).to_le_bytes())
            .collect();
        let mut bytes = source.clone();
        for gap in [b"one".as_slice(), b"another gap", b"third gap"] {
            bytes.extend_from_slice(gap);
            bytes.extend_from_slice(&source[5..]);
        }
        let tape = commands(&bytes).unwrap().unwrap();
        assert!(tape.len() < bytes.len() / 3);
        assert_eq!(lz::decode(&tape, bytes.len()).unwrap(), bytes);
        assert!(commands(&bytes[..1024]).unwrap().is_none());
    }
}
