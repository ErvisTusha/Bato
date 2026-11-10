//! Original bounded four-candidate LZ77 parser and overlap-safe copy decoder.
//! Provenance: public LZ77 distance/length principle (Ziv/Lempel, 1977) and
//! Brotli RFC 7932 insert/copy descriptions. No external compressor code.
use super::varint;
use crate::core::{memory::try_filled, types::BatoError};

const MIN_MATCH: usize = 4;
const CANDIDATES: usize = 4;
const MAX_BUCKETS: usize = 1 << 18;

fn hash(raw: &[u8], at: usize, mask: usize) -> usize {
    let key = u32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
    (key.wrapping_mul(0x9e37_79b1) >> 8) as usize & mask
}
fn insert(heads: &mut [[u32; CANDIDATES]], raw: &[u8], at: usize) {
    if at + MIN_MATCH > raw.len() {
        return;
    }
    let slot = hash(raw, at, heads.len() - 1);
    heads[slot].rotate_right(1);
    heads[slot][0] = (at + 1) as u32;
}
fn matched(raw: &[u8], a: usize, b: usize) -> usize {
    let limit = raw.len() - b;
    let mut n = 0;
    while n + 8 <= limit {
        // Match length is the same as byte-by-byte comparison, including
        // overlap, but a differing word exposes the first mismatch directly.
        let left = u64::from_le_bytes(raw[a + n..a + n + 8].try_into().unwrap());
        let right = u64::from_le_bytes(raw[b + n..b + n + 8].try_into().unwrap());
        let mismatch = left ^ right;
        if mismatch != 0 {
            return n + (mismatch.trailing_zeros() / 8) as usize;
        }
        n += 8;
    }
    while n < limit && raw[a + n] == raw[b + n] {
        n += 1;
    }
    n
}
fn best(raw: &[u8], at: usize, heads: &[[u32; CANDIDATES]]) -> (usize, usize) {
    if at + MIN_MATCH > raw.len() {
        return (0, 0);
    }
    let mut found = (0, 0);
    let mut best_saving = 0;
    for &candidate in &heads[hash(raw, at, heads.len() - 1)] {
        if candidate == 0 {
            continue;
        }
        let previous = candidate as usize - 1;
        if previous >= at {
            continue;
        }
        let n = matched(raw, previous, at);
        if n < MIN_MATCH {
            continue;
        }
        let distance = at - previous;
        let cost = varint::len((((n - MIN_MATCH) << 1) | 1) as u64) + varint::len(distance as u64);
        if n <= cost + 1 {
            continue;
        }
        let saving = n - cost;
        if saving > best_saving
            || (saving == best_saving && (n > found.0 || (n == found.0 && distance < found.1)))
        {
            found = (n, distance);
            best_saving = saving;
        }
    }
    found
}
pub(super) fn encode(raw: &[u8]) -> Result<Vec<u8>, BatoError> {
    let buckets = (raw.len() / 4).max(16).next_power_of_two().min(MAX_BUCKETS);
    let mut heads = try_filled(buckets, [0u32; CANDIDATES])?;
    let mut output = Vec::new();
    let (mut at, mut literal) = (0, 0);
    while at + MIN_MATCH <= raw.len() {
        let current = best(raw, at, &heads);
        insert(&mut heads, raw, at);
        let next = best(raw, at + 1, &heads);
        let saving = |(length, distance): (usize, usize)| -> usize {
            if length < MIN_MATCH {
                return 0;
            }
            let cost = varint::len((((length - MIN_MATCH) << 1) | 1) as u64)
                + varint::len(distance as u64);
            length.saturating_sub(cost)
        };
        if current.0 < MIN_MATCH || saving(next) > saving(current) + 1 {
            at += 1;
            continue;
        }
        if literal < at {
            varint::push(&mut output, (at - literal) << 1);
            output.extend_from_slice(&raw[literal..at]);
        }
        varint::push(&mut output, ((current.0 - MIN_MATCH) << 1) | 1);
        varint::push(&mut output, current.1);
        let end = at + current.0;
        // Index the edges and a deterministic sparse interior of long copies.
        // This bounds parser work without changing decoder state or semantics.
        for p in at + 1..end.min(at + 32) {
            insert(&mut heads, raw, p);
        }
        if end > at + 64 {
            for p in (at + 32..end - 32).step_by(64) {
                insert(&mut heads, raw, p);
            }
        }
        for p in (at + 32).max(end.saturating_sub(32))..end {
            insert(&mut heads, raw, p);
        }
        at = end;
        literal = at;
    }
    if literal < raw.len() {
        varint::push(&mut output, (raw.len() - literal) << 1);
        output.extend_from_slice(&raw[literal..]);
    }
    Ok(output)
}
pub(super) fn decode(payload: &[u8], length: usize) -> Result<Vec<u8>, BatoError> {
    let mut output = crate::core::memory::try_capacity(length)?;
    let mut at = 0;
    while output.len() < length {
        let tag = varint::take(payload, &mut at)?;
        if tag & 1 == 0 {
            let n = tag >> 1;
            if n == 0 || n > length - output.len() || n > payload.len().saturating_sub(at) {
                return Err(invalid());
            }
            output.extend_from_slice(&payload[at..at + n]);
            at += n;
            continue;
        }
        let n = (tag >> 1).checked_add(MIN_MATCH).ok_or_else(invalid)?;
        let distance = varint::take(payload, &mut at)?;
        if distance == 0 || distance > output.len() || n > length - output.len() {
            return Err(invalid());
        }
        let start = output.len() - distance;
        let mut left = n;
        while left > 0 {
            let take = left.min(output.len() - start);
            output.extend_from_within(start..start + take);
            left -= take;
        }
    }
    if at != payload.len() {
        return Err(invalid());
    }
    Ok(output)
}
fn invalid() -> BatoError {
    BatoError::CorruptStream("invalid literal or copy command".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn word_at_a_time_matches_are_byte_exact() {
        let mut bytes = b"abcdefghijklmnopqrstuvwxyz".repeat(8);
        for mismatch in 0..8 {
            bytes[32 + mismatch] = b'!';
            let expected = (0..bytes.len() - 32)
                .take_while(|&i| bytes[i] == bytes[32 + i])
                .count();
            assert_eq!(matched(&bytes, 0, 32), expected, "mismatch {mismatch}");
            bytes[32 + mismatch] = b'a' + (mismatch as u8 + 6) % 26;
        }
        for size in 1..64 {
            let data = vec![7u8; size + 1];
            assert_eq!(matched(&data, 0, 1), size);
        }
    }

    #[test]
    fn literals_long_overlap_and_arbitrary_bytes() {
        for raw in [
            vec![],
            vec![7],
            vec![9; 200000],
            b"abc".repeat(80000),
            (0..256).map(|x| x as u8).collect(),
        ] {
            let z = encode(&raw).unwrap();
            assert_eq!(decode(&z, raw.len()).unwrap(), raw);
        }
        assert!(
            encode(&b"repeated byte pattern\n".repeat(100000))
                .unwrap()
                .len()
                < 100
        );
    }
    #[test]
    fn malformed_copies_are_bounded() {
        for p in [
            vec![1, 0],
            vec![1, 1],
            vec![0],
            vec![254, 255],
            vec![128; 20],
        ] {
            assert!(decode(&p, 8).is_err());
        }
        let mut z = encode(&b"abc".repeat(100)).unwrap();
        z.push(0);
        assert!(decode(&z, 300).is_err());
    }
}
