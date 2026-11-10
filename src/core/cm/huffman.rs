//! Original canonical static Huffman tables, learned once by the encoder.
//! Provenance: Huffman's 1952 prefix-code construction and canonical code
//! lengths. Tables are serialized; decoding does no adaptive model training.
use super::varint;
use crate::core::{
    memory::{try_capacity, try_filled},
    types::BatoError,
};
use std::{cmp::Reverse, collections::BinaryHeap};
const MAX_BITS: usize = 15;

/// Minimum-weight prefix code with a hard bit-length limit (package-merge).
/// Called only when the ordinary Huffman tree exceeds MAX_BITS, so the
/// common-case encoder continues using the faster unrestricted algorithm.
/// Each package records indices into the preceding list; selecting the
/// 2n-2 cheapest objects at the deepest level gives the optimal bit lengths.
fn length_limited(counts: &[u64; 256], max_bits: usize) -> Option<[u8; 256]> {
    #[derive(Clone, Copy)]
    enum Origin {
        Symbol(usize),
        Pair(usize, usize),
    }
    #[derive(Clone, Copy)]
    struct Entry {
        weight: u64,
        origin: Origin,
    }
    let mut leaves: Vec<(u64, usize)> = counts
        .iter()
        .enumerate()
        .filter(|(_, frequency)| **frequency != 0)
        .map(|(symbol, &frequency)| (frequency, symbol))
        .collect();
    leaves.sort_unstable();
    if leaves.len() < 2 || max_bits == 0 || leaves.len() > (1usize << max_bits) {
        return None;
    }
    let leaf_entries: Vec<Entry> = leaves
        .into_iter()
        .map(|(weight, symbol)| Entry {
            weight,
            origin: Origin::Symbol(symbol),
        })
        .collect();
    let symbols = leaf_entries.len();
    let mut levels = vec![leaf_entries.clone()];
    for _ in 1..max_bits {
        let previous = levels.last()?;
        let pairs = previous.len() / 2;
        let mut merged = Vec::with_capacity(symbols + pairs);
        let (mut leaf, mut pair) = (0, 0);
        while leaf < symbols || pair < pairs {
            let package_weight = if pair < pairs {
                previous[2 * pair].weight + previous[2 * pair + 1].weight
            } else {
                u64::MAX
            };
            if leaf < symbols && (pair == pairs || leaf_entries[leaf].weight <= package_weight) {
                merged.push(leaf_entries[leaf]);
                leaf += 1;
            } else {
                merged.push(Entry {
                    weight: package_weight,
                    origin: Origin::Pair(2 * pair, 2 * pair + 1),
                });
                pair += 1;
            }
        }
        levels.push(merged);
    }
    let needed = 2 * symbols - 2;
    if levels.last()?.len() < needed {
        return None;
    }
    let mut selected: Vec<usize> = (0..needed).collect();
    let mut lengths = [0u8; 256];
    for depth in (0..max_bits).rev() {
        let mut next = Vec::new();
        for index in selected {
            match levels[depth][index].origin {
                Origin::Symbol(symbol) => lengths[symbol] += 1,
                Origin::Pair(left, right) => {
                    next.push(left);
                    next.push(right);
                }
            }
        }
        selected = next;
    }
    Some(lengths)
}

fn lengths(raw: &[u8]) -> Option<([u8; 256], [u64; 256])> {
    let mut counts = [0u64; 256];
    for &b in raw {
        counts[b as usize] += 1;
    }
    let mut nodes = Vec::new();
    let mut heap = BinaryHeap::new();
    for (b, &count) in counts.iter().enumerate() {
        if count != 0 {
            let id = nodes.len();
            nodes.push((b, usize::MAX));
            heap.push(Reverse((count, id)));
        }
    }
    if heap.len() < 2 {
        return None;
    }
    while heap.len() > 1 {
        let Reverse((a, i)) = heap.pop()?;
        let Reverse((b, j)) = heap.pop()?;
        let id = nodes.len();
        nodes.push((i, j));
        heap.push(Reverse((a + b, id)));
    }
    let Reverse((_, root)) = heap.pop()?;
    let mut pending = vec![(root, 0u8)];
    let mut lengths = [0u8; 256];
    while let Some((i, depth)) = pending.pop() {
        let (a, b) = nodes[i];
        if b == usize::MAX {
            lengths[a] = depth;
        } else {
            pending.push((a, depth + 1));
            pending.push((b, depth + 1));
        }
    }
    if lengths.iter().any(|&bits| bits as usize > MAX_BITS) {
        lengths = length_limited(&counts, MAX_BITS)?;
    }
    Some((lengths, counts))
}
fn codes(lengths: &[u8; 256]) -> Result<[u32; 256], BatoError> {
    let mut counts = [0u32; MAX_BITS + 1];
    for &n in lengths {
        if n as usize > MAX_BITS {
            return Err(invalid());
        }
        if n != 0 {
            counts[n as usize] += 1;
        }
    }
    let mut next = [0u32; MAX_BITS + 1];
    let mut code = 0;
    for bits in 1..=MAX_BITS {
        code = (code + counts[bits - 1]) << 1;
        next[bits] = code;
    }
    if code + counts[MAX_BITS] != (1 << MAX_BITS) {
        return Err(invalid());
    }
    let mut out = [0u32; 256];
    for (i, &n) in lengths.iter().enumerate() {
        if n != 0 {
            out[i] = next[n as usize].reverse_bits() >> (32 - n);
            next[n as usize] += 1;
        }
    }
    Ok(out)
}
pub(super) fn encode(raw: &[u8], limit: usize) -> Option<Vec<u8>> {
    let (lens, counts) = lengths(raw)?;
    let symbols = lens.iter().filter(|&&n| n > 0).count();
    let bits: u64 = lens
        .iter()
        .zip(&counts)
        .map(|(&length, &count)| u64::from(length) * count)
        .sum();
    // This is the exact table plus bitstream length, including zero padding.
    // Omit bit packing only when these actual bytes cannot beat the incumbent.
    let size = super::varint::len(symbols as u64) + 2 * symbols + bits.div_ceil(8) as usize;
    if size > limit {
        return None;
    }

    let code = codes(&lens).ok()?; // A failed optional candidate leaves the current representation intact.
    let mut out = Vec::new();
    varint::push(&mut out, lens.iter().filter(|&&n| n > 0).count());
    for (i, &n) in lens.iter().enumerate() {
        if n > 0 {
            out.extend_from_slice(&[i as u8, n]);
        }
    }
    let (mut buffer, mut bits) = (0u64, 0u32);
    for &b in raw {
        buffer |= u64::from(code[b as usize]) << bits;
        bits += u32::from(lens[b as usize]);
        while bits >= 8 {
            out.push(buffer as u8);
            buffer >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out.push(buffer as u8);
    }
    Some(out)
}
pub(super) fn decode(payload: &[u8], length: usize) -> Result<Vec<u8>, BatoError> {
    let mut at = 0;
    let count = varint::take(payload, &mut at)?;
    if !(2..=256).contains(&count) || count * 2 > payload.len().saturating_sub(at) {
        return Err(invalid());
    }
    let mut lens = [0u8; 256];
    let mut previous = None;
    for _ in 0..count {
        let b = payload[at];
        let n = payload[at + 1];
        at += 2;
        if n == 0 || n as usize > MAX_BITS || previous.is_some_and(|p| b <= p) {
            return Err(invalid());
        }
        previous = Some(b);
        lens[b as usize] = n;
    }
    let code = codes(&lens)?;
    let mut table = try_filled(1 << MAX_BITS, 0u16)?;
    for (b, &n) in lens.iter().enumerate() {
        if n == 0 {
            continue;
        }
        for i in (code[b] as usize..1 << MAX_BITS).step_by(1 << n) {
            table[i] = (u16::from(n) << 8) | b as u16;
        }
    }
    let mut out = try_capacity(length)?;
    let (mut buffer, mut bits) = (0u64, 0usize);
    for _ in 0..length {
        while bits < MAX_BITS && at < payload.len() {
            buffer |= u64::from(payload[at]) << bits;
            at += 1;
            bits += 8;
        }
        let entry = table[buffer as usize & ((1 << MAX_BITS) - 1)];
        let n = (entry >> 8) as usize;
        if n == 0 || bits < n {
            return Err(invalid());
        }
        out.push(entry as u8);
        buffer >>= n;
        bits -= n;
    }
    if at != payload.len() || bits >= 8 || buffer != 0 {
        return Err(invalid());
    }
    Ok(out)
}
fn invalid() -> BatoError {
    BatoError::CorruptStream("invalid canonical Huffman block".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn package_merge_minimizes_limited_prefix_cost() {
        let mut counts = [0u64; 256];
        counts[..4].copy_from_slice(&[1, 1, 1, 10]);
        let lengths = length_limited(&counts, 3).unwrap();
        assert_eq!(
            (0..4)
                .map(|i| counts[i] * u64::from(lengths[i]))
                .sum::<u64>(),
            18,
        );
        assert_eq!(
            lengths[..4]
                .iter()
                .map(|&bits| 1usize << (3 - usize::from(bits)))
                .sum::<usize>(),
            1 << 3,
        );
        for (count, limit) in [(16, 4), (17, 5), (50, 6), (256, 15)] {
            let mut counts = [0u64; 256];
            for (i, value) in counts.iter_mut().take(count).enumerate() {
                *value = (i as u64 + 1) * (i as u64 % 7 + 1);
            }
            let lengths = length_limited(&counts, limit).unwrap();
            assert!(lengths[..count]
                .iter()
                .all(|&bits| bits > 0 && usize::from(bits) <= limit));
            assert_eq!(
                lengths[..count]
                    .iter()
                    .map(|&bits| 1usize << (limit - usize::from(bits)))
                    .sum::<usize>(),
                1usize << limit,
            );
        }
    }

    #[test]
    fn very_skewed_histograms_use_valid_length_limited_codes() {
        // Fibonacci frequencies yield a Huffman tree deeper than 15 bits.
        let (mut a, mut b) = (1usize, 1usize);
        let mut input = Vec::new();
        for symbol in 0..22u8 {
            input.extend(std::iter::repeat_n(symbol, a));
            (a, b) = (b, a + b);
        }
        let (lengths, _) = lengths(&input).unwrap();
        assert!(lengths.iter().all(|&n| n <= MAX_BITS as u8));
        assert!(codes(&lengths).is_ok());
        let payload = encode(&input, usize::MAX).unwrap();
        assert_eq!(decode(&payload, input.len()).unwrap(), input);
    }

    #[test]
    fn canonical_tables_padding_and_arbitrary_symbols() {
        for raw in [
            b"aaaabbcdeff\0\xff".repeat(1500),
            (0..=255u8).cycle().take(65536).collect(),
        ] {
            let z = encode(&raw, usize::MAX).unwrap();
            assert_eq!(decode(&z, raw.len()).unwrap(), raw);
            assert!(decode(&z[..z.len() - 1], raw.len()).is_err());
            let mut extra = z.clone();
            extra.push(0);
            assert!(decode(&extra, raw.len()).is_err());
        }
        assert!(decode(&[2, 1, 1, 1, 1, 0], 100).is_err());
        assert!(encode(&[1; 100], usize::MAX).is_none());
    }
}
