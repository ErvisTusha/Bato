//! Causal binary arithmetic coding for bytes with a strong spatial period.
//! Provenance: fixed causal pixel templates in ITU-T T.82 and adaptive
//! context weighting described by Mahoney in Data Compression Explained.
//! The stride detector and nested counter models are original Rust.

use super::{coder, varint};
use crate::core::{
    memory::{try_capacity, try_filled},
    types::BatoError,
};

const MIN_STRIDE: usize = 8;
const MAX_STRIDE: usize = 1024;
const MIN_INPUT: usize = 8192;
const SAMPLES: usize = 1024;

/// Finds a long lag with substantially fewer changed bits than the adjacent
/// byte. The decoder gets the winning lag in the packet.
fn stride(raw: &[u8]) -> Option<usize> {
    if raw.len() < MIN_INPUT {
        return None;
    }
    let step = (raw.len() - MAX_STRIDE).div_ceil(SAMPLES);
    let positions = (MAX_STRIDE..raw.len()).step_by(step);
    let len = positions.len();
    let solid = positions
        .clone()
        .filter(|&i| matches!(raw[i], 0 | 255))
        .count();
    if solid * 4 < len * 3 {
        return None;
    }
    let score = |lag: usize| {
        positions
            .clone()
            .map(|i| (raw[i] ^ raw[i - lag]).count_ones())
            .sum::<u32>()
    };
    let adjacent = score(1);
    let (changes, lag) = (MIN_STRIDE..=MAX_STRIDE)
        .map(|lag| (score(lag), lag))
        .min()?;
    (changes * 4 <= len as u32 && changes * 2 < adjacent).then_some(lag)
}

#[derive(Clone, Copy, Default)]
struct History(u32);

impl History {
    fn counts(self) -> (u32, u32) {
        let zero = self.0 & 0xffff;
        (zero, zero + (self.0 >> 16))
    }

    fn observe(&mut self, bit: u8) {
        let (_, count) = self.counts();
        if count >= 2048 {
            self.0 = (self.0 >> 1) & 0x7fff_7fff;
        }
        self.0 += 1 << (16 * bit);
    }
}

fn estimate(zero: u32, count: u32) -> u32 {
    (2 * zero + 1) * 32768 / (count + 1)
}

fn blend(p: u32, h: History, strength: u32) -> u32 {
    let (zero, count) = h.counts();
    if count == 0 {
        p
    } else {
        (p * strength + estimate(zero, count) * count) / (strength + count)
    }
}

struct Model {
    small: Vec<History>,
    medium: Vec<History>,
    large: Vec<History>,
}

impl Model {
    fn new() -> Result<Self, BatoError> {
        Ok(Self {
            small: try_filled(1 << 9, History::default())?,
            medium: try_filled(1 << 16, History::default())?,
            large: try_filled(1 << 19, History::default())?,
        })
    }

    fn predict(&self, [small, medium, large]: [usize; 3]) -> u16 {
        let (zero, count) = self.small[small].counts();
        let p = blend(estimate(zero, count), self.medium[medium], 2);
        blend(p, self.large[large], 3) as u16
    }

    fn observe(&mut self, [small, medium, large]: [usize; 3], bit: u8) {
        self.small[small].observe(bit);
        self.medium[medium].observe(bit);
        self.large[large].observe(bit);
    }
}

/// Read a row window once per byte. Reversing each byte puts earlier bit
/// positions in the low bits, so each template is a shift and mask.
fn row_window(history: &[u8], index: usize, distance: usize) -> usize {
    let at = |offset: usize| {
        (index + offset)
            .checked_sub(distance + 1)
            .map_or(0, |i| usize::from(history[i].reverse_bits()))
    };
    at(0) | (at(1) << 8) | (at(2) << 16)
}

fn row_windows(history: &[u8], index: usize, width: usize) -> [usize; 3] {
    [width, 2 * width, 3 * width].map(|distance| row_window(history, index, distance))
}

#[inline]
fn indices([above, far, third]: [usize; 3], depth: usize, left: usize) -> [usize; 3] {
    let above = (above >> (5 + depth)) & 127;
    let far = (far >> (6 + depth)) & 31;
    let third = (third >> (7 + depth)) & 7;
    let small = ((left & 7) << 6) | (((above >> 1) & 31) << 1) | ((far >> 2) & 1);
    let medium = (left << 12) | (above << 5) | far;
    [small, medium, (third << 16) | medium]
}

pub(super) fn encode(raw: &[u8], limit: usize) -> Result<Option<Vec<u8>>, BatoError> {
    let Some(width) = stride(raw) else {
        return Ok(None);
    };
    let mut packet = Vec::new();
    varint::push(&mut packet, width);
    let mut coder = coder::Encoder::new(limit.saturating_sub(packet.len()))?;
    let mut model = Model::new()?;
    let mut left = 0;
    for (index, &byte) in raw.iter().enumerate() {
        let windows = row_windows(raw, index, width);
        for depth in 0..8 {
            let bit = (byte >> (7 - depth)) & 1;
            let key = indices(windows, depth, left);
            coder.bit(bit, model.predict(key))?;
            model.observe(key, bit);
            left = (left >> 1) | (usize::from(bit) << 3);
        }
        if coder.exceeded() {
            return Ok(None);
        }
    }
    let Some(payload) = coder.finish() else {
        return Ok(None);
    };
    packet.extend_from_slice(&payload);
    Ok(Some(packet))
}

pub(super) fn decode(packet: &[u8], length: usize) -> Result<Vec<u8>, BatoError> {
    let mut start = 0;
    let width = varint::take(packet, &mut start)?;
    if !(MIN_STRIDE..=MAX_STRIDE).contains(&width) || length < MIN_INPUT {
        return Err(BatoError::CorruptStream(
            "invalid spatial packet geometry".into(),
        ));
    }
    let mut output = try_capacity(length)?;
    let mut model = Model::new()?;
    let mut coder = coder::Decoder::new(&packet[start..]);
    let mut left = 0;
    for index in 0..length {
        let windows = row_windows(&output, index, width);
        let mut byte = 0;
        for depth in 0..8 {
            let key = indices(windows, depth, left);
            let bit = coder.bit(model.predict(key))?;
            model.observe(key, bit);
            left = (left >> 1) | (usize::from(bit) << 3);
            byte = (byte << 1) | bit;
        }
        output.push(byte);
    }
    coder.finish()?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_windows_match_bitwise_reference() {
        let raw: Vec<u8> = (0..4200usize)
            .map(|i| i.wrapping_mul(99_991).rotate_right(6) as u8 ^ (i >> 8) as u8)
            .collect();
        for width in [8, 37, 216, 1024] {
            let mut left = 0;
            for (index, &byte) in raw.iter().enumerate() {
                let rows = row_windows(&raw[..index], index, width);
                for depth in 0..8 {
                    let at = index * 8 + depth;
                    let scalar = |start, end| {
                        (start..=end).fold(0usize, |key, offset| {
                            let bit = at
                                .checked_sub(offset)
                                .map_or(0, |bit| usize::from((raw[bit / 8] >> (7 - bit % 8)) & 1));
                            (key << 1) | bit
                        })
                    };
                    let [a, b, c] = [width * 8, width * 16, width * 24];
                    let l = scalar(1, 4);
                    let above = scalar(a - 3, a + 3);
                    let far = scalar(b - 2, b + 2);
                    let third = scalar(c - 1, c + 1);
                    let small = ((l & 7) << 6) | (((above >> 1) & 31) << 1) | ((far >> 2) & 1);
                    let medium = (l << 12) | (above << 5) | far;
                    assert_eq!(
                        indices(rows, depth, left),
                        [small, medium, (third << 16) | medium]
                    );
                    left = (left >> 1) | (usize::from((byte >> (7 - depth)) & 1) << 3);
                }
            }
        }
    }

    #[test]
    fn spatial_round_trip_and_bounded_packet() {
        let mut row = [0; 37];
        row[2..8].fill(255);
        row[19] = 63;
        row[33] = 4;
        let raw = row.repeat(400);
        assert_eq!(stride(&raw), Some(row.len()));
        let packet = encode(&raw, raw.len()).unwrap().unwrap();
        assert_eq!(decode(&packet, raw.len()).unwrap(), raw);
        assert!(encode(&raw, 1).unwrap().is_none());
        assert!(decode(&[0], raw.len()).is_err());
        let mut trailing = packet;
        trailing.push(0);
        assert!(decode(&trailing, raw.len()).is_err());
    }
}
