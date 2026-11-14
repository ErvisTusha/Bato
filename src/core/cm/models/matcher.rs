//! Original causal segment-history match predictor.
//! Provenance: context-indexed expected-byte prediction described in Mahoney's
//! Data Compression Explained and ZPAQ section 3.4. The original Rust uses the
//! caller's already-observed segment prefix, with no duplicate history buffer.

use super::super::{coded_bits, logistic};
use crate::core::{memory::try_filled, types::BatoError};
/// Smallest match index, in entries.
const MIN_ENTRIES: usize = 16;
/// log2 of the largest match index, in entries, whatever the table cap.
const MAX_ENTRIES_LOG2: u8 = 22;
/// Bytes hashed to find a match candidate (computed by the order-6 context).
#[cfg(test)]
const ORDER: usize = 6;
/// Verified length at which a candidate becomes an active match.
const MIN_LENGTH: u16 = 6;
/// Bytes compared backwards when verifying a new candidate.
const VERIFY_LIMIT: usize = 1024;
/// Number of candidate positions tracked per 6-byte context hash bucket.
const WAYS: usize = 4;
/// Largest tracked match length.
const MAX_LENGTH: u16 = 4095;
/// Surprise numerator: a 1-byte match predicts with error 1/8.
const SURPRISE: u32 = 65536 / 8;
/// Length beyond which the expected-bit probability stops sharpening.
const PROBABILITY_CAP: u16 = 256;
/// Smallest surprise probability, bounding confidence on long matches.
const MIN_SURPRISE: u32 = 16;
/// Length beyond which the expected-bit strength input stops growing.
const STRENGTH_CAP: u16 = 64;
/// Strength input per matched byte, in logit units.
const STRENGTH_STEP: i16 = 32;
/// Largest strength input, in logit units.
const STRENGTH_MAX: i16 = 2048;
fn entries(length: usize, cap: u8) -> usize {
    length
        .max(MIN_ENTRIES)
        .next_power_of_two()
        .min(1usize << cap.min(MAX_ENTRIES_LOG2))
        .max(WAYS)
}
pub(in crate::core::cm) fn memory_required(length: usize, cap: u8) -> usize {
    entries(length, cap) * 4
}
#[derive(Clone)]
pub(in crate::core::cm) struct Matcher {
    index: Vec<u32>,
    last_position: usize,
    candidate: usize,
    length: u16,
    active: bool,
    expected: u8,
}
impl Matcher {
    pub(in crate::core::cm) fn new(length: usize, cap: u8) -> Result<Self, BatoError> {
        Ok(Self {
            index: try_filled(entries(length, cap), 0)?,
            last_position: usize::MAX,
            candidate: 0,
            length: 0,
            active: false,
            expected: 0,
        })
    }
    pub(in crate::core::cm) fn begin_byte(&mut self, history: &[u8], order6_hash: u64) {
        let pos = history.len();
        if self.last_position == pos {
            return;
        }
        self.last_position = pos;
        let bucket = (order6_hash as usize & ((self.index.len() / WAYS) - 1)) * WAYS;
        if self.active {
            self.candidate += 1;
            self.length = self.length.saturating_add(1).min(MAX_LENGTH);
            self.active = self.candidate < pos;
        }
        let min_len = usize::from(MIN_LENGTH);
        if !self.active {
            self.length = 0;
            if pos >= min_len {
                let target6 = &history[pos - min_len..pos];
                for &previous in &self.index[bucket..bucket + WAYS] {
                    if previous == 0 {
                        break;
                    }
                    let cand = previous as usize - 1;
                    if cand >= pos {
                        continue;
                    }
                    let limit = pos.min(cand).min(VERIFY_LIMIT);
                    let best = usize::from(self.length);
                    if limit < min_len || limit <= best {
                        continue;
                    }
                    if best >= min_len && history[pos - best - 1] != history[cand - best - 1] {
                        continue;
                    }
                    if &history[cand - min_len..cand] != target6 {
                        continue;
                    }
                    let mut len = min_len;
                    while len < limit && history[pos - len - 1] == history[cand - len - 1] {
                        len += 1;
                    }
                    if len as u16 > self.length {
                        self.length = len as u16;
                        self.candidate = cand;
                    }
                }
            }
            self.active = self.length >= MIN_LENGTH;
        }
        let [s0, s1, s2, _] = self.index[bucket..bucket + WAYS] else {
            unreachable!()
        };
        self.index[bucket..bucket + WAYS].copy_from_slice(&[(pos as u32) + 1, s0, s1, s2]);
        if self.active {
            self.expected = history[self.candidate];
        }
    }
    pub(in crate::core::cm) fn features(&self, prefix: u16) -> [i16; 2] {
        if !self.active {
            return [0; 2];
        }
        let depth = coded_bits(prefix);
        let bit = (self.expected >> (7 - depth)) & 1;
        let surprise =
            (SURPRISE / u32::from(self.length.min(PROBABILITY_CAP))).max(MIN_SURPRISE) as u16;
        let probability = if bit == 0 {
            logistic::MAX - surprise
        } else {
            surprise
        };
        let strength =
            (i16::from(self.length.min(STRENGTH_CAP) as u8) * STRENGTH_STEP).min(STRENGTH_MAX);
        [
            logistic::stretch(probability),
            if bit == 0 { strength } else { -strength },
        ]
    }
    /// Distance back to the match source when the match is long, for lag
    /// detection.
    pub(in crate::core::cm) fn long_distance(&self, position: usize, from: u16) -> Option<usize> {
        (self.active && self.length >= from).then(|| position - self.candidate)
    }
    pub(in crate::core::cm) fn state(&self) -> usize {
        if self.active {
            (16 - self.length.leading_zeros()) as usize
        } else {
            0
        }
    }
    pub(in crate::core::cm) fn expected(&self) -> Option<(u8, u16)> {
        self.active.then_some((self.expected, self.length))
    }
    pub(in crate::core::cm) fn observe(&mut self, prefix: u16, bit: u8) {
        let depth = coded_bits(prefix);
        if self.active && ((self.expected >> (7 - depth)) & 1) != bit {
            self.active = false;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cm::hash;
    #[test]
    fn matches_only_predict_from_the_observed_prefix_and_recover_after_a_miss() {
        let raw = b"abcdefabcdefabcdef-X-abcdefabcdefabcdef";
        let mut model = Matcher::new(raw.len(), 16).unwrap();
        let mut active = 0;
        for (pos, &byte) in raw.iter().enumerate() {
            let order6 = hash::bytes(&raw[..pos], ORDER);
            model.begin_byte(&raw[..pos], order6);
            let before = (model.candidate, model.length, model.active);
            model.begin_byte(&raw[..pos], order6);
            assert_eq!(before, (model.candidate, model.length, model.active));
            if model.active {
                assert!(model.candidate < pos);
                active += 1;
            }
            let mut prefix = 1;
            for shift in (0..8).rev() {
                let bit = (byte >> shift) & 1;
                assert!(model
                    .features(prefix)
                    .iter()
                    .all(|x| (-4096..=4096).contains(x)));
                model.observe(prefix, bit);
                prefix = prefix * 2 + u16::from(bit);
            }
        }
        assert!(active > 10);
    }
    #[test]
    fn match_index_heap_equals_the_reservation() {
        for length in [1, 65536, 1000000] {
            for cap in [0, 8, 20, 24] {
                let (_, observed) =
                    crate::core::diagnostics::capture(|| Matcher::new(length, cap).unwrap());
                assert_eq!(observed.heap.peak, memory_required(length, cap));
            }
        }
    }
}
