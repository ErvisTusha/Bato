//! Original fixed-point adaptive maps and byte-run predictor.
//! Provenance: count-limited reciprocal-rate adaptation from the public-domain
//! ZPAQ specification section 3.2, and elementary run statistics described in
//! Mahoney, Data Compression Explained. No implementation or table is copied.

use super::{coded_bits, logistic, state};
/// Largest cell probability, in 22-bit fixed point.
const Q: i64 = (1 << 22) - 1;
/// Low bits of a cell holding its observation count.
const COUNT_BITS: u32 = 10;
/// Largest storable observation count.
const COUNT_MAX: u32 = (1 << COUNT_BITS) - 1;
/// `train_weighted` weight that applies a whole update.
pub(super) const FULL_WEIGHT: u16 = 256;
/// Count limit of bit-history state maps: their rate settles near 1/512.
const STATE_LIMIT: u16 = 512;
/// Runs shorter than this give no prediction.
const MIN_RUN: u16 = 2;
/// Surprise probability of a run of length one; it shrinks with the run.
const RUN_SURPRISE: u32 = 8192;
/// Run length beyond which confidence stops growing.
const RUN_CAP: u16 = 256;
/// Smallest surprise probability, bounding confidence on long runs.
const MIN_RUN_SURPRISE: u32 = 32;
const fn reciprocals() -> [u32; COUNT_MAX as usize + 1] {
    let mut output = [0; COUNT_MAX as usize + 1];
    let mut i = 0;
    while i <= COUNT_MAX as usize {
        output[i] = (1 << 17) / (2 * i as u32 + 3);
        i += 1;
    }
    output
}
static RATES: [u32; COUNT_MAX as usize + 1] = reciprocals();

#[derive(Clone, Copy, Debug)]
pub(super) struct Cell(u32);
impl Cell {
    pub(super) fn new(probability: u16, count: u16) -> Self {
        Self((u32::from(probability) << 16) | u32::from(count).min(COUNT_MAX))
    }
    #[inline]
    pub(super) fn probability(self) -> u16 {
        ((self.0 >> 16) as u16).clamp(1, logistic::MAX)
    }
    #[inline]
    pub(super) fn train(&mut self, bit: u8, limit: u16) {
        // Equal to `train_weighted` with `FULL_WEIGHT`: multiplying by 2^8
        // before the 2^24 shift is a shift by 2^16.
        const _: () = assert!(FULL_WEIGHT == 1 << 8);
        let count = self.0 & COUNT_MAX;
        let probability = i64::from(self.0 >> COUNT_BITS);
        let target = if bit == 0 { Q } else { 0 };
        let delta = ((target - probability) * i64::from(RATES[count as usize])) >> 16;
        let next = (probability + delta).clamp(64, Q - 63) as u32;
        self.0 = (next << COUNT_BITS) | (count + 1).min(u32::from(limit).min(COUNT_MAX));
    }
    #[inline]
    pub(super) fn train_weighted(&mut self, bit: u8, limit: u16, weight: u16) {
        debug_assert!(weight <= FULL_WEIGHT);
        if weight == 0 {
            return;
        }
        let count = self.0 & COUNT_MAX;
        let probability = i64::from(self.0 >> COUNT_BITS);
        let target = if bit == 0 { Q } else { 0 };
        let delta =
            ((target - probability) * i64::from(RATES[count as usize]) * i64::from(weight)) >> 24;
        let next = (probability + delta).clamp(64, Q - 63) as u32;
        self.0 = (next << COUNT_BITS) | (count + 1).min(u32::from(limit).min(COUNT_MAX));
    }
}
pub(super) struct StateMap {
    cells: [Cell; 256],
}
impl StateMap {
    pub(super) fn new() -> Self {
        let mut cells = [Cell::new(logistic::HALF, 0); 256];
        for (i, cell) in cells.iter_mut().enumerate() {
            *cell = Cell::new(state::initial_probability(i as u8), 0);
        }
        Self { cells }
    }
    #[inline]
    pub(super) fn probability(&self, state: u8) -> u16 {
        self.cells[usize::from(state)].probability()
    }
    #[inline]
    pub(super) fn observe(&mut self, state: u8, bit: u8) {
        self.cells[usize::from(state)].train(bit, STATE_LIMIT);
    }
}

#[derive(Debug, Default)]
pub(super) struct RunMap {
    byte: u8,
    length: u16,
}
impl RunMap {
    pub(super) fn probability(&self, prefix: u16) -> u16 {
        if self.length < MIN_RUN {
            return logistic::HALF;
        }
        let depth = coded_bits(prefix);
        let expected_prefix = (256u16 | u16::from(self.byte)) >> (8 - depth);
        if prefix != expected_prefix {
            return logistic::HALF;
        }
        let bit = (self.byte >> (7 - depth)) & 1;
        let surprise =
            (RUN_SURPRISE / u32::from(self.length.min(RUN_CAP))).max(MIN_RUN_SURPRISE) as u16;
        if bit == 0 {
            logistic::MAX - surprise
        } else {
            surprise
        }
    }
    pub(super) fn observe_byte(&mut self, byte: u8) {
        self.length = if self.length != 0 && self.byte == byte {
            self.length.saturating_add(1)
        } else {
            1
        };
        self.byte = byte;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_probability_and_count_do_not_overlap() {
        for probability in [1, 2, 32768, 65534, 65535] {
            for count in [0, 1, 511, 1023] {
                let cell = Cell::new(probability, count);
                assert_eq!(cell.probability(), probability);
                assert_eq!(cell.0 & 1023, u32::from(count));
            }
        }
        assert_eq!(std::mem::size_of::<Cell>(), 4);
    }
    #[test]
    fn state_map_converges_recovers_and_keeps_probability_bounds() {
        let mut map = StateMap::new();
        for _ in 0..2000 {
            map.observe(0, 0);
        }
        assert!(map.probability(0) > 65000);
        for _ in 0..4000 {
            map.observe(0, 1);
        }
        assert!(map.probability(0) < 100);
        for i in 0..10000 {
            map.observe(7, (i & 1) as u8);
        }
        assert!((32000..33500).contains(&map.probability(7)));
        for s in 0..=255u8 {
            assert!(map.probability(s) != 0);
        }
    }
    #[test]
    fn run_prediction_tracks_prefix_and_resets_on_changed_byte() {
        let mut model = RunMap::default();
        assert_eq!(model.probability(1), 32768);
        for _ in 0..30 {
            model.observe_byte(0xa5);
        }
        assert!(model.probability(1) < 1024);
        assert!(model.probability(3) > 64000);
        assert_eq!(model.probability(2), 32768);
        model.observe_byte(0x00);
        assert_eq!(model.probability(1), 32768);
    }
}
