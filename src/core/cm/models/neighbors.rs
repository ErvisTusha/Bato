//! Original causal bit-neighborhood expert over a statistically inferred stride.
//! Provenance: neighboring-pixel/record prediction described in Mahoney's
//! Data Compression Explained. The context, fixed-point adaptation and Rust
//! here are original; there is no image parser or hard-coded record width.

use super::super::{adapt::Cell, coded_bits, logistic};

/// Cells indexed by 3 left bits, 3 above bits, 1 far bit and the bit depth.
const CELLS: usize = 1 << 10;
/// Count limit: the cells' adaptation rate settles near 1/256.
const LIMIT: u16 = 256;

pub(in crate::core::cm) struct Neighbors {
    cells: [Cell; CELLS],
    above: u32,
    far: u8,
    left: u8,
    enabled: bool,
    selected: Option<usize>,
}
impl Default for Neighbors {
    fn default() -> Self {
        Self {
            cells: [Cell::new(logistic::HALF, 0); CELLS],
            above: 0,
            far: 0,
            left: 0,
            enabled: false,
            selected: None,
        }
    }
}
impl Neighbors {
    pub(in crate::core::cm) fn begin_byte(&mut self, history: &[u8], period: Option<usize>) {
        self.enabled = false;
        self.selected = None;
        let Some(stride) = period.filter(|&s| s >= 2 && s <= history.len() / 2) else {
            return;
        };
        let n = history.len();
        // stride >= 2 keeps even the above-right byte strictly in the past.
        // Two complete prior rows also make the above-left/far accesses safe.
        self.above = (u32::from(history[n - stride - 1]) << 16)
            | (u32::from(history[n - stride]) << 8)
            | u32::from(history[n - stride + 1]);
        self.far = history[n - 2 * stride];
        self.left = history[n - 1];
        self.enabled = true;
    }

    #[inline]
    pub(in crate::core::cm) fn feature(&mut self, prefix: u16) -> i16 {
        if !self.enabled {
            return 0;
        }
        debug_assert!((1..256).contains(&prefix));
        let depth = coded_bits(prefix);
        let partial = prefix ^ (1 << depth);
        let left = ((u16::from(self.left) << depth) | partial) & 7;
        let above = (self.above >> (14 - depth)) & 7;
        let far = (self.far >> (7 - depth)) & 1;
        let context = usize::from(left)
            | ((above as usize) << 3)
            | (usize::from(far) << 6)
            | ((depth as usize) << 7);
        self.selected = Some(context);
        logistic::stretch(self.cells[context].probability())
    }

    #[inline]
    pub(in crate::core::cm) fn observe(&mut self, bit: u8) {
        if let Some(context) = self.selected {
            self.cells[context].train(bit, LIMIT);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_or_unsafe_rows_are_dormant_and_every_prefix_is_bounded() {
        let history: Vec<_> = (0..200).map(|i| (i * 37) as u8).collect();
        let mut model = Neighbors::default();
        for period in [None, Some(0), Some(1), Some(101), Some(usize::MAX)] {
            model.begin_byte(&history, period);
            assert_eq!(model.feature(1), 0);
            assert!(model.selected.is_none());
        }
        for period in [2, 17, 100] {
            model.begin_byte(&history, Some(period));
            for prefix in 1..256 {
                assert_eq!(model.feature(prefix), 0);
                assert!(model.selected.unwrap() < 1024);
            }
        }
    }

    #[test]
    fn neighborhood_cells_learn_opposite_bits_without_sharing_contexts() {
        let mut model = Neighbors::default();
        for byte in [0, 255] {
            let history = vec![byte; 34];
            for _ in 0..2000 {
                model.begin_byte(&history, Some(17));
                let mut prefix = 1;
                for shift in (0..8).rev() {
                    model.feature(prefix);
                    let bit = (byte >> shift) & 1;
                    model.observe(bit);
                    prefix = (prefix << 1) | u16::from(bit);
                }
            }
            model.begin_byte(&history, Some(17));
            let logit = model.feature(1);
            assert!(if byte == 0 {
                logit > 2000
            } else {
                logit < -2000
            });
        }
        model.begin_byte(&[0; 34], Some(17));
        assert!(model.feature(1) > 2000);
    }
}
