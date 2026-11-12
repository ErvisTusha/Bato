//! Original causal bit-neighborhood expert over a statistically inferred stride.
//! Provenance: neighboring-pixel/record prediction described in Mahoney's
//! Data Compression Explained, and the pixel-template contexts of the public
//! bilevel image coding standards (ITU-T T.82 and T.88). The templates,
//! fixed-point adaptation and Rust here are original; there is no image
//! parser or hard-coded record width.
//!
//! Each bit is predicted from the bits around the same position in the rows
//! above, a row being the record period the record model inferred: a small
//! neighbourhood with the bit depth, two 16-pixel templates and one hashed
//! 33-pixel template. On bilevel images these are the pixels around the
//! current one; on fixed-width records, the same bits of earlier records.

use super::super::{adapt::Cell, coded_bits, hash, logistic};
use crate::core::{memory::try_filled, types::BatoError};

/// Cells indexed by 3 left bits, 3 above bits, 1 far bit and the bit depth.
const CELLS: usize = 1 << 10;
/// Cells of each 16-pixel template.
const TEMPLATE_CELLS: usize = 1 << 16;
/// Count limit: the cells' adaptation rate settles near 1/256.
const LIMIT: u16 = 256;
/// Predictions per bit: the small neighbourhood and two 16-pixel templates.
pub(in crate::core::cm) const FEATURES: usize = 4;
/// Cells of the hashed 33-pixel template.
const LARGE_CELLS: usize = 1 << 18;

/// Heap bytes of one model.
pub(in crate::core::cm) fn memory_required() -> usize {
    (2 * TEMPLATE_CELLS + LARGE_CELLS) * std::mem::size_of::<Cell>()
}

pub(in crate::core::cm) struct Neighbors {
    cells: [Cell; CELLS],
    /// Template of 4 left pixels, 7 above and 5 two rows above.
    wide: Vec<Cell>,
    /// Template of 3 left pixels, 5 above, 5 two rows above and 3 three rows above.
    tall: Vec<Cell>,
    /// Hashed template of 8 left pixels, 11 above, 9 two rows above and 5
    /// three rows above.
    large: Vec<Cell>,
    /// Bytes before, at and after the position one, two and three rows up.
    rows: [u32; 3],
    /// The two bytes before the current one.
    left: u16,
    enabled: bool,
    selected: Option<[usize; FEATURES]>,
}

/// `2 * reach + 1` pixels of a three-byte row window centred above the bit
/// at `depth` of the current byte.
#[inline]
fn pixels(window: u32, depth: u32, reach: u32) -> usize {
    ((window >> (15 - depth - reach)) & ((1 << (2 * reach + 1)) - 1)) as usize
}

impl Neighbors {
    pub(in crate::core::cm) fn new() -> Result<Self, BatoError> {
        let fresh = Cell::new(logistic::HALF, 0);
        Ok(Self {
            cells: [fresh; CELLS],
            wide: try_filled(TEMPLATE_CELLS, fresh)?,
            tall: try_filled(TEMPLATE_CELLS, fresh)?,
            large: try_filled(LARGE_CELLS, fresh)?,
            rows: [0; 3],
            left: 0,
            enabled: false,
            selected: None,
        })
    }

    pub(in crate::core::cm) fn begin_byte(&mut self, history: &[u8], period: Option<usize>) {
        self.enabled = false;
        self.selected = None;
        let Some(stride) = period.filter(|&s| s >= 2 && s <= history.len() / 2) else {
            return;
        };
        let n = history.len();
        // stride >= 2 keeps even the above-right byte strictly in the past.
        // Rows that are not complete yet read as zero bytes.
        let byte = |back: usize, forward: usize| -> u32 {
            (n + forward)
                .checked_sub(back)
                .map_or(0, |i| u32::from(history[i]))
        };
        self.rows = [1, 2, 3].map(|row| {
            let back = row * stride;
            byte(back + 1, 0) << 16 | byte(back, 0) << 8 | byte(back, 1)
        });
        self.left = u16::from(history[n - 2]) << 8 | u16::from(history[n - 1]);
        self.enabled = true;
    }

    #[inline]
    pub(in crate::core::cm) fn feature(&mut self, prefix: u16) -> [i16; FEATURES] {
        if !self.enabled {
            return [0; FEATURES];
        }
        debug_assert!((1..256).contains(&prefix));
        let depth = coded_bits(prefix);
        let partial = u32::from(prefix ^ (1 << depth));
        let left = ((u32::from(self.left) << depth) | partial) as usize;
        let [above, far, third] = self.rows;
        let small = (left & 7)
            | pixels(above, depth, 1) << 3
            | pixels(far, depth, 0) << 6
            | (depth as usize) << 7;
        let wide = (left & 15) | pixels(above, depth, 3) << 4 | pixels(far, depth, 2) << 11;
        let tall = (left & 7)
            | pixels(above, depth, 2) << 3
            | pixels(far, depth, 2) << 8
            | pixels(third, depth, 1) << 13;
        let large = hash::mix(
            (left & 255) as u64
                | (pixels(above, depth, 5) as u64) << 8
                | (pixels(far, depth, 4) as u64) << 19
                | (pixels(third, depth, 2) as u64) << 28,
        ) as usize
            & (LARGE_CELLS - 1);
        self.selected = Some([small, wide, tall, large]);
        [
            logistic::stretch(self.cells[small].probability()),
            logistic::stretch(self.wide[wide].probability()),
            logistic::stretch(self.tall[tall].probability()),
            logistic::stretch(self.large[large].probability()),
        ]
    }

    #[inline]
    pub(in crate::core::cm) fn observe(&mut self, bit: u8) {
        if let Some([small, wide, tall, large]) = self.selected {
            self.cells[small].train(bit, LIMIT);
            self.wide[wide].train(bit, LIMIT);
            self.tall[tall].train(bit, LIMIT);
            self.large[large].train(bit, LIMIT);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_or_unsafe_rows_are_dormant_and_every_prefix_is_bounded() {
        let history: Vec<_> = (0..200).map(|i| (i * 37) as u8).collect();
        let mut model = Neighbors::new().unwrap();
        for period in [None, Some(0), Some(1), Some(101), Some(usize::MAX)] {
            model.begin_byte(&history, period);
            assert_eq!(model.feature(1), [0; FEATURES]);
            assert!(model.selected.is_none());
        }
        for period in [2, 17, 100] {
            model.begin_byte(&history, Some(period));
            for prefix in 1..256 {
                assert_eq!(model.feature(prefix), [0; FEATURES]);
                let [small, wide, tall, large] = model.selected.unwrap();
                assert!(small < CELLS && wide < TEMPLATE_CELLS && tall < TEMPLATE_CELLS);
                assert!(large < LARGE_CELLS);
            }
        }
    }

    #[test]
    fn neighborhood_cells_learn_opposite_bits_without_sharing_contexts() {
        let mut model = Neighbors::new().unwrap();
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
            let [logit, ..] = model.feature(1);
            assert!(if byte == 0 {
                logit > 2000
            } else {
                logit < -2000
            });
        }
        model.begin_byte(&[0; 34], Some(17));
        assert!(model.feature(1)[0] > 2000);
    }
}
