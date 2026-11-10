//! Original 33-bin interpolating log-odds calibration.
//! Provenance: adaptive secondary probability estimation described in Mahoney,
//! Data Compression Explained. Integer interpolation and weighted adaptation
//! are independent Rust, using the sibling original logistic tables.

use super::{adapt::Cell, logistic};
use crate::core::{memory::try_capacity, types::BatoError};
/// Interpolation bins spanning the stretched range [-LIMIT, LIMIT].
const BINS: usize = 33;
/// log2 of the bin width in logit units: 2 * LIMIT / (BINS - 1) = 256.
const BIN_SHIFT: u32 = 8;
/// Bin width in logit units, also the full interpolation weight.
const BIN_WIDTH: u16 = 1 << BIN_SHIFT;
/// Observations a fresh bin is assumed to have seen.
const INITIAL_COUNT: u16 = 16;
/// Count limit: the bins' adaptation rate settles near 1/512.
const LIMIT: u16 = 512;
pub(super) struct Apm {
    cells: Vec<Cell>,
    mask: usize,
    selected: usize,
    fraction: u16,
}
impl Apm {
    pub(super) fn memory_required(contexts: usize) -> usize {
        contexts * BINS * std::mem::size_of::<Cell>()
    }
    pub(super) fn new(contexts: usize) -> Result<Self, BatoError> {
        if !contexts.is_power_of_two() {
            return Err(BatoError::InvalidOptions(
                "CM APM context count must be a nonzero power of two".into(),
            ));
        }
        let mut cells = try_capacity(
            contexts
                .checked_mul(BINS)
                .ok_or(BatoError::AllocationFailed { bytes: usize::MAX })?,
        )?;
        let row: [Cell; BINS] = std::array::from_fn(|bin| {
            Cell::new(
                logistic::squash(bin as i32 * i32::from(BIN_WIDTH) - logistic::LIMIT),
                INITIAL_COUNT,
            )
        });
        for _ in 0..contexts {
            cells.extend_from_slice(&row);
        }
        Ok(Self {
            cells,
            mask: contexts - 1,
            selected: 0,
            fraction: 0,
        })
    }
    /// Starts loading the bins of `context` into the cache; the row is known
    /// well before the probability that picks the bin within it.
    #[inline]
    // Without x86-64 prefetch the context is legitimately unused.
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
    pub(super) fn prefetch(&self, context: usize) {
        // SAFETY: the masked context names a row of BINS allocated cells, and
        // every offset below stays inside that row (BINS * 4 = 132 bytes);
        // prefetch neither reads values into the model nor writes memory.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let row = self
                .cells
                .as_ptr()
                .add((context & self.mask) * BINS)
                .cast::<i8>();
            // Each row occupies 132 bytes, or three 64-byte cache lines.
            // The former offset 131 shared the line already prefetched at 128.
            for offset in [0, 64, 128] {
                std::arch::x86_64::_mm_prefetch(row.add(offset), std::arch::x86_64::_MM_HINT_T0);
            }
        }
    }
    #[inline]
    pub(super) fn predict(&mut self, probability: u16, context: usize) -> u16 {
        let position = (i32::from(logistic::stretch(probability)) + logistic::LIMIT) as usize;
        let bin = (position >> BIN_SHIFT).min(BINS - 2);
        self.fraction = (position - (bin << BIN_SHIFT)) as u16;
        self.selected = (context & self.mask) * BINS + bin;
        // The interpolation bins are adjacent. Use one checked slice instead
        // of two independent dynamic-index checks on every coded bit.
        let [left, right] = &self.cells[self.selected..self.selected + 2] else {
            unreachable!("APM interpolation always selects two bins");
        };
        let a = u32::from(left.probability());
        let b = u32::from(right.probability());
        let weighted = a * u32::from(BIN_WIDTH - self.fraction) + b * u32::from(self.fraction);
        ((weighted + u32::from(BIN_WIDTH / 2)) >> BIN_SHIFT).clamp(1, u32::from(logistic::MAX))
            as u16
    }
    #[inline]
    pub(super) fn observe(&mut self, bit: u8) {
        let [left, right] = &mut self.cells[self.selected..self.selected + 2] else {
            unreachable!("APM interpolation always selects two bins");
        };
        left.train_weighted(bit, LIMIT, BIN_WIDTH - self.fraction);
        right.train_weighted(bit, LIMIT, self.fraction);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initialization_is_monotone_near_identity_and_learning_is_contextual() {
        let mut map = Apm::new(2).unwrap();
        let mut previous = 0;
        for probability in 1..=65535u16 {
            let calibrated = map.predict(probability, 0);
            assert!(calibrated >= previous);
            previous = calibrated;
            assert!((i32::from(calibrated) - i32::from(probability)).abs() <= 1024);
        }
        for _ in 0..4000 {
            map.predict(32768, 0);
            map.observe(0);
        }
        assert!(map.predict(32768, 0) > 65000);
        assert!((32000..33500).contains(&map.predict(32768, 1)));
        let (_, observed) = crate::core::diagnostics::capture(|| Apm::new(256).unwrap());
        assert_eq!(observed.heap.peak, Apm::memory_required(256));
    }
}
