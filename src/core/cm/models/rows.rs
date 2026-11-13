//! Original two-dimensional neighbour model: finds the byte distance between
//! vertically adjacent samples (an image row, a plane line) and the distance
//! between horizontally adjacent samples of one channel, from the data's own
//! autocorrelation, then offers contexts built from the left, upper and
//! upper-left neighbours.
//! Provenance: neighbour prediction as used in lossless image coders
//! (median/gradient predictors; Mahoney, Data Compression Explained, image
//! models); detection and contexts are original.

use super::super::hash;

/// Row distances scanned, in bytes.
const MIN_ROW: usize = 16;
const MAX_ROW: usize = 16384;
/// Sample distances (bytes per sample group) tried for the left neighbour.
const MAX_UNIT: usize = 4;
/// Bytes between scans, bytes sampled per distance, and the stride between
/// sampled positions.
const SCAN_INTERVAL: usize = 32768;
const SAMPLES: usize = 512;
const SAMPLE_STEP: usize = 61;
/// A row distance is used only if its mean difference is under this share
/// (in 1/16) of the median over all scanned distances.
const MAX_RELATIVE_SIXTEENTHS: usize = 9;
const SALT: u64 = 0x5e17_c3a9_0b42_d6f8;

#[derive(Clone, Default)]
pub(in crate::core::cm) struct Rows {
    row: Option<usize>,
    unit: usize,
    next_scan: usize,
}

/// Sum of absolute differences between sampled bytes and the bytes `lag`
/// earlier, over the last `SAMPLES * SAMPLE_STEP` bytes.
fn difference(history: &[u8], lag: usize) -> usize {
    let end = history.len();
    (0..SAMPLES)
        .map(|k| end - 1 - k * SAMPLE_STEP)
        .map(|i| usize::from(history[i].abs_diff(history[i - lag])))
        .sum()
}

/// Samples summed in 16-bit lanes before widening: 256 differences of at
/// most 255 cannot overflow a `u16`.
const BLOCK_SAMPLES: usize = 256;

/// `difference` for every row distance in `MIN_ROW..=MAX_ROW` at once, indexed
/// by `MAX_ROW - lag`. Walking the earlier bytes of each sample contiguously
/// and summing in 16-bit lanes lets the compiler vectorise the sums sixteen
/// bytes at a time; the totals equal `difference`'s.
fn row_differences(history: &[u8]) -> (Vec<u32>, Vec<u16>) {
    let end = history.len();
    let mut sums = vec![0u32; MAX_ROW - MIN_ROW + 1];
    let mut block = vec![0u16; MAX_ROW - MIN_ROW + 1];
    // Samples equal to the byte one distance back; at most `SAMPLES`.
    let mut equal = vec![0u16; MAX_ROW - MIN_ROW + 1];
    for first in (0..SAMPLES).step_by(BLOCK_SAMPLES) {
        block.fill(0);
        for k in first..(first + BLOCK_SAMPLES).min(SAMPLES) {
            let i = end - 1 - k * SAMPLE_STEP;
            let current = history[i];
            // Positions i - MAX_ROW ..= i - MIN_ROW, i.e. lags MAX_ROW down to MIN_ROW.
            let earlier = &history[i - MAX_ROW..=i - MIN_ROW];
            for (sum, &byte) in block.iter_mut().zip(earlier) {
                *sum += u16::from(current.abs_diff(byte));
            }
            for (count, &byte) in equal.iter_mut().zip(earlier) {
                *count += u16::from(current == byte);
            }
        }
        for (sum, &part) in sums.iter_mut().zip(&block) {
            *sum += u32::from(part);
        }
    }
    (sums, equal)
}

/// Fewest equal samples (of `SAMPLES`) for a record period.
const MIN_EQUAL: u16 = 32;
/// A record period's equal samples must exceed this multiple of the median
/// distance's.
const EQUAL_RATIO: u16 = 8;

/// The record period of data whose bytes often equal the byte one record
/// back although their differences are not small (numeric tables): the
/// shortest distance with nearly the most equal samples, when that count
/// stands far above the typical distance's. `equal` is indexed by
/// `MAX_ROW - distance`.
fn record_period(equal: &[u16]) -> Option<usize> {
    let best = equal.iter().copied().max()?;
    let mut sorted = equal.to_vec();
    let middle = sorted.len() / 2;
    let median = *sorted.select_nth_unstable(middle).1;
    if best < MIN_EQUAL || best <= median.saturating_mul(EQUAL_RATIO) {
        return None;
    }
    // Multiples of the period score alike; the shortest is the period.
    (MIN_ROW..=MAX_ROW).find(|lag| u32::from(equal[MAX_ROW - lag]) * 16 >= u32::from(best) * 15)
}

/// A signed difference reduced to 4 bits, finer near zero.
fn quantise(difference: i16) -> u64 {
    let magnitude = difference.unsigned_abs();
    let level = match magnitude {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=8 => 4,
        9..=16 => 5,
        17..=40 => 6,
        _ => 7,
    };
    u64::from(level as u8) | u64::from(difference < 0) << 3
}

impl Rows {
    fn scan(&mut self, history: &[u8]) {
        let span = SAMPLES * SAMPLE_STEP;
        if history.len() < MAX_ROW + span {
            return;
        }
        // Samples of one pixel sit `unit` bytes apart; a row is a whole number
        // of them, which separates a row from its off-by-one neighbours.
        self.unit = (1..=MAX_UNIT)
            .map(|lag| (difference(history, lag), lag))
            .min()
            .map_or(1, |(_, lag)| lag);
        let (sums, equal) = row_differences(history);
        let mut scores: Vec<(usize, usize)> = (MIN_ROW..=MAX_ROW)
            .filter(|lag| lag % self.unit == 0)
            .map(|lag| (sums[MAX_ROW - lag] as usize, lag))
            .collect();
        let best = scores.iter().copied().min();
        let middle = scores.len() / 2;
        let median = scores.select_nth_unstable(middle).1 .0;
        self.row = best
            .filter(|&(score, _)| score * 16 <= median * MAX_RELATIVE_SIXTEENTHS)
            .map(|(_, lag)| lag);
        if self.row.is_none() {
            self.row = record_period(&equal);
            self.unit = self.row.map_or(self.unit, |_| 1);
        }
    }

    pub(in crate::core::cm) fn begin_byte(&mut self, history: &[u8]) {
        if history.len() >= self.next_scan {
            self.next_scan = history.len() + SCAN_INTERVAL;
            self.scan(history);
        }
    }

    /// The row distance, when one is known.
    pub(in crate::core::cm) fn period(&self) -> Option<usize> {
        self.row
    }

    /// Whether a row distance is known.
    pub(in crate::core::cm) fn active(&self) -> bool {
        self.row.is_some()
    }

    /// Four context hashes from the left, upper and upper-left neighbours;
    /// constant when no row distance is known.
    pub(in crate::core::cm) fn contexts(&self, history: &[u8]) -> [u64; 8] {
        let end = history.len();
        let Some(row) = self.row.filter(|&row| end > row + self.unit + 1) else {
            return [hash::mix(SALT); 8];
        };
        let unit = self.unit;
        let (left, above) = (history[end - unit], history[end - row]);
        let above_left = history[end - row - unit];
        let (previous, previous_above) = (history[end - 1], history[end - row - 1]);
        let above_right = history[end - row + unit.min(row - 1)];
        let gradient = (i16::from(left) + i16::from(above) - i16::from(above_left)).clamp(0, 255);
        let key = |a: u64, b: u64, salt: u64| hash::mix(a | b << 16 | salt);
        [
            key(u64::from(left), u64::from(above), SALT ^ 1 << 40),
            key(gradient as u64, 0, SALT ^ 2 << 40),
            key(
                u64::from(above),
                u64::from(above_right) | u64::from(left >> 2) << 8,
                SALT ^ 3 << 40,
            ),
            key(
                (i16::from(left) - i16::from(above_left)).clamp(-32, 32) as u64 & 0xff,
                u64::from(above >> 1),
                SALT ^ 4 << 40,
            ),
            // Quantised neighbour differences: robust against sensor noise.
            key(
                (gradient >> 2) as u64,
                quantise(i16::from(left) - i16::from(above))
                    | quantise(i16::from(above) - i16::from(above_right)) << 4,
                SALT ^ 5 << 40,
            ),
            key(u64::from(left), u64::from(above_left), SALT ^ 6 << 40),
            key(
                u64::from((u16::from(left) + u16::from(above)) >> 1),
                quantise(i16::from(above_right) - i16::from(above_left)),
                SALT ^ 7 << 40,
            ),
            key(
                u64::from(above >> 2),
                (i16::from(previous) - i16::from(previous_above)).clamp(-16, 16) as u64 & 0xff
                    | u64::from(left >> 2) << 8,
                SALT ^ 8 << 40,
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth synthetic image, `row` bytes wide, with small noise.
    fn image(row: usize, rows: usize) -> Vec<u8> {
        (0..row * rows)
            .map(|i| {
                let (x, y) = (i % row, i / row);
                ((x / 16 + y / 4) as u8).wrapping_add((hash::mix(i as u64) >> 50) as u8 % 3)
            })
            .collect()
    }

    #[test]
    fn the_row_distance_of_a_smooth_image_is_found() {
        let data = image(1500, 120);
        let mut rows = Rows::default();
        for end in 0..data.len() {
            rows.begin_byte(&data[..end]);
        }
        let plain = rows.contexts(&data[..10]);
        assert!(
            plain.iter().all(|&c| c == plain[0]),
            "constant on short input"
        );
        // Without colour channels any neighbouring distance fits equally.
        assert!(matches!(rows.row, Some(1499..=1501)), "{:?}", rows.row);
        let found = rows.contexts(&data);
        assert!(found.iter().any(|&c| c != found[0]));
    }

    #[test]
    fn random_bytes_and_short_inputs_get_no_row() {
        let noise: Vec<u8> = (0..200_000u64)
            .map(|i| (hash::mix(i) >> 24) as u8)
            .collect();
        let mut rows = Rows::default();
        for end in (0..noise.len()).step_by(997) {
            rows.begin_byte(&noise[..end]);
        }
        assert_eq!(rows.row, None);
        let mut short = Rows::default();
        short.begin_byte(&[1, 2, 3]);
        assert_eq!(short.row, None);
    }
}
