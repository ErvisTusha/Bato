//! Original large-stride predictor: finds a lag at which the data repeats
//! approximately (a video frame, an image plane, a table of fixed-size
//! records) from the distances of long matches, then offers contexts built
//! from the bytes one lag back.
//! Provenance: modeling of two-dimensional and temporal neighbours as in
//! lossless image and video coders (Mahoney, Data Compression Explained,
//! on record and image models); detection and contexts are original.

use super::super::hash;

/// Smallest and largest lag considered, in bytes.
const MIN_LAG: usize = 64;
const MAX_LAG: usize = 1 << 21;
const VOTE_SLOTS: usize = 16;
/// Bytes between lag evaluations, and the window each evaluation scores.
const EVALUATE_INTERVAL: usize = 4096;
const SCORE_WINDOW: usize = 2048;
/// Two bytes count as alike when they differ by at most this much.
const ALIKE: u8 = 2;
/// Votes a lag needs before it is evaluated, and the share of the window
/// (in 1/16) that must be alike for a lag to be used.
const MIN_VOTES: u16 = 16;
const MIN_ALIKE_SIXTEENTHS: usize = 8;
/// Divisors of the voted lag also tried, since long matches often reach back
/// several periods.
const DIVISORS: [usize; 6] = [1, 2, 3, 4, 6, 8];
/// Lags of the most voted distances that are scored each evaluation.
const TOP_VOTES: usize = 4;
/// A later (larger) lag must beat the best score by 1/this to replace it.
const TIE_DIVISOR: usize = 32;
const SALT: u64 = 0x2f5c_91d3_a78e_4b06;

#[derive(Clone, Copy, Default)]
struct Vote {
    lag: usize,
    count: u16,
}

#[derive(Default)]
pub(in crate::core::cm) struct Stride {
    votes: [Vote; VOTE_SLOTS],
    lag: Option<usize>,
    next_evaluation: usize,
}

impl Stride {
    fn cast(&mut self, lag: usize) {
        if let Some(vote) = self.votes.iter_mut().find(|v| v.count > 0 && v.lag == lag) {
            vote.count = vote.count.saturating_add(1);
        } else if let Some(vote) = self.votes.iter_mut().find(|v| v.count == 0) {
            *vote = Vote { lag, count: 1 };
        } else {
            for vote in &mut self.votes {
                vote.count -= 1;
            }
        }
    }

    /// Number of the last `SCORE_WINDOW` bytes within `ALIKE` of the byte
    /// `lag` earlier.
    fn alike(history: &[u8], lag: usize) -> usize {
        let end = history.len();
        (end - SCORE_WINDOW..end)
            .filter(|&i| history[i].abs_diff(history[i - lag]) <= ALIKE)
            .count()
    }

    fn evaluate(&mut self, history: &[u8]) {
        let mut voted = self.votes;
        voted.sort_by_key(|v| std::cmp::Reverse(v.count));
        for vote in &mut self.votes {
            vote.count /= 2;
        }
        let usable = |lag: usize| lag >= MIN_LAG && history.len() >= lag + SCORE_WINDOW;
        let mut candidates: Vec<usize> = self.lag.into_iter().collect();
        for vote in voted
            .iter()
            .take(TOP_VOTES)
            .filter(|v| v.count >= MIN_VOTES)
        {
            for divisor in DIVISORS {
                if vote.lag % divisor == 0 {
                    candidates.push(vote.lag / divisor);
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        // Highest score wins; a smaller lag wins a near tie.
        let mut best: Option<(usize, usize)> = None;
        for lag in candidates.into_iter().filter(|&lag| usable(lag)) {
            let score = Self::alike(history, lag);
            if best.is_none_or(|(_, s)| score > s + s / TIE_DIVISOR) {
                best = Some((lag, score));
            }
        }
        self.lag = best
            .filter(|&(_, score)| score * 16 >= SCORE_WINDOW * MIN_ALIKE_SIXTEENTHS)
            .map(|(lag, _)| lag);
    }

    /// Called at each byte boundary with the history and the distance of a
    /// current long match, if any.
    pub(in crate::core::cm) fn begin_byte(
        &mut self,
        history: &[u8],
        match_distance: Option<usize>,
    ) {
        if let Some(distance) = match_distance.filter(|d| (MIN_LAG..=MAX_LAG).contains(d)) {
            self.cast(distance);
        }
        if history.len() >= self.next_evaluation {
            self.next_evaluation = history.len() + EVALUATE_INTERVAL;
            self.evaluate(history);
        }
    }

    /// Whether a lag is known.
    pub(in crate::core::cm) fn active(&self) -> bool {
        self.lag.is_some()
    }

    /// Four context hashes from the bytes one lag back and their left
    /// neighbours; constant when no lag is known.
    pub(in crate::core::cm) fn contexts(&self, history: &[u8]) -> [u64; 4] {
        let Some(lag) = self.lag.filter(|&lag| history.len() > lag + 1) else {
            return [hash::mix(SALT); 4];
        };
        let end = history.len();
        let (above, left, left_above) =
            (history[end - lag], history[end - 1], history[end - lag - 1]);
        let below = history.get(end - lag + 1).copied().unwrap_or(0);
        let expected = (i16::from(above) + i16::from(left) - i16::from(left_above)).clamp(0, 255);
        let step = i16::from(left) - i16::from(left_above);
        let key = |a: u64, b: u64, salt: u64| hash::mix(a | b << 16 | salt);
        [
            key(u64::from(above), u64::from(left), SALT ^ 1 << 40),
            key(
                u64::from(above),
                u64::from(below) | u64::from(left_above) << 8,
                SALT ^ 2 << 40,
            ),
            key(expected as u64, 0, SALT ^ 3 << 40),
            key(
                u64::from(above),
                (step.clamp(-24, 24) + 24) as u64,
                SALT ^ 4 << 40,
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A noisy frame sequence: `lag` bytes per frame, each repeated with
    /// small deterministic noise.
    fn frames(lag: usize, count: usize) -> Vec<u8> {
        let base: Vec<u8> = (0..lag)
            .map(|i| (hash::mix(i as u64) >> 20) as u8)
            .collect();
        (0..lag * count)
            .map(|i| base[i % lag].wrapping_add((hash::mix(i as u64 ^ 77) >> 40) as u8 % 3))
            .collect()
    }

    #[test]
    fn a_voted_lag_is_found_and_changes_the_contexts() {
        let lag = 1000;
        let data = frames(lag, 12);
        let mut stride = Stride::default();
        let none = stride.contexts(&data);
        assert!(none.iter().all(|&c| c == none[0]), "constant without a lag");
        for end in 0..data.len() {
            stride.begin_byte(&data[..end], (end >= lag).then_some(lag));
        }
        let found = stride.contexts(&data);
        assert_ne!(found, none);
        assert_eq!(stride.lag, Some(lag));
    }

    #[test]
    fn unrelated_data_never_gets_a_lag() {
        let data: Vec<u8> = (0..60_000u64).map(|i| (hash::mix(i) >> 24) as u8).collect();
        let mut stride = Stride::default();
        for end in 0..data.len() {
            stride.begin_byte(&data[..end], (end > 5000).then_some(5000));
        }
        assert_eq!(stride.lag, None);
    }

    #[test]
    fn short_histories_and_extreme_votes_are_safe() {
        let mut stride = Stride::default();
        for end in 0..200 {
            stride.begin_byte(&[7u8; 200][..end], Some(usize::MAX));
        }
        assert_eq!(stride.contexts(&[1, 2, 3]).len(), 4);
        stride.lag = Some(10);
        assert_eq!(stride.contexts(&[1, 2, 3])[0], stride.contexts(&[9])[0]);
    }
}
