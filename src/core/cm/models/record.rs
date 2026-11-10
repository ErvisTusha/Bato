//! Original causal record-stride detection for arbitrary byte streams.
//! Provenance: periodic record modeling is described in Mahoney's Data
//! Compression Explained. The transition-distance voting, confirmation rule
//! and contexts here are independently designed; no file parser is used.

use super::super::hash;

/// Smallest record period considered, in bytes.
const MIN_PERIOD: u32 = 8;
/// Largest record period voted for, in bytes.
const MAX_PERIOD: u32 = 65536;
/// Vote slots; a power of two indexed by a period hash.
const VOTE_SLOTS: usize = 64;
/// Bytes between halvings of every vote count.
const DECAY_INTERVAL: u32 = 4096;
/// Bytes between re-selections of the active period.
const SELECT_INTERVAL: usize = 1024;
/// Votes a period needs before it is scored.
const MIN_VOTES: u16 = 3;
/// Recent bytes scored when choosing a period.
const SCORE_WINDOW: usize = 4096;
/// Byte transitions the window needs before any period is selected.
const MIN_TRANSITIONS: i32 = 64;
/// Score weight of a transition repeated one period back.
const MATCH_WEIGHT: i32 = 3;
/// Domain separator for record context hashes.
const SALT: u64 = 0x693a_c7e1_8d42_b509;

#[derive(Clone, Copy)]
struct Recent([u32; 3]);
impl Default for Recent {
    fn default() -> Self {
        Self([u32::MAX; 3])
    }
}
impl Recent {
    /// The gap to `position` when the byte's last three transitions were
    /// evenly spaced by a plausible record period.
    fn regular_gap(&self, position: u32) -> Option<u32> {
        let [a, b, c] = self.0;
        if c == u32::MAX {
            return None;
        }
        let gap = position - a;
        let regular = a - b == gap && b - c == gap;
        ((MIN_PERIOD..=MAX_PERIOD).contains(&gap) && regular).then_some(gap)
    }
    /// Records a transition into the byte at `position`.
    fn push(&mut self, position: u32) {
        self.0 = [position, self.0[0], self.0[1]];
    }
}
#[derive(Clone, Copy, Default)]
struct Vote {
    period: u32,
    count: u16,
}
impl Vote {
    /// Majority-vote update: reinforce a matching period, claim an empty
    /// slot, or weaken the current holder.
    fn cast(&mut self, gap: u32) {
        if self.period == gap {
            self.count = self.count.saturating_add(1);
        } else if self.count == 0 {
            *self = Vote {
                period: gap,
                count: 1,
            };
        } else {
            self.count -= 1;
        }
    }
}
pub(in crate::core::cm) struct Records {
    recent: [Recent; 256],
    votes: [Vote; VOTE_SLOTS],
    position: u32,
    previous: Option<u8>,
    period: usize,
}
impl Default for Records {
    fn default() -> Self {
        Self {
            recent: [Recent::default(); 256],
            votes: [Vote::default(); VOTE_SLOTS],
            position: 0,
            previous: None,
            period: 0,
        }
    }
}
impl Records {
    pub(in crate::core::cm) fn period(&self) -> Option<usize> {
        (self.period != 0).then_some(self.period)
    }
    pub(in crate::core::cm) fn observe_byte(&mut self, byte: u8) {
        if self.previous != Some(byte) {
            let recent = &mut self.recent[usize::from(byte)];
            if let Some(gap) = recent.regular_gap(self.position) {
                self.votes[(hash::mix(u64::from(gap)) as usize) % VOTE_SLOTS].cast(gap);
            }
            recent.push(self.position);
        }
        self.previous = Some(byte);
        self.position += 1;
        if self.position.is_multiple_of(DECAY_INTERVAL) {
            for slot in &mut self.votes {
                slot.count >>= 1;
            }
        }
    }

    pub(in crate::core::cm) fn begin_byte(&mut self, history: &[u8]) -> Option<u64> {
        debug_assert_eq!(history.len(), self.position as usize);
        if history.len().is_multiple_of(SELECT_INTERVAL) {
            self.period = self.select_period(history);
        }
        if self.period == 0 || history.len() < self.period {
            return None;
        }
        let above = history[history.len() - self.period];
        let left = history.last().copied().unwrap_or(0);
        let column = history.len() % self.period;
        Some(hash::mix(
            ((self.period as u64) << 32)
                ^ ((column as u64) << 16)
                ^ (u64::from(above) << 8)
                ^ u64::from(left)
                ^ SALT,
        ))
    }

    /// The best-scoring voted or current period, or 0 when none qualifies.
    /// Ties prefer the shorter period.
    fn select_period(&self, history: &[u8]) -> usize {
        let candidates = self
            .votes
            .iter()
            .filter(|slot| slot.count >= MIN_VOTES)
            .map(|slot| slot.period as usize)
            .chain(std::iter::once(self.period));
        let mut selected = 0;
        let mut best_score = 0;
        for period in candidates {
            let Some(score) = period_score(history, period) else {
                continue;
            };
            let better =
                score > best_score || (score == best_score && (selected == 0 || period < selected));
            if better {
                best_score = score;
                selected = period;
            }
        }
        selected
    }
}

/// How well `period` predicts the recent byte transitions, or `None` when the
/// period is out of range, the window has too few transitions, or the score
/// is not positive.
fn period_score(history: &[u8], period: usize) -> Option<i32> {
    if period < MIN_PERIOD as usize || period >= history.len() {
        return None;
    }
    let start = history.len().saturating_sub(SCORE_WINDOW).max(period + 1);
    let mut transitions = 0;
    let mut equal = 0;
    for at in start..history.len() {
        if history[at] != history[at - 1] {
            transitions += 1;
            equal += i32::from(history[at] == history[at - period]);
        }
    }
    let score = MATCH_WEIGHT * equal - transitions;
    (transitions >= MIN_TRANSITIONS && score > 0).then_some(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(model: &mut Records, bytes: &[u8], history: &mut Vec<u8>) {
        for &byte in bytes {
            model.begin_byte(history);
            model.observe_byte(byte);
            history.push(byte);
        }
    }
    fn rows(period: usize, count: usize) -> Vec<u8> {
        (0..period * count)
            .map(|at| ((at % period) * 37 % 251) as u8)
            .collect()
    }
    #[test]
    fn inferred_period_is_causal_and_adapts_to_new_records() {
        let mut model = Records::default();
        let mut history = Vec::new();
        feed(&mut model, &rows(97, 90), &mut history);
        assert_eq!(model.period, 97);
        assert!(model.begin_byte(&history).is_some());
        feed(&mut model, &rows(137, 100), &mut history);
        assert_eq!(model.period, 137);
    }
    #[test]
    fn constant_runs_and_unstructured_bytes_do_not_activate_records() {
        let mut model = Records::default();
        let mut history = Vec::new();
        feed(&mut model, &[0xff; 10000], &mut history);
        assert_eq!(model.period, 0);
        let mut seed = 0x1b73_9165u32;
        for _ in 0..100000 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            assert!(model.begin_byte(&history).is_none());
            model.observe_byte(seed as u8);
            history.push(seed as u8);
        }
    }
}
