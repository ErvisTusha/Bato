//! Original nibble context map feeding complementary learned mixer inputs.
//! Provenance: indirect bit-history modeling described in Mahoney's Data
//! Compression Explained. Context composition and fixed-point features are
//! original and do not inspect file signatures or names.

use super::{
    adapt::StateMap,
    hash, logistic, state,
    table::{Handle, Table},
};
use std::sync::OnceLock;

/// Mixer inputs produced per context: adaptive and initial state
/// probabilities, last-bit confidence and a linear probability term.
pub(super) const FEATURES: usize = 4;
/// Observations beyond which the last-bit confidence stops growing.
const CONFIDENCE_CAP: i16 = 32;
/// Logit added to the last-bit confidence per observation.
const CONFIDENCE_STEP: i16 = 32;
/// Right shift scaling `probability - 1/2` into the input range.
const LINEAR_SHIFT: u32 = 4;
/// Bits coded per nibble; the table slot changes at each nibble boundary.
pub(super) const NIBBLE_BITS: usize = 4;

// These two mixer inputs depend only on a 256-state history, not on the
// current data. Compute them once instead of redoing logit conversion for
// every context on every encoded or decoded bit.
static FIXED_FEATURES: OnceLock<[[i16; 2]; 256]> = OnceLock::new();

#[derive(Clone)]
pub(super) struct ContextMap {
    maps: [StateMap; 8],
    fixed_features: &'static [[i16; 2]; 256],
    handle: Handle,
    node: usize,
    state: u8,
    depth: usize,
}
impl ContextMap {
    pub(super) fn new() -> Self {
        Self {
            maps: std::array::from_fn(|_| StateMap::new()),
            fixed_features: FIXED_FEATURES.get_or_init(|| {
                std::array::from_fn(|index| {
                    let h = state::HISTORIES[index];
                    let strength =
                        i16::from(h.zeros + h.ones).min(CONFIDENCE_CAP) * CONFIDENCE_STEP;
                    let last = match h.last {
                        0 => strength,
                        1 => -strength,
                        _ => 0,
                    };
                    [
                        logistic::stretch(state::initial_probability(index as u8)),
                        last,
                    ]
                })
            }),
            handle: Handle::default(),
            node: 0,
            state: 0,
            depth: 0,
        }
    }
    #[inline]
    pub(super) fn predict(
        &mut self,
        table: &mut Table,
        base: u64,
        prefix: u16,
        bit_depth: usize,
    ) -> [i16; FEATURES] {
        self.depth = bit_depth;
        let depth = bit_depth % NIBBLE_BITS;
        if depth == 0 {
            self.handle = table.find(hash::mix(base ^ (u64::from(prefix) << 32)));
        }
        self.node = ((1usize << depth) | (usize::from(prefix) & ((1 << depth) - 1))) - 1;
        self.state = table.history(self.handle, self.node);
        let probability = self.maps[self.depth].probability(self.state);
        let [initial, last] = self.fixed_features[usize::from(self.state)];
        [
            logistic::stretch(probability),
            initial,
            last,
            ((i32::from(probability) - i32::from(logistic::HALF)) >> LINEAR_SHIFT) as i16,
        ]
    }
    #[inline]
    pub(super) fn hit(&self) -> bool {
        self.state != 0
    }
    #[inline]
    pub(super) fn history_state(&self) -> u8 {
        self.state
    }
    #[inline]
    pub(super) fn observe(&mut self, table: &mut Table, bit: u8) {
        self.maps[self.depth].observe(self.state, bit);
        table.observe(self.handle, self.node, bit);
    }
}
