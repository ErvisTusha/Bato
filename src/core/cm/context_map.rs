//! Original nibble context map feeding complementary learned mixer inputs.
//! Provenance: indirect bit-history modeling described in Mahoney's Data
//! Compression Explained. Context composition and fixed-point features are
//! original and do not inspect file signatures or names.

use super::{
    adapt::StateMap,
    logistic, state,
    table::{Handle, Table},
};

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

const fn build_fixed_features() -> [[i16; 2]; 256] {
    let stretch = logistic::stretch_table();
    let mut features = [[0i16; 2]; 256];
    let mut index = 0;
    while index < 256 {
        let h = state::HISTORIES[index];
        let total = (h.zeros + h.ones) as i16;
        let capped = if total < CONFIDENCE_CAP {
            total
        } else {
            CONFIDENCE_CAP
        };
        let strength = capped * CONFIDENCE_STEP;
        let last = match h.last {
            0 => strength,
            1 => -strength,
            _ => 0,
        };
        features[index] = [
            stretch[state::initial_probability(index as u8) as usize],
            last,
        ];
        index += 1;
    }
    features
}

// These two mixer inputs depend only on a 256-state history, not on the
// current data. Compute them at compile time instead of redoing logit
// conversion for every context on every encoded or decoded bit.
static FIXED_FEATURES: [[i16; 2]; 256] = build_fixed_features();

#[derive(Clone)]
#[repr(C)]
pub(super) struct ContextMap {
    handle: Handle,
    state: u8,
    maps: [StateMap; 8],
}
impl ContextMap {
    pub(super) const fn new() -> Self {
        Self {
            handle: Handle::EMPTY,
            state: 0,
            maps: [StateMap::new(); 8],
        }
    }
    #[inline]
    pub(super) fn predict(
        &mut self,
        table: &mut Table,
        mixed_key: u64,
        bit_depth: usize,
        node: usize,
    ) -> [i16; FEATURES] {
        if node == 0 {
            (self.handle, self.state) = table.find(mixed_key);
        } else {
            self.state = table.history(self.handle, node);
        }
        let probability = self.maps[bit_depth].probability(self.state);
        let [initial, last] = FIXED_FEATURES[usize::from(self.state)];
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
    pub(super) fn observe(&mut self, table: &mut Table, bit_depth: usize, node: usize, bit: u8) {
        self.maps[bit_depth].observe(self.state, bit);
        table.observe(self.handle, node, bit);
    }
}
