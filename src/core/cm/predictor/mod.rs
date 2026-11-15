//! Original learned integer context ensemble for segment-persistent coding.
//! Provenance: indirect contexts, neural log-odds mixing and online adaptation
//! described in Mahoney, Data Compression Explained and the public-domain
//! ZPAQ specification. All model composition, scaling and Rust are original.
//!
//! The model is three stages. `Front` holds every context model and depends
//! only on the data; `Mixing` combines its `Signals`; `Calibration` refines
//! the mixed probability. A decoder runs the stages in turn for each bit. An
//! encoder knows every bit ahead of time, so `pipeline` can run them on
//! separate threads over batches of bits and produce the same probabilities.

mod calibration;
mod front;
mod mixing;
pub(super) mod pipeline;

use super::{
    apm::BinHalf,
    context_map,
    models::{neighbors, order},
    profile::{Meter, Step, Timing},
    simd::{Dispatch, INPUTS},
};
use crate::core::types::BatoError;
use calibration::Calibration;
use front::Front;
use mixing::Mixing;
pub(super) use pipeline::{Engine, Stages};

const CONTEXTS: usize = order::COUNT + 4;
/// Two auxiliary experts, each four contexts with its own 256-set mixer
/// (selected by the partial byte) whose stretched output feeds the final
/// mixer. Expert 0 models text: orders 5 and 7, word + previous byte, and a
/// three-word context. Expert 1 models sparse structure: bytes 2, 3 and 4
/// back and the 2/3 pair. Kept out of the main networks, where extra inputs
/// measurably hurt text.
const EXPERTS: usize = 4;
const EXPERT_CONTEXTS: usize = 8;
/// Contexts each expert really uses; the row expert has eight.
const EXPERT_SIZES: [usize; EXPERTS] = [4, 4, 4, 8];
const EXPERT_SETS: usize = 256;
const MIX_CONTEXTS: [usize; 3] = [256, 2048, 128];
const FINAL_CONTEXTS: usize = 64;
/// Calibration stages, each with this many contexts (powers of two).
const APM_CONTEXTS: [usize; APM_STAGES] = [256, 1024, 65536, 65536, 65536, 65536, 65536, 65536];
const APM_STAGES: usize = 8;
const ISSE_MAPS: [usize; 3] = [2, 4, 5];
/// Features per context map.
const FEATURES: usize = context_map::FEATURES;
/// First-layer input slots after the context features: order-0, run, two
/// match features and three record features, in that order.
const ORDER0_INPUT: usize = CONTEXTS * FEATURES;
const RUN_INPUT: usize = ORDER0_INPUT + 1;
const MATCH_INPUTS: usize = ORDER0_INPUT + 2;
const RECORD_INPUTS: usize = ORDER0_INPUT + 4;
/// First-layer inputs the mixers read, rounded up to whole 16-lane vectors;
/// the last one holds the constant bias.
const ACTIVE_INPUTS: usize = (ORDER0_INPUT + 5).div_ceil(16) * 16;
/// Inputs each expert mixer reads (four per context plus the bias, whole 16-lane
/// vectors); the last one holds the bias.
const EXPERT_ACTIVE_INPUTS: [usize; EXPERTS] = [32, 32, 32, 48];
/// Final-mixer input slots after the three first-layer mixers.
const NEIGHBOR_INPUT: usize = 3;
const EXPERT_INPUT: usize = 4;
/// Final-mixer slots of the bit-neighbourhood templates, after the experts.
const TEMPLATE_INPUT: usize = EXPERT_INPUT + EXPERTS;
/// Final-mixer inputs the mixer reads; the last one holds the bias.
const FINAL_ACTIVE_INPUTS: usize = 16;
/// Initial weights in units of 1/256: the context features' first input
/// shares 0.75 across contexts, order-0 starts at 0.25, each first-layer
/// mixer at about 1/3 in the final mixer, and each expert context at 0.1875.
const CONTEXT_WEIGHT: i16 = (192 / CONTEXTS) as i16;
const ORDER0_WEIGHT: i16 = 64;
const FINAL_WEIGHT: i16 = 85;
const EXPERT_WEIGHT: i16 = 48;
/// Learning-rate shifts (larger is slower) of the first-layer and expert
/// mixers, and of the final mixer.
const MIXER_RATE: u8 = 23;
const FINAL_RATE: u8 = 24;
/// Count limit of the order-0 cells: their rate settles near 1/256.
const ORDER0_LIMIT: u16 = 256;
/// Value of a missing (before-start) byte in sparse contexts: outside 0..=255.
/// Match length from which its distance votes for a data lag.
const STRIDE_VOTE_LENGTH: u16 = 8;
const MISSING_BYTE: u64 = 0x1ff;
/// Domain separator for the sparse expert's context hashes.
const SPARSE_SALT: u64 = 0x7a3c_91e4_0000_0000;
/// Hashed byte orders of the text expert.
const TEXT_ORDERS: [usize; 2] = [5, 7];

/// Everything the mixing stage needs for one bit, produced by `Front`.
#[derive(Clone)]
struct Signals {
    /// First-layer inputs: context features, order-0, run, match and record.
    inputs: [i16; INPUTS],
    expert_inputs: [[i16; INPUTS]; EXPERTS],
    expert_active: [bool; EXPERTS],
    neighbors: [i16; neighbors::FEATURES],
    /// Weight sets of the three first-layer mixers; the first also selects
    /// the experts' mixers.
    selectors: [usize; 3],
    final_selectors: [usize; 2],
    isse_states: [u8; ISSE_MAPS.len()],
    calibration: [usize; APM_STAGES],
}

impl Signals {
    fn new() -> Self {
        Self {
            inputs: [0; INPUTS],
            expert_inputs: [[0; INPUTS]; EXPERTS],
            expert_active: [true; EXPERTS],
            neighbors: [0; neighbors::FEATURES],
            selectors: [0; 3],
            final_selectors: [0; 2],
            isse_states: [0; ISSE_MAPS.len()],
            calibration: [0; APM_STAGES],
        }
    }
}

/// The mixing stage's output for one bit, with what calibration still needs.
#[derive(Clone, Copy)]
struct Mixed {
    prediction: u16,
    half: BinHalf,
    isse_states: [u8; ISSE_MAPS.len()],
    calibration: [usize; APM_STAGES],
}

impl Mixed {
    fn new() -> Self {
        Self {
            prediction: 0,
            half: BinHalf::Low,
            isse_states: [0; ISSE_MAPS.len()],
            calibration: [0; APM_STAGES],
        }
    }
}

/// Attributes the time since the previous lap to `step` while timing.
#[inline(always)]
fn lap(meter: &mut Option<Box<Meter>>, step: Step) {
    if let Some(meter) = meter {
        meter.lap(step);
    }
}

/// The whole model run on one thread, bit by bit.
#[derive(Clone)]
pub(super) struct Predictor {
    front: Front,
    mixing: Mixing,
    calibration: Calibration,
    signals: Signals,
    dispatch: Dispatch,
    /// Step timer, present only while timing is requested.
    meter: Option<Box<Meter>>,
}

pub(super) fn memory_required(length: usize, table_log2: u8) -> usize {
    front::memory_required(length, table_log2)
        + mixing::memory_required()
        + calibration::memory_required()
}

impl Predictor {
    /// Starts timing the steps of every following bit.
    pub(super) fn start_meter(&mut self) {
        self.meter = Some(Box::new(Meter::new()));
    }
    /// Hands the timing gathered since `start_meter` to `timing`.
    pub(super) fn finish_meter(&mut self, timing: Timing<'_>) {
        if let Some(meter) = self.meter.take() {
            timing.merge(&meter);
        }
    }
    pub(super) fn new(
        length: usize,
        table_log2: u8,
        dispatch: Dispatch,
    ) -> Result<Self, BatoError> {
        Ok(Self {
            front: Front::new(length, table_log2)?,
            mixing: Mixing::new(dispatch)?,
            calibration: Calibration::new()?,
            signals: Signals::new(),
            dispatch: dispatch.resolve(),
            meter: None,
        })
    }
    pub(super) fn probability(&mut self, history: &[u8]) -> u16 {
        #[cfg(target_arch = "x86_64")]
        if self.dispatch.is_resolved_avx2() {
            // SAFETY: `self.dispatch` was resolved via `is_x86_feature_detected!("avx2")`.
            return unsafe { self.probability_avx2(history) };
        }
        self.probability_inner(history)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn probability_avx2(&mut self, history: &[u8]) -> u16 {
        self.probability_inner(history)
    }

    /// The stages in the order that overlaps their memory prefetches with
    /// the next stage's arithmetic.
    #[inline(always)]
    fn probability_inner(&mut self, history: &[u8]) -> u16 {
        self.front
            .predict_main(history, &mut self.signals, &mut self.meter);
        self.mixing.predict_first(&self.signals, &mut self.meter);
        self.front
            .predict_experts(&mut self.signals, &mut self.meter);
        self.calibration
            .prefetch(&self.signals.calibration, self.mixing.half());
        let mixed = self.mixing.predict_rest(&self.signals, &mut self.meter);
        self.calibration.predict(&mixed, &mut self.meter)
    }
    pub(super) fn observe(&mut self, bit: u8) {
        #[cfg(target_arch = "x86_64")]
        if self.dispatch.is_resolved_avx2() {
            // SAFETY: `self.dispatch` was resolved via `is_x86_feature_detected!("avx2")`.
            unsafe { self.observe_avx2(bit) };
            return;
        }
        self.observe_inner(bit);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn observe_avx2(&mut self, bit: u8) {
        self.observe_inner(bit);
    }

    #[inline(always)]
    fn observe_inner(&mut self, bit: u8) {
        self.front.prefetch_after(bit);
        self.calibration.observe(bit, &mut self.meter);
        self.mixing.observe(&self.signals, bit, &mut self.meter);
        self.front.observe(bit, &mut self.meter);
    }
    pub(super) fn observe_byte(&mut self, byte: u8, history: &[u8]) {
        for shift in (0..8).rev() {
            // Prediction prepares the per-bit state that `observe` trains; the
            // probability itself is only needed when coding.
            let _ = self.probability(history);
            self.observe((byte >> shift) & 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_observation_and_scalar_dispatch_preserve_predictions() {
        let mut scalar = Predictor::new(4096, 12, Dispatch::Scalar).unwrap();
        let mut vector = Predictor::new(4096, 12, Dispatch::Native).unwrap();
        let raw = b"integer context mixer reference\n".repeat(100);
        for (at, &byte) in raw.iter().enumerate() {
            for shift in (0..8).rev() {
                assert_eq!(
                    scalar.probability(&raw[..at]),
                    vector.probability(&raw[..at])
                );
                scalar.observe((byte >> shift) & 1);
                vector.observe((byte >> shift) & 1);
            }
        }
        let mut bitwise = Predictor::new(4096, 12, Dispatch::Native).unwrap();
        let mut whole = Predictor::new(4096, 12, Dispatch::Native).unwrap();
        for (at, &byte) in raw.iter().enumerate() {
            whole.observe_byte(byte, &raw[..at]);
            for shift in (0..8).rev() {
                bitwise.probability(&raw[..at]);
                bitwise.observe((byte >> shift) & 1);
            }
            assert_eq!(
                bitwise.probability(&raw[..at + 1]),
                whole.probability(&raw[..at + 1])
            );
        }
    }
    #[test]
    fn actual_model_heap_matches_the_reservation() {
        for length in [1, 256, 4096, 65536] {
            for cap in [0, 8, 16, 24] {
                let (_, observed) = crate::core::diagnostics::capture(|| {
                    Predictor::new(length, cap, Dispatch::Native).unwrap()
                });
                assert_eq!(observed.heap.peak, memory_required(length, cap));
            }
        }
    }
}
