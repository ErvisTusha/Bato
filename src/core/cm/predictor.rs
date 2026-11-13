//! Original learned integer context ensemble for segment-persistent coding.
//! Provenance: indirect contexts, neural log-odds mixing and online adaptation
//! described in Mahoney, Data Compression Explained and the public-domain
//! ZPAQ specification. All model composition, scaling and Rust are original.

use super::{
    adapt::{Cell, RunMap},
    apm::Apm,
    coded_bits,
    context_map::{self, ContextMap},
    isse::Isse,
    logistic,
    mixer::Mixer,
    models::{
        matcher::{self, Matcher},
        neighbors::{self, Neighbors},
        order,
        record::Records,
        rows::Rows,
        stride::Stride,
        word::Words,
    },
    profile::{Activity, Meter, Step, Timing},
    simd::{Dispatch, INPUTS},
    table::{self, Table},
};
use crate::core::types::BatoError;

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
const APM_CONTEXTS: [usize; APM_STAGES] = [256, 128, 65536, 65536, 65536, 65536, 65536, 65536];
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
pub(super) struct Predictor {
    table: Table,
    maps: [ContextMap; CONTEXTS],
    bases: [u64; CONTEXTS],
    base: [Cell; 256],
    run: RunMap,
    matcher: Matcher,
    words: Words,
    records: Records,
    stride: Stride,
    rows: Rows,
    neighbors: Neighbors,
    record_map: ContextMap,
    record_active: bool,
    record_base: u64,
    mixers: [Mixer; 3],
    final_mixer: Mixer,
    final_second: Mixer,
    isse: [Isse; 3],
    apms: [Apm; APM_STAGES],
    inputs: [i16; INPUTS],
    second: [i16; INPUTS],
    prefix: u16,
    dispatch: Dispatch,
    expert_maps: [Vec<ContextMap>; EXPERTS],
    expert_bases: [[u64; EXPERT_CONTEXTS]; EXPERTS],
    /// Experts with something to say for the current byte; dormant ones add
    /// a neutral input and skip their lookups and training.
    expert_active: [bool; EXPERTS],
    expert_inputs: [[i16; INPUTS]; EXPERTS],
    expert_mixers: [Mixer; EXPERTS],
    /// Step timer, present only while timing is requested.
    meter: Option<Box<Meter>>,
}

pub(super) fn memory_required(length: usize, table_log2: u8) -> usize {
    table::memory_required(length, table_log2)
        + neighbors::memory_required()
        + matcher::memory_required(length, table_log2)
        + MIX_CONTEXTS
            .into_iter()
            .map(Mixer::memory_required)
            .sum::<usize>()
        + EXPERT_SIZES.iter().sum::<usize>() * std::mem::size_of::<ContextMap>()
        + EXPERTS * Mixer::memory_required(EXPERT_SETS)
        + Mixer::memory_required(FINAL_CONTEXTS)
        + Mixer::memory_required(128)
        + APM_CONTEXTS
            .into_iter()
            .map(Apm::memory_required)
            .sum::<usize>()
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
    #[inline]
    fn lap(&mut self, step: Step) {
        if let Some(meter) = &mut self.meter {
            meter.lap(step);
        }
    }
    pub(super) fn new(
        length: usize,
        table_log2: u8,
        dispatch: Dispatch,
    ) -> Result<Self, BatoError> {
        let mut initial = [0; INPUTS];
        for i in 0..CONTEXTS {
            initial[i * FEATURES] = CONTEXT_WEIGHT;
        }
        initial[ORDER0_INPUT] = ORDER0_WEIGHT;
        let mut second = [0; INPUTS];
        second[..NEIGHBOR_INPUT].fill(FINAL_WEIGHT);
        let mut expert_initial = [0i16; INPUTS];
        for i in 0..EXPERT_CONTEXTS {
            expert_initial[i * FEATURES] = EXPERT_WEIGHT;
        }
        Ok(Self {
            table: Table::new(length, table_log2)?,
            maps: std::array::from_fn(|_| ContextMap::new()),
            bases: [0; CONTEXTS],
            base: [Cell::new(logistic::HALF, 0); 256],
            run: RunMap::default(),
            matcher: Matcher::new(length, table_log2)?,
            words: Words::default(),
            records: Records::default(),
            stride: Stride::default(),
            rows: Rows::default(),
            neighbors: Neighbors::new()?,
            record_map: ContextMap::new(),
            record_active: false,
            record_base: 0,
            mixers: [
                Mixer::with_inputs(MIX_CONTEXTS[0], initial, MIXER_RATE, ACTIVE_INPUTS)?,
                Mixer::with_inputs(MIX_CONTEXTS[1], initial, MIXER_RATE, ACTIVE_INPUTS)?,
                Mixer::with_inputs(MIX_CONTEXTS[2], initial, MIXER_RATE, ACTIVE_INPUTS)?,
            ],
            final_mixer: Mixer::with_inputs(
                FINAL_CONTEXTS,
                second,
                FINAL_RATE,
                FINAL_ACTIVE_INPUTS,
            )?,
            final_second: Mixer::with_inputs(128, second, FINAL_RATE, FINAL_ACTIVE_INPUTS)?,
            isse: std::array::from_fn(|_| Isse::new()),
            apms: [
                Apm::new(APM_CONTEXTS[0])?,
                Apm::new(APM_CONTEXTS[1])?,
                Apm::new(APM_CONTEXTS[2])?,
                Apm::new(APM_CONTEXTS[3])?,
                Apm::new(APM_CONTEXTS[4])?,
                Apm::new(APM_CONTEXTS[5])?,
                Apm::new(APM_CONTEXTS[6])?,
                Apm::new(APM_CONTEXTS[7])?,
            ],
            inputs: [0; INPUTS],
            second: [0; INPUTS],
            prefix: 1,
            dispatch,
            expert_maps: EXPERT_SIZES.map(|size| (0..size).map(|_| ContextMap::new()).collect()),
            expert_bases: [[0; EXPERT_CONTEXTS]; EXPERTS],
            expert_active: [true; EXPERTS],
            expert_inputs: [[0; INPUTS]; EXPERTS],
            expert_mixers: {
                let [a, b, c, d] = EXPERT_ACTIVE_INPUTS;
                let build =
                    |inputs| Mixer::with_inputs(EXPERT_SETS, expert_initial, MIXER_RATE, inputs);
                [build(a)?, build(b)?, build(c)?, build(d)?]
            },
            meter: None,
        })
    }
    pub(super) fn probability(&mut self, history: &[u8]) -> u16 {
        let depth = coded_bits(self.prefix) as usize;
        self.lap(Step::Coder);
        if self.prefix == 1 {
            self.bases[..order::COUNT].copy_from_slice(&order::contexts(history));
            self.bases[order::COUNT..].copy_from_slice(&self.words.contexts());
            let last = history.last().copied().unwrap_or(0);
            let [word_byte, trigram] = self.words.text_contexts(last);
            self.expert_bases[0][..4].copy_from_slice(&[
                super::hash::bytes(history, TEXT_ORDERS[0]),
                super::hash::bytes(history, TEXT_ORDERS[1]),
                word_byte,
                trigram,
            ]);
            let back = |k: usize| -> u64 {
                history
                    .len()
                    .checked_sub(k)
                    .map_or(MISSING_BYTE, |i| u64::from(history[i]))
            };
            let sparse = [back(2), back(3), back(4), back(2) | back(3) << 9];
            for (i, value) in sparse.into_iter().enumerate() {
                self.expert_bases[1][i] =
                    super::hash::mix(value ^ ((i as u64 + 1) << 48) ^ SPARSE_SALT);
            }
            self.lap(Step::ByteContexts);
            self.matcher.begin_byte(history);
            self.lap(Step::ByteMatch);
            self.stride.begin_byte(
                history,
                self.matcher
                    .long_distance(history.len(), STRIDE_VOTE_LENGTH),
            );
            self.expert_bases[2][..4].copy_from_slice(&self.stride.contexts(history));
            self.lap(Step::ByteLag);
            self.rows.begin_byte(history);
            self.expert_bases[3] = self.rows.contexts(history);
            self.expert_active[2] = self.stride.active();
            self.expert_active[3] = self.rows.active();
            self.lap(Step::ByteRows);
            let record = self.records.begin_byte(history);
            self.record_active = record.is_some();
            self.record_base = record.unwrap_or(0);
            self.neighbors
                .begin_byte(history, self.records.period().or(self.rows.period()));
            self.lap(Step::ByteRecords);
        }
        // The calibration rows are known now; load them while the contexts
        // below are looked up and mixed.
        let contexts = self.calibration_contexts(history, depth);
        for (map, &context) in self.apms.iter().zip(&contexts) {
            map.prefetch(context);
        }
        self.lap(Step::CalibrationSetup);
        if depth.is_multiple_of(context_map::NIBBLE_BITS) {
            let active = (0..EXPERTS).filter(|&e| self.expert_active[e]);
            let experts = active.flat_map(|e| self.expert_bases[e][..EXPERT_SIZES[e]].iter());
            for &base in self.bases.iter().chain(experts) {
                self.table
                    .prefetch(super::hash::mix(base ^ (u64::from(self.prefix) << 32)));
            }
        }
        self.lap(Step::Prefetch);
        let mut hits = 0;
        for i in 0..CONTEXTS {
            self.inputs[i * FEATURES..(i + 1) * FEATURES].copy_from_slice(&self.maps[i].predict(
                &mut self.table,
                self.bases[i],
                self.prefix,
                depth,
            ));
            hits += usize::from(self.maps[i].hit());
        }
        self.lap(Step::ContextMaps);
        self.inputs[ORDER0_INPUT] =
            logistic::stretch(self.base[usize::from(self.prefix) - 1].probability());
        self.inputs[RUN_INPUT] = logistic::stretch(self.run.probability(self.prefix));
        self.inputs[MATCH_INPUTS..RECORD_INPUTS]
            .copy_from_slice(&self.matcher.features(self.prefix));
        self.inputs[RECORD_INPUTS..RECORD_INPUTS + 3].fill(0);
        if self.record_active {
            let features =
                self.record_map
                    .predict(&mut self.table, self.record_base, self.prefix, depth);
            self.inputs[RECORD_INPUTS..RECORD_INPUTS + 3].copy_from_slice(&features[..3]);
        }
        self.inputs[ACTIVE_INPUTS - 1] = logistic::UNIT;
        let previous = usize::from(history.last().copied().unwrap_or(0));
        let selectors = [
            usize::from(self.prefix) - 1,
            previous * 8 + depth,
            hits * 8 + depth,
        ];
        self.lap(Step::ExtraInputs);
        for (i, mixer) in self.mixers.iter_mut().enumerate() {
            self.second[i] =
                logistic::stretch(mixer.predict(&self.inputs, selectors[i], self.dispatch));
        }
        let neighbors = self.neighbors.feature(self.prefix);
        self.second[NEIGHBOR_INPUT] = neighbors[0];
        self.second[TEMPLATE_INPUT..TEMPLATE_INPUT + neighbors::FEATURES - 1]
            .copy_from_slice(&neighbors[1..]);
        self.lap(Step::FirstMixers);
        for e in (0..EXPERTS).filter(|&e| self.expert_active[e]) {
            for i in 0..EXPERT_SIZES[e] {
                self.expert_inputs[e][i * FEATURES..(i + 1) * FEATURES].copy_from_slice(
                    &self.expert_maps[e][i].predict(
                        &mut self.table,
                        self.expert_bases[e][i],
                        self.prefix,
                        depth,
                    ),
                );
            }
            self.expert_inputs[e][EXPERT_ACTIVE_INPUTS[e] - 1] = logistic::UNIT;
            self.second[EXPERT_INPUT + e] = logistic::stretch(self.expert_mixers[e].predict(
                &self.expert_inputs[e],
                usize::from(self.prefix) - 1,
                self.dispatch,
            ));
        }
        for e in (0..EXPERTS).filter(|&e| !self.expert_active[e]) {
            self.second[EXPERT_INPUT + e] = 0;
        }
        self.lap(Step::Experts);
        self.note_activity();
        self.second[FINAL_ACTIVE_INPUTS - 1] = logistic::UNIT;
        let first_final =
            self.final_mixer
                .predict(&self.second, ((previous >> 5) * 8) + depth, self.dispatch);
        let second_final = self.final_second.predict(
            &self.second,
            self.matcher.state() * 8 + depth,
            self.dispatch,
        );
        let mut prediction = ((u32::from(first_final) + u32::from(second_final) + 1) >> 1) as u16;
        self.lap(Step::FinalMixers);
        for (expert, index) in self.isse.iter_mut().zip(ISSE_MAPS) {
            prediction = expert.predict(prediction, self.maps[index].history_state());
        }
        self.lap(Step::Isse);
        for (map, context) in self.apms.iter_mut().zip(contexts) {
            prediction = ((u32::from(prediction) + u32::from(map.predict(prediction, context)) + 1)
                >> 1) as u16;
        }
        self.lap(Step::Calibration);
        prediction
    }
    /// Counts which optional models took part in this bit, while timing.
    fn note_activity(&mut self) {
        let Some(meter) = &mut self.meter else {
            return;
        };
        if !meter.sampled() {
            return;
        }
        meter.count(Activity::Bits, true);
        meter.count(Activity::MatchPredicting, self.matcher.expected().is_some());
        meter.count(Activity::Records, self.record_active);
        let experts = [
            Activity::TextExpert,
            Activity::SparseExpert,
            Activity::LagExpert,
            Activity::RowExpert,
        ];
        for (activity, active) in experts.into_iter().zip(self.expert_active) {
            meter.count(activity, active);
        }
    }
    /// Contexts of the calibration stages, applied in this order: the partial
    /// byte; the match state; the partial byte with two and with three
    /// previous bytes; the order-4, current-word and word-pair contexts; and
    /// the byte the match model expects. Each refines the previous stage's
    /// output. A stage keyed by one previous byte made files 0.064% larger in
    /// the ablation study of docs/validation.md and was removed.
    fn calibration_contexts(&self, history: &[u8], depth: usize) -> [usize; APM_STAGES] {
        let back = |k: usize| -> u64 {
            history
                .len()
                .checked_sub(k)
                .map_or(0, |i| u64::from(history[i]))
        };
        let prefix = u64::from(self.prefix);
        let expected = self.matcher.expected();
        let match_class = expected.map_or(0, |(byte, _)| usize::from(byte >> 6));
        let expected_byte = expected.map_or(0x100, |(byte, _)| u64::from(byte));
        let hashed = |base: u64, salt: u64| super::hash::mix(base ^ (prefix << 52) ^ salt) as usize;
        [
            usize::from(self.prefix) - 1,
            (self.matcher.state() * 8 + depth) ^ (match_class << 3),
            super::hash::mix((prefix << 16) | (back(2) << 8) | back(1)) as usize,
            super::hash::mix((prefix << 24) | (back(3) << 16) | (back(2) << 8) | back(1)) as usize,
            hashed(self.bases[order::COUNT], 0),
            hashed(self.bases[4], 0x5bd1),
            hashed(self.bases[order::COUNT + 1], 0x9e37),
            super::hash::mix((prefix << 24) | (expected_byte << 8) | back(1)) as usize,
        ]
    }
    pub(super) fn observe(&mut self, bit: u8) {
        for map in &mut self.apms {
            map.observe(bit);
        }
        for expert in &mut self.isse {
            expert.observe(bit);
        }
        self.lap(Step::TrainCalibration);
        for mixer in &mut self.mixers {
            mixer.observe(&self.inputs, bit);
        }
        self.final_mixer.observe(&self.second, bit);
        self.final_second.observe(&self.second, bit);
        self.lap(Step::TrainMixers);
        for e in (0..EXPERTS).filter(|&e| self.expert_active[e]) {
            self.expert_mixers[e].observe(&self.expert_inputs[e], bit);
            for map in &mut self.expert_maps[e][..EXPERT_SIZES[e]] {
                map.observe(&mut self.table, bit);
            }
        }
        self.lap(Step::TrainExperts);
        self.neighbors.observe(bit);
        for map in &mut self.maps {
            map.observe(&mut self.table, bit);
        }
        if self.record_active {
            self.record_map.observe(&mut self.table, bit);
        }
        self.base[usize::from(self.prefix) - 1].train(bit, ORDER0_LIMIT);
        self.matcher.observe(self.prefix, bit);
        self.lap(Step::TrainMaps);
        self.prefix = (self.prefix << 1) | u16::from(bit);
        if self.prefix >= 256 {
            self.run.observe_byte(self.prefix as u8);
            self.words.observe_byte(self.prefix as u8);
            self.records.observe_byte(self.prefix as u8);
            self.prefix = 1;
            self.lap(Step::ByteEnd);
            if let Some(meter) = &mut self.meter {
                meter.end_byte();
            }
        }
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
