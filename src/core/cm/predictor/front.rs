//! Data-driven stage of the predictor: hashed contexts, context maps, the
//! match, record, stride, row and neighbour models and the experts' inputs.
//! Provenance: same as the parent module; this is the original code moved
//! out of it. Nothing here reads a mixer or calibration output, so the stage
//! can run ahead of the later ones when every bit is already known.

use super::{
    lap, Signals, ACTIVE_INPUTS, APM_STAGES, CONTEXTS, EXPERTS, EXPERT_ACTIVE_INPUTS,
    EXPERT_CONTEXTS, EXPERT_SIZES, FEATURES, ISSE_MAPS, MATCH_INPUTS, MISSING_BYTE, ORDER0_INPUT,
    ORDER0_LIMIT, RECORD_INPUTS, RUN_INPUT, SPARSE_SALT, STRIDE_VOTE_LENGTH, TEXT_ORDERS,
};
use crate::core::{
    cm::{
        adapt::{Cell, RunMap},
        coded_bits,
        context_map::{self, ContextMap},
        hash, logistic,
        models::{
            matcher::{self, Matcher},
            neighbors::{self, Neighbors},
            order,
            record::Records,
            rows::Rows,
            stride::Stride,
            word::Words,
        },
        profile::{Activity, Meter, Step},
        table::{self, Table},
    },
    types::BatoError,
};

#[derive(Clone)]
pub(super) struct Front {
    table: Table,
    maps: [ContextMap; CONTEXTS],
    bases: [u64; CONTEXTS],
    mixed_bases: [u64; CONTEXTS],
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
    mixed_record: u64,
    apm_bases: [u64; 6],
    prefix: u16,
    expert_maps: [Vec<ContextMap>; EXPERTS],
    expert_bases: [[u64; EXPERT_CONTEXTS]; EXPERTS],
    mixed_experts: [[u64; EXPERT_CONTEXTS]; EXPERTS],
    /// Experts with something to say for the current byte; dormant ones add
    /// a neutral input and skip their lookups and training.
    expert_active: [bool; EXPERTS],
}

pub(super) fn memory_required(length: usize, table_log2: u8) -> usize {
    table::memory_required(length, table_log2)
        + neighbors::memory_required()
        + matcher::memory_required(length, table_log2)
        + EXPERT_SIZES.iter().sum::<usize>() * std::mem::size_of::<ContextMap>()
}

impl Front {
    pub(super) fn new(length: usize, table_log2: u8) -> Result<Self, BatoError> {
        Ok(Self {
            table: Table::new(length, table_log2)?,
            maps: std::array::from_fn(|_| ContextMap::new()),
            bases: [0; CONTEXTS],
            mixed_bases: [0; CONTEXTS],
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
            mixed_record: 0,
            apm_bases: [0; 6],
            prefix: 1,
            expert_maps: EXPERT_SIZES.map(|size| (0..size).map(|_| ContextMap::new()).collect()),
            expert_bases: [[0; EXPERT_CONTEXTS]; EXPERTS],
            mixed_experts: [[0; EXPERT_CONTEXTS]; EXPERTS],
            expert_active: [true; EXPERTS],
        })
    }
    #[inline]
    fn mix_nibble(base: u64, prefix: u16) -> u64 {
        hash::mix(base ^ (u64::from(prefix) << 32))
    }
    #[inline]
    fn depth_and_node(prefix: u16) -> (usize, usize) {
        let depth = coded_bits(prefix) as usize;
        let nibble_depth = depth % context_map::NIBBLE_BITS;
        let node = if nibble_depth == 0 {
            0
        } else {
            ((1usize << nibble_depth) | (usize::from(prefix) & ((1 << nibble_depth) - 1))) - 1
        };
        (depth, node)
    }
    #[inline]
    fn prefetch_experts(&mut self, range: std::ops::Range<usize>, prefix: u16) {
        for e in range.filter(|&e| self.expert_active[e]) {
            for (mixed, &base) in self.mixed_experts[e][..EXPERT_SIZES[e]]
                .iter_mut()
                .zip(&self.expert_bases[e][..EXPERT_SIZES[e]])
            {
                let key = Self::mix_nibble(base, prefix);
                *mixed = key;
                self.table.prefetch(key);
            }
        }
    }
    #[inline]
    fn prefetch_nibble(&mut self, prefix: u16) {
        for (mixed, &base) in self.mixed_bases.iter_mut().zip(&self.bases) {
            let key = Self::mix_nibble(base, prefix);
            *mixed = key;
            self.table.prefetch(key);
        }
        if self.record_active {
            self.mixed_record = Self::mix_nibble(self.record_base, prefix);
            self.table.prefetch(self.mixed_record);
        }
        self.prefetch_experts(0..EXPERTS, prefix);
    }

    /// Everything the first-layer mixers need for the next bit: the byte-start
    /// models, the context maps and the extra inputs, written into `signals`.
    #[inline(always)]
    pub(super) fn predict_main(
        &mut self,
        history: &[u8],
        signals: &mut Signals,
        meter: &mut Option<Box<Meter>>,
    ) {
        let (depth, node) = Self::depth_and_node(self.prefix);
        lap(meter, Step::Coder);
        if self.prefix == 1 {
            self.bases[..order::COUNT].copy_from_slice(&order::contexts(history));
            self.bases[order::COUNT..].copy_from_slice(&self.words.contexts());
            // Mix and issue main context-table prefetches before byte-level
            // models run so DRAM fetches overlap with matcher, stride, rows
            // and record scans.
            for (mixed, &base) in self.mixed_bases.iter_mut().zip(&self.bases) {
                let key = Self::mix_nibble(base, self.prefix);
                *mixed = key;
                self.table.prefetch(key);
            }
            let last = history.last().copied().unwrap_or(0);
            let [word_byte, trigram] = self.words.text_contexts(last);
            self.expert_bases[0][..4].copy_from_slice(&[
                hash::bytes(history, TEXT_ORDERS[0]),
                hash::bytes(history, TEXT_ORDERS[1]),
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
                self.expert_bases[1][i] = hash::mix(value ^ ((i as u64 + 1) << 48) ^ SPARSE_SALT);
            }
            self.prefetch_experts(0..2, self.prefix);
            let apm_back = |k: usize| -> u64 {
                history
                    .len()
                    .checked_sub(k)
                    .map_or(0, |i| u64::from(history[i]))
            };
            let b1 = apm_back(1);
            let b12 = (apm_back(2) << 8) | b1;
            let b123 = (apm_back(3) << 16) | b12;
            self.apm_bases = [
                b1,
                b12,
                b123,
                self.bases[order::COUNT],
                self.bases[4] ^ 0x5bd1,
                self.bases[order::COUNT + 1] ^ 0x9e37,
            ];
            lap(meter, Step::ByteContexts);
            self.matcher.begin_byte(history, self.bases[5]);
            lap(meter, Step::ByteMatch);
            self.stride.begin_byte(
                history,
                self.matcher
                    .long_distance(history.len(), STRIDE_VOTE_LENGTH),
            );
            self.expert_bases[2][..4].copy_from_slice(&self.stride.contexts(history));
            lap(meter, Step::ByteLag);
            self.rows.begin_byte(history);
            self.expert_bases[3] = self.rows.contexts(history);
            self.expert_active[2] = self.stride.active();
            self.expert_active[3] = self.rows.active();
            lap(meter, Step::ByteRows);
            let record = self.records.begin_byte(history);
            self.record_active = record.is_some();
            self.record_base = record.unwrap_or(0);
            self.neighbors
                .begin_byte(history, self.records.period().or(self.rows.period()));
            lap(meter, Step::ByteRecords);
            if self.record_active {
                self.mixed_record = Self::mix_nibble(self.record_base, self.prefix);
                self.table.prefetch(self.mixed_record);
            }
            self.prefetch_experts(2..EXPERTS, self.prefix);
        } else if depth == context_map::NIBBLE_BITS
            && self.mixed_bases[0] != Self::mix_nibble(self.bases[0], self.prefix)
        {
            self.prefetch_nibble(self.prefix);
        }
        lap(meter, Step::Prefetch);
        let mut hits = 0;
        for i in 0..CONTEXTS {
            signals.inputs[i * FEATURES..(i + 1) * FEATURES].copy_from_slice(
                &self.maps[i].predict(&mut self.table, self.mixed_bases[i], depth, node),
            );
            hits += usize::from(self.maps[i].hit());
        }
        lap(meter, Step::ContextMaps);
        signals.inputs[ORDER0_INPUT] =
            logistic::stretch(self.base[usize::from(self.prefix) - 1].probability());
        signals.inputs[RUN_INPUT] = logistic::stretch(self.run.probability(self.prefix));
        signals.inputs[MATCH_INPUTS..RECORD_INPUTS]
            .copy_from_slice(&self.matcher.features(self.prefix));
        signals.inputs[RECORD_INPUTS..RECORD_INPUTS + 3].fill(0);
        if self.record_active {
            let features = self
                .record_map
                .predict(&mut self.table, self.mixed_record, depth, node);
            signals.inputs[RECORD_INPUTS..RECORD_INPUTS + 3].copy_from_slice(&features[..3]);
        }
        signals.inputs[ACTIVE_INPUTS - 1] = logistic::UNIT;
        let previous = usize::from(history.last().copied().unwrap_or(0));
        signals.selectors = [
            usize::from(self.prefix) - 1,
            previous * 8 + depth,
            hits * 8 + depth,
        ];
        signals.final_selectors[0] = ((previous >> 5) * 8) + depth;
        lap(meter, Step::ExtraInputs);
        // Overlap neighbour template prefetches with the first-layer mixers.
        self.neighbors.prefetch(self.prefix);
    }

    /// The rest of the bit's inputs: the neighbour features, the experts'
    /// context-map inputs and the calibration contexts. A serial caller runs
    /// the first-layer mixers between `predict_main` and this, so the
    /// prefetches issued there have time to land.
    #[inline(always)]
    pub(super) fn predict_experts(
        &mut self,
        signals: &mut Signals,
        meter: &mut Option<Box<Meter>>,
    ) {
        let (depth, node) = Self::depth_and_node(self.prefix);
        signals.neighbors = self.neighbors.feature(self.prefix);
        signals.calibration = self.calibration_contexts(depth);
        signals.isse_states = ISSE_MAPS.map(|index| self.maps[index].history_state());
        signals.expert_active = self.expert_active;
        signals.final_selectors[1] = self.matcher.state() * 8 + depth;
        lap(meter, Step::CalibrationSetup);
        for e in (0..EXPERTS).filter(|&e| self.expert_active[e]) {
            for i in 0..EXPERT_SIZES[e] {
                signals.expert_inputs[e][i * FEATURES..(i + 1) * FEATURES].copy_from_slice(
                    &self.expert_maps[e][i].predict(
                        &mut self.table,
                        self.mixed_experts[e][i],
                        depth,
                        node,
                    ),
                );
            }
            signals.expert_inputs[e][EXPERT_ACTIVE_INPUTS[e] - 1] = logistic::UNIT;
        }
        lap(meter, Step::Experts);
        self.note_activity(meter);
    }

    /// Counts which optional models took part in this bit, while timing.
    fn note_activity(&mut self, meter: &mut Option<Box<Meter>>) {
        let Some(meter) = meter else {
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
    /// byte; the match state, predicted bit and high class bits; the partial
    /// byte with two and with three
    /// previous bytes; the order-4, current-word and word-pair contexts; and
    /// the byte the match model expects. Each refines the previous stage's
    /// output. A stage keyed by one previous byte made files 0.064% larger in
    /// the ablation study of docs/validation.md and was removed.
    fn calibration_contexts(&self, depth: usize) -> [usize; APM_STAGES] {
        let prefix = u64::from(self.prefix);
        let expected = self.matcher.expected();
        let match_class = expected.map_or(0, |(byte, _)| usize::from(byte >> 6));
        let match_bit = expected.map_or(0, |(byte, _)| usize::from((byte >> (7 - depth)) & 1));
        let expected_byte = expected.map_or(0x100, |(byte, _)| u64::from(byte));
        let p52 = prefix << 52;
        [
            usize::from(self.prefix) - 1,
            (self.matcher.state() * 8 + depth) | (match_bit << 7) | (match_class << 8),
            hash::mix((prefix << 16) | self.apm_bases[1]) as usize,
            hash::mix((prefix << 24) | self.apm_bases[2]) as usize,
            hash::mix(self.apm_bases[3] ^ p52) as usize,
            hash::mix(self.apm_bases[4] ^ p52) as usize,
            hash::mix(self.apm_bases[5] ^ p52) as usize,
            hash::mix((prefix << 24) | (expected_byte << 8) | self.apm_bases[0]) as usize,
        ]
    }

    /// Starts loading the next nibble's table rows once `bit` completes the
    /// current nibble; a serial caller issues it before the training steps.
    #[inline(always)]
    pub(super) fn prefetch_after(&mut self, bit: u8) {
        if (8..16).contains(&self.prefix) {
            let next_prefix = (self.prefix << 1) | u16::from(bit);
            self.prefetch_nibble(next_prefix);
        }
    }

    /// Trains every context model on `bit` and moves to the next bit.
    #[inline(always)]
    pub(super) fn observe(&mut self, bit: u8, meter: &mut Option<Box<Meter>>) {
        let (depth, node) = Self::depth_and_node(self.prefix);
        for e in (0..EXPERTS).filter(|&e| self.expert_active[e]) {
            for map in &mut self.expert_maps[e][..EXPERT_SIZES[e]] {
                map.observe(&mut self.table, depth, node, bit);
            }
        }
        lap(meter, Step::TrainExperts);
        self.neighbors.observe(bit);
        for map in &mut self.maps {
            map.observe(&mut self.table, depth, node, bit);
        }
        if self.record_active {
            self.record_map.observe(&mut self.table, depth, node, bit);
        }
        self.base[usize::from(self.prefix) - 1].train(bit, ORDER0_LIMIT);
        self.matcher.observe(self.prefix, bit);
        lap(meter, Step::TrainMaps);
        self.prefix = (self.prefix << 1) | u16::from(bit);
        if self.prefix >= 256 {
            self.run.observe_byte(self.prefix as u8);
            self.words.observe_byte(self.prefix as u8);
            self.records.observe_byte(self.prefix as u8);
            self.prefix = 1;
            lap(meter, Step::ByteEnd);
            if let Some(meter) = meter {
                meter.end_byte();
            }
        }
    }
}
