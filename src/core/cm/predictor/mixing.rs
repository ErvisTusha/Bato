//! Mixing stage of the predictor: the three first-layer mixers, the experts'
//! mixers and the two final mixers. Provenance: same as the parent module;
//! this is the original code moved out of it. It reads only the data-driven
//! stage's `Signals` and its own weights.

use super::{
    lap, Mixed, Signals, ACTIVE_INPUTS, CONTEXT_WEIGHT, EXPERTS, EXPERT_ACTIVE_INPUTS,
    EXPERT_CONTEXTS, EXPERT_INPUT, EXPERT_SETS, EXPERT_SIZES, EXPERT_WEIGHT, FEATURES,
    FINAL_ACTIVE_INPUTS, FINAL_CONTEXTS, FINAL_RATE, FINAL_WEIGHT, MIXER_RATE, MIX_CONTEXTS,
    NEIGHBOR_INPUT, ORDER0_INPUT, ORDER0_WEIGHT, TEMPLATE_INPUT,
};
use crate::core::{
    cm::{
        apm::BinHalf,
        logistic,
        mixer::{LaneShape, Mixer},
        models::neighbors,
        profile::{Meter, Step},
        simd::{Dispatch, INPUTS},
    },
    types::BatoError,
};

#[derive(Clone)]
pub(super) struct Mixing {
    mixers: [Mixer; 3],
    final_mixer: Mixer,
    final_second: Mixer,
    expert_mixers: [Mixer; EXPERTS],
    /// Final-mixer inputs: the first-layer and expert outputs followed by the
    /// neighbour templates.
    second: [i16; INPUTS],
    dispatch: Dispatch,
}

pub(super) fn memory_required() -> usize {
    MIX_CONTEXTS
        .into_iter()
        .map(Mixer::memory_required)
        .sum::<usize>()
        + EXPERTS * Mixer::memory_required(EXPERT_SETS)
        + Mixer::memory_required(FINAL_CONTEXTS)
        + Mixer::memory_required(128)
}

impl Mixing {
    pub(super) fn new(dispatch: Dispatch) -> Result<Self, BatoError> {
        let mut initial = [0; INPUTS];
        for i in 0..super::CONTEXTS {
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
            expert_mixers: {
                let build = |e: usize| {
                    Mixer::with_sparse_bias(
                        EXPERT_SETS,
                        expert_initial,
                        MIXER_RATE,
                        LaneShape {
                            active: EXPERT_SIZES[e] * FEATURES,
                            total: EXPERT_ACTIVE_INPUTS[e],
                        },
                    )
                };
                [build(0)?, build(1)?, build(2)?, build(3)?]
            },
            second: [0; INPUTS],
            dispatch: dispatch.resolve(),
        })
    }

    /// Runs the first-layer mixers, which fixes the cache half the calibration
    /// stages will read.
    #[inline(always)]
    pub(super) fn predict_first(&mut self, signals: &Signals, meter: &mut Option<Box<Meter>>) {
        for (i, mixer) in self.mixers.iter_mut().enumerate() {
            self.second[i] = logistic::stretch(mixer.predict(
                &signals.inputs,
                signals.selectors[i],
                self.dispatch,
            ));
        }
        lap(meter, Step::FirstMixers);
    }

    /// Which half of an APM row the calibration of this bit reads first.
    #[inline(always)]
    pub(super) fn half(&self) -> BinHalf {
        if self.second[0] >= 0 {
            BinHalf::High
        } else {
            BinHalf::Low
        }
    }

    /// Runs the experts' and the final mixers after `predict_first`, with the
    /// neighbour templates the data-driven stage produced meanwhile.
    #[inline(always)]
    pub(super) fn predict_rest(
        &mut self,
        signals: &Signals,
        meter: &mut Option<Box<Meter>>,
    ) -> Mixed {
        self.second[NEIGHBOR_INPUT] = signals.neighbors[0];
        self.second[TEMPLATE_INPUT..TEMPLATE_INPUT + neighbors::FEATURES - 1]
            .copy_from_slice(&signals.neighbors[1..]);
        for e in 0..EXPERTS {
            self.second[EXPERT_INPUT + e] = if signals.expert_active[e] {
                logistic::stretch(self.expert_mixers[e].predict(
                    &signals.expert_inputs[e],
                    signals.selectors[0],
                    self.dispatch,
                ))
            } else {
                0
            };
        }
        lap(meter, Step::Experts);
        self.second[FINAL_ACTIVE_INPUTS - 1] = logistic::UNIT;
        let first_final =
            self.final_mixer
                .predict(&self.second, signals.final_selectors[0], self.dispatch);
        let second_final =
            self.final_second
                .predict(&self.second, signals.final_selectors[1], self.dispatch);
        let prediction = ((u32::from(first_final) + u32::from(second_final) + 1) >> 1) as u16;
        lap(meter, Step::FinalMixers);
        Mixed {
            prediction,
            half: self.half(),
            isse_states: signals.isse_states,
            calibration: signals.calibration,
        }
    }

    /// Trains every mixer on `bit`, with the inputs they predicted from.
    #[inline(always)]
    pub(super) fn observe(&mut self, signals: &Signals, bit: u8, meter: &mut Option<Box<Meter>>) {
        for mixer in &mut self.mixers {
            mixer.observe(&signals.inputs, bit);
        }
        self.final_mixer.observe(&self.second, bit);
        self.final_second.observe(&self.second, bit);
        for e in (0..EXPERTS).filter(|&e| signals.expert_active[e]) {
            self.expert_mixers[e].observe(&signals.expert_inputs[e], bit);
        }
        lap(meter, Step::TrainMixers);
    }
}
