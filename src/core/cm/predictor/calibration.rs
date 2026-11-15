//! Calibration stage of the predictor: the ISSE chain and the APM stages that
//! refine the mixers' probability. Provenance: same as the parent module;
//! this is the original code moved out of it.

use super::{lap, Mixed, APM_CONTEXTS, APM_STAGES, ISSE_MAPS};
use crate::core::{
    cm::{
        apm::{Apm, BinHalf},
        isse::Isse,
        profile::{Meter, Step},
    },
    types::BatoError,
};

#[derive(Clone)]
pub(super) struct Calibration {
    isse: [Isse; ISSE_MAPS.len()],
    apms: [Apm; APM_STAGES],
}

pub(super) fn memory_required() -> usize {
    APM_CONTEXTS
        .into_iter()
        .map(Apm::memory_required)
        .sum::<usize>()
}

impl Calibration {
    pub(super) fn new() -> Result<Self, BatoError> {
        Ok(Self {
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
        })
    }

    /// Starts loading the APM rows of a bit well before its probability is
    /// needed, so large tables do not stall the final stages.
    #[inline(always)]
    pub(super) fn prefetch(&self, contexts: &[usize; APM_STAGES], half: BinHalf) {
        for (map, &context) in self.apms.iter().zip(contexts) {
            map.prefetch(context, half);
        }
    }

    /// The final probability of a bit from the mixers' `Mixed` output.
    #[inline(always)]
    pub(super) fn predict(&mut self, mixed: &Mixed, meter: &mut Option<Box<Meter>>) -> u16 {
        let mut prediction = mixed.prediction;
        for (expert, state) in self.isse.iter_mut().zip(mixed.isse_states) {
            prediction = expert.predict(prediction, state);
        }
        lap(meter, Step::Isse);
        for (map, &context) in self.apms.iter_mut().zip(&mixed.calibration) {
            prediction = ((u32::from(prediction) + u32::from(map.predict(prediction, context)) + 1)
                >> 1) as u16;
        }
        lap(meter, Step::Calibration);
        prediction
    }

    /// Trains the ISSE chain and the APM stages on `bit`.
    #[inline(always)]
    pub(super) fn observe(&mut self, bit: u8, meter: &mut Option<Box<Meter>>) {
        for map in &mut self.apms {
            map.observe(bit);
        }
        for expert in &mut self.isse {
            expert.observe(bit);
        }
        lap(meter, Step::TrainCalibration);
    }
}
