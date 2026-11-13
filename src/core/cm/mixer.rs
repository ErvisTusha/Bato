//! Original integer online logistic network with selectable weight sets.
//! Provenance: gradient descent for log loss as described in Mahoney, Data
//! Compression Explained; all scaling, initialization and Rust are original.

use super::{
    logistic,
    simd::{self, Dispatch, INPUTS, WEIGHT_LIMIT},
};
use crate::core::{memory::try_filled, types::BatoError};
/// Weights carry SCALE_BITS extra fractional bits (1.0 = 256 << SCALE_BITS)
/// while the learning rate in real units is unchanged. With i16 weights an
/// update `round(error * input >> shift)` is zero whenever the product is
/// below 2^(shift-1); the finer scale shrinks that dead zone four-fold, so
/// mixers keep adapting on well-predicted bits. Weights saturate at
/// WEIGHT_LIMIT, i.e. 8.0 in real units.
const SCALE_BITS: u8 = 2;
/// log2 of weight 1.0 before `SCALE_BITS`: weights share the 1/256 logit unit.
const UNIT_BITS: u8 = 8;
#[derive(Clone)]
pub(super) struct Mixer {
    weights: Vec<[i16; INPUTS]>,
    selected: usize,
    prediction: u16,
    rate_shift: u8,
    inputs: usize,
    dispatch: Dispatch,
}
impl Mixer {
    pub(super) fn memory_required(contexts: usize) -> usize {
        contexts * INPUTS * 2
    }
    #[cfg(test)]
    pub(super) fn new(
        contexts: usize,
        initial: [i16; INPUTS],
        rate_shift: u8,
    ) -> Result<Self, BatoError> {
        Self::with_inputs(contexts, initial, rate_shift, INPUTS)
    }
    pub(super) fn with_inputs(
        contexts: usize,
        initial: [i16; INPUTS],
        rate_shift: u8,
        inputs: usize,
    ) -> Result<Self, BatoError> {
        if !contexts.is_power_of_two()
            || inputs == 0
            || inputs > INPUTS
            || !(20..=30).contains(&rate_shift)
            || initial
                .iter()
                .any(|&w| !(-WEIGHT_LIMIT..=WEIGHT_LIMIT).contains(&w))
        {
            return Err(BatoError::InvalidOptions(
                "invalid CM mixer geometry or rate".into(),
            ));
        }
        Ok(Self {
            weights: try_filled(
                contexts,
                initial.map(|w| {
                    (i32::from(w) << SCALE_BITS)
                        .clamp(-i32::from(WEIGHT_LIMIT), i32::from(WEIGHT_LIMIT))
                        as i16
                }),
            )?,
            selected: 0,
            prediction: 32768,
            rate_shift,
            inputs,
            dispatch: Dispatch::Native,
        })
    }
    #[inline]
    pub(super) fn predict(
        &mut self,
        inputs: &[i16; INPUTS],
        context: usize,
        dispatch: Dispatch,
    ) -> u16 {
        self.selected = context & (self.weights.len() - 1);
        self.dispatch = dispatch;
        let logit = (simd::dot_active(inputs, &self.weights[self.selected], self.inputs, dispatch)
            >> (UNIT_BITS + SCALE_BITS))
            .clamp(i64::from(-logistic::LIMIT), i64::from(logistic::LIMIT))
            as i32;
        self.prediction = logistic::squash(logit);
        self.prediction
    }
    #[inline]
    pub(super) fn observe(&mut self, inputs: &[i16; INPUTS], bit: u8) {
        let error = if bit == 0 {
            i64::from(logistic::PROBABILITY_ONE)
        } else {
            0
        } - i64::from(self.prediction);
        let update = simd::MixerUpdate {
            count: self.inputs,
            error: error as i32,
            shift: self.rate_shift - SCALE_BITS,
        };
        simd::train_active(
            inputs,
            &mut self.weights[self.selected],
            update,
            self.dispatch,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_sets_learn_opposite_outcomes_and_stay_bounded() {
        let mut network = Mixer::new(2, [0; INPUTS], 22).unwrap();
        let mut inputs = [0; INPUTS];
        inputs[0] = 1024;
        for _ in 0..3000 {
            for context in 0..2 {
                network.predict(&inputs, context, Dispatch::Scalar);
                network.observe(&inputs, context as u8);
            }
        }
        assert!(network.predict(&inputs, 0, Dispatch::Native) > 62000);
        assert!(network.predict(&inputs, 1, Dispatch::Native) < 3500);
        assert!(network
            .weights
            .iter()
            .flatten()
            .all(|&w| (-WEIGHT_LIMIT..=WEIGHT_LIMIT).contains(&w)));
    }
    #[test]
    fn training_and_second_layer_are_bit_exact_in_both_kernels() {
        let mut a = Mixer::new(8, [4; INPUTS], 22).unwrap();
        let mut b = Mixer::new(8, [4; INPUTS], 22).unwrap();
        let mut second_a = Mixer::new(8, [0; INPUTS], 23).unwrap();
        let mut second_b = Mixer::new(8, [0; INPUTS], 23).unwrap();
        let mut inputs = [0; INPUTS];
        let mut second = [0; INPUTS];
        second[1] = 256;
        for i in 0..5000usize {
            inputs[i % INPUTS] = ((i * 97) % 8193) as i16 - 4096;
            let p = a.predict(&inputs, i & 7, Dispatch::Native);
            assert_eq!(p, b.predict(&inputs, i & 7, Dispatch::Scalar));
            second[0] = logistic::stretch(p);
            assert_eq!(
                second_a.predict(&second, i & 7, Dispatch::Native),
                second_b.predict(&second, i & 7, Dispatch::Scalar)
            );
            a.observe(&inputs, (i & 1) as u8);
            b.observe(&inputs, (i & 1) as u8);
            second_a.observe(&second, (i & 1) as u8);
            second_b.observe(&second, (i & 1) as u8);
        }
        assert_eq!(a.weights, b.weights);
        assert_eq!(second_a.weights, second_b.weights);
        let (_, observed) =
            crate::core::diagnostics::capture(|| Mixer::new(128, [0; INPUTS], 22).unwrap());
        assert_eq!(observed.heap.peak, Mixer::memory_required(128));
    }
}
