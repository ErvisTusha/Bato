//! Original state-conditioned affine logit calibration for the CM ensemble.
//! Provenance: the ISSE concept in Mahoney's ZPAQ format specification 2.06,
//! a bit-history state selects a two-input adaptive mixer with
//! one constant input. Quantization, bounds, initialization and Rust are
//! original; no reference decoder code, coefficients or tables are copied.

use super::logistic;

/// log2 of weight 1.0.
const SCALE_BITS: u32 = 14;
/// Weight 1.0: a fresh ISSE passes its input through unchanged.
const SCALE: i32 = 1 << SCALE_BITS;
/// Weight saturation bound (just under 2.0).
const LIMIT: i32 = i16::MAX as i32;
/// Right shift of `error * input` giving one weight update.
const LEARNING_SHIFT: u32 = 22;

pub(super) struct Isse {
    weights: [[i16; 2]; 256],
    state: usize,
    input: i16,
    prediction: u16,
}

#[inline]
fn rounded(value: i64, shift: u32) -> i32 {
    let half = 1i64 << (shift - 1);
    if value < 0 {
        -(((-value + half) >> shift) as i32)
    } else {
        ((value + half) >> shift) as i32
    }
}

impl Isse {
    pub(super) fn new() -> Self {
        Self {
            weights: [[SCALE as i16, 0]; 256],
            state: 0,
            input: 0,
            prediction: logistic::HALF,
        }
    }

    #[inline]
    pub(super) fn predict(&mut self, probability: u16, state: u8) -> u16 {
        self.state = usize::from(state);
        self.input = logistic::stretch(probability);
        let weights = self.weights[self.state];
        // |weight| <= 32767 and |input| <= 4096, so the complete sum is
        // below 2^28. Explicit i64 intermediates keep reference semantics.
        let sum = i64::from(weights[0]) * i64::from(self.input)
            + i64::from(weights[1]) * i64::from(logistic::UNIT);
        let logit = rounded(sum, SCALE_BITS).clamp(-logistic::LIMIT, logistic::LIMIT);
        self.prediction = logistic::squash(logit);
        self.prediction
    }

    #[inline]
    pub(super) fn observe(&mut self, bit: u8) {
        debug_assert!(bit <= 1);
        let error = if bit == 0 {
            logistic::PROBABILITY_ONE
        } else {
            0
        } - i32::from(self.prediction);
        // The largest product is 65535*4096 < 2^28. This rate is an SGD
        // step of 1/4096 in natural-logit units, with symmetric rounding.
        let changes = [
            rounded(i64::from(error) * i64::from(self.input), LEARNING_SHIFT),
            rounded(i64::from(error) * i64::from(logistic::UNIT), LEARNING_SHIFT),
        ];
        for (weight, change) in self.weights[self.state].iter_mut().zip(changes) {
            *weight = (i32::from(*weight) + change).clamp(-LIMIT, LIMIT) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_calibration_and_extreme_weights_stay_bounded() {
        let mut model = Isse::new();
        for probability in 1..=65535u16 {
            let prediction = model.predict(probability, 19);
            assert_eq!(
                prediction,
                logistic::squash(i32::from(logistic::stretch(probability)))
            );
        }
        for slope in [-32767, 0, 32767] {
            for bias in [-32767, 0, 32767] {
                model.weights[255] = [slope, bias];
                for probability in [1, 2, 16384, 32768, 49152, 65534, 65535] {
                    let prediction = model.predict(probability, 255);
                    assert!((1..=65535).contains(&prediction));
                    model.observe(0);
                    model.predict(probability, 255);
                    model.observe(1);
                    assert!(model.weights[255]
                        .iter()
                        .all(|&w| i32::from(w).abs() <= LIMIT));
                }
            }
        }
        assert_eq!(rounded(3, 1), 2);
        assert_eq!(rounded(-3, 1), -2);
    }

    #[test]
    fn updates_learn_both_bit_directions_and_only_the_selected_state() {
        let mut zeros = Isse::new();
        let mut ones = Isse::new();
        let untouched = zeros.weights[1];
        for _ in 0..20000 {
            zeros.predict(32768, 0);
            zeros.observe(0);
            ones.predict(32768, 0);
            ones.observe(1);
        }
        assert!(zeros.predict(32768, 0) > 45000);
        assert!(ones.predict(32768, 0) < 20536);
        assert_eq!(zeros.weights[1], untouched);
        assert_eq!(ones.weights[1], untouched);
        assert_eq!(zeros.predict(32768, 1), 32768);
        assert_eq!(ones.predict(32768, 1), 32768);
    }

    #[test]
    fn nonneutral_input_trains_slope_and_bias_in_the_target_direction() {
        let mut model = Isse::new();
        let high = model.predict(50000, 7);
        let low = model.predict(15536, 9);
        for _ in 0..2000 {
            model.predict(50000, 7);
            model.observe(1);
            model.predict(15536, 9);
            model.observe(0);
        }
        assert!(model.weights[7][0] < SCALE as i16);
        assert!(model.weights[7][1] < 0);
        assert!(model.weights[9][0] < SCALE as i16);
        assert!(model.weights[9][1] > 0);
        assert!(model.predict(50000, 7) < high);
        assert!(model.predict(15536, 9) > low);
    }
}
