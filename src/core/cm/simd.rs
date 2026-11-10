//! Original bounded integer dot product and training with scalar references.
//! Provenance: standard integer multiply/add operations and Intel's documented
//! AVX2 intrinsics. No compressor kernel or model coefficients are copied.

/// Which kernels the model may use. Both produce identical results;
/// `Scalar` exists so tests can compare the vector kernels against it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Dispatch {
    /// Use the fastest kernels the CPU supports.
    #[default]
    Native,
    /// Use only the portable scalar reference kernels (tests compare the
    /// vector kernels against them).
    #[cfg(test)]
    Scalar,
}
impl Dispatch {
    /// Whether AVX2 kernels may run.
    #[inline]
    pub(super) fn avx2(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            self == Self::Native && std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

pub(super) const INPUTS: usize = 64;
pub(super) const INPUT_LIMIT: i16 = 4096;
pub(super) const WEIGHT_LIMIT: i16 = 8192;

#[inline]
#[cfg(test)]
pub(super) fn scalar(inputs: &[i16; INPUTS], weights: &[i16; INPUTS]) -> i64 {
    scalar_active(inputs, weights, INPUTS)
}
fn scalar_active(inputs: &[i16; INPUTS], weights: &[i16; INPUTS], count: usize) -> i64 {
    inputs
        .iter()
        .zip(weights)
        .take(count)
        .map(|(&a, &b)| i64::from(a) * i64::from(b))
        .sum()
}
#[inline]
#[cfg(test)]
pub(super) fn dot(inputs: &[i16; INPUTS], weights: &[i16; INPUTS], dispatch: Dispatch) -> i64 {
    dot_active(inputs, weights, INPUTS, dispatch)
}
#[inline]
pub(super) fn dot_active(
    inputs: &[i16; INPUTS],
    weights: &[i16; INPUTS],
    count: usize,
    dispatch: Dispatch,
) -> i64 {
    assert!(count <= INPUTS);
    debug_assert!(inputs
        .iter()
        .all(|&x| (-INPUT_LIMIT..=INPUT_LIMIT).contains(&x)));
    debug_assert!(weights
        .iter()
        .all(|&x| (-WEIGHT_LIMIT..=WEIGHT_LIMIT).contains(&x)));
    #[cfg(target_arch = "x86_64")]
    if dispatch.avx2() {
        // SAFETY: runtime dispatch checks AVX2. Both fixed arrays contain 64
        // elements; every unaligned 16-element load stays within its array.
        // Per-lane sums fit signed 32 bits under the stated model bounds.
        return unsafe { avx2(inputs, weights, count) };
    }
    scalar_active(inputs, weights, count)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2(inputs: &[i16; INPUTS], weights: &[i16; INPUTS], count: usize) -> i64 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_si256();
    let mut at = 0;
    while at + 16 <= count {
        let a = _mm256_loadu_si256(inputs.as_ptr().add(at).cast());
        let b = _mm256_loadu_si256(weights.as_ptr().add(at).cast());
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(a, b));
        at += 16;
    }
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sum);
    lanes.into_iter().map(i64::from).sum::<i64>()
        + inputs[at..count]
            .iter()
            .zip(&weights[at..count])
            .map(|(&a, &b)| i64::from(a) * i64::from(b))
            .sum::<i64>()
}

/// One mixer weight update: the first `count` inputs are trained with the
/// prediction `error` (16-bit probability scale) shifted right by `shift`.
#[derive(Clone, Copy, Debug)]
pub(super) struct MixerUpdate {
    pub(super) count: usize,
    pub(super) error: i32,
    pub(super) shift: u8,
}

#[inline]
fn train_scalar_range(
    inputs: &[i16; INPUTS],
    weights: &mut [i16; INPUTS],
    start: usize,
    update: MixerUpdate,
) {
    let MixerUpdate {
        count,
        error,
        shift,
    } = update;
    let half = 1i64 << (shift - 1);
    for (weight, &input) in weights[start..count].iter_mut().zip(&inputs[start..count]) {
        let gradient = i64::from(error) * i64::from(input);
        let delta = if gradient >= 0 {
            (gradient + half) >> shift
        } else {
            -((-gradient + half) >> shift)
        };
        *weight = (i64::from(*weight) + delta)
            .clamp(-i64::from(WEIGHT_LIMIT), i64::from(WEIGHT_LIMIT)) as i16;
    }
}

#[inline]
pub(super) fn train_active(
    inputs: &[i16; INPUTS],
    weights: &mut [i16; INPUTS],
    update: MixerUpdate,
    dispatch: Dispatch,
) {
    assert!(update.count <= INPUTS);
    debug_assert!((20..=30).contains(&update.shift));
    debug_assert!((-65535..=65535).contains(&update.error));
    debug_assert!(inputs
        .iter()
        .all(|&x| (-INPUT_LIMIT..=INPUT_LIMIT).contains(&x)));
    debug_assert!(weights
        .iter()
        .all(|&x| (-WEIGHT_LIMIT..=WEIGHT_LIMIT).contains(&x)));
    #[cfg(target_arch = "x86_64")]
    if dispatch.avx2() {
        // SAFETY: runtime dispatch establishes AVX2; count is within both
        // arrays. Each vector accesses eight live i16 elements. Under the
        // model bounds, |error*input| <= 268,431,360; adding the largest
        // rounding half (2^29) still fits i32. Rounded updates and clamped
        // weights therefore match the i64 reference, including negative ties.
        unsafe { train_avx2(inputs, weights, update) };
        return;
    }
    train_scalar_range(inputs, weights, 0, update);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn train_avx2(inputs: &[i16; INPUTS], weights: &mut [i16; INPUTS], update: MixerUpdate) {
    use std::arch::x86_64::*;
    let MixerUpdate {
        count,
        error,
        shift,
    } = update;
    let error_vector = _mm256_set1_epi32(error);
    let half = _mm256_set1_epi32(1 << (shift - 1));
    let shift_vector = _mm256_set1_epi32(i32::from(shift));
    let lower = _mm256_set1_epi32(-i32::from(WEIGHT_LIMIT));
    let upper = _mm256_set1_epi32(i32::from(WEIGHT_LIMIT));
    let zero = _mm256_setzero_si256();
    let mut at = 0;
    while at + 8 <= count {
        let input = _mm256_cvtepi16_epi32(_mm_loadu_si128(inputs.as_ptr().add(at).cast()));
        let old = _mm256_cvtepi16_epi32(_mm_loadu_si128(weights.as_ptr().add(at).cast()));
        let gradient = _mm256_mullo_epi32(input, error_vector);
        let magnitude = _mm256_add_epi32(_mm256_abs_epi32(gradient), half);
        let delta = _mm256_sign_epi32(_mm256_srlv_epi32(magnitude, shift_vector), gradient);
        let updated =
            _mm256_min_epi32(upper, _mm256_max_epi32(lower, _mm256_add_epi32(old, delta)));
        let packed = _mm256_packs_epi32(updated, zero);
        // packs works independently in 128-bit lanes. Join the two groups
        // of four updated weights before storing exactly eight elements.
        let ordered = _mm256_permute4x64_epi64::<0xd8>(packed);
        _mm_storeu_si128(
            weights.as_mut_ptr().add(at).cast(),
            _mm256_castsi256_si128(ordered),
        );
        at += 8;
    }
    train_scalar_range(inputs, weights, at, update);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_and_scalar_sums_match_at_saturation_and_random_bounds() {
        for a in [-INPUT_LIMIT, 0, INPUT_LIMIT] {
            for b in [-WEIGHT_LIMIT, 0, WEIGHT_LIMIT] {
                let inputs = [a; INPUTS];
                let weights = [b; INPUTS];
                assert_eq!(
                    dot(&inputs, &weights, Dispatch::Native),
                    scalar(&inputs, &weights)
                );
                for count in 0..=INPUTS {
                    assert_eq!(
                        dot_active(&inputs, &weights, count, Dispatch::Native),
                        scalar_active(&inputs, &weights, count)
                    );
                }
                assert_eq!(
                    dot(&inputs, &weights, Dispatch::Scalar),
                    scalar(&inputs, &weights)
                );
            }
        }
        let mut random = 0x3a71_92bdu32;
        for _ in 0..1000 {
            let mut inputs = [0; INPUTS];
            let mut weights = [0; INPUTS];
            for i in 0..INPUTS {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                inputs[i] = (random % 8193) as i16 - 4096;
                weights[i] = ((random >> 8) % 16385) as i16 - 8192;
            }
            assert_eq!(
                dot(&inputs, &weights, Dispatch::Native),
                scalar(&inputs, &weights)
            );
        }
    }

    #[test]
    fn vector_training_matches_reference_for_all_lengths_rates_and_bounds() {
        let mut seed = 0x3d71_e8a9u32;
        for shift in 20..=30 {
            for error in [-65535, -32768, -1, 0, 1, 32768, 65535] {
                for count in 0..=INPUTS {
                    let mut inputs = [0; INPUTS];
                    let mut weights = [0; INPUTS];
                    for at in 0..INPUTS {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        inputs[at] = match at % 7 {
                            0 => -INPUT_LIMIT,
                            1 => INPUT_LIMIT,
                            2 => 64,
                            3 => -64,
                            _ => (seed % 8193) as i16 - 4096,
                        };
                        weights[at] = match at % 5 {
                            0 => -WEIGHT_LIMIT,
                            1 => WEIGHT_LIMIT,
                            _ => ((seed >> 8) % 16385) as i16 - 8192,
                        };
                    }
                    let update = MixerUpdate {
                        count,
                        error,
                        shift,
                    };
                    let mut expected = weights;
                    train_scalar_range(&inputs, &mut expected, 0, update);
                    let mut vector = weights;
                    train_active(&inputs, &mut vector, update, Dispatch::Native);
                    assert_eq!(
                        vector, expected,
                        "count={count}, shift={shift}, error={error}"
                    );
                    let mut forced = weights;
                    train_active(&inputs, &mut forced, update, Dispatch::Scalar);
                    assert_eq!(forced, expected);
                }
            }
        }
    }
}
