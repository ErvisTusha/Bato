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
    /// Resolved AVX2 kernel dispatch.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// Use only the portable scalar reference kernels.
    Scalar,
}
impl Dispatch {
    /// Resolves `Native` once to the best kernel supported on this CPU.
    #[inline]
    pub(super) fn resolve(self) -> Self {
        match self {
            Self::Native => {
                #[cfg(target_arch = "x86_64")]
                if std::is_x86_feature_detected!("avx2") {
                    return Self::Avx2;
                }
                Self::Scalar
            }
            other => other,
        }
    }
    /// Whether AVX2 kernels were resolved for this instance.
    #[inline]
    pub(super) fn is_resolved_avx2(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            matches!(self, Self::Avx2)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
    /// Whether AVX2 kernels may run.
    #[inline]
    pub(super) fn avx2(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            self == Self::Avx2 || (self == Self::Native && std::is_x86_feature_detected!("avx2"))
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
#[inline]
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
    // Widen each 4x i32 half to 4x i64 in registers before reducing so extreme
    // inputs cannot wrap an intermediate 32-bit horizontal sum.
    let lo128 = _mm256_castsi256_si128(sum);
    let hi128 = _mm256_extracti128_si256::<1>(sum);
    let sum64 = _mm256_add_epi64(_mm256_cvtepi32_epi64(lo128), _mm256_cvtepi32_epi64(hi128));
    let sum128 = _mm_add_epi64(
        _mm256_castsi256_si128(sum64),
        _mm256_extracti128_si256::<1>(sum64),
    );
    let high64 = _mm_unpackhi_epi64(sum128, sum128);
    let total = _mm_cvtsi128_si64(_mm_add_epi64(sum128, high64));
    if at < count {
        total
            + inputs[at..count]
                .iter()
                .zip(&weights[at..count])
                .map(|(&a, &b)| i64::from(a) * i64::from(b))
                .sum::<i64>()
    } else {
        total
    }
}

/// One mixer weight update: the first `count` inputs are trained with the
/// prediction `error` (16-bit probability scale) shifted right by `shift`.
#[derive(Clone, Copy, Debug)]
pub(super) struct MixerUpdate {
    pub(super) count: usize,
    pub(super) error: i32,
    pub(super) shift: u8,
}

/// Updates one mixer weight using the exact symmetric-rounded SGD rule.
#[inline]
pub(super) fn train_one(input: i16, weight: &mut i16, error: i32, shift: u8) {
    debug_assert!((20..=30).contains(&shift));
    debug_assert!((-65535..=65535).contains(&error));
    debug_assert!((-INPUT_LIMIT..=INPUT_LIMIT).contains(&input));
    let half = 1i64 << (shift - 1);
    let gradient = i64::from(error) * i64::from(input);
    let delta = if gradient >= 0 {
        (gradient + half) >> shift
    } else {
        -((-gradient + half) >> shift)
    };
    *weight = (i64::from(*weight) + delta).clamp(-i64::from(WEIGHT_LIMIT), i64::from(WEIGHT_LIMIT))
        as i16;
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
    for (weight, &input) in weights[start..count].iter_mut().zip(&inputs[start..count]) {
        train_one(input, weight, error, shift);
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
unsafe fn update_eight(
    inputs: *const i16,
    weights: *const i16,
    params: (
        std::arch::x86_64::__m256i,
        std::arch::x86_64::__m256i,
        std::arch::x86_64::__m128i,
    ),
    bounds: (std::arch::x86_64::__m256i, std::arch::x86_64::__m256i),
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let (error_vector, half, shift_count) = params;
    let (lower, upper) = bounds;
    let input = _mm256_cvtepi16_epi32(_mm_loadu_si128(inputs.cast()));
    let old = _mm256_cvtepi16_epi32(_mm_loadu_si128(weights.cast()));
    let gradient = _mm256_mullo_epi32(input, error_vector);
    let magnitude = _mm256_add_epi32(_mm256_abs_epi32(gradient), half);
    let delta = _mm256_sign_epi32(_mm256_srl_epi32(magnitude, shift_count), gradient);
    _mm256_min_epi32(upper, _mm256_max_epi32(lower, _mm256_add_epi32(old, delta)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn train_avx2(inputs: &[i16; INPUTS], weights: &mut [i16; INPUTS], update: MixerUpdate) {
    use std::arch::x86_64::*;
    let MixerUpdate {
        count,
        error,
        shift,
    } = update;
    let params = (
        _mm256_set1_epi32(error),
        _mm256_set1_epi32(1 << (shift - 1)),
        _mm_cvtsi32_si128(i32::from(shift)),
    );
    let bounds = (
        _mm256_set1_epi32(-i32::from(WEIGHT_LIMIT)),
        _mm256_set1_epi32(i32::from(WEIGHT_LIMIT)),
    );
    let zero = _mm256_setzero_si256();
    let mut at = 0;
    while at + 16 <= count {
        // SAFETY: at + 16 <= count <= INPUTS, so both 8-element i16 loads
        // and the 16-element i16 store remain inside inputs and weights.
        let updated_lo = update_eight(
            inputs.as_ptr().add(at),
            weights.as_ptr().add(at),
            params,
            bounds,
        );
        let updated_hi = update_eight(
            inputs.as_ptr().add(at + 8),
            weights.as_ptr().add(at + 8),
            params,
            bounds,
        );
        let packed = _mm256_packs_epi32(updated_lo, updated_hi);
        // packs works independently in 128-bit lanes; reorder to restore [lo, hi].
        let ordered = _mm256_permute4x64_epi64::<0xd8>(packed);
        _mm256_storeu_si256(weights.as_mut_ptr().add(at).cast(), ordered);
        at += 16;
    }
    if at < count {
        while at + 8 <= count {
            // SAFETY: at + 8 <= count <= INPUTS.
            let updated = update_eight(
                inputs.as_ptr().add(at),
                weights.as_ptr().add(at),
                params,
                bounds,
            );
            let packed = _mm256_packs_epi32(updated, zero);
            let ordered = _mm256_permute4x64_epi64::<0xd8>(packed);
            _mm_storeu_si128(
                weights.as_mut_ptr().add(at).cast(),
                _mm256_castsi256_si128(ordered),
            );
            at += 8;
        }
        train_scalar_range(inputs, weights, at, update);
    }
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
