//! Original deterministic fixed-point logistic lookup generation.
//! Provenance: the mathematical squash/stretch definitions described in
//! Mahoney, Data Compression Explained. Constants and tables are generated
//! here from integer Taylor arithmetic; no PAQ/CMIX tables were copied.

/// Largest stretched magnitude, in logit units of 1/256 (logit 16.0).
pub(super) const LIMIT: i32 = 4096;
/// Logit 1.0, the value of a constant (bias) mixer input.
pub(super) const UNIT: i16 = 256;
/// Probability 1/2 on the 16-bit scale.
pub(super) const HALF: u16 = 32768;
/// Largest representable probability on the 16-bit scale.
pub(super) const MAX: u16 = 65535;
/// Probability 1.0 on the 16-bit scale, the target of a zero bit.
pub(super) const PROBABILITY_ONE: i32 = 1 << 16;
const ONE: u128 = 1 << 48;

const fn exponential_step() -> u128 {
    // exp(-1/256), evaluated as an alternating series in Q48.
    let mut sum = ONE as i128;
    let mut term = ONE;
    let mut n = 1u128;
    while n <= 12 {
        term = term / 256 / n;
        if n & 1 == 1 {
            sum -= term as i128;
        } else {
            sum += term as i128;
        }
        n += 1;
    }
    sum as u128
}

const STEP: u128 = exponential_step();

const fn squash_table() -> [u16; 8193] {
    let mut table = [0u16; 8193];
    let mut weight = ONE;
    let mut index = 0usize;
    while index <= LIMIT as usize {
        let denominator = ONE + weight;
        let scaled = (65536 * ONE + denominator / 2) / denominator;
        let probability = if scaled > 65535 { 65535 } else { scaled as u16 };
        table[LIMIT as usize + index] = probability;
        table[LIMIT as usize - index] = (65536u32 - probability as u32) as u16;
        weight = (weight * STEP + ONE / 2) / ONE;
        index += 1;
    }
    table
}

static SQUASH: [u16; 8193] = squash_table();

const fn stretch_table() -> [i16; 65536] {
    let squash = squash_table();
    let mut table = [0i16; 65536];
    let mut probability = 32768usize;
    let mut index = LIMIT as usize;
    while probability <= 65535 {
        while index + 1 < squash.len() && (squash[index + 1] as usize) < probability {
            index += 1;
        }
        let lower = squash[index] as usize;
        let upper = squash[index + 1] as usize;
        let chosen = if probability - lower <= upper - probability {
            index
        } else {
            index + 1
        };
        let stretched = (chosen as i32 - LIMIT) as i16;
        table[probability] = stretched;
        table[65536 - probability] = -stretched;
        probability += 1;
    }
    table[0] = table[1];
    table
}

static STRETCH: [i16; 65536] = stretch_table();

/// Logit units are 1/256; probabilities are multiples of 1/65536.
#[inline]
pub(super) fn squash(logit: i32) -> u16 {
    SQUASH[(logit.clamp(-LIMIT, LIMIT) + LIMIT) as usize]
}

#[inline]
pub(super) fn stretch(probability: u16) -> i16 {
    STRETCH[usize::from(probability)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probabilities_are_strict_monotonic_and_symmetric() {
        assert_eq!(squash(0), 32768);
        assert_eq!(squash(256), 47911);
        assert_eq!(squash(i32::MIN), 1);
        assert_eq!(squash(i32::MAX), 65535);
        for logit in -LIMIT..=LIMIT {
            let p = squash(logit);
            assert!(p > 0);
            assert_eq!(u32::from(p) + u32::from(squash(-logit)), 65536);
            assert!(p <= squash(logit + 1));
        }
    }

    #[test]
    fn inverse_is_monotonic_and_bounded_by_quantization() {
        assert_eq!(stretch(32768), 0);
        for probability in 1..=65535u32 {
            let p = probability as u16;
            let logit = stretch(p);
            assert_eq!(logit, -stretch((65536 - probability) as u16));
            if probability < 65535 {
                assert!(logit <= stretch(p + 1));
            }
            assert!((i32::from(squash(i32::from(logit))) - probability as i32).abs() <= 33);
        }
        for logit in -512..=512i32 {
            assert!((i32::from(stretch(squash(logit))) - logit).abs() <= 1);
        }
    }
}
