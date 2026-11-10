//! Original Rust generator for bounded bit-history states.
//! Provenance: the public-domain ZPAQ Level 2 specification, section 3.3,
//! https://mattmahoney.net/dc/zpaq203.pdf. The state set and transitions are
//! generated from its count/last-bit rules; no compressor table is copied.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct History {
    pub(super) zeros: u8,
    pub(super) ones: u8,
    /// The last bit seen, or `NO_LAST` when the state does not record it.
    pub(super) last: u8,
}
/// `History::last` of a state that does not record the last bit.
const NO_LAST: u8 = 2;
/// Largest total count any state holds.
const MAX_TOTAL: u8 = 49;
/// States with totals from 1 to this also record the last bit.
const LAST_BIT_TOTAL: u8 = 17;
const EMPTY: History = History {
    zeros: 0,
    ones: 0,
    last: NO_LAST,
};

const fn allowed(a: u8, b: u8) -> bool {
    let (small, large) = if a < b { (a, b) } else { (b, a) };
    match small {
        0 => large <= 20,
        1 => large <= 48,
        2 => large <= 15,
        3 => large <= 8,
        4 => large <= 6,
        5 => large <= 5,
        _ => false,
    }
}
const fn histories() -> [History; 256] {
    let mut output = [EMPTY; 256];
    let mut count = 0usize;
    let mut total = 0u8;
    // Priority increases first with total count, then one count, then last bit.
    while total <= MAX_TOTAL {
        let mut ones = 0u8;
        while ones <= total {
            count = push_states(&mut output, count, total - ones, ones);
            ones += 1;
        }
        total += 1;
    }
    assert!(count == 255);
    output
}
/// Appends the states for one count pair in priority order, returning the
/// new state count.
const fn push_states(output: &mut [History; 256], mut count: usize, zeros: u8, ones: u8) -> usize {
    if !allowed(zeros, ones) {
        return count;
    }
    let total = zeros + ones;
    if total == 0 || total > LAST_BIT_TOTAL {
        output[count] = History {
            zeros,
            ones,
            last: NO_LAST,
        };
        return count + 1;
    }
    if zeros != 0 {
        output[count] = History {
            zeros,
            ones,
            last: 0,
        };
        count += 1;
    }
    if ones != 0 {
        output[count] = History {
            zeros,
            ones,
            last: 1,
        };
        count += 1;
    }
    count
}
pub(super) static HISTORIES: [History; 256] = histories();
const fn discount(count: u8) -> u8 {
    if count > 7 {
        7
    } else if count > 5 {
        count - 1
    } else {
        count
    }
}
const fn major_next(a: u8, b: u8, bit: u8) -> History {
    let (zeros, ones) = match (a, b, bit) {
        (20, 0, 0) => (20, 0),
        (48, 1, 0) => (48, 1),
        (15, 2, 0) => (8, 1),
        (8, 3, 0) => (6, 2),
        (8, 3, 1) => (5, 3),
        (6, 4, 0) => (5, 3),
        (5, 5, 0) => (5, 4),
        (5, 5, 1) => (4, 5),
        (_, _, 0) => (a + 1, discount(b)),
        _ => (discount(a), b + 1),
    };
    History {
        zeros,
        ones,
        last: if zeros + ones > LAST_BIT_TOTAL {
            NO_LAST
        } else {
            bit
        },
    }
}
const fn inverse(h: History) -> History {
    History {
        zeros: h.ones,
        ones: h.zeros,
        last: if h.last != NO_LAST {
            1 - h.last
        } else {
            NO_LAST
        },
    }
}
const fn updated(h: History, bit: u8) -> History {
    if h.zeros < h.ones {
        inverse(major_next(h.ones, h.zeros, 1 - bit))
    } else {
        major_next(h.zeros, h.ones, bit)
    }
}
const fn index(states: &[History; 256], wanted: History) -> u8 {
    let mut i = 0;
    while i < 255 {
        let h = states[i];
        if h.zeros == wanted.zeros && h.ones == wanted.ones && h.last == wanted.last {
            return i as u8;
        }
        i += 1;
    }
    panic!("bit-history transition left the allowed set")
}
const fn transitions() -> [[u8; 2]; 256] {
    let states = histories();
    let mut output = [[0; 2]; 256];
    let mut i = 0;
    while i < 256 {
        output[i] = [
            index(&states, updated(states[i], 0)),
            index(&states, updated(states[i], 1)),
        ];
        i += 1;
    }
    output
}
static NEXT: [[u8; 2]; 256] = transitions();
#[inline]
pub(super) fn next(state: u8, bit: u8) -> u8 {
    NEXT[usize::from(state)][usize::from(bit)]
}
const fn initial_probabilities() -> [u16; 256] {
    let histories = histories();
    let mut probabilities = [0; 256];
    let mut i = 0;
    while i < 256 {
        let h = histories[i];
        probabilities[i] =
            ((h.zeros as u32 * 2 + 1) * 32768 / (h.zeros as u32 + h.ones as u32 + 1)) as u16;
        i += 1;
    }
    probabilities
}
static INITIAL: [u16; 256] = initial_probabilities();
#[inline]
pub(super) fn initial_probability(state: u8) -> u16 {
    INITIAL[usize::from(state)]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_transitions_close_and_preserve_zero_one_symmetry() {
        assert_eq!(HISTORIES[0], EMPTY);
        for state in 0..255usize {
            let h = HISTORIES[state];
            assert!(allowed(h.zeros, h.ones));
            assert!(!HISTORIES[..state].contains(&h));
            let opposite = index(&HISTORIES, inverse(h));
            for bit in 0..2 {
                let n = next(state as u8, bit);
                assert!(n < 255);
                assert_eq!(
                    HISTORIES[usize::from(n)],
                    inverse(HISTORIES[usize::from(next(opposite, 1 - bit))])
                );
            }
        }
        let mut reached = [false; 256];
        reached[0] = true;
        for _ in 0..255 {
            for s in 0..255 {
                if reached[s] {
                    for b in 0..2 {
                        reached[usize::from(next(s as u8, b))] = true;
                    }
                }
            }
        }
        // The specification's 255 allowed triples include 36 that are not
        // reachable from the empty state under its discounting rules.
        assert_eq!(reached[..255].iter().filter(|&&x| x).count(), 219);
        assert!(!reached[255]);
    }
    #[test]
    fn runs_saturate_and_a_surprise_discards_old_evidence() {
        let mut state = 0;
        for _ in 0..1000 {
            state = next(state, 0);
        }
        assert_eq!(HISTORIES[usize::from(state)].zeros, 20);
        assert_eq!(
            HISTORIES[usize::from(next(state, 1))],
            History {
                zeros: 7,
                ones: 1,
                last: 1
            }
        );
        assert_eq!(initial_probability(0), 32768);
        assert!(initial_probability(state) > 63000);
    }
}
