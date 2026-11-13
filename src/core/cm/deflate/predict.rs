//! Predicts which token a conventional deflate encoder emits at a position,
//! so a recompressor stores only where a stream deviates from the prediction.
//! Provenance: the hash-chain longest-match search and one-step lazy
//! evaluation commonly used by LZ77 deflate encoders, as documented in
//! RFC 1951 section 4 and the zlib manual's algorithm notes (level table:
//! good, lazy, nice and chain lengths); original implementation.

use super::Token;

/// Positions the hash chains remember: the deflate window.
const WINDOW: usize = 32768;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
/// Farthest distance a match may use: window minus the lookahead reserve.
const MAX_DISTANCE: usize = 32768 - (MAX_MATCH + MIN_MATCH + 1);
/// Length-3 matches farther than this are not worth their bits.
const TOO_FAR: usize = 4096;

/// Encoder behaviour the predictor imitates; chosen per stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Params {
    /// Candidates examined per search.
    pub(super) chain: usize,
    /// Previous match length from which the chain is cut to a quarter.
    pub(super) good: usize,
    /// Greedy mode: longest length whose interior positions are indexed.
    /// Lazy mode: longest match for which the next position is still tried.
    pub(super) lazy_max: usize,
    /// Match length that ends a search at once.
    pub(super) nice: usize,
    pub(super) greedy: bool,
    /// Hash table bits of the imitated encoder (memory level plus seven).
    pub(super) hash_bits: u32,
}

const fn level(chain: usize, good: usize, lazy_max: usize, nice: usize, greedy: bool) -> Params {
    Params {
        chain,
        good,
        lazy_max,
        nice,
        greedy,
        hash_bits: 15,
    }
}

impl Params {
    pub(super) const fn with_hash_bits(self, hash_bits: u32) -> Self {
        Self { hash_bits, ..self }
    }
}

/// Parameter sets of the nine conventional effort levels.
pub(super) const LEVELS: [Params; 9] = [
    level(4, 4, 4, 8, true),
    level(8, 4, 5, 16, true),
    level(32, 4, 6, 32, true),
    level(16, 4, 4, 16, false),
    level(32, 8, 16, 32, false),
    level(128, 8, 16, 128, false),
    level(256, 8, 32, 128, false),
    level(1024, 32, 128, 258, false),
    level(4096, 32, 258, 258, false),
];

/// Exact longest common prefix, bounded by the DEFLATE match limit.
/// The first differing byte in a 64-bit word is found without an inner
/// per-byte branch, which matters for the deepest chain-search levels.
#[inline]
fn match_length(data: &[u8], start: usize, at: usize, limit: usize) -> usize {
    let mut matched = 0;
    while matched + 8 <= limit {
        let left = u64::from_le_bytes(
            data[start + matched..start + matched + 8]
                .try_into()
                .unwrap(),
        );
        let right = u64::from_le_bytes(data[at + matched..at + matched + 8].try_into().unwrap());
        let different = left ^ right;
        if different != 0 {
            return matched + (different.trailing_zeros() / 8) as usize;
        }
        matched += 8;
    }
    while matched < limit && data[start + matched] == data[at + matched] {
        matched += 1;
    }
    matched
}

#[derive(Clone)]
pub(super) struct Predictor<'a> {
    data: &'a [u8],
    params: Params,
    head: Vec<u32>,
    prev: Vec<u32>,
    inserted: usize,
    /// Match already found for `.0` while looking ahead from the position before.
    carried: Option<(usize, usize, usize)>,
}

impl<'a> Predictor<'a> {
    pub(super) fn new(data: &'a [u8], params: Params) -> Self {
        Self {
            data,
            params,
            head: vec![0; 1 << params.hash_bits],
            prev: vec![0; WINDOW],
            inserted: 0,
            carried: None,
        }
    }

    fn hash(&self, at: usize) -> usize {
        let shift = self.params.hash_bits.div_ceil(3);
        ((usize::from(self.data[at]) << (2 * shift))
            ^ (usize::from(self.data[at + 1]) << shift)
            ^ usize::from(self.data[at + 2]))
            & ((1 << self.params.hash_bits) - 1)
    }

    /// Indexes every position before `end` that has three bytes.
    fn insert_until(&mut self, end: usize) {
        while self.inserted < end {
            let at = self.inserted;
            if at + MIN_MATCH <= self.data.len() {
                let bucket = self.hash(at);
                self.prev[at % WINDOW] = self.head[bucket];
                self.head[bucket] = at as u32 + 1;
            }
            self.inserted += 1;
        }
    }

    /// First match at `at` strictly longer than `floor`, nearest first on ties.
    fn search(&self, at: usize, floor: usize) -> Option<(usize, usize)> {
        if at + MIN_MATCH > self.data.len() {
            return None;
        }
        let limit = (self.data.len() - at).min(MAX_MATCH);
        let chain = if floor >= self.params.good {
            self.params.chain / 4
        } else {
            self.params.chain
        };
        let (mut best, mut best_distance) = (floor, 0);
        let mut candidate = self.head[self.hash(at)];
        for _ in 0..chain {
            if candidate == 0 {
                break;
            }
            let start = candidate as usize - 1;
            if at - start > MAX_DISTANCE {
                break;
            }
            let length = match_length(self.data, start, at, limit);
            if length > best {
                (best, best_distance) = (length, at - start);
                if best >= self.params.nice || best >= limit {
                    break;
                }
            }
            candidate = self.prev[start % WINDOW];
        }
        (best_distance != 0).then_some((best, best_distance))
    }

    fn usable(found: Option<(usize, usize)>) -> Option<(usize, usize)> {
        found.filter(|&(length, distance)| length > MIN_MATCH || distance <= TOO_FAR)
    }

    pub(super) fn predict(&mut self, at: usize) -> Token {
        self.insert_until(at);
        let current = match self.carried.take() {
            Some((position, length, distance)) if position == at => Some((length, distance)),
            _ if self.params.greedy => self.search(at, MIN_MATCH - 1),
            _ => Self::usable(self.search(at, MIN_MATCH - 1)),
        };
        let Some((length, distance)) = current else {
            return Token::Literal(self.data[at]);
        };
        if !self.params.greedy && length < self.params.lazy_max {
            self.insert_until(at + 1);
            if let Some((next_length, next_distance)) = Self::usable(self.search(at + 1, length)) {
                self.carried = Some((at + 1, next_length, next_distance));
                return Token::Literal(self.data[at]);
            }
        }
        Token::Match {
            length: length as u16,
            distance: distance as u16,
        }
    }

    /// Moves past `token` emitted at `at`, indexing the positions it covers.
    pub(super) fn advance(&mut self, at: usize, token: Token) {
        self.insert_until(at + 1);
        if let Token::Match { length, .. } = token {
            let length = usize::from(length);
            if !self.params.greedy || length <= self.params.lazy_max {
                self.insert_until(at + length);
            } else {
                self.inserted = at + length;
            }
        }
    }

    /// Moves past `length` stored bytes, indexing every position.
    pub(super) fn skip(&mut self, at: usize, length: usize) {
        self.carried = None;
        self.insert_until(at + length);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_match_is_identical_to_bytewise_reference() {
        for len in 9..513usize {
            let mut data: Vec<u8> = (0..len)
                .map(|i| (i.wrapping_mul(37) ^ (i >> 4) ^ (i >> 8)) as u8)
                .collect();
            for at in [1, 3, 7, 8, 11, len / 2] {
                if at >= len {
                    continue;
                }
                let limit = (len - at).min(MAX_MATCH);
                let expected = (0..limit).take_while(|&i| data[i] == data[at + i]).count();
                assert_eq!(match_length(&data, 0, at, limit), expected);
            }
            data.fill(0x42);
            for at in [1, 8, len / 2] {
                let limit = (len - at).min(MAX_MATCH);
                assert_eq!(match_length(&data, 0, at, limit), limit);
            }
        }
    }
}
