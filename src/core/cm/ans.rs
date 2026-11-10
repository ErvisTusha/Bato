//! Original byte-renormalized range ANS with encoder-built frequency tables.
//! Provenance: Duda, arXiv:1311.2540; Townsend, arXiv:2001.09186.
//! Implemented from the published state equations, without third-party code.
use super::{lz, varint};
use crate::core::{memory::try_capacity, types::BatoError};
const SCALE: u32 = 12;
const TOTAL: u32 = 1 << SCALE;
const LOWER: u32 = 1 << 23;
const UPPER: u32 = LOWER << 8;

fn frequencies(raw: &[u8]) -> [u16; 256] {
    let mut counts = [0u64; 256];
    for &b in raw {
        counts[b as usize] += 1;
    }
    let mut f = [0u16; 256];
    let mut sum = 0u32;
    for i in 0..256 {
        if counts[i] > 0 {
            f[i] = ((counts[i] * u64::from(TOTAL) / raw.len() as u64).max(1)) as u16;
            sum += u32::from(f[i]);
        }
    }
    let mut order: Vec<usize> = (0..256).filter(|&i| counts[i] > 0).collect();
    order.sort_by_key(|&i| {
        (
            std::cmp::Reverse(counts[i] * u64::from(TOTAL) % raw.len() as u64),
            i,
        )
    });
    if sum < TOTAL {
        for &i in order.iter().take((TOTAL - sum) as usize) {
            f[i] += 1;
        }
    }
    while sum > TOTAL {
        let mut i = 0;
        for j in 1..256 {
            if (f[j], counts[j], std::cmp::Reverse(j)) > (f[i], counts[i], std::cmp::Reverse(i)) {
                i = j;
            }
        }
        // sum > TOTAL with at most 256 symbols guarantees this frequency > 1.
        debug_assert!(f[i] > 1);
        f[i] -= 1;
        sum -= 1;
    }
    f
}
pub(super) fn encode(raw: &[u8]) -> Option<Vec<u8>> {
    if raw.is_empty() {
        return None;
    }
    let f = frequencies(raw);
    let mut start = [0u32; 256];
    let mut sum = 0;
    let mut head = Vec::new();
    varint::push(&mut head, f.iter().filter(|&&n| n > 0).count());
    for (i, &n) in f.iter().enumerate() {
        start[i] = sum;
        sum += u32::from(n);
        if n > 0 {
            head.push(i as u8);
            varint::push(&mut head, usize::from(n) - 1);
        }
    }
    let mut bytes = Vec::new();
    let mut state = LOWER;
    for &b in raw.iter().rev() {
        let freq = u32::from(f[b as usize]);
        let threshold = ((LOWER >> SCALE) << 8) * freq;
        while state >= threshold {
            bytes.push(state as u8);
            state >>= 8;
            if head.len() + bytes.len() + 4 >= raw.len() {
                return None;
            }
        }
        state = ((state / freq) << SCALE) + (state % freq) + start[b as usize];
    }
    if head.len() + 4 + bytes.len() >= raw.len() {
        return None;
    }
    head.extend_from_slice(&state.to_le_bytes());
    head.extend(bytes.into_iter().rev());
    Some(head)
}

pub(super) struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    state: u32,
    frequency: [u16; 256],
    start: [u16; 256],
    symbols: Vec<u8>,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, BatoError> {
        let mut at = 0;
        let count = varint::take(bytes, &mut at)?;
        if !(1..=256).contains(&count) {
            return Err(invalid());
        }
        let mut f = [0u16; 256];
        let mut start = [0u16; 256];
        let mut previous = None;
        let mut sum = 0usize;
        let mut symbols = try_capacity(TOTAL as usize)?;
        for _ in 0..count {
            let &b = bytes.get(at).ok_or_else(invalid)?;
            at += 1;
            let n = varint::take(bytes, &mut at)?
                .checked_add(1)
                .ok_or_else(invalid)?;
            if previous.is_some_and(|v| b <= v) || n > TOTAL as usize - sum {
                return Err(invalid());
            }
            previous = Some(b);
            f[b as usize] = n as u16;
            start[b as usize] = sum as u16;
            sum += n;
            symbols.resize(sum, b);
        }
        if sum != TOTAL as usize {
            return Err(invalid());
        }
        let state = u32::from_le_bytes(
            bytes
                .get(at..at + 4)
                .ok_or_else(invalid)?
                .try_into()
                .map_err(|_| invalid())?,
        );
        at += 4;
        if !(LOWER..UPPER).contains(&state) {
            return Err(invalid());
        }
        Ok(Self {
            bytes,
            at,
            state,
            frequency: f,
            start,
            symbols,
        })
    }
    fn byte(&mut self) -> Result<u8, BatoError> {
        let slot = self.state & (TOTAL - 1);
        let b = self.symbols[slot as usize];
        self.state = u32::from(self.frequency[b as usize]) * (self.state >> SCALE) + slot
            - u32::from(self.start[b as usize]);
        while self.state < LOWER {
            let &v = self.bytes.get(self.at).ok_or_else(invalid)?;
            self.at += 1;
            self.state = (self.state << 8) | u32::from(v);
        }
        Ok(b)
    }
    fn finish(&self) -> Result<(), BatoError> {
        if self.at != self.bytes.len() || self.state != LOWER {
            Err(invalid())
        } else {
            Ok(())
        }
    }
}

/// A bounded stream decoder. Frequency tables are fixed; no adaptive training.
pub(super) enum Part<'a> {
    Owned {
        bytes: Vec<u8>,
        at: usize,
    },
    Raw {
        bytes: &'a [u8],
        at: usize,
    },
    Run {
        value: u8,
        left: usize,
    },
    Ans {
        reader: Box<Reader<'a>>,
        left: usize,
    },
}
impl<'a> Part<'a> {
    pub fn parse(bytes: &'a [u8], at: &mut usize, limit: usize) -> Result<Self, BatoError> {
        let n = varint::take(bytes, at)?;
        if n > limit {
            return Err(invalid());
        }
        let &kind = bytes.get(*at).ok_or_else(invalid)?;
        *at += 1;
        let c = varint::take(bytes, at)?;
        let end = at.checked_add(c).ok_or_else(invalid)?;
        let payload = bytes.get(*at..end).ok_or_else(invalid)?;
        *at = end;
        match kind {
            0 if c == n => Ok(Self::Raw {
                bytes: payload,
                at: 0,
            }),
            1 if c == 1 && n > 0 => Ok(Self::Run {
                value: payload[0],
                left: n,
            }),
            2 if n > 0 => Ok(Self::Ans {
                reader: Box::new(Reader::new(payload)?),
                left: n,
            }),
            3 => Ok(Self::Owned {
                bytes: lz::decode(payload, n)?,
                at: 0,
            }),
            4 => {
                let mut cursor = 0;
                let size = varint::take(payload, &mut cursor)?;
                if size > n.saturating_add(10) {
                    return Err(invalid());
                }
                let mut decoder = Reader::new(&payload[cursor..])?;
                let mut tokens = try_capacity(size)?;
                for _ in 0..size {
                    tokens.push(decoder.byte()?);
                }
                decoder.finish()?;
                Ok(Self::Owned {
                    bytes: lz::decode(&tokens, n)?,
                    at: 0,
                })
            }
            _ => Err(invalid()),
        }
    }
    pub fn remaining(&self) -> usize {
        match self {
            Self::Raw { bytes, at } => bytes.len() - at,
            Self::Owned { bytes, at } => bytes.len() - at,
            Self::Run { left, .. } | Self::Ans { left, .. } => *left,
        }
    }
    pub fn byte(&mut self) -> Result<u8, BatoError> {
        match self {
            Self::Owned { bytes, at } => {
                let &b = bytes.get(*at).ok_or_else(invalid)?;
                *at += 1;
                Ok(b)
            }
            Self::Raw { bytes, at } => {
                let &b = bytes.get(*at).ok_or_else(invalid)?;
                *at += 1;
                Ok(b)
            }
            Self::Run { value, left } if *left > 0 => {
                *left -= 1;
                Ok(*value)
            }
            Self::Ans { reader, left } if *left > 0 => {
                let b = reader.byte()?;
                *left -= 1;
                Ok(b)
            }
            _ => Err(invalid()),
        }
    }
    pub fn integer(&mut self) -> Result<u64, BatoError> {
        let mut value = 0;
        for i in 0..10 {
            let b = self.byte()?;
            if i == 9 && b > 1 {
                return Err(invalid());
            }
            value |= u64::from(b & 127) << (i * 7);
            if b < 128 {
                if i > 0 && b == 0 {
                    return Err(invalid());
                }
                return Ok(value);
            }
        }
        Err(invalid())
    }
    pub fn finish(&self) -> Result<(), BatoError> {
        if self.remaining() != 0 {
            return Err(invalid());
        }
        if let Self::Ans { reader, .. } = self {
            reader.finish()?;
        }
        Ok(())
    }
}
pub(super) fn packet(raw: &[u8]) -> Result<Vec<u8>, BatoError> {
    let mut kind = 0;
    let mut body: Option<Vec<u8>> = None;
    if !raw.is_empty() && raw.iter().all(|&b| b == raw[0]) {
        if raw.len() > 1 {
            kind = 1;
            body = Some(vec![raw[0]]);
        }
    } else {
        if let Some(coded) = encode(raw) {
            kind = 2;
            body = Some(coded);
        }
        let copies = lz::encode(raw)?;
        let best = body.as_ref().map_or(raw.len(), Vec::len);
        if copies.len() < best {
            kind = 3;
            body = Some(copies.clone());
        }
        if let Some(coded) = encode(&copies) {
            let mut payload = Vec::new();
            varint::push(&mut payload, copies.len());
            payload.extend_from_slice(&coded);
            if payload.len() < body.as_ref().map_or(raw.len(), Vec::len) {
                kind = 4;
                body = Some(payload);
            }
        }
    }
    let payload = body.as_deref().unwrap_or(raw);
    let mut out = Vec::new();
    varint::push(&mut out, raw.len());
    out.push(kind);
    varint::push(&mut out, payload.len());
    out.extend_from_slice(payload);
    Ok(out)
}

fn invalid() -> BatoError {
    BatoError::CorruptStream("invalid static ANS stream".into())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packets_and_canonical_state() {
        let mut random = 0x734ab93u64;
        let raw: Vec<u8> = (0..100000)
            .map(|i| {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                if i % 5 == 0 {
                    random as u8
                } else {
                    (random % 7) as u8
                }
            })
            .collect();
        for x in [vec![], vec![7], vec![9; 20000], raw] {
            let p = packet(&x).unwrap();
            let mut at = 0;
            let mut d = Part::parse(&p, &mut at, x.len()).unwrap();
            let decoded: Vec<_> = (0..x.len()).map(|_| d.byte().unwrap()).collect();
            d.finish().unwrap();
            assert_eq!(decoded, x);
            assert_eq!(at, p.len());
            for cut in 0..p.len().min(100) {
                let mut at = 0;
                if let Ok(mut d) = Part::parse(&p[..cut], &mut at, x.len()) {
                    let mut o = Vec::new();
                    while let Ok(b) = d.byte() {
                        o.push(b);
                    }
                    assert!(d.finish().is_err() || o == x);
                }
            }
        }
    }
    #[test]
    fn invalid_tables_and_trailing_bytes() {
        for bytes in [vec![0], vec![2, 1, 0, 1, 0], vec![1, 0, 0, 0, 0, 0, 0]] {
            assert!(Reader::new(&bytes).is_err());
        }
        let raw = b"abracadabra repeated alphabet".repeat(5000);
        let mut coded = encode(&raw).unwrap();
        coded.push(0);
        let mut d = Reader::new(&coded).unwrap();
        for _ in &raw {
            d.byte().unwrap();
        }
        assert!(d.finish().is_err());
    }
}
