//! Original byte-renormalized, inclusive-interval binary arithmetic coder.
//! Provenance: standard arithmetic interval subdivision, independently written
//! in Rust; no third-party compressor implementation or tables were copied.
//! The probability is P(bit=0), in units of 1/65536, strictly between 0 and 1.
//! A chunk's decoded length and checksum belong to the outer frame. The coder
//! checks its canonical byte representation; it does not replace a checksum.

use crate::core::{memory::try_capacity, types::BatoError};

fn corrupt(message: &'static str) -> BatoError {
    BatoError::CorruptStream(message.into())
}

#[inline]
fn partition(low: u32, high: u32, probability_zero: u16) -> u32 {
    // Using the inclusive lower subinterval makes both outcomes representable
    // even when an interval straddles a byte boundary with a small width.
    low + (((u64::from(high - low) * u64::from(probability_zero)) >> 16) as u32)
}

fn flush(low: u32, high: u32) -> ([u8; 4], usize) {
    // Select the shortest byte prefix whose zero-padded code point is inside
    // the final interval; choose the lowest such point to resolve ties.
    for length in 0..=4usize {
        let step = 1u64 << (8 * (4 - length));
        let point = u64::from(low).div_ceil(step) * step;
        if point <= u64::from(high) {
            return ((point as u32).to_be_bytes(), length);
        }
    }
    unreachable!("a nonempty interval always contains a four-byte code point")
}

pub(super) struct Encoder {
    low: u32,
    high: u32,
    bytes: Vec<u8>,
    capacity: usize,
    exceeded: bool,
}

impl Encoder {
    pub(super) fn new(max_bytes: usize) -> Result<Self, BatoError> {
        Ok(Self {
            low: 0,
            high: u32::MAX,
            bytes: try_capacity(max_bytes)?,
            capacity: max_bytes,
            exceeded: false,
        })
    }

    pub(super) fn exceeded(&self) -> bool {
        self.exceeded
    }

    fn emit(&mut self, byte: u8) {
        if self.bytes.len() < self.capacity {
            self.bytes.push(byte);
        } else {
            self.exceeded = true;
        }
    }

    #[inline]
    pub(super) fn bit(&mut self, bit: u8, probability_zero: u16) -> Result<(), BatoError> {
        if bit > 1 || probability_zero == 0 {
            return Err(corrupt("invalid CM bit or probability"));
        }
        let split = partition(self.low, self.high, probability_zero);
        if bit == 0 {
            self.high = split;
        } else {
            self.low = split + 1;
        }
        while (self.low ^ self.high) < (1 << 24) {
            self.emit((self.low >> 24) as u8);
            self.low <<= 8;
            self.high = (self.high << 8) | 255;
        }
        Ok(())
    }

    pub(super) fn finish(mut self) -> Option<Vec<u8>> {
        let (tail, length) = flush(self.low, self.high);
        for &byte in &tail[..length] {
            self.emit(byte);
        }
        if self.exceeded {
            None
        } else {
            Some(self.bytes)
        }
    }
}

pub(super) struct Decoder<'a> {
    low: u32,
    high: u32,
    code: u32,
    payload: &'a [u8],
    read: usize,
    checked: usize,
}

impl<'a> Decoder<'a> {
    pub(super) fn new(payload: &'a [u8]) -> Self {
        let mut decoder = Self {
            low: 0,
            high: u32::MAX,
            code: 0,
            payload,
            read: 0,
            checked: 0,
        };
        for _ in 0..4 {
            decoder.code = (decoder.code << 8) | u32::from(decoder.byte());
        }
        decoder
    }

    fn byte(&mut self) -> u8 {
        if let Some(&byte) = self.payload.get(self.read) {
            self.read += 1;
            byte
        } else {
            // A shortest flush omits the zero-padded low bytes. The separate
            // canonical-output cursor still rejects missing emitted bytes.
            0
        }
    }

    fn check(&mut self, expected: u8) -> Result<(), BatoError> {
        if self.payload.get(self.checked) != Some(&expected) {
            return Err(corrupt("noncanonical or truncated CM payload"));
        }
        self.checked += 1;
        Ok(())
    }

    #[inline]
    pub(super) fn bit(&mut self, probability_zero: u16) -> Result<u8, BatoError> {
        if probability_zero == 0 {
            return Err(corrupt("invalid CM probability"));
        }
        let split = partition(self.low, self.high, probability_zero);
        let bit = u8::from(self.code > split);
        if bit == 0 {
            self.high = split;
        } else {
            self.low = split + 1;
        }
        while (self.low ^ self.high) < (1 << 24) {
            self.check((self.low >> 24) as u8)?;
            self.low <<= 8;
            self.high = (self.high << 8) | 255;
            self.code = (self.code << 8) | u32::from(self.byte());
        }
        Ok(bit)
    }

    pub(super) fn finish(mut self) -> Result<(), BatoError> {
        let (tail, length) = flush(self.low, self.high);
        for &byte in &tail[..length] {
            self.check(byte)?;
        }
        if self.checked != self.payload.len() {
            return Err(corrupt("trailing bytes in CM payload"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(bits: &[(u8, u16)]) -> Vec<u8> {
        let mut encoder = Encoder::new(bits.len() * 3 + 4).unwrap();
        for &(bit, probability) in bits {
            encoder.bit(bit, probability).unwrap();
        }
        encoder.finish().unwrap()
    }

    fn decode(payload: &[u8], bits: &[(u8, u16)]) -> Result<Vec<u8>, BatoError> {
        let mut decoder = Decoder::new(payload);
        let mut output = Vec::new();
        for &(_, probability) in bits {
            output.push(decoder.bit(probability)?);
        }
        decoder.finish()?;
        Ok(output)
    }

    #[test]
    fn empty_and_extreme_probabilities_roundtrip() {
        assert_eq!(encode(&[]), Vec::<u8>::new());
        assert_eq!(decode(&[], &[]).unwrap(), Vec::<u8>::new());
        for probability in [1, 2, 255, 256, 32767, 32768, 32769, 65534, 65535] {
            for pattern in [0u8, 1, 2, 3] {
                let bits: Vec<_> = (0..2048usize)
                    .map(|i| {
                        let bit = match pattern {
                            0 => 0,
                            1 => 1,
                            2 => (i & 1) as u8,
                            _ => u8::from(i % 257 == 0),
                        };
                        (bit, probability)
                    })
                    .collect();
                let payload = encode(&bits);
                assert_eq!(
                    decode(&payload, &bits).unwrap(),
                    bits.iter().map(|b| b.0).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn varying_predictions_and_canonical_tail_checks() {
        let mut random = 0x1234_5678u32;
        let mut bits = Vec::new();
        for i in 0..4096usize {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            bits.push(((random & 1) as u8, [1, 32768, 65535, 8192, 57344][i % 5]));
        }
        let payload = encode(&bits);
        let expected: Vec<_> = bits.iter().map(|b| b.0).collect();
        assert_eq!(decode(&payload, &bits).unwrap(), expected);
        for extra in [0, 1, 255] {
            let mut appended = payload.clone();
            appended.push(extra);
            if extra == 0 {
                assert!(decode(&appended, &bits).is_err());
            } else if let Ok(restored) = decode(&appended, &bits) {
                assert_ne!(restored, expected);
            }
        }
        for length in [0, 1, payload.len() / 2, payload.len() - 1] {
            // A truncated arithmetic code can encode a different message.
            // The outer CRC rejects that case; canonicality alone must never
            // accept a shorter representation of the original message.
            if let Ok(restored) = decode(&payload[..length], &bits) {
                assert_ne!(restored, expected);
            }
        }
    }

    #[test]
    fn every_interval_gets_the_shortest_zero_padded_flush() {
        for low in [0, 1, 255, 256, 65535, 65536, 0x7fff_ffff, u32::MAX] {
            for width in [0u32, 1, 255, 65535, u32::MAX] {
                let high = low.saturating_add(width);
                let (bytes, length) = flush(low, high);
                let point = u32::from_be_bytes(bytes);
                assert!((low..=high).contains(&point));
                assert_eq!(u64::from(point) % (1u64 << (8 * (4 - length))), 0);
                for shorter in 0..length {
                    let step = 1u64 << (8 * (4 - shorter));
                    let first = u64::from(low).div_ceil(step) * step;
                    assert!(first > u64::from(high));
                }
            }
        }
    }

    #[test]
    fn bounded_output_and_invalid_predictions() {
        let mut encoder = Encoder::new(1).unwrap();
        for i in 0..1000u32 {
            encoder.bit((i & 1) as u8, 32768).unwrap();
        }
        assert!(encoder.finish().is_none());
        let mut encoder = Encoder::new(0).unwrap();
        assert!(encoder.bit(2, 32768).is_err());
        assert!(encoder.bit(0, 0).is_err());
        assert!(Decoder::new(&[]).bit(0).is_err());
    }
}
