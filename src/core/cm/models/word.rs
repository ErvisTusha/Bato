//! Original case-folded word and line-column statistics for arbitrary bytes.
//! Provenance: word-context modeling described in Mahoney, Data Compression
//! Explained. No dictionary, text parser, format signature or copied code is
//! used. Bytes at least 128 are treated as letters without UTF-8 assumptions.

use super::super::hash;

/// Word length beyond which contexts stop distinguishing lengths.
const LENGTH_CAP: u16 = 32;
/// Line columns distinguished by the line context (a power of two).
const COLUMNS: u32 = 512;
/// FNV-1a 64-bit prime, the multiplier of the rolling word hash.
const WORD_PRIME: u64 = 0x100_0000_01b3;
/// Multiplier of the rolling line-shape hash.
const LINE_PRIME: u64 = 257;
/// Domain separators keeping each word context's hashes distinct.
const SALTS: [u64; 6] = [
    0x17e1_1045_8c93_a2d7,
    0x2e63_8431_297a_6b55,
    0x741e_9bd7_2340_c891,
    0x5d13_27c9_a640_8b21,
    0x3c6e_f372_fe94_f82b,
    0x6a09_e667_f3bc_c908,
];
/// Character classes of the line-shape hash.
#[derive(Clone, Copy)]
enum Class {
    Letter = 1,
    Digit = 2,
    Space = 3,
    Other = 4,
}
#[derive(Debug, Default)]
pub(in crate::core::cm) struct Words {
    current: u64,
    previous: [u64; 2],
    length: u16,
    column: u32,
    line: u64,
    last_line: u64,
    cases: u8,
}
impl Words {
    /// Extra text contexts: word + previous byte, and a three-word context.
    pub(in crate::core::cm) fn text_contexts(&self, previous: u8) -> [u64; 2] {
        let word = self.word();
        [
            hash::mix(word ^ (u64::from(previous) << 40) ^ SALTS[4]),
            hash::mix(
                word ^ self.previous[0].rotate_left(13)
                    ^ self.previous[1].rotate_left(41)
                    ^ SALTS[5],
            ),
        ]
    }
    pub(in crate::core::cm) fn contexts(&self) -> [u64; 4] {
        let word = self.word();
        [
            hash::mix(word ^ SALTS[0]),
            hash::mix(word ^ self.previous[0].rotate_left(17) ^ SALTS[1]),
            hash::mix(word ^ self.previous[1].rotate_left(29) ^ SALTS[2]),
            hash::mix(
                u64::from(self.column % COLUMNS)
                    ^ self.last_line.rotate_left(23)
                    ^ (u64::from(self.cases) << 40)
                    ^ SALTS[3],
            ),
        ]
    }
    /// The current word's hash combined with its capped length.
    fn word(&self) -> u64 {
        self.current ^ (u64::from(self.length.min(LENGTH_CAP)) << 48)
    }
    pub(in crate::core::cm) fn observe_byte(&mut self, byte: u8) {
        let folded = byte.to_ascii_lowercase();
        let letter = folded.is_ascii_lowercase() || byte >= 128;
        if letter {
            self.current = self
                .current
                .wrapping_mul(WORD_PRIME)
                .wrapping_add(u64::from(folded) + 1);
            self.length = self.length.saturating_add(1);
            self.cases = (self.cases << 1) | u8::from(byte.is_ascii_uppercase());
        } else if self.length != 0 {
            self.previous[1] = self.previous[0];
            self.previous[0] = self.current;
            self.current = 0;
            self.length = 0;
        }
        if byte == b'\n' {
            self.last_line = self.line;
            self.line = 0;
            self.column = 0;
        } else {
            self.column = self.column.saturating_add(1);
            let class = if letter {
                Class::Letter
            } else if byte.is_ascii_digit() {
                Class::Digit
            } else if byte.is_ascii_whitespace() {
                Class::Space
            } else {
                Class::Other
            };
            self.line = self
                .line
                .wrapping_mul(LINE_PRIME)
                .wrapping_add(class as u64);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn words_fold_case_accept_high_bytes_and_end_on_byte_classes() {
        let mut a = Words::default();
        let mut b = Words::default();
        for &byte in b"Universal" {
            a.observe_byte(byte);
        }
        for &byte in b"uNIVERSAL" {
            b.observe_byte(byte);
        }
        assert_eq!(a.current, b.current);
        assert_eq!(&a.contexts()[..3], &b.contexts()[..3]);
        let word = a.current;
        a.observe_byte(b'1');
        assert_eq!(a.previous[0], word);
        a.observe_byte(0xff);
        a.observe_byte(0x80);
        assert_eq!(a.length, 2);
        a.observe_byte(0);
        assert_eq!(a.length, 0);
        a.observe_byte(b'\n');
        assert_eq!(a.column, 0);
        assert_ne!(a.last_line, 0);
    }
}
