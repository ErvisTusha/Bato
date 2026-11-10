//! CRC32C (Castagnoli), with a portable slicing-by-eight implementation.
//! Provenance: the Castagnoli polynomial (Castagnoli, Bräuer and Herrmann,
//! IEEE Trans. Commun. 41(6), 1993; RFC 3720 appendix B.4), table-driven
//! slicing as published by Intel (Kounavis and Berry, 2005), and CRC
//! combination by GF(2) shift operators. The Rust implementation is original.
use std::convert::TryInto;
const POLY: u32 = 0x82F63B78;
const fn table() -> [[u32; 256]; 8] {
    let mut t = [[0; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { c >> 1 ^ POLY } else { c >> 1 };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut k = 1;
    while k < 8 {
        let mut i = 0;
        while i < 256 {
            let c = t[k - 1][i];
            t[k][i] = t[0][(c & 255) as usize] ^ c >> 8;
            i += 1;
        }
        k += 1;
    }
    t
}
const TABLE: [[u32; 256]; 8] = table();
#[inline]
fn portable(mut c: u32, mut d: &[u8]) -> u32 {
    while d.len() >= 8 {
        let x = u64::from_le_bytes(d[..8].try_into().unwrap()) ^ c as u64;
        c = TABLE[7][(x & 255) as usize]
            ^ TABLE[6][(x >> 8 & 255) as usize]
            ^ TABLE[5][(x >> 16 & 255) as usize]
            ^ TABLE[4][(x >> 24 & 255) as usize]
            ^ TABLE[3][(x >> 32 & 255) as usize]
            ^ TABLE[2][(x >> 40 & 255) as usize]
            ^ TABLE[1][(x >> 48 & 255) as usize]
            ^ TABLE[0][(x >> 56) as usize];
        d = &d[8..];
    }
    for &b in d {
        c = c >> 8 ^ TABLE[0][((c ^ b as u32) & 255) as usize];
    }
    c
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn hw(mut c: u32, mut d: &[u8]) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u32, _mm_crc32_u64, _mm_crc32_u8};
    while d.len() >= 8 {
        c = _mm_crc32_u64(c as u64, u64::from_le_bytes(d[..8].try_into().unwrap())) as u32;
        d = &d[8..];
    }
    while d.len() >= 4 {
        c = _mm_crc32_u32(c, u32::from_le_bytes(d[..4].try_into().unwrap()));
        d = &d[4..];
    }
    for &b in d {
        c = _mm_crc32_u8(c, b);
    }
    c
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn hw(mut c: u32, mut d: &[u8]) -> u32 {
    use std::arch::aarch64::{__crc32cb, __crc32cd, __crc32cw};
    while d.len() >= 8 {
        c = __crc32cd(c, u64::from_le_bytes(d[..8].try_into().unwrap()));
        d = &d[8..];
    }
    while d.len() >= 4 {
        c = __crc32cw(c, u32::from_le_bytes(d[..4].try_into().unwrap()));
        d = &d[4..];
    }
    for &b in d {
        c = __crc32cb(c, b);
    }
    c
}
#[inline]
fn state(c: u32, d: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("sse4.2") {
        // SAFETY: SSE4.2 was detected at runtime; `hw` only reads `d`.
        return unsafe { hw(c, d) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::is_aarch64_feature_detected!("crc") {
        // SAFETY: the CRC extension was detected at runtime; `hw` only reads `d`.
        return unsafe { hw(c, d) };
    }
    portable(c, d)
}
pub struct Crc32c {
    state: u32,
}
impl Default for Crc32c {
    fn default() -> Self {
        Self::new()
    }
}
impl Crc32c {
    pub fn new() -> Self {
        Self { state: u32::MAX }
    }
    pub fn update(&mut self, d: &[u8]) {
        self.state = state(self.state, d)
    }
    pub fn finalize(&self) -> u32 {
        self.state ^ u32::MAX
    }
}
pub fn crc32c(d: &[u8]) -> u32 {
    let mut c = Crc32c::new();
    c.update(d);
    c.finalize()
}
fn times(m: &[u32; 32], mut v: u32) -> u32 {
    let mut s = 0;
    let mut i = 0;
    while v != 0 {
        if v & 1 != 0 {
            s ^= m[i]
        }
        v >>= 1;
        i += 1;
    }
    s
}
fn square(o: &mut [u32; 32], m: &[u32; 32]) {
    for i in 0..32 {
        o[i] = times(m, m[i])
    }
}
pub fn crc32c_combine(a: u32, b: u32, len: u64) -> u32 {
    if len == 0 {
        return a;
    }
    let mut odd = [0; 32];
    odd[0] = POLY;
    let mut row = 1;
    for item in odd.iter_mut().skip(1) {
        *item = row;
        row <<= 1
    }
    let mut even = [0; 32];
    square(&mut even, &odd);
    square(&mut odd, &even);
    let (mut n, mut x) = (len, a);
    loop {
        square(&mut even, &odd);
        if n & 1 != 0 {
            x = times(&even, x)
        }
        n >>= 1;
        if n == 0 {
            break;
        }
        square(&mut odd, &even);
        if n & 1 != 0 {
            x = times(&odd, x)
        }
        n >>= 1;
        if n == 0 {
            break;
        }
    }
    x ^ b
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hardware_matches_portable() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 % 251) as u8).collect();
        for length in [0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 999, 1000] {
            let expected = portable(0x1234_5678, &data[..length]);
            assert_eq!(state(0x1234_5678, &data[..length]), expected, "{length}");
        }
    }
    #[test]
    fn vectors() {
        assert_eq!(crc32c(&[]), 0);
        assert_eq!(crc32c(b"123456789"), 0xe3069283)
    }
    #[test]
    fn incremental() {
        let d = b"incremental checksum test";
        let mut c = Crc32c::new();
        for x in d.chunks(3) {
            c.update(x)
        }
        assert_eq!(c.finalize(), crc32c(d))
    }
    #[test]
    fn combine() {
        let (a, b) = (b"hello ", b"world");
        assert_eq!(
            crc32c_combine(crc32c(a), crc32c(b), b.len() as u64),
            crc32c(&[a.as_slice(), b.as_slice()].concat())
        )
    }
}
#[cfg(test)]
mod property_tests {
    use super::*;
    #[test]
    fn random_chunk_boundaries() {
        let d: Vec<u8> = (0..10000u32).map(|x| x.wrapping_mul(31) as u8).collect();
        let expected = crc32c(&d);
        for step in 1..97 {
            let mut c = Crc32c::new();
            for chunk in d.chunks(step) {
                c.update(chunk)
            }
            assert_eq!(c.finalize(), expected);
        }
    }
    #[test]
    fn combine_lengths() {
        let d: Vec<u8> = (0..10000u32).map(|x| x.wrapping_mul(17) as u8).collect();
        for n in 0..=10000 {
            assert_eq!(
                crc32c_combine(crc32c(&d[..n]), crc32c(&d[n..]), (d.len() - n) as u64),
                crc32c(&d)
            );
        }
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;
    #[test]
    fn portable_and_runtime_dispatch_agree() {
        let data: Vec<u8> = (0..8193u32)
            .map(|i| i.wrapping_mul(71).wrapping_add(i / 13) as u8)
            .collect();
        for offset in 0..32 {
            for len in [0, 1, 3, 4, 7, 8, 15, 16, 4096, 8192 - offset] {
                let bytes = &data[offset..offset + len];
                assert_eq!(portable(u32::MAX, bytes) ^ u32::MAX, crc32c(bytes));
            }
        }
    }
}
