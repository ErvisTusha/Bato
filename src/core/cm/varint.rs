//! Original canonical unsigned LEB128 helpers; public base-128 representation.
use crate::core::types::BatoError;
use std::io::{Read, Write};

pub(crate) fn len(mut value: u64) -> usize {
    let mut n = 1;
    while value >= 128 {
        value >>= 7;
        n += 1;
    }
    n
}
pub(crate) fn write<W: Write>(w: &mut W, mut value: u64) -> Result<(), BatoError> {
    loop {
        let byte = (value & 127) as u8;
        value >>= 7;
        w.write_all(&[byte | if value == 0 { 0 } else { 128 }])?;
        if value == 0 {
            return Ok(());
        }
    }
}
pub(crate) fn read<R: Read>(r: &mut R) -> Result<u64, BatoError> {
    let mut value = 0u64;
    for i in 0..10 {
        let mut b = [0];
        r.read_exact(&mut b)?;
        if i == 9 && b[0] > 1 {
            return Err(invalid());
        }
        value |= u64::from(b[0] & 127) << (i * 7);
        if b[0] & 128 == 0 {
            if i > 0 && b[0] == 0 {
                return Err(invalid());
            }
            return Ok(value);
        }
    }
    Err(invalid())
}
pub(super) fn push(out: &mut Vec<u8>, mut value: usize) {
    loop {
        let byte = (value & 127) as u8;
        value >>= 7;
        out.push(byte | if value == 0 { 0 } else { 128 });
        if value == 0 {
            return;
        }
    }
}
pub(super) fn take(input: &[u8], at: &mut usize) -> Result<usize, BatoError> {
    let mut slice = input.get(*at..).ok_or_else(invalid)?;
    let before = slice.len();
    let value = read(&mut slice)?;
    *at += before - slice.len();
    usize::try_from(value).map_err(|_| invalid())
}
fn invalid() -> BatoError {
    BatoError::CorruptStream("noncanonical or overflowing integer".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_integers_and_overflows() {
        for n in [0, 1, 127, 128, 16383, 16384, u32::MAX as u64, u64::MAX] {
            let mut v = Vec::new();
            write(&mut v, n).unwrap();
            assert_eq!(v.len(), len(n));
            assert_eq!(read(&mut &v[..]).unwrap(), n);
        }
        for v in [vec![128, 0], vec![255; 11], vec![128; 10]] {
            assert!(read(&mut &v[..]).is_err());
        }
    }
}
