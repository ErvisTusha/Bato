//! Original wrapped integer context hashing, independent of file formats.
//! Provenance: elementary xor/shift/multiply avalanche construction. Constants
//! and context composition are not taken from any compressor implementation.

#[inline]
pub(super) fn mix(mut key: u64) -> u64 {
    key ^= key >> 29;
    key = key.wrapping_mul(0x9e37_79b1_85eb_ca87);
    key ^= key >> 31;
    key = key.wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    key ^ (key >> 28)
}
pub(super) fn bytes(history: &[u8], order: usize) -> u64 {
    let start = history.len().saturating_sub(order);
    let mut key = (order as u64)
        .wrapping_add(1)
        .wrapping_mul(0x1b87_3593_5bd1_e995);
    for &byte in &history[start..] {
        key = key
            .wrapping_mul(0x100_0000_01b3)
            .wrapping_add(u64::from(byte) + 1);
    }
    mix(key ^ (history.len().min(order) as u64))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn context_hashes_use_only_the_declared_history_and_order() {
        assert_eq!(bytes(b"abc", 2), bytes(b"xyzbc", 2));
        assert_ne!(bytes(b"abc", 2), bytes(b"abc", 3));
        assert_ne!(bytes(b"\0", 2), bytes(b"\0\0", 2));
        assert_ne!(mix(0), mix(u64::MAX));
    }
}
