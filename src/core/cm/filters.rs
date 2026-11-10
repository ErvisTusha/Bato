//! Original byte-exact delta and byte-plane transforms.
//! Provenance: public delta/byte-shuffle filter descriptions, including the
//! XZ delta filter; original implementations, always selected by byte cost.
use super::x86::{self, Sites};
use crate::core::{
    memory::try_filled,
    types::{BatoError, TransformId},
};

pub(super) const CANDIDATES: [TransformId; 5] = [
    TransformId::Delta1,
    TransformId::Delta2,
    TransformId::Delta4,
    TransformId::Shuffle2,
    TransformId::Shuffle4,
];
fn distance(t: TransformId) -> usize {
    match t {
        TransformId::Delta1 => 1,
        TransformId::Delta2 => 2,
        TransformId::Delta4 => 4,
        TransformId::Shuffle2 => 2,
        TransformId::Shuffle4 => 4,
        _ => 0,
    }
}
/// The deflate transforms change the segment's length, so they are not
/// in-place filters.
fn not_a_filter() -> BatoError {
    BatoError::CorruptStream("deflate transform used as an in-place filter".into())
}
pub(super) fn forward(raw: &[u8], t: TransformId) -> Result<Vec<u8>, BatoError> {
    if t.is_deflate_container() {
        return Err(not_a_filter());
    }
    let mut out = try_filled(raw.len(), 0u8)?;
    if let Some(sites) = Sites::of_transform(t) {
        out.copy_from_slice(raw);
        x86::forward(&mut out, sites);
        return Ok(out);
    }
    match t {
        TransformId::Delta1 | TransformId::Delta2 | TransformId::Delta4 => {
            let d = distance(t);
            for i in 0..raw.len() {
                out[i] = raw[i].wrapping_sub(if i >= d { raw[i - d] } else { 0 });
            }
        }
        TransformId::Shuffle2 | TransformId::Shuffle4 => {
            let width = distance(t);
            let words = raw.len() / width;
            for byte in 0..width {
                for word in 0..words {
                    out[byte * words + word] = raw[word * width + byte];
                }
            }
            out[words * width..].copy_from_slice(&raw[words * width..]);
        }
        _ => out.copy_from_slice(raw),
    }
    Ok(out)
}
pub(super) fn inverse(raw: &mut [u8], t: TransformId) -> Result<(), BatoError> {
    if t.is_deflate_container() {
        return Err(not_a_filter());
    }
    if let Some(sites) = Sites::of_transform(t) {
        x86::inverse(raw, sites);
        return Ok(());
    }
    match t {
        TransformId::Delta1 | TransformId::Delta2 | TransformId::Delta4 => {
            let d = distance(t);
            for i in d..raw.len() {
                raw[i] = raw[i].wrapping_add(raw[i - d]);
            }
        }
        TransformId::Shuffle2 | TransformId::Shuffle4 => {
            let src = crate::core::memory::try_copy(raw)?;
            let width = distance(t);
            let words = raw.len() / width;
            for byte in 0..width {
                for word in 0..words {
                    raw[word * width + byte] = src[byte * words + word];
                }
            }
        }
        _ => {}
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn filters_preserve_unaligned_and_arbitrary_bytes() {
        for n in [0, 1, 2, 3, 7, 8, 257, 65539] {
            let raw: Vec<_> = (0..n).map(|i| ((i * 117) ^ (i >> 3)) as u8).collect();
            for t in [
                TransformId::None,
                TransformId::X86Calls,
                TransformId::X86Relative,
                TransformId::X86CallsSegment,
                TransformId::X86RelativeSegment,
            ]
            .into_iter()
            .chain(CANDIDATES)
            {
                let mut v = forward(&raw, t).unwrap();
                inverse(&mut v, t).unwrap();
                assert_eq!(v, raw);
            }
        }
    }
}
