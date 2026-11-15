//! Original byte-exact delta and byte-plane transforms.
//! Provenance: public delta/byte-shuffle filter descriptions, including the
//! XZ delta filter; original implementations, always selected by byte cost.
use super::x86::{self, Sites};
use crate::core::{
    gpu::filters::{Direction, WorkerFilters},
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
pub(super) fn forward_on(
    raw: &[u8],
    transform: TransformId,
    accelerator: &mut Option<&mut WorkerFilters<'_>>,
) -> Result<Vec<u8>, BatoError> {
    if CANDIDATES.contains(&transform) {
        if let Some(accelerator) = accelerator.as_deref_mut() {
            if let Some(output) = accelerator.run(raw, transform, Direction::Forward)? {
                return Ok(output);
            }
        }
    }
    forward(raw, transform)
}

pub(super) fn inverse_on(
    raw: &mut Vec<u8>,
    transform: TransformId,
    accelerator: Option<&mut WorkerFilters<'_>>,
) -> Result<(), BatoError> {
    if CANDIDATES.contains(&transform) {
        if let Some(accelerator) = accelerator {
            if let Some(output) = accelerator.run(raw, transform, Direction::Inverse)? {
                *raw = output;
                return Ok(());
            }
        }
    }
    inverse(raw, transform)
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
    fn wgpu_filters_match_cpu_at_word_tile_and_tail_boundaries() {
        use super::super::{
            descriptor::Descriptor,
            profile::Timing,
            segment::{self, Trials},
        };
        use crate::core::{
            gpu::filters::FilterPool,
            types::{BlockType, Compute, ExecutionStats},
        };

        let Some(pool) = FilterPool::wgpu_for_test() else {
            assert!(
                std::env::var_os("BATO_TEST_REQUIRE_WGPU").is_none(),
                "no wgpu test adapter"
            );
            eprintln!("wgpu filter parity skipped: no adapter");
            return;
        };
        let stats = ExecutionStats::default();
        let mut worker = WorkerFilters::new(&pool, (Compute::Wgpu, &stats), 0);
        for length in [1, 3, 4, 63, 64, 65, 4095, 4096, 4097, 65539] {
            let input: Vec<_> = (0..length)
                .map(|i: u32| ((i.wrapping_mul(0x9e3779b1) >> 21) ^ (i >> 3)) as u8)
                .collect();
            for transform in CANDIDATES {
                let expected = forward(&input, transform).unwrap();
                let transformed = worker
                    .run(&input, transform, Direction::Forward)
                    .unwrap()
                    .unwrap();
                assert_eq!(transformed, expected, "{transform:?} ({length} bytes)");
                let restored = worker
                    .run(&transformed, transform, Direction::Inverse)
                    .unwrap()
                    .unwrap();
                assert_eq!(restored, input, "{transform:?} ({length} bytes)");
            }
        }
        let descriptor = Descriptor {
            segment_log2: 16,
            chunk_log2: 14,
            table_log2: 16,
        };
        let raw: Vec<u8> = (1..=65_535u32).map(|byte| byte as u8).collect();
        let encode = |worker| {
            segment::encode_segment(
                &mut raw.clone(),
                (descriptor, Trials::Sequential),
                Timing::off(),
                worker,
            )
            .unwrap()
        };
        let (cpu_chunks, cpu_coding) = encode(None);
        let (wgpu_chunks, wgpu_coding) = encode(Some(&mut worker));
        assert_eq!(cpu_coding, wgpu_coding);
        assert_eq!(cpu_chunks.len(), 1);
        assert_eq!(wgpu_chunks.len(), 1);
        let cpu = &cpu_chunks[0];
        let accelerated = &wgpu_chunks[0];
        assert_eq!(accelerated.kind, BlockType::Repeat);
        assert_eq!(
            (cpu.kind, cpu.length, cpu.crc, &cpu.payload),
            (
                accelerated.kind,
                accelerated.length,
                accelerated.crc,
                &accelerated.payload
            )
        );
        let restored = segment::decode_segment(
            &wgpu_chunks,
            (descriptor, raw.len()),
            wgpu_coding,
            Timing::off(),
            Some(&mut worker),
        )
        .unwrap();
        assert_eq!(restored, raw);
        assert!(stats.snapshot().wgpu_filter_dispatches > 0);
    }

    #[test]
    fn cuda_filters_match_cpu_at_word_tile_and_tail_boundaries() {
        use crate::core::gpu::cuda::{filters::apply, Cuda};

        let cuda = match Cuda::probe() {
            Ok(cuda) => cuda,
            Err(reason) => {
                assert!(
                    std::env::var_os("BATO_TEST_REQUIRE_CUDA").is_none(),
                    "{reason}"
                );
                eprintln!("CUDA filter parity skipped: {reason}");
                return;
            }
        };
        for length in [1, 3, 4, 63, 64, 65, 4095, 4096, 4097, 65539] {
            let input: Vec<_> = (0..length)
                .map(|i: u32| ((i.wrapping_mul(0x9e3779b1) >> 21) ^ (i >> 3)) as u8)
                .collect();
            let mut transformed = vec![0; input.len()];
            let mut restored = transformed.clone();
            for transform in CANDIDATES {
                let expected = forward(&input, transform).unwrap();
                apply(
                    &cuda,
                    &input,
                    &mut transformed,
                    (transform, Direction::Forward),
                )
                .unwrap();
                assert_eq!(transformed, expected, "{transform:?} ({length} bytes)");
                apply(
                    &cuda,
                    &transformed,
                    &mut restored,
                    (transform, Direction::Inverse),
                )
                .unwrap();
                assert_eq!(restored, input, "{transform:?} ({length} bytes)");
            }
        }
    }
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
