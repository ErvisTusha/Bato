#![cfg(feature = "cuda")]

use bato_engine::{
    compress_with_options, decompress_with_options, read_stream_layout, BlockType,
    CompressionOptions, Compute, DecompressionOptions, ExecutionOptions, TransformId,
};
use std::io::Cursor;

#[test]
fn cuda_archive_matches_cpu_and_uses_gpu_in_both_directions() {
    if let Err(reason) = bato_engine::core::gpu::cuda::Cuda::probe() {
        assert!(
            std::env::var_os("BATO_TEST_REQUIRE_CUDA").is_none(),
            "required CUDA device is unavailable: {reason}"
        );
        eprintln!("CUDA archive parity skipped: {reason}");
        return;
    }
    // First difference is 1 too, so every segment has a one-byte Repeat
    // independent candidate. A partial segment exercises the GPU tail.
    let input: Vec<u8> = (1..=(2 * 65536 + 4099)).map(|i| i as u8).collect();
    let mut cpu = CompressionOptions::default();
    cpu.geometry.segment_size = 65536;
    cpu.geometry.chunk_size = 16384;
    cpu.execution.threads = 1;
    cpu.execution.compute = Compute::Cpu;
    let reference = compress_with_options(&input, &cpu).unwrap();
    let gpu = CompressionOptions {
        execution: ExecutionOptions {
            threads: 3,
            compute: Compute::Cuda,
            ..Default::default()
        },
        ..cpu
    };
    let compressed = compress_with_options(&input, &gpu).unwrap();
    assert_eq!(compressed, reference);
    let layout = read_stream_layout(&mut Cursor::new(&compressed)).unwrap();
    assert!(layout.blocks.iter().any(|block| {
        block.transform_id == TransformId::Delta1 && block.block_type == BlockType::Repeat
    }));
    assert_eq!(
        gpu.execution.stats.snapshot().cuda_filter_dispatches,
        input.len().div_ceil(gpu.geometry.segment_size) as u64,
        "the first filter reaches the minimum cost in each segment"
    );

    let decode = DecompressionOptions {
        execution: ExecutionOptions {
            compute: Compute::Cuda,
            threads: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        decompress_with_options(&compressed, &decode).unwrap(),
        input
    );
    assert!(decode.execution.stats.snapshot().cuda_filter_dispatches > 0);
}
