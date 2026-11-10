use bato_engine::{
    compress_stream, compress_with_options, CompressionOptions, Geometry, MemoryBudget,
};
use std::io::Cursor;

#[test]
fn insufficient_admission_fails_before_writing_a_stream_header() {
    let options = CompressionOptions {
        memory_budget: MemoryBudget::Bytes(1),
        ..Default::default()
    };
    let mut output = Vec::new();
    assert!(compress_stream(&mut Cursor::new(b"arbitrary bytes"), &mut output, &options).is_err());
    assert!(output.is_empty());
}

#[test]
fn cm_admission_reduces_workers_with_fixed_segment_geometry() {
    let raw = b"fixed segment geometry across memory allowances\n".repeat(4000);
    let mut low = CompressionOptions {
        geometry: Geometry {
            segment_size: 65536,
            chunk_size: 16384,
            ..Default::default()
        },
        memory_budget: MemoryBudget::Bytes(224 * 1024 * 1024),
        ..Default::default()
    };
    low.execution.threads = 8;
    let mut high = low.clone();
    high.memory_budget = MemoryBudget::Bytes(1280 * 1024 * 1024);
    high.execution.stats = Default::default();
    let reference = compress_with_options(&raw, &low).unwrap();
    assert_eq!(reference, compress_with_options(&raw, &high).unwrap());
    assert_eq!(low.execution.stats.snapshot().cpu_workers, 1);
    assert_eq!(high.execution.stats.snapshot().cpu_workers, 8);
    for settings in [&low, &high] {
        let stats = settings.execution.stats.snapshot();
        // The estimate covers pool infrastructure and its admitted jobs;
        // the convenience API separately charges its growing output vector.
        assert!(stats.estimated_host_memory <= stats.memory_budget);
        assert!(stats.managed_host_peak <= stats.memory_budget);
    }
}
