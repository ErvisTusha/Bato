use bato_engine::*;
use std::sync::Arc;

fn options(stats: ExecutionStats) -> CompressionOptions {
    CompressionOptions {
        geometry: Geometry {
            segment_size: 65536,
            chunk_size: 16384,
            ..Default::default()
        },
        execution: ExecutionOptions {
            threads: 2,
            stats: Arc::new(stats),
        },
        ..Default::default()
    }
}

fn sample() -> Vec<u8> {
    b"step timing observes the coder and never steers it\n"
        .iter()
        .copied()
        .cycle()
        .take(40_000)
        .collect()
}

#[test]
fn timing_never_changes_the_archive_and_reports_every_phase() {
    let data = sample();
    let plain = compress_with_options(&data, &options(ExecutionStats::default())).unwrap();
    let timed_options = options(ExecutionStats::with_timing());
    let timed = compress_with_options(&data, &timed_options).unwrap();
    assert_eq!(plain, timed);
    assert!(options(ExecutionStats::default())
        .execution
        .stats
        .timing_report()
        .is_none());
    let report = timed_options
        .execution
        .stats
        .timing_report()
        .unwrap()
        .to_string();
    for expected in [
        "archive bytes by purpose",
        "model-coded payload",
        "primer training",
        "main context maps",
        "calibration (APM) stages",
        "activity of optional models",
    ] {
        assert!(
            report.contains(expected),
            "{expected} missing from\n{report}"
        );
    }
}

#[test]
fn decoding_with_timing_restores_the_input() {
    let data = sample();
    let archive = compress(&data).unwrap();
    let stats = Arc::new(ExecutionStats::with_timing());
    let mut restored = Vec::new();
    decompress_stream_with_options(
        &mut archive.as_slice(),
        &mut restored,
        &DecompressionOptions {
            execution: ExecutionOptions {
                threads: 2,
                stats: stats.clone(),
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(restored, data);
    let report = stats.timing_report().unwrap().to_string();
    assert!(report.contains("model decoding"), "{report}");
}
