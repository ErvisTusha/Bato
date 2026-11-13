use bato_engine::*;
use sha2::{Digest, Sha256};

fn check(expected: &str) {
    let fixture: Vec<_> = b"original integer CM framing and predictor fixture\n"
        .iter()
        .copied()
        .cycle()
        .take(70_003)
        .collect();
    assert_eq!(
        format!("{:x}", Sha256::digest(&fixture)),
        "5d858817e408165bf01bcc533b15742d45de8064cf4283f4b987aab591f5546c"
    );
    let cm = Geometry {
        segment_size: 65536,
        chunk_size: 16384,
        ..Default::default()
    };
    let options = CompressionOptions {
        geometry: cm,
        execution: ExecutionOptions {
            threads: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let encoded = compress_with_options(&fixture, &options).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&encoded)), expected);
    assert_eq!(decompress(&encoded).unwrap(), fixture);
}

#[test]
fn complete_encoded_bytes_are_pinned() {
    check("a68fdaee8c1f56742ac437603fdc880e6cb3fcfe096315ebd3e2c6819d4a642b");
}
