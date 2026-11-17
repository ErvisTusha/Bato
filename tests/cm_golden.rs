use bato_engine::*;
use sha2::{Digest, Sha256};
use std::io::Cursor;

fn explicit_sizes(encoded: &[u8]) -> Vec<u8> {
    let mut reader = Cursor::new(encoded);
    let stream = read_stream_header(&mut reader).unwrap();
    let geometry = read_geometry(&mut reader).unwrap();
    let mut legacy = encoded[..10].to_vec();
    legacy[5] = 22;
    loop {
        let start = reader.position() as usize;
        let header = read_stream_block_header(&mut reader, &stream, geometry).unwrap();
        if header.block_type == BlockType::EndOfStream {
            legacy.extend_from_slice(&encoded[start..]);
            return legacy;
        }
        let end = reader.position() as usize + header.compressed_size as usize + 4;
        if encoded[start + 1] == 0 {
            legacy.push(encoded[start]);
            let mut size = header.uncompressed_size;
            while size >= 128 {
                legacy.push(size as u8 | 128);
                size >>= 7;
            }
            legacy.push(size as u8);
            legacy.extend_from_slice(&encoded[start + 2..end]);
        } else {
            legacy.extend_from_slice(&encoded[start..end]);
        }
        reader.set_position(end as u64);
    }
}

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
    assert_eq!(encoded[5], VERSION_MINOR);
    assert_eq!(decompress(&encoded).unwrap(), fixture);
    let encoded = explicit_sizes(&encoded);
    assert_eq!(format!("{:x}", Sha256::digest(&encoded)), expected);
    assert_eq!(decompress(&encoded).unwrap(), fixture);
}

#[test]
fn complete_encoded_bytes_are_pinned() {
    check("f9fa79cae23e4d6bd69264facc73cca21e56b78355af571cac8aa83b8e52c118");
}
