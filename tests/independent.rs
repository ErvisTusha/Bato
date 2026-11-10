//! External format vectors: decode independent blocks without the encoder.
use bato_engine::core::{checksum::crc32c, frame::write_stream_header};
use bato_engine::*;
fn integer(out: &mut Vec<u8>, mut n: usize) {
    loop {
        let b = (n & 127) as u8;
        n >>= 7;
        out.push(b | if n == 0 { 0 } else { 128 });
        if n == 0 {
            break;
        }
    }
}
fn stream(kind: BlockType, transform: TransformId, payload: &[u8], raw: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    write_stream_header(&mut v, &CompressionOptions::default()).unwrap();
    v.push(kind as u8 | (transform as u8) << 3);
    integer(&mut v, raw.len());
    if matches!(
        kind,
        BlockType::Lz | BlockType::Huffman | BlockType::LzHuffman | BlockType::Numeric
    ) {
        integer(&mut v, payload.len());
    }
    v.extend_from_slice(payload);
    v.extend_from_slice(&crc32c(raw).to_le_bytes());
    v.push(255);
    integer(&mut v, raw.len());
    v.extend_from_slice(&crc32c(raw).to_le_bytes());
    v
}
#[test]
fn independently_constructed_vectors_and_integrity() {
    let repeated = vec![b'A'; 64];
    let cases = [
        (
            BlockType::Stored,
            b"abcdefgh".to_vec(),
            b"abcdefgh".to_vec(),
        ),
        (BlockType::Repeat, vec![b'A'], repeated.clone()),
        (BlockType::Lz, vec![2, b'A', 119, 1], repeated.clone()),
        (
            BlockType::Huffman,
            vec![2, b'A', 1, b'B', 1, 0xaa],
            b"ABABABAB".to_vec(),
        ),
        // Huffman table: four symbols of length 2, coding literal/copy bytes.
        (
            BlockType::LzHuffman,
            vec![4, 4, 1, 2, 2, 2, 65, 2, 119, 2, 0x36],
            repeated,
        ),
    ];
    for (kind, payload, raw) in cases {
        let v = stream(kind, TransformId::None, &payload, &raw);
        assert_eq!(decompress(&v).unwrap(), raw, "{kind:?}");
        assert!(decompress_with_limit(&v, raw.len() - 1).is_err());
        for cut in 0..v.len() {
            assert!(decompress(&v[..cut]).is_err());
        }
        let mut bad = v.clone();
        let n = bad.len();
        bad[n - 1] ^= 1;
        assert!(matches!(
            decompress(&bad),
            Err(BatoError::StreamChecksumMismatch { .. })
        ));
        for at in 10..v.len() {
            let mut bad = v.clone();
            bad[at] ^= 0x80;
            assert!(std::panic::catch_unwind(|| decompress_with_limit(&bad, 1024)).is_ok());
        }
    }
}
#[test]
fn run_filter_checksums_cover_original_bytes() {
    for (t, distance) in [
        (TransformId::Delta1, 1),
        (TransformId::Delta2, 2),
        (TransformId::Delta4, 4),
    ] {
        let mut raw = vec![3u8; 257];
        for i in distance..raw.len() {
            raw[i] = raw[i].wrapping_add(raw[i - distance]);
        }
        assert_eq!(
            decompress(&stream(BlockType::Repeat, t, &[3], &raw)).unwrap(),
            raw
        );
    }
}
#[test]
fn invalid_copy_tables_and_token_lengths_are_rejected() {
    for (kind, payload) in [
        (BlockType::Lz, vec![1, 0]),
        (BlockType::Lz, vec![1, 1]),
        (BlockType::Huffman, vec![2, b'A', 1, b'A', 1, 0]),
        (BlockType::Huffman, vec![2, b'A', 2, b'B', 2, 0]),
        (BlockType::Huffman, vec![2, b'A', 1, b'B', 1, 0xff, 0]),
        (BlockType::LzHuffman, vec![0xff, 0xff, 0xff, 0xff, 0x0f]),
    ] {
        assert!(decompress(&stream(kind, TransformId::None, &payload, &[0; 64])).is_err());
    }
}

#[test]
fn numeric_wire_vector_and_recursion_bounds() {
    let mut raw = vec![b'x'; 64];
    raw.extend_from_slice(b"1 x2");
    let payload = [
        70, 3, 11, 2, 120, 119, 1, 12, 0, 1, 32, 120, 0, 1, 2, 5, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 0,
        1, 0, 229, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
    ];
    let encoded = stream(BlockType::Numeric, TransformId::None, &payload, &raw);
    assert_eq!(decompress(&encoded).unwrap(), raw);
    for at in 10..encoded.len() {
        let mut bad = encoded.clone();
        bad[at] ^= 0x80;
        assert!(std::panic::catch_unwind(|| decompress_with_limit(&bad, 1024)).is_ok());
    }
    for kind in [0, 1, 7, 255] {
        let bad = stream(
            BlockType::Numeric,
            TransformId::None,
            &[1, kind, 0],
            &[0; 64],
        );
        assert!(decompress(&bad).is_err());
    }
}
