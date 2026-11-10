use bato_engine::{compress, decompress};

const STREAM_A: &[u8] = include_bytes!("../src/core/cm/deflate/fixtures/l6.deflate");
const STREAM_B: &[u8] = include_bytes!("../src/core/cm/deflate/fixtures/l9.deflate");

/// A buffer that embeds two zlib-made deflate streams between other bytes,
/// the way an archive embeds compressed members.
fn container() -> Vec<u8> {
    let mut raw = b"local file header, member name and flags\n".repeat(2);
    raw.extend_from_slice(STREAM_A);
    raw.extend_from_slice(b"\x50\x4b\x03\x04 next header \x00\xff");
    raw.extend_from_slice(STREAM_B);
    raw.extend_from_slice(b"central directory");
    raw
}

#[test]
fn embedded_deflate_streams_shrink_far_below_their_compressed_size() {
    let raw = container();
    let encoded = compress(&raw).unwrap();
    assert_eq!(decompress(&encoded).unwrap(), raw);
    // Without recompression the streams are incompressible and the output
    // stays near the input size.
    assert!(
        encoded.len() * 10 < raw.len() * 8,
        "{} bytes for {}",
        encoded.len(),
        raw.len()
    );
}

#[test]
fn damaged_streams_in_the_container_still_round_trip() {
    let mut raw = container();
    let middle = raw.len() / 3;
    raw[middle] ^= 0x20;
    let encoded = compress(&raw).unwrap();
    assert_eq!(decompress(&encoded).unwrap(), raw);
}

#[test]
fn truncated_and_corrupted_encodings_never_panic() {
    let encoded = compress(&container()).unwrap();
    for cut in [10, encoded.len() / 2, encoded.len() - 1] {
        assert!(decompress(&encoded[..cut]).is_err());
    }
    for position in (12..encoded.len()).step_by(encoded.len() / 5) {
        let mut damaged = encoded.clone();
        damaged[position] ^= 0x01;
        let _ = decompress(&damaged);
    }
}

fn small_segments() -> bato_engine::CompressionOptions {
    bato_engine::CompressionOptions {
        geometry: bato_engine::Geometry {
            segment_size: 65536,
            chunk_size: 16384,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// `count` segments, each a deflate stream padded with text to exactly one
/// segment, so that inflating the streams makes every segment grow beyond it.
fn full_segments(count: usize) -> Vec<u8> {
    let mut raw = Vec::new();
    for index in 0..count {
        let stream = if index % 2 == 0 { STREAM_A } else { STREAM_B };
        let mut segment = b"member header ".to_vec();
        segment.extend_from_slice(stream);
        segment.resize(65536, b' ');
        raw.extend_from_slice(&segment);
    }
    raw
}

#[test]
fn full_segments_whose_streams_inflate_beyond_the_segment_size_are_recompressed() {
    for count in [1, 3] {
        let raw = full_segments(count);
        let encoded = bato_engine::compress_with_options(&raw, &small_segments()).unwrap();
        assert_eq!(decompress(&encoded).unwrap(), raw);
        // Each segment holds 22,000 bytes of deflate that inflate to 48,000.
        // Left as they are they cost at least 22,000 bytes per segment;
        // recompressed, under 18,000.
        assert!(
            encoded.len() < count * 18_000,
            "{} bytes for {} in {count} segments",
            encoded.len(),
            raw.len()
        );
    }
}

#[test]
fn damaged_recompressed_segments_fail_cleanly() {
    let raw = full_segments(2);
    let encoded = bato_engine::compress_with_options(&raw, &small_segments()).unwrap();
    for position in (0..encoded.len()).step_by(encoded.len() / 40 + 1) {
        let mut damaged = encoded.clone();
        damaged[position] ^= 0x55;
        // Either an error or, if only coded payload bits flipped in a place
        // the checksums do not cover, never a panic.
        if let Ok(restored) = decompress(&damaged) {
            assert_eq!(restored.len(), raw.len());
        }
    }
    for cut in [10, encoded.len() / 2, encoded.len() - 1] {
        assert!(decompress(&encoded[..cut]).is_err());
    }
}
