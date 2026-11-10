//! Stream-level checks for the CM x86 call/jump address transform.
use bato_engine::*;
use std::io::Cursor;

/// Call-heavy synthetic machine code: many `E8 rel32` sites reaching a few
/// shared targets inside the same 64 KiB segment, separated by varying
/// filler bytes.
fn code(length: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(length);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    while bytes.len() + 16 < length {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let at = bytes.len() as u32;
        let target = (at & !0xffff) + 0x4000 + (state % 24) as u32 * 0x180;
        bytes.push(0xe8);
        bytes.extend_from_slice(&target.wrapping_sub(at + 5).to_le_bytes());
        let filler = 3 + (state >> 40) as usize % 9;
        bytes.extend((0..filler).map(|k| (state >> (k % 7 * 8)) as u8 & 0x7f));
    }
    bytes.resize(length, 0x90);
    bytes
}

fn options() -> CompressionOptions {
    CompressionOptions {
        geometry: Geometry {
            segment_size: 65536,
            chunk_size: 16384,
            ..Default::default()
        },
        execution: ExecutionOptions {
            threads: 2,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Offsets of every block header after the 10-byte stream header.
fn headers(encoded: &[u8]) -> Vec<(usize, BlockHeader)> {
    let mut reader = Cursor::new(encoded);
    read_stream_header(&mut reader).unwrap();
    read_geometry(&mut reader).unwrap();
    let mut found = Vec::new();
    loop {
        let at = reader.position() as usize;
        let header = read_block_header(&mut reader).unwrap();
        if header.block_type == BlockType::EndOfStream {
            return found;
        }
        reader.set_position(reader.position() + u64::from(header.compressed_size) + 4);
        found.push((at, header));
    }
}

#[test]
fn call_heavy_segments_select_the_transform_and_roundtrip_exactly() {
    let raw = code(150_000);
    let encoded = compress_with_options(&raw, &options()).unwrap();
    let blocks = headers(&encoded);
    assert!(!blocks.is_empty());
    assert!(blocks
        .iter()
        .all(|(_, h)| h.transform_id == TransformId::X86Calls));
    assert_eq!(decompress(&encoded).unwrap(), raw);

    let mut plain = raw.clone();
    for byte in &mut plain {
        if *byte & 0xfe == 0xe8 {
            *byte = 0x90;
        }
    }
    let encoded = compress_with_options(&plain, &options()).unwrap();
    // Automatic filters may also help the filler; no transform is forced.
    assert!(!headers(&encoded).is_empty());
    assert_eq!(decompress(&encoded).unwrap(), plain);
}

#[test]
fn rip_relative_segments_select_the_wider_transform_and_roundtrip_exactly() {
    // `call target` then `lea rax, [rip + d]`, both reaching few addresses.
    let mut raw = Vec::new();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    while raw.len() + 32 < 150_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let at = raw.len() as u32;
        raw.push(0xe8);
        let target = (at & !0xffff) + 0x4000 + (state % 24) as u32 * 0x180;
        raw.extend_from_slice(&target.wrapping_sub(at + 5).to_le_bytes());
        let at = raw.len() as u32;
        raw.extend_from_slice(&[0x48, 0x8d, 0x05]);
        let data = (at & !0xffff) + 0x8000 + (state >> 20) as u32 % 40 * 8;
        raw.extend_from_slice(&data.wrapping_sub(at + 7).to_le_bytes());
        let filler = 3 + (state >> 40) as usize % 9;
        raw.extend((0..filler).map(|k| (state >> (k % 7 * 8)) as u8 & 0x7f));
    }
    let encoded = compress_with_options(&raw, &options()).unwrap();
    let blocks = headers(&encoded);
    assert!(!blocks.is_empty());
    assert!(blocks
        .iter()
        .all(|(_, h)| h.transform_id == TransformId::X86Relative));
    assert_eq!(decompress(&encoded).unwrap(), raw);
}

#[test]
fn code_beside_deflate_streams_gets_both_transforms() {
    let mut raw = include_bytes!("../src/core/cm/deflate/fixtures/l6.deflate").to_vec();
    raw.extend_from_slice(&code(40_000));
    let encoded = compress_with_options(&raw, &options()).unwrap();
    let blocks = headers(&encoded);
    assert!(!blocks.is_empty());
    assert!(blocks
        .iter()
        .all(|(_, h)| h.transform_id == TransformId::DeflateX86Calls));
    assert_eq!(decompress(&encoded).unwrap(), raw);
}

#[test]
fn mixed_or_unknown_segment_transforms_are_rejected() {
    // Construct two full raw CM chunks. A segment must declare one transform
    // even when decoding can skip all predictor updates.
    let mut encoded = Vec::new();
    bato_engine::core::frame::write_stream_header(&mut encoded, &options()).unwrap();
    for transform in [TransformId::None, TransformId::X86Calls] {
        encoded.push((transform as u8) << 3);
        encoded.extend_from_slice(&[0x80, 0x80, 1]); // 16 KiB, canonical ULEB128
        encoded.resize(encoded.len() + 16384 + 4, 0);
    }
    assert!(decompress(&encoded).is_err());
    // The deflate transform rejects chunks whose bytes are not a transformed
    // container, and transform id 15 is undefined.
    encoded[10] = 7 << 3;
    assert!(decompress(&encoded).is_err());
    encoded[10] = 0x80 | 7 << 3;
    assert!(decompress(&encoded).is_err());
}
