use bato_engine::{
    compress_with_options, create_solid_archive, decompress, extract_solid_archive_with_options,
    read_stream_layout, BlockType, CompressionOptions, DecodeLimits, ExecutionOptions, Geometry,
    MemoryBudget, Overwrite, VERSION_MINOR,
};
use sha2::{Digest, Sha256};
use std::{fs, io::Cursor};

const CHUNK: usize = 64 * 1024;
const SEGMENT: usize = 8 * CHUNK;

fn opaque(length: usize) -> Vec<u8> {
    (0..length.div_ceil(32))
        .flat_map(|index| Sha256::digest((index as u64).to_le_bytes()))
        .take(length)
        .collect()
}

fn options() -> CompressionOptions {
    CompressionOptions {
        geometry: Geometry {
            segment_size: SEGMENT,
            chunk_size: CHUNK,
            ..Default::default()
        },
        memory_budget: MemoryBudget::Bytes(512 * 1024 * 1024),
        execution: ExecutionOptions {
            threads: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn exact_period_writes_one_independent_copy_with_compact_length() {
    let input = opaque(CHUNK).repeat(8);
    let encoded = compress_with_options(&input, &options()).unwrap();
    assert_eq!(encoded[5], VERSION_MINOR);
    assert_eq!(encoded[10] & 7, BlockType::Lz as u8);
    assert_eq!(encoded[11], 0);
    let layout = read_stream_layout(&mut Cursor::new(&encoded)).unwrap();
    assert_eq!(layout.blocks[0].uncompressed_size as usize, SEGMENT);
    assert_eq!(decompress(&encoded).unwrap(), input);
}

#[test]
fn cold_tail_has_an_independent_model_and_checks_its_stored_bytes() {
    let mut input = opaque(7 * CHUNK);
    input.extend_from_slice(&include_bytes!("../docs/validation.md")[..CHUNK]);
    let encoded = compress_with_options(&input, &options()).unwrap();
    assert_eq!(encoded[10] & 7, BlockType::LzHuffman as u8);
    assert_eq!(encoded[11], 0);
    let start = 12 + encoded[12..].iter().position(|byte| byte & 128 == 0).unwrap() + 1;
    assert_eq!(&encoded[start..start + 2], &[0, 0]);
    assert_eq!(decompress(&encoded).unwrap(), input);
    let primer = encoded[start + 2];
    let mut corrupted = encoded;
    corrupted[start + 2] = 2;
    assert!(decompress(&corrupted).is_err());
    corrupted[start + 2] = primer;
    corrupted[start + 12] ^= 1;
    assert!(decompress(&corrupted).is_err());
}

#[test]
fn native_directory_reuses_bytes_from_an_earlier_file() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input");
    fs::create_dir(&source).unwrap();
    let a = opaque(CHUNK);
    let mut b = a.clone();
    b.splice(1024..1152, [0xff; 175]);
    fs::write(source.join("a"), &a).unwrap();
    fs::write(source.join("b"), &b).unwrap();
    let mut encoded = Vec::new();
    create_solid_archive(&source, &mut encoded, &options()).unwrap();
    assert!(encoded.len() < 2 * CHUNK * 3 / 4);
    let restored = temp.path().join("restored");
    let limits = DecodeLimits {
        max_output_bytes: (a.len() + b.len()) as u64,
        ..Default::default()
    };
    extract_solid_archive_with_options(
        &mut Cursor::new(&encoded),
        &restored,
        &limits,
        Overwrite::Refuse,
    )
    .unwrap();
    assert_eq!(fs::read(restored.join("a")).unwrap(), a);
    assert_eq!(fs::read(restored.join("b")).unwrap(), b);
}
