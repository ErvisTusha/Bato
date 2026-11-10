use bato_engine::{
    compress_with_options, decompress_with_options, read_block_header, read_geometry,
    read_stream_header, BatoError, BlockType, CompressionOptions, DecodeLimits,
    DecompressionOptions, ExecutionOptions, Geometry, MemoryBudget,
};
use sha2::{Digest, Sha256};
use std::io::{self, Cursor, Read};

const C: usize = 16 * 1024;
const S: usize = 64 * 1024;
fn options() -> CompressionOptions {
    CompressionOptions {
        geometry: Geometry {
            segment_size: S,
            chunk_size: C,
            ..Default::default()
        },
        memory_budget: MemoryBudget::Bytes(256 * 1024 * 1024),
        execution: ExecutionOptions {
            threads: 1,
            ..Default::default()
        },
    }
}
fn decode_options() -> DecompressionOptions {
    DecompressionOptions {
        execution: options().execution,
        limits: DecodeLimits {
            max_working_memory: MemoryBudget::Bytes(256 * 1024 * 1024),
            ..Default::default()
        },
    }
}

#[test]
fn default_options_roundtrip_and_unknown_block_types_are_rejected() {
    assert_eq!(CompressionOptions::default().geometry, Geometry::default());
    let raw = b"default geometry roundtrip";
    let encoded = bato_engine::compress(raw).unwrap();
    assert_eq!(
        bato_engine::read_stream_header(&mut &encoded[..])
            .unwrap()
            .kind,
        bato_engine::StreamKind::Bytes
    );
    assert_eq!(bato_engine::decompress(&encoded).unwrap(), raw);
    for kind in [8, 18, 254] {
        assert!(bato_engine::BlockType::from_u8(kind).is_err());
    }
}
fn random(length: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    for i in 0..length.div_ceil(32) {
        bytes.extend_from_slice(&Sha256::digest((i as u64).to_le_bytes()));
    }
    bytes.truncate(length);
    bytes
}

#[test]
fn raw_coded_and_segmented_data_restore_deterministically() {
    let defaults = options();
    let mut mixed = random(C);
    mixed.extend_from_slice(&b"causal byte history and arbitrary contents\n".repeat(1400));
    for raw in [vec![], vec![255], random(C + 1), mixed] {
        let encoded = compress_with_options(&raw, &defaults).unwrap();
        let mut input = Cursor::new(&encoded);
        read_stream_header(&mut input).unwrap();
        assert_eq!(read_geometry(&mut input).unwrap().chunk_size, C);
        assert_eq!(
            decompress_with_options(&encoded, &decode_options()).unwrap(),
            raw
        );
        assert!(encoded.len() <= raw.len() + 32 + 14 * raw.len().div_ceil(C));
        let mut parallel = defaults.clone();
        parallel.execution.threads = 3;
        assert_eq!(compress_with_options(&raw, &parallel).unwrap(), encoded);
        if !raw.is_empty() {
            let mut corrupt = encoded;
            *corrupt.last_mut().unwrap() ^= 1;
            assert!(decompress_with_options(&corrupt, &decode_options()).is_err());
        }
    }
}
fn blocks(encoded: &[u8]) -> Vec<(usize, bato_engine::BlockHeader)> {
    let mut input = Cursor::new(encoded);
    read_stream_header(&mut input).unwrap();
    let descriptor = read_geometry(&mut input).unwrap();
    assert_eq!((descriptor.segment_size, descriptor.chunk_size), (S, C));
    let mut output = Vec::new();
    loop {
        let offset = input.position() as usize;
        let header = read_block_header(&mut input).unwrap();
        if header.block_type == BlockType::EndOfStream {
            break;
        }
        input.set_position(input.position() + u64::from(header.compressed_size) + 4);
        output.push((offset, header));
    }
    output
}

#[test]
fn edges_all_bytes_and_raw_bound() {
    for length in [
        0,
        1,
        2,
        255,
        4095,
        4096,
        4097,
        C - 1,
        C,
        C + 1,
        S - 1,
        S,
        S + 1,
        2 * S + 1,
    ] {
        let raw = random(length);
        let encoded = compress_with_options(&raw, &options()).unwrap();
        assert!(encoded.len() <= length + 38 + 14 * length.div_ceil(C));
        assert_eq!(
            decompress_with_options(&encoded, &decode_options()).unwrap(),
            raw
        );
    }
    for byte in 0..=255u8 {
        let encoded = compress_with_options(&[byte], &options()).unwrap();
        assert_eq!(
            decompress_with_options(&encoded, &decode_options()).unwrap(),
            [byte]
        );
    }
    let raw = random(S * 2);
    let encoded = compress_with_options(&raw, &options()).unwrap();
    assert!(blocks(&encoded)
        .iter()
        .all(|(_, h)| h.block_type == BlockType::Stored));
    let raw: Vec<_> = (0..=255u8).cycle().take(S + C).collect();
    let encoded = compress_with_options(&raw, &options()).unwrap();
    assert_eq!(
        decompress_with_options(&encoded, &decode_options()).unwrap(),
        raw
    );
}

#[test]
fn adaptive_segments_and_duplicates_cross_boundaries() {
    let mut raw = random(C);
    raw.extend_from_slice(&b"persistent statistical segment history\n".repeat(2000));
    raw.extend_from_slice(&random(C));
    raw.extend_from_within(..S);
    let encoded = compress_with_options(&raw, &options()).unwrap();
    let types: Vec<_> = blocks(&encoded)
        .into_iter()
        .map(|(_, h)| h.block_type)
        .collect();
    assert!(!types.is_empty());
    assert!(types
        .iter()
        .any(|kind| !matches!(kind, BlockType::Raw | BlockType::Stored)));
    assert_eq!(
        decompress_with_options(&encoded, &decode_options()).unwrap(),
        raw
    );
    let nested = bato_engine::compress(&raw).unwrap();
    let mut outer = b"arbitrary nested bytes\n".repeat(1000);
    outer.extend_from_slice(&nested);
    let encoded = compress_with_options(&outer, &options()).unwrap();
    assert_eq!(
        decompress_with_options(&encoded, &decode_options()).unwrap(),
        outer
    );
}

#[test]
fn threads_and_budgets_preserve_exact_bytes() {
    let raw = b"segment reset and continuing chunk models\n".repeat(4000);
    let reference = compress_with_options(&raw, &options()).unwrap();
    for threads in 1..=4 {
        for budget in [224, 640] {
            let mut settings = options();
            settings.execution.threads = threads;
            settings.memory_budget = MemoryBudget::Bytes(budget * 1024 * 1024);
            assert_eq!(compress_with_options(&raw, &settings).unwrap(), reference);
            let mut decode = decode_options();
            decode.execution = settings.execution.clone();
            decode.limits.max_working_memory = settings.memory_budget;
            assert_eq!(decompress_with_options(&reference, &decode).unwrap(), raw);
        }
    }
    let mut settings = options();
    settings.execution.threads = 257;
    assert!(matches!(
        compress_with_options(&raw, &settings),
        Err(BatoError::InvalidOptions(_))
    ));
    let mut decode = decode_options();
    decode.execution.threads = 257;
    assert!(matches!(
        decompress_with_options(&reference, &decode),
        Err(BatoError::InvalidOptions(_))
    ));
}

#[test]
fn truncation_mutation_and_invalid_descriptors_never_panic() {
    let encoded = compress_with_options(b"short mutation fixture", &options()).unwrap();
    for end in 0..encoded.len() {
        assert!(decompress_with_options(&encoded[..end], &decode_options()).is_err());
    }
    for offset in 0..encoded.len() {
        for bit in 0..8 {
            let mut mutated = encoded.clone();
            mutated[offset] ^= 1 << bit;
            assert!(std::panic::catch_unwind(|| decompress_with_options(
                &mutated,
                &decode_options()
            ))
            .is_ok());
        }
    }
    for (offset, value) in [
        (7, 31),
        (8, 0),
        (8, 2),
        (9, 25),
        (10, 7),
        (10, 0x40),
        (10, 7 << 3),
    ] {
        let mut mutated = encoded.clone();
        mutated[offset] = value;
        assert!(
            decompress_with_options(&mutated, &decode_options()).is_err(),
            "offset {offset} = {value}"
        );
    }
    let mut appended = encoded.clone();
    appended.push(0);
    assert!(decompress_with_options(&appended, &decode_options()).is_err());
    let mut bad = encoded.clone();
    *bad.last_mut().unwrap() ^= 1;
    assert!(decompress_with_options(&bad, &decode_options()).is_err());
}

#[test]
fn chunk_engine_geometry_and_checksums_are_strict() {
    let raw = b"byte statistical data without format detection\n".repeat(2000);
    let encoded = compress_with_options(&raw, &options()).unwrap();
    let chunks = blocks(&encoded);
    assert!(!chunks.is_empty());
    for kind in [7u8, 0x40, 7 << 3] {
        let mut bad = encoded.clone();
        bad[10] = kind;
        assert!(decompress_with_options(&bad, &decode_options()).is_err());
    }
    let mut bad = encoded.clone();
    bad[7] = 31;
    assert!(matches!(
        decompress_with_options(&bad, &decode_options()),
        Err(BatoError::CorruptStream(_))
    ));
    // Canonical varints: no zero size, overlong encodings, or overflow.
    for field in [
        vec![0],
        vec![0x81, 0],
        vec![0x80; 11],
        vec![0xff, 0xff, 0xff, 0xff, 0x1f],
    ] {
        let mut bad = encoded[..11].to_vec();
        bad.extend(field);
        assert!(decompress_with_options(&bad, &decode_options()).is_err());
    }
    for (offset, _) in chunks {
        let mut cursor = Cursor::new(&encoded);
        cursor.set_position(offset as u64);
        let header = read_block_header(&mut cursor).unwrap();
        let crc_at = cursor.position() as usize + header.compressed_size as usize;
        let mut crc = encoded.clone();
        crc[crc_at] ^= 1;
        assert!(matches!(
            decompress_with_options(&crc, &decode_options()),
            Err(BatoError::ChecksumMismatch { .. })
        ));
    }
}

struct Short<R>(R);
impl<R: Read> Read for Short<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let n = output.len().min(3);
        self.0.read(&mut output[..n])
    }
}
#[test]
fn fragmented_reads_limits_and_preflight() {
    let raw = b"fragmented stream\n".repeat(9000);
    let mut encoded = Vec::new();
    bato_engine::compress_stream(&mut Short(Cursor::new(&raw)), &mut encoded, &options()).unwrap();
    let mut decoded = Vec::new();
    bato_engine::decompress_stream_with_options(
        &mut Short(Cursor::new(&encoded)),
        &mut decoded,
        &decode_options(),
    )
    .unwrap();
    assert_eq!(decoded, raw);
    let mut limits = decode_options();
    limits.limits.max_output_bytes = 1;
    let mut input = Cursor::new(&encoded);
    assert!(matches!(
        bato_engine::decompress_stream_with_options(&mut input, &mut Vec::new(), &limits),
        Err(BatoError::DecompressionLimitExceeded { .. })
    ));
    assert_eq!(
        input.position(),
        {
            let mut header = Cursor::new(&encoded);
            header.set_position(10);
            read_block_header(&mut header).unwrap();
            header.position()
        },
        "oversized first chunk must be rejected before reading its payload"
    );
    let mut low = options();
    low.memory_budget = MemoryBudget::Bytes(1);
    let mut input = Cursor::new(&raw);
    let mut output = Vec::new();
    assert!(matches!(
        bato_engine::compress_stream(&mut input, &mut output, &low),
        Err(BatoError::MemoryLimitExceeded { .. })
    ));
    assert!(output.is_empty());
    assert_eq!(input.position(), 0);
    let mut limits = decode_options();
    limits.limits.max_working_memory = MemoryBudget::Bytes(1);
    let mut input = Cursor::new(&encoded);
    assert!(matches!(
        bato_engine::decompress_stream_with_options(&mut input, &mut Vec::new(), &limits),
        Err(BatoError::MemoryLimitExceeded { .. })
    ));
    assert_eq!(input.position(), 10);
    let mut defaults = options();
    defaults.memory_budget = MemoryBudget::Bytes(1);
    assert!(matches!(
        compress_with_options(&raw, &defaults),
        Err(BatoError::MemoryLimitExceeded { .. })
    ));
}
