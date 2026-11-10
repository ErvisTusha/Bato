use bato_engine::{
    compress_stream, compress_with_options, decompress, decompress_stream_with_limits,
    decompress_with_limit, decompress_with_limits, read_stream_layout, BlockType,
    CompressionOptions, DecodeLimits, ExecutionOptions, Geometry, StreamKind, VERSION_MAJOR,
    VERSION_MINOR,
};
use std::io::{self, Read};

fn compress_small(data: &[u8]) -> Result<Vec<u8>, bato_engine::BatoError> {
    compress_with_options(data, &cpu_options())
}

fn cpu_options() -> CompressionOptions {
    CompressionOptions {
        geometry: Geometry {
            segment_size: 65536,
            chunk_size: 16384,
            ..Default::default()
        },
        execution: ExecutionOptions {
            threads: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn simple_api_roundtrip_and_worker_determinism() {
    let data: Vec<u8> = (0..70_003)
        .map(|i| {
            if i % 97 < 80 {
                b'a' + (i % 11) as u8
            } else {
                (i * 31) as u8
            }
        })
        .collect();
    let one = compress_with_options(&data, &cpu_options()).unwrap();
    assert_eq!(decompress(&one).unwrap(), data);
    let many = compress_with_options(
        &data,
        &CompressionOptions {
            execution: ExecutionOptions {
                threads: 3,
                ..Default::default()
            },
            ..cpu_options()
        },
    )
    .unwrap();
    assert_eq!(one, many);
    assert_eq!(
        compress_small(&data).unwrap(),
        compress_small(&data).unwrap()
    );
}

#[test]
fn largest_chunk_roundtrip() {
    let data = (0..(1024 * 1024 + 17))
        .map(|i| if i % 13 == 0 { 7 } else { (i % 251) as u8 })
        .collect::<Vec<_>>();
    let z = compress_with_options(
        &data,
        &CompressionOptions {
            geometry: Geometry {
                segment_size: bato_engine::MAX_CHUNK_SIZE,
                chunk_size: bato_engine::MAX_CHUNK_SIZE,
                ..Default::default()
            },
            ..cpu_options()
        },
    )
    .unwrap();
    assert_eq!(&z[..4], b"BATO");
    assert_eq!(decompress(&z).unwrap(), data);
}

#[test]
fn checksums_footer_trailing_bytes_and_limits_are_strict() {
    let data = b"abcdabcdabcd".repeat(4000);
    let z = compress_with_options(&data, &cpu_options()).unwrap();
    assert_eq!(decompress(&z).unwrap(), data);
    for cut in [0, 1, 7, 8, z.len() - 1] {
        assert!(decompress(&z[..cut]).is_err());
    }
    let mut trailing = z.clone();
    trailing.push(0);
    assert!(decompress(&trailing).is_err());
    let mut corrupt = z.clone();
    let corrupt_at = corrupt.len() / 2;
    corrupt[corrupt_at] ^= 0x80;
    assert!(decompress(&corrupt).is_err());
    assert!(decompress_with_limit(&z, data.len() - 1).is_err());
    assert!(decompress_with_limits(
        &z,
        &DecodeLimits {
            max_working_memory: bato_engine::MemoryBudget::Bytes(1024),
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn old_magic_and_invalid_headers_are_rejected() {
    for magic in [
        b"BTS4".as_slice(),
        b"PK\x03\x04".as_slice(),
        b"BTA5".as_slice(),
    ] {
        assert!(decompress(&[magic, &[5, 0, 0, 0]].concat()).is_err());
    }
}

#[test]
fn archive_header_flag_is_versioned_and_rejects_unknown_bits() {
    use bato_engine::read_stream_header;
    use std::io::Cursor;
    for (major, minor) in (0..=u8::MAX).map(|minor| (VERSION_MAJOR, minor)).chain([
        (VERSION_MAJOR.wrapping_sub(1), 0),
        (VERSION_MAJOR.wrapping_add(1), 0),
    ]) {
        for flags in [0u8, 1, 2, 3, 4, 8, 128, u8::MAX] {
            let mut bytes = b"BATO".to_vec();
            bytes.extend([major, minor, flags]);
            let result = read_stream_header(&mut Cursor::new(bytes));
            let allowed = major == VERSION_MAJOR && minor == VERSION_MINOR && flags <= 1;
            assert_eq!(result.is_ok(), allowed, "{major}.{minor} flags={flags}");
            if let Ok(header) = result {
                assert_eq!(
                    header.kind == bato_engine::StreamKind::DirectoryArchive,
                    flags == 1
                );
            }
        }
    }
}

#[test]
fn arbitrary_binary_bytes_keep_raw_fallback_and_exact_framing_bounds() {
    // A reproducible cryptographic digest stream avoids accidental linear
    // dependencies in a small xorshift PRNG that adaptive contexts can exploit.
    use sha2::{Digest, Sha256};
    let mut data = Vec::with_capacity(262_145 + 32);
    for index in 0u64..8193 {
        let mut hasher = Sha256::new();
        hasher.update(b"bato-incompressible-regression");
        hasher.update(index.to_le_bytes());
        data.extend_from_slice(&hasher.finalize());
    }
    data.truncate(262_145);
    for length in [0, 1, 3, 64, 4096, 262_144, 262_145] {
        let encoded = compress_with_options(&data[..length], &cpu_options()).unwrap();
        assert_eq!(decompress(&encoded).unwrap(), data[..length]);
        // Raw storage caps growth at the fixed header/footer and 14 bytes per
        // nonempty group, even when the content has no useful pattern.
        let groups = length.div_ceil(16384);
        assert!(encoded.len() <= length + 32 + groups * 14);
        if length >= 4096 {
            assert_eq!(encoded[10] & 7, bato_engine::BlockType::Stored as u8);
        }
    }
}

struct Short<'a>(&'a [u8]);
impl Read for Short<'_> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        let n = b.len().min(3).min(self.0.len());
        b[..n].copy_from_slice(&self.0[..n]);
        self.0 = &self.0[n..];
        Ok(n)
    }
}

#[test]
fn fragmented_reads_and_streaming_limits() {
    let data = b"fragmented input ".repeat(4000);
    let mut z = Vec::new();
    compress_stream(&mut Short(&data), &mut z, &cpu_options()).unwrap();
    let mut out = Vec::new();
    decompress_stream_with_limits(&mut Short(&z), &mut out, &DecodeLimits::default()).unwrap();
    assert_eq!(out, data);
}

#[test]
fn bounded_mutation_smoke() {
    let data = b"mutate compressed payload ".repeat(200);
    let seed = compress_with_options(&data, &cpu_options()).unwrap();
    assert_eq!(
        decompress_with_limits(&seed, &DecodeLimits::default()).unwrap(),
        data
    );
    let limits = DecodeLimits {
        max_output_bytes: 1024 * 1024,
        max_working_memory: bato_engine::MemoryBudget::Bytes(256 * 1024 * 1024),
        ..Default::default()
    };
    let mut rng = 171u64;
    for _ in 0..128 {
        let mut z = seed.clone();
        for _ in 0..3 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let at = rng as usize % z.len();
            z[at] ^= (rng >> 32) as u8;
        }
        // Corrupt input may decode or fail; it only must not panic or overrun.
        let _ = decompress_with_limits(&z, &limits);
    }
}

#[test]
fn stream_layout_lists_every_block_without_decoding() {
    let data: Vec<u8> = (0..40_000u32).map(|i| (i % 251) as u8).collect();
    let encoded = compress_small(&data).unwrap();
    let layout = read_stream_layout(&mut &encoded[..]).unwrap();
    assert_eq!(layout.header.kind, StreamKind::Bytes);
    assert_eq!(layout.geometry, cpu_options().geometry);
    let (end, chunks) = layout.blocks.split_last().unwrap();
    assert_eq!(end.block_type, BlockType::EndOfStream);
    let total: usize = chunks.iter().map(|b| b.uncompressed_size as usize).sum();
    assert_eq!(total, data.len());
    let truncated = &encoded[..encoded.len() - 30];
    assert!(read_stream_layout(&mut &truncated[..]).is_err());
}

#[test]
fn default_decode_limit_spans_the_format_length_range() {
    assert_eq!(DecodeLimits::default().max_output_bytes, u64::MAX);
}
