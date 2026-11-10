use bato_engine::*;
use std::{
    fs,
    io::{self, Cursor, Read},
};
use tempfile::tempdir;
fn wrap(records: &[u8]) -> Vec<u8> {
    let mut encoded = compress_with_options(
        records,
        &CompressionOptions {
            execution: bato_engine::ExecutionOptions {
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    // These fixtures deliberately exercise the catalogue parser, inside a
    // correctly marked archive envelope rather than an ordinary byte stream.
    encoded[6] |= bato_engine::core::types::FLAG_DIRECTORY_ARCHIVE;
    encoded
}
fn path(v: &mut Vec<u8>, kind: u8, p: &str) {
    v.push(kind);
    v.extend_from_slice(&(p.len() as u16).to_le_bytes());
    v.extend_from_slice(p.as_bytes())
}
fn file(v: &mut Vec<u8>, p: &str, data: &[u8]) {
    path(v, 2, p);
    v.extend_from_slice(&(data.len() as u64).to_le_bytes());
    v.extend_from_slice(data);
    v.extend_from_slice(&bato_engine::core::checksum::crc32c(data).to_le_bytes())
}
fn extract(v: &[u8], l: &DecodeLimits) -> bool {
    let t = tempdir().unwrap();
    let options = bato_engine::DecompressionOptions {
        limits: l.clone(),
        execution: bato_engine::ExecutionOptions {
            ..Default::default()
        },
    };
    bato_engine::extract_solid_archive_with_execution(
        &mut Cursor::new(wrap(v)),
        t.path().join("out"),
        bato_engine::Overwrite::Refuse,
        &options,
    )
    .is_ok()
}
#[test]
fn unsafe_paths_duplicate_names_prefixes_and_forward_refs() {
    for name in ["../escape", "a/../escape", "/abs", "a\\escape", "", "a//b"] {
        let mut v = Vec::new();
        file(&mut v, name, b"x");
        v.push(0);
        assert!(!extract(&v, &DecodeLimits::default()), "{name}")
    }
    let mut v = Vec::new();
    file(&mut v, "a", b"one");
    file(&mut v, "a", b"two");
    v.push(0);
    assert!(!extract(&v, &DecodeLimits::default()));
    let mut v = Vec::new();
    file(&mut v, "a", b"one");
    file(&mut v, "a/b", b"two");
    v.push(0);
    assert!(!extract(&v, &DecodeLimits::default()));
    let mut v = Vec::new();
    path(&mut v, 3, "dup");
    v.extend_from_slice(&1u64.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.push(0);
    assert!(!extract(&v, &DecodeLimits::default()));
}

#[cfg(any(windows, target_os = "macos"))]
#[test]
fn filesystem_name_aliases_fail_before_publication_without_renaming_entries() {
    for (first, second) in [("Dir", "dir"), ("é", "e\u{301}")] {
        let root = tempdir().unwrap();
        let probe = root.path().join(first);
        fs::create_dir(&probe).unwrap();
        if !root.path().join(second).exists() {
            eprintln!("UNTESTED: filesystem does not alias {first:?} and {second:?}");
            continue;
        }
        let mut catalogue = Vec::new();
        file(&mut catalogue, &format!("{first}/a"), b"a");
        file(&mut catalogue, &format!("{second}/b"), b"b");
        catalogue.push(0);
        let encoded = wrap(&catalogue);
        for force in [false, true] {
            let destination = root.path().join(if force { "forced" } else { "output" });
            assert!(extract_solid_archive_with_options(
                &mut Cursor::new(&encoded),
                &destination,
                &DecodeLimits::default(),
                force
            )
            .is_err());
            assert!(!destination.exists(), "name aliases published output");
        }
    }
}
#[test]
fn duplicates_charge_materialized_bytes() {
    let data = [7; 1000];
    let mut v = Vec::new();
    file(&mut v, "original", &data);
    for p in ["dup1", "dup2"] {
        path(&mut v, 3, p);
        v.extend_from_slice(&1000u64.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&bato_engine::core::checksum::crc32c(&data).to_le_bytes())
    }
    v.push(0);
    let l = DecodeLimits {
        max_output_bytes: 2500,
        ..Default::default()
    };
    assert!(!extract(&v, &l));
    assert!(extract(
        &v,
        &DecodeLimits {
            max_output_bytes: 3000,
            ..l
        }
    ));
}
#[test]
fn entries_metadata_trailing_and_valid_dots() {
    let mut v = Vec::new();
    path(&mut v, 1, "empty");
    file(&mut v, "file..txt", b"valid");
    v.push(0);
    assert!(extract(&v, &DecodeLimits::default()));
    assert!(!extract(
        &v,
        &DecodeLimits {
            max_archive_entries: 1,
            ..Default::default()
        }
    ));
    assert!(!extract(
        &v,
        &DecodeLimits {
            max_metadata_bytes: 2,
            ..Default::default()
        }
    ));
    v.push(0);
    assert!(!extract(&v, &DecodeLimits::default()));
}
#[test]
fn bounded_record_mutation_fuzz() {
    let t = tempdir().unwrap();
    let mut seed = Vec::new();
    file(&mut seed, "nested/file..txt", b"one");
    path(&mut seed, 1, "empty");
    seed.push(0);
    assert!(extract(&seed, &DecodeLimits::default()));
    let limits = DecodeLimits {
        max_output_bytes: 65536,
        max_working_memory: bato_engine::MemoryBudget::Bytes(256 * 1024 * 1024),
        max_metadata_bytes: 65536,
        max_archive_entries: 32,
        max_path_bytes: 256,
    };
    let mut rng = 192891u64;
    for i in 0..500 {
        let mut v = seed.clone();
        for _ in 0..3 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let n = rng as usize % v.len();
            v[n] ^= (rng >> 24) as u8;
        }
        let dest = t.path().join(format!("out{i}"));
        let _ = extract_solid_archive_with_options(
            &mut Cursor::new(wrap(&v)),
            &dest,
            &limits,
            bato_engine::Overwrite::Refuse,
        );
        if dest.exists() {
            fs::remove_dir_all(dest).unwrap()
        }
    }
}

#[test]
fn exact_output_limit_excludes_record_metadata() {
    let mut v = Vec::new();
    file(&mut v, "nested/file", b"abc");
    path(&mut v, 1, "empty");
    v.push(0);
    assert!(extract(
        &v,
        &DecodeLimits {
            max_output_bytes: 3,
            ..Default::default()
        }
    ));
    assert!(!extract(
        &v,
        &DecodeLimits {
            max_output_bytes: 2,
            ..Default::default()
        }
    ));
    let mut v = Vec::new();
    path(&mut v, 1, "empty");
    v.push(0);
    assert!(extract(
        &v,
        &DecodeLimits {
            max_output_bytes: 0,
            ..Default::default()
        }
    ));
}

#[test]
fn deeply_nested_paths_fail_without_recursive_publication() {
    let mut v = Vec::new();
    file(&mut v, &vec!["a"; 129].join("/"), b"x");
    v.push(0);
    assert!(!extract(&v, &DecodeLimits::default()));
}

fn extraction_error<R: Read>(mut reader: R, limits: DecodeLimits) -> BatoError {
    let t = tempdir().unwrap();
    let destination = t.path().join("out");
    let error = extract_solid_archive_with_execution(
        &mut reader,
        &destination,
        bato_engine::Overwrite::Refuse,
        &DecompressionOptions {
            limits,
            execution: ExecutionOptions {
                threads: 1,
                ..Default::default()
            },
        },
    )
    .unwrap_err();
    assert!(!destination.exists(), "failed extraction published files");
    assert_eq!(fs::read_dir(t.path()).unwrap().count(), 0, "staging leaked");
    error
}

#[test]
fn extraction_preserves_outer_checksum_and_output_limit_errors() {
    let mut catalogue = Vec::new();
    file(&mut catalogue, "file", b"payload");
    catalogue.push(0);
    let mut encoded = wrap(&catalogue);
    let mut header = Cursor::new(&encoded);
    header.set_position(10);
    let block = read_block_header(&mut header).unwrap();
    let crc_at = header.position() as usize + block.compressed_size as usize;
    encoded[crc_at] ^= 1;
    assert!(matches!(
        extraction_error(Cursor::new(encoded), DecodeLimits::default()),
        BatoError::ChecksumMismatch { .. }
    ));
    let limits = DecodeLimits {
        max_output_bytes: 0,
        max_metadata_bytes: 1,
        ..Default::default()
    };
    assert!(matches!(
        extraction_error(Cursor::new(wrap(&catalogue)), limits),
        BatoError::DecompressionLimitExceeded { .. }
    ));
}

#[test]
fn extraction_preserves_reader_errors_and_reports_real_catalogue_truncation() {
    struct FaultAfterHeader(Cursor<Vec<u8>>);
    impl Read for FaultAfterHeader {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            if self.0.position() >= 10 {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "injected reader error",
                ));
            }
            let remaining = (10 - self.0.position()) as usize;
            let count = remaining.min(bytes.len());
            self.0.read(&mut bytes[..count])
        }
    }
    let encoded = wrap(b"\0");
    let error = extraction_error(
        FaultAfterHeader(Cursor::new(encoded)),
        DecodeLimits::default(),
    );
    assert!(
        matches!(error, BatoError::Io(error) if error.kind() == io::ErrorKind::PermissionDenied)
    );

    assert!(matches!(
        extraction_error(Cursor::new(wrap(b"")), DecodeLimits::default()),
        BatoError::TruncatedHeader
    ));
    let mut payload = Vec::new();
    path(&mut payload, 2, "file");
    payload.extend_from_slice(&3u64.to_le_bytes());
    payload.push(b'x');
    assert!(matches!(
        extraction_error(Cursor::new(wrap(&payload)), DecodeLimits::default()),
        BatoError::TruncatedPayload { expected: 3, .. }
    ));
}

#[test]
fn entry_limit_accepts_exact_count_and_rejects_zero_or_one_extra() {
    let mut catalogue = Vec::new();
    path(&mut catalogue, 1, "empty");
    file(&mut catalogue, "file", b"x");
    catalogue.push(0);
    for limit in [0, 1] {
        let error = extraction_error(
            Cursor::new(wrap(&catalogue)),
            DecodeLimits {
                max_archive_entries: limit,
                ..Default::default()
            },
        );
        assert!(
            matches!(error, BatoError::DecompressionLimitExceeded { size, limit: actual }
            if size == limit + 1 && actual == limit)
        );
    }
    assert!(extract(
        &catalogue,
        &DecodeLimits {
            max_archive_entries: 2,
            ..Default::default()
        }
    ));
    assert!(extract(
        b"\0",
        &DecodeLimits {
            max_archive_entries: 0,
            ..Default::default()
        }
    ));
}
