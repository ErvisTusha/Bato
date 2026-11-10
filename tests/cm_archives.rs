use bato_engine::*;
use std::{fs, io::Cursor};

#[test]
fn cm_solid_tree_roundtrip_and_failed_decode_preserve_destination() {
    {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source");
        fs::create_dir_all(source.join("nested/λ")).unwrap();
        let binary: Vec<_> = (0..=255u8).cycle().take(100_000).collect();
        let nested = compress(&binary).unwrap();
        let files = [
            ("empty.png", vec![]),
            ("nested/λ/duplicate", binary.clone()),
            ("duplicate", binary),
            ("nested/data.txt", nested),
        ];
        for (name, bytes) in &files {
            fs::write(source.join(name), bytes).unwrap();
        }
        let options = CompressionOptions {
            geometry: Geometry {
                segment_size: 65536,
                chunk_size: 16384,
                ..Default::default()
            },
            execution: ExecutionOptions {
                threads: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut archive = Vec::new();
        create_solid_archive(&source, &mut archive, &options).unwrap();
        let destination = root.path().join("destination");
        extract_solid_archive_with_options(
            &mut Cursor::new(&archive),
            &destination,
            &DecodeLimits::default(),
            bato_engine::Overwrite::Refuse,
        )
        .unwrap();
        for (name, bytes) in &files {
            assert_eq!(&fs::read(destination.join(name)).unwrap(), bytes);
        }
        let before = fs::read(destination.join("duplicate")).unwrap();
        *archive.last_mut().unwrap() ^= 1;
        assert!(extract_solid_archive_with_options(
            &mut Cursor::new(&archive),
            &destination,
            &DecodeLimits::default(),
            bato_engine::Overwrite::Replace
        )
        .is_err());
        assert_eq!(fs::read(destination.join("duplicate")).unwrap(), before);
        let absent = root.path().join("absent");
        assert!(extract_solid_archive_with_options(
            &mut Cursor::new(&archive),
            &absent,
            &DecodeLimits::default(),
            bato_engine::Overwrite::Refuse
        )
        .is_err());
        assert!(!absent.exists());
    }
}

#[test]
fn cm_catalogue_path_and_inner_checksum_errors_are_staged() {
    fn file(catalogue: &mut Vec<u8>, path: &str, bytes: &[u8], bad_crc: bool) {
        catalogue.push(2);
        catalogue.extend((path.len() as u16).to_le_bytes());
        catalogue.extend(path.as_bytes());
        catalogue.extend((bytes.len() as u64).to_le_bytes());
        catalogue.extend(bytes);
        catalogue.extend(
            (bato_engine::core::checksum::crc32c(bytes) ^ u32::from(bad_crc)).to_le_bytes(),
        );
    }
    fn wrap(catalogue: &[u8]) -> Vec<u8> {
        let mut encoded = compress_with_options(
            catalogue,
            &CompressionOptions {
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
            },
        )
        .unwrap();
        encoded[6] |= bato_engine::core::types::FLAG_DIRECTORY_ARCHIVE;
        encoded
    }
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("destination");
    let mut unsafe_names = vec![
        "../escape".to_owned(),
        "a//b".to_owned(),
        "a\\escape".to_owned(),
        "a/../escape".to_owned(),
    ];
    unsafe_names.push(
        root.path()
            .join("absolute-escape")
            .to_str()
            .unwrap()
            .to_owned(),
    );
    for name in unsafe_names {
        let mut catalogue = Vec::new();
        file(&mut catalogue, &name, b"bytes", false);
        catalogue.push(0);
        assert!(extract_solid_archive_with_options(
            &mut Cursor::new(wrap(&catalogue)),
            &destination,
            &DecodeLimits::default(),
            bato_engine::Overwrite::Refuse
        )
        .is_err());
        assert!(!destination.exists());
        assert!(!root.path().join("escape").exists());
        assert!(!root.path().join("absolute-escape").exists());
    }
    let mut duplicate = Vec::new();
    file(&mut duplicate, "same", b"one", false);
    file(&mut duplicate, "same", b"two", false);
    duplicate.push(0);
    let error = extract_solid_archive_with_options(
        &mut Cursor::new(wrap(&duplicate)),
        &destination,
        &DecodeLimits::default(),
        bato_engine::Overwrite::Refuse,
    )
    .unwrap_err();
    assert!(
        matches!(&error, BatoError::CorruptStream(message) if message.contains("duplicate")),
        "{error:?}"
    );
    assert!(!destination.exists());
    fs::create_dir(&destination).unwrap();
    fs::write(destination.join("sentinel"), b"old").unwrap();
    let mut corrupt = Vec::new();
    file(
        &mut corrupt,
        "new",
        b"valid outer stream, invalid inner checksum",
        true,
    );
    corrupt.push(0);
    let error = extract_solid_archive_with_options(
        &mut Cursor::new(wrap(&corrupt)),
        &destination,
        &DecodeLimits::default(),
        bato_engine::Overwrite::Replace,
    )
    .unwrap_err();
    assert!(
        matches!(error, BatoError::ChecksumMismatch { .. }),
        "{error:?}"
    );
    assert_eq!(fs::read(destination.join("sentinel")).unwrap(), b"old");
    assert!(!destination.join("new").exists());
}
