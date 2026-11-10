use bato_engine::{
    create_solid_archive, extract_solid_archive_with_options, CompressionOptions, DecodeLimits,
    Overwrite, StreamKind,
};
use std::fs;
use std::io::Cursor;
#[cfg(unix)]
use std::os::unix::fs::symlink;
use tempfile::tempdir;

fn roundtrip(src: &std::path::Path, dst: &std::path::Path, overwrite: Overwrite) {
    let mut archive = Vec::new();
    create_solid_archive(src, &mut archive, &CompressionOptions::default()).unwrap();
    extract_solid_archive_with_options(
        &mut Cursor::new(archive),
        dst,
        &DecodeLimits::default(),
        overwrite,
    )
    .unwrap();
}

#[test]
fn nested_and_duplicate_files_roundtrip() {
    let t = tempdir().unwrap();
    let src = t.path().join("src");
    let dst = t.path().join("dst");
    fs::create_dir_all(src.join("a/b")).unwrap();
    fs::write(src.join("a/b/one"), b"same payload").unwrap();
    fs::write(src.join("a/two"), b"same payload").unwrap();
    fs::write(src.join("three"), b"different").unwrap();
    roundtrip(&src, &dst, Overwrite::Refuse);
    assert_eq!(fs::read(dst.join("a/b/one")).unwrap(), b"same payload");
    assert_eq!(fs::read(dst.join("a/two")).unwrap(), b"same payload");
    assert_eq!(fs::read(dst.join("three")).unwrap(), b"different");
}

#[test]
fn archive_marker_distinguishes_a_catalogue_from_arbitrary_file_bytes() {
    use bato_engine::{compress_with_options, read_stream_header, ExecutionOptions};
    let t = tempdir().unwrap();
    let source = t.path().join("source");
    fs::create_dir(&source).unwrap();
    let options = CompressionOptions {
        execution: ExecutionOptions {
            threads: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut archive = Vec::new();
    create_solid_archive(&source, &mut archive, &options).unwrap();
    assert!(
        read_stream_header(&mut Cursor::new(&archive)).unwrap().kind
            == StreamKind::DirectoryArchive
    );
    let destination = t.path().join("destination");
    archive[6] &= !(bato_engine::core::types::FLAG_DIRECTORY_ARCHIVE);
    assert!(extract_solid_archive_with_options(
        &mut Cursor::new(archive),
        &destination,
        &DecodeLimits::default(),
        Overwrite::Refuse
    )
    .is_err());
    assert!(!destination.exists());
    // A perfectly valid empty catalogue is also a possible user file. The
    // archive parser must never infer its kind from these payload bytes.
    let bytes = compress_with_options(b"BATO\0", &options).unwrap();
    assert!(read_stream_header(&mut Cursor::new(&bytes)).unwrap().kind == StreamKind::Bytes);
    assert!(extract_solid_archive_with_options(
        &mut Cursor::new(bytes),
        &destination,
        &DecodeLimits::default(),
        Overwrite::Refuse
    )
    .is_err());
    assert!(!destination.exists());
}

#[test]
#[cfg(unix)]
fn source_and_destination_symlinks_are_rejected() {
    let t = tempdir().unwrap();
    let src = t.path().join("src");
    fs::create_dir(&src).unwrap();
    symlink("/tmp", src.join("escape")).unwrap();
    let mut out = Vec::new();
    assert!(create_solid_archive(&src, &mut out, &CompressionOptions::default()).is_err());
    let real = t.path().join("real");
    fs::create_dir(&real).unwrap();
    fs::write(real.join("x"), b"x").unwrap();
    let mut archive = Vec::new();
    create_solid_archive(&real, &mut archive, &CompressionOptions::default()).unwrap();
    let link = t.path().join("link");
    symlink(&real, &link).unwrap();
    assert!(extract_solid_archive_with_options(
        &mut Cursor::new(archive),
        &link,
        &DecodeLimits::default(),
        Overwrite::Refuse
    )
    .is_err());
}

#[test]
fn collision_failure_preserves_existing_tree() {
    let t = tempdir().unwrap();
    let src = t.path().join("src");
    let dst = t.path().join("dst");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("x"), b"new").unwrap();
    fs::create_dir_all(&dst).unwrap();
    fs::write(dst.join("x"), b"old").unwrap();
    let mut archive = Vec::new();
    create_solid_archive(&src, &mut archive, &CompressionOptions::default()).unwrap();
    assert!(extract_solid_archive_with_options(
        &mut Cursor::new(archive),
        &dst,
        &DecodeLimits::default(),
        Overwrite::Refuse
    )
    .is_err());
    assert_eq!(fs::read(dst.join("x")).unwrap(), b"old");
}

#[test]
fn force_replaces_only_archive_files_and_preserves_unrelated() {
    let t = tempdir().unwrap();
    let src = t.path().join("src");
    let dst = t.path().join("dst");
    fs::create_dir_all(&src).unwrap();
    fs::create_dir_all(&dst).unwrap();
    fs::write(src.join("x"), b"new").unwrap();
    fs::write(dst.join("x"), b"old").unwrap();
    fs::write(dst.join("keep"), b"keep").unwrap();
    roundtrip(&src, &dst, Overwrite::Replace);
    assert_eq!(fs::read(dst.join("x")).unwrap(), b"new");
    assert_eq!(fs::read(dst.join("keep")).unwrap(), b"keep");
}

#[test]
fn corrupted_archive_does_not_publish_or_overwrite() {
    let t = tempdir().unwrap();
    let src = t.path().join("src");
    let dst = t.path().join("dst");
    fs::create_dir_all(&src).unwrap();
    fs::create_dir_all(&dst).unwrap();
    fs::write(src.join("x"), b"new content").unwrap();
    fs::write(dst.join("x"), b"old").unwrap();
    let mut archive = Vec::new();
    create_solid_archive(&src, &mut archive, &CompressionOptions::default()).unwrap();
    let last = archive.len() - 1;
    archive[last] ^= 0x5a;
    assert!(extract_solid_archive_with_options(
        &mut Cursor::new(archive),
        &dst,
        &DecodeLimits::default(),
        Overwrite::Replace
    )
    .is_err());
    assert_eq!(fs::read(dst.join("x")).unwrap(), b"old");
}

#[test]
fn directory_collision_is_rejected_before_any_publication() {
    let t = tempdir().unwrap();
    let src = t.path().join("src");
    let dst = t.path().join("dst");
    fs::create_dir_all(src.join("zdir")).unwrap();
    fs::create_dir(&dst).unwrap();
    fs::write(src.join("a"), b"new").unwrap();
    fs::write(src.join("zdir/file"), b"new").unwrap();
    fs::write(dst.join("zdir"), b"existing file").unwrap();
    let mut archive = Vec::new();
    create_solid_archive(&src, &mut archive, &CompressionOptions::default()).unwrap();
    assert!(extract_solid_archive_with_options(
        &mut Cursor::new(archive),
        &dst,
        &DecodeLimits::default(),
        Overwrite::Replace
    )
    .is_err());
    assert!(!dst.join("a").exists());
    assert_eq!(fs::read(dst.join("zdir")).unwrap(), b"existing file");
}
#[test]
#[cfg(unix)]
fn source_ancestor_symlink_is_rejected() {
    let t = tempdir().unwrap();
    let src = t.path().join("real/src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("x"), b"x").unwrap();
    symlink(t.path().join("real"), t.path().join("link")).unwrap();
    assert!(create_solid_archive(
        t.path().join("link/src"),
        &mut Vec::new(),
        &CompressionOptions::default()
    )
    .is_err());
}

#[cfg(unix)]
#[test]
fn replacing_destination_ancestor_during_read_does_not_publish_or_redirect_cleanup() {
    use std::io::{self, Read};
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("source");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("file"), b"archive contents").unwrap();
    let mut packed = Vec::new();
    create_solid_archive(&source, &mut packed, &CompressionOptions::default()).unwrap();
    let parent = root.path().join("parent");
    fs::create_dir_all(parent.join("destination")).unwrap();
    fs::write(parent.join("destination/keep"), b"unrelated").unwrap();
    struct SwapReader {
        input: Cursor<Vec<u8>>,
        parent: std::path::PathBuf,
        moved: std::path::PathBuf,
        swapped: bool,
    }
    impl Read for SwapReader {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if !self.swapped {
                fs::rename(&self.parent, &self.moved)?;
                fs::create_dir(&self.parent)?;
                fs::write(self.parent.join("keep"), b"replacement tree")?;
                self.swapped = true;
            }
            self.input.read(out)
        }
    }
    let moved = root.path().join("original");
    let mut reader = SwapReader {
        input: Cursor::new(packed),
        parent: parent.clone(),
        moved: moved.clone(),
        swapped: false,
    };
    assert!(extract_solid_archive_with_options(
        &mut reader,
        parent.join("destination"),
        &DecodeLimits::default(),
        Overwrite::Refuse
    )
    .is_err());
    assert!(!parent.join("destination").exists());
    assert_eq!(fs::read(parent.join("keep")).unwrap(), b"replacement tree");
    assert_eq!(
        fs::read(moved.join("destination/keep")).unwrap(),
        b"unrelated"
    );
    assert!(!moved.join("destination/file").exists());
    assert_eq!(
        fs::read_dir(&moved).unwrap().count(),
        1,
        "staging directory was not cleaned through its original parent"
    );
}

#[cfg(unix)]
#[test]
fn source_directory_replacement_is_detected_even_with_identical_file_bytes() {
    use std::io::{self, Write};
    let root = tempdir().unwrap();
    let source = root.path().join("source");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("file"), b"same bytes").unwrap();
    struct SwapWriter {
        bytes: Vec<u8>,
        source: std::path::PathBuf,
        original: std::path::PathBuf,
        swapped: bool,
    }
    impl Write for SwapWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if !self.swapped {
                fs::rename(&self.source, &self.original)?;
                fs::create_dir(&self.source)?;
                fs::write(self.source.join("file"), b"same bytes")?;
                self.swapped = true;
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut writer = SwapWriter {
        bytes: Vec::new(),
        source: source.clone(),
        original: root.path().join("original"),
        swapped: false,
    };
    assert!(create_solid_archive(&source, &mut writer, &CompressionOptions::default()).is_err());
    assert_eq!(
        fs::read(root.path().join("original/file")).unwrap(),
        b"same bytes"
    );
    assert_eq!(fs::read(source.join("file")).unwrap(), b"same bytes");
}
#[test]
fn missing_destination_parents_roundtrip() {
    let t = tempdir().unwrap();
    let src = t.path().join("src");
    fs::create_dir(&src).unwrap();
    fs::write(src.join("x"), b"x").unwrap();
    let dst = t.path().join("new/parent/out");
    roundtrip(&src, &dst, Overwrite::Refuse);
    assert_eq!(fs::read(dst.join("x")).unwrap(), b"x");
}

#[test]
#[cfg(unix)]
fn unrepresentable_source_names_fail_before_writing_archive() {
    let t = tempdir().unwrap();
    fs::write(t.path().join("a:b"), b"data").unwrap();
    let mut output = Vec::new();
    assert!(create_solid_archive(t.path(), &mut output, &CompressionOptions::default()).is_err());
    assert!(output.is_empty());
}
