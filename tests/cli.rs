use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn tempdir() -> PathBuf {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("bato-cli-{n:x}"));
    fs::create_dir(&p).unwrap();
    p
}
fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(args)
        .output()
        .unwrap()
}
fn write(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
}
fn cpu_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_bato"));
    command.args(["--threads", "1"]);
    command
}

#[test]
fn removed_mode_options_fail_before_output() {
    let data = b"one compression mode";
    let encode = |args: &[&str]| {
        let mut child = cpu_command()
            .args(args)
            .arg("-")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        // Rejected modes may exit before reading stdin (broken pipe); their
        // status and output are checked below.
        let _ = child.stdin.take().unwrap().write_all(data);
        child.wait_with_output().unwrap()
    };
    let implicit = encode(&[]);
    assert!(implicit.status.success());
    assert_eq!(&implicit.stdout[..4], b"BATO");
    assert_eq!(bato_engine::decompress(&implicit.stdout).unwrap(), data);
    for args in [
        vec!["--engine", "cm"],
        vec!["--cm-profile", "ultra"],
        vec!["--engine", "lz"],
    ] {
        let result = encode(&args);
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
    let help = run(&["--help"]);
    let help = String::from_utf8_lossy(&help.stderr);
    assert!(!help.contains("--engine"));
    assert!(!help.contains("--cm-profile"));
}

#[test]
fn file_roundtrip_and_header_info() {
    let root = tempdir();
    let source = root.join("unknown.jpg");
    let archive = root.join("encoded");
    let restored = root.join("restored");
    let raw = b"online integer model is enabled and content-independent\0\xff".repeat(500);
    write(&source, &raw);
    let result = cpu_command()
        .args(["-o"])
        .arg(&archive)
        .arg(&source)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert_eq!(&fs::read(&archive).unwrap()[..4], b"BATO");
    let info = cpu_command().arg("-i").arg(&archive).output().unwrap();
    assert!(info.status.success(), "{info:?}");
    assert!(String::from_utf8_lossy(&info.stdout).contains(&format!(
        "Format: Bato {}.{}",
        bato_engine::VERSION_MAJOR,
        bato_engine::VERSION_MINOR
    )));
    let result = cpu_command()
        .args(["-d", "-o"])
        .arg(&restored)
        .arg(&archive)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert_eq!(fs::read(&restored).unwrap(), raw);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn cm_file_directory_info_and_ignored_decode_options() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("arbitrary.bin");
    let packed = root.path().join("packed");
    let restored = root.path().join("restored");
    let raw = b"CM file and directory CLI coverage\n".repeat(3000);
    write(&source, &raw);
    let result = cpu_command()
        .args(["--segment-size", "65536", "--chunk-size", "16384", "-o"])
        .arg(&packed)
        .arg(&source)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    let info = cpu_command().arg("-i").arg(&packed).output().unwrap();
    assert!(info.status.success(), "{info:?}");
    let info = String::from_utf8_lossy(&info.stdout);
    assert!(info.contains("Kind: byte stream"));
    assert!(info.contains("segment_size=65536, chunk_size=16384"));
    let result = cpu_command()
        .args(["-d", "--segment-size", "3", "-o"])
        .arg(&restored)
        .arg(&packed)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert_eq!(fs::read(&restored).unwrap(), raw);
    let source_dir = root.path().join("tree");
    fs::create_dir(&source_dir).unwrap();
    write(&source_dir.join("empty.jpg"), b"");
    write(&source_dir.join("unicode-λ"), &raw);
    let archive = root.path().join("archive");
    let result = cpu_command()
        .args([
            "archive",
            "--segment-size",
            "65536",
            "--chunk-size",
            "16384",
            "-o",
        ])
        .arg(&archive)
        .arg(&source_dir)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    let destination = root.path().join("destination");
    let result = cpu_command()
        .args(["extract", "-o"])
        .arg(&destination)
        .arg(&archive)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert_eq!(fs::read(destination.join("unicode-λ")).unwrap(), raw);
    assert_eq!(fs::read(destination.join("empty.jpg")).unwrap(), b"");
    for args in [
        vec!["--segment-size", "3"],
        vec!["--compute", "hybrid"],
        vec!["--engine", "cm"],
    ] {
        assert!(!cpu_command()
            .args(args)
            .arg(&source)
            .output()
            .unwrap()
            .status
            .success());
    }
}

#[test]
fn cm_stdin_stdout_roundtrip() {
    use std::process::Stdio;
    let raw = b"CM stdin and stdout exact bytes\n".repeat(200);
    let mut encode = cpu_command()
        .args(["-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    encode.stdin.take().unwrap().write_all(&raw).unwrap();
    let encoded = encode.wait_with_output().unwrap();
    assert!(encoded.status.success());
    let mut decode = cpu_command()
        .args(["-d", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    decode
        .stdin
        .take()
        .unwrap()
        .write_all(&encoded.stdout)
        .unwrap();
    let decoded = decode.wait_with_output().unwrap();
    assert!(decoded.status.success());
    assert_eq!(decoded.stdout, raw);
}

#[test]
fn repeated_output_options_use_the_last_destination() {
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("input");
    let packed = d.path().join("packed");
    let unused = d.path().join("unused");
    let restored = d.path().join("restored");
    let data = b"last output option wins";
    write(&src, data);

    let result = cpu_command()
        .args(["-o", "-", "-o"])
        .arg(&packed)
        .arg(&src)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert!(result.stdout.is_empty());
    let result = cpu_command()
        .arg(&src)
        .arg("-o")
        .arg(&unused)
        .args(["-o", "-"])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert_eq!(result.stdout, fs::read(&packed).unwrap());
    assert!(!unused.exists());

    let result = cpu_command()
        .arg("-d")
        .arg(&packed)
        .args(["-o", "-", "-o"])
        .arg(&restored)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert!(result.stdout.is_empty());
    assert_eq!(fs::read(&restored).unwrap(), data);
    let result = cpu_command()
        .arg("-d")
        .arg(&packed)
        .arg("-o")
        .arg(&unused)
        .args(["-o", "-"])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert_eq!(result.stdout, data);
    assert!(!unused.exists());

    let second = d.path().join("second");
    let result = cpu_command()
        .arg(&src)
        .arg("-o")
        .arg(&unused)
        .arg("-o")
        .arg(&second)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert_eq!(fs::read(second).unwrap(), fs::read(packed).unwrap());
    assert!(!unused.exists());
}

#[test]
fn repeated_archive_output_options_validate_the_final_destination() {
    let d = tempfile::tempdir().unwrap();
    let source = d.path().join("source");
    fs::create_dir(&source).unwrap();
    write(&source.join("file"), b"archive data");
    let packed = d.path().join("packed");
    let result = cpu_command()
        .arg("archive")
        .arg(&source)
        .args(["-o", "-", "-o"])
        .arg(&packed)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert!(result.stdout.is_empty());
    let unused = d.path().join("unused");
    let result = cpu_command()
        .arg("archive")
        .arg(&source)
        .arg("-o")
        .arg(&unused)
        .args(["-o", "-"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(!unused.exists());
    let restored = d.path().join("restored");
    let result = cpu_command()
        .arg("extract")
        .arg(&packed)
        .args(["-o", "-", "-o"])
        .arg(&restored)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert_eq!(fs::read(restored.join("file")).unwrap(), b"archive data");
    let result = cpu_command()
        .arg("extract")
        .arg(&packed)
        .arg("-o")
        .arg(&unused)
        .args(["-o", "-"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!unused.exists());
}

#[test]
fn unknown_and_misleading_file_names_use_identical_lossless_streams() {
    let d = tempfile::tempdir().unwrap();
    let data: Vec<u8> = b"BATO\0"
        .iter()
        .copied()
        .chain((0..4096).map(|i| i as u8))
        .collect();
    let mut reference = None;
    for name in [
        "extensionless",
        "opaque.future987",
        "photo.jpg",
        "archive.bato",
    ] {
        let src = d.path().join(name);
        write(&src, &data);
        let result = cpu_command().arg(&src).output().unwrap();
        assert!(result.status.success(), "{result:?}");
        let compressed = d.path().join(format!("{name}.bato"));
        let bytes = fs::read(&compressed).unwrap();
        if let Some(expected) = &reference {
            assert_eq!(&bytes, expected, "filename changed codec output: {name}");
        } else {
            reference = Some(bytes);
        }
        assert_eq!(fs::read(&src).unwrap(), data);
        fs::remove_file(&src).unwrap();
        let result = cpu_command().arg("-d").arg(&compressed).output().unwrap();
        assert!(result.status.success(), "{result:?}");
        assert_eq!(fs::read(src).unwrap(), data);
    }
}

#[test]
fn byte_stream_named_like_an_archive_is_never_interpreted_as_a_directory() {
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("source");
    // Even valid archive-catalogue bytes remain an ordinary file when written
    // through the byte-stream compressor.
    write(&src, b"BATO\0");
    let packed = d.path().join("misleading.bato");
    let result = cpu_command()
        .arg(&src)
        .arg("-o")
        .arg(&packed)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    for mode in ["-t", "-i"] {
        let result = cpu_command().arg(mode).arg(&packed).output().unwrap();
        assert!(result.status.success(), "{result:?}");
        if mode == "-i" {
            assert!(String::from_utf8(result.stdout)
                .unwrap()
                .contains("Kind: byte stream"));
        }
    }
    let result = cpu_command().arg("-d").arg(&packed).output().unwrap();
    assert!(result.status.success(), "{result:?}");
    assert_eq!(fs::read(d.path().join("misleading")).unwrap(), b"BATO\0");
    let refused = d.path().join("refused");
    let result = cpu_command()
        .arg("extract")
        .arg(&packed)
        .arg("-o")
        .arg(&refused)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!refused.exists());
}

#[test]
fn renamed_archive_tests_verify_inner_file_checksums_before_publication() {
    let d = tempfile::tempdir().unwrap();
    let mut catalogue = b"\x02\x01\0x".to_vec();
    catalogue.extend(3u64.to_le_bytes());
    catalogue.extend(b"abc");
    catalogue.extend(0u32.to_le_bytes()); // deliberately wrong file CRC32C
    catalogue.push(0);
    let mut packed = bato_engine::compress_with_options(
        &catalogue,
        &bato_engine::CompressionOptions {
            execution: bato_engine::ExecutionOptions {
                threads: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    packed[6] |= bato_engine::core::types::FLAG_DIRECTORY_ARCHIVE;
    // Outer group and stream checksums are valid, so byte-stream validation
    // alone would accept this malformed catalogue.
    assert_eq!(bato_engine::decompress(&packed).unwrap(), catalogue);
    let input = d.path().join("no-known-suffix");
    write(&input, &packed);
    for mode in ["-t", "-i", "-d"] {
        let destination = d.path().join(format!("out-{mode}"));
        let result = cpu_command()
            .arg(mode)
            .arg(&input)
            .arg("-o")
            .arg(&destination)
            .output()
            .unwrap();
        assert!(
            !result.status.success(),
            "accepted corrupt file checksum: {result:?}"
        );
        assert!(String::from_utf8(result.stderr)
            .unwrap()
            .contains("CRC32C mismatch"));
        assert!(!destination.exists());
    }
}

#[test]
fn mixed_directory_archive_follows_its_header_after_renaming_and_piping() {
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("source.with.extension");
    fs::create_dir_all(src.join("nested/empty-directory")).unwrap();
    let binary: Vec<u8> = (0..768).map(|i| i as u8).collect();
    let mut random = Vec::new();
    let mut state = 739u64;
    for _ in 0..65_537 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        random.push(state as u8);
    }
    let already_encoded = bato_engine::compress_with_options(
        &random,
        &bato_engine::CompressionOptions {
            execution: bato_engine::ExecutionOptions {
                threads: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let files: [(&str, &[u8]); 7] = [
        ("extensionless", &binary),
        ("nested/opaque.new-format", &random),
        ("duplicate.jpg", &random),
        ("picture.png", b"this happens to be text\n"),
        ("invalid-utf8.txt", b"\xff\xfe\0\x80BATO\0BATO"),
        ("nested/compressed.payload", &already_encoded),
        ("empty-file", b""),
    ];
    for (name, data) in files {
        write(&src.join(name), data);
    }
    let result = cpu_command().arg("archive").arg(&src).output().unwrap();
    assert!(result.status.success(), "{result:?}");
    let original = d.path().join("source.with.extension.bato");
    let packed = d.path().join("renamed.unknown-format");
    fs::rename(original, &packed).unwrap();
    for mode in ["-t", "-i"] {
        let result = cpu_command().arg(mode).arg(&packed).output().unwrap();
        assert!(result.status.success(), "{result:?}");
        if mode == "-i" {
            assert!(String::from_utf8(result.stdout)
                .unwrap()
                .contains("Kind: directory archive"));
        }
    }
    let result = cpu_command().arg("-d").arg(&packed).output().unwrap();
    assert!(result.status.success(), "{result:?}");
    let restored = d.path().join("renamed.unknown-format.extracted");
    for (name, data) in files {
        assert_eq!(fs::read(restored.join(name)).unwrap(), data, "{name}");
    }
    assert!(restored.join("nested/empty-directory").is_dir());
    let piped = d.path().join("from-stdin");
    let result = cpu_command()
        .args(["extract", "-", "-o"])
        .arg(&piped)
        .stdin(File::open(&packed).unwrap())
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    for (name, data) in files {
        assert_eq!(fs::read(piped.join(name)).unwrap(), data, "{name}");
    }
    let result = cpu_command()
        .args(["-t", "-"])
        .stdin(File::open(&packed).unwrap())
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    let result = cpu_command()
        .args(["-d", "-o", "-"])
        .arg(&packed)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
}

#[test]
fn rejects_missing_or_invalid_execution_values() {
    for args in [
        &["--threads"][..],
        &["--threads", "wat"][..],
        &["--threads", "257"][..],
    ] {
        assert!(!run(args).status.success(), "accepted invalid {:?}", args);
    }
    let d = tempdir();
    let src = d.join("in");
    write(&src, b"hello");
    let out = d.join("out.bato");
    let r = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["--threads", "0", "-o"])
        .arg(&out)
        .arg(&src)
        .output()
        .unwrap();
    assert!(r.status.success(), "threads=0 failed: {:?}", r);
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn same_path_and_hardlink_are_rejected() {
    let d = tempdir();
    let src = d.join("input");
    let alias = d.join("alias");
    write(&src, b"same input");
    let r = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["-f", "-o"])
        .arg(&src)
        .arg(&src)
        .output()
        .unwrap();
    assert!(!r.status.success());
    assert_eq!(fs::read(&src).unwrap(), b"same input");
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&src, &alias).unwrap();
        let r = Command::new(env!("CARGO_BIN_EXE_bato"))
            .args(["-f", "-o"])
            .arg(&alias)
            .arg(&src)
            .output()
            .unwrap();
        assert!(!r.status.success());
        assert_eq!(fs::read(&src).unwrap(), b"same input");
    }
    #[cfg(unix)]
    {
        let hard = d.join("hard");
        fs::hard_link(&src, &hard).unwrap();
        let r = Command::new(env!("CARGO_BIN_EXE_bato"))
            .args(["-f", "-o"])
            .arg(&hard)
            .arg(&src)
            .output()
            .unwrap();
        assert!(!r.status.success());
    }
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn failed_decode_preserves_existing_destination() {
    let d = tempdir();
    let bad = d.join("bad.bato");
    let dst = d.join("output");
    write(&bad, b"BTS4\x04\0\0");
    write(&dst, b"keep me");
    let r = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["-d", "-f", "-o"])
        .arg(&dst)
        .arg(&bad)
        .output()
        .unwrap();
    assert!(!r.status.success());
    assert_eq!(fs::read(&dst).unwrap(), b"keep me");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn automatic_memory_and_failed_explicit_admission_preserve_destination() {
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("input");
    let dst = d.path().join("output");
    write(&src, b"ordinary byte contents");
    let admitted = cpu_command()
        .args(["--memory-budget", "auto", "--stats", "-o"])
        .arg(&dst)
        .arg(&src)
        .output()
        .unwrap();
    assert!(admitted.status.success(), "{admitted:?}");
    let stats = String::from_utf8(admitted.stderr).unwrap();
    assert!(stats.contains("memory_budget:"));
    assert!(stats.contains("managed_host_peak:"));
    let before = fs::read(&dst).unwrap();
    let refused = cpu_command()
        .args(["--memory-budget", "1", "-f", "-o"])
        .arg(&dst)
        .arg(&src)
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert_eq!(fs::read(&dst).unwrap(), before);
    let names: Vec<_> = fs::read_dir(d.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 2, "failed admission left a temporary output");
}

#[test]
fn force_replaces_existing_regular_output() {
    let d = tempdir();
    let src = d.join("input");
    let dst = d.join("output");
    write(&src, b"new contents");
    write(&dst, b"old contents");
    let r = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["-f", "-o"])
        .arg(&dst)
        .arg(&src)
        .output()
        .unwrap();
    assert!(r.status.success(), "{r:?}");
    assert_ne!(fs::read(&dst).unwrap(), b"old contents");
    fs::remove_dir_all(d).unwrap();
}

#[cfg(unix)]
#[test]
fn force_refuses_fifos_and_sockets_without_blocking_or_replacing_them() {
    use std::time::{Duration, Instant};
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("input");
    write(&src, b"regular input");
    let fifo = d.path().join("fifo");
    let name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // Use the portable FIFO API; this host disallows generic mknodat.
    assert_eq!(
        unsafe { libc::mkfifo(name.as_ptr(), 0o600) },
        0,
        "{}",
        std::io::Error::last_os_error()
    );
    let socket = d.path().join("socket");
    let socket_name = CString::new(socket.as_os_str().as_bytes()).unwrap();
    // A socket inode is enough to test replacement; no network listener is
    // needed (this host does not permit Unix socket bind).
    assert_eq!(
        unsafe { libc::mknod(socket_name.as_ptr(), libc::S_IFSOCK | 0o600, 0) },
        0,
        "{}",
        std::io::Error::last_os_error()
    );
    for target in [&fifo, &socket] {
        let mut child = cpu_command()
            .args(["-f", "-o"])
            .arg(target)
            .arg(&src)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let start = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if start.elapsed() > Duration::from_secs(2) {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("special output blocked the CLI");
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(!status.success());
        assert!(!fs::symlink_metadata(target).unwrap().is_file());
    }
}

#[test]
fn info_rejects_malformed_stream() {
    let d = tempdir();
    let bad = d.join("bad.bato");
    write(&bad, b"BTS4\x04\0\0\0\0\0\0\0");
    let r = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["--info"])
        .arg(&bad)
        .output()
        .unwrap();
    assert!(!r.status.success());
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn v1_rejects_legacy_compressor_options() {
    let d = tempdir();
    let src = d.join("input");
    let out = d.join("input.bato");
    write(&src, &b"custom dictionary data ".repeat(5000));
    let r = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args([
            "--backend",
            "bato",
            "--chunk-size",
            "65536",
            "--threads",
            "1",
            "-o",
        ])
        .arg(&out)
        .arg(&src)
        .output()
        .unwrap();
    assert!(!r.status.success(), "legacy backend option accepted: {r:?}");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn relative_directory_with_absolute_in_tree_output_excludes_output_artifacts() {
    let d = tempdir();
    let source = d.join("project");
    fs::create_dir(&source).unwrap();
    write(&source.join("data.txt"), b"stable archive input");
    let archive = source.join("archive.bato");
    let r = Command::new(env!("CARGO_BIN_EXE_bato"))
        .current_dir(&d)
        .args(["archive", "project", "-o"])
        .arg(&archive)
        .output()
        .unwrap();
    assert!(r.status.success(), "{r:?}");
    let restored = d.join("restored");
    let r = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["extract", "-o"])
        .arg(&restored)
        .arg(&archive)
        .output()
        .unwrap();
    assert!(r.status.success(), "{r:?}");
    assert_eq!(
        fs::read(restored.join("data.txt")).unwrap(),
        b"stable archive input"
    );
    assert!(!restored.join("archive.bato").exists());
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn stdin_output_roundtrip() {
    let d = tempdir();
    let compressed = d.join("stdin.bato");
    let restored = d.join("restored");
    let input = b"stdin data repeated ".repeat(100);
    let mut c = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["-c", "-o"])
        .arg(&compressed)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    c.stdin.take().unwrap().write_all(&input).unwrap();
    assert!(c.wait().unwrap().success());
    let r = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["-d", "-o"])
        .arg(&restored)
        .stdin(File::open(&compressed).unwrap())
        .output()
        .unwrap();
    assert!(r.status.success());
    assert_eq!(fs::read(&restored).unwrap(), input);
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn explicit_dash_streams_roundtrip() {
    let input = b"dash stream".repeat(100);
    let compressed = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["-c", "-", "-o", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut compressed = compressed;
    compressed.stdin.take().unwrap().write_all(&input).unwrap();
    let output = compressed.wait_with_output().unwrap();
    assert!(output.status.success());
    assert!(!output.stdout.is_empty());
    let restored = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["-d", "-", "-o", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut restored = restored;
    restored
        .stdin
        .take()
        .unwrap()
        .write_all(&output.stdout)
        .unwrap();
    let output = restored.wait_with_output().unwrap();
    assert!(output.status.success());
    assert_eq!(output.stdout, input);
}

#[test]
fn directory_defaults_to_bato_and_roundtrips() {
    let d = tempdir();
    let source = d.join("folder");
    fs::create_dir(&source).unwrap();
    write(&source.join("a.txt"), b"archive data");
    let r = Command::new(env!("CARGO_BIN_EXE_bato"))
        .arg(&source)
        .output()
        .unwrap();
    assert!(r.status.success(), "{r:?}");
    let archive = d.join("folder.bato");
    assert!(archive.exists());
    let dest = d.join("out");
    let r = Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["extract", "-o"])
        .arg(&dest)
        .arg(&archive)
        .output()
        .unwrap();
    assert!(r.status.success(), "{r:?}");
    assert_eq!(fs::read(dest.join("a.txt")).unwrap(), b"archive data");
    fs::remove_dir_all(d).unwrap();
}

#[cfg(unix)]
#[test]
fn leaf_and_parent_symlinks_are_rejected() {
    let d = tempdir();
    let src = d.join("in");
    write(&src, b"x");
    let target = d.join("target");
    write(&target, b"safe");
    let leaf = d.join("leaf");
    std::os::unix::fs::symlink(&target, &leaf).unwrap();
    assert!(!Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["-f", "-o"])
        .arg(&leaf)
        .arg(&src)
        .status()
        .unwrap()
        .success());
    assert_eq!(fs::read(&target).unwrap(), b"safe");
    let parent = d.join("parent");
    std::os::unix::fs::symlink(&d, &parent).unwrap();
    let out = parent.join("out");
    assert!(!Command::new(env!("CARGO_BIN_EXE_bato"))
        .args(["-o"])
        .arg(&out)
        .arg(&src)
        .status()
        .unwrap()
        .success());
    fs::remove_dir_all(d).unwrap();
}
