use std::{
    fs::{self, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::Path,
    process::{Command, Output},
};

fn run(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_bato"))
        .current_dir(root)
        .args(["--threads", "2"])
        .args(args)
        .output()
        .unwrap()
}

fn ok(root: &Path, args: &[&str]) -> Output {
    let output = run(root, args);
    assert!(
        output.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn fails(root: &Path, args: &[&str]) {
    let output = run(root, args);
    assert!(!output.status.success(), "{args:?} unexpectedly succeeded");
}

fn fixture(size: usize) -> Vec<u8> {
    let mut state = 0x910a2dec89025cc1u64;
    (0..size)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn flip(path: &Path, offset: u64) {
    use std::io::Read;
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    let mut byte = [0];
    file.read_exact(&mut byte).unwrap();
    byte[0] ^= 255;
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&byte).unwrap();
}

#[test]
fn file_pause_resume_truncates_uncommitted_tail_and_preserves_encoded_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let source = fixture(170_001);
    fs::write(root.join("input"), &source).unwrap();
    let geometry = ["--segment-size", "65536", "--chunk-size", "16384"];
    let mut normal = vec!["input", "-o", "normal.bato"];
    normal.extend(geometry);
    ok(root, &normal);
    let mut pause = vec![
        "input",
        "-o",
        "resume.bato",
        "--checkpoint",
        "job",
        "--pause-after-segments",
        "1",
    ];
    pause.extend(geometry);
    let status = ok(root, &pause);
    assert!(String::from_utf8_lossy(&status.stderr).contains("Paused"));
    assert!(!root.join("resume.bato").exists());
    OpenOptions::new()
        .append(true)
        .open(root.join("job/plain.data"))
        .unwrap()
        .write_all(b"uncommitted")
        .unwrap();
    let mut resume = vec!["input", "-o", "resume.bato", "--resume", "job"];
    resume.extend(geometry);
    fs::write(root.join("input"), b"changed").unwrap();
    fails(root, &resume);
    fs::write(root.join("input"), &source).unwrap();
    ok(root, &resume);
    assert!(!root.join("job").exists());
    assert_eq!(
        fs::read(root.join("normal.bato")).unwrap(),
        fs::read(root.join("resume.bato")).unwrap()
    );

    fs::write(root.join("hold"), b"").unwrap();
    ok(
        root,
        &[
            "-d",
            "resume.bato",
            "-o",
            "restored",
            "--checkpoint",
            "decode-job",
            "--pause-file",
            "hold",
        ],
    );
    assert!(!root.join("restored").exists());
    assert_eq!(
        fs::metadata(root.join("decode-job/plain.data"))
            .unwrap()
            .len(),
        65_536
    );
    fs::remove_file(root.join("hold")).unwrap();
    ok(
        root,
        &[
            "-d",
            "resume.bato",
            "-o",
            "restored",
            "--resume",
            "decode-job",
        ],
    );
    assert_eq!(fs::read(root.join("restored")).unwrap(), source);
}

#[test]
fn protected_split_resume_seals_private_checkpoints_and_repairs_one_part() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("tree/private")).unwrap();
    let secret = fixture(95_000);
    fs::write(root.join("tree/private/payload"), &secret).unwrap();
    fs::write(root.join("tree/visible"), b"public").unwrap();
    fs::write(root.join("password"), b"checkpoint test passphrase").unwrap();
    let base = [
        "archive",
        "tree",
        "-o",
        "split",
        "--password-file",
        "password",
        "--protect",
        "private",
        "--split-size",
        "32000",
        "--parity",
        "--segment-size",
        "65536",
        "--chunk-size",
        "16384",
    ];
    let mut args = base.to_vec();
    args.extend(["--checkpoint", "job", "--pause-after-segments", "2"]);
    ok(root, &args);
    assert!(!root.join("split").exists());
    assert!(fs::read(root.join("job/private.state"))
        .unwrap()
        .starts_with(b"age-encryption.org/v1"));
    let stored = fs::read(root.join("job/private.data")).unwrap();
    assert!(stored[8..].starts_with(b"age-encryption.org/v1"));
    assert!(!stored.windows(64).any(|window| window == &secret[..64]));
    let mut resume = base.to_vec();
    resume.extend(["--resume", "job"]);
    fs::write(root.join("password"), b"wrong passphrase").unwrap();
    fails(root, &resume);
    fs::write(root.join("password"), b"checkpoint test passphrase").unwrap();
    ok(root, &resume);

    let generation = fs::read_dir(root.join("split"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.is_dir())
        .unwrap();
    fs::remove_file(generation.join("part-0000001")).unwrap();
    ok(root, &["repair-split", "split", "-o", "fixed"]);
    assert!(!generation.join("part-0000001").exists());
    let decode = [
        "extract",
        "fixed",
        "-o",
        "restored",
        "--password-file",
        "password",
    ];
    let mut args = decode.to_vec();
    args.extend(["--checkpoint", "decode", "--pause-after-segments", "2"]);
    ok(root, &args);
    assert!(!root.join("restored").exists());
    assert!(fs::read(root.join("decode/private.data")).unwrap()[8..]
        .starts_with(b"age-encryption.org/v1"));
    let mut resume = decode.to_vec();
    resume.extend(["--resume", "decode"]);
    ok(root, &resume);
    assert_eq!(
        fs::read(root.join("restored/private/payload")).unwrap(),
        secret
    );
    assert_eq!(fs::read(root.join("restored/visible")).unwrap(), b"public");

    let repaired = fs::read_dir(root.join("fixed"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.is_dir())
        .unwrap();
    flip(&repaired.join("part-0000002"), 20);
    ok(
        root,
        &[
            "extract",
            "fixed",
            "-o",
            "recovered",
            "--password-file",
            "password",
        ],
    );
    assert_eq!(
        fs::read(root.join("recovered/private/payload")).unwrap(),
        secret
    );
    fs::remove_file(repaired.join("part-0000001")).unwrap();
    fails(root, &["repair-split", "fixed", "-o", "too-damaged"]);
    assert!(!root.join("too-damaged").exists());
}

#[test]
fn salvage_recovers_later_members_and_checks_the_index() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir(root.join("tree")).unwrap();
    fs::write(root.join("tree/a-damaged"), fixture(92_000)).unwrap();
    fs::write(root.join("tree/z-intact"), b"recover this later member").unwrap();
    fs::write(root.join("tree/z-twin"), b"recover this later member").unwrap();
    ok(
        root,
        &[
            "archive",
            "tree",
            "--segment-size",
            "65536",
            "--chunk-size",
            "16384",
        ],
    );
    ok(root, &["recovery-index", "tree.bato", "-o", "index"]);
    flip(&root.join("tree.bato"), 10);
    fails(root, &["extract", "tree.bato", "-o", "normal"]);
    let output = ok(
        root,
        &["recover", "tree.bato", "--index", "index", "-o", "salvaged"],
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Missing 1"));
    assert!(!root.join("salvaged/a-damaged").exists());
    assert_eq!(
        fs::read(root.join("salvaged/z-intact")).unwrap(),
        b"recover this later member"
    );
    assert_eq!(
        fs::read(root.join("salvaged/z-twin")).unwrap(),
        b"recover this later member"
    );
    flip(&root.join("index"), 30);
    fails(
        root,
        &[
            "recover",
            "tree.bato",
            "--index",
            "index",
            "-o",
            "untrusted",
        ],
    );
    assert!(!root.join("untrusted").exists());
}

#[test]
fn byte_stream_salvage_reports_original_offsets_after_damage() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let source = fixture(150_000);
    fs::write(root.join("input"), &source).unwrap();
    ok(
        root,
        &["input", "--segment-size", "65536", "--chunk-size", "16384"],
    );
    ok(root, &["recovery-index", "input.bato", "-o", "index"]);
    flip(&root.join("input.bato"), 10);
    ok(
        root,
        &[
            "recover",
            "input.bato",
            "--index",
            "index",
            "-o",
            "segments",
        ],
    );
    assert!(!root.join("segments/segment-00000000-offset-0").exists());
    assert_eq!(
        fs::read(root.join("segments/segment-00000001-offset-65536")).unwrap(),
        source[65_536..131_072]
    );
    assert_eq!(
        fs::read(root.join("segments/segment-00000002-offset-131072")).unwrap(),
        source[131_072..]
    );
}

#[test]
fn selective_extraction_resumes_with_the_same_members() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir(root.join("input")).unwrap();
    fs::write(root.join("input/first"), fixture(75_000)).unwrap();
    fs::write(root.join("input/second"), b"selected").unwrap();
    for (archive, extra) in [
        ("plain.bato", &[][..]),
        ("split.bato", &["--split-size", "32000"][..]),
    ] {
        let mut args = vec![
            "archive",
            "input",
            "-o",
            archive,
            "--segment-size",
            "65536",
            "--chunk-size",
            "16384",
        ];
        args.extend(extra);
        ok(root, &args);
        let dst = format!("{archive}-out");
        let paused = ok(
            root,
            &[
                "extract",
                archive,
                "-o",
                &dst,
                "--checkpoint",
                "job",
                "--pause-after-segments",
                "1",
                "second",
            ],
        );
        assert!(String::from_utf8_lossy(&paused.stderr).contains("Paused"));
        assert!(!root.join(&dst).exists());
        fails(
            root,
            &["extract", archive, "-o", &dst, "--resume", "job", "first"],
        );
        assert!(!root.join(&dst).exists());
        ok(
            root,
            &["extract", archive, "-o", &dst, "--resume", "job", "second"],
        );
        assert_eq!(
            fs::read(root.join(&dst).join("second")).unwrap(),
            b"selected"
        );
        assert!(!root.join(&dst).join("first").exists());
        assert!(!root.join("job").exists());
    }
}
