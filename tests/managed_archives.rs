use std::{fs, path::Path, process::Command};

fn run(root: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_bato"))
        .current_dir(root)
        .args(["--threads", "1"])
        .args(args)
        .output()
        .unwrap()
}

fn success(root: &Path, args: &[&str]) {
    let output = run(root, args);
    assert!(
        output.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn failure(root: &Path, args: &[&str]) {
    let output = run(root, args);
    assert!(!output.status.success(), "{args:?}: {output:?}");
}

#[test]
fn edits_are_transactional_and_remove_whole_subtrees() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("source/old/empty")).unwrap();
    fs::write(root.join("source/old/item"), b"remove this").unwrap();
    fs::write(root.join("source/keep"), b"keep this").unwrap();
    fs::create_dir_all(root.join("new/nested")).unwrap();
    fs::write(root.join("new/nested/file"), b"new data").unwrap();
    fs::write(root.join("new/nested/-f"), b"literal filename").unwrap();
    success(root, &["archive", "source"]);
    success(root, &["add", "source.bato", "new"]);
    let original = fs::read(root.join("source.bato")).unwrap();
    failure(root, &["add", "source.bato", "new"]);
    failure(root, &["remove", "source.bato", "keep", "../outside"]);
    failure(root, &["remove", "source.bato", "keep", "absent"]);
    assert_eq!(fs::read(root.join("source.bato")).unwrap(), original);
    success(root, &["remove", "source.bato", "old", "old/item"]);
    success(root, &["extract", "source.bato", "-o", "restored"]);
    assert_eq!(
        fs::read(root.join("restored/new/nested/file")).unwrap(),
        b"new data"
    );
    assert_eq!(fs::read(root.join("restored/keep")).unwrap(), b"keep this");
    assert!(!root.join("restored/old").exists());
}

#[test]
fn split_roundtrip_update_and_corruption_detection() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir(root.join("input")).unwrap();
    fs::write(root.join("input/old"), (0..=255).collect::<Vec<u8>>()).unwrap();
    fs::write(root.join("new"), b"new item").unwrap();
    success(root, &["archive", "input", "--split-size", "32"]);
    success(root, &["add", "input.bato", "new"]);
    success(root, &["-i", "input.bato"]);
    success(root, &["extract", "input.bato", "-o", "out"]);
    assert_eq!(fs::read(root.join("out/new")).unwrap(), b"new item");
    assert_eq!(
        fs::read(root.join("out/old")).unwrap(),
        (0..=255).collect::<Vec<u8>>()
    );
    let generations = fs::read_dir(root.join("input.bato"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    assert_eq!(generations.len(), 1);
    let parts = fs::read_dir(&generations[0])
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    assert!(parts.len() > 1);
    assert!(parts.iter().all(|p| p.metadata().unwrap().len() <= 32));
    let original = fs::read(&parts[0]).unwrap();
    let mut tampered = original.clone();
    tampered[0] ^= 1;
    fs::write(&parts[0], tampered).unwrap();
    failure(root, &["extract", "input.bato", "-o", "corrupt"]);
    assert!(!root.join("corrupt").exists());
    fs::remove_file(&parts[0]).unwrap();
    failure(root, &["-t", "input.bato"]);
    fs::write(&parts[0], original).unwrap();
    success(root, &["remove", "input.bato", "new"]);
    success(root, &["extract", "input.bato", "-o", "after"]);
    assert!(!root.join("after/new").exists());
}

#[test]
fn whole_password_covers_paths_and_detects_wrong_password_and_tampering() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir(root.join("input")).unwrap();
    let name = "private-project-name-unique";
    fs::write(root.join("input").join(name), b"private contents").unwrap();
    fs::write(root.join("key"), b"long passphrase for tests\n").unwrap();
    fs::write(root.join("wrong"), b"wrong passphrase").unwrap();
    success(root, &["archive", "input", "--password-file", "key"]);
    let mut archive = fs::read(root.join("input.bato")).unwrap();
    assert!(!archive
        .windows(name.len())
        .any(|window| window == name.as_bytes()));
    failure(root, &["extract", "input.bato", "-o", "no-key"]);
    failure(
        root,
        &[
            "extract",
            "input.bato",
            "-o",
            "wrong-key",
            "--password-file",
            "wrong",
        ],
    );
    assert!(!root.join("no-key").exists());
    assert!(!root.join("wrong-key").exists());
    success(
        root,
        &[
            "extract",
            "input.bato",
            "-o",
            "right-key",
            "--password-file",
            "key",
        ],
    );
    assert_eq!(
        fs::read(root.join("right-key").join(name)).unwrap(),
        b"private contents"
    );
    *archive.last_mut().unwrap() ^= 1;
    fs::write(root.join("bad.bato"), archive).unwrap();
    failure(
        root,
        &[
            "extract",
            "bad.bato",
            "-o",
            "tampered",
            "--password-file",
            "key",
        ],
    );
    assert!(!root.join("tampered").exists());
}

#[test]
fn selective_protection_survives_edits_and_works_with_split_volumes() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("input/shared/private")).unwrap();
    fs::create_dir_all(root.join("private/empty")).unwrap();
    fs::write(root.join("input/shared/public"), b"public bytes").unwrap();
    fs::write(root.join("input/shared/private/old"), b"old private bytes").unwrap();
    fs::write(root.join("input/single-secret"), b"one private file").unwrap();
    fs::write(root.join("private/new"), b"new private bytes").unwrap();
    fs::write(root.join("key"), b"long passphrase for tests").unwrap();
    success(
        root,
        &[
            "archive",
            "input",
            "--password-file",
            "key",
            "--protect",
            "shared/private",
            "--protect",
            "single-secret",
            "--split-size",
            "100",
        ],
    );
    success(
        root,
        &["extract", "input.bato", "-o", "public", "--public-only"],
    );
    assert_eq!(
        fs::read(root.join("public/shared/public")).unwrap(),
        b"public bytes"
    );
    assert!(!root.join("public/shared/private").exists());
    assert!(!root.join("public/single-secret").exists());
    failure(root, &["remove", "input.bato", "single-secret"]);
    success(
        root,
        &[
            "add",
            "input.bato",
            "private",
            "--password-file",
            "key",
            "--protect",
            "private",
        ],
    );
    success(
        root,
        &[
            "remove",
            "input.bato",
            "single-secret",
            "--password-file",
            "key",
        ],
    );
    success(
        root,
        &[
            "extract",
            "input.bato",
            "-o",
            "out",
            "--password-file",
            "key",
        ],
    );
    assert_eq!(
        fs::read(root.join("out/private/new")).unwrap(),
        b"new private bytes"
    );
    assert!(root.join("out/private/empty").is_dir());
    assert_eq!(
        fs::read(root.join("out/shared/private/old")).unwrap(),
        b"old private bytes"
    );
    assert!(!root.join("out/single-secret").exists());
    success(
        root,
        &[
            "extract",
            "input.bato",
            "-o",
            "public-after",
            "--public-only",
        ],
    );
    assert!(!root.join("public-after/private").exists());
    assert!(!root.join("public-after/shared/private").exists());
}

#[test]
fn password_file_cannot_be_archived_even_via_a_hard_link() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir(root.join("input")).unwrap();
    fs::write(root.join("key"), b"long passphrase for tests").unwrap();
    fs::hard_link(root.join("key"), root.join("input/password")).unwrap();
    failure(root, &["archive", "input", "--password-file", "key"]);
    assert!(!root.join("input.bato").exists());
}

#[test]
fn selective_protection_authenticates_the_public_section_on_full_extract() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir(root.join("input")).unwrap();
    fs::create_dir(root.join("replacement")).unwrap();
    fs::write(root.join("input/public"), b"public bytes").unwrap();
    fs::write(root.join("input/private"), b"private bytes").unwrap();
    fs::write(root.join("replacement/public"), b"replacement").unwrap();
    fs::write(root.join("key"), b"long passphrase for tests").unwrap();

    failure(
        root,
        &[
            "archive",
            "input",
            "--password-file",
            "key",
            "--protect",
            "PRIVATE",
        ],
    );
    success(
        root,
        &[
            "archive",
            "input",
            "--password-file",
            "key",
            "--protect",
            "private",
        ],
    );
    success(root, &["archive", "replacement"]);
    let original = fs::read(root.join("input.bato")).unwrap();
    let public_length = u64::from_le_bytes(original[4..12].try_into().unwrap()) as usize;
    let replacement = fs::read(root.join("replacement.bato")).unwrap();
    let mut tampered = original[..4].to_vec();
    tampered.extend_from_slice(&(replacement.len() as u64).to_le_bytes());
    tampered.extend_from_slice(&replacement);
    tampered.extend_from_slice(&original[12 + public_length..]);
    fs::write(root.join("tampered.bato"), tampered).unwrap();

    success(
        root,
        &["extract", "tampered.bato", "--public-only", "-o", "public"],
    );
    assert_eq!(
        fs::read(root.join("public/public")).unwrap(),
        b"replacement"
    );
    failure(
        root,
        &[
            "extract",
            "tampered.bato",
            "--password-file",
            "key",
            "-o",
            "full",
        ],
    );
    assert!(!root.join("full").exists());
}

#[test]
fn edits_reject_authenticated_members_outside_protection_scopes() {
    use sha2::{Digest, Sha256};
    use std::io::Write;

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir(root.join("public")).unwrap();
    fs::create_dir(root.join("private")).unwrap();
    fs::write(root.join("private/secret"), b"must stay private").unwrap();
    fs::write(root.join("add-me"), b"new bytes").unwrap();
    let password = "long passphrase for tests";
    fs::write(root.join("key"), password).unwrap();
    success(root, &["archive", "public"]);
    success(root, &["archive", "private"]);

    let public = fs::read(root.join("public.bato")).unwrap();
    let private = fs::read(root.join("private.bato")).unwrap();
    let mut envelope = b"BAP1".to_vec();
    envelope.extend_from_slice(&(public.len() as u64).to_le_bytes());
    envelope.extend_from_slice(&public);
    let mut recipient = age::scrypt::Recipient::new(password.to_owned().into());
    recipient.set_work_factor(10);
    let encryptor =
        age::Encryptor::with_recipients(std::iter::once(&recipient as &dyn age::Recipient))
            .unwrap();
    let mut writer = encryptor.wrap_output(&mut envelope).unwrap();
    writer.write_all(&Sha256::digest(&public)).unwrap();
    writer.write_all(&0u16.to_le_bytes()).unwrap();
    writer.write_all(&private).unwrap();
    writer.finish().unwrap();
    fs::write(root.join("invalid.bato"), &envelope).unwrap();

    failure(
        root,
        &["add", "invalid.bato", "add-me", "--password-file", "key"],
    );
    assert_eq!(fs::read(root.join("invalid.bato")).unwrap(), envelope);
}

#[test]
fn selective_extraction_preserves_paths_and_checks_before_publication() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("input/folder/empty")).unwrap();
    fs::create_dir_all(root.join("input/folder-other")).unwrap();
    fs::write(root.join("input/a-original"), b"duplicate data").unwrap();
    fs::write(root.join("input/folder/copy"), b"duplicate data").unwrap();
    fs::write(root.join("input/folder-other/file"), b"sibling").unwrap();
    fs::write(root.join("input/-notes"), b"notes").unwrap();
    success(root, &["archive", "input"]);

    // An unselected path can collide with a destination file.
    fs::create_dir(root.join("out")).unwrap();
    fs::write(root.join("out/folder-other"), b"keep existing").unwrap();
    success(
        root,
        &[
            "extract",
            "input.bato",
            "-o",
            "out",
            "folder",
            "folder/copy",
            "--",
            "-notes",
        ],
    );
    assert!(root.join("out/folder/empty").is_dir());
    assert_eq!(
        fs::read(root.join("out/folder/copy")).unwrap(),
        b"duplicate data"
    );
    assert_eq!(fs::read(root.join("out/-notes")).unwrap(), b"notes");
    assert!(!root.join("out/a-original").exists());
    assert_eq!(
        fs::read(root.join("out/folder-other")).unwrap(),
        b"keep existing"
    );

    let output = run(
        root,
        &[
            "extract",
            "input.bato",
            "-o",
            "missing",
            "folder",
            "folder/absent",
        ],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("folder/absent"));
    assert!(!root.join("missing").exists());
    failure(
        root,
        &[
            "extract",
            "input.bato",
            "-o",
            "unsafe",
            "folder/../a-original",
        ],
    );
    assert!(!root.join("unsafe").exists());
    failure(
        root,
        &[
            "extract",
            "input.bato",
            "-o",
            "out",
            "a-original",
            "folder/copy",
        ],
    );
    assert!(!root.join("out/a-original").exists());

    fs::write(root.join("out/folder/copy"), b"old").unwrap();
    success(
        root,
        &["extract", "input.bato", "-f", "-o", "out", "folder/copy"],
    );
    assert_eq!(
        fs::read(root.join("out/folder/copy")).unwrap(),
        b"duplicate data"
    );

    let mut damaged = fs::read(root.join("input.bato")).unwrap();
    *damaged.last_mut().unwrap() ^= 1;
    fs::write(root.join("damaged.bato"), damaged).unwrap();
    failure(
        root,
        &[
            "extract",
            "damaged.bato",
            "-o",
            "damaged-out",
            "folder/copy",
        ],
    );
    assert!(!root.join("damaged-out").exists());
}

#[test]
fn selective_extraction_respects_password_and_split_sections() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("input/shared")).unwrap();
    fs::write(root.join("input/shared/public"), b"public bytes").unwrap();
    fs::write(root.join("input/shared/private"), b"private bytes").unwrap();
    fs::write(root.join("input/other"), b"other bytes").unwrap();
    fs::write(root.join("password"), b"test password").unwrap();
    success(
        root,
        &["archive", "input", "--split-size", "64", "-o", "split.bato"],
    );
    success(
        root,
        &["extract", "split.bato", "-o", "split-out", "shared/private"],
    );
    assert_eq!(
        fs::read(root.join("split-out/shared/private")).unwrap(),
        b"private bytes"
    );
    assert!(!root.join("split-out/shared/public").exists());

    success(
        root,
        &[
            "archive",
            "input",
            "--password-file",
            "password",
            "-o",
            "whole.bato",
        ],
    );
    success(
        root,
        &[
            "extract",
            "whole.bato",
            "--password-file",
            "password",
            "-o",
            "whole-out",
            "other",
        ],
    );
    assert_eq!(
        fs::read(root.join("whole-out/other")).unwrap(),
        b"other bytes"
    );
    assert!(!root.join("whole-out/shared").exists());

    success(
        root,
        &[
            "archive",
            "input",
            "--split-size",
            "64",
            "--password-file",
            "password",
            "--protect",
            "shared/private",
            "-o",
            "protected.bato",
        ],
    );
    success(
        root,
        &[
            "extract",
            "protected.bato",
            "--public-only",
            "-o",
            "public-out",
            "shared",
        ],
    );
    assert_eq!(
        fs::read(root.join("public-out/shared/public")).unwrap(),
        b"public bytes"
    );
    assert!(!root.join("public-out/shared/private").exists());
    assert!(!root.join("public-out/other").exists());

    failure(
        root,
        &[
            "extract",
            "protected.bato",
            "--public-only",
            "-o",
            "denied",
            "shared/private",
        ],
    );
    assert!(!root.join("denied").exists());
    success(
        root,
        &[
            "extract",
            "protected.bato",
            "--password-file",
            "password",
            "-o",
            "private-out",
            "shared/private",
        ],
    );
    assert_eq!(
        fs::read(root.join("private-out/shared/private")).unwrap(),
        b"private bytes"
    );
    assert!(!root.join("private-out/shared/public").exists());
    assert!(!root.join("private-out/other").exists());
}
