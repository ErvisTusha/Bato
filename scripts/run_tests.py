#!/usr/bin/env python3
"""Lossless Bato CLI round trips, archive features, recovery, and failure handling."""
import argparse
import gzip
import filecmp
import math
import hashlib
import io
import json
import os
import random
import shutil
import signal
import sqlite3
import sys
import struct
import subprocess
import tempfile
import time
import unittest
import wave
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bato_tools.codecs import resolve_bato
from bato_tools.common import PROJECT_ROOT, sha256, snapshot
from bato_tools.test_selection import DEFAULT_COPIED_CORPUS, select_samples
from bato_tools.test_campaign import ensure_test_baselines

BATO_BIN = None
CLI_TIMEOUT = 30


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bato-engine-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_cli(self, *args, data=None, discard=False):
        return subprocess.run([str(BATO_BIN), *map(str, args)], input=data,
                              stdout=subprocess.DEVNULL if discard else subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=CLI_TIMEOUT)

    def compressed(self, data):
        source, archive = self.root / "source", self.root / "stream.bato"
        source.write_bytes(data)
        result = self.run_cli("-f", source, "-o", archive)
        self.assertEqual(result.returncode, 0, result.stderr)
        return archive

    def roundtrip(self, data):
        archive = self.compressed(data)
        result = self.run_cli("-t", archive)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / "restored"
        result = self.run_cli("-d", "-f", archive, "-o", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.read_bytes(), data)

    def test_roundtrip(self):
        cases = [
            b"", b"Z", b"X" * 100000,
            bytes(range(256)) * 40,
            struct.pack("<10000f", *[float(n) * 0.125 for n in range(10000)]),
            struct.pack("<5000q", *[n * 1000 for n in range(5000)]),
            random.Random(42).randbytes(64 * 1024),
            b"Block chunking across 128KB boundaries in streaming mode. " * 6000,
        ]
        for data in cases:
            with self.subTest(size=len(data)):
                self.roundtrip(data)

    def test_exact_block_and_window_boundaries(self):
        for size in [128 * 1024 - 1, 128 * 1024, 128 * 1024 + 1,
                     256 * 1024 - 1, 256 * 1024, 256 * 1024 + 1]:
            data = (bytes(range(251)) * (size // 251 + 1))[:size]
            with self.subTest(size=size):
                self.roundtrip(data)

    def test_malformed_and_corrupt_streams(self):
        bad = self.root / "bad.bato"
        cases = [b"NOT_BATO_HEADER_1234567890"]
        original = self.compressed(b"Critical transaction integrity record." * 100).read_bytes()
        corrupt = bytearray(original)
        corrupt[16] ^= 0x55
        cases.extend([bytes(corrupt), original[:len(original) // 2]])
        checksum = bytearray(original)
        checksum[-24] ^= 1
        cases.append(bytes(checksum))
        for number, content in enumerate(cases):
            with self.subTest(case=number):
                bad.write_bytes(content)
                result = self.run_cli("-d", "-f", bad, "-o", self.root / "rejected")
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "rejected").exists())

    def test_nested_archive_paths_and_bytes(self):
        source = self.root / "project"
        for name, data in [("nested/data.txt", b"archive data" * 100), (".git/HEAD", b"ref: refs/heads/main\n"),
                           ("empty.dat", b""), ("unicode/Tiranë.txt", "日本語".encode())]:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        archive = self.root / "project.bato"
        result = self.run_cli("archive", source, "-o", archive)
        self.assertEqual(result.returncode, 0, result.stderr)
        restored = self.root / "extracted"
        result = self.run_cli("extract", archive, "-o", restored)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(snapshot(source), snapshot(restored))

    def test_removed_output_limit_option(self):
        archive = self.compressed(b"test old CLI option")
        result = self.rejected("-t", archive, "--max-output-size", 10)
        self.assertIn(b"unrecognized option", result.stderr)

    def test_streaming_roundtrip(self):
        data = b"streaming fixture " * 5000
        encoded = self.run_cli("-", data=data)
        self.assertEqual(encoded.returncode, 0, encoded.stderr)
        decoded = self.run_cli("-d", data=encoded.stdout)
        self.assertEqual(decoded.returncode, 0, decoded.stderr)
        self.assertEqual(decoded.stdout, data)

    def ok(self, *args):
        result = self.run_cli("--threads", 2, *args)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        return result

    def rejected(self, *args):
        result = self.run_cli("--threads", 2, *args)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def feature_tree(self):
        source = self.root / "tree"
        (source / "nested" / "empty").mkdir(parents=True)
        payload = random.Random(123).randbytes(130_000)
        (source / "nested" / "binary").write_bytes(payload)
        (source / "duplicate").write_bytes(payload)
        (source / ".hidden").write_bytes(b"public")
        (source / "zero").touch()
        return source

    def test_archive_edits_preserve_exact_tree(self):
        source = self.feature_tree()
        archive, restored = self.root / "tree.bato", self.root / "restored"
        self.ok("archive", source, "-o", archive)
        self.ok("-i", archive)
        before = archive.read_bytes()
        self.rejected("remove", archive, "../outside")
        self.assertEqual(archive.read_bytes(), before)
        added = self.root / "added"
        (added / "empty").mkdir(parents=True)
        (added / "extra").write_bytes(b"new member\x00")
        self.ok("add", archive, added)
        before = archive.read_bytes()
        self.rejected("add", archive, added)
        self.assertEqual(archive.read_bytes(), before)
        (added / "extra").write_bytes(b"replacement")
        self.ok("add", archive, added, "-f")
        self.ok("remove", archive, "nested", "zero")
        shutil.rmtree(source / "nested")
        (source / "zero").unlink()
        shutil.copytree(added, source / "added")
        self.ok("-t", archive)
        self.ok("extract", archive, "-o", restored)
        self.assertEqual(snapshot(restored), snapshot(source))

    def test_password_protection_and_public_extraction(self):
        source = self.feature_tree()
        password, wrong = self.root / "password", self.root / "wrong"
        password.write_text("correct password for integration test", encoding="utf-8")
        wrong.write_text("wrong password", encoding="utf-8")
        for protected in ([], ["--protect", "nested", "--protect", "duplicate"]):
            with self.subTest(selective=bool(protected)):
                archive = self.root / ("selective" if protected else "encrypted")
                restored = archive.with_suffix(".out")
                self.ok("archive", source, "-o", archive, "--password-file", password, *protected)
                self.rejected("extract", archive, "-o", restored, "--password-file", wrong)
                self.assertFalse(restored.exists())
                self.rejected("extract", archive, "-o", restored)
                self.assertFalse(restored.exists())
                self.ok("-t", archive, "--password-file", password)
                self.ok("extract", archive, "-o", restored, "--password-file", password)
                self.assertEqual(snapshot(restored), snapshot(source))
                public = archive.with_suffix(".public")
                if protected:
                    self.ok("extract", archive, "-o", public, "--public-only")
                    self.assertEqual(sorted(p.name for p in public.iterdir()), [".hidden", "zero"])
                    self.assertEqual((public / ".hidden").read_bytes(), b"public")
                    self.assertEqual((public / "zero").read_bytes(), b"")
                else:
                    self.rejected("extract", archive, "-o", public, "--public-only")
                    self.assertFalse(public.exists())

    def test_selective_extraction_preserves_exact_files_and_directories(self):
        source = self.feature_tree()
        (source / "nested-extra").mkdir()
        (source / "nested-extra" / "excluded").touch()
        (source / "Tiranë notes.txt").write_bytes(b"notes")
        (source / "-literal").touch()
        archive, restored = self.root / "tree.bato", self.root / "restored"
        self.ok("archive", source, "-o", archive)
        self.ok("extract", archive, "-o", restored, "nested", "nested/binary",
                "Tiranë notes.txt", "--", "-literal")
        self.assertEqual(sorted(p.relative_to(restored).as_posix() for p in restored.rglob("*")),
                         ["-literal", "Tiranë notes.txt", "nested", "nested/binary", "nested/empty"])
        self.assertEqual(snapshot(restored / "nested"), snapshot(source / "nested"))
        self.assertEqual((restored / "Tiranë notes.txt").read_bytes(), b"notes")
        self.assertEqual((restored / "-literal").read_bytes(), b"")

        streamed = self.root / "streamed"
        result = self.run_cli("extract", "-", "-o", streamed, "nested/binary",
                              data=archive.read_bytes())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list((streamed / "nested").iterdir()), [streamed / "nested/binary"])
        self.assertEqual(list(streamed.iterdir()), [streamed / "nested"])
        self.assertEqual(snapshot(streamed / "nested/binary"), snapshot(source / "duplicate"))

    def test_selective_extraction_rejection_preserves_destination(self):
        source = self.feature_tree()
        archive, restored = self.root / "tree.bato", self.root / "restored"
        self.ok("archive", source, "-o", archive)
        restored.mkdir()
        (restored / "zero").write_bytes(b"existing destination bytes")
        before = snapshot(restored)

        for member in ("absent", "nested/missing", "NESTED", "../outside", "zero/child"):
            with self.subTest(member=member):
                self.rejected("extract", archive, "-f", "-o", restored, "zero", member)
                self.assertEqual(snapshot(restored), before)

        self.rejected("extract", archive, "-o", restored, ".hidden", "zero")
        self.assertEqual(snapshot(restored), before)
        damaged = self.root / "damaged.bato"
        raw = archive.read_bytes()
        damaged.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        self.rejected("extract", damaged, "-f", "-o", restored, "zero")
        self.assertEqual(snapshot(restored), before)

        self.ok("extract", archive, "-f", "-o", restored, "zero")
        self.assertEqual((restored / "zero").read_bytes(), b"")
        self.assertEqual(list(restored.iterdir()), [restored / "zero"])

    def test_selective_extraction_from_encrypted_split_archive(self):
        source = self.feature_tree()
        password, archive = self.root / "password", self.root / "volumes"
        password.write_text("selective extraction passphrase", encoding="utf-8")
        self.ok("archive", source, "-o", archive, "--split-size", 32000,
                "--password-file", password, "--protect", "nested")
        self.assertGreater(len(list(archive.glob("volumes-*/part-*"))), 1)

        public = self.root / "public"
        self.ok("extract", archive, "--public-only", "-o", public, ".hidden")
        self.assertEqual(list(public.iterdir()), [public / ".hidden"])
        self.assertEqual((public / ".hidden").read_bytes(), b"public")

        private = self.root / "private"
        self.rejected("extract", archive, "--public-only", "-o", private, "nested/binary")
        self.assertFalse(private.exists())
        self.ok("extract", archive, "--password-file", password, "-o", private, "nested")
        self.assertEqual(list(private.iterdir()), [private / "nested"])
        self.assertEqual(snapshot(private / "nested"), snapshot(source / "nested"))

    def test_selective_extraction_checkpoint_binds_selection(self):
        source = self.feature_tree()
        archive, restored, job = (self.root / name for name in ("tree.bato", "restored", "job"))
        self.ok("archive", source, "-o", archive,
                "--segment-size", 65536, "--chunk-size", 16384)
        extraction = ("extract", archive, "-o", restored)
        paused = self.ok(*extraction, "--checkpoint", job, "--pause-after-segments", 1,
                         "nested/binary")
        self.assertIn(b"Paused", paused.stderr)
        self.assertFalse(restored.exists())

        self.rejected(*extraction, "--resume", job, "duplicate")
        self.assertFalse(restored.exists())
        self.ok(*extraction, "--resume", job, "nested/binary")
        self.assertEqual(list(restored.iterdir()), [restored / "nested"])
        self.assertEqual(list((restored / "nested").iterdir()), [restored / "nested/binary"])
        self.assertEqual(snapshot(restored / "nested/binary"), snapshot(source / "nested/binary"))
        self.assertFalse(job.exists())

    def test_split_volume_damage_and_parity_repair(self):
        source = self.feature_tree()
        for parity in (False, True):
            with self.subTest(parity=parity):
                archive = self.root / ("parity" if parity else "plain")
                self.ok("archive", source, "-o", archive, "--split-size", 32000,
                        *(["--parity"] if parity else []))
                parts = sorted(archive.glob("volumes-*/part-*"))
                self.assertGreater(len(parts), 2)
                self.assertTrue(all(0 < part.stat().st_size <= 32000 for part in parts))
                intact = archive.with_suffix(".intact")
                self.ok("extract", archive, "-o", intact)
                self.assertEqual(snapshot(intact), snapshot(source))
                parts[0].unlink()
                repaired = archive.with_suffix(".repaired")
                if parity:
                    self.ok("repair-split", archive, "-o", repaired)
                    restored = archive.with_suffix(".out")
                    self.ok("extract", repaired, "-o", restored)
                    self.assertEqual(snapshot(restored), snapshot(source))
                    self.assertFalse(parts[0].exists())
                    parts[1].unlink()
                self.rejected("repair-split", archive, "-o", archive.with_suffix(".failed"))
                self.assertFalse(archive.with_suffix(".failed").exists())

    def test_checkpoint_pause_resume_and_changed_input(self):
        source = self.root / "source"
        data = random.Random(321).randbytes(145_000)
        source.write_bytes(data)
        archive, ordinary = self.root / "archive", self.root / "ordinary"
        job, extract_job = self.root / "encode-job", self.root / "decode-job"
        geometry = ["--segment-size", 65536, "--chunk-size", 16384]
        self.ok(source, "-o", ordinary, *geometry)
        self.ok(source, "-o", archive, "--checkpoint", job,
                "--pause-after-segments", 1, *geometry)
        self.assertFalse(archive.exists())
        self.assertTrue((job / "plain.state").is_file())
        source.write_bytes(b"changed")
        self.rejected(source, "-o", archive, "--resume", job, *geometry)
        self.assertFalse(archive.exists())
        source.write_bytes(data)
        self.ok(source, "-o", archive, "--resume", job, *geometry)
        self.assertEqual(archive.read_bytes(), ordinary.read_bytes())
        self.assertFalse(job.exists())
        marker, restored = self.root / "hold", self.root / "restored"
        marker.touch()
        self.ok("-d", archive, "-o", restored, "--checkpoint", extract_job,
                "--pause-file", marker)
        self.assertFalse(restored.exists())
        self.assertTrue((extract_job / "plain.state").is_file())
        marker.unlink()
        self.ok("-d", archive, "-o", restored, "--resume", extract_job)
        self.assertEqual(restored.read_bytes(), data)
        self.assertFalse(extract_job.exists())

    def test_protected_split_checkpoint_roundtrip(self):
        source = self.feature_tree()
        password = self.root / "password"
        password.write_text("checkpoint passphrase", encoding="utf-8")
        archive, restored = self.root / "volumes", self.root / "restored"
        encode_job, decode_job = self.root / "encoding", self.root / "decoding"
        creation = ["archive", source, "-o", archive, "--password-file", password,
                    "--split-size", 32000, "--parity",
                    "--segment-size", 65536, "--chunk-size", 16384]
        self.ok(*creation, "--checkpoint", encode_job, "--pause-after-segments", 1)
        self.assertFalse(archive.exists())
        self.assertTrue((encode_job / "private.state").read_bytes().startswith(
            b"age-encryption.org/v1"))
        self.ok(*creation, "--resume", encode_job)
        extraction = ["extract", archive, "-o", restored, "--password-file", password]
        self.ok(*extraction, "--checkpoint", decode_job, "--pause-after-segments", 1)
        self.assertFalse(restored.exists())
        self.assertTrue((decode_job / "private.state").read_bytes().startswith(
            b"age-encryption.org/v1"))
        self.ok(*extraction, "--resume", decode_job)
        self.assertEqual(snapshot(restored), snapshot(source))
        self.assertFalse(encode_job.exists())
        self.assertFalse(decode_job.exists())

    def test_salvage_verified_segments_after_corruption(self):
        source, archive = self.root / "source", self.root / "archive"
        data = random.Random(78).randbytes(150_000)
        source.write_bytes(data)
        self.ok(source, "-o", archive, "--segment-size", 65536, "--chunk-size", 16384)
        index, output = self.root / "index", self.root / "salvaged"
        self.ok("recovery-index", archive, "-o", index)
        damaged = bytearray(archive.read_bytes())
        damaged[10] ^= 255
        archive.write_bytes(damaged)
        self.rejected("-t", archive)
        self.ok("recover", archive, "--index", index, "-o", output)
        self.assertEqual(
            {p.name: p.read_bytes() for p in output.iterdir()},
            {"segment-00000001-offset-65536": data[65536:131072],
             "segment-00000002-offset-131072": data[131072:]},
        )
        damaged_index = bytearray(index.read_bytes())
        damaged_index[-1] ^= 1
        index.write_bytes(damaged_index)
        failed = self.root / "untrusted"
        self.rejected("recover", archive, "--index", index, "-o", failed)
        self.assertFalse(failed.exists())


    def test_mutation_matrix_preserves_existing_destination(self):
        source, archive, output = (self.root / name for name in ("input", "archive", "output"))
        original = random.Random(991).randbytes(135_000)
        source.write_bytes(original)
        self.ok(source, "-o", archive, "--segment-size", 65536, "--chunk-size", 16384)
        encoded = archive.read_bytes()
        mutations = {f"truncate-{n}": encoded[:n] for n in
                     (0, 3, 9, 10, 64, len(encoded) // 2, len(encoded) - 1)}
        mutations["trailing-byte"] = encoded + b"\x00"
        for offset in (0, 4, 6, 7, 10, len(encoded) // 2, len(encoded) - 1):
            changed = bytearray(encoded)
            changed[offset] ^= 0xff
            mutations[f"flip-{offset}"] = changed
        for name, damaged in mutations.items():
            with self.subTest(mutation=name):
                archive.write_bytes(damaged)
                output.write_bytes(b"existing data")
                self.rejected("-d", archive, "-o", output, "-f")
                self.assertEqual(output.read_bytes(), b"existing data")
                self.rejected("-t", archive)

    def test_duplicate_expansion_preserves_all_copies(self):
        source, archive, output = (self.root / name for name in ("tree", "archive", "output"))
        source.mkdir()
        payload = random.Random(32).randbytes(4096)
        for index in range(96):
            (source / f"file-{index:03}").write_bytes(payload)
        self.ok("archive", source, "-o", archive)
        self.ok("extract", archive, "-o", output)
        self.assertEqual(snapshot(output), snapshot(source))

    def test_extraction_does_not_follow_destination_symlink(self):
        source = self.feature_tree()
        archive, outside, destination = (self.root / name for name in ("archive", "outside", "output"))
        outside.mkdir()
        (outside / "binary").write_bytes(b"outside data")
        destination.mkdir()
        try:
            (destination / "nested").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"directory symlinks unavailable: {error}")
        self.ok("archive", source, "-o", archive)
        before = snapshot(outside)
        self.rejected("extract", archive, "-o", destination, "-f")
        self.assertEqual(snapshot(outside), before)
        self.assertEqual([p.name for p in destination.iterdir()], ["nested"])

    def test_authenticated_archive_rejects_tampered_ciphertext(self):
        source = self.feature_tree()
        password, archive = self.root / "password", self.root / "archive"
        password.write_text("authenticated archive test passphrase", encoding="utf-8")
        self.ok("archive", source, "-o", archive, "--password-file", password)
        encoded = archive.read_bytes()
        changed = bytearray(encoded)
        changed[-1] ^= 0x80
        for name, damaged in (("tampered", changed), ("truncated", encoded[:-1]),
                              ("appended", encoded + b"\x00")):
            with self.subTest(damage=name):
                output = self.root / name
                archive.write_bytes(damaged)
                self.rejected("extract", archive, "-o", output, "--password-file", password)
                self.assertFalse(output.exists())

    def test_checkpoint_rejects_corruption_in_committed_data(self):
        source, archive, job = (self.root / name for name in ("input", "archive", "job"))
        source.write_bytes(random.Random(553).randbytes(135_000))
        args = [source, "-o", archive, "--segment-size", 65536, "--chunk-size", 16384]
        self.ok(*args, "--checkpoint", job, "--pause-after-segments", 1)
        staged = job / "plain.data"
        original = staged.read_bytes()
        altered = bytearray(original)
        altered[len(altered) // 2] ^= 1
        staged.write_bytes(altered)
        self.rejected(*args, "--resume", job)
        self.assertFalse(archive.exists())
        self.assertEqual(staged.read_bytes(), altered)
        staged.write_bytes(original + b"uncommitted tail")
        self.ok(*args, "--resume", job)
        output = self.root / "restored"
        self.ok("-d", archive, "-o", output)
        self.assertEqual(output.read_bytes(), source.read_bytes())


    @unittest.skipUnless(os.name == "posix", "requires POSIX process and file limits")
    def test_write_failure_preserves_existing_output(self):
        import resource

        source, archive, output = (self.root / name for name in ("source", "archive", "output"))
        source.write_bytes(random.Random(104).randbytes(64 * 1024))
        self.ok(source, "-o", archive)

        def restrict_file_size():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))

        for args in ([source, "-o", output], ["-d", archive, "-o", output]):
            with self.subTest(operation=str(args[0])):
                output.write_bytes(b"preserve the original destination")
                result = subprocess.run(
                    [str(BATO_BIN), "--threads", "1", "-f", *map(str, args)],
                    preexec_fn=restrict_file_size, capture_output=True, timeout=CLI_TIMEOUT,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(output.read_bytes(), b"preserve the original destination")
                self.assertEqual(set(self.root.iterdir()), {source, archive, output})

    def test_killed_compression_resumes_from_disk(self):
        source, archive, job = (self.root / name for name in ("source", "archive", "job"))
        payload = random.Random(421).randbytes(2 * 1024 * 1024)
        source.write_bytes(payload)
        args = [source, "-o", archive, "--segment-size", 65536, "--chunk-size", 16384]
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                [str(BATO_BIN), "--threads", "1", *map(str, args), "--checkpoint", str(job)],
                stdout=subprocess.DEVNULL, stderr=errors,
            )
            try:
                deadline = time.monotonic() + CLI_TIMEOUT
                while not (job / "plain.state").exists() and process.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("compression did not commit a checkpoint before timeout")
                    time.sleep(0.005)
                errors.seek(0)
                self.assertTrue((job / "plain.state").is_file(), errors.read())
                self.assertIsNone(process.poll(), "compression exited before fault injection")
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=CLI_TIMEOUT)
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(archive.exists())
        self.ok(*args, "--resume", job)
        restored = self.root / "restored"
        self.ok("-d", archive, "-o", restored)
        self.assertEqual(restored.read_bytes(), payload)
        self.assertFalse(job.exists())

    def test_seeded_mutations_cannot_publish_changed_bytes(self):
        payload = random.Random(804).randbytes(4096)
        archive = self.compressed(payload)
        encoded = archive.read_bytes()
        rng = random.Random(71239)
        output = self.root / "restored"
        for index in range(32):
            with self.subTest(mutation=index):
                mutated = bytearray(encoded)
                position = rng.randrange(len(mutated))
                if index % 3 == 0:
                    mutated.insert(position, rng.randrange(256))
                elif index % 3 == 1:
                    del mutated[position]
                else:
                    mutated[position] ^= 1 << rng.randrange(8)
                archive.write_bytes(mutated)
                output.write_bytes(b"untouched")
                result = self.run_cli("--threads", 1, "-d", archive, "-o", output, "-f")
                if result.returncode == 0:
                    self.assertEqual(output.read_bytes(), payload)
                else:
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), b"untouched")

    def test_parity_checks_equal_length_damage_and_corrupt_redundancy(self):
        source = self.feature_tree()
        archive = self.root / "split"
        self.ok("archive", source, "-o", archive, "--split-size", 32000, "--parity")
        generation, = archive.glob("volumes-*")
        part = sorted(generation.glob("part-*"))[0]
        parity = generation / "parity"

        def corrupt(path):
            data = bytearray(path.read_bytes())
            data[len(data) // 2] ^= 0x80
            path.write_bytes(data)

        corrupt(part)
        restored = self.root / "restored"
        self.ok("extract", archive, "-o", restored)
        self.assertEqual(snapshot(restored), snapshot(source))
        corrupt(parity)
        refused = self.root / "refused"
        self.rejected("extract", archive, "-o", refused)
        self.assertFalse(refused.exists())


    def test_distinct_identical_and_damaged_bato_archives_as_members(self):
        members = self.root / "members"
        members.mkdir()
        first = self.root / "first.txt"
        first.write_bytes(b"nested text\n" * 64)
        second = self.root / "second-tree"
        (second / "empty").mkdir(parents=True)
        (second / "binary").write_bytes(random.Random(44).randbytes(8192))
        self.ok(first, "-o", members / "first.bato")
        self.ok("archive", second, "-o", members / "second.bato")
        shutil.copyfile(members / "first.bato", members / "copy.bato")
        damaged = bytearray((members / "second.bato").read_bytes())
        damaged[-1] ^= 0x80
        (members / "damaged.bato").write_bytes(damaged)
        restored = self.roundtrip_archive_members(members)
        self.ok("-d", restored / "first.bato", "-o", self.root / "first.out")
        self.ok("-d", restored / "copy.bato", "-o", self.root / "copy.out")
        self.ok("extract", restored / "second.bato", "-o", self.root / "second.out")
        self.assertEqual((self.root / "first.out").read_bytes(), first.read_bytes())
        self.assertEqual((self.root / "copy.out").read_bytes(), first.read_bytes())
        self.assertEqual(snapshot(self.root / "second.out"), snapshot(second))
        self.rejected("-t", restored / "damaged.bato")

    def roundtrip_archive_members(self, members):
        outer = self.root / "outer.bato"
        restored = self.root / "restored"
        self.ok("archive", members, "-o", outer)
        self.ok("-t", outer)
        self.ok("extract", outer, "-o", restored)
        self.assertEqual(snapshot(restored), snapshot(members))
        return restored

    def test_zip_gzip_and_bato_archives_in_one_folder(self):
        members = self.root / "members"
        members.mkdir()
        payload = b"one payload, multiple archive formats\n" * 128
        source = self.root / "payload"
        source.write_bytes(payload)
        self.ok(source, "-o", members / "payload.bato")
        (members / "payload.gz").write_bytes(gzip.compress(payload, mtime=0))
        with zipfile.ZipFile(members / "payload.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("payload", payload)
        restored = self.roundtrip_archive_members(members)
        self.assertEqual(gzip.decompress((restored / "payload.gz").read_bytes()), payload)
        with zipfile.ZipFile(restored / "payload.zip") as archive:
            self.assertEqual(archive.namelist(), ["payload"])
            self.assertEqual(archive.read("payload"), payload)
        self.ok("-d", restored / "payload.bato", "-o", self.root / "payload.out")
        self.assertEqual((self.root / "payload.out").read_bytes(), payload)

    def check_frozen_archive_member(self, name, digest):
        fixture = Path(__file__).resolve().parent / "fixtures" / "nested" / name
        self.assertEqual(sha256(fixture), digest)
        members = self.root / "members"
        members.mkdir()
        shutil.copyfile(fixture, members / name)
        source = self.root / "payload"
        payload = b"Bato nested archive fixture v1\n"
        source.write_bytes(payload)
        self.ok(source, "-o", members / "payload.bato")
        restored = self.roundtrip_archive_members(members)
        self.assertEqual((restored / name).read_bytes(), fixture.read_bytes())
        decoded = self.root / "bato.out"
        self.ok("-d", restored / "payload.bato", "-o", decoded)
        self.assertEqual(decoded.read_bytes(), payload)

    def test_rar_and_bato_archives_in_one_folder(self):
        self.check_frozen_archive_member(
            "payload.rar", "f9efae3098f94b5c5cfc585e9cbdabb6ba17a9de39ea0523fdfd14b00c3df2c6"
        )

    def test_7zip_and_bato_archives_in_one_folder(self):
        self.check_frozen_archive_member(
            "payload.7z", "b60f5d7ed3ee6faee5f5a45da826a95d9922688bb5bbda10a6c344e47085b1d2"
        )

    def test_identical_archive_bundle_deduplicates_against_distinct_bundle(self):
        rng = random.Random(570)
        source = self.root / "payload"
        identical = self.root / "identical"
        distinct = self.root / "distinct"
        identical.mkdir()
        distinct.mkdir()
        payloads = {}
        for index in range(100):
            name = f"{index:03}.bato"
            payloads[name] = rng.randbytes(2048)
            source.write_bytes(payloads[name])
            self.ok(source, "-o", distinct / name)
        first = distinct / "000.bato"
        for name in payloads:
            shutil.copyfile(first, identical / name)

        sizes = {}
        for folder in (identical, distinct):
            with self.subTest(bundle=folder.name):
                outer = self.root / f"{folder.name}.bato"
                restored = self.root / f"{folder.name}.restored"
                self.ok("archive", folder, "-o", outer)
                self.ok("-t", outer)
                self.ok("extract", outer, "-o", restored)
                self.assertEqual(snapshot(restored), snapshot(folder))
                sizes[folder.name] = outer.stat().st_size
                decoded = self.root / "decoded"
                for name, payload in payloads.items():
                    self.ok("-d", restored / name, "-o", decoded, "-f")
                    self.assertEqual(decoded.read_bytes(),
                                     payloads["000.bato"] if folder == identical else payload)
        print(f"\nNested bundle sizes: 100 identical archives {sizes['identical']:,} bytes, "
              f"100 distinct archives {sizes['distinct']:,} bytes", flush=True)
        self.assertLess(sizes["identical"], sizes["distinct"])

    def test_password_protected_inner_archive_still_authenticates(self):
        source = self.feature_tree()
        members = self.root / "members"
        members.mkdir()
        password, wrong = self.root / "password", self.root / "wrong"
        password.write_text("nested archive passphrase", encoding="utf-8")
        wrong.write_text("wrong nested archive passphrase", encoding="utf-8")
        self.ok("archive", source, "-o", members / "encrypted.bato",
                "--password-file", password)
        restored = self.roundtrip_archive_members(members)
        inner = restored / "encrypted.bato"
        output = self.root / "decoded"
        self.rejected("extract", inner, "-o", output, "--password-file", wrong)
        self.assertFalse(output.exists())
        self.ok("extract", inner, "-o", output, "--password-file", password)
        self.assertEqual(snapshot(output), snapshot(source))

    def test_nested_split_archive_preserves_volume_set_and_payload(self):
        source = self.feature_tree()
        members = self.root / "members"
        members.mkdir()
        split = members / "split.bato"
        self.ok("archive", source, "-o", split, "--split-size", 32000, "--parity")
        self.assertTrue((split / "index.bato").is_file())
        self.assertGreater(len(list(split.glob("volumes-*/part-*"))), 1)
        self.assertEqual(len(list(split.glob("volumes-*/parity"))), 1)
        restored = self.roundtrip_archive_members(members)
        output = self.root / "decoded"
        self.ok("-t", restored / split.name)
        self.ok("extract", restored / split.name, "-o", output)
        self.assertEqual(snapshot(output), snapshot(source))

    def file_type_fixtures(self):
        def png_chunk(kind, data):
            body = kind + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

        png = (b"\x89PNG\r\n\x1a\n"
               + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
               + png_chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x80"))
               + png_chunk(b"IEND", b""))
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            wav.writeframes(struct.pack("<256h", *range(256)))
        zipped = io.BytesIO()
        with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("nested/data.txt", b"archive inside an archive\n" * 32)
        packed = gzip.compress(b"message=hello\n" * 64, mtime=0)
        return {
            "text": {"utf8.txt": "Tiranë 日本語\r\n".encode(),
                     "empty.txt": b"", "invalid-utf8.txt": b"\xc3\x28\xff\x00"},
            "structured": {"valid.json": b'{"items":[1,2],"ready":true}',
                           "broken.json": b'{"items":[1,',
                           "records.csv": b"id,amount\r\n1,42\r\n2,300\r\n"},
            "images": {"valid.png": png, "truncated.png": png[:-15]},
            "audio": {"valid.wav": audio.getvalue(), "truncated.wav": audio.getvalue()[:25]},
            "compressed": {"valid.zip": zipped.getvalue(), "broken.zip": zipped.getvalue()[:-22],
                           "valid.gz": packed, "broken.gz": packed[:-8]},
            "unknown": {"no-extension": bytes(range(256)) * 3,
                        "opaque.unknown": random.Random(318).randbytes(8192),
                        "looks-like-an-image.png": b"unrecognized format\x00\xff",
                        "looks-like-an-archive": b"BATO\x00\xffbroken header"},
        }

    def test_file_type_matrix_preserves_valid_and_damaged_source_bytes(self):
        for group, fixtures in self.file_type_fixtures().items():
            for name, data in fixtures.items():
                with self.subTest(group=group, name=name):
                    source = self.root / name
                    archive = self.root / "stream.bato"
                    output = self.root / "restored"
                    source.write_bytes(data)
                    self.ok(source, "-o", archive, "-f")
                    self.ok("-t", archive)
                    self.ok("-d", archive, "-o", output, "-f")
                    self.assertEqual(output.read_bytes(), data)

    def test_same_type_and_mixed_folders_with_damaged_members(self):
        for group, fixtures in self.file_type_fixtures().items():
            folder = self.root / "fixtures" / group
            (folder / "nested" / "empty").mkdir(parents=True)
            for index, (name, data) in enumerate(fixtures.items()):
                path = (folder / "nested" if index % 2 else folder) / name
                path.write_bytes(data)
            (folder / ".hidden").write_bytes(next(iter(fixtures.values())))
        root = self.root / "fixtures"
        for source in [*sorted(root.iterdir()), root]:
            with self.subTest(folder=source.name):
                archive = self.root / f"{source.name}.bato"
                restored = self.root / f"{source.name}.restored"
                self.ok("archive", source, "-o", archive)
                self.ok("-t", archive)
                self.ok("extract", archive, "-o", restored)
                self.assertEqual(snapshot(restored), snapshot(source))
                # Damage Bato's stream checksum, independently of any damaged source members.
                damaged = bytearray(archive.read_bytes())
                damaged[-1] ^= 0x80
                archive.write_bytes(damaged)
                self.rejected("-t", archive)
                refused = self.root / f"{source.name}.refused"
                self.rejected("extract", archive, "-o", refused)
                self.assertFalse(refused.exists())
                self.rejected("extract", archive, "-o", restored, "-f")
                self.assertEqual(snapshot(restored), snapshot(source))


    def test_indexed_extract_seeks_to_duplicate_data_and_checks_binding(self):
        source = self.root / "source"
        source.mkdir()
        rng = random.Random(913)
        payload = rng.randbytes(4000)
        (source / "a-original").write_bytes(payload)
        (source / "m-unselected").write_bytes(rng.randbytes(200_000))
        (source / "z-duplicate").write_bytes(payload)
        archive, index, output = (self.root / name for name in ("archive", "index", "output"))
        self.ok("archive", source, "-o", archive, "--segment-size", 65536, "--chunk-size", 16384)
        self.ok("recovery-index", archive, "-o", index)
        result = self.ok("extract", archive, "z-duplicate", "--index", index, "-o", output, "--stats")
        self.assertIn(b"decoded 1 segments", result.stderr)
        self.assertEqual([p.name for p in output.iterdir()], ["z-duplicate"])
        self.assertEqual((output / "z-duplicate").read_bytes(), payload)
        for offset in (0, 10, len(archive.read_bytes()) - 1):
            with self.subTest(offset=offset):
                original = archive.read_bytes()
                changed = bytearray(original)
                changed[offset] ^= 0xff
                archive.write_bytes(changed)
                self.rejected("extract", archive, "z-duplicate", "--index", index, "-o", output, "-f")
                self.assertEqual((output / "z-duplicate").read_bytes(), payload)
                archive.write_bytes(original)

    def test_list_encrypted_public_and_full_json(self):
        source = self.feature_tree()
        archive, password = self.root / "archive", self.root / "password"
        password.write_text("listing test passphrase", encoding="utf-8")
        self.ok("archive", source, "-o", archive, "--password-file", password,
                "--protect", "nested", "--protect", "duplicate")
        self.rejected("list", archive, "--json")
        public = json.loads(self.ok("list", archive, "--public-only", "--json").stdout)
        self.assertEqual([row["path"] for row in public], [".hidden", "zero"])
        full = json.loads(self.ok("list", archive, "--password-file", password, "--json").stdout)
        self.assertEqual({row["path"]: row["size"] for row in full},
                         {p.relative_to(source).as_posix(): None if p.is_dir() else p.stat().st_size
                          for p in source.rglob("*")})

    @unittest.skipUnless(os.name == "posix", "TMPDIR is POSIX-specific")
    def test_list_without_extraction_scratch(self):
        source = self.feature_tree()
        archive = self.root / "archive"
        self.ok("archive", source, "-o", archive)
        result = subprocess.run(
            [str(BATO_BIN), "list", str(archive), "--json"],
            env={**os.environ, "TMPDIR": str(self.root / "missing")},
            capture_output=True, timeout=CLI_TIMEOUT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads(result.stdout)
        self.assertEqual({row["path"]: row["size"] for row in rows},
                         {p.relative_to(source).as_posix(): None if p.is_dir() else p.stat().st_size
                          for p in source.rglob("*")})

    def test_archive_diff_detects_same_size_content_and_type_changes(self):
        first, second = self.root / "first", self.root / "second"
        first.mkdir()
        second.mkdir()
        (first / "removed").write_bytes(b"old")
        (second / "added").write_bytes(b"new")
        (first / "changed").write_bytes(b"a")
        (second / "changed").write_bytes(b"b")
        (first / "type").mkdir()
        (second / "type").write_bytes(b"")
        for directory in (first, second):
            (directory / "same").write_bytes(b"equal")
        archive, other = self.root / "archive", self.root / "other"
        self.ok("archive", first, "-o", archive)
        self.ok("archive", second, "-o", other)
        expected = {"added": "added", "changed": "changed", "removed": "removed", "type": "changed"}
        for target in (second, other):
            rows = json.loads(self.ok("diff", archive, target, "--json").stdout)
            self.assertEqual({row["path"]: row["status"] for row in rows}, expected)
        self.assertEqual(json.loads(self.ok("diff", archive, first, "--json").stdout), [])

    def test_embedded_plain_index_survives_edit_and_rejects_damage(self):
        source = self.feature_tree()
        archive = self.root / "archive"
        self.ok("archive", source, "--indexed", "-o", archive)
        original = archive.read_bytes()
        self.assertEqual(original[:4], b"BAI1")
        output = self.root / "output"
        self.ok("extract", archive, "nested", "-o", output)
        self.assertEqual(snapshot(output / "nested"), snapshot(source / "nested"))
        self.ok("remove", archive, "duplicate")
        self.assertEqual(archive.read_bytes()[:4], b"BAI1")
        self.rejected("extract", archive, "duplicate", "-o", self.root / "absent")
        self.ok("extract", archive, "nested", "-o", output, "-f")
        changed = bytearray(original)
        changed[20] ^= 1
        archive.write_bytes(changed)
        self.rejected("list", archive)
        self.rejected("extract", archive, "nested", "-o", output, "-f")
        self.assertEqual(snapshot(output / "nested"), snapshot(source / "nested"))

    def test_embedded_encrypted_index_authenticates_selected_chunks(self):
        source = self.root / "source"
        source.mkdir()
        rng = random.Random(501)
        payload = rng.randbytes(4000)
        (source / "a-original").write_bytes(payload)
        (source / "m-unselected").write_bytes(rng.randbytes(400_000))
        (source / "z-duplicate").write_bytes(payload)
        archive, password, wrong = (self.root / name for name in ("archive", "password", "wrong"))
        password.write_text("correct embedded passphrase", encoding="utf-8")
        wrong.write_text("incorrect embedded passphrase", encoding="utf-8")
        access = ("--password-file", password)
        self.ok("archive", source, "--indexed", *access, "-o", archive,
                "--segment-size", 65536, "--chunk-size", 16384)
        original = archive.read_bytes()
        self.assertEqual(original[:4], b"BAE2")
        self.assertNotIn(b"z-duplicate", original)
        self.rejected("list", archive, "--password-file", wrong)
        self.ok("list", archive, *access)
        full = self.root / "full"
        self.ok("extract", archive, *access, "-o", full)
        self.assertEqual(snapshot(full), snapshot(source))

        # An unrelated middle age chunk is outside the selected Bato extent.
        damaged = bytearray(original)
        damaged[200_000] ^= 1
        archive.write_bytes(damaged)
        selected = self.root / "selected"
        self.ok("extract", archive, "z-duplicate", *access, "-o", selected)
        self.assertEqual({p.name for p in selected.iterdir()}, {"z-duplicate"})
        self.assertEqual((selected / "z-duplicate").read_bytes(), payload)
        self.rejected("-t", archive, *access)
        self.rejected("extract", archive, *access, "-o", full, "-f")
        self.assertEqual(snapshot(full), snapshot(source))
        self.rejected("extract", archive, "m-unselected", *access, "-o", self.root / "damaged")
        self.assertFalse((self.root / "damaged").exists())

        damaged = bytearray(original)
        damaged[512] ^= 1
        archive.write_bytes(damaged)
        self.rejected("extract", archive, "z-duplicate", *access, "-o", selected, "-f")
        self.assertEqual((selected / "z-duplicate").read_bytes(), payload)

    def test_archive_health_reports_degraded_and_repairable_volumes(self):
        source = self.feature_tree()
        archive = self.root / "split"
        self.ok("archive", source, "-o", archive, "--split-size", 32000, "--parity-volumes", 2)
        parts = sorted(archive.glob("volumes-*/part-*"))
        parity = sorted(archive.glob("volumes-*/parity-*"))
        for status, remove in (("healthy", None), ("degraded", parity[0]), ("repairable", parts[0])):
            if remove:
                remove.unlink()
            report = json.loads(self.ok("health", archive, "--json").stdout)
            self.assertEqual(report["status"], status)
            self.assertTrue(report["archive_sha256_verified"])
        parts[1].unlink()
        result = self.run_cli("health", archive, "--json")
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "unrecoverable")
        self.assertFalse(report["archive_sha256_verified"])
        self.assertEqual(len(report["affected_volumes"]), 3)

    def test_multi_parity_recovers_two_losses_and_rejects_three(self):
        source = self.feature_tree()
        archive, restored = self.root / "split", self.root / "restored"
        self.ok("archive", source, "-o", archive, "--split-size", 32000, "--parity-volumes", 2)
        parts = sorted(archive.glob("volumes-*/part-*"))
        self.assertGreaterEqual(len(parts), 3)
        self.assertEqual(len(list(archive.glob("volumes-*/parity-*"))), 2)
        parts[0].unlink()
        damaged = bytearray(parts[-1].read_bytes())
        damaged[-1] ^= 1
        parts[-1].write_bytes(damaged)
        repaired = self.root / "repaired"
        self.ok("repair-split", archive, "-o", repaired)
        self.ok("extract", repaired, "-o", restored)
        self.assertEqual(snapshot(restored), snapshot(source))
        parts[1].unlink()
        missing = self.root / "missing"
        self.rejected("extract", archive, "-o", missing)
        self.assertFalse(missing.exists())

    def test_checkpoint_progress_status_and_competing_resume(self):
        source, archive, job = (self.root / name for name in ("input", "archive", "job"))
        source.write_bytes(random.Random(151).randbytes(140_000))
        args = [source, "-o", archive, "--segment-size", 65536, "--chunk-size", 16384]
        result = self.ok(*args, "--checkpoint", job, "--pause-after-segments", 1, "--progress")
        self.assertIn(b"committed 65536 input bytes", result.stderr)
        self.assertIn(b"65536 decoded bytes", self.ok("job-status", job).stdout)
        if sys.platform.startswith("linux"):
            import fcntl
            with (job / "job").open("rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.rejected(*args, "--resume", job)
            self.assertFalse(archive.exists())
        self.ok(*args, "--resume", job)
        decoded = self.root / "decoded"
        self.ok("-d", archive, "-o", decoded)
        self.assertEqual(snapshot(decoded), snapshot(source))

    def test_alternating_entropy_across_segment_boundaries(self):
        rng = random.Random(179)
        data = b"".join((rng.randbytes(65535), b"A" * 65537,
                         gzip.compress(rng.randbytes(65533), mtime=0),
                         bytes(range(256)) * 260))
        source, archive, restored = (self.root / name for name in ("input", "archive", "output"))
        source.write_bytes(data)
        self.ok(source, "-o", archive, "--segment-size", 65536, "--chunk-size", 16384)
        self.ok("-d", archive, "-o", restored)
        self.assertEqual(restored.read_bytes(), data)

    def test_portable_unicode_and_deep_paths(self):
        source = self.root / "tree"
        names = ["Tiranë_日本語.txt", "é.txt", "Case.txt"]
        source.mkdir()
        for index, name in enumerate(names):
            (source / name).write_bytes(bytes([index]))
        case_variant = source / "case.txt"
        if not case_variant.exists():
            case_variant.write_bytes(b"distinct case")
        nested = source.joinpath(*["deep"] * 32)
        nested.mkdir(parents=True)
        (nested / "payload").write_bytes(b"deep payload")
        archive, restored = self.root / "archive", self.root / "restored"
        self.ok("archive", source, "-o", archive)
        self.ok("extract", archive, "-o", restored)
        self.assertEqual(snapshot(restored), snapshot(source))

    @unittest.skipUnless(os.environ.get("BATO_MULTIGIB_STRESS") == "1",
                         "set BATO_MULTIGIB_STRESS=1 for high file-count tests")
    def test_ten_thousand_small_files_and_empty_directories(self):
        source, archive, output = (self.root / name for name in ("tree", "archive", "output"))
        sample = random.Random(882).randbytes(1024)
        for group in range(100):
            directory = source / f"group-{group:03}"
            (directory / "empty").mkdir(parents=True)
            for index in range(100):
                (directory / f"{index:03}").write_bytes(sample)
        (source / ".hidden").write_bytes(b"manifest fixture")
        self.ok("archive", source, "-o", archive)
        self.ok("extract", archive, "-o", output)
        self.assertEqual(snapshot(output), snapshot(source))

    @unittest.skipUnless(os.environ.get("BATO_MULTIGIB_STRESS") == "1",
                         "set BATO_MULTIGIB_STRESS=1 for multi-GiB boundary tests")
    def test_multigib_file_size_boundaries(self):
        source, archive, output = (self.root / name for name in ("input", "archive", "output"))
        for boundary in (2 ** 31, 2 ** 32):
            for delta in (-1, 0, 1):
                with self.subTest(size=boundary + delta):
                    with source.open("wb") as stream:
                        stream.write(b"begin")
                        stream.seek(boundary + delta - 3)
                        stream.write(b"end")
                    expected = snapshot(source)
                    self.ok(source, "-o", archive, "-f")
                    self.ok("-d", archive, "-o", output, "-f")
                    self.assertEqual(snapshot(output), expected)
                    output.unlink()


    def test_hostile_split_manifests_leave_destination_unpublished(self):
        source = self.feature_tree()
        archive = self.root / "split"
        output = self.root / "output"
        self.ok("archive", source, "-o", archive, "--split-size", 32000, "--parity-volumes", 2)
        manifest = archive / "index.bato"
        original = manifest.read_bytes()
        for offset, patch in ((0, b"NOPE"), (4, bytes(8)), (12, b"\xff" * 8),
                              (52, b"../"), (64, b"\xff")):
            with self.subTest(offset=offset):
                manifest.write_bytes(original[:offset] + patch + original[offset + len(patch):])
                self.rejected("extract", archive, "-o", output)
                self.assertFalse(output.exists())
        manifest.write_bytes(original)
        self.ok("extract", archive, "-o", output)
        self.assertEqual(snapshot(output), snapshot(source))

    def test_hostile_recovery_index_with_valid_outer_checksum(self):
        source = self.feature_tree()
        archive, index, output = (self.root / name for name in ("archive", "index", "output"))
        self.ok("archive", source, "-o", archive)
        self.ok("recovery-index", archive, "-o", index)
        body = index.read_bytes()[:-32]
        self.assertEqual(body[:4], b"BAR2")
        mutations = [b"JUNK" + body[4:], body[:4] + bytes(32) + body[36:],
                     body[:46] + b"\xff" * 8 + body[54:], body[:-1], body + b"trailing"]
        for number, mutated in enumerate(mutations):
            with self.subTest(mutation=number):
                index.write_bytes(mutated + hashlib.sha256(mutated).digest())
                self.rejected("extract", archive, "nested", "--index", index, "-o", output)
                self.assertFalse(output.exists())

    def test_concurrent_split_edit_and_extract_publish_consistent_trees(self):
        source = self.feature_tree()
        archive = self.root / "split"
        self.ok("archive", source, "-o", archive, "--split-size", 32000)
        files = [self.root / name for name in ("first", "second")]
        for path in files:
            path.write_bytes(path.name.encode() * 128)
        original = snapshot(source)
        output = self.root / "concurrent"
        with ThreadPoolExecutor(max_workers=3) as pool:
            edits = [pool.submit(self.run_cli, "--threads", 1, "add", archive, path)
                     for path in files]
            extraction = pool.submit(self.run_cli, "--threads", 1, "extract", archive, "-o", output)
            successful = [path for path, future in zip(files, edits) if future.result().returncode == 0]
            extracted = extraction.result()
        self.assertTrue(successful)
        final = self.root / "final"
        self.ok("extract", archive, "-o", final)
        for path in successful:
            shutil.copyfile(path, source / path.name)
        self.assertEqual(snapshot(final), snapshot(source))
        if extracted.returncode:
            self.assertFalse(output.exists())
        else:
            concurrent = snapshot(output)
            original_files = {row["path"]: row for row in original["files"]}
            final_files = {row["path"]: row for row in snapshot(source)["files"]}
            observed = {row["path"]: row for row in concurrent["files"]}
            self.assertTrue(all(observed.get(name) == row for name, row in original_files.items()))
            self.assertTrue(all(final_files.get(name) == row for name, row in observed.items()))
            self.assertEqual(concurrent["directories"], original["directories"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux mount namespaces")
    def test_real_filesystem_exhaustion_preserves_output(self):
        unshare, mount = shutil.which("unshare"), shutil.which("mount")
        if not unshare or not mount or subprocess.run(
            [unshare, "-Urnm", "true"], capture_output=True
        ).returncode:
            self.skipTest("private mount namespace unavailable")
        source = self.root / "input"
        source.write_bytes(random.Random(629).randbytes(2 * 1024 * 1024))
        archive = self.root / "archive"
        self.ok(source, "-o", archive)
        mountpoint = self.root / "limited"
        mountpoint.mkdir()
        code = """
import os, pathlib, shutil, subprocess, sys
binary, mount, mountpoint, source, archive, indexed = sys.argv[1:]
subprocess.run([mount, '-t', 'tmpfs', '-o', 'size=1m', 'tmpfs', mountpoint], check=True)
target = pathlib.Path(mountpoint) / 'output'
for args in ([source], ['-d', archive]):
    target.write_bytes(b'existing')
    result = subprocess.run([binary, '--threads', '1', *args, '-f', '-o', str(target)], capture_output=True)
    assert result.returncode > 0 and b'No space left' in result.stderr, result.stderr
    assert target.read_bytes() == b'existing'
    assert sorted(path.name for path in target.parent.iterdir()) == ['output']
target.unlink()
edited = target.parent / 'edited'
shutil.copyfile(indexed, edited)
original = edited.read_bytes()
# Keep staging for the extracted tree outside the constrained output mount.
result = subprocess.run([binary, '--threads', '1', 'add', edited, source],
                        capture_output=True, env={**os.environ, 'TMPDIR': str(pathlib.Path(source).parent)})
assert result.returncode > 0 and b'No space left' in result.stderr, result.stderr
assert edited.read_bytes() == original
assert sorted(path.name for path in edited.parent.iterdir()) == ['edited']
subprocess.run([binary, '-t', edited], check=True)
"""
        tree = self.root / "tree"
        tree.mkdir()
        (tree / "original").write_bytes(b"preserve")
        indexed = self.root / "indexed"
        self.ok("archive", tree, "--indexed", "-o", indexed)
        result = subprocess.run([unshare, "-Urnm", sys.executable, "-c", code,
                                 str(BATO_BIN), mount, str(mountpoint), str(source), str(archive), str(indexed)],
                                capture_output=True, timeout=CLI_TIMEOUT)
        self.assertEqual(result.returncode, 0, result.stderr)


    def test_interrupted_indexed_edits_preserve_complete_generation(self):
        source = self.feature_tree()
        archive = self.root / "archive"
        addition = self.root / "addition"
        addition.write_bytes(random.Random(713).randbytes(512 * 1024))
        original = snapshot(source)
        for operation, argument in (("add", addition), ("remove", "nested")):
            with self.subTest(operation=operation):
                expected = self.root / f"expected-{operation}"
                shutil.copytree(source, expected)
                if operation == "add":
                    shutil.copyfile(addition, expected / addition.name)
                else:
                    shutil.rmtree(expected / "nested")
                self.ok("archive", source, "--indexed", "-f", "-o", archive)
                scratch = self.root / f"scratch-{operation}"
                scratch.mkdir()
                process = subprocess.Popen(
                    [str(BATO_BIN), "--threads", "1", operation, str(archive), str(argument)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env={**os.environ, "TMPDIR": str(scratch)},
                )
                try:
                    deadline = time.monotonic() + CLI_TIMEOUT
                    while process.poll() is None and not any(self.root.glob(".bato-*")):
                        if time.monotonic() >= deadline:
                            self.fail("edit did not reach publication staging")
                        time.sleep(0.001)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=CLI_TIMEOUT)
                if process.returncode > 0:
                    self.fail(f"edit exited with status {process.returncode}")
                self.ok("-t", archive)
                restored = self.root / f"restored-{operation}"
                self.ok("extract", archive, "-o", restored)
                states = [snapshot(expected)]
                if process.returncode < 0:
                    states.append(original)
                self.assertIn(snapshot(restored), states)
                for path in self.root.glob(".bato-*"):
                    if path.is_file():
                        path.unlink()

    def test_indexed_edit_file_size_limit_rolls_back(self):
        if os.name != "posix":
            self.skipTest("requires POSIX file-size limits")
        source = self.root / "tree"
        source.mkdir()
        (source / "original").write_bytes(b"original")
        archive = self.root / "archive"
        self.ok("archive", source, "--indexed", "-o", archive)
        original = sha256(archive)
        addition = self.root / "addition"
        addition.write_bytes(random.Random(330).randbytes(256 * 1024))
        code = """
import os, resource, signal, sys
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
os.execv(sys.argv[1], sys.argv[1:])
"""
        result = subprocess.run([sys.executable, "-c", code, str(BATO_BIN), "--threads", "1",
                                 "add", str(archive), str(addition)],
                                capture_output=True, timeout=CLI_TIMEOUT)
        self.assertGreater(result.returncode, 0, result.stderr)
        self.assertIn(b"File too large", result.stderr)
        self.assertEqual(sha256(archive), original)
        self.ok("-t", archive)
        restored = self.root / "restored"
        self.ok("extract", archive, "-o", restored)
        self.assertEqual(snapshot(restored), snapshot(source))

    def test_extended_hundred_thousand_file_edits(self):
        source = self.root / "source"
        for group in range(100):
            directory = source / f"group-{group:03}"
            (directory / "empty").mkdir(parents=True)
            for member in range(1000):
                (directory / f"{member:03}").write_bytes(f"{member:03}".encode())
        archive = self.root / "archive"
        rejection = self.rejected("archive", source, "--indexed", "-o", archive)
        self.assertRegex(rejection.stderr, rb"Working memory limit exceeded|archive entry limit exceeded")
        self.assertFalse(archive.exists())
        # Inventory accounting depends on path lengths and can reach its cap first.
        for group in range(30, 100):
            shutil.rmtree(source / f"group-{group:03}")
        self.ok("archive", source, "--indexed", "-o", archive)
        replacement = self.root / "group-001"
        replacement.mkdir()
        (replacement / "000").write_bytes(b"new")
        (replacement / "new" / "empty").mkdir(parents=True)
        (replacement / "new" / "duplicate").write_bytes(b"001")
        self.ok("add", archive, replacement, "-f")
        shutil.copytree(replacement, source / replacement.name, dirs_exist_ok=True)
        self.ok("remove", archive, "group-000")
        shutil.rmtree(source / "group-000")
        restored = self.root / "restored"
        self.ok("extract", archive, "-o", restored)
        self.assertEqual(snapshot(restored), snapshot(source))

    def test_extended_multigib_mixed_restore(self):
        source = self.root / "source"
        source.mkdir()
        payload = random.Random(734).randbytes(1024 * 1024)
        with (source / "mixed").open("wb") as stream:
            stream.write(payload)
            stream.seek(2**31)
            stream.write(payload)
        (source / "z-last").write_bytes(payload[:4096])
        archive = self.root / "archive"
        self.ok("archive", source, "--indexed", "--segment-size", 1024 * 1024, "-o", archive)
        restored = self.root / "restored"
        self.ok("extract", archive, "-o", restored)
        self.assertEqual(snapshot(restored), snapshot(source))
        selected = self.root / "selected"
        self.ok("extract", archive, "z-last", "-o", selected)
        self.assertEqual(list(selected.iterdir()), [selected / "z-last"])
        self.assertEqual((selected / "z-last").read_bytes(), payload[:4096])

    def test_split_publication_survives_process_death(self):
        source = self.root / "tree"
        source.mkdir()
        (source / "binary").write_bytes(random.Random(270).randbytes(2 * 1024 * 1024))
        archive = self.root / "split"
        args = ["archive", source, "-o", archive, "--split-size", 32000]
        process = subprocess.Popen([str(BATO_BIN), "--threads", "2", *map(str, args)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + CLI_TIMEOUT
            while not any(self.root.glob(".bato-stage-*")) and process.poll() is None:
                if time.monotonic() >= deadline:
                    self.fail("split publication staging did not start")
                time.sleep(0.001)
            self.assertTrue(archive.exists() or any(self.root.glob(".bato-stage-*")),
                            "split job exited before publication staging")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=CLI_TIMEOUT)
        # The rename may have won the race. Any published generation must be complete.
        if not archive.exists():
            self.ok(*args)
        self.ok("-t", archive)
        restored = self.root / "restored"
        self.ok("extract", archive, "-o", restored)
        self.assertEqual(snapshot(restored), snapshot(source))


class RealFileTest(EngineTests):
    """One independently reported round trip for a verified corpus member."""

    def __init__(self, sample):
        super().__init__('runTest')
        self.sample = sample

    def id(self):
        return f'real_file:{self.sample.relative_path}'

    def __str__(self):
        return self.id()

    def runTest(self):
        source = self.sample.verified_path()
        archive, restored = self.root / 'sample.bato', self.root / 'restored'
        phases = [('compress', ('-f', source, '-o', archive)),
                  ('test', ('-t', archive)),
                  ('decompress', ('-d', '-f', archive, '-o', restored))]
        timings = {}
        for phase, command in phases:
            started = time.monotonic()
            result = self.run_cli(*command)
            timings[phase] = time.monotonic() - started
            self.assertEqual(result.returncode, 0, f'{phase}: {result.stderr!r}')
        self.sample.verified_path()
        self.assertTrue(filecmp.cmp(source, restored, shallow=False), 'Restored file bytes differ')
        print(f"  {self.sample.relative_path}: original={self.sample.size} bytes "
              f"compressed={archive.stat().st_size} bytes "
              + ' '.join(f'{phase}={seconds:.3f}s' for phase, seconds in timings.items()), flush=True)


def main(argv=None):
    global BATO_BIN, CLI_TIMEOUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bato-bin')
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--test', action='append', default=[],
                        help='Run only the named EngineTests method. Can be repeated.')
    parser.add_argument('--extended', action='store_true', help='Include high-disk and high-file-count stress cases')
    parser.add_argument('--min', action='store_true', help='Alias for --sample-mode fast')
    parser.add_argument('--sample-mode', choices=['fast', 'normal', 'full'], default='normal', help='Number of copied files per extension')
    parser.add_argument('--seed', type=int, default=42, help='Reproducible file selection seed')
    parser.add_argument('--corpus', default=str(DEFAULT_COPIED_CORPUS), help='Copied real-file corpus directory')
    parser.add_argument('--database', default=str(PROJECT_ROOT / 'data-research' / 'measurements.sqlite3'), help='Original file sizes and competitor measurements')
    parser.add_argument('--benchmark-output', default=str(PROJECT_ROOT / 'data-research' / 'test-results'), help='Baseline reports directory')
    parser.add_argument('--scratch-dir', default=str(Path(tempfile.gettempdir()) / 'bato-test-scratch'), help='Temporary benchmark workspace')
    parser.add_argument('--competitors', type=lambda v: tuple(v.split(',')), default=None,
                        help='Comma-separated competitor subset for the baseline gate (default: all, including paq8px, cmix and nncp)')
    parser.add_argument('--competitor-timeout', type=int, default=600, help='Seconds per competitor measurement phase')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    if args.competitor_timeout <= 0:
        parser.error('--competitor-timeout must be positive')
    if args.min and (args.test or args.extended or args.sample_mode != 'normal'):
        parser.error('--min cannot be combined with --test, --extended, or --sample-mode')
    if args.min:
        args.sample_mode = 'fast'
    known = unittest.defaultTestLoader.getTestCaseNames(EngineTests)
    if args.test and any(name not in known for name in args.test):
        parser.error('--test requires an EngineTests method name')
    try:
        samples = [] if args.test else select_samples(args.corpus, args.sample_mode, args.seed)
        BATO_BIN, version = resolve_bato(args.bato_bin)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    CLI_TIMEOUT = args.timeout
    print(f'Testing {version}: {BATO_BIN}', flush=True)
    for sample in samples:
        print(f'Corpus input: {sample.relative_path} | {sample.size} bytes | SHA-256 {sample.digest}', flush=True)
    baseline_status = 0
    if samples:
        try:
            baseline_status = ensure_test_baselines(samples,args)
        except (OSError, ValueError, sqlite3.Error) as error:
            print(f'Competitor baseline gate failed: {error}',file=sys.stderr)
            return 2
        if baseline_status == 2:
            print('Competitor campaign could not start; Bato correctness tests were not started.',file=sys.stderr)
            return baseline_status
        if baseline_status:
            print('Competitor results are partial or contain failures; continuing Bato correctness tests.',file=sys.stderr)
    names = args.test or [name for name in known if args.extended or not name.startswith('test_extended_')]
    suite = unittest.TestSuite(EngineTests(name) for name in names)
    suite.addTests(RealFileTest(sample) for sample in samples)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and not baseline_status else 1


if __name__ == '__main__':
    raise SystemExit(main())
