"""Archive-operation campaign inputs, isolation, and reporting."""
import contextlib
import io
import json
import sqlite3
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bato_tools import archive_ops
from bato_tools.common import ProcessFailure


class ArchiveOperationTests(unittest.TestCase):
    def test_frozen_fixture_rejects_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = archive_ops.fixtures(root)
            frozen = root / "manifest.json"
            frozen.write_text(json.dumps(manifest))
            archive_ops.validate_fixture(manifest, frozen)
            manifest["changed"]["files"][0]["sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "differs"):
                archive_ops.validate_fixture(manifest, frozen)

    def test_trials_copy_seed_and_record_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = archive_ops.fixtures(root)
            seed = root / "seed"
            seed.write_bytes(b"original")
            starts = []

            def interrupted(command, *args, **kwargs):
                archive = command[-2]
                starts.append(archive.read_bytes())
                archive.write_bytes(b"partial edit")
                raise ProcessFailure(command, "interrupted", timed_out=True)

            with patch.object(archive_ops, "run_process", side_effect=interrupted):
                for _ in range(2):
                    result = archive_ops.trial("bato", {"threads": 1, "memory": 10}, "add",
                                               (root, seed, manifest, 1))
                    self.assertEqual(result["status"], "timeout")
            self.assertEqual(starts, [b"original", b"original"])
            self.assertEqual(seed.read_bytes(), b"original")

    def test_campaign_interleaves_builds_and_persists_failures(self):
        with tempfile.TemporaryDirectory() as directory, sqlite3.connect(":memory:") as db:
            root = Path(directory)
            manifest = archive_ops.fixtures(root)
            frozen = root / "frozen"
            frozen.write_text(json.dumps(manifest))
            args = SimpleNamespace(frozen_workloads=frozen, bato_modes="cpu1", threads=4,
                                   bato_memory_budget=100, bato_bin="new", bato_revision="new-sha",
                                   baseline_bato_bin="old", baseline_bato_revision="old-sha",
                                   warmups=0, repetitions=2, timeout=3)
            order = []

            def identity(binary, revision):
                return binary, dict(sha256=binary, version="same-version", indexed=True,
                                    operations=archive_ops.OPERATIONS, declared_git_sha=revision)

            def prepare(command, *args, **kwargs):
                if "extract" in command:
                    shutil.copytree(root / "source", command[-1])
                else:
                    command[-1].write_bytes(b"BAI1seed")

            def trial(binary, track, operation, inputs):
                if operation == "add":
                    order.append(binary)
                return {"status": "timeout" if operation == "add" and binary == "old" else "passed"}

            archive_ops.init_database(db)
            with (patch.object(archive_ops, "configure_resources"),
                  patch.object(archive_ops, "environment", return_value={}),
                  patch.object(archive_ops, "storage_identity", return_value="scratch"),
                  patch.object(archive_ops, "build_identity", side_effect=identity),
                  patch.object(archive_ops, "run_process", side_effect=prepare),
                  patch.object(archive_ops, "trial", side_effect=trial),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(archive_ops.campaign(args, root, manifest, db), 1)
            self.assertEqual(order, ["old", "new", "new", "old"])
            records = db.execute("SELECT record FROM archive_operation_trials").fetchall()
            self.assertEqual(len(records), 20)
            self.assertEqual(sum(json.loads(record)["status"] == "timeout" for record, in records), 2)

    def test_comparison_retains_failure_and_refuses_unmatched_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks = {"cpu1": {"threads": 1, "memory": 100}}
            with sqlite3.connect(root / "measurements.sqlite3") as db:
                archive_ops.init_database(db)
                for run in ("old", "new"):
                    meta = dict(fixture="same", tracks=tracks, host={"host_id_sha256": run}, storage="same",
                                build={"version": "0.7", "sha256": run, "executable_target": "ELF"},
                                warmups=0, repetitions=1, timeout=1, cache_policy="same")
                    db.execute("INSERT INTO archive_operation_runs VALUES (?, ?)", (run, json.dumps(meta)))
                    for op in ("add", "remove"):
                        record = dict(status="timeout" if op == "remove" and run == "new" else "passed",
                                      wall_s=2 if run == "old" else 1, archive_bytes=200 if run == "old" else 180)
                        db.execute("INSERT INTO archive_operation_trials VALUES (?, ?, ?, ?, ?, ?)",
                                   (run, "cpu1", op, "measured", 0, json.dumps(record)))
            with contextlib.redirect_stdout(io.StringIO()):
                archive_ops.compare_operations(root, "old", "new")
            report = json.loads(next((root / "archive-operation-comparisons").glob("*.json")).read_text())
            cases = {case["operation"]: case for case in report["cases"]}
            self.assertFalse(cases["add"]["timing_conditions_match"])
            self.assertEqual(set(cases["add"]["metrics"]), {"archive_bytes"})
            self.assertEqual(cases["remove"]["status"], "unmatched_or_failed")
            self.assertEqual(cases["remove"]["outcomes"][1], ["timeout"])
