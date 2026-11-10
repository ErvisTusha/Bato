"""Extreme settings, mandatory preflight, native paths, and verified exports."""
import contextlib
import csv
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_suite
from bato_tools.benchmark import export, measure, print_archive_summary
from bato_tools.codecs import Codec, DEFAULT_CODECS, WORKER, check_tools, create_codecs, probe_command
from bato_tools.common import Measurement, ProcessFailure, run_process, snapshot


class ArchiverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "input"
        (self.source / "nested space").mkdir(parents=True)
        (self.source / "nested space/日本語.bin").write_bytes(bytes(range(256)) * 8)
        (self.source / ".hidden.txt").write_text("hidden payload\n" * 64)
        (self.source / "empty.dat").touch()

    def tearDown(self):
        self.temporary.cleanup()

    def test_modes_select_identical_extreme_settings(self):
        with patch("bato_tools.codecs.shutil.which", return_value=None):
            quick = create_codecs(Path(sys.executable), "fixture")
            rigorous = create_codecs(Path(sys.executable), "fixture")
        self.assertEqual([codec.identifier for codec in quick], ["bato-v1", *DEFAULT_CODECS[1:]])
        self.assertEqual([codec.settings() for codec in quick], [codec.settings() for codec in rigorous])
        expected = {"bato-v1": 0, "kgb": 9, "paq8px": 12, "paq8l": 8, "zpaq": 5, "rar": 5, "7z": 9, "xz": 9, "zstd": 22, "gzip": 9, "lz4": 12, "zip": 9, "bzip2": 9}
        self.assertEqual({codec.identifier: codec.level for codec in quick}, expected)

    def test_cmix_nncp_selection_and_identity(self):
        with patch("bato_tools.codecs.shutil.which", return_value=None):
            chosen = create_codecs(None, None, "cmix,nncp", threads=4, nncp_profile="enwik9", nncp_cuda=True)
        cmix, nncp = chosen
        self.assertEqual([codec.identifier for codec in chosen], ["cmix", "nncp"])
        self.assertEqual(cmix.archive_suffix(), ".cmix")
        self.assertEqual(nncp.archive_suffix(), ".nncp")
        self.assertEqual(cmix.compression_options(), [])
        self.assertEqual(nncp.compression_options(), ["-T", "4", "--profile", "enwik9", "--cuda"])
        self.assertEqual(nncp.settings()["compute"], "cuda")
        self.assertEqual(nncp.settings()["nncp_profile"], "enwik9")
        self.assertEqual(nncp.settings()["directory_container"], "tar")
        self.assertEqual(nncp.settings()["decompression_workers"], 4)
        with self.assertRaisesRegex(ValueError, "NNCP profile"):
            create_codecs(None, None, "nncp", nncp_profile="unrecognized")

    def test_cmix_nncp_commands_and_directory_container(self):
        for name, mode in [("cmix", "-c"), ("nncp", "c")]:
            with self.subTest(name=name):
                codec = Codec(name, name, "/test/tool", threads=3)
                work = self.root / name
                work.mkdir()
                payload = self.source / ".hidden.txt"
                with patch("bato_tools.codecs.run_process", return_value=Measurement(0.1, None, None, [])) as runner:
                    codec.step("compress", payload, work / ("data" + codec.archive_suffix()), work, 10, 100)
                    self.assertIn(mode, runner.call_args.args[0])
                    codec.step("decompress", work / ("data" + codec.archive_suffix()), work / "restored", work, 10, 100)
                    self.assertIn("-d" if name == "cmix" else "d", runner.call_args.args[0])
                    codec.workflow("compress", self.source, work / ("folder" + codec.archive_suffix()), work, 10,
                                   snapshot(self.source), True)
                    command = runner.call_args.args[0]
                    self.assertIn(work / "payload.tar", command)
                self.assertFalse((work / "payload.tar").exists())

    def test_command_arguments_use_extreme_settings(self):
        expected = {"xz": ["-9", "--extreme", "--no-adjust", "-T1"], "zstd": ["--ultra", "-22", "--single-thread"],
                    "gzip": ["-9", "-n"], "lz4": ["-12", "-T1"], "7z": ["-mx=9", "-mmt=1", "-ms=on"],
                    "rar": ["-m5", "-mt1", "-s", "-cfg-"], "kgb": ["-9"]}
        for name, options in expected.items():
            with self.subTest(codec=name):
                work = self.root / name
                work.mkdir()
                codec = Codec(name, name, sys.executable, lz4_workers=True)
                directory = name in {"rar", "7z", "kgb"}
                source = self.source if directory else self.source / ".hidden.txt"
                with patch("bato_tools.codecs.run_process", return_value=Measurement(0.1, None, None, [])) as runner:
                    codec.step("compress", source, work / ("compressed" + codec.archive_suffix(directory)), work, 1, 4096, directory)
                command = runner.call_args.args[0]
                for option in options:
                    self.assertIn(option, command)

    def test_bato_commands_apply_reported_thread_and_chunk_settings(self):
        codec = Codec("bato-v1", "fixture", sys.executable)
        work = self.root / "bato"
        work.mkdir()
        with patch("bato_tools.codecs.run_process", return_value=Measurement(0.1, None, None, [])) as runner:
            codec.step("compress", self.source, work / "archive.bato", work, 1, 4096, True)
        command = runner.call_args.args[0]
        self.assertEqual(command[command.index("--threads") + 1], "1")
        self.assertEqual(command[command.index("--chunk-size") + 1], str(256 * 1024))
        self.assertIn("--memory-budget", command)
        self.assertNotIn("dictionary_size", Codec("bato-v1", "fixture", None).settings())

    def test_kgb_file_list_and_extraction_parents(self):
        codec = Codec("kgb", "KGB", sys.executable)
        work = self.root / "work"
        work.mkdir()
        with patch("bato_tools.codecs.run_process", return_value=Measurement(0.1, None, None, [])) as runner:
            codec.workflow("compress", self.source, work / "compressed.kgb", work, 1, snapshot(self.source), True)
            names = (work / "kgb-files.txt").read_text().splitlines()
            self.assertEqual(names, ["./.hidden.txt", "./empty.dat", "./nested space/日本語.bin"])
            self.assertEqual(runner.call_args.args[0][3:], names)
            codec.workflow("decompress", work / "compressed.kgb", work / "restored", work, 1, snapshot(self.source), True)
        self.assertTrue((work / "restored/nested space").is_dir())
        self.assertEqual(len(snapshot(self.source)["files"]), 3)

    def test_kgb_ascii_files_use_listing_without_argument_count_limits(self):
        codec = Codec("kgb", "KGB", sys.executable)
        work = self.root / "work"
        work.mkdir()
        with patch("bato_tools.codecs.run_process", return_value=Measurement(0.1, None, None, [])) as runner:
            codec.step("compress", self.source / ".hidden.txt", work / "compressed.kgb", work, 1, 4096)
        self.assertEqual(runner.call_args.args[0][3:], ["@" + str(work / "kgb-files.txt")])

    def test_kgb_unrepresentable_filename_is_explicit_failure(self):
        (self.source / "bad\nname").write_bytes(b"data")
        codec = Codec("kgb", "KGB", sys.executable)
        workload = dict(target="input", path=str(self.source), type="directory_archive", category="test", origin="real", snapshot=snapshot(self.source))
        with patch("bato_tools.codecs.run_process", side_effect=AssertionError("Must reject before invocation")):
            record = measure(codec, workload, 1, 0, 1)
        self.assertEqual(record["status"], "FAIL")
        self.assertIn("control characters", record["error"])

    def test_zip_level_nine_is_applied_to_every_entry(self):
        original = zipfile._get_compressor
        output = self.root / "payload.zip"
        with patch("sys.argv", [str(WORKER), "zip", "compress", str(self.source), str(output), "--directory"]), patch("zipfile._get_compressor", wraps=original) as compressor:
            runpy.run_path(str(WORKER), run_name="__main__")
        calls = [call for call in compressor.call_args_list if call.args[0] == zipfile.ZIP_DEFLATED]
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call.args == (zipfile.ZIP_DEFLATED, 9) for call in calls))
        with zipfile.ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())

    def test_missing_tools_block_before_corpus_cache_or_output(self):
        output, scratch = self.root / "reports", self.root / "scratch"
        with patch("bato_tools.codecs.shutil.which", return_value=None), patch("bato_tools.runner.discover", side_effect=AssertionError("Must not hash inputs")), patch("bato_tools.runner.MeasurementStore", side_effect=AssertionError("Must not open store")), contextlib.redirect_stdout(io.StringIO()) as console:
            result = benchmark_suite.main(["--codecs", "rar,kgb,7z,lz4", "-c", str(self.root / "missing-corpus"),
                                           "--output-dir", str(output), "--scratch-dir", str(scratch)])
        self.assertEqual(result, 2)
        self.assertEqual(console.getvalue().count("[MISSING]"), 4)
        self.assertFalse(output.exists())
        self.assertFalse(scratch.exists())

    def test_check_tools_without_corpus_requires_only_selected_codecs(self):
        with patch("bato_tools.codecs.resolve_bato", side_effect=AssertionError("Rust not selected")), contextlib.redirect_stdout(io.StringIO()):
            result = benchmark_suite.main(["--check-tools", "--codecs", "zip,bzip2", "-c", str(self.root / "missing"), "--output-dir", str(self.root / "reports")])
        self.assertEqual(result, 0)
        self.assertFalse((self.root / "reports").exists())

    def test_non_executable_tool_is_missing(self):
        tool = self.root / "gzip"
        tool.write_text("not executable")
        with patch.dict(os.environ, {"PATH": str(self.root)}), contextlib.redirect_stdout(io.StringIO()):
            _, report = check_tools("gzip")
        self.assertEqual(report["tools"][0]["status"], "MISSING")

    def test_wrong_binary_identity_blocks_preflight(self):
        with patch("bato_tools.codecs.shutil.which", return_value=sys.executable), patch("bato_tools.codecs.probe_command", return_value=(0, "UNRAR 7.20")), contextlib.redirect_stdout(io.StringIO()):
            _, report = check_tools("rar")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["tools"][0]["status"], "FAILED")

    def test_nonzero_help_exit_is_allowed_when_roundtrip_succeeds(self):
        def workflow(codec, action, source, output, work, timeout, expected, directory):
            if action == "compress":
                (work / "original-path").write_text(str(source))
                output.write_bytes(b"fixture archive")
            else:
                original = Path((work / "original-path").read_text())
                shutil.copytree(original, output) if directory else shutil.copyfile(original, output)
            return Measurement(0.01, None, None, ["fixture"])
        with patch("bato_tools.codecs.shutil.which", return_value=sys.executable), patch("bato_tools.codecs.probe_command", return_value=(7, "7-Zip 25.01")), patch.object(Codec, "workflow", workflow), contextlib.redirect_stdout(io.StringIO()):
            _, report = check_tools("7z")
        self.assertEqual(report["status"], "ready")

    def test_unsupported_extreme_options_block_preflight(self):
        with patch("bato_tools.codecs.shutil.which", return_value=sys.executable), patch("bato_tools.codecs.probe_command", return_value=(0, "gzip fixture")), patch.object(Codec, "workflow", side_effect=ProcessFailure(["fixture"], "unsupported -9")), contextlib.redirect_stdout(io.StringIO()):
            _, report = check_tools("gzip")
        self.assertEqual(report["status"], "failed")
        self.assertIn("unsupported -9", report["tools"][0]["error"])

    def test_hanging_probe_times_out(self):
        with self.assertRaises(ProcessFailure) as failure:
            probe_command([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.05)
        self.assertTrue(failure.exception.timed_out)

    def test_archiver_errors_written_to_stdout_are_reported(self):
        with self.assertRaises(ProcessFailure) as failure:
            run_process([sys.executable, "-c", "import sys; print('Invalid archive file list'); sys.exit(1)"], self.root, timeout=1)
        self.assertIn("Invalid archive file list", str(failure.exception))

    def test_lz4_old_serial_and_new_worker_options(self):
        for help_text, workers in [("LZ4 old serial CLI", False), ("LZ4 CLI -T# number of workers", True)]:
            with self.subTest(workers=workers), patch("bato_tools.codecs.shutil.which", return_value=sys.executable), patch("bato_tools.codecs.probe_command", side_effect=[(0, "LZ4 version"), (0, help_text)]):
                codec = create_codecs(None, None, "lz4")[0]
            self.assertEqual(codec.lz4_workers, workers)
            self.assertEqual("-T1" in codec.compression_options(), workers)
            self.assertIn("-12", codec.compression_options())

    def test_settings_change_cache_identity(self):
        codec = Codec("bato-v1", "fixture", sys.executable, version="version")
        before = codec.identity()
        codec.version = "updated"
        self.assertNotEqual(before, codec.identity())
        self.assertEqual(before["settings"]["compression_profile"], "bato-v1")

    def test_archive_export_preserves_failures_and_settings(self):
        run = self.root / "run"
        run.mkdir()
        passing = dict(target="input", type="directory_archive", category="test", codec="zip", compressor="Python ZIP",
                       compression_profile="extreme", effective_settings={"entry_compresslevel": 9}, verified_lossless=True,
                       status="PASS & VERIFIED", raw_size=100, comp_size=50, ratio=2, comp_time_s=0.1, decomp_time_s=0.1,
                       comp_mb_s=1, decomp_mb_s=1, cache_origin="fresh")
        failure = {**passing, "codec": "rar", "status": "FAIL", "verified_lossless": False, "error": "corrupt archive"}
        export(run, [passing, failure], {"compression_profile": "extreme"})
        with (run / "archive_comparison.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["compression_profile"] == "extreme" for row in rows))
        self.assertEqual(json.loads(rows[0]["effective_settings"])["entry_compresslevel"], 9)
        with contextlib.redirect_stdout(io.StringIO()) as console:
            print_archive_summary([passing, failure])
        self.assertIn("Failures", console.getvalue())
        self.assertIn("n/a", console.getvalue())

    def integration(self, name):
        codec = create_codecs(None, None, name)[0]
        self.assertIsNone(codec.probe_error)
        before = snapshot(self.source)
        for source, directory in [(self.source / "nested space/日本語.bin", False), (self.source, True)]:
            workload = dict(target="input", path=str(source), type="directory_archive" if directory else "single_file",
                            category="test", origin="real", snapshot=snapshot(source))
            record = measure(codec, workload, 1, 0, 30)
            self.assertTrue(record["verified_lossless"], record.get("error"))
            self.assertEqual(record["compression_profile"], "extreme")
        self.assertEqual(snapshot(self.source), before)

    @unittest.skipUnless(shutil.which("rar"), "RAR is not installed")
    def test_rar_real_roundtrip(self):
        self.integration("rar")

    @unittest.skipUnless(shutil.which("kgb"), "KGB is not installed")
    def test_kgb_real_roundtrip(self):
        self.integration("kgb")

    @unittest.skipUnless(shutil.which("7zz") or shutil.which("7z"), "7-Zip is not installed")
    def test_7z_real_roundtrip(self):
        self.integration("7z")

    @unittest.skipUnless(shutil.which("lz4"), "LZ4 is not installed")
    def test_lz4_real_roundtrip(self):
        self.integration("lz4")


if __name__ == "__main__":
    unittest.main()
