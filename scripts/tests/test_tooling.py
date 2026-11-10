"""Network-independent regressions for integrity, selection, measurement, and recovery."""
import contextlib
import hashlib
import http.client
import http.server
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_suite
import download_corpus
from bato_tools.benchmark import discover, measure, quick_sample, summarize
from bato_tools.catalog import asset
from bato_tools.codecs import Codec
from bato_tools.common import ProcessFailure, atomic_json, digest_json, run_process, snapshot
from bato_tools.corpus import prepare_corpus, read_index, select_assets, synthesize, unzip
from bato_tools.download import DownloadError, download
from bato_tools.store import MeasurementStore


class FixtureServer:
    def __enter__(self):
        owner = self
        self.content = b"abcdefgh" * 8192
        self.version = '"v1"'
        self.requests = []
        self.active = self.peak = self.slow_active = self.slow_peak = 0
        self.lock = threading.Lock()
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args):
                pass
            def do_GET(self):
                with owner.lock:
                    owner.requests.append((self.path, self.headers.get("Range"), self.headers.get("If-Range")))
                    owner.active += 1
                    owner.peak = max(owner.peak, owner.active)
                try:
                    if self.path.startswith("/redirect"):
                        self.send_response(302)
                        self.send_header("Location", owner.url + "/slow-redirected")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    if self.path == "/missing":
                        self.send_response(404)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    if self.path == "/transient" and sum(path == "/transient" for path, _, _ in owner.requests) == 1:
                        self.send_response(503)
                        self.send_header("Retry-After", "0")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    if self.path.startswith("/slow"):
                        with owner.lock:
                            owner.slow_active += 1
                            owner.slow_peak = max(owner.slow_peak, owner.slow_active)
                        time.sleep(0.07)
                    offset = 0
                    if self.headers.get("Range") and self.path != "/ignore-range" and self.headers.get("If-Range") == owner.version:
                        offset = int(self.headers["Range"].split("=")[1].rstrip("-"))
                    if offset >= len(owner.content):
                        self.send_response(416)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206 if offset else 200)
                    self.send_header("Content-Length", str(len(owner.content) - offset))
                    self.send_header("ETag", owner.version)
                    if offset:
                        self.send_header("Content-Range", f"bytes {offset}-{len(owner.content)-1}/{len(owner.content)}")
                    self.end_headers()
                    payload = owner.content[offset:]
                    if self.path == "/truncated":
                        payload = payload[:5]
                    self.wfile.write(payload)
                    self.wfile.flush()
                    if self.path == "/truncated":
                        self.close_connection = True
                finally:
                    with owner.lock:
                        owner.active -= 1
                        if self.path.startswith("/slow"):
                            owner.slow_active -= 1
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()


class DownloadTests(TemporaryTest):
    def partial(self, url, data, validator='"v1"', expected_size=None):
        destination = self.root / "file.bin"
        destination.with_name("file.bin.part").write_bytes(data)
        atomic_json(self.root / "file.bin.part.json", {"url": url, "validator": validator,
                    "expected_sha256": None, "expected_size": expected_size})
        return destination

    def test_truncated_response_is_not_published_and_preserves_old_file(self):
        with FixtureServer() as server:
            output = self.root / "file.bin"
            output.write_bytes(b"previous")
            with self.assertRaises(DownloadError):
                download(server.url + "/truncated", output, attempts=1, expected_size=len(server.content))
            self.assertEqual(output.read_bytes(), b"previous")
            self.assertEqual((self.root / "file.bin.part").stat().st_size, 5)

    def test_resume_and_ignored_range(self):
        with FixtureServer() as server:
            for endpoint in ["/normal", "/ignore-range"]:
                with self.subTest(endpoint=endpoint):
                    destination = self.partial(server.url + endpoint, server.content[:17])
                    download(server.url + endpoint, destination, attempts=1)
                    self.assertEqual(destination.read_bytes(), server.content)
                    self.assertEqual(server.requests[-1][1], "bytes=17-")

    def test_changed_validator_restarts(self):
        with FixtureServer() as server:
            destination = self.partial(server.url + "/normal", b"OLD", validator='"old"')
            download(server.url + "/normal", destination, attempts=1)
            self.assertEqual(destination.read_bytes(), server.content)

    def test_mirror_changes_restart_instead_of_splicing(self):
        with FixtureServer() as server:
            destination = self.partial(server.url + "/missing", b"OLD")
            download([server.url + "/missing", server.url + "/normal"], destination, attempts=1)
            self.assertEqual(destination.read_bytes(), server.content)
            self.assertIsNone(server.requests[-1][1])

    def test_checksum_failure_and_force_refresh(self):
        with FixtureServer() as server:
            destination = self.root / "file.bin"
            with self.assertRaises(DownloadError):
                download(server.url + "/normal", destination, attempts=1, expected_sha256="0" * 64)
            self.assertFalse(destination.exists())
            digest = hashlib.sha256(server.content).hexdigest()
            destination.write_bytes(server.content)
            download(server.url + "/normal", destination, expected_sha256=digest, attempts=1)
            before = len(server.requests)
            download(server.url + "/normal", destination, force=True, expected_sha256=digest, attempts=1)
            self.assertEqual(len(server.requests), before + 1)

    def test_transient_retry(self):
        with FixtureServer() as server:
            download(server.url + "/transient", self.root / "file.bin", attempts=2, sleep=lambda delay: None)
            self.assertEqual(len(server.requests), 2)

    def test_416_restarts_unverified_partial(self):
        with FixtureServer() as server:
            destination = self.partial(server.url + "/normal", server.content)
            download(server.url + "/normal", destination, attempts=2, sleep=lambda delay: None)
            self.assertEqual(destination.read_bytes(), server.content)
            self.assertIsNone(server.requests[-1][1])

    def test_partial_symlink_is_rejected_without_touching_target(self):
        marker = self.root / "marker"
        marker.write_bytes(b"preserve")
        (self.root / "file.bin.part").symlink_to(marker)
        with self.assertRaises(DownloadError):
            download("https://fixture/file", self.root / "file.bin", attempts=1)
        self.assertEqual(marker.read_bytes(), b"preserve")

    def test_incomplete_http_read_is_retried(self):
        with FixtureServer() as server:
            opener = __import__("urllib.request", fromlist=["build_opener"]).build_opener()
            real_open = opener.open
            with patch.object(opener, "open", side_effect=[http.client.IncompleteRead(b"", 8), real_open(server.url + "/normal")]):
                download(server.url + "/normal", self.root / "file.bin", opener=opener, attempts=2, sleep=lambda delay: None)
            self.assertEqual((self.root / "file.bin").read_bytes(), server.content)


class CorpusTests(TemporaryTest):
    def test_queue_enforces_per_host_limit_and_reuses_verified_assets(self):
        with FixtureServer() as server:
            assets = [asset(f"file-{number}", "test", server.url + f"/slow-{number}", f"real/file-{number}.bin", len(server.content)) for number in range(6)]
            with contextlib.redirect_stdout(io.StringIO()):
                report = prepare_corpus(self.root, assets, jobs=6, per_host=2)
                second = prepare_corpus(self.root, assets, jobs=6, per_host=2)
            self.assertFalse(report["failures"])
            self.assertEqual(server.peak, 2)
            self.assertEqual(len(server.requests), 6)
            self.assertEqual(len(second["reused"]), 6)
            self.assertEqual(len(read_index(self.root)["files"]), 6)

    def test_redirects_respect_destination_host_limit(self):
        with FixtureServer() as server:
            origin = server.url.replace("127.0.0.1", "localhost")
            assets = [asset(f"redirect-{n}", "test", (origin + f"/redirect-{n}" if n < 3 else server.url + f"/slow-{n}"),
                            f"real/file-{n}.bin", len(server.content)) for n in range(6)]
            with contextlib.redirect_stdout(io.StringIO()):
                report = prepare_corpus(self.root, assets, jobs=6, per_host=2)
            self.assertFalse(report["failures"])
            self.assertEqual(server.slow_peak, 2)

    def test_catalog_change_refreshes_asset(self):
        with FixtureServer() as server:
            item = asset("versioned", "test", server.url + "/normal", "real/file.bin", len(server.content), version="v1")
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_corpus(self.root, [item])
                server.content = b"new content"
                item["version"] = "v2"
                report = prepare_corpus(self.root, [item])
            self.assertFalse(report["failures"])
            self.assertEqual((self.root / "real/file.bin").read_bytes(), server.content)
            self.assertEqual(len(server.requests), 2)

    def test_force_refresh_preserves_unmanaged_files(self):
        with FixtureServer() as server:
            incoming = io.BytesIO()
            with zipfile.ZipFile(incoming, "w") as archive:
                archive.writestr("managed.txt", "managed")
            server.content = incoming.getvalue()
            item = asset("archive", "test", server.url + "/normal", "real/archive", 1000, archive="zip")
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_corpus(self.root, [item])
                marker = self.root / "real/archive/user.txt"
                marker.write_text("preserve")
                report = prepare_corpus(self.root, [item], force=True)
            self.assertTrue(report["failures"])
            self.assertEqual(marker.read_text(), "preserve")
            self.assertEqual(len(server.requests), 1)

    def test_synthetic_budget_is_enforced_before_payload_writes(self):
        with self.assertRaises(ValueError):
            synthesize(self.root, ["binary"], max_bytes=10)
        self.assertFalse((self.root / "synthetic").exists())

    def test_byte_budget_prevents_network(self):
        with FixtureServer() as server:
            item = asset("too-large", "test", server.url + "/normal", "real/file.bin", 100)
            with contextlib.redirect_stdout(io.StringIO()):
                report = prepare_corpus(self.root, [item], max_bytes=10)
            self.assertTrue(report["failures"])
            self.assertFalse(server.requests)

    def test_category_selection_is_precise(self):
        catalog = [asset("word", "word", "https://fixture/word", "word", 1), asset("excel", "excel", "https://fixture/excel", "excel", 1)]
        self.assertEqual([item["id"] for item in select_assets(catalog, ["excel"])], ["excel"])
        self.assertEqual(len(select_assets(catalog, ["office"])), 2)
        with self.assertRaises(ValueError):
            select_assets(catalog, ["typo"])

    def test_synthetic_cli_has_no_network_and_real_default_excludes_it(self):
        with patch("urllib.request.build_opener", side_effect=AssertionError("network call")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(download_corpus.main(["--synthetic", "--target", "binary", "-o", str(self.root)]), 0)
        self.assertFalse(discover(self.root)[0])
        self.assertEqual(len(discover(self.root, origins=["synthetic"])[0]), 3)

    def test_archive_traversal_and_expansion_budget(self):
        archive = self.root / "input.zip"
        for name, limit in [("../outside", 100), ("safe", 2)]:
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr(name, b"12345")
            with self.assertRaises(ValueError):
                unzip(archive, self.root / "out", limit)
        self.assertFalse((self.root.parent / "outside").exists())


class BenchmarkTests(TemporaryTest):
    def workload(self, data=b"good", directory=False):
        source = self.root / "input"
        if directory:
            source.mkdir()
            (source / "file.txt").write_bytes(data)
        else:
            source.write_bytes(data)
        return {"target": "input", "path": str(source), "type": "directory_archive" if directory else "single_file",
                "category": "test", "origin": "real", "snapshot": snapshot(source)}

    def test_same_size_directory_corruption_fails(self):
        workload = self.workload(directory=True)
        codec = Codec("bato-2", "fake fixture", sys.executable)
        def step(action, source, output, work, timeout, expected, directory):
            from bato_tools.common import Measurement
            if action == "compress":
                output.write_bytes(b"archive")
            else:
                output.mkdir()
                (output / "file.txt").write_bytes(b"evil")
            return Measurement(0.01, None, None, ["fixture"])
        with patch.object(codec, "workflow", side_effect=step):
            record = measure(codec, workload, 1, 0, 1)
        self.assertEqual(record["status"], "FAIL")
        self.assertFalse(record["verified_lossless"])

    def test_temporary_outputs_do_not_touch_source_neighbors(self):
        workload = self.workload()
        marker = self.root / "input.bato_l2"
        marker.write_bytes(b"preserve")
        codec = Codec("bato-2", "fixture", sys.executable)
        with patch.object(codec, "workflow", side_effect=ProcessFailure(["fixture"], "failure")):
            self.assertEqual(measure(codec, workload, 1, 0, 1)["status"], "FAIL")
        self.assertEqual(marker.read_bytes(), b"preserve")
        self.assertEqual(Path(workload["path"]).read_bytes(), b"good")

    def test_content_identity_detects_collisions_and_nested_edits(self):
        first = self.root / "a"
        second = self.root / "b"
        first.mkdir(); second.mkdir()
        (first / "same").write_bytes(b"foo")
        (second / "same").write_bytes(b"bar")
        self.assertNotEqual(snapshot(first)["sha256"], snapshot(second)["sha256"])
        before = snapshot(first)
        (first / "same").write_bytes(b"new")
        self.assertNotEqual(snapshot(first), before)

    def test_measurement_store_survives_reopen_without_aliasing(self):
        path = self.root / "measurements.sqlite"
        store = MeasurementStore(path)
        store.save_case("run", {"case_id": "case", "status": "PASS & VERIFIED"})
        record = store.cases("run")[0]; record["status"] = "modified"
        store.close()
        store = MeasurementStore(path)
        self.assertEqual(store.cases("run")[0]["status"], "PASS & VERIFIED")
        store.close()

    def test_recursive_discovery_deduplicates_and_rejects_index_drift(self):
        synthesize(self.root, ["text"])
        files, excluded = discover(self.root, origins=["synthetic"])
        self.assertEqual(len(files), 2)
        Path(files[0]["path"]).write_bytes(b"changed")
        after, excluded = discover(self.root, origins=["synthetic"])
        self.assertEqual(len(after), 1)
        self.assertTrue(any(row["reason"] == "invalid" for row in excluded))

    def test_quick_sampling_is_deterministic_and_bounded(self):
        workloads = [{"target": str(n), "category": str(n % 4), "snapshot": {"size": 100}} for n in range(100)]
        first = quick_sample(workloads, 42, max_files=10, max_bytes=500)
        self.assertEqual(first, quick_sample(workloads, 42, max_files=10, max_bytes=500))
        self.assertEqual(len(first), 5)

    def test_process_timeout_kills_child_group(self):
        with self.assertRaises(ProcessFailure) as result:
            run_process([sys.executable, "-c", "import time; time.sleep(30)"], self.root, timeout=0.05)
        self.assertTrue(result.exception.timed_out)

    def test_malformed_office_is_excluded(self):
        synthesize(self.root, ["text"])
        bad = self.root / "bad.xlsx"
        with zipfile.ZipFile(bad, "w") as stream:
            stream.writestr("xl/workbook.xml", "<workbook r:id='x'/>")
        _, excluded = discover(self.root, include_local=True)
        self.assertTrue(any(row["target"] == "bad.xlsx" and row["reason"] == "invalid" for row in excluded))

    def test_missing_codec_does_not_empty_comparable_workloads(self):
        passing = dict(type="single_file", category="text", codec="zip", status="PASS & VERIFIED", target="a",
                       verified_lossless=True, raw_size=10, comp_size=5, comp_time_s=0.1, decomp_time_s=0.1)
        skipped = {**passing, "codec": "missing", "status": "SKIPPED", "verified_lossless": False}
        rows = summarize([passing, skipped])
        self.assertEqual(next(row for row in rows if row["codec"] == "zip")["common_workloads"], 1)

    def test_cli_fresh_runs_resume_and_rigorous_trials(self):
        corpus, output = self.root / "corpus", self.root / "results"
        synthesize(corpus, ["text"])
        # Exercise fresh-run/recovery/trial policy independently of volatile host
        # headroom and Python thread address-space reservations.
        args = ["-c", str(corpus), "--origin", "synthetic", "--codecs", "zip", "--workload", "files", "--output-dir", str(output), "--database", str(output / "measurements.sqlite3"),
                "--memory-budget", str(2**31)]
        with patch("bato_tools.codecs.resolve_bato", side_effect=AssertionError("ZIP must not require Rust")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(benchmark_suite.main(args), 0)
            self.assertEqual(benchmark_suite.main(args), 0)
            latest = json.loads((output / "latest.json").read_text())
            run = Path(latest["path"])
            records = json.loads((run / "results.json").read_text())
            self.assertEqual({row["cache_origin"] for row in records}, {"fresh"})
            journal = (run / "results.jsonl").read_text()
            self.assertEqual(benchmark_suite.main(args + ["--resume", latest["run_id"]]), 0)
            self.assertEqual((run / "results.jsonl").read_text(), journal)
            self.assertEqual(benchmark_suite.main(args + ["--mode", "rigorous"]), 0)
        run = Path(json.loads((output / "latest.json").read_text())["path"])
        records = json.loads((run / "results.json").read_text())
        self.assertEqual({row["cache_origin"] for row in records}, {"fresh"})
        self.assertTrue(all(row["verified_lossless"] and row["repetitions"] == 5 and row["warmups"] == 1 and len(row["trials"]) == 5 for row in records))


if __name__ == "__main__":
    unittest.main()
