"""Concurrent preparation, provenance, size budgets, and atomic corpus indexes."""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import lzma
import os
import queue
import random
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse
import urllib.error
import urllib.request

from .catalog import ALIASES, default_catalog
from .common import atomic_json, copy_bounded, digest_json, extract_tar, regular_files, run_process, safe_member, sha256, utc_now
from .download import download
from .validation import validate_file

DEFAULT_BUDGET = 10_000_000_000


class LimitedResponse:
    def __init__(self, response, semaphore):
        self.response, self.semaphore = response, semaphore

    def __getattr__(self, name):
        return getattr(self.response, name)

    def __enter__(self):
        return self.response.__enter__()

    def __exit__(self, *args):
        try:
            return self.response.__exit__(*args)
        finally:
            self.semaphore.release()


class LimitedOpener:
    def __init__(self, hosts, lock):
        self.hosts, self.lock = hosts, lock
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        self.opener = urllib.request.build_opener(NoRedirect())

    def open(self, request, timeout):
        # Release the original host before acquiring a redirected host's slot.
        for _ in range(11):
            with self.lock:
                semaphore = self.hosts[urlparse(request.full_url).netloc]
            semaphore.acquire()
            try:
                return LimitedResponse(self.opener.open(request, timeout=timeout), semaphore)
            except urllib.error.HTTPError as error:
                semaphore.release()
                if error.code not in {301, 302, 303, 307, 308} or not error.headers.get("Location"):
                    raise
                location = urljoin(request.full_url, error.headers["Location"])
                error.close()
                if urlparse(location).scheme not in {"http", "https"}:
                    raise ValueError("Redirect must use HTTP(S)")
                request = urllib.request.Request(location, headers=dict(request.header_items()))
            except BaseException:
                semaphore.release()
                raise
        raise urllib.error.URLError("Too many HTTP redirects")


def read_index(root):
    path = Path(root) / ".corpus.json"
    if not path.exists():
        return {"schema_version": 1, "files": {}, "assets": {}}
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or not isinstance(value.get("files"), dict) or not isinstance(value.get("assets"), dict):
        raise ValueError("Unsupported corpus index")
    return value


def load_catalog(manifests=()):
    catalog = default_catalog()
    for path in manifests:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("assets"), list):
            raise ValueError("Manifests require schema_version=1 and an assets array")
        catalog.extend(value["assets"])
    ids, destinations = set(), []
    root = Path.cwd() / "manifest-root"
    for item in catalog:
        for field in ["id", "category", "urls", "destination", "prepared_bytes", "transfer_bytes"]:
            if field not in item:
                raise ValueError(f"Asset missing {field}: {item.get('id')}")
        if item["id"] in ids:
            raise ValueError(f"Duplicate asset id: {item['id']}")
        ids.add(item["id"])
        relative = safe_member(root, item["destination"]).relative_to(root)
        if not relative.parts or relative.parts[0] in {"_corpus", ".corpus.json", ".corpus.lock"}:
            raise ValueError("Asset destination overlaps corpus state")
        for other in destinations:
            if relative == other or relative.is_relative_to(other) or other.is_relative_to(relative):
                raise ValueError(f"Overlapping destinations: {relative} and {other}")
        destinations.append(relative)
        if not isinstance(item["urls"], list) or not item["urls"] or not all(isinstance(url, str) and url.startswith(("https://", "http://")) for url in item["urls"]):
            raise ValueError("Asset URLs must be a nonempty HTTP(S) array")
        if any(not isinstance(item[key], int) or item[key] <= 0 for key in ["prepared_bytes", "transfer_bytes"]):
            raise ValueError("Asset budgets must be positive integers")
        if item.get("archive") not in {None, "zip", "tar", "xz"} or item.get("kind") not in {None, "git"}:
            raise ValueError("Unsupported asset preparation method")
        if item.get("origin", "real") not in {"real", "synthetic", "local"}:
            raise ValueError("Invalid asset origin")
        digest = item.get("sha256")
        if digest is not None and (len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
    return catalog


def select_assets(catalog, targets=(), tier="standard", all_suites=False):
    categories = {item["category"] for item in catalog}
    chosen = set()
    for group in targets:
        for target in group.lower().split(","):
            target = target.strip()
            matched = ALIASES.get(target, {target}) & categories
            if not matched:
                raise ValueError(f"Unknown or unavailable target {target!r}; categories: {', '.join(sorted(categories))}")
            chosen.update(matched)
    return [item for item in catalog if (item["category"] in chosen if targets else
            all_suites or tier != "standard" or item.get("tier", "standard") == "standard")]


def unzip(source, destination, max_bytes):
    total = 0
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            path = safe_member(destination, member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlinks are unsupported")
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            total += member.file_size
            if total > max_bytes:
                raise ValueError("Archive exceeds its prepared-size budget")
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as incoming, path.open("xb") as output:
                if copy_bounded(incoming, output, member.file_size) != member.file_size:
                    raise ValueError("Truncated ZIP member")


def asset_valid(root, index, item):
    record = index["assets"].get(item["id"])
    if not record or not record.get("files") or record.get("catalog_sha256") != digest_json(item):
        return False
    for name in record["files"]:
        metadata = index["files"].get(name, {})
        path = safe_member(root, name)
        if not path.is_file() or path.is_symlink() or path.stat().st_size != metadata.get("size") or sha256(path) != metadata.get("sha256"):
            return False
    return True


def prepare_asset(root, item, force, hosts, host_lock, extraction_slots, events, stop):
    destination = safe_member(root, item["destination"])
    identifier = hashlib.sha256(item["id"].encode()).hexdigest()[:24]
    state = root / "_corpus"
    state.mkdir(exist_ok=True)
    with host_lock:
        limiter = hosts[urlparse(item["urls"][0]).netloc]
    def progress(current, total, speed):
        events.put((item["id"], current, total, speed))
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=state) as temporary:
        work = Path(temporary)
        prepared = work / ("payload" if item.get("archive") or item.get("kind") == "git" else destination.name)
        if item.get("kind") == "git":
            with limiter:
                run_process(["git", "clone", "--depth", str(item.get("depth", 100)), "--branch", item["revision"],
                             "--", item["urls"][0], prepared], work, timeout=300)
            result = subprocess.run(["git", "-C", str(prepared), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True)
            source_metadata = {"url": item["urls"][0], "revision": result.stdout.strip(), "depth": item.get("depth", 100)}
        else:
            cached = state / "downloads" / identifier
            source_metadata = download(item["urls"], cached, expected_size=item.get("expected_size"),
                expected_sha256=item.get("sha256"), force=force, progress=progress,
                max_bytes=item["transfer_bytes"], stop=stop, opener=LimitedOpener(hosts, host_lock),
                validate=(lambda path: validate_file(path, item.get("validation", destination.suffix.lstrip(".")))) if not item.get("archive") else None)
            with extraction_slots:
                if item.get("archive") == "zip":
                    prepared.mkdir()
                    unzip(cached, prepared, item["prepared_bytes"])
                elif item.get("archive") == "tar":
                    extract_tar(cached, prepared, item["prepared_bytes"])
                elif item.get("archive") == "xz":
                    with lzma.open(cached, "rb") as incoming, prepared.open("wb") as output:
                        copy_bounded(incoming, output, item["prepared_bytes"])
                else:
                    shutil.copyfile(cached, prepared)
        files = regular_files(prepared)
        if not files:
            raise ValueError("Asset produced no files")
        total = sum(path.stat().st_size for path in files)
        if total > item["prepared_bytes"]:
            raise ValueError("Prepared asset exceeds its declared budget")
        records = {}
        for path in files:
            final = destination / path.relative_to(prepared) if prepared.is_dir() else destination
            name = final.relative_to(root).as_posix()
            validation = validate_file(path, item.get("validation", "auto")) if not item.get("kind") else {"validation": "passed", "validation_kind": "git", "validation_scope": "git-clone"}
            records[name] = {"size": path.stat().st_size, "sha256": sha256(path), "category": item["category"],
                "origin": item.get("origin", "real"), "asset": item["id"], "version": item.get("version", item["urls"][0]), "source": source_metadata, **validation}
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            backup = work / "previous"
            os.replace(destination, backup)
            try:
                os.replace(prepared, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
        else:
            os.replace(prepared, destination)
        if item.get("kind") != "git":
            cached.unlink(missing_ok=True)
        return records, {"files": list(records), "prepared_bytes": total, "source": source_metadata,
            "version": item.get("version", item["urls"][0]), "catalog_sha256": digest_json(item), "completed_at": utc_now()}


def prepare_corpus(root, assets, *, jobs=8, max_bytes=DEFAULT_BUDGET, force=False, per_host=2, extraction_jobs=2):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".corpus.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another corpus manager is using this destination") from error
        index = read_index(root)
        existing_bytes = sum(path.stat().st_size for path in regular_files(root, exclude_state=True))
        reserved = existing_bytes
        selected, failures, reused = [], [], []
        for item in assets:
            if not force and asset_valid(root, index, item):
                reused.append(item["id"])
                continue
            destination = safe_member(root, item["destination"])
            if destination.exists() and item["id"] not in index["assets"]:
                failures.append({"asset": item["id"], "error": "Unmanaged destination already exists; choose another destination"})
                continue
            managed_names = set(index["assets"].get(item["id"], {}).get("files", []))
            if destination.exists() and any(path.relative_to(root).as_posix() not in managed_names for path in regular_files(destination)):
                failures.append({"asset": item["id"], "error": "Destination contains unmanaged files; refresh would remove them"})
                continue
            prior = sum(path.stat().st_size for path in regular_files(destination)) if destination.exists() else 0
            extra = max(0, item["prepared_bytes"] - prior)
            if reserved + extra > max_bytes:
                failures.append({"asset": item["id"], "error": "Prepared-corpus budget exceeded"})
                continue
            reserved += extra
            selected.append(item)
        scratch = sum(sorted((item["transfer_bytes"] + item["prepared_bytes"] for item in selected), reverse=True)[:jobs])
        if shutil.disk_usage(root).free < scratch + reserved - existing_bytes:
            raise ValueError("Insufficient staging space; lower --jobs")
        hosts = defaultdict(lambda: threading.BoundedSemaphore(per_host))
        host_lock = threading.Lock()
        extraction_slots = threading.BoundedSemaphore(extraction_jobs)
        events, stop = queue.Queue(), threading.Event()
        live, completed = {}, len(reused)
        print(f"Selected {len(assets)} assets; {len(reused)} verified existing; downloads={jobs}, per-host={per_host}, extractors={extraction_jobs}")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
        futures = {executor.submit(prepare_asset, root, item, force, hosts, host_lock, extraction_slots, events, stop): item for item in selected}
        last_update = 0.0
        try:
            while futures:
                done, _ = concurrent.futures.wait(futures, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED)
                while not events.empty():
                    identifier, current, total, speed = events.get_nowait()
                    live[identifier] = (current, total, speed)
                for future in done:
                    item = futures.pop(future)
                    live.pop(item["id"], None)
                    try:
                        records, asset_record = future.result()
                        for name in index["assets"].get(item["id"], {}).get("files", []):
                            index["files"].pop(name, None)
                        index["files"].update(records)
                        index["assets"][item["id"]] = asset_record
                        atomic_json(root / ".corpus.json", index)
                        completed += 1
                        print(f"[PASS] {item['id']}: {len(records)} files, {asset_record['prepared_bytes'] / 2**20:.2f} MiB", flush=True)
                    except Exception as error:
                        failures.append({"asset": item["id"], "error": str(error)})
                        print(f"[FAIL] {item['id']}: {error}", flush=True)
                if live and time.monotonic() - last_update >= 1:
                    speed = sum(value[2] for value in live.values())
                    remaining = sum(max(0, (total or current) - current) for current, total, _ in live.values())
                    eta = f"{remaining / speed:.0f}s" if speed and all(total for _, total, _ in live.values()) else "unknown"
                    print(f"[PROGRESS] {completed}/{len(assets)} complete, {len(live)} transferring, {speed / 2**20:.2f} MiB/s, active ETA {eta}", flush=True)
                    last_update = time.monotonic()
        except KeyboardInterrupt:
            stop.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        report = {"completed": completed, "selected": len(assets), "failures": failures, "reused": reused, "finished_at": utc_now()}
        atomic_json(root / "_corpus" / "last_run.json", report)
        return report


def _audit_existing(root, adopt=False):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Corpus directory does not exist")
    index = read_index(root)
    records, invalid, duplicates, seen = {}, [], [], {}
    for path in regular_files(root, exclude_state=True):
        name = path.relative_to(root).as_posix()
        try:
            validation, digest = validate_file(path), sha256(path)
            if digest in seen:
                duplicates.append({"path": name, "same_as": seen[digest]})
            else:
                seen[digest] = name
            records[name] = {"size": path.stat().st_size, "sha256": digest, "category": path.parent.name,
                            "origin": "local", "source": {"path": name}, **validation}
        except Exception as error:
            invalid.append({"path": name, "error": str(error)})
    if adopt:
        for name, record in records.items():
            if name not in index["files"]:
                index["files"][name] = record
        atomic_json(root / ".corpus.json", index)
    return {"valid_files": len(records), "invalid": invalid, "duplicates": duplicates, "indexed": len(index["files"]), "adopted": adopt}


def _synthesize(root, targets=(), max_bytes=DEFAULT_BUDGET):
    root = Path(root).resolve()
    allowed = {"text", "structured", "binary", "images", "audio"}
    selected = {name for group in targets for name in group.split(",")} if targets else allowed
    if not selected <= allowed:
        raise ValueError(f"Synthetic categories: {', '.join(sorted(allowed))}")
    root.mkdir(parents=True, exist_ok=True)
    index = read_index(root)
    generators = {}
    if "text" in selected:
        generators["text/prose.txt"] = ("Compression tooling should verify every measured result.\n" * 4096).encode()
        generators["text/unicode.txt"] = ("Tiranë — Ελληνικά — 日本語\n" * 2048).encode()
    if "structured" in selected:
        generators["structured/telemetry.json"] = json.dumps([{"id": n, "temperature": (n * 17 % 101) / 10, "active": n % 3 != 0} for n in range(10000)]).encode()
        generators["structured/measurements.csv"] = ("id,value\n" + "".join(f"{n},{n * 37 % 10007}\n" for n in range(10000))).encode()
    if "binary" in selected:
        generators["binary/random.bin"] = random.Random(42).randbytes(1024 * 1024)
        generators["binary/zeros.bin"] = bytes(1024 * 1024)
        generators["binary/stride.dat"] = b"".join(struct.pack("<d", n * 0.125) for n in range(131072))
    if "images" in selected:
        def chunk(kind, payload):
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
        raw = b"".join(b"\0" + bytes((x ^ y) % 256 for x in range(256)) for y in range(256))
        generators["images/gradient.png"] = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    if "audio" in selected:
        payload = b"".join(struct.pack("<h", (n % 100) * 300 - 15000) for n in range(44100))
        generators["audio/tone.wav"] = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 44100, 88200, 2, 16) + b"data" + struct.pack("<I", len(payload)) + payload
    existing_bytes = sum(path.stat().st_size for path in regular_files(root, exclude_state=True))
    additional = sum(max(0, len(content) - ((root / "synthetic" / name).stat().st_size if (root / "synthetic" / name).exists() else 0)) for name, content in generators.items())
    if existing_bytes + additional > max_bytes:
        raise ValueError("Synthetic corpus exceeds --max-bytes")
    for name, content in generators.items():
        relative = "synthetic/" + name
        path = root / relative
        if path.exists() and (relative not in index["files"] or index["files"][relative].get("origin") != "synthetic"):
            raise ValueError(f"Unmanaged synthetic destination exists: {relative}")
        if path.is_file() and sha256(path) == hashlib.sha256(content).hexdigest():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            output.write(content)
            temporary = Path(output.name)
        try:
            validation = validate_file(temporary, path.suffix.lstrip("."))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        index["files"][relative] = {"size": len(content), "sha256": sha256(path), "category": name.split("/")[0],
                                  "origin": "synthetic", "source": {"generator": "bato-stress-v1", "seed": 42}, **validation}
    atomic_json(root / ".corpus.json", index)
    return {"completed": len(generators), "failures": [], "origin": "synthetic"}


def _with_write_lock(root, function, *args, **kwargs):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".corpus.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Corpus is currently being prepared or benchmarked") from error
        return function(root, *args, **kwargs)


def synthesize(root, targets=(), max_bytes=DEFAULT_BUDGET):
    return _with_write_lock(root, _synthesize, targets, max_bytes)


def audit_existing(root, adopt=False):
    if adopt:
        if not Path(root).is_dir():
            raise ValueError("Corpus directory does not exist")
        return _with_write_lock(root, _audit_existing, True)
    return _audit_existing(root)
