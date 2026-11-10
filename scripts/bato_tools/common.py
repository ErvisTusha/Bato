"""Shared hashing, atomic persistence, archive handling, and process measurement."""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_NAMES = {".corpus.json", ".corpus.lock", "_corpus"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def regular_files(root, exclude_state=False):
    """Include hidden payloads such as .git, but never follow symlinks."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError(f"Symlink workloads are unsupported: {root}")
    if root.is_file():
        return [root]
    files = []
    for directory, subdirs, names in os.walk(root, followlinks=False):
        for name in list(subdirs):
            candidate = Path(directory) / name
            if candidate.is_symlink():
                raise ValueError(f"Symlink workloads are unsupported: {candidate}")
        subdirs[:] = sorted(name for name in subdirs if not (exclude_state and Path(directory) == root and name in STATE_NAMES))
        for name in sorted(names):
            if exclude_state and Path(directory) == root and (name in STATE_NAMES or name.endswith((".part", ".part.json"))):
                continue
            candidate = Path(directory) / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"Non-regular file workload: {candidate}")
            files.append(candidate)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def snapshot(path):
    path = Path(path)
    if path.is_symlink() or not path.exists():
        raise ValueError(f"Workload is missing or a symlink: {path}")
    if path.is_file():
        return {"size": path.stat().st_size, "sha256": sha256(path)}
    entries = [
        {"path": file.relative_to(path).as_posix(), "size": file.stat().st_size, "sha256": sha256(file)}
        for file in regular_files(path)
    ]
    directories = regular_directories(path)
    return {"size": sum(entry["size"] for entry in entries),
            "sha256": digest_json({"files": entries, "directories": directories}),
            "files": entries, "directories": directories}


def regular_directories(root):
    root = Path(root)
    result = []
    for parent, names, _ in os.walk(root, followlinks=False):
        for name in names:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError(f"Symlink directory: {path}")
            result.append(path.relative_to(root).as_posix())
    return sorted(result)


def safe_member(root, name):
    relative = PurePosixPath(name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or (relative.parts and ":" in relative.parts[0]):
        raise ValueError(f"Unsafe archive member: {name!r}")
    target = Path(root).joinpath(*relative.parts)
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Archive member escapes destination: {name!r}")
    return target


def make_tar(source, output, expected=None):
    source = Path(source)
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        for name in (expected["directories"] if expected is not None else regular_directories(source)):
            info = tarfile.TarInfo(name + "/")
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(info)
        for path in ([safe_member(source, entry["path"]) for entry in expected["files"]] if expected is not None else regular_files(source)):
            info = archive.gettarinfo(str(path), path.relative_to(source).as_posix())
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)


def extract_tar(source, output, max_bytes):
    Path(output).mkdir(parents=True, exist_ok=True)
    total = 0
    with tarfile.open(source, "r:*") as archive:
        for member in archive:
            target = safe_member(output, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member: {member.name}")
            total += member.size
            if total > max_bytes:
                raise ValueError("Archive exceeds its prepared-size budget")
            if target.exists():
                raise ValueError(f"Duplicate archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream, target.open("xb") as destination:
                copy_bounded(stream, destination, member.size)


def copy_bounded(source, destination, limit):
    total = 0
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        total += len(chunk)
        if total > limit:
            raise ValueError("Decoded data exceeds the expected size")
        destination.write(chunk)
    return total


@dataclass
class Measurement:
    wall_s: float
    peak_rss_kib: int | None
    cpu_s: float | None
    command: list[str]
    sampled_tree_peak_rss_kib: int | None = None
    container_time_s: float = 0.0
    input_bytes: int | None = None
    execution_stats: dict | None = None
    gpu_vram_peak_bytes: int | None = None
    energy_joules: float | None = None


def parse_execution_stats(stderr):
    """Read both the original Rust Debug stats and future extended snapshots."""
    lines = [line for line in stderr.splitlines() if line.startswith("execution stats: ")]
    if not lines:
        return None
    fields = re.findall(r'(\w+): ("(?:\\.|[^"\\])*"|\d+)', lines[-1])
    return {name: json.loads(value) for name, value in fields}


class ProcessFailure(RuntimeError):
    def __init__(self, command, diagnostic, timed_out=False, returncode=None):
        super().__init__(diagnostic)
        self.command = command
        self.timed_out = timed_out
        self.returncode = returncode


def bounded_reader(stream, result, key, limit=65536):
    """Drain a pipe while retaining only its diagnostic tail, never an unbounded log."""
    tail = bytearray()
    try:
        while block := stream.read(4096):
            tail.extend(block)
            if len(tail) > limit:
                del tail[:-limit]
    finally:
        result[key] = bytes(tail)
        stream.close()


def kill_group(process):
    try:
        group = getattr(process, 'benchmark_group', None)
        os.killpg(group if group is not None else os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def wait_process(process, timeout, progress=None, monitor=None):
    """Blocking reap for accurate short-job timings; an independent watchdog enforces deadlines."""
    done, expired = threading.Event(), threading.Event()
    started = time.monotonic()

    def supervise():
        next_progress = started + 30
        while not done.is_set():
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                expired.set()
                kill_group(process)
                return
            if monitor:
                monitor.sample(process.pid)
            if progress and time.monotonic() >= next_progress:
                progress(time.monotonic() - started)
                next_progress += 30
            if done.wait(min(remaining, 0.1 if monitor or progress else remaining)):
                return

    watchdog = threading.Thread(target=supervise, daemon=True)
    watchdog.start()
    try:
        process.wait()
    except BaseException:
        kill_group(process)
        process.wait()
        raise
    finally:
        done.set()
        watchdog.join()
    return expired.is_set()


def run_process(command, work, timeout=600, stdout_path=None, cwd=None, progress=None, monitor_tree=False, env=None):
    """Measure a child with bounded diagnostics, blocking wait, and group cancellation."""
    command = [str(part) for part in command]
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    usage = work / "usage.txt"
    usage.unlink(missing_ok=True)
    timed_command = command
    if Path("/usr/bin/time").is_file():
        timed_command = ["/usr/bin/time", "-f", "%M %U %S", "-o", str(usage), "--", *command]
    from .resources import ProcessTreeMonitor
    monitor = ProcessTreeMonitor() if monitor_tree else None
    output = open(stdout_path, "wb") if stdout_path is not None else subprocess.PIPE
    logs, readers = {}, []
    process = None
    own_group = not os.environ.get("BATO_BENCH_STAGE_WORKER")
    try:
        start = time.perf_counter()
        process = subprocess.Popen(timed_command, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.PIPE, cwd=cwd, env=env,
                                   start_new_session=own_group)
        if own_group:
            process.benchmark_group = process.pid
        for key, stream in [("stderr", process.stderr), ("stdout", process.stdout)]:
            if stream is not None:
                thread = threading.Thread(target=bounded_reader, args=(stream, logs, key), daemon=True)
                thread.start()
                readers.append(thread)
        timed_out = wait_process(process, timeout, progress, monitor)
        wall = time.perf_counter() - start
        if own_group:
            kill_group(process)
        for thread in readers:
            thread.join(timeout=2)
        if timed_out:
            raise ProcessFailure(command, f"Timed out after {timeout}s", timed_out=True)
        if process.returncode:
            message = "\n".join(logs.get(key, b"").decode("utf-8", errors="replace").strip() for key in ["stderr", "stdout"]).strip()
            raise ProcessFailure(command, message or f"Process exited {process.returncode}", returncode=process.returncode)
    finally:
        if process is not None and own_group:
            kill_group(process)
        for thread in readers:
            thread.join(timeout=2)
        if stdout_path is not None:
            output.close()
    peak = cpu = None
    if usage.exists():
        try:
            fields = usage.read_text().strip().splitlines()[-1].split()
            peak, user, system = int(fields[0]), float(fields[1]), float(fields[2])
            cpu = user + system
        except (ValueError, IndexError):
            pass
    return Measurement(wall, peak, cpu, command, monitor.peak_kib if monitor else None,
                       execution_stats=parse_execution_stats(logs.get("stderr", b"").decode("utf-8", errors="replace")))
