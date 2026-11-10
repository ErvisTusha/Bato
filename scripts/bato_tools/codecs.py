"""Real command adapters with explicit binary identity, options, and thread counts."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import zlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .common import PROJECT_ROOT, ProcessFailure, bounded_reader, extract_tar, kill_group, make_tar, regular_files, run_process, safe_member, sha256, snapshot, utc_now, wait_process

WORKER = Path(__file__).with_name("codec_worker.py")
DEFAULT_CODECS = ("bato", "kgb", "paq8px", "paq8l", "zpaq", "rar", "7z", "xz", "zstd", "gzip", "lz4", "zip", "bzip2")
OPTIONAL_CODECS = ("cmix", "nncp", "tar", "brotli")  # Explicit opt-in: high memory and very slow models.
SUPPORTED_CODECS = DEFAULT_CODECS + OPTIONAL_CODECS
PROFILE_LEVELS = {
    "tar": (0, 0, 0), "brotli": (1, 6, 11), "7z": (1, 5, 9), "rar": (1, 3, 5), "kgb": (1, 5, 9), "gzip": (1, 6, 9),
    "paq8l": (1, 5, 8), "paq8px": (1, 5, 12), "zpaq": (1, 3, 5), "xz": (1, 6, 9),
    "cmix": (0, 0, 0), "nncp": (0, 0, 0),  # No universal fast/default/extreme presets in these CLIs.
    "zstd": (1, 3, 22), "lz4": (1, 1, 12), "zip": (1, 6, 9), "bzip2": (1, 9, 9),
}


def validate_tuning(tuning):
    """Only compression parameters may vary; retain worker and framing controls."""
    if not isinstance(tuning, dict) or set(tuning) - {"zip", "7z", "rar", "xz"}:
        raise ValueError("Codec tuning must map zip, 7z, rar, or xz to settings")
    patterns = {
        "zip": r"-m(?:m|fb|pass|d|mem|o)=[A-Za-z0-9:.=+-]+",
        "7z": r"-m(?:0|d|fb|mc|yx|qs|f)=[A-Za-z0-9:.=+-]+",
        "rar": r"(?:-md[0-9]+[kmg]?|-mc[lx][+-]?)",
        "xz": r"--lzma2=[A-Za-z0-9,=+-]+",
    }
    for name, config in tuning.items():
        if not isinstance(config, dict) or set(config) - {"backend", "parameters"}:
            raise ValueError(f"Invalid tuning settings for {name}")
        backend = config.get("backend", "native")
        if not isinstance(backend, str) or backend not in {"native", "7z"} or (name != "zip" and "backend" in config):
            raise ValueError("Only ZIP can select the 7z backend")
        parameters = config.get("parameters", [])
        if not isinstance(parameters, list) or any(not isinstance(p, str) or not re.fullmatch(patterns[name], p) for p in parameters):
            raise ValueError(f"Unsupported compression parameter for {name}; worker, path, and archive options cannot be overridden")
        if name == "7z" and any(p.startswith("-m0=") and not re.fullmatch(
                r"-m0=(?:LZMA2?|PPMd|BZip2)(?::(?:a|d|mf|fb|mc|lc|lp|pb|c|mem|o)=[A-Za-z0-9]+)*", p)
                for p in parameters):
            raise ValueError("Unsupported 7z method parameters")
        if name == "zip" and parameters and backend != "7z":
            raise ValueError("ZIP tuning parameters require backend 7z")
    return tuning


def probe_command(command, timeout=5):
    """Capture bounded help/version text, allowing documented nonzero help exits."""
    command = list(map(str, command))
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    process.benchmark_group = process.pid
    logs = {}
    reader = threading.Thread(target=bounded_reader, args=(process.stdout, logs, 'output', 8192), daemon=True)
    reader.start()
    try:
        expired = wait_process(process, timeout)
    finally:
        kill_group(process)
        reader.join(timeout=2)
    if expired:
        raise ProcessFailure(command, f"Tool probe timed out after {timeout}s", timed_out=True)
    return process.returncode, logs.get('output', b'').decode("utf-8", errors="replace")


def selected_names(requested=None, only_bato=False):
    names = ["bato"] if only_bato else ([name.strip().lower() for name in requested.split(",")] if requested else list(DEFAULT_CODECS))
    if not names or not set(names) <= set(SUPPORTED_CODECS):
        raise ValueError(f"Supported codecs: {', '.join(SUPPORTED_CODECS)}")
    return list(dict.fromkeys(names))


def resolve_bato(explicit=None):
    candidates = [Path(explicit)] if explicit else [PROJECT_ROOT / "bin/bato", PROJECT_ROOT / "target/release/bato", PROJECT_ROOT / "target/x86_64-unknown-linux-musl/release/bato"]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            path = path.resolve()
            try:
                returncode, help_text = probe_command([path, "--help"])
            except (OSError, ProcessFailure):
                if explicit:
                    raise
                continue
            if returncode or "Pure Rust Bato" not in help_text or any(option not in help_text for option in ("--segment-size", "--chunk-size")):
                if explicit:
                    raise ValueError(f"Expected a Bato binary with --segment-size and --chunk-size options: {path}; run cargo build --release.")
                continue
            returncode, version = probe_command([path, "--version"])
            if returncode or not re.fullmatch(r"bato \d+\.\d+\.\d+(?:[-+][^ ]+)? \(format \d+\.\d+\)", version.strip()):
                if explicit:
                    raise ValueError(f"Rust Bato v1 version probe failed: {path}")
                continue
            return path, version.strip()
    raise ValueError("Rust Bato binary not found; run cargo build --release")


@dataclass
class Codec:
    identifier: str
    name: str
    program: str | None
    version: str = ""
    probe_error: str | None = None
    lz4_workers: bool = False
    profile: str = "high"
    threads: int = 1
    memory_budget: int = 256 * 2**20
    python_worker: bool = True
    extract_program: str | None = None
    chunk_size: int = 256 * 1024
    zip_backend: str = "native"
    compression_parameters: list[str] = field(default_factory=list)
    nncp_profile: str = "default"
    nncp_cuda: bool = False
    segment_size: int = 64 * 2**20

    @property
    def level(self):
        if self.identifier.startswith("bato"):
            return 0
        return PROFILE_LEVELS[self.identifier][("fast", "default", "high").index(self.profile)]

    def compression_options(self, directory=False):
        if self.identifier.startswith("bato"):
            return ["--segment-size", str(self.segment_size),
                    "--chunk-size", str(self.chunk_size), "--threads", str(self.threads),
                    "--memory-budget", str(self.memory_budget), "--stats"]
        # CMIX has no universal compression level; NNCP profiles are explicitly
        # workload-specific rather than a truthful equivalent to PAQ -12L.
        if self.identifier == "cmix":
            return []
        if self.identifier == "nncp":
            return ["-T", str(self.threads), "--profile", self.nncp_profile, *(["--cuda"] if self.nncp_cuda else [])]
        high = self.profile == "high"
        level = self.level
        if self.identifier == "zip" and self.zip_backend == "7z":
            return ["-tzip", f"-mx={level}", f"-mmt={self.threads}", "-mtm=off", "-mtc=off", "-mta=off", "-y", *self.compression_parameters]
        options = {
            "7z": ["-t7z", f"-mx={level}", f"-mmt={self.threads}", "-mtm=off", "-mtc=off", "-mta=off", "-y", *(["-ms=on"] if directory else [])],
            "tar": [], "brotli": ["-q", str(level), "--large_window=29", "-c"],
            "rar": ["-ma5", "-cfg-", f"-m{level}", f"-mt{self.threads}", "-ts-", "-y", *(["-s", "-r"] if directory else [])],
            "paq8l": [f"-{level}"],
            "paq8px": [f"-{level}" + ("L" if high else ""), *(["-lstmlayers=5"] if high else [])],
            "zpaq": ["-method", str(level), "-threads", str(self.threads), "-noattributes"],
            "kgb": [f"-{level}"], "gzip": ["-n", f"-{level}", "-c"],
            "xz": [f"-T{self.threads}", f"-{level}", *(["--extreme"] if high else []), "--no-adjust", "-c"],
            "zstd": ["-q", *(["--single-thread"] if self.threads == 1 else [f"-T{self.threads}"]), *(["--ultra"] if high else []), f"-{level}", "-c"],
            "lz4": ["-q", f"-{level}", *([f"-T{self.threads}"] if self.lz4_workers else []), "-c"],
            "zip": ["-q", "-X", f"-{level}"] if not self.python_worker else ["DEFLATE", f"compresslevel={level}", f"entry_compresslevel={level}"],
            "bzip2": [f"-{level}", "-c"] if not self.python_worker else [f"compresslevel={level}"],
        }[self.identifier]
        return options + self.compression_parameters

    def settings(self):
        bato = self.identifier.startswith("bato")
        return {"compression_profile": "bato-v1" if bato else "extreme" if self.profile == "high" else self.profile,
                "benchmark_profile": self.profile, "level": self.level, "threads": self.threads,
                "metadata_policy": "normalized-v1",
                "decompression_workers": self.threads if bato or self.identifier in {"xz", "zstd", "7z", "rar", "zpaq", "nncp"} or self.lz4_workers or self.zip_backend == "7z" else 1,
                "dictionary": "explicit_parameters" if self.compression_parameters else "preset", "file_options": self.compression_options(),
                "directory_options": self.compression_options(True),
                "directory_container": "native" if bato or self.identifier in {"zip", "rar", "7z", "kgb"} else "tar",
                "backend": "bato-v1" if bato else "7z-zip" if self.identifier == "zip" and self.zip_backend == "7z" else self.identifier,
                "compute": ("cuda" if self.nncp_cuda else "cpu") if self.identifier == "nncp" else None,
                "nncp_profile": self.nncp_profile if self.identifier == "nncp" else None,
                "chunk_size": self.chunk_size if bato else None,
                "segment_size": self.segment_size if bato else None,
                "memory_budget": self.memory_budget if bato else None,
                "checksums": True if bato else None}

    def archive_suffix(self, directory=False):
        if self.identifier.startswith("bato"):
            return ".bato"
        if self.identifier == "paq8px":
            match = re.search(r"\bpaq8px(?: data compressor)? v(\d+)", self.version, re.I)
            if not match:
                raise ValueError("PAQ8PX versioned archive suffix unavailable")
            return ".paq8px" + match.group(1)
        return {"tar": ".tar", "brotli": ".br", "paq8l": ".paq8l", "zpaq": ".zpaq", "cmix": ".cmix", "nncp": ".nncp", "7z": ".7z", "rar": ".rar", "kgb": ".kgb", "gzip": ".gz", "xz": ".xz",
                "zstd": ".zst", "lz4": ".lz4", "zip": ".zip", "bzip2": ".bz2"}[self.identifier]

    def identity(self):
        return {"id": self.identifier, "name": self.name, "binary": self.program, "binary_sha256": sha256(self.program) if self.program else None,
                "level": self.level, "version": self.version, "threads": self.threads, "settings": self.settings(),
                "worker_sha256": sha256(WORKER) if self.identifier in {"zip", "bzip2"} and self.python_worker else None,
                "extract_binary": self.extract_program,
                "extract_binary_sha256": sha256(self.extract_program) if self.extract_program else None}

    def step(self, action, source, output, work, timeout, size, directory=False):
        source, output, work = Path(source), Path(output), Path(work)
        if self.identifier == "tar":
            if action == "compress":
                return run_process([self.program, "--format=ustar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0", "--numeric-owner", "-cf", output, "-C", source if directory else source.parent, "." if directory else source.name], work, timeout)
            extraction = output if directory else work / "tar-extracted"
            extraction.mkdir(parents=True, exist_ok=True)
            measured = run_process([self.program, "-xf", source, "-C", extraction], work, timeout)
            if not directory:
                self.move_single(extraction, output)
            return measured
        if self.identifier.startswith("bato"):
            if action == "compress":
                cmd = [self.program, *(["archive"] if directory else []), source, *self.compression_options(directory), "-o", output, "-f"]
            else:
                cmd = [self.program, "extract" if directory else "-d", source, "-o", output, "-f", *self.compression_options(directory)]
            return run_process(cmd, work, timeout)
        if self.identifier == "paq8l":
            if action == "compress":
                # The first (nonexistent) argument sets the archive basename;
                # PAQ8L appends .paq8l and archives the following real input.
                return run_process([self.program, *self.compression_options(), output.with_suffix(""),
                                    "./" + source.name], work, timeout, cwd=source.parent)
            extraction = work / "paq8l-extracted"
            extraction.mkdir(parents=True, exist_ok=True)
            measured = run_process([self.program, "-d", source, extraction], work, timeout)
            self.move_single(extraction, output)
            return measured
        if self.identifier == "paq8px":
            command = ([self.program, *self.compression_options(), "./" + source.name, output]
                       if action == "compress" else [self.program, "-d", source, output])
            feedback = work / "paq-feedback.txt"
            measured = run_process(command, work, timeout, cwd=source.parent if action == "compress" else None,
                                   stdout_path=feedback)
            # PAQ8PX intentionally exits zero on allocation failure and leaves
            # a partial archive. Never accept its exit status alone as success.
            with feedback.open("rb") as stream:
                prefix = stream.read(8192)
                stream.seek(max(0, feedback.stat().st_size - 8192))
                text = (prefix + b"\n" + stream.read(8192)).decode("utf-8", errors="replace")
            if any(line.strip() == "Out of memory." for line in text.splitlines()):
                raise MemoryError("PAQ8PX reports Out of memory at the requested settings")
            return measured
        if self.identifier == "zpaq":
            if action == "compress":
                return run_process([self.program, "add", output, "./" + source.name, *self.compression_options()],
                                   work, timeout, cwd=source.parent)
            extraction = work / "zpaq-extracted"
            extraction.mkdir(parents=True, exist_ok=True)
            measured = run_process([self.program, "extract", source, "-to", extraction,
                                    "-threads", str(self.threads), "-force"], work, timeout)
            self.move_single(extraction, output)
            return measured
        if self.identifier in {"cmix", "nncp"}:
            # Both programs accept individual streams, never directory paths.
            # workflow() creates and verifies a deterministic tar for folders.
            if self.identifier == "cmix":
                command = [self.program, "-c" if action == "compress" else "-d", source, output]
            else:
                command = [self.program, *self.compression_options(), "c" if action == "compress" else "d", source, output]
            return run_process(command, work, timeout)
        if self.identifier == "zip" and self.zip_backend == "7z":
            if action == "compress":
                return run_process([self.program, "a", *self.compression_options(directory), output, "--",
                                    "." if directory else "./" + source.name], work, timeout, cwd=source if directory else source.parent)
            extraction = output if directory else work / "zip-extracted"
            measured = run_process([self.extract_program, "x", f"-mmt={self.threads}", "-y", f"-o{extraction}", source], work, timeout)
            if not directory:
                self.move_single(extraction, output)
            return measured
        if self.identifier == "zip" and not self.python_worker:
            if action == "compress":
                return run_process([self.program, *self.compression_options(directory), output,
                                    *(["-r", "."] if directory else ["--", "./" + source.name])],
                                   work, timeout, cwd=source if directory else source.parent)
            extraction = output if directory else work / "zip-extracted"
            measured = run_process([self.extract_program, "-q", "-o", source, "-d", extraction], work, timeout)
            if not directory:
                self.move_single(extraction, output)
            return measured
        if self.identifier in {"zip", "bzip2"} and self.python_worker:
            cmd = [self.program, str(WORKER), self.identifier, action, source, output, "--max-bytes", str(max(1, size))]
            cmd += ["--level", str(self.level)]
            if directory:
                cmd.append("--directory")
            return run_process(cmd, work, timeout)
        if self.identifier == "7z":
            if action == "compress":
                return run_process([self.program, "a", *self.compression_options(directory), output, "--",
                                    "." if directory else "./" + source.name], work, timeout, cwd=source if directory else source.parent)
            extraction = output if directory else work / "7z-extracted"
            measured = run_process([self.program, "x", f"-mmt={self.threads}", "-y", f"-o{extraction}", source], work, timeout)
            if not directory:
                self.move_single(extraction, output)
            return measured
        if self.identifier == "rar":
            if action == "compress":
                return run_process([self.program, "a", *self.compression_options(directory), "--", output,
                                    "." if directory else "./" + source.name], work, timeout, cwd=source if directory else source.parent)
            extraction = output if directory else work / "rar-extracted"
            extraction.mkdir(parents=True, exist_ok=True)
            measured = run_process([self.program, "x", "-cfg-", f"-mt{self.threads}", "-y", "--", source, str(extraction) + os.sep], work, timeout)
            if not directory:
                self.move_single(extraction, output)
            return measured
        if self.identifier == "kgb":
            if action == "compress":
                files = regular_files(source) if directory else [source]
                base = source if directory else source.parent
                names = ["./" + file.relative_to(base).as_posix() for file in files]
                if any(any(ord(char) < 32 or ord(char) == 127 for char in name) or "../" in name or "..\\" in name for name in names):
                    raise ValueError("KGB cannot represent control characters or parent-path markers in filenames")
                listing = work / "kgb-files.txt"
                listing.write_text("\n".join(names) + "\n", encoding="utf-8")
                # KGB's legacy @list parser accepts ASCII only and may silently
                # omit everything after a non-ASCII byte. argv preserves UTF-8.
                arguments = ["@" + str(listing)] if all(name.isascii() for name in names) else names
                return run_process([self.program, *self.compression_options(directory), output, *arguments], work, timeout, cwd=base)
            extraction = output if directory else work / "kgb-extracted"
            extraction.mkdir(parents=True, exist_ok=True)
            measured = run_process([self.program, source], work, timeout, cwd=extraction)
            if not directory:
                self.move_single(extraction, output)
            return measured
        if action == "compress":
            options = self.compression_options()
        else:
            options = {"brotli": ["-d", "-c"], "gzip": ["-d", "-c"], "xz": ["-d", f"-T{self.threads}", "-c"],
                       "zstd": ["-d", "-q", f"-T{self.threads}", "-c"], "bzip2": ["-d", "-c"],
                       "lz4": ["-d", "-q", *([f"-T{self.threads}"] if self.lz4_workers else []), "-c"]}[self.identifier]
        return run_process([self.program, *options, "--", source], work, timeout, stdout_path=output)

    @staticmethod
    def move_single(extraction, output):
        files = regular_files(extraction)
        if len(files) != 1:
            raise ValueError("Unexpected single-file archive inventory")
        shutil.move(files[0], output)

    def workflow(self, action, source, output, work, timeout, expected, directory=False):
        """Directory stream codecs include tar packing/unpacking in elapsed time."""
        size = expected["size"]
        start = time.perf_counter()
        if self.identifier == "kgb" and directory and action == "decompress":
            Path(output).mkdir(parents=True, exist_ok=True)
            for entry in expected["files"]:
                safe_member(output, entry["path"]).parent.mkdir(parents=True, exist_ok=True)
        if not directory or self.identifier.startswith("bato") or self.identifier in {"zip", "7z", "rar", "kgb", "tar"}:
            measured = self.step(action, source, output, work, timeout, size, directory)
            measured.input_bytes = size if action == 'compress' else None
            measured.wall_s = time.perf_counter() - start
            return measured
        tar_path = Path(work) / "payload.tar"
        if action == "compress":
            container_start = time.perf_counter()
            make_tar(source, tar_path, expected)
            container_time = time.perf_counter() - container_start
            measurement = self.step(action, tar_path, output, work, timeout, tar_path.stat().st_size)
            measurement.input_bytes = tar_path.stat().st_size
        else:
            measurement = self.step(action, source, tar_path, work, timeout, size + 1024 * 1024 + len(expected["files"]) * 2048)
            container_start = time.perf_counter()
            extract_tar(tar_path, output, size)
            container_time = time.perf_counter() - container_start
        measurement.container_time_s = container_time
        measurement.wall_s = time.perf_counter() - start
        tar_path.unlink(missing_ok=True)
        return measurement


def create_codecs(binary, version, requested=None, only_bato=False, profile="high", threads=1,
                  bato_modes="cpu1", memory_budget=256 * 2**20,
                  chunk_size=256 * 1024, codec_tuning=None, nncp_profile="default", nncp_cuda=False,
                  segment_size=64 * 2**20):
    if not 2**16 <= segment_size <= 2**30 or segment_size & (segment_size - 1):
        raise ValueError("Bato segment size must be a power of two from 64 KiB to 1 GiB")
    if chunk_size < 4096 or chunk_size > 2**20 or chunk_size & (chunk_size - 1) or chunk_size > segment_size:
        raise ValueError("Bato chunk size must be a power of two from 4 KiB to min(1 MiB, segment size)")
    names = selected_names(requested, only_bato)
    if nncp_profile not in {"default", "enwik8", "enwik9", "lstm", "lstm_fast"}:
        raise ValueError("NNCP profile must be default, enwik8, enwik9, lstm, or lstm_fast")
    tuning = validate_tuning({} if codec_tuning is None else codec_tuning)
    if set(tuning) - set(names):
        raise ValueError("Codec tuning includes a codec not selected for this run")
    modes = list(dict.fromkeys(bato_modes.split(",")))
    if not modes or not set(modes) <= {"cpu1", "cpu"}:
        raise ValueError("Bato modes must be cpu1 and/or cpu")
    codecs = []
    for name in dict.fromkeys(names):
        if name == "bato":
            for execution in modes:
                count = 1 if execution == "cpu1" else threads
                identifier = "bato-v1" if execution == "cpu1" else f"bato-v1-{execution}{count}"
                codecs.append(Codec(identifier, f"Bato Rust ({execution}, {count} workers)", str(binary) if binary else None,
                                    version or "unavailable", profile=profile, threads=count,
                                    memory_budget=memory_budget, chunk_size=chunk_size,
                                    segment_size=segment_size))
            continue
        if name == "zip" and tuning.get(name, {}).get("backend") == "7z":
            program = shutil.which("7zz") or shutil.which("7z")
            error, codec_version = None, "unavailable"
            if program:
                program = str(Path(program).resolve())
                try:
                    _, codec_version = probe_command([program])
                    if "7-zip" not in codec_version.lower():
                        raise ValueError("ZIP backend does not identify as 7-Zip")
                except (OSError, ValueError, ProcessFailure) as failure:
                    error = str(failure)
            else:
                error = "ZIP backend requires 7zz or 7z on PATH"
            codecs.append(Codec(name, f"ZIP via 7-Zip ({profile})", program, codec_version.strip()[:2048], error,
                                profile=profile, threads=threads, python_worker=False,
                                extract_program=program, zip_backend="7z"))
            continue
        if name in {"zip", "bzip2"}:
            program = shutil.which(name)
            extractor = shutil.which("unzip") if name == "zip" else None
            level = PROFILE_LEVELS[name][("fast", "default", "high").index(profile)]
            if program and (name != "zip" or extractor):
                program = str(Path(program).resolve())
                extractor = str(Path(extractor).resolve()) if extractor else None
                error, native_version = None, "unavailable"
                try:
                    _, native_version = probe_command([program, "-v" if name == "zip" else "--version"])
                    if name not in native_version.lower():
                        raise ValueError(f"Executable does not identify as {name}")
                    if extractor:
                        _, text = probe_command([extractor, "-v"])
                        if "unzip" not in text.lower():
                            raise ValueError("ZIP extractor does not identify as unzip")
                        native_version += "\n" + text
                except (OSError, ValueError, ProcessFailure) as failure:
                    error = str(failure)
                codecs.append(Codec(name, f"Native {name.upper()} ({profile} level {level})", program,
                               native_version.strip()[:4096], error, profile=profile,
                                    python_worker=False, extract_program=extractor))
                continue
            codecs.append(Codec(name, f"Python ZIP ({profile})" if name == "zip" else f"Python BZIP2 ({profile})",
                                sys.executable,
                                f"Python {platform.python_version()}, zlib {zlib.ZLIB_RUNTIME_VERSION}", profile=profile))
            continue
        program = (shutil.which("7zz") or shutil.which("7z")) if name == "7z" else shutil.which(name)
        level = PROFILE_LEVELS[name][("fast", "default", "high").index(profile)]
        codec_version, error, workers = "unavailable", None, False
        if program:
            program = str(Path(program).resolve())
            try:
                _, text = probe_command([program] if name in {"7z", "rar", "kgb", "paq8px", "paq8l", "zpaq"} else [program, "-h"] if name == "cmix" else [program, "--help"] if name == "nncp" else [program, "--version"])
                signatures = {"tar": ("tar",), "brotli": ("brotli",),"paq8l": ("paq8l",), "paq8px": ("paq8px",), "zpaq": ("zpaq",), "cmix": ("cmix",), "nncp": ("nncp",), "7z": ("7-zip",), "rar": ("rar ",), "kgb": ("kgb",), "gzip": ("gzip",),
                              "xz": ("xz",), "zstd": ("zstd", "zstandard"), "lz4": ("lz4",)}
                if not any(signature in text.lower() for signature in signatures[name]) or (name == "rar" and "unrar" in text.lower()):
                    raise ValueError(f"Executable does not identify as {name}")
                codec_version = text.strip()[:2048]
                if name == "lz4":
                    _, help_text = probe_command([program, "--help"])
                    workers = "-T" in help_text or "--threads" in help_text
            except (OSError, ValueError, ProcessFailure) as failure:
                error = str(failure)
        else:
            error = "Executable not found on PATH" + (" (requires rar; unrar cannot compress)" if name == "rar" else "")
        codecs.append(Codec(name, f"{name.upper()} ({profile} level {level})", program, codec_version, error, workers,
                            profile=profile, threads=threads if name in {"7z", "rar", "xz", "zstd", "zpaq", "nncp"} or workers else 1,
                            nncp_profile=nncp_profile, nncp_cuda=nncp_cuda if name == "nncp" else False))
    for codec in codecs:
        codec.compression_parameters = list(tuning.get(codec.identifier, {}).get("parameters", []))
    return codecs


def check_tools(requested=None, only_bato=False, bato_bin=None, workload="both", timeout=30, allow_unavailable=False, **execution):
    """Preflight fixtures never enter the corpus, metrics, or baseline cache."""
    names = selected_names(requested, only_bato)
    binary, version, bato_error = None, None, None
    if "bato" in names:
        try:
            binary, version = resolve_bato(bato_bin)
        except (OSError, ValueError, ProcessFailure) as failure:
            bato_error = str(failure)
    codecs = create_codecs(binary, version, ",".join(names), **execution)
    rows = []
    print(f"Tool check: {execution.get('profile', 'high')} profile; requested workers={execution.get('threads', 1)}", flush=True)
    with tempfile.TemporaryDirectory(prefix="bato-tool-check-") as temporary:
        root = Path(temporary)
        payload = root / "input"
        (payload / "nested space").mkdir(parents=True)
        single = payload / "nested space" / "日本語.bin"
        single.write_bytes(bytes(range(256)) * 8 + b"tool capability check\n" * 64)
        (payload / ".hidden.txt").write_bytes(b"preflight capability fixture\n")
        (payload / "empty.dat").touch()
        for codec in codecs:
            row = {"codec": codec.identifier, "name": codec.name, "path": codec.program,
                   "version": codec.version, "settings": codec.settings(), "status": "READY"}
            error = bato_error if codec.identifier.startswith("bato") else codec.probe_error
            if not codec.program or error:
                row.update(status="MISSING" if not codec.program else "FAILED", error=error or "Executable not found")
            else:
                try:
                    for directory in ([False, True] if workload == "both" else [workload == "directories"]):
                        source = payload if directory else single
                        expected = snapshot(source)
                        work = root / codec.identifier / ("directory" if directory else "file")
                        work.mkdir(parents=True)
                        archive = work / ("compressed" + codec.archive_suffix(directory))
                        restored = work / "restored"
                        codec.workflow("compress", source, archive, work, timeout, expected, directory)
                        if not archive.is_file() or not archive.stat().st_size:
                            raise ValueError("Tool produced no archive with its extreme settings")
                        codec.workflow("decompress", archive, restored, work, timeout, expected, directory)
                        if snapshot(restored) != expected or snapshot(source) != expected:
                            raise ValueError("Tool capability round trip changed file contents or paths")
                except (OSError, ValueError, ProcessFailure, tarfile.TarError, zipfile.BadZipFile, EOFError, MemoryError) as failure:
                    row.update(status="RESOURCE_LIMIT" if isinstance(failure, MemoryError) else "FAILED", error=str(failure))
            rows.append(row)
            headline = codec.version.splitlines()[0] if codec.version else "unknown"
            print(f"[{row['status']}] {codec.name} | {codec.program or 'not found'} | {headline}", flush=True)
            if row.get("error"):
                print(f"  {row['error']}", flush=True)
    report = {"checked_at": utc_now(), "compression_profile": "extreme", "status": "ready" if all(row["status"] == "READY" for row in rows) or allow_unavailable else "failed", "tools": rows}
    if report["status"] == "failed":
        print("Tool check failed; no benchmark measurements or cache lookups started.", flush=True)
        print("Ubuntu/Debian packages: 7zip rar kgb lz4 xz-utils zstd gzip. Install missing tools manually; rebuild Rust Bato with cargo build --release.", flush=True)
        print("Use --codecs to select an installed subset, or --only-bato.", flush=True)
    if allow_unavailable:
        ready = {row["codec"] for row in rows if row["status"] == "READY"}
        codecs = [codec for codec in codecs if codec.identifier in ready]
    return codecs, report
