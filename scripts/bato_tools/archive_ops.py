"""Frozen Bato archive-operation campaigns in the canonical SQLite database."""
import hashlib
import itertools
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import tempfile
import uuid
from contextlib import closing
from pathlib import Path

from .codecs import resolve_bato
from .common import (
    PROJECT_ROOT, ProcessFailure, atomic_json, digest_json, run_process, sha256, snapshot, utc_now,
)
from .resources import configure_resources
from .runner import environment, host_measurement_lock

OPERATIONS = ("list", "verify", "diff", "add", "remove")
GEOMETRY = ("--segment-size", 65536, "--chunk-size", 16384)
DIFF_PATH = "data"
ADD_PATH = "added"
REMOVE_PATH = "remove-me"


def fixtures(root, archive_format="indexed"):
    source = root / "source"
    (source / REMOVE_PATH / "empty").mkdir(parents=True)
    (source / "empty").mkdir()
    payload = hashlib.shake_256(b"Bato archive ops v1").digest(512 * 1024)
    (source / DIFF_PATH).write_bytes(payload)
    (source / REMOVE_PATH / "duplicate").write_bytes(payload)
    (source / ".records").write_bytes(b'{"id":42,"event":"archive"}\n' * 1000)
    (source / "zero").touch()
    changed = root / "changed"
    shutil.copytree(source, changed)
    (changed / DIFF_PATH).write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    addition = root / ADD_PATH
    addition.write_bytes(payload[:4096])
    return {
        "version": 1, "format": archive_format, "geometry": list(GEOMETRY),
        "operations": {
            "list": {"output": "json"},
            "verify": {"scope": "full"},
            "diff": {"output": "json", "changed_path": DIFF_PATH},
            "add": {"path": ADD_PATH},
            "remove": {"path": REMOVE_PATH},
        },
        "source": snapshot(source), "changed": snapshot(changed), "added": snapshot(addition),
    }


def validate_fixture(manifest, frozen):
    if manifest != json.loads(Path(frozen).read_text()):
        raise ValueError("Archive-operation fixture differs from frozen manifest")


def expected_tree(manifest, operation):
    tree = manifest["source"]
    files, directories = tree["files"].copy(), tree["directories"].copy()
    if operation == "add":
        files.append(dict(manifest["added"], path=ADD_PATH))
    if operation == "remove":
        files = [entry for entry in files if not entry["path"].startswith(f"{REMOVE_PATH}/")]
        directories = [name for name in directories if name != REMOVE_PATH and not name.startswith(f"{REMOVE_PATH}/")]
    return sorted(files, key=lambda entry: entry["path"]), directories


def verify_result(operation, invoke, paths, manifest, original):
    archive, output, work = paths
    files, directories = expected_tree(manifest, operation)
    if operation in ("list", "diff"):
        rows = json.loads(output.read_text())
        if operation == "list":
            expected = {entry["path"]: entry["size"] for entry in files}
            expected.update(dict.fromkeys(directories))
            actual = {entry["path"]: entry["size"] for entry in rows}
        else:
            expected = {DIFF_PATH: "changed"}
            actual = {entry["path"]: entry["status"] for entry in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError(f"Incorrect {operation} output")
    if operation in ("add", "remove"):
        with archive.open("rb") as stream:
            expected_magic = b"BAI1" if manifest["format"] == "indexed" else b"BATO"
            if stream.read(4) != expected_magic:
                raise ValueError("Edit changed the archive format")
        invoke("extract", archive, "-o", work / "restored")
        restored = snapshot(work / "restored")
        if (restored["files"], restored["directories"]) != (files, directories):
            raise ValueError("Edited archive restored a different tree")
    elif sha256(archive) != original:
        raise ValueError("Read-only operation modified its input")


def trial(build, track, operation, inputs):
    root, seed, manifest, timeout = inputs
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        work = Path(temporary)
        archive = work / "archive.bato"
        shutil.copyfile(seed, archive)
        original = sha256(archive)
        output = work / "stdout"
        commands = {
            "list": ("list", archive, "--json"),
            "verify": ("-t", archive),
            "diff": ("diff", archive, root / "changed", "--json"),
            "add": ("add", archive, root / ADD_PATH),
            "remove": ("remove", archive, REMOVE_PATH),
        }

        def invoke(*command, stdout=None):
            return run_process([build, "--threads", track["threads"], "--memory-budget",
                                track["memory"], *manifest["geometry"], "--stats", *command],
                               work, timeout, stdout_path=stdout, monitor_tree=True, env={**os.environ, "TMPDIR": str(work)})

        record = {"status": "passed", "start_bytes": archive.stat().st_size,
                  "start_sha256": original, "peak_scratch_bytes": None}
        try:
            result = invoke(*commands[operation], stdout=output)
        except ProcessFailure as error:
            return dict(record, status="timeout" if error.timed_out else "command_failed",
                        returncode=error.returncode, error=str(error))
        except OSError as error:
            return dict(record, status="command_failed", error=str(error))
        record.update(wall_s=result.wall_s, cpu_s=result.cpu_s, peak_rss_kib=result.peak_rss_kib,
                      process_tree_peak_kib=result.sampled_tree_peak_rss_kib, execution_stats=result.execution_stats,
                      archive_bytes=archive.stat().st_size if archive.is_file() and operation in ("add", "remove") else None)
        try:
            verify_result(operation, invoke, (archive, output, work), manifest, original)
        except (ProcessFailure, OSError, ValueError, KeyError, TypeError) as error:
            return dict(record, status="verification_failed", error=str(error))
        return record


def init_database(db):
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS archive_operation_runs (run TEXT PRIMARY KEY, record TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS archive_operation_trials (
            run TEXT, track TEXT, operation TEXT, phase TEXT, number INTEGER, record TEXT NOT NULL,
            PRIMARY KEY(run, track, operation, phase, number));
    """)


def build_identity(path, revision):
    binary, version = resolve_bato(path)
    source = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options", revision + "^{commit}"],
                                     cwd=PROJECT_ROOT, text=True).strip()
    target = None
    if shutil.which("file"):
        description = subprocess.check_output(["file", "-b", binary], text=True)
        target = description.split(", BuildID")[0].strip()
    help_text = subprocess.check_output([binary, "--help"], stderr=subprocess.STDOUT, text=True)
    supported = {"verify"} | {name for name in OPERATIONS if name + " ARCHIVE" in help_text}
    return binary, {"path": str(binary), "sha256": sha256(binary), "version": version,
                    "declared_git_sha": source, "executable_target": target,
                    "indexed": "--indexed" in help_text, "operations": sorted(supported)}


def storage_identity(root):
    if shutil.which("findmnt") is None:
        return None
    result = subprocess.run(["findmnt", "-n", "-o", "SOURCE,FSTYPE,OPTIONS", "-T", root],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def campaign(args, root, manifest, db):
    validate_fixture(manifest, args.frozen_workloads)
    modes = args.bato_modes.split(",")
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"cpu1", "cpu"}:
        raise ValueError("Expected distinct cpu1,cpu tracks")
    configure_resources(args)
    tracks = {mode: {"threads": 1 if mode == "cpu1" else args.threads,
                     "memory": args.bato_memory_budget} for mode in modes}
    builds = [build_identity(args.bato_bin, args.bato_revision)]
    if args.baseline_bato_bin:
        builds.insert(0, build_identity(args.baseline_bato_bin, args.baseline_bato_revision))
    warmups = args.warmups if args.warmups is not None else 1
    repetitions = args.repetitions if args.repetitions is not None else 5
    ids = [uuid.uuid4().hex for _ in builds]
    host = environment(root)
    storage = storage_identity(root)
    for run, (_, identity) in zip(ids, builds):
        meta = dict(build=identity, fixture=digest_json(manifest), manifest=manifest,
                    tracks=tracks, host=host, storage=storage, created=utc_now(),
                    warmups=warmups, repetitions=repetitions, timeout=args.timeout,
                    cache_policy="fresh prehashed copy; OS cache uncontrolled",
                    scratch_measurement="unavailable: transient file high-water mark is not sampled",
                    status="running")
        with db:
            db.execute("INSERT INTO archive_operation_runs VALUES (?, ?)", (run, json.dumps(meta)))
        print(f"{run} {identity['version']} {identity['sha256']}", flush=True)
    failed = False
    for mode, track in tracks.items():
        seeds = []
        for run, (binary, identity) in zip(ids, builds):
            seed = root / f"{run}-{mode}.bato"
            if manifest["format"] == "indexed" and not identity["indexed"]:
                seeds.append(("unsupported", "--indexed unavailable"))
                continue
            indexed = ["--indexed"] if manifest["format"] == "indexed" else []
            try:
                run_process([binary, "--threads", track["threads"], "--memory-budget", track["memory"],
                             *manifest["geometry"], "archive", root / "source", *indexed, "-o", seed], root, args.timeout,
                            env={**os.environ, "TMPDIR": str(root)})
                with seed.open("rb") as stream:
                    if stream.read(4) != (b"BAI1" if indexed else b"BATO"):
                        raise ValueError("Starting archive has a different format")
                run_process([binary, "--threads", track["threads"], "--memory-budget", track["memory"],
                             "extract", seed, "-o", root / f"{run}-{mode}-verify"], root, args.timeout,
                            env={**os.environ, "TMPDIR": str(root)})
                if snapshot(root / f"{run}-{mode}-verify") != manifest["source"]:
                    raise ValueError("Starting archive failed round trip")
                seeds.append(("passed", seed))
            except (ProcessFailure, OSError, ValueError) as error:
                seeds.append(("timeout" if isinstance(error, ProcessFailure) and error.timed_out else "setup_failed", str(error)))
        schedule = itertools.chain((("warmup", n) for n in range(warmups)),
                                   (("measured", n) for n in range(repetitions)))
        for index, (phase, number) in enumerate(schedule):
            order = list(range(len(builds)))
            if index % 2:
                order.reverse()
            for operation in OPERATIONS:
                for slot in order:
                    binary, identity = builds[slot]
                    status, seed = seeds[slot]
                    if status != "passed":
                        result = {"status": status, "error": seed}
                    elif operation not in identity["operations"]:
                        result = {"status": "unsupported"}
                    else:
                        try:
                            result = trial(binary, track, operation, (root, seed, manifest, args.timeout))
                        except OSError as error:
                            result = {"status": "setup_failed", "error": str(error)}
                    failed |= result["status"] != "passed"
                    with db:
                        db.execute("INSERT INTO archive_operation_trials VALUES (?, ?, ?, ?, ?, ?)",
                                   (ids[slot], mode, operation, phase, number, json.dumps(result)))
    with db:
        for run in ids:
            meta = load_meta(db, run)
            meta["status"] = "complete"
            db.execute("UPDATE archive_operation_runs SET record=? WHERE run=?", (json.dumps(meta), run))
    return int(failed)


def run_archive_ops(args):
    if args.checkpoint_dir and Path(args.checkpoint_dir).resolve() == Path(args.output_dir).resolve():
        raise ValueError("Checkpoint directory must differ from output directory")
    if args.resume:
        raise ValueError("Archive-operation campaigns preserve partial trials but do not support --resume")
    if not args.list_workloads and not (args.frozen_workloads and args.bato_bin and args.bato_revision):
        raise ValueError("Specify --frozen-workloads, --bato-bin and --bato-revision")
    if bool(args.baseline_bato_bin) != bool(args.baseline_bato_revision):
        raise ValueError("Specify both baseline binary and revision")
    with tempfile.TemporaryDirectory(dir=args.scratch_dir, prefix="bato-ops-") as temporary:
        root = Path(temporary)
        manifest = fixtures(root, args.archive_operation_format)
        if args.list_workloads:
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        from .database import DEFAULT_DATABASE, database_path, prepare_database
        with host_measurement_lock():
            destination = database_path(args)
            if destination == DEFAULT_DATABASE.resolve():
                prepare_database(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            from .store import MeasurementStore
            if args.checkpoint_dir and Path(args.checkpoint_dir).resolve() / 'measurements.sqlite3' == destination:
                raise ValueError('Checkpoint database must differ from canonical database')
            store = MeasurementStore(destination)
            store.close()
        with host_measurement_lock(), closing(sqlite3.connect(destination)) as db:
            init_database(db)
            try:
                return campaign(args, root, manifest, db)
            finally:
                if args.checkpoint_dir:
                    mirror = Path(args.checkpoint_dir).resolve()
                    mirror.mkdir(parents=True, exist_ok=True)
                    with closing(sqlite3.connect(mirror / "measurements.sqlite3")) as backup:
                        db.backup(backup)


def load_meta(db, run):
    row = db.execute("SELECT record FROM archive_operation_runs WHERE run=?", (run,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown archive-operation run {run}")
    return json.loads(row[0])


def summarize(samples, metric):
    values = [sample.get(metric) for sample in samples]
    if not values or any(value is None for value in values):
        return None
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def compare_operations(output, baseline, candidate, database=None):
    root = Path(output)
    with closing(sqlite3.connect((Path(database) if database else root / "measurements.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        metadata = [load_meta(db, run) for run in (baseline, candidate)]
        def observations(run, mode, operation):
            return [dict(json.loads(record), phase=phase) for phase, record in db.execute(
                "SELECT phase, record FROM archive_operation_trials WHERE run=? AND track=? AND operation=? ORDER BY phase,number",
                (run, mode, operation))]
        left, right = metadata
        cases = []
        for mode in sorted(left["tracks"].keys() | right["tracks"].keys()):
            for operation in OPERATIONS:
                samples = [observations(run, mode, operation) for run in (baseline, candidate)]
                outcomes = [[sample["status"] for sample in side] for side in samples]
                measured = [[sample for sample in side if sample["phase"] == "measured"] for side in samples]
                size_match = (left["fixture"] == right["fixture"] and mode in left["tracks"] and
                              mode in right["tracks"])
                timing_match = (size_match and left["tracks"][mode] == right["tracks"][mode]
                                and left["storage"] is not None and left["host"]["host_id_sha256"] is not None
                                and all(left[key] == right[key] for key in
                                        ("host", "storage", "cache_policy", "warmups", "repetitions", "timeout"))
                                and left["build"]["executable_target"] is not None
                                and left["build"]["executable_target"] == right["build"]["executable_target"])
                case = dict(track=mode, operation=operation, outcomes=outcomes,
                            timing_conditions_match=timing_match, metrics={})
                complete = all(len(side) == meta["repetitions"] and len(statuses) == meta["repetitions"] + meta["warmups"]
                               and all(status == "passed" for status in statuses)
                               for side, meta, statuses in zip(measured, metadata, outcomes))
                case["status"] = "complete" if complete and size_match else "unmatched_or_failed"
                if complete:
                    for metric in ("start_bytes", "archive_bytes", "wall_s", "cpu_s", "peak_rss_kib", "process_tree_peak_kib"):
                        if not (size_match if metric in ("start_bytes", "archive_bytes") else timing_match):
                            continue
                        before, after = (summarize(side, metric) for side in measured)
                        if before is None or after is None:
                            continue
                        verdict = ("lower" if after["max"] < before["min"] else
                                   "higher" if after["min"] > before["max"] else
                                   "equal" if metric in ("start_bytes", "archive_bytes") and before == after else "uncertain")
                        case["metrics"][metric] = dict(baseline=before, candidate=after, verdict=verdict,
                                                       delta=after["median"] - before["median"])
                cases.append(case)
    report = root / "archive-operation-comparisons" / f"{uuid.uuid4().hex}.json"
    atomic_json(report, dict(baseline=baseline, candidate=candidate, runs=metadata, cases=cases))
    for case in cases:
        wall = case["metrics"].get("wall_s")
        detail = (f"{wall['baseline']['median']:.6f}s → {wall['candidate']['median']:.6f}s ({wall['verdict']})"
                  if wall else "timing unavailable or unmatched")
        print(f"{case['track']} {case['operation']} {case['status']} {detail}")
    print(f"Report {report}")
    return 0
