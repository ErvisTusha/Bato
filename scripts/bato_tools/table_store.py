"""SQLite store of table-benchmark results, so an unchanged binary is not re-measured.

A result is reused only for the same binary bytes, input identity, compute
mode, resource limits, geometry and host hardware, and only when it was
verified with at least the requested warmups and repetitions.
"""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sqlite3
import subprocess

from .common import digest_json
from .runner import environment

DEFAULT_TABLE_DATABASE = Path.home() / 'data-research' / 'bato-table.sqlite3'
VERIFIED = 'PASS & VERIFIED'
# Host facts that change timings; the harness hash is left out so that editing
# the scripts does not discard measurements of an unchanged binary.
HARDWARE_FIELDS = ('os', 'architecture', 'cpu_model', 'physical_cores', 'logical_cpus', 'affinity',
                   'host_id_sha256', 'governor', 'total_ram_bytes', 'effective_memory_bytes',
                   'cpu_quota', 'gpu_devices')
SCHEMA = '''CREATE TABLE IF NOT EXISTS table_results (
    id INTEGER PRIMARY KEY,
    condition_key TEXT NOT NULL,
    measured_at TEXT NOT NULL,
    binary_sha256 TEXT NOT NULL,
    revision TEXT NOT NULL,
    workload_id TEXT NOT NULL,
    target TEXT NOT NULL,
    compute TEXT NOT NULL,
    warmups INTEGER NOT NULL,
    repetitions INTEGER NOT NULL,
    status TEXT NOT NULL,
    condition TEXT NOT NULL,
    result TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS table_results_key ON table_results (condition_key, status);'''


def nvidia_devices():
    """Name and driver of each NVIDIA GPU; empty when nvidia-smi is absent or fails."""
    if not shutil.which('nvidia-smi'):
        return []
    query = ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']
    try:
        found = subprocess.run(query, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted(line.strip() for line in found.stdout.splitlines() if line.strip())


def hardware_identity():
    host = environment()
    return {**{field: host.get(field) for field in HARDWARE_FIELDS}, 'nvidia': nvidia_devices()}


def condition(workload, build, compute, settings):
    """Everything that may change the archive bytes or the timings of one variant."""
    return {'binary_sha256': build['sha256'], 'workload_id': workload['workload_id'],
            'input_sha256': workload['snapshot']['sha256'], 'compute': compute, **settings}


class TableStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(SCHEMA)

    def close(self):
        self.connection.close()

    def lookup(self, wanted, warmups, repetitions):
        """Return (result, measured_at) of the newest sufficient verified result, or None."""
        with closing(self.connection.execute(
                'SELECT result, measured_at FROM table_results WHERE condition_key=? AND status=? '
                'AND warmups>=? AND repetitions>=? ORDER BY id DESC LIMIT 1',
                (digest_json(wanted), VERIFIED, warmups, repetitions))) as cursor:
            found = cursor.fetchone()
        return (json.loads(found[0]), found[1]) if found else None

    def save(self, measured, build, result, runs):
        """Record one variant result, including failures, which never satisfy a lookup."""
        warmups, repetitions = runs
        with self.connection:
            self.connection.execute(
                'INSERT INTO table_results (condition_key, measured_at, binary_sha256, revision, workload_id, '
                'target, compute, warmups, repetitions, status, condition, result) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (digest_json(measured), datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 build['sha256'], build['revision'], measured['workload_id'], result.get('target', ''),
                 measured['compute'], warmups, repetitions, result.get('status', 'FAIL'),
                 json.dumps(measured, sort_keys=True), json.dumps(result, sort_keys=True, default=str)))
