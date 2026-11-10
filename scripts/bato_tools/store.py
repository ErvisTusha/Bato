"""Authoritative v4 run/trial transactions; reports are recoverable projections."""
import json
import sqlite3
from pathlib import Path
from .common import digest_json
from .result_views import install_views
from .competitor_table import install_competitor_table, save_unavailable

SCHEMA_VERSION = 4


def expected_case_ids(metadata):
    """Derive canonical case identities from a run manifest, when complete."""
    if not metadata.get('workloads') or not metadata.get('codecs'):
        return None
    result = set()
    for item in metadata['workloads']:
        tracks = ['direct'] if item['type'] == 'single_file' else (
            ['native', 'common-tar'] if metadata['track'] == 'both' else [metadata['track']])
        for track in tracks:
            for codec in metadata['codecs']:
                result.add(digest_json([item['workload_id'], track, codec['id']]))
    return result - set(metadata.get('baseline_references', {}))


def validate_case_records(cases, trials_by_case, expected_case_ids=None, complete=False):
    """Validate canonical inventory and successful case projections without writes."""
    actual_case_ids = {case['case_id'] for case in cases}
    if complete and expected_case_ids is not None and actual_case_ids != set(expected_case_ids):
        raise ValueError('Incomplete canonical case inventory')
    for case in cases:
        fresh_verified = case.get('verified_lossless') and case.get('cache_origin') == 'fresh'
        if case.get('status') == 'PASS & VERIFIED' and not case.get('verified_lossless'):
            raise ValueError(f'Inconsistent successful case verification: {case["case_id"]}')
        if fresh_verified and case.get('status') != 'PASS & VERIFIED':
            raise ValueError(f'Inconsistent fresh case status: {case["case_id"]}')
        if not fresh_verified:
            continue
        trials = trials_by_case.get(case['case_id'], [])
        for phase, required in [('warmup', case['warmups']), ('measured', case['repetitions'])]:
            accepted = [t for t in trials if t.get('phase') == phase and t.get('status') == 'PASS' and t.get('verified_lossless')]
            if len(accepted) != required or {t['number'] for t in accepted} != set(range(required)):
                raise ValueError(f'Missing committed {phase} trials for {case["case_id"]}')
        if case.get('trials') != [t for t in trials if t.get('phase') == 'measured']:
            raise ValueError(f'Case projection differs from committed trials: {case["case_id"]}')


class MeasurementStore:
    def __init__(self, path):
        self.path = Path(path)
        self.connection = sqlite3.connect(path, timeout=30)
        # The workflow has one writer protected by its host measurement lock.
        # Rollback journals avoid relying on externally shared WAL/shm files.
        self.connection.execute('PRAGMA journal_mode=DELETE')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS files (
                relative_path TEXT NOT NULL, sha256 TEXT NOT NULL, original_bytes INTEGER NOT NULL CHECK(original_bytes>=0),
                PRIMARY KEY(relative_path,sha256));
            CREATE TABLE IF NOT EXISTS runs (run TEXT PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS workloads (run TEXT, id TEXT, record TEXT NOT NULL, PRIMARY KEY(run,id));
            CREATE TABLE IF NOT EXISTS cases (run TEXT, id TEXT, record TEXT NOT NULL, PRIMARY KEY(run,id));
            CREATE TABLE IF NOT EXISTS trials (run TEXT, id TEXT, phase TEXT, number INTEGER, record TEXT NOT NULL,
                                             PRIMARY KEY(run,id,phase,number));
        ''')
        self.connection.execute('CREATE INDEX IF NOT EXISTS cases_by_id ON cases(id)')
        install_views(self.connection)
        self.connection.commit()
        install_competitor_table(self.connection)
        with self.connection:
            for run, encoded in self.connection.execute('SELECT run,record FROM runs').fetchall():
                save_unavailable(self.connection, run, json.loads(encoded))
        self.identity = self._identity()

    def _identity(self):
        state = self.path.stat()
        return state.st_dev, state.st_ino

    def _refresh(self):
        # Workspace synchronization can replace an otherwise writable file.
        # SQLite refuses writes through the old inode (READONLY_DBMOVED).
        # Reopen the current path; never alter permissions or reuse stale pages.
        identity = self._identity()
        if identity != self.identity:
            self.connection.close()
            self.connection = sqlite3.connect(self.path, timeout=30)
            self.connection.execute('PRAGMA synchronous=FULL')
            self.identity = identity

    def run(self, run):
        self._refresh()
        row = self.connection.execute('SELECT record FROM runs WHERE run=?', (run,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_run(self, run, record):
        self._refresh()
        with self.connection:
            self.connection.execute('INSERT OR REPLACE INTO runs VALUES (?,?)', (run, json.dumps(record)))
            save_unavailable(self.connection, run, record)
            self.connection.execute('UPDATE competitor_measurements SET run_status=? WHERE run_id=?',
                                    (record.get('status'), run))

    def register_input(self, sample):
        """Record verified original size without inventing compression measurements."""
        sample.verified_path()
        self._refresh()
        with self.connection:
            previous = self.connection.execute('SELECT original_bytes FROM files WHERE relative_path=? AND sha256=?',
                                               (sample.relative_path,sample.digest)).fetchone()
            if previous and previous[0] != sample.size:
                raise ValueError('Original size conflicts with recorded file identity: ' + sample.relative_path)
            self.connection.execute('INSERT OR IGNORE INTO files VALUES (?,?,?)',
                                    (sample.relative_path,sample.digest,sample.size))

    def save_workload(self, run, item):
        self._refresh()
        with self.connection:
            self.connection.execute('INSERT OR REPLACE INTO workloads VALUES (?,?,?)', (run, item['workload_id'], json.dumps(item)))

    def save_case(self, run, record):
        self._refresh()
        with self.connection:
            self.connection.execute('INSERT INTO cases VALUES (?,?,?) ON CONFLICT(run,id) DO UPDATE SET record=excluded.record',
                                    (run, record['case_id'], json.dumps(record)))

    def cases(self, run):
        self._refresh()
        return [json.loads(r[0]) for r in self.connection.execute('SELECT record FROM cases WHERE run=? ORDER BY id', (run,))]

    def trials(self, run, case):
        self._refresh()
        return [json.loads(r[0]) for r in self.connection.execute('SELECT record FROM trials WHERE run=? AND id=? ORDER BY phase DESC,number', (run, case))]

    def save_trial(self, run, case, phase, number, record):
        self._refresh()
        with self.connection:
            previous = self.connection.execute('SELECT record FROM trials WHERE run=? AND id=? AND phase=? AND number=?', (run, case, phase, number)).fetchone()
            encoded = json.dumps(record, sort_keys=True)
            if previous and json.loads(previous[0]) != record:
                raise ValueError('Cannot overwrite an already committed trial')
            self.connection.execute('INSERT OR IGNORE INTO trials VALUES (?,?,?,?,?)', (run, case, phase, number, encoded))

    def close(self):
        self.connection.close()

    def validate(self, run, expected_case_ids):
        from .baselines import validate_references
        self._refresh()
        validate_references(self, self.run(run))
        integrity = self.connection.execute('PRAGMA integrity_check').fetchall()
        if integrity != [('ok',)]:
            raise ValueError(f'Benchmark database integrity check failed: {integrity[:3]}')
        cases = self.cases(run)
        trials = {case['case_id']: self.trials(run, case['case_id']) for case in cases}
        validate_case_records(cases, trials, expected_case_ids, complete=True)
