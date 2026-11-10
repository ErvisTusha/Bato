"""Typed, transactional projection of competitor cases and unavailable tools."""
import json

from .common import digest_json

# The authoritative case JSON is retained; missing metrics remain SQL NULL.
FIELDS = {
    'workload_id': 'workload_id', 'filename': 'target', 'input_sha256': 'input_sha256',
    'original_bytes': 'raw_size', 'competitor': 'codec', 'compressed_bytes': 'comp_size',
    'compress_seconds': 'comp_time_s', 'decompress_seconds': 'decomp_time_s',
    'compress_cpu_seconds': 'comp_cpu_s', 'decompress_cpu_seconds': 'decomp_cpu_s',
    'compress_peak_rss_kib': 'comp_peak_rss_kib', 'decompress_peak_rss_kib': 'decomp_peak_rss_kib',
    'compress_tree_rss_kib': 'comp_sampled_tree_peak_rss_kib',
    'decompress_tree_rss_kib': 'decomp_sampled_tree_peak_rss_kib',
    'status': 'status', 'verified_lossless': 'verified_lossless', 'error': 'error',
    'measured_at': 'measured_at', 'settings': 'effective_settings', 'track': 'track',
    'compression_profile': 'compression_profile', 'binary_identity': 'binary_identity',
    'resource_policy': 'resource_policy', 'cpu_scope': 'cpu_scope', 'tree_rss_scope': 'tree_rss_scope',
}


def install_competitor_table(connection):
    """Install a projection synchronized by SQLite triggers, and backfill existing cases."""
    exists = connection.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name='competitor_measurements'").fetchone()
    if exists and 'run_status' not in {row[1] for row in connection.execute('PRAGMA table_info(competitor_measurements)')}:
        connection.execute('ALTER TABLE competitor_measurements ADD COLUMN run_status TEXT')
        connection.commit()
    columns = ', '.join(['run_id', 'case_id', *FIELDS, 'source', 'run_status'])
    def projection(alias):
        fields = ', '.join(f"json_extract({alias}.record, '$.{key}')" for key in FIELDS.values())
        return f"{alias}.run, {alias}.id, {fields}, 'case', (SELECT json_extract(record, '$.status') FROM runs WHERE run={alias}.run)"
    connection.executescript(f'''
        BEGIN;
        CREATE TABLE IF NOT EXISTS competitor_measurements (
            run_id TEXT NOT NULL, case_id TEXT NOT NULL, workload_id TEXT, filename TEXT,
            input_sha256 TEXT, original_bytes INTEGER, competitor TEXT, compressed_bytes INTEGER,
            compress_seconds REAL, decompress_seconds REAL, compress_cpu_seconds REAL,
            decompress_cpu_seconds REAL, compress_peak_rss_kib INTEGER, decompress_peak_rss_kib INTEGER,
            compress_tree_rss_kib INTEGER, decompress_tree_rss_kib INTEGER, status TEXT,
            verified_lossless INTEGER, error TEXT, measured_at TEXT, settings TEXT, track TEXT,
            compression_profile TEXT, binary_identity TEXT, resource_policy TEXT, cpu_scope TEXT,
            tree_rss_scope TEXT, source TEXT NOT NULL, run_status TEXT, PRIMARY KEY (run_id, case_id));
        CREATE INDEX IF NOT EXISTS competitor_measurements_file ON competitor_measurements(filename, competitor);
        DROP TRIGGER IF EXISTS competitor_case_insert;
        DROP TRIGGER IF EXISTS competitor_case_update;
        DROP TRIGGER IF EXISTS competitor_case_delete;
        CREATE TRIGGER competitor_case_insert AFTER INSERT ON cases
        WHEN json_extract(NEW.record, '$.codec') NOT LIKE 'bato%'
        BEGIN
            INSERT OR REPLACE INTO competitor_measurements ({columns}) SELECT {projection('NEW')};
        END;
        CREATE TRIGGER competitor_case_update AFTER UPDATE ON cases
        BEGIN
            DELETE FROM competitor_measurements WHERE run_id=OLD.run AND case_id=OLD.id AND source='case';
            INSERT OR REPLACE INTO competitor_measurements ({columns})
                SELECT {projection('NEW')} WHERE json_extract(NEW.record, '$.codec') NOT LIKE 'bato%';
        END;
        CREATE TRIGGER competitor_case_delete AFTER DELETE ON cases
        BEGIN
            DELETE FROM competitor_measurements WHERE run_id=OLD.run AND case_id=OLD.id AND source='case';
        END;
        INSERT OR IGNORE INTO competitor_measurements ({columns})
            SELECT {projection('c')} FROM cases c WHERE json_extract(c.record, '$.codec') NOT LIKE 'bato%';
        UPDATE competitor_measurements SET run_status=(SELECT json_extract(record, '$.status')
            FROM runs WHERE run=competitor_measurements.run_id);
        COMMIT;
    ''')


def save_unavailable(connection, run, metadata):
    """Project preflight failures without pretending that trials were performed."""
    connection.execute("DELETE FROM competitor_measurements WHERE run_id=? AND source='preflight'", (run,))
    for tool in metadata.get('tool_preflight', {}).get('tools', []):
        if tool.get('status') == 'READY' or tool.get('codec', '').startswith('bato'):
            continue
        for item in metadata.get('workloads', []):
            if item.get('type') != 'single_file':
                continue
            case_id = digest_json([item['workload_id'], 'direct', tool['codec']])
            connection.execute('''INSERT OR IGNORE INTO competitor_measurements
                (run_id, case_id, workload_id, filename, input_sha256, original_bytes, competitor,
                 status, verified_lossless, error, settings, track, compression_profile, source, run_status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (run, case_id, item['workload_id'], item['target'], item.get('snapshot', {}).get('sha256'),
                 item.get('logical_payload_bytes'), tool['codec'], 'UNAVAILABLE', 0,
                 tool.get('error') or tool.get('status'), json.dumps(tool.get('settings')), 'direct',
                 metadata.get('compression_profile'), 'preflight', metadata.get('status')))
