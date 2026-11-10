"""Canonical database relocation and SQL projection acceptance tests."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_suite
from bato_tools.database import prepare_database
from bato_tools.store import MeasurementStore


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_init_does_not_discover_or_measure(self):
        path = self.root / 'new' / 'measurements.sqlite3'
        with patch('benchmark_suite.run_suite', side_effect=AssertionError('benchmark invoked')):
            for _ in range(2):
                self.assertEqual(benchmark_suite.main(['--init-db', '--database', str(path)]), 0)
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute('select count(*) from runs').fetchone()[0], 0)
            self.assertEqual(db.execute('select count(*) from bato_results').fetchone()[0], 0)
            self.assertEqual(db.execute('pragma integrity_check').fetchone()[0], 'ok')

    def test_competitor_table_backfill_updates_and_rolls_back(self):
        path = self.root / 'legacy.db'
        with sqlite3.connect(path) as db:
            db.execute('create table cases (run text, id text, record text not null, primary key(run,id))')
            db.execute('insert into cases values (?,?,?)', ('old', 'case', json.dumps({
                'case_id':'case', 'codec':'zip', 'target':'real/txt/a.txt', 'raw_size':42,
                'input_sha256':'hash', 'comp_size':30, 'comp_time_s':1.5,
                'status':'PASS & VERIFIED', 'verified_lossless':True})))
        store = MeasurementStore(path)
        self.addCleanup(store.close)
        db = store.connection
        row = db.execute('select run_id,filename,competitor,compressed_bytes,compress_seconds,decompress_seconds from competitor_measurements').fetchone()
        self.assertEqual(row, ('old', 'real/txt/a.txt', 'zip', 30, 1.5, None))
        store.save_case('old', {'case_id':'case', 'codec':'zip', 'status':'FAIL', 'error':'failed'})
        self.assertEqual(db.execute('select status,compressed_bytes from competitor_measurements').fetchone(), ('FAIL', None))
        with self.assertRaises(RuntimeError):
            with db:
                db.execute('update cases set record=?', (json.dumps({'codec':'gzip'}),))
                raise RuntimeError('rollback')
        self.assertEqual(db.execute('select competitor from competitor_measurements').fetchone()[0], 'zip')
        store.save_case('old', {'case_id':'case', 'codec':'bato-v1', 'status':'FAIL'})
        self.assertEqual(db.execute('select count(*) from competitor_measurements').fetchone()[0], 0)

    def test_unavailable_tool_has_no_fabricated_measurements(self):
        store = MeasurementStore(self.root / 'unavailable.db')
        self.addCleanup(store.close)
        metadata = {'status':'running', 'workloads':[{'workload_id':'wid', 'type':'single_file', 'target':'real/a',
                                  'logical_payload_bytes':10, 'snapshot':{'sha256':'sha'}}],
                    'tool_preflight':{'tools':[{'codec':'nncp','status':'MISSING','error':'not found'}]}}
        store.save_run('run', metadata)
        row = store.connection.execute('select competitor,status,compressed_bytes,compress_seconds,error from competitor_measurements').fetchone()
        self.assertEqual(row, ('nncp', 'UNAVAILABLE', None, None, 'not found'))
        store.save_run('run', metadata)
        self.assertEqual(store.connection.execute('select count(*) from competitor_measurements').fetchone()[0], 1)
        self.assertEqual(store.connection.execute('select count(*) from cases').fetchone()[0], 0)
        store.save_run('run', dict(metadata, status='interrupted'))
        store.save_case('run', {'case_id':'ready', 'codec':'zip', 'status':'RUNNING'})
        self.assertEqual(store.connection.execute('select distinct run_status from competitor_measurements').fetchall(), [('interrupted',)])

    def test_relocation_conflict_and_writer(self):
        source, dest = self.root / 'old.db', self.root / 'new.db'
        with sqlite3.connect(source) as db:
            db.execute('create table evidence (id integer primary key, value blob)')
            db.execute('insert into evidence values (1, ?)', (b'original',))
        with sqlite3.connect(source) as writer:
            writer.execute('begin immediate')
            with self.assertRaises(sqlite3.OperationalError):
                prepare_database(dest, source)
            self.assertFalse(dest.exists())
        prepare_database(dest, source)
        self.assertFalse(source.exists())
        with sqlite3.connect(dest) as db:
            self.assertEqual(db.execute('select value from evidence').fetchone()[0], b'original')
        with sqlite3.connect(source) as db:
            db.execute('create table unrelated (value text)')
        with self.assertRaisesRegex(ValueError, 'conflict'):
            prepare_database(dest, source)
        self.assertTrue(source.exists())

    def test_views_keep_campaigns_versions_failures_and_units(self):
        store = MeasurementStore(self.root / 'views.db')
        self.addCleanup(store.close)
        for run in ['old', 'new']:
            store.save_run(run, {'label':run, 'checkout_revision':run, 'requested_threads':8,
                                'environment':{'host':run}, 'resource_policy':{'memory_budget':1024},
                                'codecs':[{'id':'bato-v1', 'binary_sha256':run}]})
            for codec, size, status in [('zip', 30, 'PASS & VERIFIED'), ('gzip', 0, 'FAIL'),
                                        ('bato-v1', 20, 'PASS & VERIFIED')]:
                store.save_case(run, {'case_id':codec, 'workload_id':'file', 'track':'direct',
                                     'target':'real/txt/file.txt', 'input_sha256':'sha', 'raw_size':100,
                                     'codec':codec, 'comp_size':size, 'status':status,
                                     'verified_lossless':status=='PASS & VERIFIED', 'effective_settings':{},
                                     'comp_cpu_s':1.25, 'comp_peak_rss_kib':4096,
                                     'comp_sampled_tree_peak_rss_kib':8192})
        db = store.connection
        rows = db.execute('select run_id,zip_bytes,gzip_bytes,cmix_bytes from competitor_results order by run_id').fetchall()
        self.assertEqual(rows, [('new',30,None,None),('old',30,None,None)])
        rows = db.execute('select binary_sha256,comp_cpu_s,comp_peak_rss_kib,comp_sampled_tree_peak_rss_kib from bato_results order by run_id').fetchall()
        self.assertEqual(rows, [('new',1.25,4096,8192),('old',1.25,4096,8192)])
        detail = json.loads(db.execute('select codec_results from competitor_results limit 1').fetchone()[0])
        self.assertEqual(next(x for x in detail if x['codec']=='gzip')['status'], 'FAIL')
        self.assertEqual(db.execute('select count(*) from trials').fetchone()[0],0)


if __name__ == '__main__':
    unittest.main()
