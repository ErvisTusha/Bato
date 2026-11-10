"""Namespace-independent workflow I/O and versioned-engine preflight."""
import subprocess
import json
import sqlite3
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bato_tools.codecs import resolve_bato
from bato_tools.stage_worker import IO_FIELDS, io_counters, io_delta, execute
from bato_tools.common import snapshot, make_tar
from bato_tools.store import MeasurementStore
from bato_tools.compare import load_run


class WorkflowIOTests(unittest.TestCase):
    def test_comparison_rejects_missing_trials_and_stale_case_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = MeasurementStore(root/'measurements.sqlite3')
            row = {'case_id':'case','status':'PASS & VERIFIED','verified_lossless':True,
                   'cache_origin':'fresh','warmups':1,'repetitions':1,'trials':[]}
            store.save_run('run',{'schema_version':4,'status':'complete'})
            store.save_case('run',row)
            with self.assertRaisesRegex(ValueError,'Missing committed'):
                load_run(root,'run')
            for phase in ['warmup','measured']:
                trial = {'phase':phase,'number':0,'status':'PASS','verified_lossless':True}
                store.save_trial('run','case',phase,0,trial)
            with self.assertRaisesRegex(ValueError,'projection differs'):
                load_run(root,'run')
            row['trials'] = [trial]
            store.save_case('run',row)
            self.assertEqual(load_run(root,'run')[1],[row])
            store.close()

    def test_comparison_requires_canonical_database_for_v4(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root/'runs/run'
            run.mkdir(parents=True)
            (run/'run.json').write_text(json.dumps({'schema_version':4}))
            (run/'results.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'canonical measurements'):
                load_run(root,'run')

    def test_replaced_database_reopens_and_keeps_trial_immutability(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'measurements.sqlite3'
            store = MeasurementStore(path)
            try:
                trial = {'phase':'measured','number':0,'status':'PASS','verified_lossless':True}
                store.save_trial('run','case','measured',0,trial)
                replacement = Path(temporary)/'replacement.sqlite3'
                with sqlite3.connect(replacement) as copy:
                    store.connection.backup(copy)
                os.replace(replacement,path)
                store.save_run('run',{'status':'running'})
                self.assertEqual(store.trials('run','case'),[trial])
                with self.assertRaisesRegex(ValueError,'already committed'):
                    store.save_trial('run','case','measured',0,{**trial,'status':'FAIL'})
            finally: store.close()
    def test_database_completeness_rejects_missing_committed_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = MeasurementStore(Path(temporary)/'measurements.sqlite3')
            try:
                self.assertEqual(store.connection.execute('PRAGMA journal_mode').fetchone()[0], 'delete')
                row = {'case_id':'case','status':'PASS & VERIFIED','verified_lossless':True,
                       'cache_origin':'fresh','warmups':1,'repetitions':1}
                store.save_case('run',row)
                with self.assertRaisesRegex(ValueError, 'Missing committed'):
                    store.validate('run', {'case'})
                for phase in ['warmup','measured']:
                    store.save_trial('run','case',phase,0,{'phase':phase,'number':0,'status':'PASS','verified_lossless':True})
                row['trials'] = [{'phase':'measured','number':0,'status':'PASS','verified_lossless':True}]
                store.save_case('run',row)
                store.validate('run', {'case'})
                with self.assertRaisesRegex(ValueError, 'Incomplete canonical case inventory'):
                    store.validate('run', {'case','other'})
            finally: store.close()
    def test_common_tar_verification_checks_files_and_empty_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'
            (source/'empty/nested').mkdir(parents=True)
            (source/'payload').write_bytes(b'original file bytes')
            archive = root/'restored.tar'
            make_tar(source, archive)
            original = snapshot(source)
            spec = {'action':'verify', 'source':str(archive), 'expected':snapshot(archive), 'tar_directory_snapshot':original}
            self.assertTrue(execute(spec)['verified_lossless'])
            (source/'empty/nested').rmdir()
            make_tar(source, archive)
            spec['expected'] = snapshot(archive)
            with self.assertRaisesRegex(ValueError, 'files or directories differ'):
                execute(spec)
    def test_unavailable_or_reset_io_is_unknown(self):
        counters = {k: 100 for k in IO_FIELDS}
        self.assertTrue(all(v is None for v in io_delta(None, counters).values()))
        self.assertTrue(all(v is None for v in io_delta(counters, None).values()))
        self.assertTrue(all(v is None for v in io_delta(counters, {**counters, 'wchar': 99}).values()))

    def test_waited_child_io_is_included_when_linux_accounting_is_available(self):
        if io_counters() is None:
            self.skipTest('Linux self I/O counters unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            before = io_counters()
            subprocess.run([sys.executable, '-c',
                            'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.write_bytes(b"x"*131072); assert len(p.read_bytes())==131072',
                            str(Path(temporary)/'child-data')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            delta = io_delta(before, io_counters())
            self.assertGreaterEqual(delta['io_rchar'], 131072)
            self.assertGreaterEqual(delta['io_wchar'], 131072)
            # Page-cache hits may legitimately yield zero storage reads.
            self.assertGreaterEqual(delta['io_read_bytes'], 0)

    def test_current_binary_is_accepted_by_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary)/'bato'
            binary.write_text('fixture')
            binary.chmod(0o700)
            with patch('bato_tools.codecs.probe_command', side_effect=[
                (0, 'Pure Rust Bato --segment-size --chunk-size'), (0, 'bato 0.3.0 (format 1.0)')]):
                path, version = resolve_bato(binary)
            self.assertEqual(path, binary.resolve())
            self.assertIn('format 1.0', version)
