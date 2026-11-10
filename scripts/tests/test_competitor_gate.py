"""Competitor baselines must precede tests and remain original observations."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_tests
from bato_tools.baselines import BaselineRequest, CONDITION_FIELDS, find_baseline, validate_references
from bato_tools.measurement import aggregate
from bato_tools.runner import _run_suite
from bato_tools.store import MeasurementStore
from test_benchmark_v4 import observation, options


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = MeasurementStore(self.root / 'measurements.sqlite3')
        self.addCleanup(self.store.close)
        self.meta = {k:'same' for k in CONDITION_FIELDS}
        self.meta.update(schema_version=4,run_id='original')
        base = dict(case_id='case',workload_id='file',track='direct',codec='zip',input_sha256='sha',
                    raw_size=100,binary_identity={'binary_sha256':'binary'},effective_settings={'level':9},
                    resource_policy={'limit':1000},repetitions=1,warmups=0,cache_origin='fresh')
        trial = dict(observation(),phase='measured',number=0)
        self.case = aggregate(base,[trial])
        self.store.save_run('original',self.meta)
        self.store.save_case('original',self.case)
        self.store.save_trial('original','case','measured',0,trial)
        self.request = BaselineRequest(dict(self.meta,run_id='candidate'),base)

    def test_exact_baseline_and_missing_original_size(self):
        ref = find_baseline(self.store,self.request)
        self.assertEqual(ref['run_id'],'original')
        self.assertEqual(self.store.cases('candidate'),[])
        self.assertEqual(self.store.trials('candidate','case'),[])
        for field in ['raw_size','input_sha256','binary_identity','effective_settings','warmups','repetitions']:
            changed = dict(self.request.case)
            changed[field] = 'changed'
            self.assertIsNone(find_baseline(self.store,BaselineRequest(self.request.metadata,changed)),field)
        for field in CONDITION_FIELDS:
            changed = dict(self.request.metadata)
            changed[field] = 'different'
            self.assertIsNone(find_baseline(self.store,BaselineRequest(changed,self.request.case)),field)

    def test_failed_or_uncommitted_measurements_do_not_satisfy_gate(self):
        for field,value in [('status','TIMEOUT'),('verified_lossless',False),('raw_size',None),('comp_size',None)]:
            self.store.save_case('original',dict(self.case,**{field:value}))
            self.assertIsNone(find_baseline(self.store,self.request))
        self.store.save_case('original',self.case)
        self.store.connection.execute('delete from trials')
        self.store.connection.commit()
        self.assertIsNone(find_baseline(self.store,self.request))

    def test_deleted_reference_cannot_be_validated(self):
        ref = find_baseline(self.store,self.request)
        meta = dict(self.request.metadata,baseline_references={'case':ref})
        validate_references(self.store,meta)
        self.store.connection.execute('delete from cases')
        self.store.connection.commit()
        with self.assertRaisesRegex(ValueError,'missing'):
            validate_references(self.store,meta)


class GateOrderTests(unittest.TestCase):
    def test_unavailable_baseline_stops_before_bato_rules(self):
        sample = type('Sample',(),dict(relative_path='real/txt/a.txt',size=1,digest='sha'))()
        with patch('run_tests.select_samples',return_value=[sample]), \
             patch('run_tests.resolve_bato',return_value=(Path('/unused'),'fixture')), \
             patch('run_tests.ensure_test_baselines',return_value=2), \
             patch.object(unittest.TextTestRunner,'run',side_effect=AssertionError('Bato started')), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_tests.main(['--min']),2)


    def test_partial_baselines_continue_correctness_and_return_nonzero(self):
        sample = type('Sample',(),dict(relative_path='real/txt/a.txt',size=1,digest='sha'))()
        with patch('run_tests.select_samples',return_value=[sample]), \
             patch('run_tests.resolve_bato',return_value=(Path('/unused'),'fixture')), \
             patch('run_tests.ensure_test_baselines',return_value=1), \
             patch.object(unittest.TextTestRunner,'run',return_value=unittest.TestResult()) as run, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_tests.main(['--min']),1)
        run.assert_called_once()


class EnsureCampaignTests(unittest.TestCase):
    def test_only_missing_cases_are_measured_and_references_are_not_copied(self):
        # Reuse the established harness fixtures, not a second measurement implementation.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            from bato_tools.codecs import Codec
            codec = Codec('zip','ZIP',sys.executable,version='fixture')
            (root/'corpus').mkdir()
            (root/'corpus'/'file.txt').write_bytes(b'original input' * 10)
            args = options(root,suite='corpus',include_local=True,workload='files',track='native',
                           ensure_competitors=True,profile='high',cpu_limit=1,threads=1)
            for run in ['first','second']:
                args.run_id = run
                (Path(args.output_dir)/'runs'/run).mkdir(parents=True)
                Path(args.scratch_dir).mkdir(parents=True,exist_ok=True)
                with patch('bato_tools.runner.environment',return_value={'host':'same'}), \
                     patch('bato_tools.runner.measure_trial',return_value=observation()) as measure, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(_run_suite(args,[codec],{'status':'ready'}),0)
                self.assertEqual(measure.call_count,2 if run=='first' else 0)
            args.run_id = 'partial'
            (Path(args.output_dir)/'runs'/args.run_id).mkdir(parents=True)
            gzip = Codec('gzip','GZIP',sys.executable,version='fixture')
            with patch('bato_tools.runner.environment',return_value={'host':'same'}), \
                 patch('bato_tools.runner.measure_trial',return_value=observation()) as measure, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(_run_suite(args,[codec,gzip],{'status':'ready'}),0)
            self.assertEqual(measure.call_count,2)  # Only gzip is missing, ZIP references its original trials.
            store = MeasurementStore(Path(args.output_dir)/'measurements.sqlite3')
            try:
                self.assertEqual(len(store.cases('first')),1)
                self.assertEqual(store.cases('second'),[])
                self.assertEqual(len(store.run('second')['baseline_references']),1)
                self.assertEqual([c['codec'] for c in store.cases('partial')],['gzip'])
                self.assertEqual(len(store.run('partial')['baseline_references']),1)
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
