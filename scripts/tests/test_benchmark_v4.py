"""Regression coverage for selection, trial durability, limits, and fair comparisons."""
import contextlib
import copy
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_suite
from compare_benchmarks import reference_competitors
from bato_tools.codecs import Codec, WORKER, probe_command
from bato_tools.common import ProcessFailure, make_tar, run_process, snapshot
from bato_tools.compare import compare_runs, load_run, write_comparison
from bato_tools.measurement import aggregate, base_record, measure_trial, run_stage
from bato_tools.reporting import leaderboards, summarize
from bato_tools.resources import ResourceLimit, admit_disk, proc_pids_match
from bato_tools.runner import _run_suite, host_measurement_lock, repetitions_for, select_workloads
from bato_tools.store import MeasurementStore, validate_case_records
from bato_tools.workloads import (GENERATOR_VERSION, NORMALIZED_TIME, directory_workloads,
                                    discover, identify, materialize, stress_workloads)


def options(root, **changes):
    values = dict(corpus=str(root/'corpus'), output_dir=str(root/'reports'), scratch_dir=str(root/'scratch'),
                  suite='stress', mode='rigorous', track='both', workload='both', target=[], category=[],
                  origin=None, include_local=False, include_duplicates=False, seed=42, max_files=50,
                  sample_bytes=256*2**20, max_archives=10, max_file_bytes=4*2**30,
                  max_directory_bytes=4*2**30, max_directory_files=100000, repetitions=2, warmups=0,
                  large_repetitions=None, large_warmups=None, memory_budget=2*2**30, timeout=10,
                  run_id='fixture', resume=None, label='fixture build')
    return SimpleNamespace(**(values | changes))


def observation(size=50):
    return dict(status='PASS', verified_lossless=True, comp_size=size, input_bytes=100,
                comp_time_s=.01, decomp_time_s=.02, comp_cpu_s=.008, decomp_cpu_s=.012,
                comp_peak_rss_kib=1024, decomp_peak_rss_kib=2048, cpu_scope='isolated_workflow_including_container_and_children')


class BenchmarkV4Tests(unittest.TestCase):
    def test_extreme_paq_and_zpaq_settings_are_explicit(self):
        paq = Codec('paq8px', 'PAQ8PX', '/tool/paq8px', version='paq8px v217')
        self.assertEqual(paq.compression_options(), ['-12L', '-lstmlayers=5'])
        self.assertEqual(paq.archive_suffix(), '.paq8px217')
        zpaq = Codec('zpaq', 'ZPAQ', '/tool/zpaq', threads=3)
        self.assertEqual(zpaq.compression_options(), ['-method', '5', '-threads', '3', '-noattributes'])
        self.assertEqual(zpaq.settings()['decompression_workers'], 3)
        self.assertEqual(zpaq.settings()['directory_container'], 'tar')

    def test_opt_in_cmix_and_nncp_models_are_explicit(self):
        from bato_tools.codecs import DEFAULT_CODECS, selected_names
        self.assertNotIn("cmix", DEFAULT_CODECS)
        self.assertNotIn("nncp", DEFAULT_CODECS)
        self.assertEqual(selected_names("bato,cmix,nncp,zpaq,paq8px"),
                         ["bato", "cmix", "nncp", "zpaq", "paq8px"])
        cmix = Codec("cmix", "CMIX", "/tool/cmix")
        nncp = Codec("nncp", "NNCP", "/tool/nncp", threads=6)
        self.assertEqual(cmix.level, 0)
        self.assertEqual(nncp.level, 0)
        self.assertEqual(nncp.compression_options(), ["-T", "6", "--profile", "default"])
        self.assertEqual(cmix.settings()["directory_container"], "tar")

    def test_common_tar_uses_frozen_members_without_dropping_hidden_payload(self):
        import tarfile
        source = self.root/'tree'
        source.mkdir()
        (source/'.legitimate').write_bytes(b'payload')
        frozen = snapshot(source)
        (source/'.unexpected.partial').write_bytes(b'not in frozen inputs')
        archive = self.root/'data.tar'
        make_tar(source, archive, frozen)
        with tarfile.open(archive) as packed:
            self.assertEqual(packed.getnames(), ['.legitimate'])
            self.assertEqual(packed.extractfile('.legitimate').read(), b'payload')

    def test_prepare_rejects_input_added_during_tar_creation(self):
        from bato_tools.stage_worker import execute
        source = self.root/'original'
        source.mkdir()
        (source/'file').write_bytes(b'original')
        work = {'type':'directory_archive','path':str(source),'snapshot':snapshot(source)}
        destination = self.root/'staged'
        real_make_tar = make_tar
        def changed_tar(src, output, expected):
            real_make_tar(src, output, expected)
            (Path(src)/'.injected').write_bytes(b'changed')
        with patch.dict(os.environ), patch('bato_tools.stage_worker.make_tar', side_effect=changed_tar):
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                execute({'action':'prepare','workload':work,'destination':str(destination),'tar_path':str(self.root/'payload.tar')})

    def test_paq_zero_exit_allocation_failure_is_not_a_success(self):
        paq = Codec('paq8px', 'PAQ8PX', '/tool/paq8px', version='paq8px v217')
        work = self.root/'paq'
        work.mkdir()
        def allocation_failure(*args, **kwargs):
            Path(kwargs['stdout_path']).write_text('Creating archive...\nOut of memory.\n')
            return SimpleNamespace()
        with patch('bato_tools.codecs.run_process', side_effect=allocation_failure):
            with self.assertRaisesRegex(MemoryError, 'Out of memory'):
                paq.step('compress', self.root/'input', self.root/'archive', work, 10, 100)

    def test_optional_checkpoint_preserves_committed_measurements(self):
        import sqlite3
        args = options(self.root, target=['exact/random/100x1024'], workload='directories',
                       checkpoint_dir=str(self.root/'mirror'))
        self.initialize(args)
        with patch('bato_tools.runner.environment', return_value={'fixture':'same'}), \
             patch('bato_tools.runner.measure_trial', return_value=observation()), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(_run_suite(args, [self.codec], {'status':'ready'}), 0)
        with sqlite3.connect(self.root/'mirror/measurements.sqlite3') as database:
            self.assertEqual(database.execute('select count(*) from trials').fetchone()[0], 4)
            self.assertEqual(database.execute('pragma integrity_check').fetchone()[0], 'ok')
        self.assertTrue((self.root/'mirror/runs/fixture/run.json').is_file())

    def reference_fixture(self):
        meta = {'schema_version':4, 'run_id':'candidate', 'environment':{'host':'fixture'},
                'effective_memory_budget':2**30, 'resource_policy':{'cap':'fixture'},
                'cpu_limit':1, 'requested_threads':1, 'compression_profile':'high',
                'mode':'rigorous', 'suite':'corpus', 'track':'both', 'seed':42, 'timeout':120,
                'codecs':[{'id':'bato-v1'}, {'id':'zip'}], 'status':'complete'}
        row = {'codec':'bato-v1', 'target':'real/file', 'workload_id':'fixture', 'track':'direct',
               'status':'PASS & VERIFIED', 'verified_lossless':True, 'cache_origin':'fresh',
               'warmups':1, 'trials':[observation(40) for _ in range(5)], 'comp_size':40,
               'comp_time_s':.01, 'decomp_time_s':.02, 'comp_cpu_s':.008, 'decomp_cpu_s':.012,
               'input_sha256':'input', 'codec_input_sha256':'input', 'metadata_policy':'fixture',
               'raw_size':100, 'compression_profile':'extreme'}
        peer = dict(row, codec='zip', comp_size=50, trials=[observation(50) for _ in range(5)])
        return meta, row, dict(meta, run_id='reference'), peer

    def test_reference_competitors_preserve_trial_provenance(self):
        meta, row, reference, peer = self.reference_fixture()
        result = reference_competitors(meta, [row], reference, [peer], 'bato-v1')
        self.assertEqual(result['comparisons'][0]['outcomes']['comp_size'], 'win')
        self.assertEqual(result['competitor_run_id'], 'reference')
        self.assertEqual(result['comparisons'][0]['bato_run_id'], 'candidate')
        self.assertEqual(result['missing_reference_cases'], [])
        self.assertFalse(result['overall_goal_complete'])

    def test_reference_competitors_reject_mismatched_allocations_and_limits(self):
        meta, row, reference, peer = self.reference_fixture()
        for field, value in [('cpu_limit',8), ('requested_threads',8), ('effective_memory_budget',2**31),
                             ('environment',{'host':'other'}), ('compression_profile','fast')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                reference_competitors(meta, [row], dict(reference, **{field:value}), [peer], 'bato-v1')

    def test_reference_competitors_exclude_cached_short_or_failed_observations(self):
        meta, row, reference, peer = self.reference_fixture()
        for changes in [{'cache_origin':'cached'}, {'warmups':0}, {'trials':[]},
                        {'verified_lossless':False, 'status':'TIMEOUT'}]:
            result = reference_competitors(meta, [row], reference, [dict(peer, **changes)], 'bato-v1')
            self.assertEqual(set(result['comparisons'][0]['outcomes'].values()), {'unknown'})
        result = reference_competitors(meta, [row], reference, [], 'bato-v1')
        self.assertEqual(len(result['missing_reference_cases']), 1)
        self.assertEqual(result['comparisons'], [])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.codec = Codec('zip', 'ZIP', sys.executable, version='fixture')

    def tearDown(self):
        self.temporary.cleanup()

    def initialize(self, args):
        (Path(args.output_dir)/'runs'/args.run_id).mkdir(parents=True)
        Path(args.scratch_dir).mkdir(parents=True, exist_ok=True)

    def test_duplicate_selection_keeps_all_archive_members_and_nested_root(self):
        corpus = self.root/'corpus'
        (corpus/'nested').mkdir(parents=True)
        for i in range(100):
            (corpus/('nested' if i%2 else '')/f'{i:03d}.bin').write_bytes(b'repeat\n'*100)
        args = options(self.root, suite='corpus', include_local=True)
        selected, exclusions, _ = select_workloads(args)
        files = [row for row in selected if row['type']=='single_file']
        directories = [row for row in selected if row['type']=='directory_archive']
        self.assertEqual(len(files), 1)
        self.assertEqual(len(directories), 1)
        self.assertEqual(directories[0]['file_count'], 100)
        self.assertEqual(directories[0]['logical_payload_bytes'], 70000)
        self.assertEqual(sum(row['reason']=='duplicate single-file case' for row in exclusions), 99)

    def test_directory_filter_never_silently_archives_unselected_files(self):
        corpus = self.root/'corpus'
        corpus.mkdir()
        (corpus/'a.bin').write_bytes(b'one')
        (corpus/'b.bin').write_bytes(b'two')
        files, _ = discover(corpus, include_local=True, include_duplicates=True)
        directories, excluded = directory_workloads(corpus, files[:1])
        self.assertEqual(directories, [])
        self.assertIn('outside selection', excluded[0]['reason'])

    def test_generated_manifest_is_lazy_bounded_and_versioned(self):
        args = options(self.root)
        with patch('bato_tools.workloads.materialize', side_effect=AssertionError('Must stay lazy')):
            selected, _, _ = select_workloads(args)
        self.assertFalse((self.root/'corpus').exists())
        self.assertEqual(max(row['logical_payload_bytes'] for row in selected), 4*2**30)
        self.assertEqual(max(row['file_count'] for row in selected), 100000)
        self.assertTrue(all(row['recipe']['generator']==GENERATOR_VERSION for row in selected))
        changed, excluded, _ = select_workloads(options(self.root, max_file_bytes=2**20, max_directory_bytes=2**20))
        self.assertTrue(all(row['logical_payload_bytes']<=2**20 for row in changed))
        self.assertTrue(any(row['reason']=='workload limits' for row in excluded))

    def test_cli_manifest_needs_neither_tools_nor_output(self):
        output = self.root/'reports'
        with patch('bato_tools.runner.check_tools', side_effect=AssertionError('Must not preflight')), contextlib.redirect_stdout(io.StringIO()) as console:
            result = benchmark_suite.main(['--suite','stress','--list-workloads','--mode','rigorous',
                                           '--target','exact/random/100x1024','--output-dir',str(output)])
        self.assertEqual(result,0)
        self.assertEqual(json.loads(console.getvalue())['workloads'][0]['file_count'],100)
        self.assertFalse(output.exists())

    def test_exact_near_and_mixed_duplicate_contents(self):
        catalog = stress_workloads()
        fractions = {}
        for scenario in ['exact_duplicates','near_duplicates','mixed_duplicates']:
            item = next(row for row in catalog if row['scenario']==scenario and row['recipe']['content']=='random')
            item = copy.deepcopy(item)
            item['recipe'].update(file_size=10000, count=10)
            details = materialize(item, self.root/scenario)
            fractions[scenario] = details['duplicate_byte_fraction']
            self.assertEqual(len(details['snapshot']['files']),10)
            repeated = materialize(item, self.root/(scenario+'-again'))
            self.assertEqual(details['snapshot'],repeated['snapshot'])
            if scenario=='near_duplicates':
                # Each member mutates one percent of the shared base, at distinct offsets.
                original = copy.deepcopy(item)
                original['recipe']['scenario']='exact_duplicates'
                materialize(original,self.root/'base')
                before = (self.root/'base/000000.bin').read_bytes()
                for path in (self.root/scenario).glob('*.bin'):
                    self.assertEqual(sum(a!=b for a,b in zip(before,path.read_bytes())),100)
        self.assertAlmostEqual(fractions['exact_duplicates'],.9)
        self.assertEqual(fractions['near_duplicates'],0)
        self.assertAlmostEqual(fractions['mixed_duplicates'],.4)

    def test_staging_normalizes_real_copies_without_touching_source_metadata(self):
        source = self.root/'original'
        source.mkdir()
        payload = source/'payload.bin'
        payload.write_bytes(b'payload')
        payload.chmod(0o600)
        os.utime(payload,(1000000000,1000000000))
        before = payload.stat()
        item = dict(target='original',path=str(source),type='directory_archive',category='binary',origin='local',snapshot=snapshot(source))
        staged = self.root/'staged'
        details = materialize(item,staged)
        self.assertEqual(details['snapshot'],item['snapshot'])
        self.assertEqual(payload.stat().st_mtime_ns,before.st_mtime_ns)
        self.assertEqual(payload.stat().st_mode,before.st_mode)
        self.assertNotEqual(payload.stat().st_ino,(staged/'payload.bin').stat().st_ino)
        self.assertEqual((staged/'payload.bin').stat().st_mode&0o777,0o644)
        self.assertEqual((staged/'payload.bin').stat().st_mtime,NORMALIZED_TIME)
        a,b = self.root/'a.tar',self.root/'b.tar'
        make_tar(source,a)
        make_tar(staged,b)
        self.assertEqual(snapshot(a),snapshot(b))

    def test_large_trial_defaults_and_explicit_overrides(self):
        item = next(row for row in stress_workloads() if row['target']=='stress/files/random/4294967296.bin')
        args = options(self.root,repetitions=None,warmups=None)
        self.assertEqual(repetitions_for(args,item),(1,0))
        small = next(row for row in stress_workloads() if row['target']=='stress/files/random/4096.bin')
        self.assertEqual(repetitions_for(args,small),(5,1))
        args.large_repetitions=5
        args.large_warmups=1
        self.assertEqual(repetitions_for(args,item),(5,1))

    def test_size_history_and_single_observation_do_not_invent_dispersion(self):
        item = identify(next(row for row in stress_workloads() if row['target']=='stress/files/random/64.bin'))
        base = base_record(self.codec,item,1,0)
        record = aggregate(base,[observation()])
        self.assertEqual(record['timing_quality'],'single_observation')
        self.assertIsNone(record['comp_time_stdev_s'])
        self.assertIsNone(record['comp_time_mad_s'])
        self.assertIsNone(record['size_deterministic'])
        base['repetitions']=2
        record = aggregate(base,[observation(50),observation(60)])
        self.assertFalse(record['size_deterministic'])
        self.assertEqual((record['comp_size_min'],record['comp_size_max']),(50,60))

    def comparison_fixture(self):
        item = identify(next(row for row in stress_workloads() if row['target']=='stress/files/random/64.bin'))
        row = aggregate(base_record(self.codec,item,2,0),[observation(),observation()])
        row.update(input_sha256='bytes',codec_input_sha256='bytes',binary_identity={'binary_sha256':'old'})
        metadata = dict(schema_version=4,run_id='old',status='complete',mode='rigorous',environment={'host':'same'},
                        effective_memory_budget=2**30,resource_policy={'memory':'per process'})
        return metadata,row

    def test_comparison_allows_build_changes_and_rejects_wrong_inputs(self):
        metadata,row = self.comparison_fixture()
        candidate = {**row,'binary_identity':{'binary_sha256':'new'},'comp_size':40,'comp_size_min':40,'comp_size_max':40}
        result = compare_runs(metadata,[row],{**metadata,'run_id':'new'},[candidate])
        self.assertEqual(result['comparisons'][0]['size_improvement_percent'],20)
        self.assertTrue(result['comparisons'][0]['timing_comparable'])
        self.assertEqual(result['summary'][0]['smaller'],1)
        invalid = compare_runs(metadata,[row],{**metadata,'run_id':'new'},[{**candidate,'codec_input_sha256':'changed'}])
        self.assertEqual(invalid['matched_cases'],0)
        self.assertFalse(invalid['complete_coverage'])
        with self.assertRaises(ValueError):
            compare_runs({**metadata,'schema_version':3},[row],metadata,[candidate])

    def test_cached_single_trial_and_different_host_disable_timing_comparisons(self):
        metadata,row = self.comparison_fixture()
        for changes, meta_changes in [({'cache_origin':'baseline'},{}),({'timing_quality':'single_observation'},{}),
                                     ({},{'environment':{'host':'different'}}),({},{'effective_memory_budget':2**29}),
                                     ({},{'environment':None}),({},{'resource_policy':None})]:
            result = compare_runs(metadata,[row],{**metadata,'run_id':'new',**meta_changes},[{**row,**changes}])
            self.assertEqual(result['matched_cases'],1)
            self.assertFalse(result['comparisons'][0]['timing_comparable'])
            self.assertIsNone(result['comparisons'][0]['comp_time_change_percent'])

    def test_overlapping_size_ranges_and_missing_input_identity_are_visible(self):
        metadata,row = self.comparison_fixture()
        previous = {**row,'size_deterministic':False,'comp_size_min':40,'comp_size_max':60}
        candidate = {**row,'comp_size':45,'size_deterministic':False,'comp_size_min':35,'comp_size_max':55}
        result = compare_runs(metadata,[previous],{**metadata,'run_id':'new'},[candidate])
        self.assertEqual(result['comparisons'][0]['size_outcome'],'variable_overlap')
        self.assertEqual(result['summary'][0]['variable_overlap'],1)
        result = compare_runs(metadata,[row],{**metadata,'run_id':'new'},[{**row,'input_sha256':None}])
        self.assertEqual(result['matched_cases'],0)
        self.assertIn('identity is missing',result['exclusions'][0]['reason'])

    def test_reports_keep_origin_track_and_failure_coverage_separate(self):
        metadata,row = self.comparison_fixture()
        rows = [row,{**row,'codec':'gzip','status':'FAIL','verified_lossless':False},
                {**row,'origin':'real'},{**row,'track':'common-tar'},{**row,'cache_origin':'baseline','codec':'xz'}]
        summaries = summarize(rows)
        cached = next(summary for summary in summaries if summary['codec']=='xz')
        self.assertIsNone(cached['comp_mib_s'])
        boards = leaderboards(rows,metadata)
        self.assertEqual(len(boards),3)
        failed = next(board for board in boards if board['origin']=='synthetic' and board['track']=='direct')
        self.assertFalse(failed['winner_claim_eligible'])
        self.assertEqual(failed['ranking'][0]['common_workloads'],0)
        self.assertEqual(failed['ranking'][0]['intended_workloads'],1)

    def test_committed_trials_are_immutable_and_comparison_uses_database(self):
        database = self.root/'measurements.sqlite3'
        store = MeasurementStore(database)
        metadata,row = self.comparison_fixture()
        trials = [{**observation(),'phase':'measured','number':number} for number in range(2)]
        row = aggregate(row,trials)
        trial = {**observation(),'phase':'measured','number':0}
        store.save_trial('old',row['case_id'],'measured',0,trial)
        store.save_trial('old',row['case_id'],'measured',0,trial)
        with self.assertRaises(ValueError):
            store.save_trial('old',row['case_id'],'measured',0,{**trial,'comp_size':99})
        store.save_run('old',metadata)
        store.save_case('old',row)
        store.save_run('new',{**metadata,'run_id':'new'})
        store.save_case('new',row)
        store.save_trial('old',row['case_id'],'measured',1,trials[1])
        for trial in trials:
            store.save_trial('new',row['case_id'],'measured',trial['number'],trial)
        store.close()
        self.assertEqual(load_run(self.root,'old')[1],[row])
        with patch('bato_tools.runner.check_tools',side_effect=AssertionError('No compressors needed')):
            directory,result = write_comparison(self.root,'old','new')
        self.assertEqual(result['matched_cases'],1)
        self.assertTrue((directory/'comparison.md').is_file())

    def test_partial_database_load_does_not_require_complete_inventory(self):
        database = self.root/'measurements.sqlite3'
        store = MeasurementStore(database)
        metadata, row = self.comparison_fixture()
        metadata.update(status='running', run_id='partial')
        row = {**row, 'status': 'FAIL', 'verified_lossless': False}
        store.save_run('partial', metadata)
        store.save_case('partial', row)
        store.close()
        loaded, records = load_run(self.root, 'partial')
        self.assertEqual(loaded['status'], 'running')
        self.assertEqual(records, [row])

    def test_fresh_verified_cases_cannot_bypass_projection_validation_with_wrong_status(self):
        _, row = self.comparison_fixture()
        row = {**row, 'status': 'FAIL', 'verified_lossless': True, 'cache_origin': 'fresh'}
        with self.assertRaises(ValueError):
            validate_case_records([row], {row['case_id']: []})

    def test_recovery_rebuilds_missing_reports_and_only_measures_missing_trial(self):
        args = options(self.root,target=['stress/files/random/64.bin'])
        self.initialize(args)
        original = MeasurementStore.save_case
        interrupted = []
        def crash_after_trial(store,run,record):
            if store.trials(run,record['case_id']) and not interrupted:
                interrupted.append(True)
                raise KeyboardInterrupt()
            return original(store,run,record)
        with patch('bato_tools.runner.environment',return_value={'fixture':'same'}), patch('bato_tools.runner.measure_trial',return_value=observation()), contextlib.redirect_stdout(io.StringIO()), patch.object(MeasurementStore,'save_case',crash_after_trial):
            with self.assertRaises(KeyboardInterrupt):
                _run_suite(args,[self.codec],{'status':'ready'})
        run = Path(args.output_dir)/'runs'/args.run_id
        (run/'trials.jsonl').write_text('partial broken journal')
        (run/'results.json').unlink()
        (run/'run.json').unlink()
        args.resume=args.run_id
        with patch('bato_tools.runner.environment',return_value={'fixture':'same'}), patch('bato_tools.runner.measure_trial',return_value=observation()) as measure, patch('bato_tools.runner.check_tools',return_value=([self.codec],{'status':'ready'})), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(benchmark_suite.main(['--suite','stress','--mode','rigorous','--target',args.target[0],
                                                  '--repetitions','2','--warmups','0','--timeout','10','--memory-budget',str(args.memory_budget),
                                                  '--output-dir',args.output_dir,'--database',str(Path(args.output_dir)/'measurements.sqlite3'),'--scratch-dir',args.scratch_dir,'--resume',args.run_id]),0)
        self.assertEqual(measure.call_count,1)
        rows = [json.loads(line) for line in (run/'trials.jsonl').read_text().splitlines()]
        self.assertEqual([row['number'] for row in rows],[0,1])
        self.assertEqual(json.loads((run/'results.json').read_text())[0]['completed_trials'],2)
        self.assertEqual(json.loads((run/'run.json').read_text())['status'],'complete')

    def test_common_tar_is_shared_and_codec_order_rotates_between_rounds(self):
        args = options(self.root,target=['exact/random/100x1024'],workload='directories')
        self.initialize(args)
        calls = []
        def trial(codec,item,*unused):
            calls.append((codec.identifier,item['track'],item['tar_path'],item.get('tar_snapshot')))
            return observation()
        codecs = [self.codec,Codec('bzip2','BZIP2',sys.executable,version='fixture')]
        with patch('bato_tools.runner.measure_trial',side_effect=trial), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(_run_suite(args,codecs,{'status':'ready'}),0)
        self.assertEqual(len(calls),8)
        self.assertEqual([(row[0],row[1]) for row in calls[4:]],[(row[0],row[1]) for row in calls[1:4]+calls[:1]])
        tars = [row for row in calls if row[1]=='common-tar']
        self.assertEqual(len({row[2] for row in tars}),1)
        self.assertTrue(all(row[3]==tars[0][3] for row in tars))
        results = json.loads((Path(args.output_dir)/'runs'/args.run_id/'results.json').read_text())
        self.assertTrue(all(row['duplicate_byte_fraction']==.99 for row in results))

    def test_disk_limit_is_recorded_for_every_intended_case(self):
        args = options(self.root,target=['exact/random/100x1024'],workload='directories')
        self.initialize(args)
        with patch('bato_tools.runner.admit_disk',side_effect=ResourceLimit('no disk')), patch('bato_tools.runner.measure_trial',side_effect=AssertionError('Must not measure')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(_run_suite(args,[self.codec],{'status':'ready'}),1)
        results = json.loads((Path(args.output_dir)/'runs'/args.run_id/'results.json').read_text())
        self.assertEqual(len(results),2)
        self.assertTrue(all(row['status']=='RESOURCE_LIMIT' for row in results))
        with patch('bato_tools.resources.shutil.disk_usage',return_value=SimpleNamespace(free=1)):
            with self.assertRaises(ResourceLimit):
                admit_disk(self.root,1024,100)

    def test_isolated_phases_verify_zero_byte_and_duplicate_round_trips(self):
        for target in ['stress/files/random/0.bin','stress/duplicates/exact/random/100x1024']:
            item = identify(next(row for row in stress_workloads() if row['target']==target))
            work = self.root/str(item['file_count'])
            work.mkdir()
            source = work/'input'
            prepared = run_stage({'action':'prepare','workload':item,'destination':str(source)},work,10,2**30)
            self.assertEqual(prepared['status'],'PASS',prepared)
            row = measure_trial(self.codec,{**item,'path':str(source),'snapshot':prepared['snapshot']},10,work,True,2**30)
            self.assertEqual(row['status'],'PASS',row)
            self.assertGreater(row['comp_time_s'],0)
            self.assertIsNotNone(row['comp_cpu_s'])
            self.assertIsNotNone(row['comp_worker_peak_rss_kib'])

    def test_address_space_limit_becomes_structured_resource_failure(self):
        item = identify(next(row for row in stress_workloads() if row['target']=='stress/files/random/16777216.bin'))
        result = run_stage({'action':'prepare','workload':item,'destination':str(self.root/'input')},self.root,5,24*2**20)
        self.assertEqual(result['status'],'RESOURCE_LIMIT',result)
        self.assertEqual(result['stage'],'prepare')

    @unittest.skipUnless(shutil.which('gzip'), 'gzip is not installed')
    def test_native_stream_reports_tar_bytes_separately_from_logical_payload(self):
        item = identify(next(row for row in stress_workloads() if row['target']=='stress/duplicates/exact/random/100x1024'))
        source = self.root/'input'
        prepared = materialize(item,source)
        tar = self.root/'expected.tar'
        make_tar(source,tar)
        codec = Codec('gzip','GZIP',shutil.which('gzip'),version='fixture')
        result = measure_trial(codec,{**item,'path':str(source),'snapshot':prepared['snapshot']},10,self.root,True,2**30)
        self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(result['input_bytes'],tar.stat().st_size)
        self.assertGreater(result['input_bytes'],item['logical_payload_bytes'])

    def test_timeout_and_normal_exit_clean_up_grandchildren(self):
        if not proc_pids_match():
            self.skipTest('Host /proc cannot observe namespace child PIDs')
        for timeout, pause in [(.3,30),(5,0)]:
            pid_file = self.root/f'pid-{pause}'
            program = ('import subprocess,sys,time; '
                       'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                       'open(sys.argv[1],"w").write(str(p.pid)); time.sleep(float(sys.argv[2]))')
            if pause:
                with self.assertRaises(ProcessFailure) as error:
                    run_process([sys.executable,'-c',program,pid_file,str(pause)],self.root,timeout)
                self.assertTrue(error.exception.timed_out)
            else:
                run_process([sys.executable,'-c',program,pid_file,str(pause)],self.root,timeout)
            status = Path('/proc')/pid_file.read_text()/'status'
            try:
                contents = status.read_text()
            except (FileNotFoundError, ProcessLookupError):
                continue  # The kernel may finish reaping it while /proc is being read.
            self.assertIn('State:\tZ',contents)

    def test_large_diagnostics_remain_bounded(self):
        with self.assertRaises(ProcessFailure) as error:
            run_process([sys.executable,'-c','import sys; print("x"*2000000); print("y"*2000000,file=sys.stderr); sys.exit(1)'],self.root,5)
        self.assertLess(len(str(error.exception)),132000)
        _, text = probe_command([sys.executable,'-c','print("x"*2000000); print("probe tail")'])
        self.assertLessEqual(len(text),8192)
        self.assertIn('probe tail',text)

    def test_host_measurement_lock_rejects_a_second_runner(self):
        with patch.dict(os.environ,{'XDG_RUNTIME_DIR':str(self.root)}), host_measurement_lock():
            with self.assertRaises(ValueError):
                with host_measurement_lock():
                    self.fail('Concurrent measurements must be serialized')

    def test_zip64_only_applies_when_sizes_require_it(self):
        payload = self.root/'payload.bin'
        payload.write_bytes(b'x'*1024)
        for threshold, version in [(zipfile.ZIP64_LIMIT,20),(100,45)]:
            output = self.root/f'payload-{threshold}.zip'
            with patch('sys.argv',[str(WORKER),'zip','compress',str(payload),str(output)]), patch('zipfile.ZIP64_LIMIT',threshold):
                runpy.run_path(str(WORKER),run_name='__main__')
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.infolist()[0].extract_version,version)
                self.assertEqual(archive.read('payload'),payload.read_bytes())


if __name__=='__main__':
    unittest.main()
