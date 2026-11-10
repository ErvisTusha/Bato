"""CM configuration identities and strict reuse of disjoint reference campaigns."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_suite
from compare_benchmarks import reference_campaigns
from bato_tools.codecs import Codec, create_codecs, resolve_bato
from bato_tools.compare import compare_runs
from test_benchmark_v4 import observation


class CmGroundworkTests(unittest.TestCase):
    def fixture(self, codec='bato-v1'):
        metadata = dict(schema_version=4, run_id='candidate', status='complete',
                        environment={'host': 'same', 'harness_sha256': 'frozen'},
                        effective_memory_budget=2**30, resource_policy={'cap': 'same'},
                        cpu_limit=1, requested_threads=1, compression_profile='high',
                        mode='rigorous', suite='corpus', track='both', seed=42, timeout=120,
                        codecs=[{'id': 'gzip', 'binary_sha256': 'gzip-fixed'}])
        row = dict(codec=codec, target='real/input', workload_id='fixed-input', track='direct',
                   status='PASS & VERIFIED', verified_lossless=True, cache_origin='fresh',
                   warmups=1, trials=[observation() for _ in range(5)],
                   timing_quality='repeated', comp_size=50, comp_size_min=50, comp_size_max=50,
                   comp_time_s=.01, decomp_time_s=.02, input_sha256='input',
                   codec_input_sha256='input', metadata_policy='normalized-v1',
                   raw_size=100, compression_profile='extreme', origin='real',
                   category='compression', scenario='corpus', type='single_file', size_band='tiny',
                   effective_settings=Codec(codec, codec, None).settings())
        return metadata, row

    def test_cm_settings_reach_commands_both_directions_and_identities(self):
        codec = create_codecs(None, 'fixture', 'bato', segment_size=2**20, chunk_size=2**16)[0]
        settings = codec.settings()
        self.assertEqual((settings['segment_size'], settings['chunk_size']), (2**20, 2**16))
        self.assertNotIn('engine', settings)
        self.assertNotIn('cm_profile', settings)
        for action in ('compress', 'decompress'):
            with patch('bato_tools.codecs.run_process') as run:
                codec.step(action, 'input', 'output', '.', 10, 100)
            command = run.call_args.args[0]
            for flag, value in [('--segment-size', str(2**20)), ('--chunk-size', str(2**16))]:
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertNotIn('--engine', command)
            self.assertNotIn('--cm-profile', command)

    def test_cm_geometry_rejects_unrepresentable_settings(self):
        for changes in [dict(segment_size=2**16-1), dict(segment_size=2**30+1),
                        dict(segment_size=2**20+1), dict(chunk_size=4095),
                        dict(chunk_size=65537), dict(segment_size=2**16, chunk_size=2**17)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                create_codecs(None, 'fixture', 'bato', **changes)

    def test_cli_forwards_cm_settings(self):
        with patch('benchmark_suite.run_suite', return_value=0) as run:
            self.assertEqual(benchmark_suite.main(['--bato-segment-size', str(2**20)]), 0)
        self.assertEqual(run.call_args.args[0].bato_segment_size, 2**20)

    def test_binary_without_expected_options_is_rejected(self):
        help_text = 'Pure Rust Bato --chunk-size'
        with patch('bato_tools.codecs.probe_command', return_value=(0, help_text)):
            with self.assertRaisesRegex(ValueError, '--segment-size'):
                resolve_bato(sys.executable)
        with patch('bato_tools.codecs.probe_command', side_effect=[
                (0, help_text + ' --segment-size'), (0, 'bato 0.3.0 (format 1.0)')]):
            self.assertEqual(resolve_bato(sys.executable)[0], Path(sys.executable).resolve())

    def test_model_changes_require_exact_explicit_allowance(self):
        metadata, before = self.fixture()
        before['effective_settings']['segment_size'] = 2**26
        after = copy.deepcopy(before)
        after['effective_settings'] = Codec('bato-v1', 'Bato', None, segment_size=2**20).settings()
        original = copy.deepcopy(before)
        args = (metadata, [before], dict(metadata, run_id='after'), [after])
        self.assertEqual(compare_runs(*args)['matched_cases'], 0)
        result = compare_runs(*args, ['segment_size'])
        self.assertEqual(result['matched_cases'], 1)
        self.assertEqual(result['comparisons'][0]['bato_setting_changes']['segment_size'],
                         {'baseline': 2**26, 'candidate': 2**20})
        self.assertEqual(before, original)
        for field, value in [('threads', 8), ('memory_budget', 2**31), ('chunk_size', 2**16)]:
            changed = copy.deepcopy(after)
            changed['effective_settings'][field] = value
            result = compare_runs(metadata, [before], metadata, [changed], ['segment_size'])
            self.assertEqual(result['matched_cases'], 0)
        with self.assertRaises(ValueError):
            compare_runs(*args, ['threads'])

    def test_setting_allowance_does_not_apply_to_peers_or_changed_inputs(self):
        metadata, peer = self.fixture('gzip')
        changed = copy.deepcopy(peer)
        changed['effective_settings']['segment_size'] = 2**20
        self.assertEqual(compare_runs(metadata, [peer], metadata, [changed], ['segment_size'])['matched_cases'], 0)
        metadata, row = self.fixture()
        changed = dict(row, codec_input_sha256='other')
        self.assertEqual(compare_runs(metadata, [row], metadata, [changed], ['segment_size'])['matched_cases'], 0)

    def test_reference_campaigns_keep_original_provenance_and_missing_cases(self):
        metadata, row = self.fixture()
        gzip = dict(row, codec='gzip')
        xz = dict(row, codec='xz')
        fast = dict(metadata, run_id='fast')
        slow = dict(metadata, run_id='slow', codecs=[{'id': 'xz', 'binary_sha256': 'xz-fixed'}])
        result = reference_campaigns(metadata, [row], [(fast, [gzip]), (slow, [xz])], 'bato-v1')
        self.assertEqual(result['competitor_run_ids'], ['fast', 'slow'])
        self.assertEqual([r['competitor_run_id'] for r in result['comparisons']], ['fast', 'slow'])
        self.assertEqual(result['missing_reference_cases'], [])
        self.assertEqual(len(reference_campaigns(metadata, [row], [(fast, []), (slow, [xz])],
                                               'bato-v1')['missing_reference_cases']), 1)
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            reference_campaigns(metadata, [row], [(fast, [gzip]), (dict(fast, run_id='again'), [gzip])], 'bato-v1')
        with self.assertRaisesRegex(ValueError, 'Duplicate reference'):
            reference_campaigns(metadata, [row], [(fast, [gzip]), (fast, [])], 'bato-v1')

    def test_reference_campaigns_reject_changed_identity_host_or_harness(self):
        metadata, row = self.fixture()
        fast = dict(metadata, run_id='fast')
        for changes in [dict(environment={'host': 'other', 'harness_sha256': 'frozen'}),
                        dict(environment={'host': 'same', 'harness_sha256': 'changed'}),
                        dict(codecs=[{'id': 'gzip', 'binary_sha256': 'changed'}])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                reference_campaigns(metadata, [row], [(fast, []),
                    (dict(fast, run_id='other', **changes), [])], 'bato-v1')

    def test_failed_peers_are_unknown_and_trial_policies_must_match(self):
        metadata, row = self.fixture()
        reference = dict(metadata, run_id='reference')
        peer = dict(row, codec='gzip', status='FAIL')
        result = reference_campaigns(metadata, [row], [(reference, [peer])], 'bato-v1')
        self.assertEqual(set(result['comparisons'][0]['outcomes'].values()), {'unknown'})
        peer = dict(row, codec='gzip', trials=[observation() for _ in range(6)])
        with self.assertRaisesRegex(ValueError, 'trial policy'):
            reference_campaigns(metadata, [row], [(reference, [peer])], 'bato-v1')


if __name__ == '__main__':
    unittest.main()
