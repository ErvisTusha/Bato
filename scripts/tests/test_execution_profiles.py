"""Execution identities, immutable inputs, uncertainty, and directory verification."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bato_tools.codecs import Codec, create_codecs, check_tools, validate_tuning
from bato_tools.common import Measurement, snapshot, make_tar, extract_tar, parse_execution_stats, sha256
from bato_tools.measurement import measure_trial
from bato_tools.compare import metric_outcome
from bato_tools.resources import ProcessTreeMonitor
from bato_tools.runner import select_workloads
from test_benchmark_v4 import options


class ExecutionProfileTests(unittest.TestCase):
    def test_both_bato_operations_apply_execution_settings(self):
        codec = Codec('bato-v1-cpu8', 'Bato', sys.executable, threads=8, memory_budget=2**30)
        for action in ['compress', 'decompress']:
            with patch('bato_tools.codecs.run_process', return_value=Measurement(.1, None, None, [])) as runner:
                codec.step(action, Path('input'), Path('output'), Path('.'), 10, 1024)
            command = runner.call_args.args[0]
            self.assertEqual(command[command.index('--threads') + 1], '8')
            self.assertNotIn('--compute', command)
            self.assertIn('--stats', command)
            self.assertIn(str(2**30), command)

    def test_execution_and_profile_settings_have_distinct_identities(self):
        with patch('bato_tools.codecs.shutil.which', return_value=None):
            codecs = create_codecs(Path(sys.executable), 'fixture', 'bato',
                                   threads=8, bato_modes='cpu1,cpu')
        self.assertEqual(len({json.dumps(c.identity(), sort_keys=True) for c in codecs}), 2)
        original = codecs[0].identity()
        for field, value in [('profile', 'fast'), ('memory_budget', 2**30), ('chunk_size', 2**16)]:
            old = getattr(codecs[0], field)
            setattr(codecs[0], field, value)
            self.assertNotEqual(original, codecs[0].identity())
            setattr(codecs[0], field, old)

    def test_external_profiles_and_workers_are_applied(self):
        for profile, level in [('fast', 1), ('default', 3), ('high', 22)]:
            opts = Codec('zstd', 'ZSTD', None, profile=profile, threads=8).compression_options()
            self.assertIn(f'-{level}', opts)
            self.assertIn('-T8', opts)
        self.assertNotIn('--extreme', Codec('xz', 'XZ', None, profile='default').compression_options())

    def test_bato_chunk_size_reaches_command_and_identity(self):
        codec = create_codecs(Path(sys.executable), 'fixture', 'bato', chunk_size=1024 * 1024)[0]
        command = codec.compression_options()
        self.assertEqual(command[command.index('--chunk-size') + 1], str(1024 * 1024))
        identity = codec.identity()
        self.assertEqual(identity['settings']['chunk_size'], 1024 * 1024)
        codec.chunk_size = 256 * 1024
        self.assertNotEqual(identity, codec.identity())
        for invalid in [0, 1024 * 1024 + 1, 2 * 1024 * 1024]:
            with self.assertRaises(ValueError):
                create_codecs(Path(sys.executable), 'fixture', 'bato', chunk_size=invalid)

    def test_tuned_zip_uses_matching_backend_for_both_phases(self):
        tuning = {'zip': {'backend':'7z', 'parameters':['-mm=PPMd', '-mmem=256m', '-mo=16']}}
        with patch('bato_tools.codecs.shutil.which', return_value=sys.executable), \
             patch('bato_tools.codecs.probe_command', return_value=(0, '7-Zip fixture')):
            codec = create_codecs(None, None, 'zip', threads=2, codec_tuning=tuning)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for action in ['compress', 'decompress']:
                with patch('bato_tools.codecs.run_process', return_value=Measurement(.1, None, None, [])) as run:
                    codec.step(action, root/'source', root/'output', root, 10, 100, directory=True)
                command = run.call_args.args[0]
                self.assertEqual(command[0], str(Path(sys.executable).resolve()))
                self.assertEqual(command[1], 'a' if action == 'compress' else 'x')
                self.assertIn('-mmt=2', command)
                if action == 'compress':
                    self.assertIn('-tzip', command)
                    self.assertIn('-mm=PPMd', command)
        identity = codec.identity()
        self.assertEqual(identity['settings']['backend'], '7z-zip')
        self.assertEqual(identity['settings']['decompression_workers'], 2)
        codec.compression_parameters[-1] = '-mo=8'
        self.assertNotEqual(identity, codec.identity())

    def test_tuning_rejects_execution_and_framing_overrides(self):
        for tuning in [[], {'bato':{}}, {'zip':{'backend':'other'}},
                       {'rar':{'backend':'native'}}, {'zip':{'parameters':['-mm=PPMd']}},
                       {'rar':{'parameters':'-mcx'}}, {'7z':{'parameters':['-mmt=8']}},
                       {'7z':{'parameters':['-m0=LZMA2:mt=8']}},
                       {'7z':{'parameters':['-tzip']}}, {'xz':{'parameters':['-T8']}},
                       {'rar':{'parameters':['-df']}}]:
            with self.subTest(tuning=tuning), self.assertRaises(ValueError):
                validate_tuning(tuning)
        with self.assertRaises(ValueError):
            create_codecs(None, None, 'rar', codec_tuning={'7z':{}})

    def test_tuned_stream_and_rar_options_enter_commands_and_identity(self):
        tuning = {'xz':{'parameters':['--lzma2=preset=9e,dict=256MiB,nice=273,depth=10000']},
                  'rar':{'parameters':['-md256m', '-mcx']},
                  '7z':{'parameters':['-m0=PPMd:mem=2g:o=32', '-myx=9']}}
        with patch('bato_tools.codecs.shutil.which', return_value=None):
            codecs = create_codecs(None, None, 'xz,rar,7z', codec_tuning=tuning)
        for codec in codecs:
            for option in tuning[codec.identifier]['parameters']:
                self.assertIn(option, codec.compression_options(True))
                self.assertIn(option, codec.identity()['settings']['directory_options'])

    def test_only_verified_archives_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source.txt'
            source.write_bytes(b'archive retention and full byte verification\n' * 30)
            workload = {'path':str(source), 'target':'source.txt', 'snapshot':snapshot(source), 'type':'single_file'}
            codec = Codec('zip', 'ZIP', sys.executable, python_worker=True)
            kept = root/'kept/archive.zip'
            result = measure_trial(codec, workload, 10, retain_archive=kept)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(sha256(kept), result['archive_sha256'])
            self.assertFalse(kept.with_name('archive.zip.pending').exists())
            kept.unlink()
            with patch.object(Codec, 'workflow', return_value=Measurement(.1, None, None, [])):
                result = measure_trial(codec, workload, 10, retain_archive=kept)
            self.assertEqual(result['status'], 'FAIL')
            self.assertFalse(kept.exists())

    def test_directory_hashes_and_tar_include_empty_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, restored = root/'source', root/'restored'
            (source/'empty/nested').mkdir(parents=True)
            (source/'data').write_bytes(b'payload')
            before = snapshot(source)
            make_tar(source, root/'archive.tar')
            extract_tar(root/'archive.tar', restored, 100)
            self.assertEqual(before, snapshot(restored))
            (restored/'empty/nested').rmdir()
            self.assertNotEqual(before, snapshot(restored))

    def test_frozen_manifest_rejects_changed_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = options(root, target=['files/text/4096.bin'])
            workloads, _, _ = select_workloads(args)
            args.frozen_workloads = str(root/'frozen.json')
            Path(args.frozen_workloads).write_text(json.dumps({'schema_version': 4, 'workloads': workloads}))
            self.assertEqual(select_workloads(args)[0], workloads)
            args.seed += 1
            with self.assertRaisesRegex(ValueError, 'frozen manifest'):
                select_workloads(args)

    def test_timing_noise_and_unknown_resources_never_become_wins(self):
        def row(values):
            return {'comp_time_s': sorted(values)[2], 'trials': [{'comp_time_s': v} for v in values],
                    'warmups': 1, 'cache_origin': 'fresh'}
        before = row([.9, 1, 1, 1, 1.1])
        self.assertEqual(metric_outcome(before, row([.5] * 5), 'comp_time_s'), 'win')
        self.assertEqual(metric_outcome(before, row([.8, .8, .8, .8, 1.2]), 'comp_time_s'), 'uncertain')
        self.assertEqual(metric_outcome(before, row([.99] * 5), 'comp_time_s'), 'tie')
        self.assertEqual(metric_outcome(before, row([.5] * 5), 'comp_gpu_vram_peak_bytes'), 'unknown')

    def test_namespace_mismatch_disables_sampling(self):
        with patch('bato_tools.resources.proc_pids_match', return_value=False):
            monitor = ProcessTreeMonitor()
        with patch('bato_tools.resources.Path', side_effect=AssertionError('Do not inspect unrelated PIDs')):
            monitor.sample(1)
        self.assertIsNone(monitor.peak_kib)

    def test_original_execution_stats_are_preserved(self):
        result = parse_execution_stats('execution stats: ExecutionStatsSnapshot { resolved_mode: "cpu", fallback_reason: "no Vulkan compute adapter", gpu_kernel_dispatches: 0, overlap_events: 0 }')
        self.assertEqual(result['gpu_kernel_dispatches'], 0)
        self.assertEqual(result['fallback_reason'], 'no Vulkan compute adapter')

    def test_allow_unavailable_keeps_explicit_preflight_failures(self):
        with patch('bato_tools.codecs.shutil.which', return_value=None), contextlib.redirect_stdout(io.StringIO()):
            codecs, report = check_tools('rar,zip', allow_unavailable=True)
        self.assertEqual([c.identifier for c in codecs], ['zip'])
        self.assertEqual(report['tools'][0]['status'], 'MISSING')


if __name__ == '__main__':
    unittest.main()
