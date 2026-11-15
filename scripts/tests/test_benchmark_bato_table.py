import contextlib
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_bato_table as table


class TableBenchmarkTests(unittest.TestCase):
    def test_selection_samples_complete_directories_and_is_seeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'corpus'
            for index in range(5):
                directory = root / f'dir-{index}'
                directory.mkdir(parents=True)
                for number in range(3):
                    (directory / str(number)).write_bytes(b'x' * 100_000)
            with patch.object(table, 'MIN_DIR', 250_000), patch.object(table, 'MAX_DIR', 400_000):
                files, directories = table.select_inputs(root, random.Random(12))
                self.assertEqual((files, directories), table.select_inputs(root, random.Random(12)))
                other = table.select_inputs(root, random.Random(13))
            self.assertNotEqual((files, directories), other)
            self.assertEqual(len(files), 5)
            self.assertEqual(len(directories), 2)
            self.assertTrue(all(path.is_file() for path in files))
            self.assertTrue(all(path.is_dir() and len(list(path.iterdir())) == 3 for path in directories))

    def test_table_adds_one_row_with_variant_columns(self):
        out = io.StringIO()
        rows = {
            'main CPU': {'status': 'PASS & VERIFIED', 'comp_size': 123, 'comp_time_s': 1.0, 'decomp_time_s': 0.2},
            'branch CPU': {'status': 'FAIL'},
            'branch wgpu': {'status': 'UNSUPPORTED'},
            'branch CUDA': {'status': 'UNSUPPORTED'},
        }
        with contextlib.redirect_stdout(out):
            table.print_table_header()
            table.print_table_row({'target': '/data/sample', 'snapshot': {'size': 12345}}, rows)
        text = out.getvalue()
        self.assertIn('Original KB', text)
        self.assertIn('12.35', text)
        self.assertIn('Main CPU', text)
        self.assertIn('0.12 KB 1,00/0,20', text)
        self.assertIn('UNSUPPORTED', text)
        self.assertEqual(text.count('sample'), 1)

    def test_main_displays_each_variant_without_saving_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / 'corpus'
            corpus.mkdir()
            workload = {'target': 'sample', 'snapshot': {'size': 12345}}
            output = io.StringIO()
            measured = {'status': 'PASS & VERIFIED', 'comp_size': 123,
                        'comp_time_s': 1.0, 'decomp_time_s': 0.2}
            def measure(item, name, build, run_dir, args, resources):
                self.assertEqual((args.warmups, args.repetitions), (0, 1))
                self.assertFalse((run_dir / 'comparison.json').exists())
                if name == 'main':
                    self.assertIn('PENDING', output.getvalue())
                elif name == 'branch':
                    self.assertIn('0.12 KB 1,00/0,20', output.getvalue())
                else:
                    self.assertEqual(name, 'cuda')
                    self.assertEqual(output.getvalue().splitlines()[-1].count('PENDING'), 1)
                    return {**measured, 'cuda_phases': 'ED'}
                return measured.copy()
            argv = ['benchmark', '--corpus', str(corpus), '--output', str(root / 'results'),
                    '--scratch', str(root / 'scratch'), '--branch-revision', 'test']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output), \
                 patch.object(table, 'prepare_workloads', return_value=[workload]), \
                 patch.object(table, 'build_revision', return_value={'revision': 'test'}), \
                 patch.object(table, 'build_capabilities', side_effect=lambda build: {**build, 'cuda_filters': True}), \
                 patch.object(table, 'perf_counter', side_effect=[0, 1, 3, 6, 9, 10]), \
                 patch.object(table, 'run_case', side_effect=measure):
                self.assertEqual(table.main(), 0)
            self.assertEqual(list((root / 'scratch').iterdir()), [])
            self.assertFalse((root / 'results').exists())
            lines = [line for line in output.getvalue().splitlines() if line.startswith('sample')]
            self.assertEqual([line.count('PENDING') for line in lines], [3, 2, 1, 0])
            self.assertIn('0.12 KB 1,00*/0,20*', lines[-1])
            self.assertTrue(lines[-1].endswith('8.00 S'))
            self.assertIn('Benchmark elapsed: 10.00 S', output.getvalue())

    def test_reuses_verified_previous_build_and_rebuilds_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'old-run'
            old.mkdir()
            binary = old / 'bato'
            binary.write_bytes(b'verified build')
            binary.chmod(0o755)
            revision = 'a' * 40
            build = {'revision': revision, 'binary': str(binary), 'sha256': table.sha256(binary), 'version': 'v1'}
            (old / 'builds.json').write_text(json.dumps({'builds': {'main': build}}))
            with patch.object(table.subprocess, 'check_output', return_value=revision + '\n'), \
                 patch.object(table.subprocess, 'run', side_effect=AssertionError('must not compile')):
                first = table.build_revision('origin/main', root)
                self.assertEqual(Path(first['binary']).read_bytes(), b'verified build')
                Path(first['binary']).write_bytes(b'corrupt')
                second = table.build_revision('origin/main', root)
                self.assertEqual(Path(second['binary']).read_bytes(), b'verified build')
                third = table.build_revision('origin/main', root)
                self.assertEqual(second, third)


if __name__ == '__main__':
    unittest.main()
