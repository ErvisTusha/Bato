import contextlib
import io
import json
import random
import sqlite3
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
            table.MAIN_CPU: {'status': 'PASS & VERIFIED', 'comp_size': 123, 'comp_time_s': 1.0, 'decomp_time_s': 0.2},
            table.BRANCH_CPU: {'status': 'FAIL'},
            table.BRANCH_GPU: {'status': 'UNSUPPORTED'},
        }
        with contextlib.redirect_stdout(out):
            table.print_table_header('wgpu')
            table.print_table_row({'target': '/data/sample', 'snapshot': {'size': 12345}}, rows)
        text = out.getvalue()
        self.assertIn('Original KB', text)
        self.assertIn('12.35', text)
        self.assertIn('Main CPU', text)
        self.assertIn('Branch wgpu (GPU + CPU)', text)
        self.assertIn('0.12 KB 1,00/0,20', text)
        self.assertIn('UNSUPPORTED', text)
        self.assertEqual(text.count('sample'), 1)

    def run_main(self, root, extra=(), measure=None, binaries=('main-hash', 'branch-hash')):
        """Run the table on one fake input; return (status, output, measured variant names)."""
        corpus = root / 'corpus'
        corpus.mkdir(exist_ok=True)
        workload = {'target': 'sample', 'workload_id': 'w1', 'snapshot': {'size': 12345, 'sha256': 'input-hash'}}
        output = io.StringIO()
        measured = []

        def default_measure(item, variant, run_dir, run):
            measured.append(variant['name'])
            row = {'status': 'PASS & VERIFIED', 'comp_size': 123, 'comp_time_s': 1.0, 'decomp_time_s': 0.2}
            return {**row, 'gpu_phases': 'ED'} if variant['name'] == 'cuda' else row

        builds = iter([{'revision': 'main', 'sha256': binaries[0], 'gpu_compute_modes': []},
                       {'revision': 'test', 'sha256': binaries[1], 'gpu_compute_modes': ['auto', 'cpu', 'cuda']}])
        argv = ['benchmark', '--corpus', str(corpus), '--output', str(root / 'results'),
                '--scratch', str(root / 'scratch'), '--database', str(root / 'store' / 'table.sqlite3'),
                '--branch-revision', 'test', '--gpu-compute', 'cuda', *extra]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output), \
             patch.object(table, 'prepare_workloads', return_value=[workload]), \
             patch.object(table, 'build_revision', return_value={}), \
             patch.object(table, 'build_capabilities', side_effect=lambda build: next(builds)), \
             patch.object(table, 'hardware_identity', return_value={'cpu_model': self.cpu_model}), \
             patch.object(table, 'run_case', side_effect=measure or default_measure):
            status = table.main()
        self.assertEqual(list((root / 'scratch').iterdir()), [])
        return status, output.getvalue(), measured

    cpu_model = 'test cpu'

    def test_main_displays_each_variant_and_marks_gpu_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, text, measured = self.run_main(Path(tmp))
            self.assertEqual(status, 0)
            self.assertEqual(measured, ['main', 'branch', 'cuda'])
            lines = [line for line in text.splitlines() if line.startswith('sample')]
            self.assertEqual([line.count('PENDING') for line in lines], [3, 2, 1, 0])
            self.assertIn('0.12 KB 1,00*/0,20*', lines[-1])
            self.assertNotIn('=0.12', lines[-1])
            self.assertNotIn('Reused stored results', text)

    def test_stored_results_are_reused_for_unchanged_binaries_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.run_main(root)
            # Same binaries: nothing is measured again and every cell is marked as stored.
            status, text, measured = self.run_main(root)
            self.assertEqual((status, measured), (0, []))
            line = [line for line in text.splitlines() if line.startswith('sample')][-1]
            self.assertEqual(line.count('=0.12 KB'), 3)
            self.assertIn('sample (main CPU): measured ', text)
            # A changed branch binary is measured; the stored main result is still reused.
            _, _, measured = self.run_main(root, binaries=('main-hash', 'new-branch-hash'))
            self.assertEqual(measured, ['branch', 'cuda'])
            # More repetitions than stored, a geometry change, other hardware or --refresh measure again.
            for extra in (['--repetitions', '2'], ['--bato-segment-size', '1048576'], ['--refresh']):
                _, _, measured = self.run_main(root, extra)
                self.assertEqual(measured, ['main', 'branch', 'cuda'], extra)
            self.cpu_model = 'other cpu'
            self.addCleanup(setattr, self, 'cpu_model', 'test cpu')
            _, _, measured = self.run_main(root)
            self.assertEqual(measured, ['main', 'branch', 'cuda'])

    def test_failed_results_are_stored_but_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            def fail(item, variant, run_dir, run):
                calls.append(variant['name'])
                return {'status': 'FAIL', 'error': 'boom'}

            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.run_main(root, measure=fail)[0], 1)
                self.assertEqual(self.run_main(root, measure=fail)[0], 1)
            self.assertEqual(calls, ['main', 'branch', 'cuda'] * 2)
            with contextlib.closing(sqlite3.connect(root / 'store' / 'table.sqlite3')) as connection:
                self.assertEqual(connection.execute('SELECT count(*), min(status) FROM table_results').fetchone(),
                                 (6, 'FAIL'))

    def test_named_inputs_replace_the_random_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'one.bin').write_bytes(b'abc')
            workloads = table.prepare_workloads(root, None, [root / 'one.bin'])
            self.assertEqual([Path(item['target']).name for item in workloads], ['one.bin'])
            self.assertEqual(workloads[0]['snapshot']['size'], 3)

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
