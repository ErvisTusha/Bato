"""Selection and command-line behavior of corpus-backed minimum CLI tests."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_tests
from bato_tools.test_corpus import MINIMUM_EXTENSIONS, minimum_samples


class MinimumCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entries = {}
        for extension in MINIMUM_EXTENSIONS:
            self.add(f'real/{extension}/b.{extension}', extension.encode() + b'1')
            self.add(f'real/{extension}/a.{extension}', extension.encode() + b'2')
            self.add(f'real/{extension}/large.{extension}', extension.encode() * 20)
        self.save()

    def add(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.entries[name] = {'size':len(data), 'sha256':hashlib.sha256(data).hexdigest()}

    def save(self):
        (self.root / '.corpus.json').write_text(json.dumps({'files':self.entries}))

    def test_smallest_then_path_and_no_duplicate_contents(self):
        self.add('real/txt/0.txt', b'txt2')
        self.save()
        samples = minimum_samples(self.root)
        self.assertEqual(len(samples), 10)
        self.assertEqual(len({s.digest for s in samples}), 10)
        self.assertEqual([s.relative_path for s in samples[:2]], ['real/txt/0.txt','real/txt/b.txt'])
        self.assertTrue(all('large' not in s.relative_path for s in samples))
        self.entries = dict(reversed(list(self.entries.items())))
        self.save()
        self.assertEqual(minimum_samples(self.root), samples)

    def test_missing_type_and_changed_file_fail(self):
        entry = self.entries.pop('real/pdf/b.pdf')
        self.entries.pop('real/pdf/large.pdf')
        self.save()
        with self.assertRaisesRegex(ValueError, '.pdf'):
            minimum_samples(self.root)
        self.entries['real/pdf/b.pdf'] = entry
        self.save()
        (self.root / 'real/pdf/b.pdf').write_bytes(b'xxxx')
        with self.assertRaisesRegex(ValueError, 'differs'):
            minimum_samples(self.root)

    def test_symlink_and_traversal_rejected(self):
        path = self.root / 'real/txt/a.txt'
        data = path.read_bytes()
        path.unlink()
        target = self.root / 'other'
        target.write_bytes(data)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            minimum_samples(self.root)
        path.unlink()
        path.write_bytes(data)
        self.entries['real/../escape.txt'] = {'size':0,'sha256':hashlib.sha256(b'').hexdigest()}
        self.save()
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            minimum_samples(self.root)

    def test_default_and_min_select_distinct_modes_and_keep_rules(self):
        captured = []
        def run(suite):
            captured.append([case.id() for case in suite])
            return unittest.TestResult()
        with patch('run_tests.select_samples', side_effect=lambda root, mode, seed: minimum_samples(root)) as select, \
             patch('run_tests.resolve_bato', return_value=(Path('/unused/bato'),'test')), \
             patch('run_tests.ensure_test_baselines', return_value=0) as gate, \
             patch.object(unittest.TextTestRunner, 'run', side_effect=run), contextlib.redirect_stdout(io.StringIO()):
            for flags in [[],['--min']]:
                self.assertEqual(run_tests.main(['--corpus',str(self.root)] + flags),0)
        self.assertEqual(gate.call_count,2)
        self.assertEqual([call.args[1] for call in select.call_args_list], ['normal', 'fast'])
        self.assertEqual(captured[0],captured[1])
        self.assertEqual(sum(name.startswith('real_file:') for name in captured[0]),10)
        self.assertTrue(any('test_exact_block_and_window_boundaries' in name for name in captured[0]))
        self.assertFalse(any('test_extended_' in name for name in captured[0]))

    def test_targeted_rules_need_no_corpus_and_conflicts_fail(self):
        with patch('run_tests.resolve_bato', return_value=(Path('/unused/bato'),'test')), \
             patch.object(unittest.TextTestRunner,'run',return_value=unittest.TestResult()), \
             patch('run_tests.select_samples',side_effect=AssertionError('unexpected corpus read')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_tests.main(['--test','test_roundtrip']),0)
        with contextlib.redirect_stderr(io.StringIO()):
            for flags in [['--min','--test','test_roundtrip'],['--min','--extended'],['--timeout','nan']]:
                with self.assertRaises(SystemExit) as raised:
                    run_tests.main(flags)
                self.assertEqual(raised.exception.code,2)


if __name__ == '__main__':
    unittest.main()
