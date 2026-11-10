"""Selection and drift checks for the copied corpus."""
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bato_tools.test_selection import EXTENSIONS, select_samples


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for ext in EXTENSIONS:
            directory = self.root / 'real' / ext[1:]
            directory.mkdir(parents=True)
            for number in range(10):
                (directory / f'{number}{ext}').write_bytes(bytes([number]) * (number + 1))

    def test_modes_and_seed(self):
        for mode, count in [('fast', 2), ('normal', 5), ('full', 10)]:
            samples = select_samples(self.root, mode, 42)
            self.assertEqual(Counter(Path(s.relative_path).suffix for s in samples),
                             Counter({extension:count for extension in EXTENSIONS}))
            self.assertEqual(samples, select_samples(self.root, mode, 42))
        fast = select_samples(self.root, 'fast', 42)
        for extension in EXTENSIONS:
            sizes = sorted(s.size for s in fast if s.relative_path.endswith(extension))
            self.assertLessEqual(sizes[0], 5)
            self.assertGreater(sizes[1], 5)
        self.assertNotEqual(select_samples(self.root, 'normal', 1), select_samples(self.root, 'normal', 2))

    def test_change_and_missing_file(self):
        sample = select_samples(self.root, 'fast', 42)[0]
        sample.verified_path().write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            sample.verified_path()
        (self.root / sample.relative_path).unlink()
        with self.assertRaisesRegex(ValueError, 'Expected 10'):
            select_samples(self.root, 'full', 42)

    def test_symlink_rejected(self):
        path = self.root / 'real' / 'txt' / '0.txt'
        path.unlink()
        path.symlink_to(self.root / 'real' / 'txt' / '1.txt')
        with self.assertRaisesRegex(ValueError, 'regular file'):
            select_samples(self.root, 'fast', 42)
