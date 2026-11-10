import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ablate", ROOT / "scripts" / "ablate.py")
ablate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablate)


class AblationPatchesTest(unittest.TestCase):
    def test_every_anchor_occurs_exactly_once_in_the_sources(self):
        for relative, anchor, _ in ablate.PATCHES:
            text = (ROOT / relative).read_text()
            self.assertEqual(text.count(anchor), 1, f"{relative}: {anchor!r}")

    def test_switches_are_unique(self):
        switches = [switch for switch, _, _ in ablate.ABLATIONS.values()]
        self.assertEqual(len(switches), len(set(switches)))

    def test_combinations_parse_to_bit_masks(self):
        self.assertEqual(ablate.parse_combo("apm=0,2;isse=1"), (0b101, 0b10))
        with self.assertRaises(SystemExit):
            ablate.parse_combo("mixer=1")


if __name__ == "__main__":
    unittest.main()
