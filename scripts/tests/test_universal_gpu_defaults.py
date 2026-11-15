"""Regression coverage for universal GPU defaults and benchmark selectors."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_suite
import benchmark_bato_table


class UniversalGpuDefaults(unittest.TestCase):
    def test_canonical_benchmark_defaults_to_all_available_gpu_backends(self):
        observed = {}

        def capture(args, *, result_rows=None):
            observed["compute"] = args.bato_compute
            return 0

        with patch.object(benchmark_suite, "run_suite", side_effect=capture):
            self.assertEqual(benchmark_suite.main(["--only-bato"]), 0)
        self.assertEqual(observed["compute"], "auto")

    def test_legacy_benchmark_baseline_can_omit_compute_flag(self):
        observed = {}

        def capture(args, *, result_rows=None):
            observed["compute"] = args.bato_compute
            return 0

        with patch.object(benchmark_suite, "run_suite", side_effect=capture):
            self.assertEqual(
                benchmark_suite.main(["--only-bato", "--bato-compute", "legacy"]), 0
            )
        self.assertIsNone(observed["compute"])

    def test_backend_detection_handles_all_vendor_names(self):
        help_text = (
            "Options: --compute auto|cpu|cuda|hip|level-zero|metal|wgpu --stats"
        )
        result = SimpleNamespace(stdout=help_text, stderr="")
        with patch.object(benchmark_bato_table.subprocess, "run", return_value=result):
            found = benchmark_bato_table.build_capabilities({"binary": "fake-bato"})
        self.assertTrue(found["cuda_filters"])
        self.assertTrue(found["wgpu_filters"])
        self.assertEqual(
            set(found["gpu_compute_modes"]),
            {"auto", "cpu", "cuda", "hip", "level-zero", "metal", "wgpu"},
        )


if __name__ == "__main__":
    unittest.main()
