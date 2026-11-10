"""Automatic resource defaults, explicit controls and stable resumed budgets."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bato_tools.resources import available_cpu_count, configure_resources, resolve_memory_budget


class AutomaticResourcesTests(unittest.TestCase):
    def options(self, **overrides):
        values = dict(cpu_limit=None, threads=None, memory_budget=None,
                      bato_memory_budget=None, resume=None)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_defaults_use_cpu_capacity_and_ninety_percent_once(self):
        args = self.options()
        with patch('bato_tools.resources.available_cpu_count', return_value=8), \
             patch('bato_tools.resources.memory_state', return_value={
                 'available_ram_bytes': 1001, 'effective_memory_bytes': 2000}):
            configure_resources(args)
            self.assertEqual((args.cpu_limit, args.threads), (8, 8))
            self.assertEqual(args.memory_budget, 2000)
            self.assertEqual(args.bato_memory_budget, 900)
            self.assertEqual(resolve_memory_budget(), 2000)

    def test_explicit_limits_are_preserved(self):
        args = self.options(cpu_limit=2, threads=1, memory_budget=500, bato_memory_budget=123)
        with patch('bato_tools.resources.memory_state', return_value={
                'available_ram_bytes': 1001, 'effective_memory_bytes': 2000}):
            configure_resources(args)
        self.assertEqual((args.cpu_limit, args.threads, args.memory_budget, args.bato_memory_budget),
                         (2, 1, 500, 123))

    def test_managed_default_respects_an_explicit_process_cap(self):
        args = self.options(cpu_limit=2, memory_budget=500)
        with patch('bato_tools.resources.memory_state', return_value={
                'available_ram_bytes': 1001, 'effective_memory_bytes': 2000}):
            configure_resources(args)
        self.assertEqual((args.threads, args.bato_memory_budget), (2, 450))

    def test_resume_reuses_frozen_values_despite_free_memory_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'runs'/'saved';path.mkdir(parents=True)
            (path/'run.json').write_text(json.dumps(dict(cpu_limit=8, requested_threads=8,
                effective_memory_budget=2000, resource_policy={'bato_managed_memory_budget': 900})))
            args = self.options(resume='saved', output_dir=tmp)
            with patch('bato_tools.resources.available_cpu_count', return_value=8), \
                 patch('bato_tools.resources.memory_state', return_value={
                     'available_ram_bytes': 400, 'effective_memory_bytes': 2000}):
                configure_resources(args)
            self.assertEqual((args.cpu_limit, args.threads, args.memory_budget, args.bato_memory_budget),
                             (8, 8, 2000, 900))

    def test_resume_rejects_path_traversal_before_reading_projection(self):
        with self.assertRaises(ValueError):
            configure_resources(self.options(resume='../saved', output_dir='/tmp'))

    def test_cpu_capacity_obeys_affinity_and_ancestor_quotas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root/'job').mkdir()
            (root/'cpu.max').write_text('800000 100000')
            (root/'job/cpu.max').write_text('150000 100000')
            member = root/'membership';member.write_text('0::/job\n')
            def mapped(value):
                return root if value == '/sys/fs/cgroup' else member if value == '/proc/self/cgroup' else Path(value)
            with patch('bato_tools.resources.Path', side_effect=mapped), \
                 patch('bato_tools.resources.os.sched_getaffinity', return_value=set(range(16))):
                self.assertEqual(available_cpu_count(), 2)
            with patch('bato_tools.resources.Path', side_effect=mapped), \
                 patch('bato_tools.resources.os.sched_getaffinity', return_value={4}):
                self.assertEqual(available_cpu_count(), 1)


if __name__ == '__main__':
    unittest.main()
