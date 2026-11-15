"""Cross-platform process cleanup regression tests."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bato_tools import common


class KillGroupTests(unittest.TestCase):
    def test_windows_uses_taskkill_for_child_process_tree(self):
        process = Mock(pid=13579)
        process.poll.return_value = 0
        with patch.object(common.os, 'name', 'nt'), \
                patch.object(common.subprocess, 'run') as command:
            common.kill_group(process)
        self.assertEqual(
            command.call_args.args[0],
            ['taskkill', '/PID', '13579', '/T', '/F'],
        )
        process.kill.assert_not_called()

    def test_windows_falls_back_to_process_kill(self):
        process = Mock(pid=24680)
        process.poll.return_value = None
        with patch.object(common.os, 'name', 'nt'), \
                patch.object(common.subprocess, 'run', side_effect=OSError):
            common.kill_group(process)
        process.kill.assert_called_once()

    def test_unix_kills_isolated_process_group(self):
        process = Mock(pid=1234, benchmark_group=6789)
        with patch.object(common.os, 'name', 'posix'), \
                patch.object(common.os, 'killpg', create=True) as kill:
            common.kill_group(process)
        kill.assert_called_once_with(6789, common.signal.SIGKILL)


if __name__ == '__main__':
    unittest.main()
