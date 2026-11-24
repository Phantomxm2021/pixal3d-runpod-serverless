import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.worker_diagnostics import STAGE_FD, _cgroup_memory, _memory_cgroup, _monitor, _sample


class WorkerDiagnosticsTests(unittest.TestCase):
    def test_records_peak_and_oom_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"
            (proc / "self").mkdir(parents=True)
            (proc / "self/cgroup").write_text("0::/worker\n")
            (proc / "123").mkdir()
            (proc / "123/status").write_text("VmRSS:\t800 kB\nVmHWM:\t900 kB\n")
            cgroup = root / "cgroup"
            member = cgroup / "worker"
            member.mkdir(parents=True)
            (member / "memory.current").write_text("1048576\n")
            (member / "memory.peak").write_text("2097152\n")
            (member / "memory.max").write_text("4194304\n")
            (member / "memory.events").write_text("oom 1\noom_kill 0\n")
            group = _memory_cgroup(proc, cgroup)
            self.assertEqual(group, (member, 2))
            baseline = _cgroup_memory(group)
            self.assertEqual(baseline, {"cgroup_version": 2, "current_bytes": 1048576,
                                       "peak_bytes": 2097152, "limit_bytes": 4194304,
                                       "oom": 1, "oom_kill": 0})
            self.assertEqual(_sample(123, group, proc), {**baseline, "rss_kib": 800, "rss_peak_kib": 900})

            # Simulate updated kernel counters and a hard child exit with a pending stage marker.
            (member / "memory.events").write_text("oom 2\noom_kill 1\n")
            read_fd, write_fd = os.pipe()
            with os.fdopen(read_fd, "rb") as pipe:
                try:
                    worker = subprocess.Popen(
                        [sys.executable, "-c", "import os, signal; "
                         f"os.write(int(os.environ[{STAGE_FD!r}]), b'moge_create_model\\n'); "
                         "os.kill(os.getpid(), signal.SIGKILL)"],
                        env={**os.environ, STAGE_FD: str(write_fd)}, pass_fds=(write_fd,),
                    )
                finally:
                    os.close(write_fd)
                with worker, redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(_monitor(worker, pipe.fileno(), proc, group, baseline), 137)
            exit_line = next(line for line in output.getvalue().splitlines() if "worker exit" in line)
            data = json.loads(exit_line.removeprefix("[memory] worker exit "))
            self.assertEqual((data["stage"], data["returncode"], data["oom_kill_delta"]),
                             ("moge_create_model", -signal.SIGKILL, 1))
