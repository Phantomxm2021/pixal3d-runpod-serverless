"""Observe worker exits and Linux memory counters outside the inference process."""

from __future__ import annotations

import json
import os
import resource
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path


STAGE_FD = "PIXAL3D_MEMORY_STAGE_FD"


def memory_stage(name: str) -> None:
    fd = os.environ.get(STAGE_FD)
    if fd is not None:
        try:
            os.write(int(fd), (name + "\n").encode())
        except (OSError, ValueError):
            pass  # Diagnostics must not interrupt inference.


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _memory_cgroup(proc: Path, cgroup: Path) -> tuple[Path, int] | None:
    for entry in _read(proc / "self/cgroup").splitlines():
        _, controllers, member = entry.split(":", 2)
        member = member.lstrip("/")
        if not controllers:
            root, version, marker = cgroup, 2, "memory.current"
        elif "memory" in controllers.split(","):
            root, version, marker = cgroup / "memory", 1, "memory.usage_in_bytes"
        else:
            continue
        for candidate in (root / member, root):
            if (candidate / marker).is_file():
                return candidate, version
    return None


def _cgroup_memory(group: tuple[Path, int] | None) -> dict:
    if group is None:
        return {}
    folder, version = group
    usage, peak, limit, events = (
        ("memory.current", "memory.peak", "memory.max", "memory.events") if version == 2 else
        ("memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.limit_in_bytes", "memory.oom_control")
    )
    values = {"cgroup_version": version}
    for name, filename in (("current_bytes", usage), ("peak_bytes", peak), ("limit_bytes", limit)):
        value = _read(folder / filename).strip()
        if value.isdigit():
            number = int(value)
            values[name] = "max" if name == "limit_bytes" and number >= 2**60 else number
        elif value == "max":
            values[name] = value
    for line in _read(folder / events).splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in {"oom", "oom_kill", "oom_group_kill", "under_oom"} and fields[1].isdigit():
            values[fields[0]] = int(fields[1])
    return values


def _sample(pid: int, group: tuple[Path, int] | None, proc: Path) -> dict:
    values = _cgroup_memory(group)
    for line in _read(proc / str(pid) / "status").splitlines():
        key, _, value = line.partition(":")
        fields = value.split()
        if key in {"VmRSS", "VmHWM"} and fields and fields[0].isdigit():
            values["rss_kib" if key == "VmRSS" else "rss_peak_kib"] = int(fields[0])
    return values


def _log(event: str, stage: str, **details) -> None:
    print(f"[memory] {event} {json.dumps({'stage': stage, **details}, sort_keys=True)}", flush=True)


def _monitor(worker: subprocess.Popen, pipe: int, proc: Path, group: tuple[Path, int] | None, baseline: dict) -> int:
    stage, pending = "", b""
    peak_rss, sampled_cgroup_peak = 0, 0
    next_log = 0.0
    with selectors.DefaultSelector() as selector:
        selector.register(pipe, selectors.EVENT_READ)
        while True:
            exited = worker.poll() is not None
            if selector.select(0 if exited else (0.25 if stage else 1)):
                chunk = os.read(pipe, 4096)
                if not chunk:
                    selector.unregister(pipe)
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    previous, stage = stage, line.decode()
                    if previous:
                        _log("stage end", previous, **_sample(worker.pid, group, proc))
                    next_log = 0.0
            if exited:
                break
            if stage:
                sample = _sample(worker.pid, group, proc)
                peak_rss = max(peak_rss, sample.get("rss_peak_kib", 0))
                sampled_cgroup_peak = max(sampled_cgroup_peak, sample.get("current_bytes", 0))
                if time.monotonic() >= next_log:
                    _log("sample", stage, **sample)
                    next_log = time.monotonic() + 3

    status = worker.wait()
    if status or stage:
        final = _cgroup_memory(group)
        if sys.platform.startswith("linux"):
            final["worker_rss_peak_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        deltas = {
            f"{key}_delta": final[key] - baseline[key]
            for key in ("oom", "oom_kill", "oom_group_kill") if key in final and key in baseline
        }
        _log("worker exit", stage, returncode=status, last_rss_peak_kib=peak_rss,
             sampled_cgroup_peak_bytes=sampled_cgroup_peak, **final, **deltas)
    return 128 - status if status < 0 else status


def main() -> int:
    proc = Path("/proc")
    group = _memory_cgroup(proc, Path("/sys/fs/cgroup"))
    baseline = _cgroup_memory(group)
    _log("baseline", "", **baseline)
    read_fd, write_fd = os.pipe()
    try:
        worker = subprocess.Popen(
            [sys.executable, "-u", "-X", "faulthandler", "rp_handler.py", *sys.argv[1:]],
            env={**os.environ, STAGE_FD: str(write_fd)}, pass_fds=(write_fd,),
        )
    finally:
        os.close(write_fd)

    def forward(signum, _frame):
        if worker.poll() is None:
            worker.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        return _monitor(worker, read_fd, proc, group, baseline)
    finally:
        os.close(read_fd)


if __name__ == "__main__":
    raise SystemExit(main())
