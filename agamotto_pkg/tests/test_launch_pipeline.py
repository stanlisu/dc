# agamotto_pkg/tests/test_launch_pipeline.py
"""Tests for bin/launch_pipeline.sh behavior via subprocess."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

LAUNCHER = str(Path(__file__).resolve().parents[2] / "bin" / "launch_pipeline.sh")
MARVEL_ROOT = str(Path(__file__).resolve().parents[2])

# Skip all tests when launcher script doesn't exist (dc/ standalone runs).
pytestmark = pytest.mark.skipif(
    not Path(LAUNCHER).exists(),
    reason=f"launcher not found at {LAUNCHER}"
)


def _run(cmd, env=None, timeout=10):
    e = {**os.environ, "MARVEL_ROOT": MARVEL_ROOT, **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=e)


def test_launcher_exists():
    assert Path(LAUNCHER).exists(), f"launcher not found at {LAUNCHER}"
    assert os.access(LAUNCHER, os.X_OK), "launcher is not executable"


def test_launcher_writes_pidfile(tmp_path):
    pid_file = tmp_path / "test.pid"
    log_file = tmp_path / "test.log"
    base_args = [
        LAUNCHER,
        "--name", "test_job",
        "--pidfile", str(pid_file),
        "--log", str(log_file),
        "--no-telegram",
        "--", "sleep", "3"
    ]
    # Run in background so we can check PID file while it's running
    p = subprocess.Popen(base_args, env={**os.environ, "MARVEL_ROOT": MARVEL_ROOT})
    # Wait past the launcher's own 1s validation sleep, then check
    time.sleep(1.5)
    assert pid_file.exists(), "PID file was not written"
    p.terminate()
    p.wait()


def test_launcher_prevents_duplicate(tmp_path):
    pid_file = tmp_path / "test.pid"
    log_file = tmp_path / "test.log"
    base_args = [
        LAUNCHER,
        "--name", "test_job",
        "--pidfile", str(pid_file),
        "--log", str(log_file),
        "--no-telegram",
        "--", "sleep", "2"
    ]
    # Start first instance
    p1 = subprocess.Popen(base_args, env={**os.environ, "MARVEL_ROOT": MARVEL_ROOT})
    time.sleep(0.3)
    # Try to start second instance — should fail
    result = _run(base_args)
    p1.terminate()
    p1.wait()
    assert result.returncode != 0, "Duplicate launch should have been rejected"
    assert "already running" in result.stderr.lower() or "already running" in result.stdout.lower()


def test_launcher_cleans_pidfile_on_exit(tmp_path):
    pid_file = tmp_path / "test.pid"
    log_file = tmp_path / "test.log"
    _run([
        LAUNCHER,
        "--name", "test_job",
        "--pidfile", str(pid_file),
        "--log", str(log_file),
        "--no-telegram",
        "--", "true"  # exits immediately
    ], timeout=15)
    time.sleep(0.3)
    assert not pid_file.exists(), "PID file should be cleaned up after process exits"
