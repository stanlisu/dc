"""Ensure the obfuscation map + vendored codec copies exist before any test.

The per-package `<pkg>/_obf/` copies are gitignored (derived from obfuscation/).
Regenerate them once per test session so a fresh clone can import
`from <pkg>._obf.codec import default` without a manual build step.
"""
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def pytest_configure(config):
    for script in ("obfuscation/build_map.py", "obfuscation/sync_vendor.py"):
        subprocess.run([sys.executable, str(_ROOT / script)], cwd=_ROOT, check=True)
