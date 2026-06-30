"""CI gate: no dc algo may produce a feature column that ships real-named.

Runs the static coverage audit; fails if any feature base is neither mapped nor
explicit passthrough (which would silently leak a real name into parquet/meta).
"""
import subprocess
import sys
from pathlib import Path

DC_ROOT = Path(__file__).resolve().parent.parent.parent


def test_all_feature_bases_mapped_or_passthrough():
    src = ":".join(str(p) for p in DC_ROOT.glob("*_pkg/src"))
    r = subprocess.run(
        [sys.executable, "obfuscation/audit_feature_coverage.py"],
        cwd=DC_ROOT, env={"PYTHONPATH": src, "PATH": ""},
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"unmapped feature bases would leak:\n{r.stdout}"
