"""WRITE_VERTICAL_FEATURES_CSV gates only the artifact, never the computation.

`AgamottoResearch.create()` wrote `vertical_features.csv` unconditionally. Nothing
in the gauntlet pipeline reads it, and it is expensive: measured on shield
2026-08-07 for `pred_orb.base.15m_1`, 4,996,355 rows x 324 cols = 18.9 GB at
~11 MB/s to the NAS-backed mirror, i.e. ~26 min of a 9h29m Step 1.

The contract these tests lock in:

1. **Absent means True.** Every one of the ~775 existing setting.json files
   predates the key, and marvel's per-window panel workflows consume the file
   (`regime_discover.py` defaults its panel to
   `~/marvel-explore/<algo>_perwindow_<tf>_<market>/vertical_features.csv`), so a
   False default would break them silently.
2. **False skips the CSV and changes nothing else.** The filter parquets from a
   flag-False run must be identical to those from a flag-absent run — that is what
   makes this an artifact switch rather than a behaviour switch, and therefore not
   the kind of defaulted config CLAUDE.md bans.
"""
from __future__ import annotations

import csv
import sys

sys.path.insert(0, "agamotto_pkg/src")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest

from agamotto.research import AgamottoResearch

SYMBOL = "BTCUSDT"
REGIME = "rsi_oversold"


def _make_raw(n: int = 400) -> pd.DataFrame:
    """OHLCV long enough for the 99-bar MA and the 14-bar rolling stats."""
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(11)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            f"{SYMBOL}_open": close - 0.05,
            f"{SYMBOL}_high": close + 0.5,
            f"{SYMBOL}_low": close - 0.5,
            f"{SYMBOL}_close": close,
            f"{SYMBOL}_volume": rng.integers(100, 1000, n).astype(float),
        },
        index=idx,
    )


def _stack_csv(tmp_path):
    p = tmp_path / "regime_stack.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["regime", "position", "model"])
        w.writeheader()
        w.writerow({"regime": REGIME, "position": "long", "model": "Ridge"})
    return p


def _run_create(tmp_path, out_name, flag):
    """Run create() into its own out dir; `flag` None means the key is absent."""
    out_dir = tmp_path / out_name
    out_dir.mkdir()
    cfg = {
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "EXCHANGE": "BINANCE",
        "DATA": "liquid",
        "TIME_UNIT": "1h",
        "LADDER": 1,
        "LADDER_BPS": 1.0,
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
        "FEE": 2.25,
        "VERSION": "test.vertical_csv_flag",
        "OUTPUT_DIR": str(out_dir),
        "REGIME_STACK_PATH": str(_stack_csv(tmp_path)),
    }
    if flag is not None:
        cfg["WRITE_VERTICAL_FEATURES_CSV"] = flag
    research = AgamottoResearch(cfg, str(tmp_path))
    research.raw = _make_raw()
    research.create()
    return out_dir


def _filter_frames(out_dir):
    """Every filter artifact under `out_dir`, keyed by filename."""
    files = sorted(out_dir.rglob("filter_*.parquet")) + sorted(out_dir.rglob("filter_*.csv"))
    assert files, f"create() wrote no filter artifacts under {out_dir}"
    out = {}
    for f in files:
        out[f.name] = (pd.read_parquet(f) if f.suffix == ".parquet"
                       else pd.read_csv(f))
    return out


def test_absent_key_writes_the_csv(tmp_path):
    """Back-compat: the ~775 settings that predate the key must be unaffected."""
    out_dir = _run_create(tmp_path, "absent", flag=None)
    assert (out_dir / "vertical_features.csv").exists()


def test_true_writes_the_csv(tmp_path):
    out_dir = _run_create(tmp_path, "true", flag=True)
    assert (out_dir / "vertical_features.csv").exists()


def test_false_skips_the_csv(tmp_path):
    out_dir = _run_create(tmp_path, "false", flag=False)
    assert not (out_dir / "vertical_features.csv").exists()


def test_false_leaves_the_filter_output_identical(tmp_path):
    """The flag skips an ARTIFACT — it must not move a single computed value."""
    on = _filter_frames(_run_create(tmp_path, "on", flag=None))
    off = _filter_frames(_run_create(tmp_path, "off", flag=False))
    assert on.keys() == off.keys()
    for name, frame_on in on.items():
        pd.testing.assert_frame_equal(frame_on, off[name])


@pytest.mark.parametrize("flag", [None, True, False])
def test_create_returns_the_out_dir_either_way(tmp_path, flag):
    out_dir = _run_create(tmp_path, f"ret_{flag}", flag=flag)
    assert out_dir.exists()
