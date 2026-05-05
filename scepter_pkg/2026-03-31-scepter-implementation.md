# Scepter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `ScepterResearch(OrbResearch)` — adds BTC/ETH cross-symbol features and crossed (own-state × BTC-state) regimes to the ORB pipeline.

**Architecture:** `ScepterResearch` extends `OrbResearch`. Anchors (BTC, ETH) are loaded alongside altcoins but never verticalized as targets. After `super().verticalize()`, `_attach_anchor_features()` joins cross-symbol columns per (altcoin, timestamp). A new `_apply_filter_mask()` override handles ANCHOR_REGIMES conditions. All downstream pipeline scripts are unchanged.

**Tech Stack:** Python, pandas, numpy, agamotto_pkg, orb, OrbResearch.verticalize()

---

### Task 1: Scaffold `scepter/` package

**Files:**
- Create: `scepter/__init__.py`
- Create: `scepter/research.py`

**Step 1: Write the failing test**

```python
# scepter/tests/test_scepter_research.py
def test_import():
    from scepter.research import ScepterResearch
    assert ScepterResearch is not None
```

**Step 2: Run to verify it fails**

```bash
PYTHONPATH="agamotto_pkg/src:orb:." pytest scepter/tests/test_scepter_research.py::test_import -v
```
Expected: `ModuleNotFoundError: No module named 'scepter'`

**Step 3: Create `scepter/__init__.py`**

```python
"""Scepter: cross-symbol anchor features on top of ORB."""
from .research import ScepterResearch

__all__ = ["ScepterResearch"]
```

**Step 4: Create `scepter/research.py` stub**

```python
"""Scepter research: BTC/ETH anchor features on top of OrbResearch."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from agamotto.utils import _symbol_to_native
from orb.research import OrbResearch

logger = logging.getLogger(__name__)


class ScepterResearch(OrbResearch):
    """OrbResearch + BTC/ETH cross-symbol anchor features."""

    def __init__(self, config: Dict, home_root: str) -> None:
        if "ANCHOR_SYMBOLS" not in config:
            raise KeyError("ANCHOR_SYMBOLS is required in config but not set")
        self.anchor_symbols: list[str] = config["ANCHOR_SYMBOLS"]
        self.anchor_windows: list[int] = config.get("ANCHOR_WINDOWS", [14, 28])
        self.anchor_regimes: dict = config.get("ANCHOR_REGIMES", {})

        # Expand SYMBOLS to include anchors for loading/feature engineering.
        # Anchors are excluded from verticalization (not prediction targets).
        original_symbols: list[str] = list(config["SYMBOLS"])
        self._altcoin_symbols: list[str] = original_symbols
        all_symbols = original_symbols + [
            s for s in self.anchor_symbols if s not in original_symbols
        ]
        expanded = {**config, "SYMBOLS": all_symbols}
        super().__init__(expanded, home_root)
```

**Step 5: Run test to verify it passes**

```bash
PYTHONPATH="agamotto_pkg/src:orb:." pytest scepter/tests/test_scepter_research.py::test_import -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add scepter/__init__.py scepter/research.py scepter/tests/test_scepter_research.py
git commit -m "feat: scaffold scepter/ package — ScepterResearch(OrbResearch) stub

[shield]"
```

---

### Task 2: `__init__` validation — raise KeyError on missing ANCHOR_SYMBOLS

**Files:**
- Modify: `scepter/research.py` (already done in Task 1)
- Modify: `scepter/tests/test_scepter_research.py`

**Step 1: Write the failing test**

```python
import pytest
from scepter.research import ScepterResearch

def test_missing_anchor_symbols_raises():
    cfg = {
        "SYMBOLS": ["BINANCE_PERP_SOL_USDT"],
        "TIMEFRAMES": ["1h"],
        "BASE_TF": "1h",
        "TARGET_TF": "1h",
        "TIME_UNIT": "1h",
        "FEE": 0,
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
    }
    with pytest.raises(KeyError, match="ANCHOR_SYMBOLS"):
        ScepterResearch(cfg, "/tmp/fake")

def test_anchor_symbols_expanded_in_load_config():
    """Anchors are added to SYMBOLS for loading but stored separately."""
    cfg = {
        "SYMBOLS": ["BINANCE_PERP_SOL_USDT"],
        "ANCHOR_SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "TIMEFRAMES": ["1h"],
        "BASE_TF": "1h",
        "TARGET_TF": "1h",
        "TIME_UNIT": "1h",
        "FEE": 0,
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
    }
    sc = ScepterResearch(cfg, "/tmp/fake")
    assert sc._altcoin_symbols == ["BINANCE_PERP_SOL_USDT"]
    assert "BINANCE_PERP_BTC_USDT" in sc.config["SYMBOLS"]
    assert "BINANCE_PERP_SOL_USDT" in sc.config["SYMBOLS"]
```

**Step 2: Run to verify they fail (or pass — __init__ is already correct from Task 1)**

```bash
PYTHONPATH="agamotto_pkg/src:orb:." pytest scepter/tests/test_scepter_research.py -v -k "missing_anchor or expanded"
```
Expected: both PASS (already implemented in Task 1 stub)

**Step 3: Commit**

```bash
git add scepter/tests/test_scepter_research.py
git commit -m "test: scepter __init__ validation — ANCHOR_SYMBOLS required

[shield]"
```

---

### Task 3: `verticalize()` override — anchor symbols excluded from targets

**Files:**
- Modify: `scepter/research.py`
- Modify: `scepter/tests/test_scepter_research.py`

**Step 1: Write the failing test**

Add this helper + test to `test_scepter_research.py`. Read `orb/tests/test_orb_research.py` for the pattern — it injects pre-built AgamottoResearch instances with synthetic raw data to avoid disk I/O.

```python
import numpy as np
import pandas as pd
from agamotto import AgamottoResearch
from scepter.research import ScepterResearch


def _make_ohlcv(n, symbol, start, freq):
    idx = pd.date_range(start, periods=n, freq=freq)
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        f"{symbol}_open": close - 0.1,
        f"{symbol}_high": close + 0.5,
        f"{symbol}_low": close - 0.5,
        f"{symbol}_close": close,
        f"{symbol}_volume": np.random.randint(100, 1000, n).astype(float),
    }, index=idx)


def _make_scepter(altcoin_sym="BINANCE_PERP_SOL_USDT",
                  anchor_sym="BINANCE_PERP_BTC_USDT"):
    """Build ScepterResearch with injected raw data, no disk I/O."""
    altcoin_native = "SOLUSDT"
    anchor_native = "BTCUSDT"
    cfg = {
        "SYMBOLS": [altcoin_sym],
        "ANCHOR_SYMBOLS": [anchor_sym],
        "ANCHOR_WINDOWS": [5, 10],
        "ANCHOR_REGIMES": {
            "btc_trending_up":   {"col": "btc_close_vs_ma", "op": ">", "val": 0},
            "btc_high_vol":      {"col": "btc_atr_ratio",   "op": ">", "val": 1.2},
        },
        "TIMEFRAMES": ["1h"],
        "BASE_TF": "1h",
        "TARGET_TF": "1h",
        "TIME_UNIT": "1h",
        "FEE": 0,
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
        "LADDER": 0,
    }
    sc = ScepterResearch(cfg, "/tmp/fake")
    n = 200
    for tf in cfg["TIMEFRAMES"]:
        inst = AgamottoResearch({**cfg, "SYMBOLS": [altcoin_sym, anchor_sym], "TIME_UNIT": tf}, "/tmp/fake")
        inst.raw = pd.concat([
            _make_ohlcv(n, altcoin_native, "2025-01-01", "1h"),
            _make_ohlcv(n, anchor_native, "2025-01-01", "1h"),
        ], axis=1)
        sc._tf_instances[tf] = inst
    sc.raw = sc._tf_instances[cfg["BASE_TF"]].raw
    return sc


def test_verticalize_excludes_anchors():
    """BTC/ETH rows must not appear in vertical_features."""
    sc = _make_scepter()
    sc.engineer_features()
    sc.verticalize()
    vf = sc.vertical_features
    assert "BINANCE_PERP_BTC_USDT" not in vf["symbol"].values
    assert "BINANCE_PERP_SOL_USDT" in vf["symbol"].values
```

**Step 2: Run to verify it fails**

```bash
PYTHONPATH="agamotto_pkg/src:orb:." pytest scepter/tests/test_scepter_research.py::test_verticalize_excludes_anchors -v
```
Expected: FAIL — BTC rows appear because super().verticalize() uses expanded SYMBOLS

**Step 3: Override `verticalize()` in `research.py`**

Add to `ScepterResearch`:

```python
def verticalize(self) -> None:
    """Verticalize altcoins only (skip anchors), then attach anchor features."""
    # Temporarily restrict SYMBOLS to altcoins so super() only builds altcoin rows.
    original = self.config["SYMBOLS"]
    self.config["SYMBOLS"] = self._altcoin_symbols
    try:
        super().verticalize()
    finally:
        self.config["SYMBOLS"] = original

    if self.vertical_features is not None and not self.vertical_features.empty:
        self._attach_anchor_features()
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH="agamotto_pkg/src:orb:." pytest scepter/tests/test_scepter_research.py::test_verticalize_excludes_anchors -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add scepter/research.py scepter/tests/test_scepter_research.py
git commit -m "feat: scepter verticalize() — exclude anchor symbols from prediction targets

[shield]"
```

---

### Task 4: `_attach_anchor_features()` — cross-symbol columns in vertical_features

**Files:**
- Modify: `scepter/research.py`
- Modify: `scepter/tests/test_scepter_research.py`

This is the core of Scepter. For each anchor symbol, two categories of features are joined onto `vertical_features`:

**Category A — anchor-only (same value for all altcoins at timestamp `t`):**
- `{pfx}_ret_lag1/2/3` — anchor bar return at t-1, t-2, t-3
- `{pfx}_atr_ratio` — anchor ATR / 28-bar MA of anchor ATR
- `{pfx}_close_vs_ma` — anchor close − anchor mvg1 (used by ANCHOR_REGIMES)

**Category B — per-altcoin (value depends on altcoin × anchor pair at `t`):**
- `{pfx}_corr_{w}` — rolling Pearson correlation (altcoin return vs anchor return) over window w
- `{pfx}_spread` — cointegration residual: altcoin_close − β·anchor_close (28-bar rolling OLS β)
- `{pfx}_rel_strength` — altcoin cumulative return − anchor cumulative return over 14 bars

All computations use only data available at `t` (causal).

**Step 1: Write the failing tests**

```python
def test_anchor_features_present_in_vf():
    """btc_ columns must appear in vertical_features after verticalize()."""
    sc = _make_scepter()
    sc.engineer_features()
    sc.verticalize()
    vf = sc.vertical_features
    for col in ["btc_ret_lag1", "btc_atr_ratio", "btc_close_vs_ma",
                "btc_corr_5", "btc_spread", "btc_rel_strength"]:
        assert col in vf.columns, f"Missing column: {col}"

def test_anchor_features_no_lookahead():
    """Anchor features must be NaN for the first window rows (causal check)."""
    sc = _make_scepter()
    sc.engineer_features()
    sc.verticalize()
    vf = sc.vertical_features.sort_values("timestamp").reset_index(drop=True)
    # Rolling window=10: first 9 rows should be NaN for corr_10
    assert vf["btc_corr_10"].iloc[:9].isna().all(), "Lookahead in btc_corr_10"

def test_anchor_atr_ratio_positive():
    """btc_atr_ratio must be positive where not NaN."""
    sc = _make_scepter()
    sc.engineer_features()
    sc.verticalize()
    vf = sc.vertical_features
    valid = vf["btc_atr_ratio"].dropna()
    assert (valid > 0).all()
```

**Step 2: Run to verify they fail**

```bash
PYTHONPATH="agamotto_pkg/src:orb:." pytest scepter/tests/test_scepter_research.py -k "anchor_features" -v
```
Expected: all FAIL — `_attach_anchor_features` not yet implemented

**Step 3: Implement `_attach_anchor_features()` and helper `_rolling_beta()`**

Add to `ScepterResearch` in `research.py`:

```python
@staticmethod
def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta: cov(y, x) / var(x). Causal (shift=0 uses [i-w+1:i+1])."""
    beta = np.full(len(y), np.nan)
    y_arr = y.values
    x_arr = x.values
    for i in range(window - 1, len(y)):
        xi = x_arr[i - window + 1: i + 1]
        yi = y_arr[i - window + 1: i + 1]
        vx = np.var(xi)
        if vx > 1e-12:
            beta[i] = np.cov(yi, xi, ddof=1)[0, 1] / vx
    return pd.Series(beta, index=y.index)

def _attach_anchor_features(self) -> None:
    """Join BTC/ETH cross-symbol columns onto self.vertical_features."""
    vf = self.vertical_features
    base_tf = self.base_tf
    windows = self.anchor_windows
    spread_window = max(windows)  # use largest window for OLS spread

    new_cols: dict[str, pd.Series] = {}

    for anchor_sym in self.anchor_symbols:
        anchor_native = _symbol_to_native(anchor_sym)
        if anchor_native is None:
            logger.warning(f"Cannot map anchor symbol {anchor_sym} to native — skipping")
            continue
        pfx = anchor_native[:3].lower()  # "btc" or "eth"

        # ── Category A: anchor-only features (indexed by timestamp) ────────
        ts_feats: dict[str, pd.Series] = {}

        # Lagged returns
        for lag in [1, 2, 3]:
            col = f"{base_tf}_{anchor_native}_ret_lag{lag}"
            if col in self.features.columns:
                ts_feats[f"{pfx}_ret_lag{lag}"] = self.features[col]

        # ATR ratio
        atr_col = f"{base_tf}_{anchor_native}_atr"
        if atr_col in self.features.columns:
            atr = self.features[atr_col]
            atr_ma = atr.rolling(spread_window, min_periods=1).mean()
            ts_feats[f"{pfx}_atr_ratio"] = (atr / atr_ma.replace(0, np.nan)).astype(float)

        # close_vs_ma (used by btc_trending_up/down anchor regimes)
        close_col = f"{base_tf}_{anchor_native}_close"
        mvg1_col  = f"{base_tf}_{anchor_native}_mvg1"
        if close_col in self.features.columns and mvg1_col in self.features.columns:
            ts_feats[f"{pfx}_close_vs_ma"] = (
                self.features[close_col] - self.features[mvg1_col]
            ).astype(float)

        # Merge anchor-only features by timestamp
        if ts_feats:
            anchor_df = pd.DataFrame(ts_feats, index=self.features.index)
            anchor_df = anchor_df.reset_index().rename(columns={"index": "timestamp"})
            vf = vf.merge(anchor_df, on="timestamp", how="left")

        # ── Category B: per-altcoin features ───────────────────────────────
        anchor_ret_col = f"{base_tf}_{anchor_native}_return"
        anchor_close_col = f"{base_tf}_{anchor_native}_close"
        if anchor_ret_col not in self.features.columns:
            logger.warning(f"Anchor return column {anchor_ret_col} not in features — skipping per-altcoin features")
            continue

        anchor_ret = self.features[anchor_ret_col]
        anchor_close = self.features[anchor_close_col] if anchor_close_col in self.features.columns else None

        for altcoin_sym in self._altcoin_symbols:
            alt_native = _symbol_to_native(altcoin_sym)
            if alt_native is None:
                continue
            mask = vf["symbol"] == altcoin_sym

            alt_ret_col   = f"{base_tf}_{alt_native}_return"
            alt_close_col = f"{base_tf}_{alt_native}_close"

            if alt_ret_col not in self.features.columns:
                continue

            alt_ret   = self.features[alt_ret_col]
            alt_close = self.features[alt_close_col] if alt_close_col in self.features.columns else None

            # Rolling correlation for each window
            for w in windows:
                col_name = f"{pfx}_corr_{w}"
                corr = alt_ret.rolling(w, min_periods=w).corr(anchor_ret)
                corr_df = corr.reset_index().rename(columns={"index": "timestamp", 0: col_name})
                sub = vf.loc[mask, ["timestamp"]].merge(corr_df, on="timestamp", how="left")
                if col_name not in vf.columns:
                    vf[col_name] = np.nan
                vf.loc[mask, col_name] = sub[col_name].values

            # Cointegration spread: alt_close - beta * anchor_close (28-bar rolling OLS)
            if alt_close is not None and anchor_close is not None:
                beta = self._rolling_beta(alt_close, anchor_close, spread_window)
                spread = alt_close - beta * anchor_close
                spread_df = spread.reset_index().rename(columns={"index": "timestamp", 0: f"{pfx}_spread"})
                sub = vf.loc[mask, ["timestamp"]].merge(spread_df, on="timestamp", how="left")
                if f"{pfx}_spread" not in vf.columns:
                    vf[f"{pfx}_spread"] = np.nan
                vf.loc[mask, f"{pfx}_spread"] = sub[f"{pfx}_spread"].values

            # Relative strength: alt_cum_ret_14 - anchor_cum_ret_14
            short_w = min(windows)
            alt_cum   = alt_ret.rolling(short_w, min_periods=short_w).sum()
            anch_cum  = anchor_ret.rolling(short_w, min_periods=short_w).sum()
            rel = alt_cum - anch_cum
            rel_df = rel.reset_index().rename(columns={"index": "timestamp", 0: f"{pfx}_rel_strength"})
            sub = vf.loc[mask, ["timestamp"]].merge(rel_df, on="timestamp", how="left")
            if f"{pfx}_rel_strength" not in vf.columns:
                vf[f"{pfx}_rel_strength"] = np.nan
            vf.loc[mask, f"{pfx}_rel_strength"] = sub[f"{pfx}_rel_strength"].values

    self.vertical_features = vf
```

**Step 4: Run tests to verify they pass**

```bash
PYTHONPATH="agamotto_pkg/src:orb:." pytest scepter/tests/test_scepter_research.py -k "anchor_features or no_lookahead or atr_ratio" -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add scepter/research.py scepter/tests/test_scepter_research.py
git commit -m "feat: scepter _attach_anchor_features — lagged returns, corr, spread, rel_strength, atr_ratio

[shield]"
```

---

### Task 5: `_apply_filter_mask()` override — ANCHOR_REGIMES conditions

**Files:**
- Modify: `scepter/research.py`
- Modify: `scepter/tests/test_scepter_research.py`

ANCHOR_REGIMES conditions are defined in setting.json as `{"col": "btc_close_vs_ma", "op": ">", "val": 0}`. The override intercepts these condition names before they reach super's filter logic.

**Step 1: Write the failing test**

```python
def test_filter_mask_anchor_regime_btc_trending_up():
    """btc_trending_up filter must use btc_close_vs_ma > 0."""
    sc = _make_scepter()
    sc.engineer_features()
    sc.verticalize()
    vf = sc.vertical_features.copy()

    # Inject known values for btc_close_vs_ma
    vf["btc_close_vs_ma"] = [1.0, -1.0, 0.5, -0.5] * (len(vf) // 4) + [1.0] * (len(vf) % 4)

    mask = sc._apply_filter_mask(vf, "btc_trending_up", "long")
    assert mask.sum() > 0
    # All unmasked rows must have btc_close_vs_ma > 0
    assert (vf.loc[mask, "btc_close_vs_ma"] > 0).all()

def test_filter_mask_unknown_anchor_regime_passthrough():
    """Unknown filter names pass through to super (not silently dropped)."""
    sc = _make_scepter()
    sc.engineer_features()
    sc.verticalize()
    vf = sc.vertical_features.copy()
    # "baseline" is handled by super — should not raise
    mask = sc._apply_filter_mask(vf, "baseline", "long")
    assert len(mask) == len(vf)
```

**Step 2: Run to verify they fail**

```bash
PYTHONPATH="agamotto_pkg/src:orb:." pytest scepter/tests/test_scepter_research.py -k "filter_mask_anchor" -v
```
Expected: FAIL — `btc_trending_up` falls through to super and returns all-True or errors

**Step 3: Implement `_apply_filter_mask()` override**

Add to `ScepterResearch` in `research.py`:

```python
def _apply_filter_mask(
    self,
    df: pd.DataFrame,
    filter_name: str | list,
    position: str,
) -> pd.Series:
    """Override: check ANCHOR_REGIMES before falling through to OrbResearch."""
    if isinstance(filter_name, str):
        # Handle _and_ compounds that may include anchor regime components
        if "_and_" in filter_name:
            parts = filter_name.split("_and_")
            mask = None
            for part in parts:
                sub = self._apply_filter_mask(df, part.strip(), position)
                mask = sub if mask is None else (mask & sub)
            return mask if mask is not None else pd.Series(True, index=df.index)

        if "_or_" in filter_name:
            parts = filter_name.split("_or_")
            mask = None
            for part in parts:
                sub = self._apply_filter_mask(df, part.strip(), position)
                mask = sub if mask is None else (mask | sub)
            return mask if mask is not None else pd.Series(True, index=df.index)

        # Strip _long/_short suffix before looking up anchor regime
        base_name = filter_name.replace("_long", "").replace("_short", "")
        cond = self.anchor_regimes.get(base_name)
        if cond is not None:
            col = cond["col"]
            op  = cond["op"]
            val = float(cond["val"])
            if col not in df.columns:
                logger.warning(f"ANCHOR_REGIMES column '{col}' missing — defaulting to True")
                return pd.Series(True, index=df.index)
            ops = {">": df[col] > val, "<": df[col] < val,
                   ">=": df[col] >= val, "<=": df[col] <= val,
                   "==": df[col] == val}
            if op not in ops:
                raise ValueError(f"Unsupported op '{op}' in ANCHOR_REGIMES['{base_name}']")
            return ops[op].fillna(False)

    return super()._apply_filter_mask(df, filter_name, position)
```

**Step 4: Run tests to verify they pass**

```bash
PYTHONPATH="agamotto_pkg/src:orb:." pytest scepter/tests/test_scepter_research.py -k "filter_mask" -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add scepter/research.py scepter/tests/test_scepter_research.py
git commit -m "feat: scepter _apply_filter_mask — ANCHOR_REGIMES conditions (btc_trending_up/down, btc_high/low_vol)

[shield]"
```

---

### Task 6: `gauntlet/generate_scepter_regimes.py` — crossed regime generator

**Files:**
- Create: `gauntlet/generate_scepter_regimes.py`

Scepter regimes are crossed: own-state condition × BTC-state condition. Each regime is a compound string joined with `_and_`. The `position` column comes from `AgamottoResearch.allowed_positions()` applied to the own-state part only (anchor conditions do not constrain direction).

**Step 1: Write the failing test**

```python
# scepter/tests/test_generate_scepter_regimes.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gauntlet"))

from generate_scepter_regimes import generate_regimes


def test_crossed_regimes_produced():
    regimes = generate_regimes()
    assert len(regimes) > 0
    # All regimes must be crossed (contain an anchor condition)
    anchor_conds = {"btc_trending_up", "btc_trending_down", "btc_high_vol", "btc_low_vol"}
    for r in regimes:
        parts = set(r["regime"].split("_and_"))
        assert parts & anchor_conds, f"No anchor component in {r['regime']}"

def test_no_long_short_conflict():
    """macd_bearish must only appear as SHORT."""
    regimes = generate_regimes()
    bearish = [r for r in regimes if "macd_bearish" in r["regime"]]
    assert all(r["position"] == "short" for r in bearish)

def test_regime_has_position():
    for r in generate_regimes():
        assert r["position"] in ("long", "short")
```

**Step 2: Run to verify they fail**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest scepter/tests/test_generate_scepter_regimes.py -v
```
Expected: `ModuleNotFoundError: No module named 'generate_scepter_regimes'`

**Step 3: Create `gauntlet/generate_scepter_regimes.py`**

```python
#!/usr/bin/env python3
"""Generate crossed regime stack for Scepter experiments.

Scepter regimes = own-state condition × BTC-state condition.
Regime name: "{own_state}_and_{btc_state}"
Position: determined by own-state alone (anchor conditions are directionally neutral).
"""

from __future__ import annotations

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
_agamotto_src = os.path.join(_repo_root, "agamotto_pkg", "src")
for p in [_agamotto_src, _repo_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from agamotto import AgamottoResearch

# Own-state filters (altcoin's own regime — same as Agamotto base filters)
_OWN_STATE_FILTERS = [
    "above_all_mas",
    "high_volume",
    "adx_trend",
    "vol_breakout",
    "strong_trend",
    "ma_momentum",
    "rsi_oversold",
    "rsi_overbought",
    "macd_bullish",
    "macd_bearish",
    "stoch_bullish",
    "adx_trend",
    "bb_rebound",
    "mom_positive",
]

# BTC-state conditions (defined here, must match ANCHOR_REGIMES in setting.json)
_BTC_STATE_CONDITIONS = [
    "btc_trending_up",
    "btc_trending_down",
    "btc_high_vol",
    "btc_low_vol",
]


def generate_regimes() -> list[dict]:
    """Return list of {regime, position} crossed-regime dicts."""
    regimes = []
    seen = set()

    for own in _OWN_STATE_FILTERS:
        positions = AgamottoResearch.allowed_positions(own)
        for btc in _BTC_STATE_CONDITIONS:
            regime_name = f"{own}_and_{btc}"
            for pos in positions:
                key = (regime_name, pos)
                if key in seen:
                    continue
                seen.add(key)
                regimes.append({"regime": regime_name, "position": pos})

    return regimes


if __name__ == "__main__":
    import json
    regs = generate_regimes()
    print(json.dumps(regs[:5], indent=2))
    print(f"Total: {len(regs)} regimes")
```

**Step 4: Run tests to verify they pass**

```bash
PYTHONPATH="agamotto_pkg/src:." pytest scepter/tests/test_generate_scepter_regimes.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add gauntlet/generate_scepter_regimes.py scepter/tests/test_generate_scepter_regimes.py
git commit -m "feat: generate_scepter_regimes — crossed own-state × BTC-state regime generator

[shield]"
```

---

### Task 7: `gauntlet/run_scepter_research.py` entry point

**Files:**
- Create: `gauntlet/run_scepter_research.py`

Mirrors `gauntlet/run_orb_research.py`. Differences: imports `ScepterResearch` from `scepter`, calls `generate_scepter_regimes` instead of `generate_orb_regimes`, writes the same CSV/JSON regime stack format.

**Step 1: Write the failing test**

```python
# scepter/tests/test_run_scepter_research.py
import subprocess, sys, os

def test_help_flag():
    """Entry point must be runnable and accept --help."""
    result = subprocess.run(
        [sys.executable, "gauntlet/run_scepter_research.py", "--help"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "agamotto_pkg/src:orb:scepter:."},
        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
    )
    assert result.returncode == 0
    assert "-c" in result.stdout
```

**Step 2: Run to verify it fails**

```bash
PYTHONPATH="agamotto_pkg/src:orb:scepter:." pytest scepter/tests/test_run_scepter_research.py -v
```
Expected: FAIL

**Step 3: Create `gauntlet/run_scepter_research.py`**

```python
#!/usr/bin/env python3
"""Scepter research pipeline wrapper."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Ensure scepter/ and orb/ are importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
for p in [os.path.join(_repo_root, "orb"), os.path.join(_repo_root, "scepter"), _repo_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scepter import ScepterResearch
from generate_scepter_regimes import generate_regimes as _generate_scepter_regimes


def _write_scepter_regime_stack(out_dir: Path, models: list[str]) -> Path:
    """Write regime_stack.csv for crossed Scepter regimes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    regimes = _generate_scepter_regimes()  # list of {regime, position}

    json_path = out_dir / "regime_stack.json"
    with json_path.open("w") as f:
        json.dump(regimes, f, indent=2)

    csv_path = out_dir / "regime_stack.csv"
    fields = ["regime", "position", "model"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in regimes:
            for model in models:
                writer.writerow({"regime": r["regime"], "position": r["position"], "model": model})

    n = len(regimes)
    print(f"Wrote {n} Scepter regimes to {json_path} and {n * len(models)} rows to {csv_path}")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scepter cross-symbol research pipeline")
    parser.add_argument("-c", required=True, help="Path to setting.json")
    parser.add_argument("--output-dir", help="Optional output directory override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setting_path = Path(args.c).resolve()
    with setting_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    if args.output_dir:
        config["OUTPUT_DIR"] = str(Path(args.output_dir).resolve())
    elif "OUTPUT_DIR" not in config:
        config["OUTPUT_DIR"] = str(setting_path.parent)

    out_dir = Path(config["OUTPUT_DIR"])
    models = config.get("SWEEP_MODELS", ["LightGBM", "XGBoost", "Ridge", "HistGBR"])

    json_path = _write_scepter_regime_stack(out_dir, models)
    config["REGIME_STACK_PATH"] = str(json_path)

    home_root = _repo_root + "/"
    research = ScepterResearch(config, home_root)
    research.load()
    research.engineer_features()
    out_dir_str = research.create()
    print(f"Scepter research output directory: {out_dir_str}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH="agamotto_pkg/src:orb:scepter:." pytest scepter/tests/test_run_scepter_research.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add gauntlet/run_scepter_research.py scepter/tests/test_run_scepter_research.py
git commit -m "feat: gauntlet/run_scepter_research.py — entry point for Scepter pipeline

[shield]"
```

---

### Task 8: Experiment directories and setting.json files

**Files:**
- Create: `gauntlet/pred_scepter15m_1/setting.json`
- Create: `gauntlet/pred_scepter1h_1/setting.json`
- Create: `gauntlet/pred_scepter4h_1/setting.json`
- Create: `gauntlet/pred_scepter1d_1/setting.json`

Copy the relevant ORB setting.json as base and make these changes:
1. Set `STRATEGY`, `VERSION`, TF fields
2. Add `ANCHOR_SYMBOLS`, `ANCHOR_WINDOWS`, `ANCHOR_REGIMES`
3. Remove BTC/ETH from `SYMBOLS` (they're prediction targets in ORB; in Scepter they're anchors)
4. Set `WINDOW_SIZE` to 6 (per design doc)
5. Set `OUTPUT_DIR` to `/mnt/tardis-data-archive/marvel-research/gauntlet/pred_scepter{tf}_1`

**Step 1: No test needed — just create the files**

**Step 2: Create `gauntlet/pred_scepter1h_1/setting.json`** (use as template, adjust others accordingly)

```json
{
    "VERSION": "scepter1h_1",
    "STRATEGY": "scepter",
    "EXCHANGE": "BINANCE",
    "DATA": "liquid",
    "CAPITAL": 100,
    "LEVERAGE": 3,
    "TIME_UNIT": "1h",
    "TARGET_TF": "1h",
    "TIMEFRAMES": ["15m", "1h", "4h", "1d"],
    "BASE_TF": "1h",
    "LADDER": 5,
    "ANCHOR_SYMBOLS": [
        "BINANCE_PERP_BTC_USDT",
        "BINANCE_PERP_ETH_USDT"
    ],
    "ANCHOR_WINDOWS": [14, 28],
    "ANCHOR_REGIMES": {
        "btc_trending_up":   {"col": "btc_close_vs_ma", "op": ">", "val": 0},
        "btc_trending_down": {"col": "btc_close_vs_ma", "op": "<", "val": 0},
        "btc_high_vol":      {"col": "btc_atr_ratio",   "op": ">", "val": 1.2},
        "btc_low_vol":       {"col": "btc_atr_ratio",   "op": "<", "val": 0.8}
    },
    "SYMBOLS": [
        "BINANCE_PERP_XRP_USDT",
        "BINANCE_PERP_BNB_USDT",
        "BINANCE_PERP_SOL_USDT",
        "BINANCE_PERP_DOGE_USDT",
        "BINANCE_PERP_TRX_USDT",
        "BINANCE_PERP_ADA_USDT",
        "BINANCE_PERP_LINK_USDT",
        "BINANCE_PERP_AVAX_USDT",
        "BINANCE_PERP_HYPE_USDT",
        "BINANCE_PERP_SUI_USDT",
        "BINANCE_PERP_XLM_USDT",
        "BINANCE_PERP_BCH_USDT",
        "BINANCE_PERP_HBAR_USDT",
        "BINANCE_PERP_LTC_USDT",
        "BINANCE_PERP_1000SHIB_USDT",
        "BINANCE_PERP_TON_USDT",
        "BINANCE_PERP_DOT_USDT",
        "BINANCE_PERP_XMR_USDT",
        "BINANCE_PERP_WLFI_USDT",
        "BINANCE_PERP_UNI_USDT",
        "BINANCE_PERP_AAVE_USDT",
        "BINANCE_PERP_1000PEPE_USDT",
        "BINANCE_PERP_ENA_USDT",
        "BINANCE_PERP_NEAR_USDT",
        "BINANCE_PERP_APT_USDT",
        "BINANCE_PERP_TAO_USDT",
        "BINANCE_PERP_ETC_USDT"
    ],
    "FEE": 2.25,
    "SWEEP_MODELS": ["LightGBM", "XGBoost", "Ridge", "HistGBR"],
    "WINDOW_SIZE": 6,
    "MA_PERIODS": [7, 25, 99],
    "STATS_WINDOW": 14,
    "OUTPUT_DIR": "/mnt/tardis-data-archive/marvel-research/gauntlet/pred_scepter1h_1"
}
```

For 15m: `"TIME_UNIT": "15m"`, `"TARGET_TF": "15m"`, `"BASE_TF": "15m"`, VERSION/OUTPUT_DIR accordingly.
For 4h: same pattern with `"4h"`.
For 1d: same with `"1d"`.

**Step 3: Verify setting files parse correctly**

```bash
for tf in 15m 1h 4h 1d; do
    python -c "import json; cfg=json.load(open('gauntlet/pred_scepter${tf}_1/setting.json')); print('${tf}:', cfg['VERSION'], len(cfg['SYMBOLS']), 'altcoins', len(cfg['ANCHOR_SYMBOLS']), 'anchors')"
done
```
Expected: each prints e.g. `1h: scepter1h_1 27 altcoins 2 anchors`

**Step 4: Commit**

```bash
git add gauntlet/pred_scepter*/setting.json
git commit -m "feat: scepter experiment dirs — pred_scepter{15m,1h,4h,1d}_1 setting.json

[shield]"
```

---

### Task 9: Run full test suite + flake8

**Step 1: Run all Scepter tests**

```bash
PYTHONPATH="agamotto_pkg/src:orb:scepter:." pytest scepter/tests/ -v
```
Expected: all PASS

**Step 2: Run existing test suite to confirm no regressions**

```bash
PYTHONPATH="agamotto_pkg/src:orb:scepter:." pytest agamotto_pkg/tests/ orb/tests/ -v --tb=short
```
Expected: all PASS

**Step 3: Lint**

```bash
flake8 scepter/ gauntlet/run_scepter_research.py gauntlet/generate_scepter_regimes.py \
    --count --select=E9,F63,F7,F82 --show-source --statistics
```
Expected: 0 errors

**Step 4: Final commit if any lint fixes needed**

```bash
git add -p  # stage only lint fixes
git commit -m "fix: scepter lint — E9/F63/F7/F82 clean

[shield]"
```

---

## Running the Scepter Pipeline

Once implementation is complete, run the full pipeline for one TF to validate end-to-end:

```bash
# Resource check first
free -h && df -h /home/ubuntu/ && pgrep -f "rolling_predict\|run_agamotto\|run_orb" && echo "PIPELINE RUNNING — ABORT"

PY="/home/ubuntu/miniconda3/envs/py313/bin/python"
export PYTHONPATH="agamotto_pkg/src:orb:scepter:."

DIR="gauntlet/pred_scepter1h_1"

# Step 1: Feature engineering (generates vertical_features.csv + filter/)
$PY gauntlet/run_scepter_research.py -c ${DIR}/setting.json 2>&1 | tee ${DIR}/research.log

# Step 2: Rolling predict
$PY gauntlet/rolling_predict_returns.py -c ${DIR} --workers 180 2>&1 | tee ${DIR}/rolling_predict.log

# Step 3: Optimize thresholds (1h grid)
$PY gauntlet/optimize_thresholds.py -c ${DIR} --step 0.001 --max-thresh 0.01 2>&1 | tee ${DIR}/optimize.log

# Step 4: Filter and rank
$PY gauntlet/filter_regime_stacks.py --project gauntlet --top-n 120 2>&1 | tee gauntlet/filter.log
```

**Execution options:**

**1. Subagent-Driven (this session)** — dispatch fresh subagent per task, review between tasks

**2. Parallel Session (separate)** — open new session in this worktree, use superpowers:executing-plans

Which approach?
