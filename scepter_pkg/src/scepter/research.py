"""Scepter research: BTC/ETH anchor features on top of OrbResearch."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from agamotto.utils import _symbol_to_native
from orb.research import OrbResearch

logger = logging.getLogger(__name__)


def _obf():
    """Lazy accessor for the vendored obfuscation codec (see _obf/codec.py)."""
    from ._obf.codec import default
    return default()


# Regime definitions moved out of the public marvel generator (real names live
# only here). Scepter regime = own-state × BTC-state, "{own}_and_{btc}".
_SCEPTER_OWN_STATE = [
    "above_all_mas", "high_volume", "low_volume", "adx_trend", "vol_breakout",
    "low_vol", "high_vol", "strong_trend", "ma_momentum",
    "rsi_oversold", "rsi_overbought", "macd_bullish", "macd_bearish",
    "stoch_bullish", "bb_rebound", "mom_positive",
]
_SCEPTER_BTC_STATE = [
    "btc_trending_up", "btc_trending_down", "btc_high_vol", "btc_low_vol",
]


class ScepterResearch(OrbResearch):
    """OrbResearch + BTC/ETH cross-symbol anchor features."""

    @classmethod
    def generate_regime_stack(cls) -> list[dict]:
        """Coded [{regime, position}] for own-state × BTC-state crossed regimes.

        Position is determined by own-state alone (anchors are directionally
        neutral). Regime names returned OBFUSCATED (structure preserved).
        """
        c = _obf()
        seen, out = set(), []
        for own in _SCEPTER_OWN_STATE:
            positions = cls.allowed_positions(own)
            for btc in _SCEPTER_BTC_STATE:
                name = f"{own}_and_{btc}"
                for pos in positions:
                    key = (name, pos)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"regime": c.encode_regime(name), "position": pos})
        return out

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

    def _apply_filter_mask(
        self,
        df: pd.DataFrame,
        filter_name,
        position: str,
    ) -> pd.Series:
        """Override: check ANCHOR_REGIMES before falling through to OrbResearch."""
        if isinstance(filter_name, str):
            # Accept coded regimes: decode to real before matching anchor names
            # against the real-name ANCHOR_REGIMES dict (code->real, real->real).
            filter_name = _obf().decode_regime_tolerant(filter_name)
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
                op = cond["op"]
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
        spread_window = max(windows)

        for anchor_sym in self.anchor_symbols:
            anchor_native = _symbol_to_native(anchor_sym)
            if anchor_native is None:
                logger.warning(f"Cannot map anchor symbol {anchor_sym} to native — skipping")
                continue
            pfx = anchor_native[:3].lower()  # "btc" or "eth"

            # ── Category A: anchor-only features (indexed by timestamp) ────────
            ts_feats: dict[str, pd.Series] = {}

            # Lagged returns (pre-computed by engineer_features)
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

            # close_vs_ma (used by ANCHOR_REGIMES btc_trending_up/down)
            close_col = f"{base_tf}_{anchor_native}_close"
            mvg1_col = f"{base_tf}_{anchor_native}_mvg1"
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

            # `{tf}_{sym}_return` is the FORWARD return (price_return = hist_return.shift(-1)),
            # i.e. the prediction target. Using it in rolling corr / rel_strength leaks the
            # future (the 2026-06-13 scepter leak: corr(rel_strength, y_true)=0.24, Sharpe→11).
            # .shift(1) recovers the historical return (== hist_return) so these anchor
            # features are causal — only data through bar T. (Spread uses close, already causal.)
            anchor_ret = self.features[anchor_ret_col].shift(1)
            anchor_close = self.features[anchor_close_col] if anchor_close_col in self.features.columns else None

            for altcoin_sym in self._altcoin_symbols:
                alt_native = _symbol_to_native(altcoin_sym)
                if alt_native is None:
                    continue
                mask = vf["symbol"] == altcoin_sym

                alt_ret_col = f"{base_tf}_{alt_native}_return"
                alt_close_col = f"{base_tf}_{alt_native}_close"

                if alt_ret_col not in self.features.columns:
                    continue

                alt_ret = self.features[alt_ret_col].shift(1)   # forward->historical (causal), see above
                alt_close = self.features[alt_close_col] if alt_close_col in self.features.columns else None

                # Rolling correlation for each window
                for w in windows:
                    col_name = f"{pfx}_corr_{w}"
                    corr = alt_ret.rolling(w, min_periods=w).corr(anchor_ret).rename(col_name)
                    corr_df = corr.reset_index().rename(columns={"index": "timestamp"})
                    sub = vf.loc[mask, ["timestamp"]].merge(corr_df, on="timestamp", how="left")
                    if col_name not in vf.columns:
                        vf[col_name] = np.nan
                    vf.loc[mask, col_name] = sub[col_name].values

                # Cointegration spread: alt_close - beta * anchor_close (rolling OLS)
                if alt_close is not None and anchor_close is not None:
                    beta = self._rolling_beta(alt_close, anchor_close, spread_window)
                    spread = (alt_close - beta * anchor_close).rename(f"{pfx}_spread")
                    spread_df = spread.reset_index().rename(columns={"index": "timestamp"})
                    sub = vf.loc[mask, ["timestamp"]].merge(spread_df, on="timestamp", how="left")
                    if f"{pfx}_spread" not in vf.columns:
                        vf[f"{pfx}_spread"] = np.nan
                    vf.loc[mask, f"{pfx}_spread"] = sub[f"{pfx}_spread"].values

                # Relative strength: alt_cum_ret - anchor_cum_ret over min(windows)
                short_w = min(windows)
                alt_cum = alt_ret.rolling(short_w, min_periods=short_w).sum()
                anch_cum = anchor_ret.rolling(short_w, min_periods=short_w).sum()
                rel = (alt_cum - anch_cum).rename(f"{pfx}_rel_strength")
                rel_df = rel.reset_index().rename(columns={"index": "timestamp"})
                sub = vf.loc[mask, ["timestamp"]].merge(rel_df, on="timestamp", how="left")
                if f"{pfx}_rel_strength" not in vf.columns:
                    vf[f"{pfx}_rel_strength"] = np.nan
                vf.loc[mask, f"{pfx}_rel_strength"] = sub[f"{pfx}_rel_strength"].values

        self.vertical_features = vf
