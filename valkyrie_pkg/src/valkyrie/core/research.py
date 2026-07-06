"""ValkyrieResearch: weekly options classification pipeline."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .features import DailyFeatureBuilder
from .loader import ValkyrieLoader
from .utils import get_dates_in_range


def _obf():
    """Lazy accessor for the vendored obfuscation codec (see _obf/codec.py)."""
    from .._obf.codec import default
    return default()

logger = logging.getLogger(__name__)


def _process_one_date(
    tardis_root: str,
    assets: List[str],
    date_str: str,
    settlement_hour: int,
    max_weekly_days: int,
) -> Dict[str, Dict]:
    """Worker: load one day and build feature rows for each asset.

    Returns {asset: row_dict} for assets with valid data.
    Module-level so it is picklable for ProcessPoolExecutor.
    """
    loader = ValkyrieLoader(tardis_root, assets)
    builder = DailyFeatureBuilder(
        settlement_hour_utc=settlement_hour,
        max_weekly_days=max_weekly_days,
    )
    df_all = loader.load_day(date_str)
    if df_all.empty:
        return {}
    result: Dict[str, Dict] = {}
    for asset in assets:
        df_asset = df_all[df_all["asset"] == asset].copy()
        if df_asset.empty:
            continue
        row = builder.build_daily_row(df_asset, asset, date_str)
        if row is not None:
            result[asset] = row
    return result

# Columns excluded from ML feature selection
_META_COLS = {
    "date", "asset", "symbol", "timestamp", "year", "month",
    "y", "ret", "ret_short_call", "ret_short_put", "ret_short_straddle",
    "entry_price_used", "target_price", "position", "regime",
    "spot_open", "spot_close", "spot_settlement",
    "iso_year", "iso_week",
    # Option price OHLC — used for P&L computation, not ML features
    "atm_call_price_open", "atm_call_price_high", "atm_call_price_close",
    "atm_put_price_open", "atm_put_price_high", "atm_put_price_close",
    "atm_straddle_price_open", "atm_straddle_price_high", "atm_straddle_price_close",
    "atm_strike_used",
}

# Default regime stack — SHORT options only
# positive_skew: puts overpriced → sell put; negative_skew: calls overpriced → sell call
_DEFAULT_REGIME_STACK: List[Dict] = [
    # baseline (unconditional) removed 2026-06-18 — no-brainer, banned (CLAUDE.md)
    {"regime": "high_iv_rank", "position": "short_straddle"},
    {"regime": "low_iv_rank", "position": "short_straddle"},
    {"regime": "backwardation", "position": "short_straddle"},
    {"regime": "positive_skew", "position": "short_put"},
    {"regime": "negative_skew", "position": "short_call"},
    {"regime": "oi_expansion", "position": "short_straddle"},
]

# Map position name → ret column
_POSITION_TO_RET_COL: Dict[str, str] = {
    "short_call": "ret_short_call",
    "short_put": "ret_short_put",
    "short_straddle": "ret_short_straddle",
}


class ValkyrieResearch:
    """Research pipeline: Deribit options data → weekly samples → filter parquets.

    Mirrors the MjolnirResearch API pattern so downstream scripts are consistent.

    Args:
        config:    Experiment config dict (from setting.json).
        home_root: Repository root directory.
    """

    @classmethod
    def generate_regime_stack(cls) -> List[Dict]:
        """Coded [{regime, position}] for the canonical Valkyrie regime set.

        Options-specific positions (short_straddle/short_put/short_call) are not
        long/short suffixes and pass through unchanged; only the regime name is
        obfuscated. Real names live only here (private package).
        """
        c = _obf()
        return [{"regime": c.encode_regime(r["regime"]), "position": r["position"]}
                for r in _DEFAULT_REGIME_STACK]

    def __init__(self, config: Dict, home_root: str) -> None:
        self.config = config
        self.home_root = home_root.rstrip("/")
        self._daily_frames: Dict[str, pd.DataFrame] = {}  # asset -> daily features
        self.vertical_features: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load cached daily feature parquets (if available).

        Expects files at: {out_dir}/daily_features_{ASSET}.parquet
        If missing, call engineer_features() to build them first.
        """
        out_dir = self._resolve_out_dir()
        assets = self.config.get("ASSETS", [])
        for asset in assets:
            cache_path = os.path.join(out_dir, f"daily_features_{asset}.parquet")
            if os.path.exists(cache_path):
                try:
                    df = pd.read_parquet(cache_path)
                    self._daily_frames[asset] = df
                    logger.info("Loaded cached daily features for %s (%d days)", asset, len(df))
                except Exception as exc:
                    logger.warning("Failed to load cache for %s: %s", asset, exc)
            else:
                logger.info("No cache for %s at %s — run engineer_features() first", asset, cache_path)

    def engineer_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        workers: int = 1,
    ) -> None:
        """Build daily feature rows from raw Deribit options data.

        For each date in [start_date, end_date]:
          1. Load options_chain file via ValkyrieLoader
          2. Build raw daily feature row per asset via DailyFeatureBuilder
          3. After all dates, add rolling features (IV rank, spot returns, HV)
          4. Cache to {out_dir}/daily_features_{ASSET}.parquet

        Args:
            start_date: YYYYMMDD (defaults to config START_DATE).
            end_date:   YYYYMMDD (defaults to config END_DATE).
            workers:    Number of parallel worker processes (default 1).
        """
        start_date = start_date or self.config.get("START_DATE", "20230101")
        end_date = end_date or self.config.get("END_DATE", "20251020")
        assets = self.config.get("ASSETS", [])
        tardis_root = self.config.get("TARDIS_ROOT", "/mnt/tardis/deribit")
        settlement_hour = int(self.config.get("SETTLEMENT_HOUR_UTC", 8))
        max_weekly_days = int(self.config.get("MAX_WEEKLY_DAYS", 7))

        dates = get_dates_in_range(start_date, end_date)
        raw_rows: Dict[str, List[Dict]] = {a: [] for a in assets}
        done = 0

        logger.info(
            "Processing %d dates from %s to %s (workers=%d)",
            len(dates), start_date, end_date, workers,
        )

        if workers > 1:
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for date_str in dates:
                    fut = pool.submit(
                        _process_one_date,
                        tardis_root, assets, date_str, settlement_hour, max_weekly_days,
                    )
                    futures[fut] = date_str
                for fut in as_completed(futures):
                    done += 1
                    if done % 50 == 0:
                        logger.info("  %d/%d dates done", done, len(dates))
                    try:
                        result = fut.result()
                    except Exception as exc:
                        logger.warning("Date %s failed: %s", futures[fut], exc)
                        continue
                    for asset, row in result.items():
                        raw_rows[asset].append(row)
        else:
            loader = ValkyrieLoader(tardis_root, assets)
            builder = DailyFeatureBuilder(
                settlement_hour_utc=settlement_hour,
                max_weekly_days=max_weekly_days,
            )
            for i, date_str in enumerate(dates):
                if i % 50 == 0:
                    logger.info("  date %d/%d: %s", i + 1, len(dates), date_str)
                df_all = loader.load_day(date_str)
                if df_all.empty:
                    continue
                for asset in assets:
                    df_asset = df_all[df_all["asset"] == asset].copy()
                    if df_asset.empty:
                        continue
                    row = builder.build_daily_row(df_asset, asset, date_str)
                    if row is not None:
                        raw_rows[asset].append(row)

        # Add rolling features and cache
        builder = DailyFeatureBuilder(
            settlement_hour_utc=settlement_hour,
            max_weekly_days=max_weekly_days,
        )
        out_dir = self._resolve_out_dir()
        os.makedirs(out_dir, exist_ok=True)

        for asset in assets:
            if not raw_rows[asset]:
                logger.warning("No daily rows for %s", asset)
                continue
            df = pd.DataFrame(raw_rows[asset]).sort_values("date").reset_index(drop=True)
            df = builder.add_rolling_features(df)
            self._daily_frames[asset] = df
            cache_path = os.path.join(out_dir, f"daily_features_{asset}.parquet")
            df.to_parquet(cache_path, index=False)
            logger.info("Saved %d daily rows for %s → %s", len(df), asset, cache_path)

    def verticalize(self) -> None:
        """Construct weekly samples from daily feature frames.

        For each ISO week × asset × entry day (Mon–Thu):
          - y: 0 (DOWN ≤ -threshold%), 1 (FLAT), 2 (UP ≥ +threshold%)
          - ret_short_call:     option premium P&L for short ATM call
          - ret_short_put:      option premium P&L for short ATM put
          - ret_short_straddle: option premium P&L for short ATM straddle
          - ret: same as ret_short_straddle (default, for backward compat)

        Option P&L formula (e.g. short call):
          ret = (entry_premium - settlement_value) / entry_premium
          Stop-loss: if any hold-day's high > entry_premium * (1 + STOPLOSS_PCT),
          ret = -STOPLOSS_PCT (position closed at stop).

        Settlement value (intrinsic at 08:00 UTC Friday):
          Inverse (BTC/ETH): max(0, S - K) / S for call, max(0, K - S) / S for put
          Linear (SOL_USDC): max(0, S - K) for call, max(0, K - S) for put

        Populates self.vertical_features.
        """
        if not self._daily_frames:
            raise RuntimeError("Call engineer_features() or load() before verticalize().")

        all_daily = pd.concat(
            [df.copy() for df in self._daily_frames.values()], ignore_index=True
        )

        # Parse date → pandas Timestamp for ISO week extraction
        all_daily["_dt"] = pd.to_datetime(all_daily["date"], format="%Y%m%d")
        all_daily["iso_year"] = all_daily["_dt"].dt.isocalendar().year.astype(int)
        all_daily["iso_week"] = all_daily["_dt"].dt.isocalendar().week.astype(int)

        threshold = float(self.config.get("TARGET_THRESHOLD", 0.10))
        stoploss_pct = float(self.config.get("STOPLOSS_PCT", 0.10))
        assets = self.config.get("ASSETS", [])

        rows: List[Dict] = []

        for (iso_year, iso_week), week_group in all_daily.groupby(["iso_year", "iso_week"]):
            for asset in assets:
                asset_week = (
                    week_group[week_group["asset"] == asset]
                    .sort_values("day_of_week")
                    .copy()
                )
                if len(asset_week) < 2:
                    continue

                # Friday row (day_of_week == 4) for settlement
                friday_rows = asset_week[asset_week["day_of_week"] == 4]
                if friday_rows.empty:
                    continue

                fri_row = friday_rows.iloc[0]
                fri_spot = float(fri_row.get("spot_settlement") or np.nan)
                if not np.isfinite(fri_spot) or fri_spot <= 0:
                    fri_spot = float(fri_row.get("spot_close") or np.nan)
                if not np.isfinite(fri_spot) or fri_spot <= 0:
                    continue

                is_linear = "USDC" in str(asset)

                # Entry days: Mon–Thu (day_of_week 0–3)
                entry_days = asset_week[asset_week["day_of_week"].isin([0, 1, 2, 3])]
                for _, entry_row in entry_days.iterrows():
                    entry_dow = int(entry_row["day_of_week"])
                    entry_price = float(entry_row.get("spot_open") or 0)
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        continue

                    change = fri_spot / entry_price - 1

                    # 3-class target
                    if change <= -threshold:
                        y = 0  # DOWN
                    elif change >= threshold:
                        y = 2  # UP
                    else:
                        y = 1  # FLAT

                    # ATM strike at entry (fixed for this option position)
                    atm_k = float(entry_row.get("atm_strike_used") or np.nan)

                    # Settlement intrinsic values
                    settle_call = _settlement_value(atm_k, fri_spot, "call", is_linear)
                    settle_put = _settlement_value(atm_k, fri_spot, "put", is_linear)
                    settle_straddle = (
                        settle_call + settle_put
                        if np.isfinite(settle_call) and np.isfinite(settle_put)
                        else np.nan
                    )

                    # Hold days = days after entry up to and including Friday
                    hold_days = asset_week[asset_week["day_of_week"] > entry_dow]

                    ret_short_call = _compute_option_ret(
                        entry_price=float(entry_row.get("atm_call_price_open") or np.nan),
                        hold_df=hold_days,
                        price_high_col="atm_call_price_high",
                        stoploss_pct=stoploss_pct,
                        settlement=settle_call,
                    )
                    ret_short_put = _compute_option_ret(
                        entry_price=float(entry_row.get("atm_put_price_open") or np.nan),
                        hold_df=hold_days,
                        price_high_col="atm_put_price_high",
                        stoploss_pct=stoploss_pct,
                        settlement=settle_put,
                    )
                    ret_short_straddle = _compute_option_ret(
                        entry_price=float(entry_row.get("atm_straddle_price_open") or np.nan),
                        hold_df=hold_days,
                        price_high_col="atm_straddle_price_high",
                        stoploss_pct=stoploss_pct,
                        settlement=settle_straddle,
                    )

                    row_dict = entry_row.to_dict()
                    row_dict["y"] = int(y)
                    row_dict["ret_short_call"] = float(ret_short_call)
                    row_dict["ret_short_put"] = float(ret_short_put)
                    row_dict["ret_short_straddle"] = float(ret_short_straddle)
                    row_dict["ret"] = float(ret_short_straddle)  # default ret
                    row_dict["target_price"] = float(fri_spot)
                    row_dict["entry_price_used"] = float(entry_price)
                    row_dict["symbol"] = asset
                    row_dict["timestamp"] = row_dict.get("_dt", pd.NaT)
                    rows.append(row_dict)

        if rows:
            df_vert = pd.DataFrame(rows)
            df_vert = df_vert.drop(columns=["_dt"], errors="ignore")
            df_vert = df_vert.sort_values(["year", "month", "asset", "date"]).reset_index(drop=True)
            self.vertical_features = df_vert
            logger.info(
                "Verticalized: %d weekly samples | class distribution: %s",
                len(df_vert),
                df_vert["y"].value_counts().to_dict(),
            )
        else:
            self.vertical_features = pd.DataFrame()

    def filter_signals(
        self,
        regime: Dict,
        save: bool = False,
        out_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply a regime filter to vertical features.

        Args:
            regime: Dict with 'regime' and 'position' keys.
            save:   Save result to {out_dir}/filter/ as parquet.
            out_dir: Output directory.

        Returns:
            Filtered DataFrame with 'y', 'ret', 'position', 'regime' columns.
        """
        if self.vertical_features is None or self.vertical_features.empty:
            raise RuntimeError("Call verticalize() before filter_signals().")

        df = self.vertical_features.copy()
        if "regime" not in regime:
            raise KeyError("regime row missing required 'regime' key (no baseline default)")
        regime_name = regime["regime"]
        position = regime.get("position", "short")

        mask = self._apply_regime_mask(df, regime_name)
        filtered = df[mask].copy()

        if filtered.empty:
            logger.warning("Regime %r filtered to 0 rows", regime_name)
            return pd.DataFrame()

        filtered["position"] = position
        filtered["regime"] = regime_name

        # Override ret with the appropriate option P&L column for this position
        ret_col = _POSITION_TO_RET_COL.get(position)
        if ret_col and ret_col in filtered.columns:
            filtered["ret"] = filtered[ret_col]

        if save and out_dir:
            safe_name = "".join(
                c if c.isalnum() or c == "_" else "_"
                for c in f"{regime_name}_{position}"
            )
            save_dir = os.path.join(out_dir, "filter")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"filter_{safe_name}.parquet")
            try:
                # Obfuscation: code feature columns before write (greeks/IV/straddle
                # bases are in the feature map; targets/metadata pass through).
                _to_save = filtered.rename(columns=_obf().encode_columns(filtered.columns))
                _to_save.to_parquet(save_path, index=False, row_group_size=200_000)
                logger.info(
                    "Saved filter %s: %d rows (DOWN=%d, FLAT=%d, UP=%d)",
                    safe_name,
                    len(filtered),
                    (filtered["y"] == 0).sum(),
                    (filtered["y"] == 1).sum(),
                    (filtered["y"] == 2).sum(),
                )
            except Exception as exc:
                logger.error("Failed to save %s: %s", save_path, exc)

        return filtered

    def create(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """Orchestrate the full pipeline: load/engineer → verticalize → filter.

        Returns:
            Output directory path.
        """
        out_dir = self._resolve_out_dir()
        os.makedirs(out_dir, exist_ok=True)

        # Load cached daily features or build from raw data
        if not self._daily_frames:
            self.load()

        if not self._daily_frames:
            logger.info("No cached features found — running engineer_features()")
            self.engineer_features(start_date=start_date, end_date=end_date)

        self.verticalize()

        if self.vertical_features is None or self.vertical_features.empty:
            logger.warning("No weekly samples produced — aborting filter step")
            return out_dir

        # Save vertical features
        vf_path = os.path.join(out_dir, "vertical_features.parquet")
        self.vertical_features.to_parquet(vf_path, index=False)
        logger.info("Saved vertical_features.parquet (%d rows)", len(self.vertical_features))

        # Load and run regime stack
        regime_stack = self._load_regime_stack(out_dir)

        logger.info("Processing %d regimes", len(regime_stack))
        for reg in regime_stack:
            try:
                self.filter_signals(reg, save=True, out_dir=out_dir)
            except Exception as exc:
                logger.error("filter_signals failed for %s: %s", reg, exc)

        return out_dir

    # ------------------------------------------------------------------
    # Regime filter masks
    # ------------------------------------------------------------------

    def _apply_regime_mask(self, df: pd.DataFrame, regime_name: str) -> pd.Series:
        """Return boolean Series: True = include in this regime."""
        true = pd.Series(True, index=df.index)

        if df.empty:
            return pd.Series(dtype=bool)

        name = str(regime_name).lower().strip()
        # Accept coded regimes: decode code->real before the real-name chain.
        name = _obf().decode_regime_tolerant(name)

        if name == "baseline":
            raise ValueError(
                "baseline regime removed 2026-06-18 — unconditional fires-every-bar "
                "no-brainer; drop it from the regime stack. See CLAUDE.md.")

        if name == "high_iv_rank":
            col = "iv_rank"
            if col not in df.columns:
                return true
            return df[col] > 70

        if name == "low_iv_rank":
            col = "iv_rank"
            if col not in df.columns:
                return true
            return df[col] < 30

        if name == "backwardation":
            # term_structure_ratio < 1: weekly IV > monthly IV (stress)
            col = "term_structure_ratio"
            if col not in df.columns:
                return true
            return df[col] < 1.0

        if name == "contango":
            col = "term_structure_ratio"
            if col not in df.columns:
                return true
            return df[col] > 1.0

        if name == "positive_skew":
            # 25δ RR > 0: puts more expensive than calls (downside fear)
            col = "skew_25d_rr"
            if col not in df.columns:
                return true
            return df[col] > 0

        if name == "negative_skew":
            col = "skew_25d_rr"
            if col not in df.columns:
                return true
            return df[col] < 0

        if name == "oi_expansion":
            col = "oi_1d_change"
            if col not in df.columns:
                col = "total_oi"
                if col not in df.columns:
                    return true
                return df[col].diff() > 0
            return df[col] > 0

        if name == "high_hv":
            col = "hv_20d"
            if col not in df.columns:
                return true
            return df[col] > df[col].quantile(0.66)

        if name == "low_hv":
            col = "hv_20d"
            if col not in df.columns:
                return true
            return df[col] < df[col].quantile(0.33)

        # ── Spot OHLC-derived regimes ────────────────────────────────────────
        # These use columns added by add_rolling_features():
        #   rsi_7, spot_return_20d, spot_vs_ma20, hv_10d, hv_20d, iv_vs_hv20

        if name == "rsi_oversold":
            # 7-day RSI < 30: spot sold off hard, elevated IV → good time to sell vol
            col = "rsi_7"
            if col not in df.columns:
                return true
            return df[col] < 30

        if name == "rsi_overbought":
            # 7-day RSI > 70: spot extended, vol may compress → sell premium
            col = "rsi_7"
            if col not in df.columns:
                return true
            return df[col] > 70

        if name == "above_ma20":
            # Spot above 20-day MA: bullish trend, puts likely to expire OTM
            col = "spot_vs_ma20"
            if col not in df.columns:
                return true
            return df[col] > 0

        if name == "below_ma20":
            # Spot below 20-day MA: bearish trend, calls likely to expire OTM
            col = "spot_vs_ma20"
            if col not in df.columns:
                return true
            return df[col] < 0

        if name == "momentum_up":
            # 20-day return > +3%: sustained upward momentum
            col = "spot_return_20d"
            if col not in df.columns:
                return true
            return df[col] > 0.03

        if name == "momentum_down":
            # 20-day return < -3%: sustained downward momentum
            col = "spot_return_20d"
            if col not in df.columns:
                return true
            return df[col] < -0.03

        if name == "iv_premium":
            # ATM IV > 120% of 20-day HV: options overpriced vs realised vol
            col = "iv_vs_hv20"
            if col not in df.columns:
                return true
            return df[col] > 1.2

        if name == "iv_discount":
            # ATM IV < 80% of 20-day HV: options cheap vs realised vol
            col = "iv_vs_hv20"
            if col not in df.columns:
                return true
            return df[col] < 0.8

        if name == "vol_compression":
            # Short-term HV falling below longer-term: volatility regime contracting
            col10, col20 = "hv_10d", "hv_20d"
            if col10 not in df.columns or col20 not in df.columns:
                return true
            return df[col10] < df[col20]

        if name == "vol_expansion":
            # Short-term HV rising above longer-term: volatility regime expanding
            col10, col20 = "hv_10d", "hv_20d"
            if col10 not in df.columns or col20 not in df.columns:
                return true
            return df[col10] > df[col20]

        # Unknown regime: return all True (log warning once)
        logger.warning("Unknown regime filter %r — returning all rows", regime_name)
        return true

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_out_dir(self) -> str:
        version = self.config.get("VERSION")
        if not version:
            raise ValueError("VERSION missing from config.")
        if "OUTPUT_DIR" in self.config:
            out_dir = self.config["OUTPUT_DIR"]
        else:
            project = self.config.get("PROJECT", "valkyrie/gauntlet")
            out_dir = os.path.join(project, f"pred_{version}")
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(self.home_root, out_dir)
        return out_dir

    def _load_regime_stack(self, out_dir: str) -> List[Dict]:
        """Load regime stack CSV from REGIME_STACK_PATH in config. CSV-only, no fallbacks.

        Raises KeyError immediately if REGIME_STACK_PATH is not configured.
        Raises FileNotFoundError if the configured path does not exist.
        Raises ValueError if path does not end in .csv.
        """
        from utils.lib import load_regime_stack

        # Resolve relative paths before calling shared loader
        if "REGIME_STACK_PATH" not in self.config:
            raise KeyError("REGIME_STACK_PATH is required in config but not set")
        p = self.config["REGIME_STACK_PATH"]
        if not os.path.isabs(p):
            p = os.path.join(self.home_root, p)
        return load_regime_stack(p)


# ---------------------------------------------------------------------------
# Module-level helpers for option P&L simulation
# ---------------------------------------------------------------------------

def _settlement_value(
    atm_strike: float,
    spot_fri: float,
    opt_type: str,
    is_linear: bool,
) -> float:
    """Compute option intrinsic value at Friday settlement.

    Args:
        atm_strike: ATM strike used at entry.
        spot_fri:   Spot price at Friday 08:00 UTC settlement.
        opt_type:   "call" or "put".
        is_linear:  True for USDC-settled (linear); False for inverse (BTC/ETH-settled).

    Returns:
        Intrinsic value in the same units as mark_price (BTC/ETH or USDC).
        Returns np.nan if inputs are invalid.
    """
    if not (np.isfinite(atm_strike) and atm_strike > 0):
        return np.nan
    if not (np.isfinite(spot_fri) and spot_fri > 0):
        return np.nan

    if opt_type == "call":
        intrinsic = max(0.0, spot_fri - atm_strike)
    else:
        intrinsic = max(0.0, atm_strike - spot_fri)

    if is_linear:
        return float(intrinsic)  # USDC linear: value in USDC
    else:
        return float(intrinsic / spot_fri)  # Inverse: value in BTC/ETH


def _compute_option_ret(
    entry_price: float,
    hold_df: pd.DataFrame,
    price_high_col: str,
    stoploss_pct: float,
    settlement: float,
) -> float:
    """Compute short option P&L for one position, with daily stop-loss check.

    The position is entered at `entry_price` (premium received).
    On each hold day, if the option's daily high exceeds
    `entry_price * (1 + stoploss_pct)`, the position is stopped out
    and ret = -stoploss_pct.  Otherwise ret = (entry - settlement) / entry.

    Args:
        entry_price:    Option premium at entry (open of entry day).
        hold_df:        Rows for days after entry day (Tue–Fri for Mon entry).
        price_high_col: Column name for daily high option price.
        stoploss_pct:   Stop-loss threshold (e.g. 0.10 for 10%).
        settlement:     Intrinsic value at Friday settlement.

    Returns:
        Float P&L, or np.nan if entry_price is invalid.
    """
    if not (np.isfinite(entry_price) and entry_price > 0):
        return np.nan

    stop_threshold = entry_price * (1.0 + stoploss_pct)

    for _, hold_row in hold_df.iterrows():
        daily_high = hold_row.get(price_high_col)
        if daily_high is None:
            continue
        try:
            daily_high = float(daily_high)
        except (TypeError, ValueError):
            continue
        if np.isfinite(daily_high) and daily_high >= stop_threshold:
            return float(-stoploss_pct)  # stopped out

    # No stop triggered — P&L at settlement
    if not np.isfinite(settlement) or settlement < 0:
        return np.nan
    return float((entry_price - settlement) / entry_price)
