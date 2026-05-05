"""DailyFeatureBuilder: build daily ATM-IV, skew, term structure features."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import (
    get_atm_strike,
    get_delta_strike,
    get_expiry_buckets,
    iv_percentile,
    iv_rank,
    rsi_manual,
)

logger = logging.getLogger(__name__)


class DailyFeatureBuilder:
    """Compute daily option-based features from intra-day snapshot DataFrames.

    Produces one row per (asset, date) with:
    - Spot: open/close/settlement prices (rolling returns added separately)
    - ATM IV: open/high/low/close for call, put, and average
    - Skew: 25δ and 10δ risk reversals, 25δ butterfly
    - Term structure: weekly/monthly IV ratio
    - OI: total, put/call ratio, call/put breakdown
    - Greeks: ATM gamma, vega, theta, delta_call
    - Temporal: day_of_week, days_to_friday, week_number, days_to_expiry

    Rolling/historical features (IV rank, spot returns, HV) are added by
    add_rolling_features() which requires the full daily history.

    Args:
        settlement_hour_utc: Hour of Deribit daily settlement (default 8).
        max_weekly_days: Max days-to-expiry for "weekly" expiry bucket.
    """

    def __init__(self, settlement_hour_utc: int = 8, max_weekly_days: int = 7) -> None:
        self.settlement_hour_utc = settlement_hour_utc
        self.max_weekly_days = max_weekly_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_daily_row(
        self,
        df_day: pd.DataFrame,
        asset: str,
        date_str: str,
    ) -> Optional[Dict]:
        """Build one raw daily feature dict from intra-day option snapshots.

        Rolling features are NOT computed here; call add_rolling_features()
        on the accumulated daily DataFrame afterward.

        Args:
            df_day: All option snapshots for this day, already filtered to asset.
            asset:  Asset name (e.g. "BTC").
            date_str: Date in YYYYMMDD format.

        Returns:
            Feature dict, or None if data is insufficient.
        """
        if df_day.empty or "underlying_price" not in df_day.columns:
            return None

        df_day = df_day.sort_values("timestamp").copy()
        spot_series = df_day.groupby("timestamp")["underlying_price"].first()

        if spot_series.empty:
            return None

        spot_open = float(spot_series.iloc[0])
        spot_close = float(spot_series.iloc[-1])

        if spot_open <= 0:
            return None

        # Current date as UTC Timestamp
        d = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}", tz="UTC")
        settlement_ts = d + pd.Timedelta(hours=self.settlement_hour_utc)

        # Spot closest to settlement time (08:00 UTC)
        time_diff = (df_day["timestamp"] - settlement_ts).abs()
        spot_at_settlement = float(df_day.loc[time_diff.idxmin(), "underlying_price"])

        # Reference (midday) for expiry bucketing — microseconds
        midday_ts = d + pd.Timedelta(hours=12)
        midday_us = int(midday_ts.value // 1000)  # ns → µs

        # Weekly and monthly expiry buckets
        if "expiration" not in df_day.columns:
            return None

        weekly_exp, monthly_exp = get_expiry_buckets(
            df_day["expiration"], midday_us, self.max_weekly_days
        )
        if weekly_exp is None:
            return None

        df_weekly = df_day[df_day["expiration"] == weekly_exp].copy()
        if df_weekly.empty:
            return None

        # ATM strike fixed at market open using first snapshot of the day
        first_ts = df_weekly["timestamp"].min()
        df_open = df_weekly[df_weekly["timestamp"] == first_ts]
        atm_strike = get_atm_strike(df_open["strike_price"].values, spot_open)
        if atm_strike is None:
            return None

        # ATM call/put IV OHLC and mark_price OHLC
        atm_call_iv = self._iv_ohlc(df_weekly, atm_strike, "call")
        atm_put_iv = self._iv_ohlc(df_weekly, atm_strike, "put")
        opt_prices = self._option_price_ohlc(df_weekly, atm_strike)

        # Average ATM IV (call + put mean for each OHLC point)
        avg_close = _safe_mean([
            atm_call_iv.get("close") if atm_call_iv else None,
            atm_put_iv.get("close") if atm_put_iv else None,
        ])
        avg_open = _safe_mean([
            atm_call_iv.get("open") if atm_call_iv else None,
            atm_put_iv.get("open") if atm_put_iv else None,
        ])
        avg_high = _safe_mean([
            atm_call_iv.get("high") if atm_call_iv else None,
            atm_put_iv.get("high") if atm_put_iv else None,
        ])
        avg_low = _safe_mean([
            atm_call_iv.get("low") if atm_call_iv else None,
            atm_put_iv.get("low") if atm_put_iv else None,
        ])

        # Skew and term structure
        skew_25, skew_10, butterfly_25 = self._compute_skew(df_weekly)
        term_ratio = self._compute_term_structure(
            df_weekly, df_day, monthly_exp, weekly_exp, atm_strike, spot_open
        )

        # OI and Greeks
        oi = self._compute_oi(df_weekly)
        greeks = self._compute_greeks(df_weekly, atm_strike)

        # Temporal
        weekly_exp_dt = pd.Timestamp(weekly_exp, unit="us", tz="UTC")
        days_to_expiry = max(0, (weekly_exp_dt.date() - d.date()).days)
        dow = d.dayofweek  # 0=Mon, ..., 6=Sun
        days_to_friday = (4 - dow) % 7

        row: Dict = {
            "date": date_str,
            "asset": asset,
            "year": int(date_str[:4]),
            "month": int(date_str[4:6]),
            "day_of_week": dow,
            "days_to_friday": days_to_friday,
            "week_number": int(d.isocalendar()[1]),
            "days_to_expiry": days_to_expiry,
            # Spot
            "spot_open": spot_open,
            "spot_close": spot_close,
            "spot_settlement": spot_at_settlement,
            # ATM call IV
            "atm_call_iv_open": atm_call_iv.get("open") if atm_call_iv else np.nan,
            "atm_call_iv_high": atm_call_iv.get("high") if atm_call_iv else np.nan,
            "atm_call_iv_low": atm_call_iv.get("low") if atm_call_iv else np.nan,
            "atm_call_iv_close": atm_call_iv.get("close") if atm_call_iv else np.nan,
            # ATM put IV
            "atm_put_iv_open": atm_put_iv.get("open") if atm_put_iv else np.nan,
            "atm_put_iv_high": atm_put_iv.get("high") if atm_put_iv else np.nan,
            "atm_put_iv_low": atm_put_iv.get("low") if atm_put_iv else np.nan,
            "atm_put_iv_close": atm_put_iv.get("close") if atm_put_iv else np.nan,
            # ATM average IV
            "atm_avg_iv_open": avg_open,
            "atm_avg_iv_high": avg_high,
            "atm_avg_iv_low": avg_low,
            "atm_avg_iv_close": avg_close,
            # Skew
            "skew_25d_rr": skew_25,
            "skew_10d_rr": skew_10,
            "butterfly_25d": butterfly_25,
            # Term structure
            "term_structure_ratio": term_ratio,
            # OI
            "total_oi": oi.get("total_oi", np.nan),
            "call_oi": oi.get("call_oi", np.nan),
            "put_oi": oi.get("put_oi", np.nan),
            "put_call_oi_ratio": oi.get("put_call_oi_ratio", np.nan),
            # Greeks
            "atm_gamma": greeks.get("gamma", np.nan),
            "atm_vega": greeks.get("vega", np.nan),
            "atm_theta": greeks.get("theta", np.nan),
            "atm_delta_call": greeks.get("delta_call", np.nan),
            # ATM mark_price OHLC (for option P&L / stop-loss simulation)
            # Mark_price unit: BTC/ETH (inverse) or USDC (linear) — consistent within asset
            "atm_call_price_open": opt_prices.get("atm_call_price_open", np.nan),
            "atm_call_price_high": opt_prices.get("atm_call_price_high", np.nan),
            "atm_call_price_close": opt_prices.get("atm_call_price_close", np.nan),
            "atm_put_price_open": opt_prices.get("atm_put_price_open", np.nan),
            "atm_put_price_high": opt_prices.get("atm_put_price_high", np.nan),
            "atm_put_price_close": opt_prices.get("atm_put_price_close", np.nan),
            "atm_straddle_price_open": opt_prices.get("atm_straddle_price_open", np.nan),
            "atm_straddle_price_high": opt_prices.get("atm_straddle_price_high", np.nan),
            "atm_straddle_price_close": opt_prices.get("atm_straddle_price_close", np.nan),
            # ATM strike used (needed for settlement value calculation in verticalize)
            "atm_strike_used": atm_strike,
        }
        return row

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling/historical features to a daily feature DataFrame.

        Requires df sorted by (asset, date) with spot_close and atm_avg_iv_close.

        Adds per asset (sorted chronologically):
        - spot_return_{5,10,20,60}d: N-day percentage return
        - hv_{10,20}d: annualised historical volatility
        - spot_vs_ma20: (spot / 20d MA) - 1
        - rsi_7: 7-day RSI of spot_close
        - iv_rank: rolling 252-day IV rank (0-100)
        - iv_pctile: rolling 252-day IV percentile
        - iv_vs_hv20: ATM IV / (HV20 * 100)
        - oi_1d_change: day-over-day change in total_oi

        Args:
            df: DataFrame returned by stacking build_daily_row() results.

        Returns:
            New DataFrame with additional feature columns.
        """
        df = df.copy()
        result_frames: List[pd.DataFrame] = []

        for asset, grp in df.groupby("asset"):
            grp = grp.sort_values("date").reset_index(drop=True).copy()
            spot = grp["spot_close"]

            # Spot returns
            for n in [5, 10, 20, 60]:
                grp[f"spot_return_{n}d"] = spot.pct_change(n)

            # Historical volatility (annualised)
            log_ret = np.log(spot / spot.shift(1))
            for n in [10, 20]:
                grp[f"hv_{n}d"] = log_ret.rolling(n, min_periods=max(2, n // 2)).std() * np.sqrt(252)

            # Spot vs 20-day MA
            ma20 = spot.rolling(20, min_periods=5).mean()
            grp["spot_vs_ma20"] = (spot - ma20) / ma20.replace(0, np.nan)

            # RSI-7 (daily spot)
            grp["rsi_7"] = rsi_manual(spot, period=7).values

            # IV rank and percentile
            iv_col = "atm_avg_iv_close"
            if iv_col not in grp.columns:
                iv_col = "atm_call_iv_close"

            if iv_col in grp.columns:
                iv_series = grp[iv_col]
                grp["iv_rank"] = iv_rank(iv_series).values
                grp["iv_pctile"] = iv_percentile(iv_series).values
                hv20 = grp.get("hv_20d", pd.Series(np.nan, index=grp.index))
                grp["iv_vs_hv20"] = iv_series / (hv20 * 100).replace(0, np.nan)
            else:
                grp["iv_rank"] = 50.0
                grp["iv_pctile"] = 50.0
                grp["iv_vs_hv20"] = np.nan

            # OI 1-day change
            if "total_oi" in grp.columns:
                grp["oi_1d_change"] = grp["total_oi"].diff()

            result_frames.append(grp)

        if result_frames:
            return pd.concat(result_frames, ignore_index=True)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _option_price_ohlc(self, df_weekly: pd.DataFrame, atm_strike: float) -> Dict:
        """Compute OHLC of ATM mark_price for call, put, and straddle.

        For the straddle, uses synchronized timestamps so max(call+put) is accurate
        (not the overestimate of max_call + max_put).  Falls back to summing
        per-leg highs/lows if the two legs have no overlapping timestamps.

        Args:
            df_weekly: Options DataFrame filtered to weekly expiry.
            atm_strike: ATM strike determined at market open.

        Returns:
            Dict with keys: atm_call_price_{open,high,close},
            atm_put_price_{open,high,close},
            atm_straddle_price_{open,high,close}.
        """
        call_sub = (
            df_weekly[(df_weekly["strike_price"] == atm_strike) & (df_weekly["type"] == "call")]
            [["timestamp", "mark_price"]].dropna().sort_values("timestamp")
        )
        put_sub = (
            df_weekly[(df_weekly["strike_price"] == atm_strike) & (df_weekly["type"] == "put")]
            [["timestamp", "mark_price"]].dropna().sort_values("timestamp")
        )

        result: Dict = {}

        def _ohlc_from(sub: pd.DataFrame, prefix: str) -> None:
            if not sub.empty:
                prices = sub["mark_price"]
                result[f"{prefix}_open"] = float(prices.iloc[0])
                result[f"{prefix}_high"] = float(prices.max())
                result[f"{prefix}_close"] = float(prices.iloc[-1])
            else:
                result[f"{prefix}_open"] = np.nan
                result[f"{prefix}_high"] = np.nan
                result[f"{prefix}_close"] = np.nan

        _ohlc_from(call_sub, "atm_call_price")
        _ohlc_from(put_sub, "atm_put_price")

        # Straddle: try synchronized timestamps first
        if not call_sub.empty and not put_sub.empty:
            merged = (
                call_sub.rename(columns={"mark_price": "c"})
                .merge(put_sub.rename(columns={"mark_price": "p"}), on="timestamp", how="inner")
            )
            if not merged.empty:
                straddle = merged["c"] + merged["p"]
                result["atm_straddle_price_open"] = float(straddle.iloc[0])
                result["atm_straddle_price_high"] = float(straddle.max())
                result["atm_straddle_price_close"] = float(straddle.iloc[-1])
            else:
                # No timestamp overlap — sum of per-leg values (conservative for high)
                result["atm_straddle_price_open"] = (
                    result["atm_call_price_open"] + result["atm_put_price_open"]
                )
                result["atm_straddle_price_high"] = (
                    result["atm_call_price_high"] + result["atm_put_price_high"]
                )
                result["atm_straddle_price_close"] = (
                    result["atm_call_price_close"] + result["atm_put_price_close"]
                )
        else:
            result["atm_straddle_price_open"] = np.nan
            result["atm_straddle_price_high"] = np.nan
            result["atm_straddle_price_close"] = np.nan

        return result

    def _iv_ohlc(
        self,
        df: pd.DataFrame,
        strike: float,
        opt_type: str,
    ) -> Optional[Dict]:
        """Compute OHLC for mark_iv of a given strike and option type."""
        sub = df[(df["strike_price"] == strike) & (df["type"] == opt_type)]
        sub = sub.sort_values("timestamp")
        iv = sub["mark_iv"].dropna()
        if iv.empty:
            return None
        return {
            "open": float(iv.iloc[0]),
            "high": float(iv.max()),
            "low": float(iv.min()),
            "close": float(iv.iloc[-1]),
        }

    def _compute_skew(
        self,
        df_weekly: pd.DataFrame,
    ) -> Tuple[float, float, float]:
        """Compute 25δ RR, 10δ RR, and 25δ butterfly.

        Returns:
            (skew_25d_rr, skew_10d_rr, butterfly_25d) — np.nan if unavailable.
        """
        if df_weekly.empty:
            return np.nan, np.nan, np.nan

        df_calls = df_weekly[df_weekly["type"] == "call"]
        df_puts = df_weekly[df_weekly["type"] == "put"]

        strike_25c = get_delta_strike(df_calls, 0.25)
        strike_25p = get_delta_strike(df_puts, -0.25)
        strike_10c = get_delta_strike(df_calls, 0.10)
        strike_10p = get_delta_strike(df_puts, -0.10)

        def _get_iv(df: pd.DataFrame, strike: Optional[float]) -> float:
            if strike is None:
                return np.nan
            vals = df[df["strike_price"] == strike]["mark_iv"].dropna()
            return float(vals.mean()) if not vals.empty else np.nan

        iv_25c = _get_iv(df_calls, strike_25c)
        iv_25p = _get_iv(df_puts, strike_25p)
        iv_10c = _get_iv(df_calls, strike_10c)
        iv_10p = _get_iv(df_puts, strike_10p)

        rr_25 = iv_25p - iv_25c
        rr_10 = iv_10p - iv_10c
        butterfly_25 = (
            0.5 * (iv_25c + iv_25p)
            if not (np.isnan(iv_25c) or np.isnan(iv_25p))
            else np.nan
        )

        return float(rr_25), float(rr_10), float(butterfly_25)

    def _compute_term_structure(
        self,
        df_weekly: pd.DataFrame,
        df_day: pd.DataFrame,
        monthly_exp: Optional[int],
        weekly_exp: int,
        atm_strike: float,
        spot_price: float,
    ) -> float:
        """Compute weekly ATM IV / monthly ATM IV ratio (contango > 1, backwardation < 1)."""
        if monthly_exp is None or monthly_exp == weekly_exp:
            return np.nan

        # Weekly ATM call IV (median across day)
        weekly_iv = df_weekly[
            (df_weekly["strike_price"] == atm_strike) &
            (df_weekly["type"] == "call")
        ]["mark_iv"].median()

        if pd.isna(weekly_iv):
            return np.nan

        df_monthly = df_day[df_day["expiration"] == monthly_exp]
        if df_monthly.empty:
            return np.nan

        atm_monthly = get_atm_strike(df_monthly["strike_price"].values, spot_price)
        if atm_monthly is None:
            return np.nan

        monthly_iv = df_monthly[
            (df_monthly["strike_price"] == atm_monthly) &
            (df_monthly["type"] == "call")
        ]["mark_iv"].median()

        if pd.isna(monthly_iv) or monthly_iv <= 0:
            return np.nan

        return float(weekly_iv / monthly_iv)

    def _compute_oi(self, df_weekly: pd.DataFrame) -> Dict:
        """Open interest features for the weekly expiry."""
        if df_weekly.empty or "open_interest" not in df_weekly.columns:
            return {}

        # Use last snapshot for most up-to-date OI
        last_ts = df_weekly["timestamp"].max()
        last_snap = df_weekly[df_weekly["timestamp"] == last_ts]

        call_oi = float(last_snap[last_snap["type"] == "call"]["open_interest"].sum())
        put_oi = float(last_snap[last_snap["type"] == "put"]["open_interest"].sum())
        total_oi = call_oi + put_oi
        pc_ratio = (put_oi / call_oi) if call_oi > 0 else np.nan

        return {
            "total_oi": total_oi,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "put_call_oi_ratio": pc_ratio,
        }

    def _compute_greeks(self, df_weekly: pd.DataFrame, atm_strike: float) -> Dict:
        """Extract mean ATM Greeks across all snapshots of the day."""
        if df_weekly.empty:
            return {}

        atm = df_weekly[df_weekly["strike_price"] == atm_strike]
        if atm.empty:
            return {}

        result: Dict = {}
        for col in ("gamma", "vega", "theta"):
            if col in atm.columns:
                val = atm[col].dropna().mean()
                result[col] = float(val) if not pd.isna(val) else np.nan

        atm_call = atm[atm["type"] == "call"]
        if "delta" in atm_call.columns and not atm_call.empty:
            result["delta_call"] = float(atm_call["delta"].mean())

        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: List) -> float:
    """Mean of non-None, non-NaN values; returns np.nan if none valid."""
    valid = [v for v in values if v is not None and not np.isnan(float(v))]
    return float(np.mean(valid)) if valid else np.nan
