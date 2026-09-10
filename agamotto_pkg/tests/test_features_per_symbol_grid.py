"""Per-symbol features are computed on the symbol's OWN bars, not the joined grid.

`AgamottoResearch.load()` OUTER-joins every symbol's timestamp index
(`pd.concat(frames, axis=1)`), so a bar one symbol has and another lacks is a
NaN row in the other's columns. Before 2026-09-10 `engineer_features()` fed
that joined grid straight into TA-Lib and pandas, so an injected NaN row was
read as data. Measured on a 115-symbol US 15m panel: 24 of 73 feature columns
finite on exactly 17.2% of rows — every one a TA-Lib output whose C
implementation carries an interior NaN forward for good (running-sum SMA,
Wilder smoothing, EMA) — against 100% on the crypto control panel. The raw
per-symbol bars were clean (AAPL: 26 bars every day, 0 NaN); the NaN was
manufactured by the join (one symbol with 31 trading days the other 114 lack,
four ADRs with pre-market bars: 1,100 injected rows into AAPL's 2,522).

The pandas features were contaminated by the same mechanism, less visibly:
on a synthetic two-symbol panel where B lacks 5 interior bars, 64 of 84 B
columns differed from the own-bar computation (mvg1 on 155 rows, acf_lag1
fabricated 0.0 on 203, std/skew/kurt NaN on 271-294, the forward target NaN on
the bar before every gap, ret_lag*/vol_ret_lag* losing the return across it).

TWO CONTRACTS, in order of importance:

1. **PARITY.** This package feeds the live `pred_agamotto.base.15m_1` arm, and
   Binance perps share one exact grid, so on an aligned panel the change must
   be a bit-for-bit no-op. `_reference_old_path` below is the pre-change
   per-symbol block (research.py @ 209c899) transcribed verbatim: wide-frame
   pandas arithmetic, talib called on the joined arrays. `assert_frame_equal(
   check_exact=True)` against it, every engineered column.

2. **THE FIX.** B lacking 5 interior timestamps A has: B's TA columns equal
   talib on B's own bars reindexed; B's rsi is finite on >95% of B's own rows
   after warm-up (the old path, run as the negative control, is well under);
   A is unaffected; rows B genuinely lacks stay NaN — no forward-fill.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "agamotto_pkg/src")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib")

from agamotto.features_scalefree import scale_free_levels  # noqa: E402
from agamotto.ladder import compute_ladder_multiplier, ladder_params  # noqa: E402
from agamotto.research import AgamottoResearch  # noqa: E402

N = 600
GAP = list(range(300, 305))          # the 5 interior bars B lacks
FEE = 2.25
CFG_BASE = {
    "EXCHANGE": "BINANCE", "DATA": "liquid", "TIME_UNIT": "1h",
    "LADDER": 2, "LADDER_BPS": 5.0, "MA_PERIODS": [7, 25, 99],
    "STATS_WINDOW": 14, "FEE": FEE, "DUAL_HORIZON": True,
}


def _cfg(natives):
    return {**CFG_BASE, "SYMBOLS": [f"BINANCE_PERP_{s}_USDT" for s in natives]}


def _bars(rng, sym, idx):
    n = len(idx)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        f"{sym}_open": close - 0.05,
        f"{sym}_high": close + 0.5 + rng.random(n) * 0.2,
        f"{sym}_low": close - 0.5 - rng.random(n) * 0.2,
        f"{sym}_close": close,
        f"{sym}_volume": rng.integers(100, 1000, n).astype(float),
        f"{sym}_quote_volume": rng.integers(10_000, 100_000, n).astype(float),
        f"{sym}_number_of_trades": rng.integers(10, 100, n).astype(float),
        f"{sym}_taker_buy_quote_volume": rng.integers(1_000, 50_000, n).astype(float),
    }, index=idx)


def _features(raw, natives):
    r = AgamottoResearch(_cfg(natives), "/tmp/unused")
    r.raw = raw
    r.engineer_features()
    return r.features


# --------------------------------------------------------------------------
# The OLD code path, transcribed from research.py @ 209c899 (the last commit
# before the per-symbol change). Every series is built on the WIDE frame `df`
# and talib is called on the joined arrays — exactly what shipped. Kept
# private to this test so the parity assertion has an independent oracle.
# --------------------------------------------------------------------------
def _reference_old_path(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    ladder_long, ladder_short, step_bps = ladder_params(config)
    fee_rate = float(config["FEE"]) / 10000.0
    frames = [df]
    for col in df.columns:
        if not col.endswith("_close"):
            continue
        base = col[:-6]
        close = df[col]
        open_series = df.get(f"{base}_open", close)
        high_series = df.get(f"{base}_high", close)
        low_series = df.get(f"{base}_low", close)

        price_range = (high_series - low_series).rename(f"{base}_price_range")
        price_range_pct = ((high_series - low_series) / (open_series + 1e-8)).rename(f"{base}_price_range_pct")
        price_range_pct_q50 = price_range_pct.rolling(700, min_periods=1).quantile(0.5).rename(f"{base}_price_range_pct_q50")
        vol_quantile_cols = [
            price_range_pct.rolling(700, min_periods=700).quantile(level).rename(f"{base}_{name}")
            for level, name in zip((0.80, 0.90, 0.95),
                                   ("price_range_pct_q80", "price_range_pct_q90", "price_range_pct_q95"))
        ]
        open_close_diff = (close - open_series).rename(f"{base}_open_close_diff")
        open_close_pct = (open_close_diff / (open_series + 1e-8)).rename(f"{base}_open_close_pct")
        high_open_pct = ((high_series - open_series) / (open_series + 1e-8)).rename(f"{base}_high_open_pct")
        low_open_pct = ((low_series - open_series) / (open_series + 1e-8)).rename(f"{base}_low_open_pct")

        hist_return = close.pct_change(fill_method=None)
        price_return = hist_return.shift(-1)
        ret_lag1 = hist_return.shift(1).rename(f"{base}_ret_lag1")
        ret_lag2 = hist_return.shift(2).rename(f"{base}_ret_lag2")
        ret_lag3 = hist_return.shift(3).rename(f"{base}_ret_lag3")

        low_next = low_series.shift(-1)
        high_next = high_series.shift(-1)
        close_safe = close.replace(0, np.nan)
        size_long = compute_ladder_multiplier(close_safe, low_next, ladder_long, step_bps)
        size_short = compute_ladder_multiplier(close_safe, 2.0 * close_safe - high_next, ladder_short, step_bps)

        fee_cost = fee_rate * 2.0
        price_return_long = ((price_return - fee_cost) * size_long).rename(f"{base}_return_long")
        price_return_short = ((price_return + fee_cost) * size_short).rename(f"{base}_return_short")
        price_return_long_raw = (price_return * size_long).rename(f"{base}_return_long_raw")
        price_return_short_raw = (price_return * size_short).rename(f"{base}_return_short_raw")

        price_return_2bar = (close.shift(-2) / close_safe - 1)
        low_min2 = pd.concat([low_series.shift(-1), low_series.shift(-2)], axis=1).min(axis=1)
        high_max2 = pd.concat([high_series.shift(-1), high_series.shift(-2)], axis=1).max(axis=1)
        size_long2 = compute_ladder_multiplier(close_safe, low_min2, ladder_long, step_bps)
        size_short2 = compute_ladder_multiplier(close_safe, 2.0 * close_safe - high_max2, ladder_short, step_bps)
        ret_2bar = price_return_2bar.rename(f"{base}_ret_2bar")
        return_long_2bar = ((price_return_2bar - fee_cost) * size_long2).rename(f"{base}_return_long_2bar")
        return_short_2bar = ((price_return_2bar + fee_cost) * size_short2).rename(f"{base}_return_short_2bar")
        return_long_2bar_raw = (price_return_2bar * size_long2).rename(f"{base}_return_long_2bar_raw")
        return_short_2bar_raw = (price_return_2bar * size_short2).rename(f"{base}_return_short_2bar_raw")

        return_dip = (low_next / close_safe - 1).rename(f"{base}_return_dip")
        return_rip = (high_next / close_safe - 1).rename(f"{base}_return_rip")
        price_return_combined = price_return.rename(f"{base}_return")

        ma1_period, ma2_period, ma3_period = config["MA_PERIODS"]
        ma1 = close.rolling(int(ma1_period), min_periods=1).mean()
        ma2 = close.rolling(int(ma2_period), min_periods=1).mean()
        ma3 = close.rolling(int(ma3_period), min_periods=1).mean()

        volume_features = []
        vol = df[f"{base}_volume"]
        vol_ma = vol.rolling(7, min_periods=1).mean()
        volume_features.append((vol / (vol_ma + 1e-8)).rename(f"{base}_vol_ratio"))
        vol_ret = vol.pct_change(fill_method=None)
        volume_features.append(vol_ret.shift(1).rename(f"{base}_vol_ret_lag1"))
        volume_features.append(vol_ret.shift(2).rename(f"{base}_vol_ret_lag2"))
        volume_features.append(vol_ret.shift(3).rename(f"{base}_vol_ret_lag3"))
        quote_vol = df[f"{base}_quote_volume"]
        quote_vol_ma = quote_vol.rolling(7, min_periods=1).mean()
        volume_features.append((quote_vol / (quote_vol_ma + 1e-8)).rename(f"{base}_quote_vol_ratio"))
        taker_buy = df[f"{base}_taker_buy_quote_volume"]
        volume_features.append((taker_buy / (quote_vol + 1e-8)).rename(f"{base}_buy_pressure"))
        num_trades = df[f"{base}_number_of_trades"]
        trades_ma = num_trades.rolling(7, min_periods=1).mean()
        volume_features.append((num_trades / (trades_ma + 1e-8)).rename(f"{base}_trade_intensity"))

        c_vals = close.values.astype(float)
        h_vals = high_series.values.astype(float)
        l_vals = low_series.values.astype(float)
        v_vals = vol.values.astype(float)
        ix = df.index
        ta = []
        ta.append(pd.Series(talib.RSI(c_vals, timeperiod=14), index=ix, name=f"{base}_rsi"))
        ta.append(pd.Series(talib.RSI(c_vals, timeperiod=7), index=ix, name=f"{base}_rsi_7"))
        ta.append(pd.Series(talib.RSI(c_vals, timeperiod=28), index=ix, name=f"{base}_rsi_28"))
        macd, _, macdhist = talib.MACD(c_vals, fastperiod=12, slowperiod=26, signalperiod=9)
        ta.append(pd.Series(macd, index=ix, name=f"{base}_macd"))
        ta.append(pd.Series(macdhist, index=ix, name=f"{base}_macdhist"))
        slowk, slowd = talib.STOCH(h_vals, l_vals, c_vals, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        ta.append(pd.Series(slowk, index=ix, name=f"{base}_stoch_k"))
        ta.append(pd.Series(slowd, index=ix, name=f"{base}_stoch_d"))
        ta.append(pd.Series(talib.CCI(h_vals, l_vals, c_vals, timeperiod=14), index=ix, name=f"{base}_cci"))
        ta.append(pd.Series(talib.ADX(h_vals, l_vals, c_vals, timeperiod=14), index=ix, name=f"{base}_adx"))
        ta.append(pd.Series(talib.DX(h_vals, l_vals, c_vals, timeperiod=14), index=ix, name=f"{base}_dx"))
        ta.append(pd.Series(talib.PLUS_DI(h_vals, l_vals, c_vals, timeperiod=14), index=ix, name=f"{base}_plus_di"))
        ta.append(pd.Series(talib.MINUS_DI(h_vals, l_vals, c_vals, timeperiod=14), index=ix, name=f"{base}_minus_di"))
        ta.append(pd.Series(talib.MOM(c_vals, timeperiod=10), index=ix, name=f"{base}_mom"))
        ta.append(pd.Series(talib.ROC(c_vals, timeperiod=10), index=ix, name=f"{base}_roc"))
        ta.append(pd.Series(talib.WILLR(h_vals, l_vals, c_vals, timeperiod=14), index=ix, name=f"{base}_willr"))
        ta.append(pd.Series(talib.CMO(c_vals, timeperiod=14), index=ix, name=f"{base}_cmo"))
        ta.append(pd.Series(talib.TRIX(c_vals, timeperiod=30), index=ix, name=f"{base}_trix"))
        ta.append(pd.Series(talib.ULTOSC(h_vals, l_vals, c_vals, timeperiod1=7, timeperiod2=14, timeperiod3=28), index=ix, name=f"{base}_ultosc"))
        fastk, fastd = talib.STOCHRSI(c_vals, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        ta.append(pd.Series(fastk, index=ix, name=f"{base}_stochrsi_k"))
        ta.append(pd.Series(fastd, index=ix, name=f"{base}_stochrsi_d"))
        obv_raw = pd.Series(talib.OBV(c_vals, v_vals), index=ix)
        ad_raw = pd.Series(talib.AD(h_vals, l_vals, c_vals, v_vals), index=ix)
        ta.append(obv_raw.diff(14).fillna(0.0).rename(f"{base}_obv"))
        ta.append(ad_raw.diff(14).fillna(0.0).rename(f"{base}_ad"))
        ta.append(pd.Series(talib.MFI(h_vals, l_vals, c_vals, v_vals, timeperiod=14), index=ix, name=f"{base}_mfi"))
        ta.append(pd.Series(talib.BOP(open_series.values.astype(float), h_vals, l_vals, c_vals), index=ix, name=f"{base}_bop"))
        ta.append(pd.Series(talib.ATR(h_vals, l_vals, c_vals, timeperiod=14), index=ix, name=f"{base}_atr"))
        ta.append(pd.Series(talib.NATR(h_vals, l_vals, c_vals, timeperiod=14), index=ix, name=f"{base}_natr"))
        ta.append(np.sqrt(1.0 / (4.0 * np.log(2)) * (np.log(high_series / low_series) ** 2))
                  .rolling(14).mean().rename(f"{base}_parkinson_vol"))
        upper, _, lower = talib.BBANDS(c_vals, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        ta.append(pd.Series(upper, index=ix, name=f"{base}_bb_upper"))
        ta.append(pd.Series(lower, index=ix, name=f"{base}_bb_lower"))
        ta.append(pd.Series(talib.SAR(h_vals, l_vals, acceleration=0.02, maximum=0.2), index=ix, name=f"{base}_sar"))

        w = int(config["STATS_WINDOW"])
        rolling_stats = [
            hist_return.rolling(window=w).std().rename(f"{base}_std"),
            hist_return.rolling(window=w).skew().rename(f"{base}_skew"),
            hist_return.rolling(window=w).kurt().rename(f"{base}_kurt"),
            hist_return.rolling(window=w - 1).corr(hist_return.shift(1)).fillna(0.0).rename(f"{base}_acf_lag1"),
        ]

        frames.extend([
            price_range, price_range_pct, price_range_pct_q50, open_close_diff,
            open_close_pct, high_open_pct, low_open_pct, price_return_combined,
            price_return_long, price_return_short, price_return_long_raw,
            price_return_short_raw, return_dip, return_rip, ret_lag1, ret_lag2,
            ret_lag3, ma1.rename(f"{base}_mvg1"), ma2.rename(f"{base}_mvg2"),
            ma3.rename(f"{base}_mvg3"),
        ] + vol_quantile_cols + volume_features + ta + rolling_stats)

        by_name = {s.name: s for s in ta}
        src = {f"{base}_close": close, f"{base}_volume": vol}
        for n in ("sar", "bb_upper", "bb_lower", "macd", "macdhist", "obv", "ad"):
            src[f"{base}_{n}"] = by_name[f"{base}_{n}"]
        frames.extend([s for _, s in scale_free_levels(
            pd.DataFrame(src), prefix=f"{base}_", obv_is_cumulative=False).items()])
        frames.extend([ret_2bar, return_long_2bar, return_short_2bar,
                       return_long_2bar_raw, return_short_2bar_raw])

    out = pd.concat(frames, axis=1)
    out["year"] = out.index.year
    out["month"] = out.index.month
    out["close_timestamp"] = out.index + pd.Timedelta(seconds=3600)
    return out


TA_NAMES = ["rsi", "rsi_7", "rsi_28", "macd", "macdhist", "stoch_k", "stoch_d",
            "cci", "adx", "dx", "plus_di", "minus_di", "mom", "roc", "willr",
            "cmo", "trix", "ultosc", "stochrsi_k", "stochrsi_d", "obv", "ad",
            "mfi", "bop", "atr", "natr", "bb_upper", "bb_lower", "sar"]


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=N, freq="1h")
    a = _bars(rng, "AAA", idx)
    b_full = _bars(rng, "BBB", idx)
    c = _bars(rng, "CCC", idx)
    b_gapped = b_full.drop(b_full.index[GAP])
    return {
        "aligned": pd.concat([a, b_full, c], axis=1).sort_index(),
        "gapped": pd.concat([a, b_gapped], axis=1).sort_index(),
        "a": a, "b_gapped": b_gapped,
    }


# --------------------------------------------------------------------------
# 1. PARITY — the aligned (crypto) panel is bit-identical to the old path.
# --------------------------------------------------------------------------
def test_aligned_panel_is_bit_identical_to_old_path(panels):
    raw = panels["aligned"]
    new = _features(raw, ["AAA", "BBB", "CCC"])
    old = _reference_old_path(raw, _cfg(["AAA", "BBB", "CCC"]))

    assert list(new.columns) == list(old.columns), "column set/order changed"
    pd.testing.assert_frame_equal(new, old, check_exact=True)
    # Belt and braces on the float payload: NaN-aware, bit-for-bit.
    num = [c for c in new.columns if c != "close_timestamp"]
    np.testing.assert_array_equal(new[num].to_numpy(dtype=float),
                                  old[num].to_numpy(dtype=float))


# --------------------------------------------------------------------------
# 2. THE FIX — B lacks 5 interior bars A has.
# --------------------------------------------------------------------------
def test_gapped_symbol_ta_equals_talib_on_own_bars(panels):
    raw = panels["gapped"]
    b = panels["b_gapped"]
    feats = _features(raw, ["AAA", "BBB"])

    c = b["BBB_close"].to_numpy(dtype=float)
    h = b["BBB_high"].to_numpy(dtype=float)
    lo = b["BBB_low"].to_numpy(dtype=float)
    o = b["BBB_open"].to_numpy(dtype=float)
    v = b["BBB_volume"].to_numpy(dtype=float)
    own = lambda arr: pd.Series(arr, index=b.index).reindex(raw.index).to_numpy(dtype=float)  # noqa: E731

    expected = {
        "rsi": talib.RSI(c, timeperiod=14),
        "rsi_7": talib.RSI(c, timeperiod=7),
        "rsi_28": talib.RSI(c, timeperiod=28),
        "macd": talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0],
        "macdhist": talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[2],
        "stoch_k": talib.STOCH(h, lo, c, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0],
        "stoch_d": talib.STOCH(h, lo, c, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[1],
        "cci": talib.CCI(h, lo, c, timeperiod=14),
        "adx": talib.ADX(h, lo, c, timeperiod=14),
        "dx": talib.DX(h, lo, c, timeperiod=14),
        "plus_di": talib.PLUS_DI(h, lo, c, timeperiod=14),
        "minus_di": talib.MINUS_DI(h, lo, c, timeperiod=14),
        "mom": talib.MOM(c, timeperiod=10),
        "roc": talib.ROC(c, timeperiod=10),
        "willr": talib.WILLR(h, lo, c, timeperiod=14),
        "cmo": talib.CMO(c, timeperiod=14),
        "trix": talib.TRIX(c, timeperiod=30),
        "ultosc": talib.ULTOSC(h, lo, c, timeperiod1=7, timeperiod2=14, timeperiod3=28),
        "stochrsi_k": talib.STOCHRSI(c, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[0],
        "stochrsi_d": talib.STOCHRSI(c, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[1],
        "obv": pd.Series(talib.OBV(c, v)).diff(14).fillna(0.0).to_numpy(),
        "ad": pd.Series(talib.AD(h, lo, c, v)).diff(14).fillna(0.0).to_numpy(),
        "mfi": talib.MFI(h, lo, c, v, timeperiod=14),
        "bop": talib.BOP(o, h, lo, c),
        "atr": talib.ATR(h, lo, c, timeperiod=14),
        "natr": talib.NATR(h, lo, c, timeperiod=14),
        "bb_upper": talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0],
        "bb_lower": talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2],
        "sar": talib.SAR(h, lo, acceleration=0.02, maximum=0.2),
    }
    assert set(expected) == set(TA_NAMES)
    for name, arr in expected.items():
        np.testing.assert_array_equal(
            feats[f"BBB_{name}"].to_numpy(dtype=float), own(arr), err_msg=name)

    # The scale-free twins derive from these outputs and inherit the fix.
    close = feats["BBB_close"]
    bbu, bbl = feats["BBB_bb_upper"], feats["BBB_bb_lower"]
    pd.testing.assert_series_equal(
        feats["BBB_bb_pctb"], ((close - bbl) / (bbu - bbl)).rename("BBB_bb_pctb"),
        check_exact=True)
    pd.testing.assert_series_equal(
        feats["BBB_macd_norm"], (feats["BBB_macd"] / close).rename("BBB_macd_norm"),
        check_exact=True)


def test_gapped_symbol_rsi_finite_after_warmup_and_old_path_was_not(panels):
    raw = panels["gapped"]
    b_rows = panels["b_gapped"].index
    feats = _features(raw, ["AAA", "BBB"])

    after_warmup = b_rows[14:]                       # RSI(14) lookback
    frac_new = np.isfinite(feats.loc[after_warmup, "BBB_rsi"]).mean()
    assert frac_new > 0.95, frac_new
    assert frac_new == 1.0                           # every own bar after warm-up

    # Negative control: the old path on the same panel loses everything after
    # the gap — TA-Lib's RSI is Wilder-smoothed, so one interior NaN is final.
    old = _reference_old_path(raw, _cfg(["AAA", "BBB"]))
    frac_old = np.isfinite(old.loc[after_warmup, "BBB_rsi"]).mean()
    assert frac_old < 0.95, frac_old
    assert not np.isfinite(old.loc[b_rows[GAP[0] + 1:], "BBB_rsi"]).any()


def test_gapped_panel_leaves_other_symbol_unaffected(panels):
    feats = _features(panels["gapped"], ["AAA", "BBB"])
    alone = _features(panels["a"], ["AAA"])
    a_cols = [c for c in feats.columns if c.startswith("AAA_")]
    assert a_cols == [c for c in alone.columns if c.startswith("AAA_")]
    np.testing.assert_array_equal(feats[a_cols].to_numpy(dtype=float),
                                  alone[a_cols].to_numpy(dtype=float))


def test_gapped_symbol_missing_rows_stay_nan(panels):
    raw = panels["gapped"]
    feats = _features(raw, ["AAA", "BBB"])
    missing = raw.index.difference(panels["b_gapped"].index)
    assert len(missing) == len(GAP)
    b_cols = [c for c in feats.columns if c.startswith("BBB_")]
    assert feats.loc[missing, b_cols].isna().all().all(), \
        "a row B has no bar for must carry no fabricated B feature"


def test_partial_nan_row_in_own_data_keeps_old_semantics(panels):
    """A NaN CELL in a bar the symbol does have is not a missing bar.

    The join manufactures rows that are NaN in EVERY one of the symbol's
    columns; a bar with a NaN close beside a finite volume is the symbol's own
    data and must keep pandas/TA-Lib's NaN semantics unchanged — that is what
    agamotto_core's feature engine (the live sentinel bots) reproduces, and
    what its parity harness pins on its "NaN holes" scenario. A close-only
    `own` rule breaks that gate; the any-column rule below must not.
    """
    raw = panels["a"].copy()
    for c in ("AAA_open", "AAA_high", "AAA_low", "AAA_close"):
        raw.iloc[13, raw.columns.get_loc(c)] = np.nan
        raw.iloc[100:104, raw.columns.get_loc(c)] = np.nan
        raw.iloc[N - 2, raw.columns.get_loc(c)] = np.nan
    raw.iloc[50:53, raw.columns.get_loc("AAA_volume")] = np.nan
    raw.iloc[200, raw.columns.get_loc("AAA_high")] = np.nan
    assert raw.notna().any(axis=1).all()             # no all-NaN row: every row is a bar

    new = _features(raw, ["AAA"])
    old = _reference_old_path(raw, _cfg(["AAA"]))
    assert list(new.columns) == list(old.columns)
    pd.testing.assert_frame_equal(new, old, check_exact=True)
    # And the propagation really is there to preserve: RSI is Wilder-smoothed,
    # so the interior NaN close at row 100 is final on the old and new path alike.
    assert not np.isfinite(new["AAA_rsi"].iloc[100:]).any()


def test_gapped_symbol_pandas_features_equal_own_bar_computation(panels):
    """The pandas features were contaminated by the same join; pin the fix."""
    raw = panels["gapped"]
    b = panels["b_gapped"]
    feats = _features(raw, ["AAA", "BBB"])
    alone = _features(b, ["BBB"])
    b_cols = [c for c in alone.columns if c.startswith("BBB_")]
    np.testing.assert_array_equal(
        feats.loc[b.index, b_cols].to_numpy(dtype=float),
        alone.loc[b.index, b_cols].to_numpy(dtype=float))
