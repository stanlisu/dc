"""TA indicators priced off the BOOK, not off the last trade.

``StreamAligner(fill_zero_trade=False)`` leaves ``open/high/low/close/vwap``
NaN on a bar that had no trade — which is the truth — but TA-Lib has no NaN
semantics: one interior NaN propagates to the END of every RSI/MACD/ATR/
BBANDS/ADX/STDDEV series. Measured on the 5s corpus, the 21 columns in
``POISONED_21`` go 99.7% NaN on XMRUSDT and 53.4% on SOLUSDT and are dropped
by the IC screen. An A/B on those bars compares fabricated indicators against
NO indicators.

The book is observed on 100% of bars, zero-trade ones included: on 6 symbols x
20 days of the 5s mirror, ``bids_0_price``/``asks_0_price`` are NaN on 0.0000%
of zero-trade bars. So ``ta_price_source="book_mid"`` prices the TA indicators
off ``(bids_0_price + asks_0_price) / 2`` and fabricates nothing.

Two things this file exists to pin, because both are invisible in the output:

  1. The book mid is the **book_snapshot_25 level-0** mid,
     ``(bids_0_price + asks_0_price) / 2`` — NOT the **book_ticker L1** mid
     ``(bid_price + ask_price) / 2`` that ``features.py`` already publishes as
     ``mid_price``. The two are usually equal (they agree on ~97.5-98.7% of
     bars) and NOT always, so a swap survives every smoke test.
  2. The TARGET's price source is untouched by this flag. The target rides on
     ``mid_price``; only the FEATURES move.

Every test here has a paired mutant in ``run_ta_price_source_mutants.sh``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mjolnir.core.aligner import StreamAligner
from mjolnir.core.features import MjolnirFeatures, book_mid_price
from mjolnir.core.research import resolve_ta_price_source

talib = pytest.importorskip("talib")

DATE = "20240101"
BASE_TS = pd.Timestamp("2024-01-01", tz="UTC")
BAR_FREQ = "5s"
DAY_S = 24 * 3600

# Same 21 columns the zero-trade pass measured as NaN-propagating.
POISONED_21 = [
    "rsi", "rsi_7", "rsi_28", "macd", "macdhist", "macd_norm", "macdhist_norm",
    "atr", "natr", "adx", "dx", "plus_di", "minus_di",
    "bb_upper", "bb_lower", "bb_width", "bb_pctb",
    "cmo", "std", "stoch_k", "stoch_d",
]

# An interior trade gap: bars in [GAP_START_S, GAP_END_S) have no trade, so
# under fill_zero_trade=False their close is NaN. TA-Lib skips a LEADING NaN
# run (the numpy wrapper starts at the first finite value) but propagates an
# INTERIOR one to the end of the series, which is the failure mode this whole
# change exists for. The gap is 20.8% of the day so a test cannot pass by
# hiding under a NaN-rate tolerance.
GAP_START_S = 3600
GAP_END_S = 21600


def _paths(n: int) -> tuple:
    """Two DIFFERENT price paths: book_ticker L1 and book_snapshot_25 L0.

    They must not coincide, or a test that reads the wrong one still passes.
    """
    rng = np.random.default_rng(11)
    ticker = 50_000 + np.cumsum(rng.normal(0, 3.0, n))
    # A distinct, always-nonzero offset — the snapshot is a different stream
    # sampled at a different cadence, not a copy of the ticker.
    snapshot = ticker + 7.5 + rng.normal(0, 1.5, n)
    return ticker, snapshot


def _streams() -> dict:
    secs = np.arange(0, DAY_S, 5)
    ts = pd.DatetimeIndex([BASE_TS + pd.Timedelta(seconds=int(s)) for s in secs])
    ticker, snapshot = _paths(len(secs))

    trade_mask = ~((secs >= GAP_START_S) & (secs < GAP_END_S))
    t_ts, t_px = ts[trade_mask], ticker[trade_mask]
    rng = np.random.default_rng(3)
    trades = pd.DataFrame({
        "timestamp": t_ts,
        "side": np.where(rng.random(len(t_px)) > 0.5, "buy", "sell"),
        "price": t_px,
        "amount": rng.uniform(0.01, 1.0, len(t_px)),
    })

    book_ticker = pd.DataFrame({
        "timestamp": ts,
        "bid_price": ticker - 0.5,
        "bid_amount": np.full(len(ts), 1.0),
        "ask_price": ticker + 0.5,
        "ask_amount": np.full(len(ts), 2.0),
    })

    snap_cols = {"timestamp": ts}
    for lvl in range(5):
        snap_cols[f"bids[{lvl}].price"] = snapshot - 0.5 - lvl
        snap_cols[f"bids[{lvl}].amount"] = np.full(len(ts), 10.0 + lvl)
        snap_cols[f"asks[{lvl}].price"] = snapshot + 0.5 + lvl
        snap_cols[f"asks[{lvl}].amount"] = np.full(len(ts), 20.0 + lvl)
    snapshot_df = pd.DataFrame(snap_cols)

    derivative = pd.DataFrame({
        "timestamp": ts,
        "mark_price": ticker + 0.1,
        "index_price": ticker,
        "open_interest": np.full(len(ts), 1_000.0),
        "funding_rate": np.full(len(ts), 0.0001),
    })

    return {
        "trades": trades,
        "book_ticker": book_ticker,
        "book_snapshot_25": snapshot_df,
        "derivative_ticker": derivative,
    }


def _align(fill_zero_trade: bool) -> pd.DataFrame:
    aligner = StreamAligner(bar_freq=BAR_FREQ, fill_zero_trade=fill_zero_trade)
    return aligner.align(_streams(), DATE)


def _engine(ta_price_source: str, zero_fill_prices: bool) -> MjolnirFeatures:
    return MjolnirFeatures(
        feature_windows=[30], target_horizon=1, fee_rate=0.0,
        bar_tf=BAR_FREQ, target_tf=BAR_FREQ,
        zero_fill_prices=zero_fill_prices,
        ta_price_source=ta_price_source,
    )


def _features(ta_price_source: str, fill_zero_trade: bool,
              zero_fill_prices: bool = None) -> pd.DataFrame:
    """`zero_fill_prices` defaults to the matching half of the ONE switch."""
    bars = _align(fill_zero_trade)
    if zero_fill_prices is None:
        zero_fill_prices = fill_zero_trade
    eng = _engine(ta_price_source, zero_fill_prices=zero_fill_prices)
    return eng.compute(bars)


def _talib_ref(series: pd.Series, fn, drop_invalid: bool) -> pd.Series:
    """Reference indicator, computed independently of features.py."""
    if drop_invalid:
        keep = series.notna()
        vals = fn(series[keep].to_numpy(dtype=float))
        return pd.Series(vals, index=series.index[keep]).reindex(series.index)
    return pd.Series(fn(series.to_numpy(dtype=float)), index=series.index)


# ---------------------------------------------------------------------------
# 1. The flag itself
# ---------------------------------------------------------------------------

class TestFlag:
    def test_ta_price_source_has_no_default(self):
        """A caller that does not state its choice must fail, not guess."""
        with pytest.raises(TypeError):
            MjolnirFeatures(feature_windows=[30], target_horizon=1,
                            fee_rate=0.0, bar_tf=BAR_FREQ, target_tf=BAR_FREQ)

    def test_unknown_source_is_rejected(self):
        with pytest.raises(ValueError, match="TA_PRICE_SOURCE|ta_price_source"):
            _engine("microprice", zero_fill_prices=True)

    def test_both_documented_values_construct(self):
        for src in ("close", "book_mid"):
            assert _engine(src, zero_fill_prices=True).ta_price_source == src


# ---------------------------------------------------------------------------
# 2. Identity at "close"
# ---------------------------------------------------------------------------

class TestCloseIsUnchanged:
    """`close` must reproduce today's behaviour exactly, NaN poisoning and all.

    The references below are transcribed from the TA-Lib calls directly, so an
    identity claim is not the new code compared against itself.
    """

    def test_close_ta_is_talib_on_the_raw_close(self):
        bars = _align(True)
        # zero_fill_prices=False so the TA-Lib warmup NaN is not blanket-filled
        # to 0.0 before the comparison — the fill is a separate switch.
        feats = _features("close", fill_zero_trade=True, zero_fill_prices=False)
        close = bars["close"]
        for col, fn in (("rsi", lambda a: talib.RSI(a, timeperiod=14)),
                        ("cmo", lambda a: talib.CMO(a, timeperiod=14)),
                        ("std", lambda a: talib.STDDEV(a, timeperiod=14))):
            want = _talib_ref(close, fn, drop_invalid=False)
            pd.testing.assert_series_equal(feats[col], want, check_names=False)

    def test_close_does_not_drop_nan_rows(self):
        """No compaction on the legacy path — NaN propagation is preserved."""
        feats = _features("close", fill_zero_trade=False)
        bars = _align(False)
        want = _talib_ref(bars["close"], lambda a: talib.RSI(a, timeperiod=14),
                          drop_invalid=False)
        pd.testing.assert_series_equal(feats["rsi"], want, check_names=False)


# ---------------------------------------------------------------------------
# 3. The book mid is the SNAPSHOT L0 mid, not the book_ticker mid
# ---------------------------------------------------------------------------

class TestBookMidSource:
    def test_helper_is_snapshot_level0(self):
        bars = _align(False)
        got = book_mid_price(bars)
        want = (bars["bids_0_price"] + bars["asks_0_price"]) / 2
        pd.testing.assert_series_equal(got, want, check_names=False)
        # ... and is NOT the book_ticker mid the frame also carries.
        ticker_mid = (bars["bid_price"] + bars["ask_price"]) / 2
        assert not np.allclose(got.to_numpy(dtype=float),
                               ticker_mid.to_numpy(dtype=float))

    def test_ta_reads_snapshot_level0_and_not_book_ticker(self):
        """The mutant that swaps the source must fail here."""
        bars = _align(False)
        feats = _features("book_mid", fill_zero_trade=False)
        # Point-in-time book columns are shifted one bar in compute(), so the
        # TA input is the PRIOR bar's book — causal, and NaN on row 0.
        snap_mid = ((bars["bids_0_price"] + bars["asks_0_price"]) / 2).shift(1)
        ticker_mid = ((bars["bid_price"] + bars["ask_price"]) / 2).shift(1)

        rsi = lambda a: talib.RSI(a, timeperiod=14)     # noqa: E731
        want = _talib_ref(snap_mid, rsi, drop_invalid=True)
        wrong = _talib_ref(ticker_mid, rsi, drop_invalid=True)
        pd.testing.assert_series_equal(feats["rsi"], want, check_names=False)
        assert not np.allclose(feats["rsi"].to_numpy(dtype=float),
                               wrong.to_numpy(dtype=float), equal_nan=True)

    def test_high_low_are_the_same_mid(self):
        """No intrabar high/low of the book exists — h == lo == c, stated."""
        bars = _align(False)
        feats = _features("book_mid", fill_zero_trade=False)
        snap_mid = ((bars["bids_0_price"] + bars["asks_0_price"]) / 2).shift(1)
        keep = snap_mid.notna()
        arr = snap_mid[keep].to_numpy(dtype=float)
        want = pd.Series(talib.ATR(arr, arr, arr, timeperiod=14),
                         index=snap_mid.index[keep]).reindex(bars.index)
        pd.testing.assert_series_equal(feats["atr"], want, check_names=False)


# ---------------------------------------------------------------------------
# 4. Missing book columns RAISE — never fill
# ---------------------------------------------------------------------------

class TestMissingBookColumnsRaise:
    """10 of 13,800 day-files in the 5s mirror carry no bids_*/asks_* at all."""

    @pytest.mark.parametrize("drop", ["bids_0_price", "asks_0_price",
                                      ["bids_0_price", "asks_0_price"]])
    def test_missing_column_raises(self, drop):
        bars = _align(False).drop(columns=drop)
        # The MESSAGE is load-bearing: a bare pandas KeyError would also name
        # the column, so match the explanation that forbids the fallback.
        with pytest.raises(KeyError, match="cannot price TA indicators"):
            _engine("book_mid", zero_fill_prices=False).compute(bars)

    def test_missing_column_is_fine_for_close(self):
        """The legacy source must not acquire a new dependency."""
        bars = _align(True).drop(columns=["bids_0_price", "asks_0_price"])
        out = _engine("close", zero_fill_prices=True).compute(bars)
        assert out["rsi"].notna().any()


# ---------------------------------------------------------------------------
# 5. The 21 columns: dense at book_mid, poisoned at close
# ---------------------------------------------------------------------------

def _nan_rate(feats: pd.DataFrame, col: str) -> float:
    return float(feats[col].isna().mean())


class TestDensity:
    def test_close_plus_no_fill_poisons_the_21(self):
        """The measured baseline this change exists to remove."""
        feats = _features("close", fill_zero_trade=False)
        bad = [c for c in POISONED_21 if _nan_rate(feats, c) < 0.5]
        assert not bad, f"expected NaN-poisoned at close+no-fill, dense: {bad}"

    def test_book_mid_makes_the_21_dense(self):
        feats = _features("book_mid", fill_zero_trade=False)
        # 1 shift row + the longest TA warmup (rsi_28 / ADX 2x14) is well under
        # 1% of a 17,280-bar day.
        bad = {c: _nan_rate(feats, c) for c in POISONED_21
               if _nan_rate(feats, c) > 0.01}
        assert not bad, f"still sparse under book_mid: {bad}"

    def test_no_fabricated_zero_prices_anywhere(self):
        feats = _features("book_mid", fill_zero_trade=False)
        for col in ("close", "open", "high", "low", "vwap", "mid_price",
                    "bid_price", "ask_price", "bids_0_price", "asks_0_price",
                    "bb_upper", "bb_lower"):
            if col not in feats.columns:
                continue
            zeros = int((feats[col] == 0.0).sum())
            assert zeros == 0, f"{col} has {zeros} fabricated 0.0 values"


# ---------------------------------------------------------------------------
# 6. The TARGET is untouched
# ---------------------------------------------------------------------------

class TestTargetUnchanged:
    TARGET_COLS = ("return", "return_long", "return_short",
                   "return_long_raw", "return_short_raw")

    def test_target_is_identical_across_sources(self):
        """The mutant that repoints the target at the book mid must fail."""
        a = _features("close", fill_zero_trade=False)
        b = _features("book_mid", fill_zero_trade=False)
        for col in self.TARGET_COLS:
            if col not in a.columns:
                continue
            pd.testing.assert_series_equal(a[col], b[col], check_names=False)

    def test_target_still_rides_on_the_book_ticker_mid(self):
        bars = _align(False)
        feats = _features("book_mid", fill_zero_trade=False)
        ticker_mid = (bars["bid_price"] + bars["ask_price"]) / 2
        want = ticker_mid.shift(-1) / ticker_mid - 1.0
        got = feats["return"]
        both = got.notna() & want.notna()
        assert both.sum() > 1000
        np.testing.assert_allclose(got[both].to_numpy(dtype=float),
                                   want[both].to_numpy(dtype=float),
                                   rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. The setting.json resolver
# ---------------------------------------------------------------------------

class TestResolver:
    """`TA_PRICE_SOURCE` is required in setting.json, exactly like
    `FILL_ZERO_TRADE`: a guess would silently decide which price a whole
    corpus of features was built on."""

    def test_missing_key_raises(self):
        # "no default" pins the explanation, not just a bare dict KeyError.
        with pytest.raises(KeyError, match="no default"):
            resolve_ta_price_source({"FILL_ZERO_TRADE": True})

    @pytest.mark.parametrize("bad", ["", "mid_price", "CLOSE", True, 1, None])
    def test_invalid_value_raises(self, bad):
        with pytest.raises(ValueError, match="TA_PRICE_SOURCE"):
            resolve_ta_price_source({"TA_PRICE_SOURCE": bad})

    @pytest.mark.parametrize("good", ["close", "book_mid"])
    def test_valid_value_passes_through(self, good):
        assert resolve_ta_price_source({"TA_PRICE_SOURCE": good}) == good


# ---------------------------------------------------------------------------
# 8. Gaps in the BOOK itself
# ---------------------------------------------------------------------------

class TestBookGaps:
    """The snapshot stream is not infallible either.

    Per day-file the aligner ffills the snapshot, so the only NaN it can leave
    is at the START of a day — before that day's first book_snapshot_25 tick.
    Research CONCATENATES day-files per symbol, which turns those leading runs
    into INTERIOR NaN runs (measured 0.0035% of bars on XMRUSDT, 0.0119% on
    TAOUSDT over 20 days). TA-Lib propagates an interior NaN forever, so those
    rows are excluded from the TA input rather than fed to it — and they must
    come back as NaN, never as a filled value.
    """

    GAP = slice(5_000, 5_060)

    def _bars_with_a_book_gap(self) -> pd.DataFrame:
        bars = _align(False)
        rows = bars.index[self.GAP]
        bars.loc[rows, ["bids_0_price", "asks_0_price"]] = np.nan
        return bars, rows

    def test_interior_book_gap_does_not_poison_the_tail(self):
        bars, rows = self._bars_with_a_book_gap()
        feats = _engine("book_mid", zero_fill_prices=False).compute(bars)
        # The point-in-time shift moves the hole one bar later.
        after = feats.index > rows[-1] + pd.Timedelta(BAR_FREQ)
        for col in POISONED_21:
            rate = float(feats.loc[after, col].isna().mean())
            assert rate == 0.0, f"{col} still {rate:.4%} NaN after the gap"

    def test_rows_without_a_book_stay_nan(self):
        """Excluded, not filled: a dropped row must not come back with a value."""
        bars, rows = self._bars_with_a_book_gap()
        feats = _engine("book_mid", zero_fill_prices=False).compute(bars)
        holed = rows + pd.Timedelta(BAR_FREQ)
        for col in ("rsi", "atr", "std", "bb_upper", "macd"):
            assert feats.loc[holed, col].isna().all(), \
                f"{col} was filled on rows with no observed book"
