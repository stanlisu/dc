"""Zero-trade bars: the builder flag, the load guard, and the fill split.

"No trade in 5s" is information. The historical convention erased it by
carrying the previous bar's price across a zero-trade bar
(``StreamAligner._agg_trades``), and then re-applying the identical fill again
at load time (``MjolnirResearch.load``).

Removing only that fill is WORSE than leaving it: a single interior NaN
propagates to the end of every TA-Lib series (RSI/MACD/ATR/BBANDS/ADX/STDDEV),
and the blanket ``fillna(0.0)`` in ``MjolnirFeatures.compute`` then pins 21
columns to exactly 0.0 for the remainder of the day. So the change has three
inseparable halves, and this file locks all three:

  1. ``StreamAligner(fill_zero_trade=...)``   — the fill becomes a choice, and
     ``has_trade`` records the truth in BOTH modes.
  2. ``MjolnirResearch.load``                 — must not silently re-fill, and
     must refuse bars whose build convention disagrees with the tree's.
  3. ``MjolnirFeatures(zero_fill_prices=...)`` — 0.0 stays valid for flow, and
     stops being applied to prices and price-derived indicators.

Every test here has a paired mutant in ``run_zero_trade_mutants.sh``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mjolnir.core.aligner import StreamAligner
from mjolnir.core.features import MjolnirFeatures
from mjolnir.core.research import (
    _apply_zero_trade_policy,
    resolve_fill_zero_trade,
)
from mjolnir.core.utils import bars_per_day

DATE = "20240101"
BASE_TS = pd.Timestamp("2024-01-01", tz="UTC")
BAR_FREQ = "5s"

# The 21 columns the architecture pass measured as pinned to exactly 0.0 for
# the rest of the day by ONE interior NaN + the blanket fillna(0.0).
POISONED_21 = [
    "rsi", "rsi_7", "rsi_28", "macd", "macdhist", "macd_norm", "macdhist_norm",
    "atr", "natr", "adx", "dx", "plus_di", "minus_di",
    "bb_upper", "bb_lower", "bb_width", "bb_pctb",
    "cmo", "std", "stoch_k", "stoch_d",
]


# ---------------------------------------------------------------------------
# Fixtures — raw streams with a deliberate trade GAP
# ---------------------------------------------------------------------------

# Trades cover 00:00:00 -> 00:04:59 except the gap below, so the day's
# remaining bars are all zero-trade too. GAP_BARS are interior zero-trade bars,
# which is the case that poisons TA-Lib.
GAP_START_S = 60      # 00:01:00
GAP_END_S = 120       # 00:02:00 (exclusive)
TRADE_SPAN_S = 300    # trades exist in [0, 300) minus the gap


def _make_trades() -> pd.DataFrame:
    secs = [s for s in range(TRADE_SPAN_S) if not (GAP_START_S <= s < GAP_END_S)]
    ts = pd.DatetimeIndex([BASE_TS + pd.Timedelta(seconds=s) for s in secs])
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "timestamp": ts,
        "side": np.where(rng.random(len(secs)) > 0.5, "buy", "sell"),
        "price": 50_000 + rng.normal(0, 25, len(secs)),
        "amount": rng.uniform(0.01, 1.0, len(secs)),
    })


def _make_book_ticker() -> pd.DataFrame:
    """Quotes only in the FIRST bar — everything after must come from ffill."""
    ts = pd.DatetimeIndex([BASE_TS + pd.Timedelta(seconds=s) for s in (0, 1)])
    return pd.DataFrame({
        "timestamp": ts,
        "bid_price": [49_999.0, 49_998.0],
        "bid_amount": [1.0, 2.0],
        "ask_price": [50_001.0, 50_002.0],
        "ask_amount": [1.0, 2.0],
    })


def _make_snapshot() -> pd.DataFrame:
    ts = pd.DatetimeIndex([BASE_TS + pd.Timedelta(seconds=s) for s in (0, 1)])
    cols = {"timestamp": ts}
    for lvl in range(5):
        cols[f"bids[{lvl}].price"] = [49_999.0 - lvl, 49_998.0 - lvl]
        cols[f"bids[{lvl}].amount"] = [10.0 + lvl, 11.0 + lvl]
        cols[f"asks[{lvl}].price"] = [50_001.0 + lvl, 50_002.0 + lvl]
        cols[f"asks[{lvl}].amount"] = [20.0 + lvl, 21.0 + lvl]
    return pd.DataFrame(cols)


def _make_derivative() -> pd.DataFrame:
    ts = pd.DatetimeIndex([BASE_TS + pd.Timedelta(seconds=s) for s in (0, 1)])
    return pd.DataFrame({
        "timestamp": ts,
        "mark_price": [50_000.5, 50_000.7],
        "index_price": [50_000.0, 50_000.2],
        "open_interest": [1_000.0, 1_001.0],
        "funding_rate": [0.0001, 0.0002],
    })


def _all_streams() -> dict:
    return {
        "trades": _make_trades(),
        "book_ticker": _make_book_ticker(),
        "book_snapshot_25": _make_snapshot(),
        "derivative_ticker": _make_derivative(),
    }


def _align(fill_zero_trade: bool, streams=None) -> pd.DataFrame:
    aligner = StreamAligner(bar_freq=BAR_FREQ, fill_zero_trade=fill_zero_trade)
    return aligner.align(streams if streams is not None else _all_streams(), DATE)


def _legacy_agg_trades_reference(streams: dict) -> pd.DataFrame:
    """The pre-change algorithm, transcribed verbatim from git history.

    An identity test against the NEW code in its ``True`` mode is only
    meaningful if the thing it is compared to is not itself the new code, so
    this is an independent transcription of the old ``_agg_trades`` tail.
    """
    df = streams["trades"].copy()
    bar_index = pd.date_range(start=BASE_TS, periods=bars_per_day(BAR_FREQ),
                              freq=BAR_FREQ)
    df["bar"] = df["timestamp"].dt.floor(BAR_FREQ)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    is_buy = df["side"].str.lower().isin(["buy", "b"])

    agg = df.groupby("bar").agg(
        open=("price", "first"), high=("price", "max"),
        low=("price", "min"), close=("price", "last"),
        volume=("amount", "sum"), n_trades=("price", "count"),
    )
    agg["buy_vol"] = df[is_buy].groupby("bar")["amount"].sum()
    agg["sell_vol"] = df[~is_buy].groupby("bar")["amount"].sum()
    df["pv"] = df["price"] * df["amount"]
    agg["vwap"] = df.groupby("bar")["pv"].sum() / (agg["volume"] + 1e-10)
    agg["trade_imbalance"] = (
        (agg["buy_vol"].fillna(0) - agg["sell_vol"].fillna(0))
        / (agg["volume"].fillna(0) + 1e-10)
    )
    agg = agg.reindex(bar_index)

    agg["close"] = agg["close"].ffill()
    no_trade = agg["n_trades"].isna()
    agg.loc[no_trade, "open"] = agg.loc[no_trade, "close"]
    agg.loc[no_trade, "high"] = agg.loc[no_trade, "close"]
    agg.loc[no_trade, "low"] = agg.loc[no_trade, "close"]
    agg.loc[no_trade, "vwap"] = agg.loc[no_trade, "close"]
    for col in ["volume", "n_trades", "buy_vol", "sell_vol"]:
        agg[col] = agg[col].fillna(0.0)
    agg["trade_imbalance"] = agg["trade_imbalance"].fillna(0.0)
    return agg


# ---------------------------------------------------------------------------
# 1. The aligner flag
# ---------------------------------------------------------------------------

class TestAlignerFlag:
    def test_fill_zero_trade_has_no_default(self):
        """A caller that does not state its choice must fail, not guess."""
        with pytest.raises(TypeError):
            StreamAligner(bar_freq=BAR_FREQ)      # noqa: F821 - deliberate

    def test_true_reproduces_legacy_columns_exactly(self):
        """Identity: every legacy column is bit-for-bit the old algorithm."""
        streams = _all_streams()
        got = _align(True, streams)
        want = _legacy_agg_trades_reference(streams)
        assert len(got) == bars_per_day(BAR_FREQ)
        for col in want.columns:
            pd.testing.assert_series_equal(
                got[col], want[col], check_names=False,
                obj=f"legacy column {col!r} under fill_zero_trade=True",
            )

    def test_false_leaves_ohlc_nan_on_exactly_the_zero_trade_rows(self):
        bars = _align(False)
        no_trade = bars["n_trades"] == 0
        assert no_trade.sum() > 0
        assert (~no_trade).sum() > 0
        for col in ("open", "high", "low", "close", "vwap"):
            # NaN on every zero-trade row ...
            assert bars.loc[no_trade, col].isna().all(), col
            # ... and NOWHERE else.
            assert bars.loc[~no_trade, col].notna().all(), col

    def test_false_still_zero_fills_the_flow_columns(self):
        """0.0 is the TRUE value for "no trades", so flows are not NaN."""
        bars = _align(False)
        no_trade = bars["n_trades"] == 0
        for col in ("volume", "n_trades", "buy_vol", "sell_vol",
                    "trade_imbalance"):
            assert bars[col].notna().all(), col
            assert (bars.loc[no_trade, col] == 0.0).all(), col

    def test_true_fills_the_interior_gap(self):
        """The mirror image of the test above — the fill really does happen."""
        bars = _align(True)
        gap = bars.index[(bars.index >= BASE_TS + pd.Timedelta(seconds=GAP_START_S))
                         & (bars.index < BASE_TS + pd.Timedelta(seconds=GAP_END_S))]
        assert len(gap) == (GAP_END_S - GAP_START_S) // 5
        assert bars.loc[gap, "close"].notna().all()
        assert (bars.loc[gap, "n_trades"] == 0).all()


class TestHasTrade:
    @pytest.mark.parametrize("fill", [True, False])
    def test_present_and_equals_n_trades_gt_zero(self, fill):
        bars = _align(fill)
        assert "has_trade" in bars.columns
        assert bars["has_trade"].dtype == bool
        pd.testing.assert_series_equal(
            bars["has_trade"], bars["n_trades"] > 0, check_names=False,
        )

    @pytest.mark.parametrize("fill", [True, False])
    def test_is_not_constant(self, fill):
        """A column that is all-True proves nothing about either mode."""
        bars = _align(fill)
        assert bars["has_trade"].any()
        assert not bars["has_trade"].all()


class TestStateFfillsAreUntouched:
    """A quote is STATE and persists; a trade is an EVENT and does not.

    book_ticker / book_snapshot_25 / derivative_ticker are ffilled
    unconditionally, and must keep being ffilled in BOTH modes. The fixtures
    supply those streams only in the first bar, so every later row can ONLY be
    non-NaN because the ffill ran.
    """

    LATE = BASE_TS + pd.Timedelta(seconds=3600)

    @pytest.mark.parametrize("fill", [True, False])
    def test_book_ticker_still_ffills(self, fill):
        bars = _align(fill)
        assert bars.loc[self.LATE, "bid_price"] == 49_998.0
        assert bars.loc[self.LATE, "ask_price"] == 50_002.0
        assert bars["bid_price"].notna().all()

    @pytest.mark.parametrize("fill", [True, False])
    def test_book_snapshot_still_ffills(self, fill):
        bars = _align(fill)
        assert bars.loc[self.LATE, "bids_0_price"] == 49_998.0
        assert bars.loc[self.LATE, "asks_0_price"] == 50_002.0
        assert bars["depth_bid_L5"].notna().all()

    @pytest.mark.parametrize("fill", [True, False])
    def test_derivative_ticker_still_ffills(self, fill):
        bars = _align(fill)
        assert bars.loc[self.LATE, "mark_price"] == 50_000.7
        assert bars.loc[self.LATE, "funding_rate"] == 0.0002
        assert bars["open_interest"].notna().all()


# ---------------------------------------------------------------------------
# 2. The load-time guard
# ---------------------------------------------------------------------------

class TestResolveFillZeroTrade:
    def test_missing_key_raises(self):
        with pytest.raises(KeyError, match="FILL_ZERO_TRADE"):
            resolve_fill_zero_trade({"TIME_UNIT": "5s"})

    @pytest.mark.parametrize("val", ["false", "true", 0, 1, None])
    def test_non_boolean_raises(self, val):
        with pytest.raises(TypeError, match="FILL_ZERO_TRADE"):
            resolve_fill_zero_trade({"FILL_ZERO_TRADE": val})

    @pytest.mark.parametrize("val", [True, False])
    def test_boolean_round_trips(self, val):
        assert resolve_fill_zero_trade({"FILL_ZERO_TRADE": val}) is val


class TestLoadTimePolicy:
    """`load()` re-applied the identical ffill. Without this, a whole no-fill
    rebuild would be invisible: the bars on disk would be right and every
    downstream feature identical to before."""

    def test_no_fill_tree_does_not_refill_no_fill_bars(self):
        bars = _align(False)
        out = _apply_zero_trade_policy(bars.copy(), "t", fill_zero_trade=False)
        no_trade = out["n_trades"] == 0
        assert out.loc[no_trade, "close"].isna().all()
        assert out.loc[no_trade, "vwap"].isna().all()

    def test_fill_tree_still_refills(self):
        bars = _align(True)
        # Blank an interior price the way a mid-corpus daily boundary can.
        bars.iloc[100, bars.columns.get_loc("close")] = np.nan
        out = _apply_zero_trade_policy(bars, "t", fill_zero_trade=True)
        assert out["close"].iloc[100] == out["close"].iloc[99]

    def test_no_fill_tree_rejects_bars_built_with_the_fill(self):
        """The stale-corpus case: FILL_ZERO_TRADE=false but filled bars."""
        bars = _align(True)
        with pytest.raises(RuntimeError, match="already carry a close price"):
            _apply_zero_trade_policy(bars, "stale", fill_zero_trade=False)

    def test_no_fill_tree_rejects_bars_predating_has_trade(self):
        bars = _align(False).drop(columns=["has_trade"])
        with pytest.raises(RuntimeError, match="has_trade"):
            _apply_zero_trade_policy(bars, "legacy", fill_zero_trade=False)

    def test_no_fill_tree_rejects_nan_price_on_a_traded_bar(self):
        """Damage the legacy loader would have papered over."""
        bars = _align(False)
        traded = bars.index[bars["has_trade"]][5]
        bars.loc[traded, "close"] = np.nan
        with pytest.raises(RuntimeError, match="has_trade=True"):
            _apply_zero_trade_policy(bars, "damaged", fill_zero_trade=False)

    def test_flow_columns_are_zero_filled_in_both_modes(self):
        for fill in (True, False):
            bars = _align(fill)
            bars.loc[bars.index[10], "volume"] = np.nan
            out = _apply_zero_trade_policy(bars, "t", fill_zero_trade=fill)
            assert out["volume"].notna().all()


# ---------------------------------------------------------------------------
# 3. The features fill split
# ---------------------------------------------------------------------------

def _features(zero_fill_prices: bool, fill_zero_trade: bool) -> pd.DataFrame:
    bars = _align(fill_zero_trade)
    eng = MjolnirFeatures(
        feature_windows=[30], target_horizon=1,
        fee_rate=0.0, bar_tf=BAR_FREQ, target_tf=BAR_FREQ,
        zero_fill_prices=zero_fill_prices,
    )
    return eng.compute(bars)


class TestFeatureFillSplit:
    PRICE_COLS = ("close", "open", "high", "low", "vwap", "mid_price",
                  "bid_price", "ask_price", "mark_price", "index_price")

    def test_price_columns_are_never_zero_filled_when_disabled(self):
        feats = _features(zero_fill_prices=False, fill_zero_trade=False)
        for col in self.PRICE_COLS:
            if col not in feats.columns:
                continue
            zeros = (feats[col] == 0.0).sum()
            assert zeros == 0, f"{col} has {zeros} fabricated 0.0 prices"

    def test_flow_columns_are_still_zero_filled_when_disabled(self):
        feats = _features(zero_fill_prices=False, fill_zero_trade=False)
        for col in ("trade_imbalance", "volume", "n_trades", "ofi_agg"):
            assert col in feats.columns, col
            assert feats[col].notna().all(), f"{col} left NaN — flow must fill"
        no_trade = feats["n_trades"] == 0
        assert no_trade.sum() > 0
        assert (feats.loc[no_trade, "trade_imbalance"] == 0.0).all()

    def test_enabled_is_the_legacy_blanket_fill(self):
        """True keeps today's behaviour: no non-target NaN survives."""
        feats = _features(zero_fill_prices=True, fill_zero_trade=True)
        meta = MjolnirFeatures._META_COLS
        cols = [c for c in feats.columns if c not in meta]
        assert not feats[cols].isna().any().any()

    def test_default_is_the_legacy_blanket_fill(self):
        """The LIVE default must be today's behaviour (knull/mjolnir_bridge)."""
        eng = MjolnirFeatures(feature_windows=[30], target_horizon=1,
                              fee_rate=0.0, bar_tf=BAR_FREQ, target_tf=BAR_FREQ)
        assert eng.zero_fill_prices is True


class TestTwentyOneColumnPoisoning:
    """The measured regression: ONE interior zero-trade bar + the blanket
    fillna(0.0) pins 21 TA columns to exactly 0.0 for the rest of the series.

    Reproduced at zero_fill_prices=True, absent at False. This is the whole
    reason the aligner change alone would have been a regression.
    """

    @staticmethod
    def _pinned(feats: pd.DataFrame) -> list:
        """Columns whose tail is a run of exactly 0.0 to the last row."""
        out = []
        for col in POISONED_21:
            if col not in feats.columns:
                continue
            s = feats[col]
            # Exclude the target-tail rows the pipeline drops anyway.
            tail = s.iloc[-1000:]
            if (tail == 0.0).all():
                out.append(col)
        return out

    def test_present_columns_cover_the_measured_set(self):
        feats = _features(zero_fill_prices=True, fill_zero_trade=False)
        present = [c for c in POISONED_21 if c in feats.columns]
        assert len(present) >= 15, (
            "the poisoning fixture must exercise most of the measured 21; "
            f"only {present} exist in this build"
        )

    def test_poisoning_reproduced_at_true(self):
        feats = _features(zero_fill_prices=True, fill_zero_trade=False)
        pinned = self._pinned(feats)
        assert pinned, (
            "expected the blanket fillna(0.0) to pin TA columns to 0.0 on "
            "no-fill bars; if this is empty the fixture no longer produces an "
            "interior NaN and the regression test below proves nothing"
        )

    def test_poisoning_absent_at_false(self):
        feats = _features(zero_fill_prices=False, fill_zero_trade=False)
        pinned = self._pinned(feats)
        assert pinned == [], f"still pinned to 0.0: {pinned}"

    def test_absent_means_nan_not_a_different_number(self):
        """The columns are missing DATA, and must say so rather than guess."""
        feats = _features(zero_fill_prices=False, fill_zero_trade=False)
        for col in POISONED_21:
            if col not in feats.columns:
                continue
            tail = feats[col].iloc[-1000:]
            assert tail.isna().all() or (tail != 0.0).any(), col
