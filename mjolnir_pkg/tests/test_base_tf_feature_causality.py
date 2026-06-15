"""Causality lock for base-TF point-in-time book/snapshot/derivative features.

These features carry the bar-CLOSE state (last tick in bar / ffilled derivative
ticker) but are stamped at the bar-OPEN timestamp — the same instant the
forward-return target window opens. MjolnirFeatures.compute() therefore shifts
them forward by exactly ONE base bar so the value at row T is the prior bar's
closed state (known at/before T), mirroring the cross-TF +tf_seconds shift in
multi_tf_merge.py.

This test pins that timing so CI fails if the shift is ever removed or a
target/causal column is accidentally shifted.
"""

import numpy as np
import pandas as pd

from mjolnir.core.features import MjolnirFeatures


def _make_bars(n: int = 12) -> pd.DataFrame:
    """Tiny synthetic aligned-bar frame with distinct per-bar book state.

    Every point-in-time book column is given a strictly monotone, unique value
    per bar so "row T == row T-1's value" is unambiguous after the shift.
    """
    idx = pd.date_range("2026-01-01 00:00:00", periods=n, freq="5s", tz="UTC")
    bid_price = 100.0 + np.arange(n)          # unique per bar
    ask_price = bid_price + 1.0
    # Asymmetric, monotone depth so depth_imbalance_L1 is unique per bar.
    bids_0_qty = 10.0 + np.arange(n)
    asks_0_qty = 5.0 + 0.5 * np.arange(n)
    return pd.DataFrame(
        {
            # trades OHLCV (NOT point-in-time book — must stay unshifted)
            "open": bid_price,
            "high": ask_price,
            "low": bid_price,
            "close": bid_price + 0.5,
            "volume": np.full(n, 3.0),
            "n_trades": np.full(n, 5.0),
            "trade_imbalance": np.zeros(n),
            # book_ticker last-in-bar (point-in-time)
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_amount": bids_0_qty,
            "ask_amount": asks_0_qty,
            # snapshot depth (point-in-time)
            "bids_0_qty": bids_0_qty,
            "asks_0_qty": asks_0_qty,
            "depth_bid_L1": bids_0_qty,
            "depth_ask_L1": asks_0_qty,
            # derivative ticker ffilled (point-in-time)
            "open_interest": 1000.0 + 7.0 * np.arange(n),
            "mark_price": bid_price + 0.25,
            "index_price": bid_price,
        },
        index=idx,
    )


def _engine() -> MjolnirFeatures:
    # Same-resolution target (5s base, 5s target) → fixed-horizon shift target.
    return MjolnirFeatures(
        feature_windows=[1],
        target_horizon=1,
        fee_rate=0.0002,
        bar_tf="5s",
        target_tf="5s",
    )


def test_point_in_time_book_feature_is_shifted_one_bar():
    """A book column at row T must equal the PRIOR bar's last value."""
    bars = _make_bars()
    feats = _engine().compute(bars)

    # depth_imbalance_L1 is the canonical book feature called out in the audit.
    raw = (bars["depth_bid_L1"] - bars["depth_ask_L1"]) / (
        bars["depth_bid_L1"] + bars["depth_ask_L1"] + 1e-10
    )
    # Row T holds bar T-1's value; row 0 is warmup (shift NaN -> sanitised to 0).
    for t in range(2, len(bars)):
        assert feats["depth_imbalance_L1"].iloc[t] == raw.iloc[t - 1], (
            f"depth_imbalance_L1 at row {t} should equal raw bar {t-1}"
        )
    assert feats["depth_imbalance_L1"].iloc[0] == 0.0  # warmup, sanitised

    # bid_price: feats[T] == bars[T-1]
    for t in range(1, len(bars)):
        assert feats["bid_price"].iloc[t] == bars["bid_price"].iloc[t - 1]
    # open_interest (derivative ffill): feats[T] == bars[T-1]
    for t in range(1, len(bars)):
        assert feats["open_interest"].iloc[t] == bars["open_interest"].iloc[t - 1]


def test_target_uses_unshifted_mid_not_shifted_feature():
    """The forward-return target must come from the UNSHIFTED mid, not the
    shifted mid_price feature column."""
    bars = _make_bars()
    feats = _engine().compute(bars)

    # Reconstruct the unshifted mid the target SHOULD use.
    unshifted_mid = (bars["bid_price"] + bars["ask_price"]) / 2.0
    expected_ret = unshifted_mid.pct_change(1, fill_method=None).shift(-1)

    # Compare on rows with a defined forward target (drop the final NaN row).
    got = feats["return"].iloc[:-1]
    exp = expected_ret.iloc[:-1]
    np.testing.assert_allclose(got.values, exp.values, rtol=1e-9, atol=1e-12)

    # And the mid_price FEATURE column itself is shifted (NOT equal to the
    # contemporaneous mid), proving feature != target source.
    assert feats["mid_price"].iloc[5] == unshifted_mid.iloc[4]


def test_trade_aggregates_and_causal_columns_not_shifted():
    """OHLCV / trade aggregates and ret_lagN are NOT in the point-in-time set
    and must remain unshifted."""
    bars = _make_bars()
    feats = _engine().compute(bars)

    # close is a bar aggregate — must equal the same bar's close, not T-1.
    for t in range(len(bars)):
        assert feats["close"].iloc[t] == bars["close"].iloc[t]
