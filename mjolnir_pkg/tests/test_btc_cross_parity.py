"""Byte-parity tests for the 2026-07-26 BTC compute-once refactor.

The legacy path recomputed the full BTC feature frame inside every non-BTC
``predict_from_inputs`` (27 redundant computes per 30s cycle, raw 1000-bar
BTC buffer pickled to every worker — 2026-07-24 latency forensics). The new
contract computes the slim BTC frame once (``compute_btc_features``) and
injects it as ``inputs["btc_feats"]``.

REQUIRED invariant: the final non-BTC feature frame is byte-identical between
the legacy composition (``add_btc_cross_features(feats, compute(btc_df))``)
and the new one (``add_btc_cross_features(feats, slim)``). Feature-frame
parity implies PredictDiag parity — the model consume is deterministic.

Fixture: real recorded live bars (xmen pred_mjolnir.base.30s_1_xmen
bars_2026-07-24.csv, first 400 bars of BTC + ETH).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mjolnir.core.features import MjolnirFeatures
from mjolnir.trading import MjolnirTrading

_FIXTURE = Path(__file__).parent / "fixtures" / "bars_btc_eth_400x30s_2026-07-24.csv"
_BTC = "BINANCE_PERP_BTC_USDT"
_ETH = "BINANCE_PERP_ETH_USDT"


def _base_config():
    return {
        "TIME_UNIT": "30s",
        "SYMBOLS": [_BTC, _ETH],
        "TARGET_HORIZON_BARS": 1,
        "FEE": 2.0,
        "OUTPUT_DIR": "/tmp/mjolnir_test",
        "MIN_SIGNAL_COUNT": 1,
        "REVERSE": 1,
        "REGIME_STACK_PATH": "/tmp/regime_stack.csv",
    }


@pytest.fixture(scope="module")
def bars():
    df = pd.read_csv(_FIXTURE)
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp_ns")
        ts = pd.to_datetime(g.pop("timestamp_ns"), unit="ns", utc=True)
        g = g.drop(columns=["symbol"])
        out[sym] = (g.to_dict(orient="records"), list(ts))
    assert set(out) == {_BTC, _ETH}
    return out


@pytest.fixture
def mj(bars):
    """Real feature engine; regime stack / models mocked empty."""
    with patch.object(MjolnirTrading, "_load_regime_stack", return_value=[]), \
         patch.object(MjolnirTrading, "_load_models"):
        inst = MjolnirTrading(config=_base_config(), home_root="/tmp")
    inst._research = MagicMock()
    for sym in (_BTC, _ETH):
        recs, tss = bars[sym]
        for bar, ts in zip(recs, tss):
            inst.add_bar(sym, bar, ts)
    return inst


def _frame_from_snap(snap):
    df = pd.DataFrame(snap["buf"])
    idx = pd.DatetimeIndex(snap["ts"])
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx
    return df


# ---------------------------------------------------------------------------
# Byte parity: legacy full-frame recompute vs slim compute-once frame
# ---------------------------------------------------------------------------


def test_final_feature_frame_byte_identical(mj):
    eng = mj._feat_engine
    btc_snap = mj.snapshot_inputs(_BTC)
    eth_snap = mj.snapshot_inputs(_ETH)
    assert btc_snap is not None and eth_snap is not None

    # Legacy oracle: full BTC frame recomputed inside the worker.
    btc_full = eng.compute(_frame_from_snap(btc_snap))
    feats_old = eng.add_btc_cross_features(
        eng.compute(_frame_from_snap(eth_snap)), btc_full)

    # New path: slim frame computed once by compute_btc_features.
    slim = mj.compute_btc_features(btc_snap)
    feats_new = eng.add_btc_cross_features(
        eng.compute(_frame_from_snap(eth_snap)), slim)

    pd.testing.assert_frame_equal(feats_old, feats_new, check_exact=True)


def test_slim_frame_shape(mj):
    slim = mj.compute_btc_features(mj.snapshot_inputs(_BTC))
    assert list(slim.columns) == [
        c for c in MjolnirFeatures.BTC_CROSS_INPUT_COLS if c in slim.columns]
    # Every declared input column actually exists in compute() output — a
    # rename in compute() must break here, not silently drop a cross-feature.
    assert set(slim.columns) == set(MjolnirFeatures.BTC_CROSS_INPUT_COLS)
    assert isinstance(slim.index, pd.DatetimeIndex)
    assert slim.index.tz is not None


def test_cross_input_cols_match_add_btc_reader(mj):
    """add_btc_cross_features given ONLY the slim columns must produce every
    btc_* column it produces when given the full frame (no hidden reads)."""
    eng = mj._feat_engine
    btc_full = eng.compute(_frame_from_snap(mj.snapshot_inputs(_BTC)))
    eth_feats = eng.compute(_frame_from_snap(mj.snapshot_inputs(_ETH)))
    with_full = eng.add_btc_cross_features(eth_feats, btc_full)
    with_slim = eng.add_btc_cross_features(
        eth_feats, btc_full[list(MjolnirFeatures.BTC_CROSS_INPUT_COLS)])
    assert list(with_full.columns) == list(with_slim.columns)
    pd.testing.assert_frame_equal(with_full, with_slim, check_exact=True)


# ---------------------------------------------------------------------------
# Contract: loud raises on skew, None only for legit warmup
# ---------------------------------------------------------------------------


def test_snapshot_no_longer_ships_btc_buf(mj):
    snap = mj.snapshot_inputs(_ETH)
    assert "btc_buf" not in snap
    assert "btc_ts" not in snap


def test_legacy_btc_buf_key_raises(mj):
    snap = mj.snapshot_inputs(_ETH)
    snap["btc_buf"] = ["stale"]
    with pytest.raises(RuntimeError, match="btc_buf"):
        mj.predict_from_inputs(_ETH, snap)


def test_non_btc_missing_btc_feats_raises(mj):
    snap = mj.snapshot_inputs(_ETH)
    assert "btc_feats" not in snap
    with pytest.raises(KeyError, match="btc_feats"):
        mj.predict_from_inputs(_ETH, snap)


def test_btc_feats_none_is_legit_warmup_skip(mj):
    snap = mj.snapshot_inputs(_ETH)
    snap["btc_feats"] = None
    # Empty stack → result None; the point is it does NOT raise.
    assert mj.predict_from_inputs(_ETH, snap) is None


def test_return_btc_features_only_for_btc(mj):
    snap = mj.snapshot_inputs(_ETH)
    snap["btc_feats"] = None
    with pytest.raises(ValueError, match="return_btc_features"):
        mj.predict_from_inputs(_ETH, snap, return_btc_features=True)


def test_btc_predict_returns_slim_frame_matching_compute_once(mj):
    """BTC's own predict with return_btc_features=True must hand back the
    same slim frame compute_btc_features builds — never a second compute."""
    snap = mj.snapshot_inputs(_BTC)
    diag, slim = mj.predict_from_inputs(_BTC, snap, return_btc_features=True)
    assert diag is None  # empty regime stack
    pd.testing.assert_frame_equal(
        slim, mj.compute_btc_features(snap), check_exact=True)


def test_predict_recomposes_split_path(mj):
    """predict(sym) — the offline-replay path — must run the same
    compute-once composition and return without raising."""
    assert mj.predict(_ETH) is None  # empty stack; no exception = contract ok
