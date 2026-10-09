"""``_process_combined`` must know whether the frame's newest row is in flight.

It used to drop the newest row UNCONDITIONALLY ("drop the current incomplete
bar"). That is right for the REST frame — Binance serves the in-flight candle as
the last row — and for marvel's in-flight ``KlineStreamer`` buffer, which appends
the in-flight candle on purpose so its frame matches REST. It is wrong for a
buffer that holds CLOSED bars only (marvel knull ``KLINE_WAKE_ON_CLOSE``, which
wakes the cycle on Binance's ``x=true`` close frame instead of sleeping
``DELAY``): there the newest row is the just-closed bar the decision is about,
and dropping it leaves the frame one full TIME_UNIT stale.

So the caller states it. REST passes ``True``; the WS buffer path reads the
buffer's own ``last_row_in_flight``. Both shapes must hand agamotto the SAME
``limit - 1`` closed bars, or the closed-bar path silently changes features.

These tests use a local fake buffer, not marvel's ``symbiote.KlineBuffer``, so
they run in dc's own CI (``test_ws_buffer_integration.py`` skips there).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

MARVEL_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(MARVEL_ROOT, "agamotto_pkg", "src"))

from agamotto.trading import AgamottoTrading  # noqa: E402

TF = "15m"
TF_SEC = 900
NATIVES = ["BTCUSDT", "ETHUSDT"]
COLS = ["open", "high", "low", "close", "volume", "quote_volume",
        "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]


class _FakeBuffer:
    """The three members ``_fetch_and_prepare_data`` reads, nothing else."""

    def __init__(self, frames, last_row_in_flight):
        self._frames = frames
        self.last_row_in_flight = last_row_in_flight

    def is_ready(self, symbols, tf):
        return all((s, tf) in self._frames for s in symbols)

    def get_dataframe(self, symbol, tf):
        return self._frames.get((symbol, tf))


def _closed_frames(n_closed, seed=0):
    """``{(native, tf): df}`` of ``n_closed`` CLOSED bars ending at the bar
    that closed at the most recent boundary, in ``{sym}_{tf}_{col}`` layout."""
    rng = np.random.default_rng(seed)
    newest_closed = pd.Timestamp.now("UTC").tz_localize(None).floor(
        f"{TF_SEC}s") - pd.Timedelta(seconds=TF_SEC)
    idx = pd.date_range(end=newest_closed, periods=n_closed,
                        freq=pd.Timedelta(seconds=TF_SEC))
    return {
        (n, TF): pd.DataFrame(
            {f"{n}_{TF}_{c}": rng.random(n_closed) + 1.0 for c in COLS},
            index=idx)
        for n in NATIVES
    }


def _with_in_flight(frames, seed=1):
    """Append one in-flight row (the bar that opened at the boundary)."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, df in frames.items():
        nxt = df.index[-1] + pd.Timedelta(seconds=TF_SEC)
        row = pd.DataFrame({c: [rng.random() + 1.0] for c in df.columns},
                           index=[nxt])
        out[key] = pd.concat([df, row])
    return out


def _trading():
    config = {
        "TIME_UNIT": TF,
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT", "BINANCE_PERP_ETH_USDT"],
        "CAPITAL": 10000, "LEVERAGE": 5, "SIZES": [0.001, 0.01],
        "WEIGHTS_PATH": "/tmp/mock_weights", "WEIGHTS_PERIOD": "window_2026_01",
        "TRADING_MODE": "both",
        "LONG_PRED_THRESHOLD": 0.0, "SHORT_PRED_THRESHOLD": 0.0,
        "REGIME_STACK_PATH": "/tmp/fake_regime_stack.csv",
    }

    def _no_stack(self_inner):
        self_inner.regime_stack = []

    with patch.object(AgamottoTrading, "_load_regime_stack", _no_stack), \
         patch("agamotto.trading.AgamottoTrading._calculate_sizes"), \
         patch("agamotto.trading.AgamottoTrading.load_data"), \
         patch("agamotto.trading.fetch_futures_klines"), \
         patch("agamotto.trading.klines_to_dataframe"):
        return AgamottoTrading(config=config, home_root="/tmp",
                               period="window_test")


def _raw_from(buf, limit):
    inst = _trading()
    inst._kline_buffer = buf
    with patch("agamotto.trading.fetch_futures_klines") as rest, \
         patch.object(inst, "engineer_features"), \
         patch.object(inst, "verticalize"):
        inst._fetch_and_prepare_data(limit=limit)
        rest.assert_not_called()
    return inst.raw


def test_closed_only_frame_keeps_the_just_closed_bar():
    frames = _closed_frames(50)
    raw = _raw_from(_FakeBuffer(frames, last_row_in_flight=False), limit=100)
    newest_closed = frames[("BTCUSDT", TF)].index[-1]
    assert raw.index.max() == newest_closed
    assert len(raw) == 50


def test_in_flight_frame_still_drops_its_last_row():
    frames = _closed_frames(50)
    raw = _raw_from(_FakeBuffer(_with_in_flight(frames), last_row_in_flight=True),
                    limit=100)
    assert raw.index.max() == frames[("BTCUSDT", TF)].index[-1]
    assert len(raw) == 50


@pytest.mark.parametrize("n_closed", [120, 99, 40])
def test_both_shapes_hand_agamotto_the_same_closed_bars(n_closed):
    """The closed-bar path must be a pure latency change: same rows, same
    values, same count (``limit - 1`` closed bars) as the in-flight path."""
    limit = 100
    frames = _closed_frames(n_closed)
    raw_closed = _raw_from(_FakeBuffer(frames, last_row_in_flight=False), limit)
    raw_flight = _raw_from(
        _FakeBuffer(_with_in_flight(frames), last_row_in_flight=True), limit)
    # check_freq=False: the fixture's date_range index carries freq=15min and
    # the concatenated one does not. Index metadata only — every timestamp and
    # value is still compared, and real buffer/REST frames never carry a freq.
    pd.testing.assert_frame_equal(raw_closed, raw_flight, check_freq=False)
    assert len(raw_closed) == min(n_closed, limit - 1)


def test_rest_path_drops_the_in_flight_row():
    """REST always serves the in-flight candle last; behaviour unchanged."""
    inst = _trading()
    inst._kline_buffer = None
    frames = _with_in_flight(_closed_frames(30))

    def _to_df(rows, native, tf):
        return frames[(native, tf)]

    with patch("agamotto.trading.fetch_futures_klines", return_value=[[0]]), \
         patch("agamotto.trading.klines_to_dataframe", side_effect=_to_df), \
         patch.object(inst, "engineer_features"), \
         patch.object(inst, "verticalize"):
        inst._fetch_and_prepare_data(limit=100)
    assert len(inst.raw) == 30
    assert inst.raw.index.max() == frames[("BTCUSDT", TF)].index[-2]


@pytest.mark.parametrize("bad", [None, 1, "true"])
def test_non_bool_flag_raises(bad):
    with pytest.raises(TypeError, match="last_row_in_flight"):
        _raw_from(_FakeBuffer(_closed_frames(10), last_row_in_flight=bad),
                  limit=100)


def test_buffer_without_the_flag_raises():
    """A buffer that cannot say what its last row is must not be guessed at."""
    buf = _FakeBuffer(_closed_frames(10), last_row_in_flight=False)
    del buf.last_row_in_flight
    with pytest.raises(AttributeError, match="last_row_in_flight"):
        _raw_from(buf, limit=100)
