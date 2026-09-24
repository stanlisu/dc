"""FILTER_PARQUET_FEATURE_TFS: opt-in TF narrowing of the written filter parquet.

A cross-TF (orb) vertical panel is mostly context — measured on
pred_orb.base.15m_1, 305 columns of which 210 (69%) are `1h_`/`4h_`/`1d_`
prefixed. The key lets an arm write only the timeframes it names.

Two properties matter and are both locked here:
  * ABSENT key  -> the written parquet is byte-for-byte what it is today.
  * PRESENT key -> only `<tf>_`-prefixed columns of other timeframes go; every
    target/metadata column and every BARE column survives, and the frame
    filter_signals RETURNS is never narrowed.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "agamotto_pkg/src")
sys.path.insert(0, ".")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from agamotto.research import (  # noqa: E402
    AgamottoResearch,
    narrow_filter_parquet_timeframes,
)

_PANEL_COLUMNS = [
    # bare (base-TF) feature + filter columns
    "rsi", "macd", "close", "mvg1", "price_range_pct", "price_range_pct_q80",
    # base TF, prefixed
    "15m_close", "15m_mvg1", "15m_rsi",
    # context TFs, prefixed
    "1h_rsi", "1h_close", "1h_price_range_pct_q90",
    "4h_rsi", "4h_close",
    "1d_rsi", "1d_close",
    # targets + metadata
    "return_long", "return_short", "return_long_raw", "return_short_raw",
    "year", "month", "symbol", "timestamp",
]

_PROTECTED = (
    "ret", "ret_raw", "position", "regime", "symbol", "timestamp",
    "year", "month", "return_long", "return_short",
    "return_long_raw", "return_short_raw",
)


def _panel(n: int = 8) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    data = {c: [float(i) for i in range(n)] for c in _PANEL_COLUMNS}
    data["symbol"] = ["BINANCE_PERP_BTC_USDT"] * n
    data["timestamp"] = idx
    data["year"] = [2025] * n
    data["month"] = [1] * n
    return pd.DataFrame(data)


class _Panel(AgamottoResearch):
    """filter_signals() over a hand-built panel, with the mask stubbed all-True.

    Only the write path is under test; building a real regime mask would drag in
    TA-Lib and a full engineer_features run for no extra coverage.
    """

    def __init__(self, config, vertical_features):
        self.config = config
        self.vertical_features = vertical_features

    def _apply_filter_mask(self, df, filter_name, position):
        return pd.Series(True, index=df.index)


def _write(tmp_path, config):
    res = _Panel(config, _panel())
    returned = res.filter_signals(
        {"regime": "rsi_oversold", "position": "long"},
        save=True,
        out_dir=str(tmp_path),
    )
    written = pd.read_parquet(tmp_path / "filter" / "filter_rsi_oversold_long.parquet")
    return returned, written


# ---------------------------------------------------------------- helper unit


def test_helper_drops_only_the_unlisted_timeframes():
    out = narrow_filter_parquet_timeframes(_panel(), ["15m"])
    assert [c for c in out.columns if c.startswith(("1h_", "4h_", "1d_"))] == []
    assert {"15m_close", "15m_mvg1", "15m_rsi"} <= set(out.columns)


def test_helper_keeps_every_bare_and_protected_column():
    out = narrow_filter_parquet_timeframes(_panel(), [])
    bare = [c for c in _PANEL_COLUMNS if not c[0].isdigit()]
    assert set(bare) <= set(out.columns)
    assert not [c for c in out.columns if c[0].isdigit()]


def test_helper_refuses_a_timeframe_the_panel_does_not_carry():
    with pytest.raises(KeyError, match=r"\['30m'\]"):
        narrow_filter_parquet_timeframes(_panel(), ["15m", "30m"])


def test_helper_refuses_a_malformed_timeframe_token():
    with pytest.raises(ValueError, match="not a timeframe token"):
        narrow_filter_parquet_timeframes(_panel(), ["15minutes"])


def test_helper_is_a_no_op_object_when_nothing_is_dropped():
    df = _panel()
    assert narrow_filter_parquet_timeframes(df, ["15m", "1h", "4h", "1d"]) is df


# ------------------------------------------------------------ through the write


def test_absent_key_writes_todays_full_panel(tmp_path):
    returned, written = _write(tmp_path, {})
    # 24 panel columns + position + regime + ret + ret_raw
    assert len(written.columns) == len(_PANEL_COLUMNS) + 4
    assert len(returned.columns) == len(written.columns)
    for tf in ("1h", "4h", "1d"):
        assert any(c.startswith(f"{tf}_") for c in written.columns)


def test_narrowed_write_drops_the_other_timeframes_only(tmp_path):
    full = _write(tmp_path / "full", {})[1]
    returned, written = _write(
        tmp_path / "narrow", {"FILTER_PARQUET_FEATURE_TFS": ["15m"]})

    dropped = set(full.columns) - set(written.columns)
    assert dropped == {c for c in full.columns if c.startswith(("1h_", "4h_", "1d_"))}
    assert dropped, "expected the context timeframes to be dropped"

    # Nothing load-bearing left with them. Target/metadata columns keep their
    # real names; feature columns are coded, so look them up through the codec.
    from agamotto._obf.codec import default as _codec

    for col in _PROTECTED + ("close", "mvg1"):
        assert col in written.columns, col
    for real in ("rsi", "price_range_pct", "price_range_pct_q80"):
        assert _codec().encode_feature(real) in written.columns, real

    # The RETURNED frame is untouched — narrowing is a write-time cut only.
    # It carries REAL (uncoded) names, so compare against the panel, not `full`.
    assert set(returned.columns) == set(_PANEL_COLUMNS) | {
        "position", "regime", "ret", "ret_raw"}


def test_tf_prefix_survives_obfuscation_so_the_cut_is_well_defined(tmp_path):
    """The written columns are coded (`1h_f0NN`), but the TF prefix is literal.

    If encode_columns ever stopped re-emitting the prefix, the narrowing would
    become a no-op on coded names — this pins the property the cut relies on.
    """
    written = _write(tmp_path, {})[1]
    coded_1h = [c for c in written.columns if c.startswith("1h_")]
    assert coded_1h
    assert "1h_rsi" not in coded_1h, "expected the base name to be coded"


def test_empty_allow_list_drops_every_timeframe(tmp_path):
    written = _write(tmp_path, {"FILTER_PARQUET_FEATURE_TFS": []})[1]
    assert not [c for c in written.columns if c[0].isdigit()]
    for col in _PROTECTED:
        assert col in written.columns, col


def test_bad_allow_list_raises_instead_of_skipping_the_local_write(tmp_path):
    """The local write path log-and-continues on exceptions; a config error
    must NOT be swallowed into a silently missing parquet."""
    with pytest.raises(KeyError):
        _write(tmp_path, {"FILTER_PARQUET_FEATURE_TFS": ["30m"]})
    assert not (tmp_path / "filter" / "filter_rsi_oversold_long.parquet").exists()
