"""KLINE_PARQUET_ROOT — `AgamottoResearch.load()` reading a parquet kline tree.

Contract: the parquet tree `{KLINE_PARQUET_ROOT}/{TIME_UNIT}/{DATA}/{SYMBOL}/
*_{TIME_UNIT}.parquet` holds the SAME rows and columns as the CSV tree, so
`load()` must produce a frame IDENTICAL to the CSV path's. And unlike the CSV
reader (which logs and skips a bad file), the parquet reader never skips: a bad
file, a missing symbol or a missing root raises.
"""
import logging

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from agamotto.research import AgamottoResearch  # noqa: E402

TF = "15m"
DATA = "liquid"
COLUMNS = ["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms",
           "quote_volume", "number_of_trades", "taker_buy_base_volume",
           "taker_buy_quote_volume"]
BAR_MS = 15 * 60 * 1000
T0_MS = 1_785_542_400_000          # 2026-08-01 00:00:00 UTC


def _month(rng, start_bar, n_bars):
    """One monthly file's rows. `number_of_trades` stays int64 on purpose, so
    the float cast is exercised on both readers."""
    open_ms = T0_MS + (start_bar + np.arange(n_bars, dtype=np.int64)) * BAR_MS
    close = 100.0 + rng.normal(0, 1, n_bars).cumsum()
    return pd.DataFrame({
        "open_time_ms": open_ms.astype(np.int64),
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": rng.random(n_bars) * 10,
        "close_time_ms": (open_ms + BAR_MS - 1).astype(np.int64),
        "quote_volume": rng.random(n_bars) * 1000,
        "number_of_trades": rng.integers(1, 500, n_bars).astype(np.int64),
        "taker_buy_base_volume": rng.random(n_bars) * 5,
        "taker_buy_quote_volume": rng.random(n_bars) * 500,
    }, columns=COLUMNS)


def _symbol_months(rng, sym):
    """Two months that OVERLAP by one open_time_ms (a different row each side,
    so keep="last" is observable), plus a within-file duplicate in month 1."""
    m1 = _month(rng, 0, 6)
    m1 = pd.concat([m1, m1.iloc[[2]].assign(close=999.0)], ignore_index=True)
    m2 = _month(rng, 5, 6)            # bar 5 is in both files
    if sym == "ETHUSDT":
        m2 = m2.drop(index=3).reset_index(drop=True)   # a bar BTC has and ETH lacks
    return {"2026-08": m1, "2026-09": m2}


def _write_trees(tmp_path, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT")):
    """Write the same rows as a CSV tree (under home_root/data) and a parquet tree."""
    rng = np.random.default_rng(3)
    home = tmp_path / "home"
    pq_root = tmp_path / "pq"
    for sym in symbols:
        csv_dir = home / "data" / "BINANCEFUTURES" / TF / DATA / sym
        pq_dir = pq_root / TF / DATA / sym
        csv_dir.mkdir(parents=True)
        pq_dir.mkdir(parents=True)
        for month, df in _symbol_months(rng, sym).items():
            df.to_csv(csv_dir / f"{sym}-{month}_{TF}.csv", index=False)
            df.to_parquet(pq_dir / f"{sym}-{month}_{TF}.parquet", index=False)
    return home, pq_root


def _cfg(symbols=("BTC", "ETH"), **over):
    cfg = {"EXCHANGE": "BINANCE", "DATA": DATA, "TIME_UNIT": TF,
           "SYMBOLS": [f"BINANCE_PERP_{s}_USDT" for s in symbols]}
    cfg.update(over)
    return cfg


def _load(home, cfg):
    r = AgamottoResearch(cfg, str(home))
    r.load()
    return r.raw


class TestParity:
    def test_parquet_frame_equals_csv_frame(self, tmp_path):
        home, pq_root = _write_trees(tmp_path)
        csv_raw = _load(home, _cfg())
        pq_raw = _load(home, _cfg(KLINE_PARQUET_ROOT=str(pq_root)))
        pd.testing.assert_frame_equal(csv_raw, pq_raw)

        # Not a vacuous equality: both symbols, whitelist honoured, outer join.
        assert {c.split("_")[0] for c in pq_raw.columns} == {"BTCUSDT", "ETHUSDT"}
        assert "BTCUSDT_close_time_ms" not in pq_raw.columns
        assert pq_raw.index.tz is None
        assert pq_raw.index.is_unique and pq_raw.index.is_monotonic_increasing
        assert len(pq_raw) == 11
        assert pq_raw["ETHUSDT_close"].isna().sum() == 1
        assert (pq_raw.dtypes == float).all()

    def test_duplicates_keep_last(self, tmp_path):
        home, pq_root = _write_trees(tmp_path)
        pq_raw = _load(home, _cfg(KLINE_PARQUET_ROOT=str(pq_root)))
        # Within-file duplicate of bar 2: the appended close=999 row wins.
        assert pq_raw["BTCUSDT_close"].iloc[2] == 999.0
        # Bar 5 is in both months: the later file (2026-09) wins.
        rng = np.random.default_rng(3)
        btc = _symbol_months(rng, "BTCUSDT")
        assert pq_raw["BTCUSDT_close"].iloc[5] == btc["2026-09"]["close"].iloc[0]

    def test_empty_whitelist_loads_every_symbol_on_both_paths(self, tmp_path):
        home, pq_root = _write_trees(tmp_path)
        csv_raw = _load(home, _cfg(symbols=()))
        pq_raw = _load(home, _cfg(symbols=(), KLINE_PARQUET_ROOT=str(pq_root)))
        pd.testing.assert_frame_equal(csv_raw, pq_raw)
        assert {c.split("_")[0] for c in pq_raw.columns} == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


class TestParquetFailsLoud:
    def test_missing_whitelisted_symbol_raises(self, tmp_path):
        home, pq_root = _write_trees(tmp_path)
        with pytest.raises(FileNotFoundError, match="XRPUSDT") as ei:
            _load(home, _cfg(symbols=("BTC", "XRP"), KLINE_PARQUET_ROOT=str(pq_root)))
        assert str(pq_root / TF / DATA) in str(ei.value)

    def test_whitelisted_symbol_with_no_files_raises(self, tmp_path):
        home, pq_root = _write_trees(tmp_path)
        for f in (pq_root / TF / DATA / "ETHUSDT").iterdir():
            f.unlink()
        with pytest.raises(FileNotFoundError, match="ETHUSDT"):
            _load(home, _cfg(KLINE_PARQUET_ROOT=str(pq_root)))

    def test_corrupt_parquet_raises_with_path(self, tmp_path):
        home, pq_root = _write_trees(tmp_path)
        bad = pq_root / TF / DATA / "ETHUSDT" / f"ETHUSDT-2026-09_{TF}.parquet"
        bad.write_bytes(b"not a parquet file")
        with pytest.raises(ValueError, match="unreadable kline parquet") as ei:
            _load(home, _cfg(KLINE_PARQUET_ROOT=str(pq_root)))
        assert str(bad) in str(ei.value)

    def test_parquet_missing_required_column_raises_with_path(self, tmp_path):
        home, pq_root = _write_trees(tmp_path)
        path = pq_root / TF / DATA / "BTCUSDT" / f"BTCUSDT-2026-08_{TF}.parquet"
        pd.read_parquet(path).drop(columns=["low"]).to_parquet(path, index=False)
        with pytest.raises(ValueError, match="low") as ei:
            _load(home, _cfg(KLINE_PARQUET_ROOT=str(pq_root)))
        assert str(path) in str(ei.value)

    def test_missing_root_raises(self, tmp_path):
        home, _ = _write_trees(tmp_path)
        with pytest.raises(FileNotFoundError, match="KLINE_PARQUET_ROOT"):
            _load(home, _cfg(KLINE_PARQUET_ROOT=str(tmp_path / "nope")))

    def test_relative_root_raises(self, tmp_path):
        home, _ = _write_trees(tmp_path)
        with pytest.raises(ValueError, match="absolute"):
            _load(home, _cfg(KLINE_PARQUET_ROOT="pq"))

    def test_stocks_with_parquet_raises(self, tmp_path):
        home, pq_root = _write_trees(tmp_path)
        with pytest.raises(ValueError, match="STOCKS"):
            _load(home, _cfg(EXCHANGE="STOCKS", KLINE_PARQUET_ROOT=str(pq_root)))

    def test_mm_minute_bars_refuse_parquet_root(self, tmp_path):
        home, pq_root = _write_trees(tmp_path)
        r = AgamottoResearch(_cfg(KLINE_PARQUET_ROOT=str(pq_root)), str(home))
        with pytest.raises(ValueError, match="KLINE_PARQUET_ROOT"):
            r._load_minute_bars("BTCUSDT")


class TestCsvUnchanged:
    def test_corrupt_csv_is_still_logged_and_skipped(self, tmp_path, caplog):
        """The CSV path's historical catch-log-skip is untouched by the refactor."""
        home, _ = _write_trees(tmp_path)
        bad = home / "data" / "BINANCEFUTURES" / TF / DATA / "ETHUSDT" / f"ETHUSDT-2026-09_{TF}.csv"
        pd.read_csv(bad).drop(columns=["low"]).to_csv(bad, index=False)
        with caplog.at_level(logging.WARNING, logger="agamotto.research"):
            raw = _load(home, _cfg())
        assert any(str(bad) in rec.getMessage() for rec in caplog.records)
        # ETH kept only its August rows (6 unique bars); BTC still has all 11.
        assert raw["ETHUSDT_close"].notna().sum() == 6
        assert raw["BTCUSDT_close"].notna().sum() == 11
