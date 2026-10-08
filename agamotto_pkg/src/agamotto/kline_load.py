"""Kline tree readers behind ``AgamottoResearch.load()`` -- CSV and parquet.

Moved out of research.py (2026-10-08) for a build reason: PyArmor's TRIAL
licence refuses to obfuscate a script past a size limit ("out of license"), and
dc PR #97 pushed research.py past it. Same code, moved; ``load()`` delegates.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import List

import pandas as pd

# The historical logger name, so moved log lines read exactly as before.
logger = logging.getLogger("agamotto.research")


# Kline columns `load()` keeps, in this order. Shared by the CSV and parquet
# readers so the two can never produce different frames.

KLINE_REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

KLINE_OPTIONAL_COLS = [
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def prepare_kline_file(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """One monthly kline file -> UTC-indexed float OHLCV frame.

    The per-file post-processing both `load()` readers share: index from
    `open_time_ms`, duplicate timestamps dropped keep="last", required columns
    enforced, optional columns kept when present, everything cast to float.
    Raises on a missing required column; what the caller does with that raise
    is the caller's policy (the CSV reader logs and skips, the parquet reader
    propagates).
    """
    if "open_time_ms" not in df.columns:
        raise ValueError(f"Missing required column 'open_time_ms' in {path}")

    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    missing_required = [col for col in KLINE_REQUIRED_COLS if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    existing_cols = [col for col in KLINE_REQUIRED_COLS + KLINE_OPTIONAL_COLS
                     if col in df.columns]
    return df[existing_cols].astype(float)


def finish_symbol_frame(symbol: str, symbol_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate one symbol's monthly frames and prefix its columns.

    The sort MUST be stable: files are concatenated in path order, and the
    keep="last" dedupe below means "the later file wins" only if equal
    timestamps keep that order. The default (quicksort) does not — on two
    fully overlapping 5,000-row files it kept the EARLIER file's row on
    ~2,500 of them (tests/test_load_parquet.py).
    """
    symbol_df = pd.concat(symbol_frames).sort_index(kind="stable")
    symbol_df = symbol_df[~symbol_df.index.duplicated(keep="last")]
    symbol_df.columns = [f"{symbol}_{col}" for col in symbol_df.columns]
    return symbol_df


def load_csv_frames(source_dir: str, whitelist: set, timeframe: str) -> List[pd.DataFrame]:
    """The CSV tree reader — the historical `load()` body, behaviour unchanged."""
    frames: List[pd.DataFrame] = []
    for symbol_dir in sorted(glob.glob(f"{source_dir}/*")):
        if not os.path.isdir(symbol_dir):
            continue

        symbol = os.path.basename(symbol_dir)
        if whitelist and symbol.upper() not in whitelist:
            continue

        symbol_frames = []
        for csv_path in sorted(glob.glob(f"{symbol_dir}/*_{timeframe}.csv")):
            logger.debug(f"Loading {csv_path}")
            try:
                df = pd.read_csv(csv_path, header=0)
                symbol_frames.append(prepare_kline_file(df, csv_path))
            except Exception as exc:
                # WHY: historical CSV behaviour, kept byte-for-byte so no
                # existing arm's frame changes. The parquet reader below
                # does NOT copy this — it raises on any bad file.
                logger.warning(f"Failed to load {csv_path}: {exc}")
                continue

        if symbol_frames:
            frames.append(finish_symbol_frame(symbol, symbol_frames))
    return frames


def load_parquet_frames(parquet_root: str, whitelist: set, timeframe: str,
                        data_family: str) -> List[pd.DataFrame]:
    """Read `{KLINE_PARQUET_ROOT}/{TIME_UNIT}/{DATA}/{SYMBOL}/*_{TIME_UNIT}.parquet`.

    Same symbol directories, same whitelist matching and same per-file
    post-processing as the CSV reader, so the frame is identical. Unlike the
    CSV reader nothing is skipped: an unreadable or malformed file raises with
    its path, and a whitelisted symbol with no parquet files raises naming the
    symbol and the directory searched.
    """
    if not os.path.isabs(parquet_root):
        raise ValueError(
            f"KLINE_PARQUET_ROOT={parquet_root!r} must be an absolute local directory "
            f"(s3:// and relative paths are not supported).")
    if not os.path.isdir(parquet_root):
        raise FileNotFoundError(
            f"KLINE_PARQUET_ROOT={parquet_root!r} is not an existing directory.")

    source_dir = f"{parquet_root.rstrip('/')}/{timeframe}/{data_family}"
    frames: List[pd.DataFrame] = []
    loaded: set = set()
    for symbol_dir in sorted(glob.glob(f"{source_dir}/*")):
        if not os.path.isdir(symbol_dir):
            continue

        symbol = os.path.basename(symbol_dir)
        if whitelist and symbol.upper() not in whitelist:
            continue

        paths = sorted(glob.glob(f"{symbol_dir}/*_{timeframe}.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"no *_{timeframe}.parquet files for {symbol} under {symbol_dir}.")

        symbol_frames = []
        for pq_path in paths:
            logger.debug(f"Loading {pq_path}")
            try:
                df = pd.read_parquet(pq_path)
            except Exception as exc:
                raise ValueError(f"unreadable kline parquet {pq_path}: {exc!r}") from exc
            try:
                symbol_frames.append(prepare_kline_file(df, pq_path))
            except Exception as exc:
                raise ValueError(f"malformed kline parquet {pq_path}: {exc}") from exc

        frames.append(finish_symbol_frame(symbol, symbol_frames))
        loaded.add(symbol.upper())

    missing = sorted(whitelist - loaded)
    if missing:
        raise FileNotFoundError(
            f"no kline parquet directory for SYMBOLS {missing}: searched "
            f"{[f'{source_dir}/{sym}' for sym in missing]}.")
    if not frames:
        raise RuntimeError(f"No parquet files matched in {source_dir}")
    return frames
