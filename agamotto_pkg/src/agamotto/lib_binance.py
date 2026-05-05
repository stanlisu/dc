"""Utility helpers for interacting with Binance Futures REST APIs."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Dict, Iterable, List
from pathlib import Path
import json
import logging
import threading
import time

import pandas as pd
import requests

from .utils import _symbol_to_native  # noqa: F401 — canonical location

_logger = logging.getLogger(__name__)

BINANCE_FUTURES_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_FUTURES_KLINE_URL = "https://fapi.binance.com/fapi/v1/klines"

# ---------------------------------------------------------------------------
# Module-level cache for exchangeInfo (Fix 5)
# ---------------------------------------------------------------------------
_exchange_info_cache: dict = {}
_exchange_info_cache_ts: float = 0.0
_exchange_info_cache_lock = threading.Lock()
_EXCHANGE_INFO_TTL = 60.0  # seconds


def _fetch_exchange_info(max_retries: int = 4, pause: float = 1.0) -> dict:
    """Fetch exchangeInfo with TTL cache and retry+backoff on 429/418.

    Returns the parsed JSON payload (dict). Raises RuntimeError after
    exhausting retries.
    """
    global _exchange_info_cache, _exchange_info_cache_ts

    with _exchange_info_cache_lock:
        if _exchange_info_cache and (time.monotonic() - _exchange_info_cache_ts) < _EXCHANGE_INFO_TTL:
            return _exchange_info_cache

    # Cache miss — fetch with retry
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                BINANCE_FUTURES_EXCHANGE_INFO_URL, timeout=30)

            if response.status_code in (429, 418):
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else pause * (2 ** (attempt - 1))
                except (TypeError, ValueError):
                    wait = pause * (2 ** (attempt - 1))
                _logger.warning(
                    "exchangeInfo rate-limited (%s), retry %d/%d in %.1fs",
                    response.status_code, attempt, max_retries, wait)
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Binance exchangeInfo rate-limited after "
                        f"{max_retries} retries (HTTP {response.status_code})")
                time.sleep(wait)
                continue

            response.raise_for_status()
            payload = response.json()

            with _exchange_info_cache_lock:
                _exchange_info_cache = payload
                _exchange_info_cache_ts = time.monotonic()

            return payload

        except requests.RequestException as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to query Binance Futures exchange info: "
                    f"{exc}") from exc
            wait = pause * (2 ** (attempt - 1))
            _logger.warning(
                "exchangeInfo request failed (%s), retry %d/%d in %.1fs",
                exc, attempt, max_retries, wait)
            time.sleep(wait)

    # Should not reach here, but satisfy type checkers
    raise RuntimeError("Failed to query Binance Futures exchange info")


def _interval_to_millis(interval: str) -> int:
    mapping = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
        "1M": 30 * 24 * 60 * 60 * 1000,
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported Binance interval: {interval}")
    return mapping[interval]


def _precision_from_tick(tick: str) -> int:
    try:
        value = Decimal(tick)
    except Exception:
        return 0
    if value == 0:
        return 0
    normalized = value.normalize()
    exponent = -normalized.as_tuple().exponent
    return max(0, exponent)


def fetch_exchange_filters(symbols: Iterable[str]) -> Dict[str, Dict[str, int]]:
    native_map: Dict[str, str] = {}
    for sym in symbols:
        converted = _symbol_to_native(sym)
        if converted:
            native_map[converted] = sym

    if not native_map:
        return {}

    payload = _fetch_exchange_info()

    symbol_map = {
        entry.get("symbol"): entry
        for entry in payload.get("symbols", [])
        if isinstance(entry, dict)
    }

    result: Dict[str, Dict[str, int]] = {}
    for native, original in native_map.items():
        info = symbol_map.get(native)
        if not info:
            continue

        filters = info.get("filters", [])
        price_filter = next((f for f in filters if f.get("filterType") == "PRICE_FILTER"), None)
        lot_filter = next((f for f in filters if f.get("filterType") in {"LOT_SIZE", "MARKET_LOT_SIZE"}), None)
        if not price_filter or not lot_filter:
            continue

        price_precision = _precision_from_tick(price_filter.get("tickSize", "0"))
        volume_precision = _precision_from_tick(lot_filter.get("stepSize", "0"))

        result[original] = {
            "price_tick_size": price_precision,
            "volume_tick_size": volume_precision,
        }

    return result


def fetch_price_lot_sizes(symbols: Iterable[str]) -> Dict[str, Dict[str, float]]:
    native_map: Dict[str, str] = {}
    for sym in symbols:
        converted = _symbol_to_native(sym)
        if converted:
            native_map[converted] = sym

    if not native_map:
        return {}

    payload = _fetch_exchange_info()

    symbol_map = {
        entry.get("symbol"): entry
        for entry in payload.get("symbols", [])
        if isinstance(entry, dict)
    }

    result: Dict[str, Dict[str, float]] = {}
    for native, original in native_map.items():
        info = symbol_map.get(native)
        if not info:
            continue

        filters = info.get("filters", [])
        price_filter = next((f for f in filters if f.get("filterType") == "PRICE_FILTER"), None)
        lot_filter = next((f for f in filters if f.get("filterType") in {"LOT_SIZE", "MARKET_LOT_SIZE"}), None)
        if not price_filter or not lot_filter:
            continue

        notional_filter = next(
            (f for f in filters if f.get("filterType") == "MIN_NOTIONAL"), None)

        try:
            tick_size = float(price_filter.get("tickSize", "0"))
            step_size = float(lot_filter.get("stepSize", "0"))
            min_notional = float(notional_filter.get("notional", "0")) if notional_filter else 0.0
        except (TypeError, ValueError):
            continue

        result[original] = {
            "tick_size": tick_size,
            "step_size": step_size,
            "min_notional": min_notional,
        }

    return result


def get_price_lot_size(symbol: str) -> Dict[str, float]:
    data = fetch_price_lot_sizes([symbol])
    return data.get(symbol, {})


def fetch_latest_closes(symbols: Iterable[str], interval: str = "1d") -> Dict[str, float]:
    """Fetch latest close prices using batch ticker endpoint (single API call)."""
    symbols = list(symbols)

    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v2/ticker/price", timeout=10)
        resp.raise_for_status()
        ticker_data = {
            item["symbol"]: float(item["price"]) for item in resp.json()
        }

        result: Dict[str, float] = {}
        for sym in symbols:
            native = _symbol_to_native(sym)
            if not native:
                continue
            if native in ticker_data:
                result[sym] = ticker_data[native]
            else:
                _logger.warning(f"No ticker price for {sym} ({native})")
        return result
    except Exception as e:
        _logger.warning(
            f"Batch ticker failed ({e}), falling back to kline fetch")

    # Fallback: sequential kline fetch (original behavior)
    now_ms = int(dt.datetime.utcnow().timestamp() * 1000)
    interval_ms = _interval_to_millis(interval)
    start_ms = now_ms - interval_ms * 200

    result = {}
    for sym in symbols:
        native = _symbol_to_native(sym)
        if not native:
            continue
        rows = fetch_futures_klines(
            native, interval, start_ms, now_ms, limit=200)
        if not rows:
            continue
        last_close = float(rows[-1][4])
        result[sym] = last_close
    return result


def load_symbols_from_setting(setting_path: str | Path) -> List[str]:
    setting_path = Path(setting_path)
    with setting_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("SYMBOLS", [])


def fetch_tick_sizes_from_setting(setting_path: str | Path) -> pd.DataFrame:
    symbols = load_symbols_from_setting(setting_path)
    filters = fetch_exchange_filters(symbols)
    if not filters:
        return pd.DataFrame()
    df = pd.DataFrame(filters).T
    df.index.name = "symbol"
    return df


def fetch_futures_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1500,
    pause: float = 0.5,
    max_retries: int = 5,
) -> List[List[str]]:
    interval_ms = _interval_to_millis(interval)

    rows: List[List[str]] = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": min(limit, 1500),
        }

        payload = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(BINANCE_FUTURES_KLINE_URL, params=params, timeout=15)

                if response.status_code in (429, 418):
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after else pause * (2 ** (attempt - 1))
                    except (TypeError, ValueError):
                        wait = pause * (2 ** (attempt - 1))
                    _logger.warning(
                        "Kline rate-limited (%s) for %s, retry %d/%d in %.1fs",
                        response.status_code, symbol, attempt, max_retries, wait)
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"Binance kline rate-limited after {max_retries} "
                            f"retries for {symbol} (HTTP {response.status_code})")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                payload = response.json()
                break
            except requests.RequestException as exc:
                if attempt == max_retries:
                    raise RuntimeError(f"Binance kline request failed for {symbol}: {exc}") from exc
                time.sleep(pause * (2 ** (attempt - 1)))

        if payload is None or not isinstance(payload, list) or not payload:
            break

        rows.extend(payload)

        last_open_ms = int(payload[-1][0])
        next_start = last_open_ms + interval_ms
        if next_start <= current_start:
            next_start = current_start + interval_ms

        current_start = next_start

        if len(payload) < limit:
            break

        time.sleep(pause)

    rows.sort(key=lambda entry: int(entry[0]))
    return rows


def klines_to_dataframe(rows: List[List[str]], symbol: str, interval: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "open_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time_ms",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ])

    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df[numeric_cols]
    df.columns = [f"{symbol}_{interval}_{col}" for col in df.columns]
    df.index = df.index.tz_convert(None)
    return df


def parse_ws_kline_to_row(
    kline_data: dict, symbol: str, interval: str
) -> pd.DataFrame:
    """Convert a Binance WS kline 'k' object to a single-row DataFrame.

    Output format matches klines_to_dataframe() exactly:
    - Index: tz-naive UTC Timestamp from open_time_ms
    - Columns: {symbol}_{interval}_{col} for each OHLCV field
    - All values float64
    """
    ts = pd.Timestamp(kline_data["t"], unit="ms", tz="UTC").tz_localize(None)

    col_map = [
        ("open", "o"),
        ("high", "h"),
        ("low", "l"),
        ("close", "c"),
        ("volume", "v"),
        ("quote_volume", "q"),
        ("number_of_trades", "n"),
        ("taker_buy_base_volume", "V"),
        ("taker_buy_quote_volume", "Q"),
    ]

    data = {
        f"{symbol}_{interval}_{col}": [float(kline_data[ws_key])]
        for col, ws_key in col_map
    }

    return pd.DataFrame(data, index=[ts])
