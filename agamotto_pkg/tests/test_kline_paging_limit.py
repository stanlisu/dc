"""`fetch_futures_klines` must page against the CLAMPED page size, not the
caller's total.

Binance caps a single /klines page at 1500 rows, so the request sends
`min(limit, 1500)`. The page-exhaustion test used the caller's *unclamped*
`limit`: a caller asking for 3000 rows got a full 1500-row page, `1500 < 3000`
evaluated True, the loop broke, and the function silently returned 1500 rows —
a plausible-looking short result instead of the data or an error.

Latent, not live (`trading.py` asks for 700, `orb/trading.py` for `limit + 1`,
the function default is 1500), but it is exactly the silent-truncation class
CLAUDE.md's no-silent-fallback rule targets.

The rate-limit contract established by 30efda1 must survive the fix: sleeps are
always exactly `requests - 1`, i.e. a pause only ever precedes a request that is
actually issued.
"""
import pytest

from agamotto import lib_binance


# The real endpoint's hard per-page ceiling. The mocked transport below serves
# at most this many rows regardless of the `limit` it is sent, which is the
# whole point: the production loop must notice a "full" page is 1500, not
# `limit`.
EXCHANGE_PAGE_CAP = 1500


class _Resp:
    status_code = 200
    headers: dict = {}

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _install(monkeypatch, interval_ms, data_end_ms=None):
    """Model the real endpoint: serve klines from `startTime`, never past
    `endTime` (nor past `data_end_ms`, the exchange's own data horizon), and
    never more than EXCHANGE_PAGE_CAP rows in one page.

    Records every request and every sleep.
    """
    calls, sleeps = [], []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params))
        start = params["startTime"]
        end = params["endTime"]
        if data_end_ms is not None and data_end_ms < end:
            end = data_end_ms
        available = (end - start) // interval_ms
        if available < 0:
            available = 0
        # Exchange-side truncation of a caller-supplied request; not a clamp on
        # a derived quantity.
        n = min(params["limit"], EXCHANGE_PAGE_CAP, available)
        return _Resp([[start + i * interval_ms] + ["1"] * 11
                      for i in range(n)])

    monkeypatch.setattr(lib_binance.requests, "get", fake_get)
    monkeypatch.setattr(lib_binance.time, "sleep", lambda s: sleeps.append(s))
    return calls, sleeps


def _ceil_div(a, b):
    return -(-a // b)


@pytest.mark.parametrize("limit", [3000, 4000, 1501, 4500])
def test_over_cap_limit_pages_until_the_window_is_covered(monkeypatch, limit):
    """(a) A caller asking for > 1500 rows gets all of them.

    Pre-fix this returned exactly 1500 rows after a single request for every
    `limit` above the cap.
    """
    interval_ms = 60 * 1000
    calls, sleeps = _install(monkeypatch, interval_ms)

    start = 1_700_000_000_000
    rows = lib_binance.fetch_futures_klines(
        "BTCUSDT", "1m", start, start + limit * interval_ms,
        limit=limit, pause=0.5)

    assert len(rows) == limit, (
        f"asked for {limit} rows, got {len(rows)} — silent truncation")
    expected_calls = _ceil_div(limit, EXCHANGE_PAGE_CAP)
    assert len(calls) == expected_calls, (
        f"expected {expected_calls} requests, got {len(calls)}")
    # (d) a pause only ever precedes a request that is actually issued.
    assert sleeps == [0.5] * (expected_calls - 1), (
        f"expected {expected_calls - 1} pauses, got {sleeps}")
    # Every request must ask for the clamped page size, never the raw total.
    assert all(c["limit"] == EXCHANGE_PAGE_CAP for c in calls), \
        [c["limit"] for c in calls]
    opens = [int(r[0]) for r in rows]
    assert opens == [start + i * interval_ms for i in range(limit)]


def test_over_cap_limit_stops_when_the_exchange_runs_out(monkeypatch):
    """(c) A short final page still terminates paging when limit > 1500."""
    interval_ms = 60 * 1000
    available = 2200  # exchange has less data than the requested window
    start = 1_700_000_000_000
    calls, sleeps = _install(
        monkeypatch, interval_ms, data_end_ms=start + available * interval_ms)

    rows = lib_binance.fetch_futures_klines(
        "BTCUSDT", "1m", start, start + 5000 * interval_ms,
        limit=5000, pause=0.5)

    assert len(rows) == available
    # page 1 = 1500 (full), page 2 = 700 (short) -> stop.
    assert len(calls) == 2, f"expected 2 requests, got {len(calls)}"
    assert sleeps == [0.5], f"expected exactly one pause, got {sleeps}"


def test_at_cap_limit_is_unchanged(monkeypatch):
    """(b) limit == 1500 (the function default) behaves as before."""
    interval_ms = 60 * 1000
    calls, sleeps = _install(monkeypatch, interval_ms)

    start = 1_700_000_000_000
    rows = lib_binance.fetch_futures_klines(
        "BTCUSDT", "1m", start, start + 1500 * interval_ms, pause=0.5)

    assert len(rows) == 1500
    assert len(calls) == 1
    assert sleeps == []


def test_under_cap_single_page_is_unchanged(monkeypatch):
    """(b) The live shape — 700 bars, one page, no pause."""
    interval_ms = 15 * 60 * 1000
    calls, sleeps = _install(monkeypatch, interval_ms)

    start = 1_700_000_000_000
    rows = lib_binance.fetch_futures_klines(
        "BTCUSDT", "15m", start, start + 700 * interval_ms,
        limit=700, pause=0.5)

    assert len(rows) == 700
    assert len(calls) == 1
    assert calls[0]["limit"] == 700, "sub-cap limit must pass through unclamped"
    assert sleeps == []


def test_under_cap_short_final_page_is_unchanged(monkeypatch):
    """(c) The pre-existing `len(payload) < limit` exit for limit <= 1500."""
    interval_ms = 60 * 60 * 1000
    start = 1_700_000_000_000
    calls, sleeps = _install(
        monkeypatch, interval_ms, data_end_ms=start + 4 * interval_ms)

    rows = lib_binance.fetch_futures_klines(
        "BTCUSDT", "1h", start, start + 100 * interval_ms,
        limit=10, pause=0.5)

    assert len(rows) == 4
    assert len(calls) == 1
    assert sleeps == []
