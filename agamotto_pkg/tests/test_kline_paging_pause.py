"""`fetch_futures_klines` must only pause BETWEEN requests it actually makes.

The paging loop used to `time.sleep(pause)` after every full-sized page, then
immediately exit on `while current_start < end_ms`. The live path asks for
exactly 700 bars and gets exactly 700 back, so that sleep was pure dead time on
every symbol of every cycle (~1.24 s per cycle across the thread pool).

The pause between two REAL consecutive pages must survive — this is the only
rate limiting on the endpoint.
"""
from agamotto import lib_binance


class _Resp:
    status_code = 200
    headers: dict = {}

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _install(monkeypatch, interval_ms, rows_per_page):
    """Serve `rows_per_page` klines per request, starting at the requested
    startTime. Records every request and every sleep."""
    calls, sleeps = [], []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params))
        start = params["startTime"]
        return _Resp([[start + i * interval_ms] + ["1"] * 11
                      for i in range(rows_per_page)])

    monkeypatch.setattr(lib_binance.requests, "get", fake_get)
    monkeypatch.setattr(lib_binance.time, "sleep",
                        lambda s: sleeps.append(s))
    return calls, sleeps


def test_no_pause_when_one_full_page_covers_the_window(monkeypatch):
    """The live shape: ask for exactly `limit` bars, get exactly `limit`."""
    interval_ms = 15 * 60 * 1000
    limit = 700
    calls, sleeps = _install(monkeypatch, interval_ms, rows_per_page=limit)

    start = 1_700_000_000_000
    rows = lib_binance.fetch_futures_klines(
        "BTCUSDT", "15m", start, start + limit * interval_ms,
        limit=limit, pause=0.5)

    assert len(calls) == 1, f"expected 1 request, got {len(calls)}"
    assert len(rows) == limit
    assert sleeps == [], f"slept {sleeps} with no further request to make"


def test_pause_is_kept_between_two_real_consecutive_pages(monkeypatch):
    """Two full pages are needed -> exactly one pause, before the 2nd request."""
    interval_ms = 15 * 60 * 1000
    limit = 700
    calls, sleeps = _install(monkeypatch, interval_ms, rows_per_page=limit)

    start = 1_700_000_000_000
    rows = lib_binance.fetch_futures_klines(
        "BTCUSDT", "15m", start, start + 2 * limit * interval_ms,
        limit=limit, pause=0.5)

    assert len(calls) == 2, f"expected 2 requests, got {len(calls)}"
    assert len(rows) == 2 * limit
    assert sleeps == [0.5], f"expected exactly one 0.5s pause, got {sleeps}"


def test_pause_count_is_always_requests_minus_one(monkeypatch):
    interval_ms = 60 * 60 * 1000
    limit = 10
    for pages in (1, 2, 3, 5):
        calls, sleeps = _install(monkeypatch, interval_ms, rows_per_page=limit)
        start = 1_700_000_000_000
        lib_binance.fetch_futures_klines(
            "BTCUSDT", "1h", start, start + pages * limit * interval_ms,
            limit=limit, pause=0.25)
        assert len(calls) == pages
        assert sleeps == [0.25] * (pages - 1), (
            f"{pages} pages: expected {pages - 1} pauses, got {sleeps}")


def test_short_final_page_still_ends_without_a_trailing_pause(monkeypatch):
    """A partial page ends paging via the `len(payload) < limit` branch."""
    interval_ms = 60 * 60 * 1000
    calls, sleeps = _install(monkeypatch, interval_ms, rows_per_page=4)

    start = 1_700_000_000_000
    rows = lib_binance.fetch_futures_klines(
        "BTCUSDT", "1h", start, start + 100 * interval_ms,
        limit=10, pause=0.5)

    assert len(calls) == 1
    assert len(rows) == 4
    assert sleeps == []


def test_rows_are_still_sorted_and_complete(monkeypatch):
    interval_ms = 60 * 60 * 1000
    limit = 10
    calls, sleeps = _install(monkeypatch, interval_ms, rows_per_page=limit)
    start = 1_700_000_000_000
    rows = lib_binance.fetch_futures_klines(
        "BTCUSDT", "1h", start, start + 3 * limit * interval_ms,
        limit=limit, pause=0.0)

    opens = [int(r[0]) for r in rows]
    assert opens == sorted(opens)
    assert opens == [start + i * interval_ms for i in range(3 * limit)]
