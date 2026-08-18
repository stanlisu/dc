# agamotto_core — private C++ core for the agamotto sentinel port

The **IP side** of the agamotto port. Implements the opaque contract declared in
the public `sentinel` repo
(`Strategy/ltp_release/ltp_strat_sdk/stan_code/agamotto_core.hpp`) and builds as
`libagamotto_core`, which the public strategy links against.

**Nothing in here may be copied into the sentinel repo.** Same split, and same
reasoning, as [`../sentinel_core`](../sentinel_core/README.md).

## Status: Phase 1 — the bar layer only

| stage | state |
|---|---|
| 15m kline builder from ticks | ✅ 27/27 self-test assertions |
| Backfill ingestion + validation | ✅ (loader lives in the strategy; CSV is the seam) |
| Parity vs Binance's own klines | ⏳ harness written, live numbers pending |
| Feature engine (~60 cols, 25 TA-Lib) | ❌ Phase 2 |
| Regime gate (33 base predicates) | ❌ Phase 3 |
| Ridge runner (top-5 IC per regime) | ❌ Phase 4 |

`decide()` returns `{fired=false}` unconditionally and the strategy says so in
its log. There is no ta-lib or LightGBM dependency yet, deliberately.

## Why not mjolnir's bar_builder with bar_sec=900

The two disagree on both things the bar layer does:

- **Bucketing.** Binance assigns a trade to a kline by exchange event time, on
  buckets tiling the UTC day. mjolnir closes on the *first trade of a new
  bucket* against its own stream clock — deliberate and parity-tested there.
  Agamotto trained on Binance's klines, so bar membership must be theirs.
- **Fields.** mjolnir's bar carries book depth, OI, funding and liquidations;
  agamotto needs Binance's nine kline columns. The overlap is OHLC + volume.

## What the live feed actually populates

Measured on hydra 2026-08-18 against the deployed (Aug-12) `tsBinanceFeedPublisher`.
This table is the reason several design decisions look paranoid:

| field | observed | consequence |
|---|---|---|
| `header.exchange_timestamp` | valid epoch ms | **the bucketing clock** |
| `message.last_trade_id` | valid, monotonic | dedupes replays |
| `message.last_trade_ts` | `18359008543379257232` (uninitialised) | unusable; range-checked and counted |
| `message.is_buyer_maker` | `0` on all 613 trades of a 45s capture | `taker_buy_*` cannot be exact → quote rule |
| `mark` / `index` / `funding` | `0` on every event | unusable (no `MARK_PRICE_UPDATE` seen) |

The maker-flag gap is the same one `sentinel/README.md` documents for mjolnir's
`f091`. The field now *exists* in the 20260804 `Quote`; it is simply never set.
The fix belongs in the Feed Publisher, which has Binance's own flag at parse
time — no amount of porting recovers it.

## The five bar rules

Each exists because getting it wrong yields a plausible-but-wrong bar:

1. **Only `TRADE_UPDATE` (6) contributes.** `AGG_TRADE_UPDATE` (7) is a distinct
   kind carrying the same fills re-aggregated; counting both doubles volume,
   quote_volume and number_of_trades while OHLC still looks perfect.
2. **Trade ids dedupe** replays and repeated SHM slots.
3. **The first bucket is discarded.** Attaching mid-bucket misses the trades
   before we attached; that bar would read as real with implausibly low volume.
4. **Empty buckets emit flat bars** (`o=h=l=c=prev close`, `v=0`), as Binance
   does. Skipping them shifts every rolling window downstream.
5. **A trade stamped in a closed bucket is dropped and counted**, never applied.

Trades strictly inside the spread are counted `unclassified` rather than
guessed, and every bar carries `aggressor_source` so no consumer can mistake a
quote-rule approximation for an exact figure.

## Build

```bash
cmake -S . -B build -DSENTINEL_REPO=$HOME/sandbox/sentinel -DAGAMOTTO_CORE_GITSHA=$(git rev-parse --short HEAD)
cmake --build build -j
```

On dev105 build inside `devbox-v5.1` (see `sentinel/README.md`); `docker run -it`
dies over ssh with no TTY, so invoke docker directly.

Then point the public strategy at it:

```bash
cmake -S <sentinel>/Strategy/ltp_release/ltp_strat_sdk -B <...>/build \
      -DLTP_SDK_LIB_TYPE=Release -DAGAMOTTO_CORE_DIR=$PWD/build
```

**The SDK must be 20260804 or newer.** The deployed feed publisher emits the
979-byte `Quote`; a binary built against the 954-byte 20260728 layout dies at
startup with `Mismatched shared memory segment, sizeof(T) = 954`.

## Tests

```bash
./build/kline_parity_driver --selftest                       # 27 assertions
./build/kline_parity_driver --ticks T.csv --klines K.csv      # offline replay diff

python tests/fetch_binance_klines.py --symbol BTCUSDT --interval 15m --limit 700 \
    --out <bundle>/config/backfill_BTCUSDT_15m.csv
python tests/compare_agbar_vs_binance.py --log <strategy.log> --symbol BTCUSDT --interval 1m
```

Run the live parity at `bar_sec=60` first: the code path is identical to 900 and
a Binance kline to diff against arrives every minute instead of every fifteen.

Parity tolerance is **relative (1e-9), never exact** — both sides accumulate the
same trades into a double in a different order, so identical values still differ
in the last ULP.

## Warmup

Agamotto needs **700 bars** (`VOL_Q_WINDOW=700`, `min_periods=700` fails closed;
the live bot fetches `limit=700`). At 15m that is **7.3 days** of live bars, so
backfill is not optional in practice. Network I/O stays out of the core — as it
does for mjolnir, whose weights also come from an external tool — so the seam is
a CSV written by `fetch_binance_klines.py` and loaded by the strategy before it
subscribes.
