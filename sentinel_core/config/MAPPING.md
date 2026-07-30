# setting.json → sentinel config mapping

Source: live `xmen/pred_mjolnir.base.30s_1_xmen/setting.json`.

## There is no executable to deploy

`libtsMjolnir.so` is a **plugin**, not a program. The executable is the vendor's
`tsLtpBaseAlgo` (`Strategy/ltp_release/bin/Release/`), which loads the plugin
through `sp_config.json → algo_details[].library_name`:

```
tsLtpBaseAlgo -f sp_config.mjolnir30s.json
   └── loads libtsMjolnir.so
         └── links libmjolnir_core.so   (LD_LIBRARY_PATH or rpath)
```

## Direct mappings

| setting.json | sentinel `algo_params` | value |
|---|---|---|
| `TIME_UNIT: "30s"` | `bar_sec`, `target_sec` | 30, 30 |
| `MIN_SIGNAL_COUNT` | `min_signal_count` | 1 |
| `REVERSE` | `reverse` | 1 |
| `HOLD_TTL_BARS` | `hold_ttl_bars` | 2 |
| `WARMUP_FIRE_GATE_BARS` | `warmup_fire_gate_bars` | 1080 |
| `REGIME_SUBSET: "trade_imbalance"` | `regime_subset` | **`"r068"`** (coded) |
| `REGIME_STACK_PATH` | `regime_stack_csv` | coded copy — see below |
| `WEIGHTS_PATH` | `weights_dir` | exported + coded — see below |
| `MULTI_TF_BARS: []` | (none) | single-TF; the C++ engine is single-TF only |

## Deliberately NOT mapped

| setting.json | why |
|---|---|
| `FEE`, `CAPITAL`, `LEVERAGE`, `TRADING_MODE` | execution-side. This build never trades, so they have no effect and are omitted rather than carried as dead config. |
| `STREAMS` | the Python bot opens its own websockets. Under LTP the feed arrives via shared memory; subscription is `subscribeProduct(product_id, …)`, driven by `product_id`/`levels`. |
| `TARGET_HORIZON_BARS`, `WINDOW_SIZE` | training-time parameters, already baked into the deployed weights. |
| `CANONICAL_BRIDGE_VENUE`, `INFERENCE_WORKERS` | Python bridge orchestration; no equivalent — each sentinel process is one symbol on one core. |

## Two differences that will change what you see

**1. `BACKFILL_WARMUP_BARS` has no C++ equivalent — and this matters.**
The Python bot backfills its warmup buffer from Tardis, so it fires shortly after
boot. The C++ core has **no backfill**: it warms from live bars only. At
`warmup_fire_gate_bars: 1080` and a 30 s bar that is **1080 × 30 s ≈ 9 hours**
before the first `[MJDEC]` line.

For a same-day smoke test, lower it (e.g. `240` ≈ 2 h) — but understand what you
are trading away: rolling windows run to 900 bars, so below ~900 the features are
still filling and will **not** match the Python bot. Use a low gate to prove the
pipeline runs; use 1080 to compare decisions.

**2. `product_id` cannot be derived.**
The Python bot addresses symbols by name (`BINANCE_PERP_BTC_USDT`); LTP addresses
them numerically. Look the id up in the contract file named by
`sp_config.contract_file`. It is the one value left at `0`, deliberately, so a
wrong run fails loudly instead of subscribing to the wrong instrument.

## Weights and stack must be CODED

The core accepts coded regime directories only (`r068_short`), and reads
`model.txt` / `scaler.txt` / `features.txt` rather than joblib pickles. Real names
are refused by design — accepting them would require embedding the name table,
which is what made the built `.so` recoverable with `strings`.

```bash
PYTHONPATH=mjolnir_pkg/src python marvel/mjolnir/gauntlet/export_sentinel_weights.py \
    --weights <exp>/.weight_cache/weights/window_YYYY_MM_DD \
    --out     <bundle>/weights
```

The exporter lives in **marvel**, not here, on purpose: the only thing it needs
from the obfuscation system is the regime encoder, and that codec is already
vendored into the deployed mjolnir package (`mjolnir._obf.codec`). So dc never
has to exist on a machine that exports weights or runs the bundle.

Live `WEIGHTS_PATH` is currently `window_2026_07_07`; export **that** window, not
an older one, or the shadow will disagree with the bot for reasons that have
nothing to do with the port.

## One shape difference from the SDK sample

The SDK sample's `algo_params` takes a `products[]` array. `MjolnirStrategy` takes
a scalar `product_id` + `levels`, because this build is one symbol per process
(one core per process, confirmed 2026-07-29). Multi-symbol fan-out is a separate
process per symbol, not a longer array.
