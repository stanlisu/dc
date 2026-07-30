# Mjolnir shadow bundle — dry-run on the live feed

A two-file drop-in for running the C++ mjolnir strategy against a **live LTP feed
in shadow mode**: it computes bars, features, regimes and predictions exactly as
production does, logs every decision, and **sends nothing**.

## Why this is safe

**There is no order path compiled into the strategy.** `send_order` is never
called anywhere in `MjolnirStrategy.cpp` — this is not a `dry_run` flag that
could be flipped by a config edit or an operator typo. `handleOrderResponse` is
wired only to log an ERROR, because in this build receiving one means something
*else* is trading that strategy id.

What it does on a firing bar is exactly one thing:

```
[MJDEC] bar_ts=<ns> side=<+1|-1> y_pred=<f> thr=<f> n_trig=<n> regime=r<NNN>
```

Regimes are logged as **codes**, never names.

## Contents

```
lib/libmjolnir_core.so    the private core (self-contained: ta-lib 0.6.4 and
                          LightGBM 4.6.0 are statically linked; only stock
                          system libs remain)
lib/libtsMjolnir.so       the LTP strategy plugin, loaded by tsLtpBaseAlgo
config/algo_params.example.json
config/regime_stack.example.csv
```

## Install

1. Copy the bundle to the target host.
2. Point `tsLtpBaseAlgo` at `libtsMjolnir.so` via its `sp_config.json`
   (`algo_details[]`), the same way the sample strategies are wired.
3. Make `libmjolnir_core.so` resolvable — either keep the layout and set
   `LD_LIBRARY_PATH=<bundle>/lib`, or drop it beside the strategy. The plugin was
   linked with an rpath to its build-time core directory, so a moved bundle needs
   the env var.
4. Export weights for the deploy window (see below) into `<bundle>/weights`.
5. Copy `algo_params.example.json` → `algo_params.json` and fill in
   `product_id`, the paths, and `logger_core`.

## Weights — required, and they must be CODED

The core reads `model.txt` / `scaler.txt` / `features.txt`, not joblib pickles,
and it accepts **coded** regime directories only (`r068_short`, not the real
name). Real names are refused deliberately: accepting them would mean embedding
the name table, which is what made the built `.so` recoverable with `strings`.

Produce them with:

```bash
PYTHONPATH=mjolnir_pkg/src python marvel/mjolnir/gauntlet/export_sentinel_weights.py \
    --weights <exp>/.weight_cache/weights/window_YYYY_MM_DD \
    --out     <bundle>/weights
```

The exporter lives in **marvel**, not here, on purpose: the only thing it needs
from the obfuscation system is the regime encoder, and that codec is already
vendored into the deployed mjolnir package (`mjolnir._obf.codec`). So dc never
has to exist on a machine that exports weights or runs the bundle.

That tool re-predicts through both the original sklearn object and the exported
booster and refuses to write if they disagree, so a silently-different model
cannot ship.

The regime-stack CSV must use the same coded directory names.

## Config that must match the deployed Python bot

Mismatches here change the signal without erroring:

| key | live value | why it matters |
|---|---|---|
| `bar_sec` / `target_sec` | 30 / 30 | bar cadence and the cycle columns |
| `warmup_fire_gate_bars` | 1080 | nothing fires before this many bars |
| `min_signal_count` | 1 | vote threshold |
| `reverse` | 1 | flips side when -1 |
| `hold_ttl_bars` | 2 | holds a fired signal through N non-firing bars |
| `is_anchor` | true for the anchor symbol | peers need the anchor frame |

Required keys have **no defaults** — a missing key raises at construction rather
than silently picking a plausible value.

## What to expect in the log

```
[Mjolnir] SHADOW start core=core-<sha> product_id=... dry_run=0 guard_ms=1000
[Mjolnir] warming bars=NNN            (until warmup_fire_gate_bars)
[MJDEC] bar_ts=... side=-1 y_pred=... thr=... n_trig=1 regime=r068
[Mjolnir] shutdown ticks=... bars=... fires=... conv_err=...
```

`conv_err` counts dropped events whose fixed-point fields would not convert
(`toDouble` throws on NaN/Inf); a nonzero value is logged loudly rather than
swallowed.

## Verification status of what is in here

| stage | evidence |
|---|---|
| Bar builder | 53/53 fields vs the reference, 5 seeds |
| Feature engine | 160/160 columns, incl. all 25 TA-Lib indicators |
| Anchor cross-features | 191 columns, 31 cross, 3 anchor/peer seed pairs |
| Regime gate | 30/30 masks + negative controls |
| Model runner | bit-identical to the deployed model (`max_abs_diff` 0.0) |
| Numerical chain | **2085/2085 live firings reproduced, corr 1.000000** |
| Live path | replay shadow: 28.5k ticks → 3.4k bars → 85 decisions, hold-TTL correct |
| IP seam | 0 of 138 distinctive names present in the built artifact |

## What is NOT yet verified — read before trusting the output

**The `Quote` → `TickEvent` adaptation has never seen real exchange data.** Every
result above comes from the reference's own bars or synthetic ticks. This bundle
exists precisely to close that gap, so treat the FIRST live run as the test of
that adapter, not as a confirmation of the strategy.

Sanity checks worth doing on the first run, in order:

1. `bars=` climbs at roughly one bar per `bar_sec` per symbol.
2. `conv_err=0`.
3. Bars reconcile against the Python bot's `bars_<date>.csv` for the same window
   (this is the adapter check).
4. Only then compare `[MJDEC]` lines against the bot's `decisions_<date>.csv`.

Multi-symbol fan-out is not wired in this bundle: the anchor bus contract exists
and is parity-verified, but no transport publishes it between processes yet, so
run the anchor symbol alone for now.
