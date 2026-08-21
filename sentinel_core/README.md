# sentinel_core — private C++ core for the mjolnir sentinel port

This is the **IP side** of the sentinel port. It implements the opaque contract
declared in the public `sentinel` repo
(`Strategy/ltp_release/ltp_strat_sdk/stan_code/mjolnir_core.hpp`) and is built as
`libmjolnir_core`, which the public strategy links against.

**Nothing in here may be copied into the sentinel repo.**

## Why the split

The public repo vendors a third-party SDK and is a publication candidate. The
contract header reveals only arity and codes; the parts that would disclose the
strategy live here:

| module | holds | why private |
|---|---|---|
| `bar_builder.cpp` | tick bar schema, stream handling, bar-close rule | equivalent of the reference `live_bar.py`, already excluded from public today |
| `feature_engine.cpp` | the ~396-feature math, rolling windows, TA calls | the alpha |
| `regime_gate.cpp` | regime predicates | the alpha |
| `model_runner.cpp` | LightGBM C-API load/predict, scaler | reveals feature ordering + weight layout |

The public side gets `mjolnir_core_stub.cpp`: compiles, links, never emits a bar,
reports `coreIsRealImplementation() == false` so the strategy refuses to run
against it outside explicit dry-run.

## Codes, not names — non-negotiable

Use the **same** global map as the rest of dc (`dc/obfuscation/map.json`,
72 regimes → `rNNN`, 100 features → `fNNN`). Never invent a second numbering:
weights, regime-stack CSVs and filter parquets are already coded against this
map, and a parallel scheme silently breaks interop.

- Regime codes cross the boundary as `uint16_t` (`r068` → `68`).
- Feature vectors cross as `double*` **ordered by code**; never by name.
- Deployed weights already carry coded `meta.feature_columns`, so the loader
  resolves **code → index** and needs no name path at all.

## Leak gate

`dc/obfuscation/audit_public_surface.py` scans the public checkout and fails on
any distinctive real name. It runs from the sentinel pre-commit hook
(`tools/githooks/pre-commit`). It scans the PUBLIC repo — this directory is
private and is deliberately *not* scanned.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Then point the public strategy build at it:

```bash
cmake -S <sentinel>/Strategy/ltp_release/ltp_strat_sdk \
      -B <sentinel>/.../build \
      -DMJOLNIR_CORE_DIR=$PWD/build
```

## Parity tests

Each harness in `tests/` is a Python script that drives the SAME synthetic
stream through the reference and through a small C++ driver, then diffs.
`CMakeLists.txt` builds only the library — the drivers are built by:

```bash
SENTINEL_REPO=~/sandbox/sentinel ./build_parity_drivers.sh
```

`./build_parity_drivers.sh --list` shows each driver and its sources; pass a
name (`./build_parity_drivers.sh regime_parity`) to build just one. It builds
inside `mjolnir-core-build`, which is what pins TA-Lib 0.6.4 and LightGBM
4.6.0 to the production versions — **do not hand-roll a `g++` line on the
host**, since a driver built against different library versions is a silent
parity break rather than a rounding difference. Every `tests/*_driver.cpp` must
have a recipe in the script; one without an entry fails the build instead of
being skipped.

`warmup_gate_driver` is the exception to "each harness is driven by a Python
script": it is a self-contained C++ unit test over `src/warmup_gate.hpp` with no
TA-Lib, LightGBM, weights or stack, so it just runs (`./build-linux/warmup_gate_driver`,
exit 0 = pass). It pins the semantics the port got wrong — the warmup gate counts
bars **ever fed**, not bar-buffer occupancy, so a gate above `BUFFER_MAXLEN`
still opens. See DEPLOY_SHADOW.md.

### Three prerequisites, all of which fail loud

**1. `mjolnir/_obf` must be generated.** It is derived and gitignored, so a
fresh checkout does not have it and the reference's `regime_filters` import
raises. The harness then SKIPs every regime and reports
`compared 0 regimes` — a FAIL, not a pass. Generate it first:

```bash
cd ~/sandbox/dc && python3.11 obfuscation/build_map.py && python3.11 obfuscation/sync_vendor.py
```

**2. The reference must have TA-Lib, at the pinned version.** The C++ links
`libta-lib` unconditionally (`CMakeLists.txt` raises `FATAL_ERROR` without it),
but `features.py` wraps its TA-Lib block in `except ImportError` and falls back
to `_numpy_indicators`, which stubs `adx`/`dx`/`cci`/`willr`/`stoch`/`sar`/
`obv`/`ad`/`mfi` to NaN and hand-rolls RSI and MACD. A reference without TA-Lib
therefore grades the C++ against **different math**.

This is not hypothetical. On dev105 (2026-08-04) `regime_parity.py` reported
five failures — `rsi_oversold`, `rsi_overbought`, `macd_bullish`,
`macd_bearish`, `adx_trend` — with `adx_trend` firing 0 times on the reference
for the dull reason that `NaN > 25` is False on every bar. The C++ was correct
throughout: installing TA-Lib 0.6.4 made all 30 masks identical, and
`feature_parity` all 160 shared columns identical, with no source change.

The three panel harnesses (`feature_parity`, `regime_parity`,
`btc_cross_parity`) now refuse to run without it, and additionally assert the
computed panel does not carry the `_numpy_indicators` stub signature. Install
the same version production runs:

```bash
# Take the C library straight out of the image that pins it, so the reference
# and the C++ are the same 0.6.4 by construction. The `cd` matters: the glob is
# expanded in the container's CWD, so `tar -C /usr/local` alone matches nothing.
# Copy the .so SYMLINKS too — the linker resolves -lta-lib via libta-lib.so.
docker run --rm --platform linux/amd64 mjolnir-core-build:latest \
    bash -c "cd /usr/local && tar -cf - lib/libta-lib.* include/ta-lib" | tar -C ~/.local -xf -
TA_INCLUDE_PATH=$HOME/.local/include TA_LIBRARY_PATH=$HOME/.local/lib \
    python3.11 -m pip install --user 'TA-Lib==0.6.4'
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
```

Verify with `python3.11 -c "import talib; print(talib.__ta_version__)"` — the
harnesses check `__ta_version__` (the C library) rather than the wrapper
version, because the library is what computes the indicators.

**3. The reference environment needs the feature module's PACKAGE deps.** The
three panel harnesses load `features.py` as a package member
(`mjolnir.core.features`, via `load_reference_pkg` in `tests/bar_parity.py`),
because since dc 3fe8e57 it does `from .features_scalefree import
scale_free_levels` and a relative import has nothing to resolve against when
the file is loaded standalone. Importing it as a package member also runs
`mjolnir/core/__init__.py`, which imports `research` and therefore **pyarrow**
— a dependency the old standalone load did not pull in. Missing it is reported
as `BLOCKED: cannot import mjolnir.core.features`, never as a parity failure.

```bash
python3.11 -m pip install --user numpy pandas pyarrow
```

`bar_parity.py` deliberately keeps the standalone load for `knull/live_bar.py`:
that reference is self-contained, so importing it as a package member would run
`knull/__init__.py` and drag the bot's import graph into a bar-builder test for
no gain.

### Running

`python3` is 3.6 on some build hosts; use `python3.11` or newer.

```bash
python3.11 tests/bar_parity.py    --ref ~/sandbox/marvel/knull/live_bar.py \
                                  --driver ./build-linux/bar_parity_driver
python3.11 tests/feature_parity.py --ref-bar ~/sandbox/marvel/knull/live_bar.py \
    --ref-feat ~/sandbox/dc/mjolnir_pkg/src/mjolnir/core/features.py \
    --driver ./build-linux/feature_parity_driver
python3.11 tests/regime_parity.py --ref-bar ~/sandbox/marvel/knull/live_bar.py \
    --ref-feat ~/sandbox/dc/mjolnir_pkg/src/mjolnir/core/features.py \
    --pkg-src ~/sandbox/dc/mjolnir_pkg/src \
    --driver ./build-linux/regime_parity_driver
```

## Status — M1

Scaffold only. `core_impl.cpp` wires the contract and returns a real (non-stub)
build tag; the four IP modules above are **not yet implemented**. The M1 exit
gate is variant-A parity (corr 1.000 vs live `y_pred` on the bot's own dumped
bars) — see the port plan's parity section. Do not wire to any executor before
that gate passes; M1 is shadow-only.
