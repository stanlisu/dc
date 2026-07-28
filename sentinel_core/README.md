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

## Status — M1

Scaffold only. `core_impl.cpp` wires the contract and returns a real (non-stub)
build tag; the four IP modules above are **not yet implemented**. The M1 exit
gate is variant-A parity (corr 1.000 vs live `y_pred` on the bot's own dumped
bars) — see the port plan's parity section. Do not wire to any executor before
that gate passes; M1 is shadow-only.
