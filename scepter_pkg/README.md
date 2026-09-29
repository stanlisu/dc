# Scepter

Cross-symbol anchor features on top of Orb. Each prediction symbol's row gets columns describing an **anchor** symbol (BTC/ETH for crypto, QQQ for the adamantium equities arms). The regime stack crosses each own-state regime with an anchor-state regime, e.g. `rsi_oversold_and_btc_trending_up`.

Research only. There is no `ScepterTrading`, and the live loaders (marvel `symbiote/common.py::LOADABLE_STRATEGIES`, `knull/orb_bridge.py`) refuse `STRATEGY=scepter`. Design history: [2026-03-31-scepter-design.md](2026-03-31-scepter-design.md), [2026-03-31-scepter-implementation.md](2026-03-31-scepter-implementation.md).

## Architecture

```
ScepterResearch (research/backtesting)
  +-- OrbResearch (cross-TF alignment, TF-prefixed filter routing)
      +-- AgamottoResearch (single-TF base)
```

`ScepterResearch` overrides four things. Everything else, including `load`, `engineer_features`, `create` and `filter_signals`, is Orb's:

| Method | What it adds |
|--------|--------------|
| `__init__` | Appends `ANCHOR_SYMBOLS` to `SYMBOLS` so the anchors are loaded and engineered. Normalises `ANCHOR_REGIMES` keys from codes to real names. |
| `verticalize` | Verticalizes the non-anchor symbols only (anchors are never prediction targets), then calls `_attach_anchor_features`. |
| `_apply_filter_mask` | Resolves an `ANCHOR_REGIMES` part from its config entry before falling through to Orb. Handles `_and_` / `_or_` compounds and coded names. |
| `generate_regime_stack` (classmethod) | The coded own-state × anchor-state stack. Called by marvel `gauntlet/generate_scepter_regimes.py`. |

## Key Files

| File | Description |
|------|-------------|
| `src/scepter/research.py` | `ScepterResearch(OrbResearch)`, the regime atoms (`_SCEPTER_OWN_STATE`, `_ANCHOR_STATE_TEMPLATE`) and the anchor features |
| `tests/test_regime_stack.py` | Regime-stack direction, anchor crossing, `anchor_prefix` generalisation |
| `tests/test_anchor_regime_mask.py` | `ANCHOR_REGIMES` masking: fails loud on a missing column, handles compounds and coded names |

The runner (`gauntlet/run_scepter_research.py`), the arm configs and the marvel-side tests (`scepter/tests/`) live in **marvel**, not here.

## Anchor features

The column prefix is `anchor_native[:3].lower()`: `BINANCE_PERP_BTC_USDT` → `btc`, `QQQ` → `qqq`. Two anchors whose native names share their first 3 characters would collide.

| Column | Built from | Notes |
|--------|-----------|-------|
| `{p}_ret_lag1..3` | anchor lagged returns | per timestamp |
| `{p}_atr_ratio` | ATR / rolling-mean ATR over `max(ANCHOR_WINDOWS)` | per timestamp |
| `{p}_close_vs_ma` | `close − mvg1` | per timestamp; used by the trending_up/down regimes |
| `{p}_corr_{w}` | rolling corr of symbol vs anchor returns, per `w` in `ANCHOR_WINDOWS` | per symbol |
| `{p}_spread` | `symbol_close − β·anchor_close`, with β from rolling OLS | per symbol |
| `{p}_rel_strength` | symbol minus anchor cumulative return over `min(ANCHOR_WINDOWS)` | per symbol |

**Causality:** `{tf}_{sym}_return` is the forward return (the target). The correlation and relative-strength features `.shift(1)` it back to the historical return. Without that shift they leak the target: in the 2026-06-13 leak, `corr(rel_strength, y_true)` was 0.24 and Sharpe reached 11.

## Configuration

Scepter keys, in addition to the Orb/Agamotto `setting.json` keys:

| Key | Example | Description |
|-----|---------|-------------|
| `STRATEGY` | `"scepter"` | |
| `ANCHOR_SYMBOLS` | `["QQQ"]` | Required, and raises if absent. The first entry sets the regime-stack prefix in the runner. |
| `ANCHOR_WINDOWS` | `[14, 28]` | Correlation windows. `min` sets the relative-strength window, `max` the spread and ATR window. |
| `ANCHOR_REGIMES` | see below | Maps each anchor-state regime to a column condition. Keys may be codes or real names. |

```json
"ANCHOR_REGIMES": {
  "r080": {"col": "qqq_close_vs_ma", "op": ">", "val": 0.0},
  "r079": {"col": "qqq_close_vs_ma", "op": "<", "val": 0.0},
  "r077": {"col": "qqq_atr_ratio",   "op": ">", "val": 1.2},
  "r078": {"col": "qqq_atr_ratio",   "op": "<", "val": 0.8}
}
```

`op` must be one of `> < >= <= ==`. A NaN in the column counts as False.

## Gotchas

- **`ANCHOR_REGIMES` must match the arm's anchor.** Every `col` has to carry the prefix built from this arm's `ANCHOR_SYMBOLS`. If you copy a BTC arm's block onto a QQQ arm, `_apply_filter_mask` raises a `ValueError` that names the regime, the column and the prefixes the anchors actually built. Until 2026-09-29 it returned an all-True mask instead, so the anchor leg vanished silently.
- **That error is a plain `ValueError` on purpose.** `create()` catches `MissingFilterColumnError` and skips the regime. Since every scepter regime has an anchor leg, that type would skip the whole stack and the run would still exit 0.
- **Missing anchor klines surface at the mask, not at feature build.** `_attach_anchor_features` logs a warning and skips an anchor it cannot map, or one whose return column is absent. The first `ANCHOR_REGIMES` lookup then raises.
- **Onboarding a new anchor prefix** requires adding its 4 atoms (`{p}_trending_up/_down`, `{p}_high_vol/_low_vol`) to `obfuscation/extract_inventory.py::_MARVEL_ATOMS`. Otherwise `generate_regime_stack(anchor_prefix=...)` raises on the unmapped atom.
- **`ANCHOR_WINDOWS` silently defaults** to `[14, 28]` when absent (`config.get`). Every current arm sets it explicitly.

## Usage

```bash
# Research: run on shield2, never on the Mac (marvel CLAUDE.md)
PYTHONPATH="agamotto_pkg/src:." python gauntlet/run_scepter_research.py \
  -c adamantium/pred_scepter.base.1d.semiconductor/setting.json [--start-date YYYY-MM-DD]

# Unit tests (from the dc root; packages need not be installed)
PYTHONPATH=agamotto_pkg/src:orb_pkg/src:scepter_pkg/src:. pytest scepter_pkg/tests -q

# Build / deploy: see the top-level README
./build_distribution.sh --build-only scepter
```
