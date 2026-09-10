"""Kline ladder sizing — the size multiplier the training target is scaled by.

Shared by every kline research engine: AgamottoResearch and, through it,
OrbResearch / ScepterResearch / AetherResearch. Lives in agamotto_pkg because
orb_pkg already depends on it (`orb/research.py:12`), and because the dc packages
ship standalone and must NOT import marvel's `gauntlet` at runtime.

WHAT THE MULTIPLIER MODELS. `knull`'s LadderEngine opens a position at the
signal bar and stacks additional rungs as price moves against it:

  * rung 1 fills at entry and needs no adverse move — `knull/ladder.py:165-172`
    sets `_level = 1` as soon as `filled_qty >= rung_qty`. In LadderMaker (the
    kline mode) the entry rests post-only but chases the book
    (`knull/execution_style.py:185-196`, `should_reprice_entry() -> True`), and
    Taker mode fills outright, so the base rung is real either way.
  * rungs 2..LADDER each need a further `LADDER_BPS` against entry
    (`knull/ladder.py:186,200-211`). `ladder_max = self._LADDER` and levels run
    1..ladder_max inclusive (`:157,183`), so LADDER is the TOTAL rung count and
    only LADDER-1 rungs are reachable by moving.

Hence `multiplier = 1 + clip(floor(adverse_excursion / step), 0, LADDER-1)`,
which lives in [1, LADDER].

WHY NOT `min(long_layers, short_layers)`. That was the previous kline sizing. It
required a favourable excursion as well, which nothing in the executor does — the
exit is the next non-BUY decision, chased into the book with no give-up
(`knull/base_executor.py:3890-3891,4166-4172`). It zeroed exactly the
dip-and-keep-falling bars, whose losses are real and are realized live. See
`agamotto_pkg/tests/test_kline_ladder_sizing.py` for the measured damage.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# The MEANING of the kline target this package computes. A weights window is
# stamped with the convention it trained under and every launcher refuses a
# mismatch (marvel `gauntlet/target_convention.py`, xmen
# `scripts/verify_weights_convention.py`). Bump it whenever the target's meaning
# changes, so no pre-change window can be launched against the new gate.
#   unsigned-v1          dc #23: return_{long,short}[_raw] are UNSIGNED market returns.
#   ladder-avg-entry-v2  2026-09-10: each rung priced from ITS OWN entry
#                        (`compute_ladder_return`), not `price_return * rungs`.
TARGET_CONVENTION = "ladder-avg-entry-v2"


def ladder_params(config: Dict) -> Tuple[int, int, float]:
    """Read (LADDER_LONG, LADDER_SHORT, LADDER_BPS) from a setting.json dict.

    Per-leg since 2026-08-06. Measured on 1.7-6.4M rows, direction-only IC
    (features vs the PLAIN return, so uncontaminated by the size term) moves in
    OPPOSITE directions with ladder depth on the two legs:

        top-16 |IC|      LADDER=1   LADDER=10   LADDER=20
        15m long           0.0293     0.0245      0.0220    falling
        15m short          0.0490     0.0563      0.0638    rising

    and comparing all bars against cap-size bars shows why: the multiplier
    upweights LESS predictable bars on longs (15m 0.0293 -> 0.0261) and MORE
    predictable ones on shorts (0.0490 -> 0.0580). One shared value is therefore
    wrong for one leg whichever number is chosen.

    Resolution order per leg, ending in a fail-fast (CLAUDE.md):
        LADDER_LONG / LADDER_SHORT  ->  LADDER  ->  KeyError

    WARNING — target/executor divergence. `LADDER` is also what the EXECUTOR
    fills (`knull/ladder.py:157`, `ladder_max = self._LADDER`). Setting a per-leg
    TARGET ladder that differs from it trains on rungs that will never fill,
    which is structurally the same defect as the `min()` gate this module
    replaced. These keys are for research arms; reconcile them with the executor
    before anything built on them is deployed.

    Raises:
        KeyError: if LADDER_BPS is missing, or if a leg has neither its own key
            nor the shared LADDER.
    """
    if config.get("LADDER_BPS") is None:
        raise KeyError(
            "LADDER_BPS is required in setting.json (adverse move between rungs, "
            "in bps; matches knull/ladder.py:186 bps_frac = LADDER_BPS * 1e-4) — "
            "no default, see CLAUDE.md 'no silent fallbacks'")

    def _leg(key: str) -> int:
        val = config.get(key)
        if val is None:
            # DEPRECATED: drop after 2027-01-01. 159 kline settings carry only
            # the shared LADDER; breaking them all at once to force per-leg keys
            # would strand every experiment. Chain ends in the raise below.
            val = config.get("LADDER")
        if val is None:
            raise KeyError(
                f"{key} (or the shared LADDER) is required in setting.json — "
                "total ladder rungs for this leg, including the entry rung. "
                "No default, see CLAUDE.md 'no silent fallbacks'")
        return int(val)

    return _leg("LADDER_LONG"), _leg("LADDER_SHORT"), float(config["LADDER_BPS"])


def compute_ladder_multiplier(close, adverse_extreme, ladder: int,
                              step_bps: float) -> pd.Series:
    """Number of rungs filled, in [1, ladder], for ONE position direction.

    The two legs are independent: neither gates the other.

    Args:
        close: Series of close prices at the signal bar.
        adverse_extreme: the forward price extreme in the ADVERSE direction —
            next-bar low for a long. For a short, mirror it before calling
            (`2*close - high_next`) so the distance is still measured downward.
        ladder: TOTAL rung count including the entry rung (setting.json LADDER).
        step_bps: adverse move between rungs, in bps (setting.json LADDER_BPS).

    Returns:
        Integer Series in [1, ladder], indexed like `close`.
    """
    step_size = float(step_bps) * 1e-4
    # Only ladder-1 rungs are reachable by moving; rung 1 fills at entry.
    # max(...,0) keeps LADDER=0/1 meaning "entry rung only" rather than a
    # negative clip bound.
    max_extra = max(int(ladder) - 1, 0)
    close_s = pd.Series(close).replace(0, np.nan)
    other = pd.Series(adverse_extreme)
    # clip(lower=0): a FAVOURABLE excursion fills no extra rung and must never
    # push the multiplier below the base rung.
    distance = ((close_s - other) / close_s).clip(lower=0).replace(
        [np.inf, -np.inf], np.nan)
    # +1e-9 so an exact k-step excursion counts k rungs despite binary float
    # representation (e.g. 2bp / 1bp landing on 1.9999999).
    layers = np.floor(distance / step_size + 1e-9).clip(
        0, max_extra).fillna(0).astype(int)
    return pd.Series(1 + layers, index=close_s.index)


def compute_ladder_return(price_return, rungs, step_bps: float,
                          side: str) -> pd.Series:
    """Return earned by a k-rung ladder, each rung priced from ITS OWN entry.

    The rung COUNT comes from `compute_ladder_multiplier`; this is the
    aggregation over those rungs. Before 2026-09-10 the target was
    `price_return * k`, which books every rung at the anchor — a k-rung stack
    that fills `(k-1)*LADDER_BPS` against the anchor is worth strictly more
    than that (long: cheaper entries; short: dearer ones), and inside the
    `(k-1)*step` band the SIGN of the label differs. On the live 15m arm
    (LADDER=2) it read as exactly `2 x close-to-close return`.

        long   sum_{j=1..k} (exit/ent_j - 1),   ent_j = a * (1 - (j-1)*step)
        short  sum_{j=1..k} (exit/ent_j - 1),   ent_j = a * (1 + (j-1)*step)

    Both columns are UNSIGNED market returns from each rung's entry (the
    `unsigned-v1` convention, dc #23): the consumer applies the trade
    direction, so the short column is what it NEGATES. Rung 1 contributes
    `price_return` itself, exactly. Fees are NOT charged here — the callers
    keep charging them per rung, as before.

    Args:
        price_return: `exit/anchor - 1` per bar (NaN where there is no exit
            bar; NaN stays NaN — never a silent 0.0 label).
        rungs: integer rung count per bar, in [1, LADDER]; same index.
        step_bps: adverse move between rungs, in bps (LADDER_BPS), > 0.
        side: "long" or "short" — which way rung j>=2 is priced off the anchor.

    Raises:
        ValueError: on an unknown side, a non-positive step, a rung count
            below 1 or non-integer, or misaligned indexes. Never realigns.
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    step = float(step_bps) * 1e-4
    if not step > 0:
        raise ValueError(f"step_bps must be > 0, got {step_bps!r}")
    r = pd.Series(price_return, dtype=float)
    k = pd.Series(rungs)
    if not r.index.equals(k.index):
        raise ValueError("price_return and rungs must share the same index — "
                         "refusing to realign silently")
    kv = k.to_numpy(dtype=float)
    if np.isnan(kv).any() or (kv < 1).any() or (kv != np.floor(kv)).any():
        raise ValueError("rungs must be integers >= 1 (rung 1 is the entry rung); "
                         f"got {sorted(set(kv.tolist()))[:5]}")
    kv = kv.astype(int)
    # Long rungs fill BELOW the anchor, short rungs ABOVE it.
    adverse = -1.0 if side == "long" else 1.0
    rv = r.to_numpy()
    out = rv.copy()                       # rung 1: the anchor return, exactly
    for j in range(2, int(kv.max()) + 1):
        filled = kv >= j
        entry_factor = 1.0 + adverse * (j - 1) * step
        out[filled] += (1.0 + rv[filled]) / entry_factor - 1.0
    return pd.Series(out, index=r.index)
