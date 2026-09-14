"""Ladder-adjusted return computation for MjolnirResearch.

Standalone function — takes a config dict as first arg instead of reading
self.config.

Extracted from research.py to keep each module under ~700 lines for
PyArmor trial compatibility.

SOURCE OF TRUTH for the sizing (`compute_ladder_multiplier`) and per-rung
pricing (`compute_ladder_return`) maths is `agamotto.ladder` (dc `b696288`,
`6b17fa3`). mjolnir keeps byte-equivalent COPIES rather than importing them:
xmen vendors ONLY `mjolnir_pkg` (`build_distribution.sh:43,76-84`) and
launches with `PYTHONPATH="$XROOT:$XROOT/mjolnir_pkg/src"`
(`xmen/launch_xmen_bot.sh:45`) — there is no `agamotto_pkg` in the xmen
checkout, and `mjolnir/trading.py` imports `core.research`, which imports
this module, so a cross-package import would kill the live tick bot at boot.
`test_tick_ladder_avg_entry.py::test_parity_with_agamotto_*` is the anti-drift
guard: if it fails, the copies have diverged and one of them is wrong.

THE TWO FIXES THIS FILE CARRIES (2026-09-14), both applying to mjolnir/
stormbreaker tick targets what agamotto/orb already have for kline:

  1. BASE RUNG (was dc #79/`0744ea0`, 2026-08-08, never merged). Sizing used
     to be `floor(adverse / step).clip(0, LADDER)` — "no free base rung", so
     a bar whose adverse excursion was under one step got size 0, i.e. was
     labelled as a position that never opened. Nothing in the executor works
     that way: rung 1 fills AT ENTRY with no adverse move needed
     (`knull/ladder.py:170`), so the discarded bars were exactly the ones
     where the trade was immediately right. Fixed by `compute_ladder_multiplier`
     below: size is now `1 + clip(floor(adverse/step), 0, LADDER-1)`, in
     [1, LADDER].
  2. PER-RUNG ENTRY PRICING (was dc `6b17fa3`/PR #74 for kline ONLY,
     2026-09-10; that commit explicitly flagged
     "mjolnir_pkg core/ladder.py:287-290 still books `ret * size` — tick
     twin, separate ticket" and left it unfixed). The target used to book
     EVERY rung at the anchor price (`price_return * k`) — a k-rung stack
     that fills `(j-1)*LADDER_BPS` against the anchor is worth strictly more
     than that (long: cheaper entries; short: dearer ones), and inside the
     `(k-1)*step` band the SIGN of the label differs from the anchor-priced
     one. Fixed by `compute_ladder_return` below: each rung is priced from
     ITS OWN entry and summed.

BOTH fixes change the target column — this is a repeat of PR #74's own
warning, now for tick: **every cached mjolnir/stormbreaker tick filter
parquet and every model/threshold/IC/daily-PnL/Sharpe derived from one is
invalidated.** No tick result is trustworthy until the filters are
regenerated and the arms retrained. Requires a package rebuild before marvel
or xmen sees any of it.

NOT TOUCHED, and deliberately so: `size_short = high_layers` being
correlated with the realized return (TODO.md P4 in marvel, "the short
drought, chased since May 2026... diagnosed, formula unfixed"). That is a
different, still-open research question — no committed fix exists for it —
and bundling a guess at one into this change would make the two impossible
to evaluate independently. `high_layers`/`low_layers` are still the sizing
inputs; only the floor (fix 1) and the pricing (fix 2) changed.

The MEANING of the tick target this module computes. A weights window
should be stamped with the convention it trained under so a pre-fix window
can be told apart from a post-fix one; nothing currently reads this stamp on
the mjolnir/xmen launch path (unlike agamotto's `gauntlet/target_convention.py`
/ xmen `scripts/verify_weights_convention.py` gates), so wiring enforcement is
a separate, cross-repo follow-up. Declaring it here regardless, matching
`agamotto.ladder.TARGET_CONVENTION`:
    unsigned-v1          dc #23: return_{long,short}[_raw] are UNSIGNED
                         market returns.
    ladder-avg-entry-v2  2026-09-14: base rung + each rung priced from ITS
                         OWN entry (this file), not `price_return * rungs`.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

VALID_FILL_MODES = ("ladder", "flat", "limit_then_taker")

TARGET_CONVENTION = "ladder-avg-entry-v2"


def resolve_fill_mode(config: Dict) -> str:
    """Read the REQUIRED ``LADDER_FILL_MODE`` key. No default.

    Single source of truth for the key. It used to be read as
    ``config.get("LADDER_FILL_MODE", "ladder")`` at THREE independent
    sites (``ladder.py``, ``research.py::verticalize``,
    ``streaming.py::stream_research``); an arm whose ``setting.json``
    omitted the key therefore got a silently DIFFERENT TARGET from one
    that set it — ``mjolnir.base.5s_1`` (omitted -> "ladder") vs
    ``mjolnir.base.30s_1`` ("limit_then_taker") — with nothing in the
    logs to say so. Magic-value defaults on required numeric/string
    config are banned by CLAUDE.md's no-silent-fallback rule, so a
    missing key now raises.

    Args:
        config: Dictionary loaded from setting.json.

    Returns:
        The lower-cased fill mode, guaranteed to be in VALID_FILL_MODES.

    Raises:
        KeyError:   LADDER_FILL_MODE absent from config.
        ValueError: LADDER_FILL_MODE present but not a known mode.
    """
    if "LADDER_FILL_MODE" not in config:
        raise KeyError(
            "LADDER_FILL_MODE is required in setting.json and has NO default "
            f"(VERSION={config.get('VERSION', '<unset>')!r}). It selects the "
            "target construction, so defaulting it silently changes what the "
            f"model is trained on. Set it explicitly to one of "
            f"{VALID_FILL_MODES}.")
    mode = str(config["LADDER_FILL_MODE"]).lower()
    if mode not in VALID_FILL_MODES:
        raise ValueError(
            "LADDER_FILL_MODE must be 'ladder', 'flat' or 'limit_then_taker', "
            f"got {mode!r}")
    return mode


def resolve_ladder(config: Dict) -> int:
    """Read the REQUIRED ``LADDER`` key. No default.

    Single source of truth for the key on the mjolnir/stormbreaker target
    path. It used to be read as ``int(config.get("LADDER", 1) or 0)`` —
    a magic-number default AND the banned ``get(K, X) or Y`` idiom in one
    expression (CLAUDE.md "NEVER add silent fallbacks"), sitting in the
    same function as the ``LADDER_FILL_MODE`` default that was removed in
    ``b5e71bd``. Two distinct silent paths:

      * key ABSENT -> ``1``. All six ``pred_stormbreaker.base.*_1`` arms
        omitted ``LADDER`` entirely and were therefore trained on a
        1-rung target while every ``pred_mjolnir.base.*_1`` arm set the
        key explicitly (one of them at ``10``) — a different target, with
        nothing in the logs saying so.
      * key present but FALSY-non-zero (``null``, ``""``, ``false``) ->
        ``0`` via the ``or``, i.e. an unusable rung cap accepted in
        silence instead of a type error. (``LADDER: 0`` itself survived
        the ``or`` unchanged, because the right operand was also ``0`` —
        the idiom was still banned, and still hid the other two cases.)

    ``LADDER`` is the TOTAL rung count (including the entry rung) the
    TARGET the model is trained on is capped at (:func:`compute_ladder_multiplier`
    clips extra layers to ``LADDER - 1``), so defaulting it silently changes
    what is being learned. ``LADDER: 0`` and ``LADDER: 1`` both mean "entry
    rung only" (size 1) — neither means "no position" (2026-09-14; see
    ``compute_ladder_multiplier``).

    Args:
        config: Dictionary loaded from setting.json.

    Returns:
        The rung cap as a non-negative int.

    Raises:
        KeyError:   LADDER absent from config.
        ValueError: LADDER present but not a whole, non-negative number.
    """
    if "LADDER" not in config:
        raise KeyError(
            "LADDER is required in setting.json and has NO default "
            f"(VERSION={config.get('VERSION', '<unset>')!r}). It caps the rungs "
            "of the ladder TARGET, so defaulting it silently changes what the "
            "model is trained on. Set it explicitly to a whole number >= 0.")
    value = config["LADDER"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "LADDER must be a whole number of rungs, got "
            f"{value!r} of type {type(value).__name__}.")
    if float(value) != int(value):
        raise ValueError(
            f"LADDER must be a WHOLE number of rungs, got {value!r}.")
    ladder = int(value)
    if ladder < 0:
        raise ValueError(f"LADDER must be >= 0 rungs, got {ladder}.")
    return ladder


def _venue_blocks(config: Dict):
    """Yield (label, block dict) for every venue-override block a live
    knull bot could resolve LADDER_BPS from.

    Mirrors marvel `knull/venue_config.py`'s two resolution paths:
    `merge_venue_config` overlays `EXECUTORS.<venue>` and the overlay WINS
    (`:107-109`), and `_select_venue_block` (`:157-158`) falls back to a
    legacy single-venue `executor` block when `EXECUTORS` is absent.
    `pred_mjolnir.base.{5s,15s}_1` carry `LADDER_BPS` under the legacy key,
    so both must be checked or those two arms pass vacuously.
    """
    executors = config.get("EXECUTORS")
    if isinstance(executors, dict):
        for venue, block in executors.items():
            if isinstance(block, dict):
                yield f"EXECUTORS.{venue}", block
    legacy = config.get("executor")
    if isinstance(legacy, dict):
        yield "executor", legacy


def resolve_ladder_bps(config: Dict) -> float:
    """Read the REQUIRED ``LADDER_BPS`` key (adverse move between rungs, in
    bps). No default, and no default is possible: this used to be the module
    constant ``LADDER_STEP_BPS = 1.0``, so the target's rung spacing and the
    live executor's (``knull/ladder.py:186`` ``bps_frac = LADDER_BPS * 1e-4``)
    were merely both UNDECLARED rather than verified equal — an arm running
    non-1bp executor spacing would have trained on rungs that never fill,
    the same shape of defect as the missing base rung.

    Also checks every venue-override block (see :func:`_venue_blocks`)
    agrees with the top-level value — a per-venue ``LADDER_BPS`` that
    differs IS the live rung spacing for that venue (the overlay wins), so a
    silent mismatch there would train the target at a spacing the executor
    on that venue never fills at.

    Args:
        config: Dictionary loaded from setting.json.

    Returns:
        The rung spacing in bps, as a positive finite float.

    Raises:
        KeyError:   LADDER_BPS absent from config.
        ValueError: LADDER_BPS present but not a positive finite number, or
            a venue-override block disagrees with the top-level value.
    """
    if "LADDER_BPS" not in config:
        raise KeyError(
            "LADDER_BPS is required in setting.json (adverse move between "
            "rungs, in bps; matches knull/ladder.py:186 "
            "bps_frac = LADDER_BPS * 1e-4) — no default, see CLAUDE.md "
            "'no silent fallbacks'.")
    raw = config["LADDER_BPS"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"LADDER_BPS must be a number, got {raw!r} of type "
            f"{type(raw).__name__}.")
    value = float(raw)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"LADDER_BPS must be a positive finite number, got {raw!r}. A "
            "zero or non-finite step makes distance/step_size infinite, "
            "which the clip then renders as 'every bar at the rung cap' — "
            "an out-of-range result silently made plausible.")
    for label, block in _venue_blocks(config):
        if "LADDER_BPS" not in block:
            continue
        block_value = float(block["LADDER_BPS"])
        if block_value != value:
            raise ValueError(
                f"{label}.LADDER_BPS={block_value!r} disagrees with the "
                f"top-level LADDER_BPS={value!r}. The venue overlay wins at "
                "boot (knull/venue_config.py:107-109), so this IS the live "
                "rung spacing for that venue — the target would be trained "
                "at a spacing the executor never fills at.")
    return value


def compute_ladder_multiplier(close, adverse_extreme, ladder: int,
                              step_bps: float) -> pd.Series:
    """Number of rungs filled, in [1, ladder], for ONE position direction.

    Byte-equivalent copy of ``agamotto.ladder.compute_ladder_multiplier`` —
    see the module docstring for why this is a copy, not an import. The two
    legs are independent: neither gates the other.

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

    Byte-equivalent copy of ``agamotto.ladder.compute_ladder_return`` — see
    the module docstring for why this is a copy, not an import.

    The rung COUNT comes from `compute_ladder_multiplier`; this is the
    aggregation over those rungs. Before 2026-09-14 the tick target was
    `price_return * k`, which books every rung at the anchor — a k-rung stack
    that fills `(k-1)*LADDER_BPS` against the anchor is worth strictly more
    than that (long: cheaper entries; short: dearer ones), and inside the
    `(k-1)*step` band the SIGN of the label differs.

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


def compute_ladder_returns(
    config: Dict,
    df: pd.DataFrame,
    close_col: str,
    low_col: str,
    high_col: str,
    horizon_bars: int = 1,
) -> pd.DataFrame:
    """Compute ladder-adjusted return columns for a single symbol.

    Mirrors `agamotto.research.AgamottoResearch._compute_ladder_returns`
    (same sizing via `compute_ladder_multiplier`, same per-rung pricing via
    `compute_ladder_return`, same column names), generalized to a variable
    forward horizon so the fill window matches the prediction window.

    At horizon_bars == 1 the sizing/pricing is identical to agamotto's
    (`test_parity_with_agamotto_full_target_at_horizon_one`), preserving
    native-mode parity (TIME_UNIT == bar resolution, e.g. mjolnir.base.5s_1).
    For boundary-aligned experiments (e.g. mjolnir.base.30s_1: 5s bars
    predicting the next 30s boundary close) callers should pass
    horizon_bars = TIME_UNIT_seconds / bar_tf_seconds so the low/high
    lookahead spans the full prediction horizon, instead of only the
    next single bar (which under-counted ladder fills and produced
    a frictionless close-to-close target).

    Args:
        config:       Dictionary loaded from setting.json.
        df:           DataFrame with at least close_col, low_col, high_col.
        close_col:    Name of the close price column.
        low_col:      Name of the low price column.
        high_col:     Name of the high price column.
        horizon_bars: Forward horizon used both for the price return
                      (close[t+h] / close[t] - 1) and for the
                      low/high min/max lookahead window. Must be >= 1.

    Returns:
        DataFrame with columns: return_long, return_short,
        return_long_raw, return_short_raw.
    """
    if horizon_bars < 1:
        raise ValueError(
            f"horizon_bars must be >= 1, got {horizon_bars}")

    ladder = resolve_ladder(config)
    step_bps = resolve_ladder_bps(config)
    # FEE is required (no fallback): see commit 0500d8fa for rationale.
    # The historical `or 0.0` collapsed any falsy FEE to the default.
    fee_rate = float(config["FEE"]) / 10000.0

    close = df[close_col]
    low_series = df[low_col]
    high_series = df[high_col]
    close_safe = close.replace(0, np.nan)

    # Forward h-bar price return: close[t+h] / close[t] - 1. Off close_safe,
    # not close, so a literal zero close yields NaN rather than +/-inf — see
    # test_zero_close_is_nan_not_inf. Before the base-rung fix this was masked
    # by accident (the old sizing floored to 0 rungs there, and inf * 0 was
    # NaN); with size >= 1 always, an inf computed off raw close would now
    # survive into the label.
    price_return = close_safe.pct_change(
        horizon_bars, fill_method=None).shift(-horizon_bars)

    # Forward-rolling min low / max high over (t, t+horizon_bars]:
    # shift(-1) so position t holds low[t+1], then reverse-rolling so
    # the window covers the next horizon_bars bars (exclusive of t).
    # At horizon_bars=1 this reduces to low.shift(-1) / high.shift(-1).
    low_shifted = low_series.shift(-1)
    high_shifted = high_series.shift(-1)
    low_window = (
        low_shifted.iloc[::-1]
        .rolling(horizon_bars, min_periods=1)
        .min()
        .iloc[::-1]
    )
    high_window = (
        high_shifted.iloc[::-1]
        .rolling(horizon_bars, min_periods=1)
        .max()
        .iloc[::-1]
    )

    # Rung count from each side's OWN forward excursion, capped at LADDER —
    # neither side gates the other. size_long from the forward LOW (long
    # entries dig down); size_short from the forward HIGH, mirrored about
    # close so the same downward-measuring helper serves both legs.
    size_long = compute_ladder_multiplier(close_safe, low_window, ladder, step_bps)
    size_short = compute_ladder_multiplier(
        close_safe, 2.0 * close_safe - high_window, ladder, step_bps)

    # Per-unit UNSIGNED price return on each side's own exit path, BEFORE fee and
    # size. Default ("ladder") marks the whole opened stack at the horizon close
    # (close-to-close), so both sides see the same close-to-close move; fill_mode
    # may override either the size (flat) or the exit price (limit_then_taker),
    # which is why the two are tracked separately.
    #
    # UNSIGNED, i.e. the MARKET move, NOT the trade's P&L: a short is profitable
    # when this is NEGATIVE. This matches agamotto (agamotto/research.py:379-380,
    # `return_{long,short}_raw = price_return * size`, now per-rung via
    # compute_ladder_return) so the shared marvel PnL engine — which books
    # `signal * y_true_raw`, with signal = -1 for a short — is correct for BOTH
    # algos. Until 2026-07-29 mjolnir negated the short here, making its target
    # position-SIGNED while agamotto's stayed unsigned under the SAME column
    # names; the engine then signed mjolnir's a second time and every short leg
    # booked +price_return. See tasks/lessons.md 2026-07-29.
    ret_long_px = price_return
    ret_short_px = price_return

    fill_mode = resolve_fill_mode(config)
    if fill_mode == "flat":
        # Two-way TAKER model — fixed size 1 per bar, filled at the decision
        # (horizon-close) price, no laddered size and no maker rungs. This is
        # what aggressive taker execution actually realizes (precise fill
        # PRICE, fixed SIZE), as opposed to the laddered maker accumulation.
        # A constant rung count of 1 makes compute_ladder_return an identity
        # (the loop over rungs 2..k never runs), so this is unaffected by the
        # per-rung pricing fix.
        size_long = pd.Series(1, index=price_return.index)
        size_short = pd.Series(1, index=price_return.index)
    elif fill_mode == "limit_then_taker":
        # Two-stage maker-close-then-taker-fallback exit (Stan 2026-06-22).
        # Open n on the entry-side penetration (size_long/size_short as
        # above), then try to close the WHOLE position with a single limit at
        # the next-boundary close close[t+h]. Decide the fill from the
        # FOLLOWING horizon window (t+h, t+2h]:
        #   LONG  (sell limit): fills if that window rallies back up to
        #          close_h (max high >= close_h) -> exit at close_h; else
        #          taker-close the leftover at close[t+2h].
        #   SHORT (buy limit):  fills if that window dips to close_h
        #          (min low <= close_h) -> exit at close_h; else taker-close
        #          the leftover at close[t+2h].
        # Unfilled ("leftover") inventory therefore books the real later move
        # at close[t+2h], not the optimistic single horizon close. Both
        # branches are charged the taker FEE (a deliberately strict
        # assumption). Requires the full t+2 window: the final 2h bars per
        # symbol have no close[t+2h] and are masked to NaN (dropped in
        # verticalize, which under this mode also gates on return_long/short).
        #
        # The rung SIZE is still the entry-side penetration (unchanged); only
        # the EXIT price changes, and each rung is still priced from its own
        # entry against this same exit via compute_ladder_return below.
        close_h = close_safe.shift(-horizon_bars)
        close_2h = close_safe.shift(-2 * horizon_bars)
        high_w2 = high_window.shift(-horizon_bars)   # max high over (t+h, t+2h]
        low_w2 = low_window.shift(-horizon_bars)      # min low over (t+h, t+2h]
        full = close_2h.notna()                       # full t+2 window observed
        exit_long = close_h.where(high_w2 >= close_h, close_2h)
        exit_short = close_h.where(low_w2 <= close_h, close_2h)
        # Both UNSIGNED (see ret_{long,short}_px above): each side's own realized
        # price move on its own exit path. The short is NOT negated here — the
        # consumer applies the direction.
        ret_long_px = (exit_long / close_safe - 1.0).where(full)
        ret_short_px = (exit_short / close_safe - 1.0).where(full)
    # No trailing `elif fill_mode != "ladder": raise` — resolve_fill_mode()
    # already rejected anything outside VALID_FILL_MODES, so reaching here
    # means fill_mode == "ladder" (the close-to-close mark, set above).

    # Each rung is priced from ITS OWN entry (rung j fills (j-1)*LADDER_BPS
    # against the anchor), not `price_return * rungs`, which booked every rung
    # at the anchor — see compute_ladder_return. The fee stays per rung.
    fee_cost = fee_rate * 2.0
    long_raw = compute_ladder_return(ret_long_px, size_long, step_bps, "long")
    short_raw = compute_ladder_return(ret_short_px, size_short, step_bps, "short")

    return_long = (long_raw - fee_cost * size_long).rename("return_long")
    return_short = (short_raw + fee_cost * size_short).rename("return_short")
    return_long_raw = long_raw.rename("return_long_raw")
    return_short_raw = short_raw.rename("return_short_raw")

    return pd.concat(
        [return_long, return_short, return_long_raw, return_short_raw], axis=1)
