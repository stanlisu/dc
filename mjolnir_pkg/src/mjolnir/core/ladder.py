"""Ladder-adjusted return computation for MjolnirResearch.

Standalone function — takes a config dict as first arg instead of reading
self.config.

Extracted from research.py to keep each module under ~700 lines for
PyArmor trial compatibility.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

VALID_FILL_MODES = ("ladder", "flat", "limit_then_taker")


def resolve_ladder_bps(config: Dict) -> float:
    """Read the REQUIRED top-level ``LADDER_BPS`` key. No default.

    The adverse move between ladder rungs, in bps — the TARGET-side twin of the
    executor's ``bps_frac = LADDER_BPS * 1e-4`` (``knull/ladder.py:190``). It is
    the *between-ladder* adverse trigger measured against average cost, NOT the
    within-ladder rung spacing (which is 1 tick); see
    ``.claude/memory/reference_ladder_bps_trigger.md``.

    Until 2026-08-08 this was the module constant ``LADDER_STEP_BPS = 1.0``,
    while ``LADDER``, ``FEE`` and ``LADDER_FILL_MODE`` in the very same function
    were already required-no-default config reads. The justification was that no
    tick ``setting.json`` declared the key, so there was nothing to read — true,
    but it left the executor's rung spacing and the target's merely both
    UNDECLARED rather than verified equal. An arm running non-1bp spacing would
    have trained on rungs that never fill, which is structurally the same defect
    as the missing base rung fixed in ``0744ea0``. All 10 tick arms (4
    ``pred_mjolnir.base.*_1`` + 6 ``pred_stormbreaker.base.*_1``) now declare
    ``LADDER_BPS: 1.0`` top-level, so the resolved value is unchanged and the
    cached filter parquets stay valid.

    WHY TOP-LEVEL, PLUS A CROSS-CHECK. The target is venue-agnostic: research
    hands this function the RAW ``setting.json`` dict
    (``core/research.py:714``, ``core/streaming.py:345``) — the per-venue merge
    happens only at bot boot, in ``knull.venue_config.merge_venue_config``. So
    there is no venue in scope to read. But top-level alone would not close the
    gap either, because that merge overlays the venue block and **the block wins
    on conflict** (``knull/venue_config.py:107-109``): the executor's EFFECTIVE
    spacing is the venue block's ``LADDER_BPS`` whenever it is present. Hence
    every declared per-venue value must equal the top-level one. This is
    stricter than agamotto's ``agamotto/ladder.py::ladder_params``, which only
    requires the key to exist.

    BOTH executor schemas are checked. ``_select_venue_block``
    (``knull/venue_config.py:143-162``) resolves the venue block from
    ``EXECUTORS[venue]`` when that container exists, and otherwise falls back to
    a legacy single-venue ``executor`` block. 2 of the 4 tick arms that declare
    ``LADDER_BPS`` at all carry it under the legacy key
    (``pred_mjolnir.base.{5s,15s}_1``), so checking only ``EXECUTORS`` would
    pass them vacuously — the exact hole this function exists to close.

    Args:
        config: Dictionary loaded from setting.json (UNMERGED — top-level plus
            an optional ``EXECUTORS`` container or legacy ``executor`` block).

    Returns:
        The rung spacing in bps, guaranteed finite and > 0.

    Raises:
        KeyError:   LADDER_BPS absent from the top level of config.
        ValueError: LADDER_BPS is not a finite number, is <= 0, or disagrees
            with an executor block's ``LADDER_BPS``.
    """
    if "LADDER_BPS" not in config:
        raise KeyError(
            "LADDER_BPS is required at the TOP LEVEL of setting.json and has NO "
            f"default (VERSION={config.get('VERSION', '<unset>')!r}). It is the "
            "adverse move between ladder rungs, in bps, and it sizes the TARGET "
            "the model is trained on — defaulting it silently trains on rungs "
            "the executor may never fill. Set it explicitly to the same value "
            "the executor uses (knull/ladder.py:190).")
    value = config["LADDER_BPS"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "LADDER_BPS must be a number of bps, got "
            f"{value!r} of type {type(value).__name__}.")
    step_bps = float(value)
    # > 0, not >= 0: a zero step makes `distance / step_size` infinite, which the
    # clip in compute_ladder_multiplier then turns into "every bar at the rung
    # cap" — an out-of-range result silently rendered plausible, exactly the
    # failure shape CLAUDE.md's clamp rule exists to stop. Negative is nonsense
    # outright (rungs fill on ADVERSE movement).
    if not np.isfinite(step_bps) or step_bps <= 0:
        raise ValueError(
            f"LADDER_BPS must be a finite number of bps > 0, got {step_bps!r}.")

    # Target/executor agreement. Report EVERY offending block in one message —
    # fixing one and re-running to discover the next wastes a pipeline start.
    blocks = []
    executors = config.get("EXECUTORS")
    if isinstance(executors, dict):
        blocks = [(f"EXECUTORS.{venue}", blk)
                  for venue, blk in sorted(executors.items())]
    elif isinstance(config.get("executor"), dict):
        # DEPRECATED: drop after 2026-09-01, with knull/venue_config.py's own
        # legacy branch. Checked, not ignored — it is a live source of the
        # executor's spacing for pred_mjolnir.base.{5s,15s}_1.
        blocks = [("executor", config["executor"])]

    offenders = [
        (label, blk["LADDER_BPS"]) for label, blk in blocks
        if isinstance(blk, dict) and "LADDER_BPS" in blk
        and float(blk["LADDER_BPS"]) != step_bps]
    if offenders:
        detail = ", ".join(f"{label}.LADDER_BPS={val!r}" for label, val in offenders)
        raise ValueError(
            f"LADDER_BPS={step_bps!r} at the top level disagrees with "
            f"{detail} (VERSION={config.get('VERSION', '<unset>')!r}). The "
            "executor merge overlays the venue block and the block WINS "
            "(knull/venue_config.py:107-109), so the live rung spacing "
            "would differ from the one this target is built at — the model "
            "would be trained on rungs that never fill. Make them equal.")
    return step_bps


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

    ``LADDER`` is the TOTAL rung count of the TARGET the model is trained
    on, INCLUDING the entry rung: :func:`compute_ladder_multiplier` returns
    a size in ``[1, LADDER]``, so ``LADDER`` and ``LADDER - 1`` differ by a
    whole rung of position and defaulting it silently changes what is being
    learned. ``LADDER: 0`` and ``LADDER: 1`` both mean "entry rung only"
    (size == 1) — since 2026-08-08 neither means "no position".

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


def compute_ladder_multiplier(close, adverse_extreme, ladder: int,
                              step_bps: float) -> pd.Series:
    """Number of rungs filled, in [1, ladder], for ONE position direction.

    DUPLICATE OF `agamotto_pkg/src/agamotto/ladder.py::compute_ladder_multiplier`
    (dc `b696288`), WHICH IS THE SOURCE OF TRUTH. Kept byte-equivalent, not
    imported, because the tick deploy path cannot see agamotto: xmen vendors
    ONLY `mjolnir_pkg` (`build_distribution.sh:43,76-84`, `XMEN_ALGO="mjolnir"`)
    and launches with `PYTHONPATH="$XROOT:$XROOT/mjolnir_pkg/src"`
    (`xmen/launch_xmen_bot.sh:45`) — there is no `agamotto_pkg` in the xmen
    checkout at all. `mjolnir/trading.py` imports `core.research`, which imports
    this module (`core/research.py:27`), so a cross-package import here would
    kill the live bot at boot with ModuleNotFoundError. orb gets away with
    `from agamotto.ladder import ...` (`orb/research.py:12`) only because marvel
    deploys all nine packages together.

    `mjolnir_pkg/tests/test_tick_ladder_sizing.py::test_parity_with_agamotto_*`
    is the guard that keeps the two copies from drifting.

    WHY A BASE RUNG (changed 2026-08-08). The previous mjolnir sizing was
    `size = floor(adverse / step).clip(0, LADDER)` — "2026-06-20 refinement #1:
    use n layers, NOT 1 + n — no free base rung". That premise holds only where
    the entry order needs price to come to it, and nothing in the executor works
    that way:

      * `knull/ladder.py:170` sets `_level[symbol] = 1` as soon as
        `filled_qty >= rung_qty` — rung 1 fills at ENTRY, with no adverse move.
        Only rungs 2..ladder_max need a further `LADDER_BPS` against entry
        (`:177` advance loop, `:190` `bps_frac`, `:201,208` the triggers).
      * `knull/execution_style.py:181,193-194` — LadderMakerStyle rests the
        entry post-only but `should_reprice_entry()` returns True ("Maker
        chases: _handle_entering runs its idempotent reprice"), so the entry
        follows the book instead of waiting for a dip.
      * `knull/execution_style.py:385,401` — TakerStyle sets `ioc_open` True and
        crosses the touch outright. `_regen_a26/setting.json` has
        `TRADING_MODE: "Taker"`, so the tick arms measured here fill immediately
        today, and move to LadderMaker next.

    (marvel/knull and xmen/knull are line-identical for both files, checked
    2026-08-08 — the tick bot runs the xmen copy.)

    So a signalled bar ALWAYS opens its first rung, and a bar whose adverse
    excursion was under one step was being labelled as a position that never
    opened — discarding exactly the bars where the trade was immediately right.

    Args:
        close: Series of close prices at the signal bar.
        adverse_extreme: the forward price extreme in the ADVERSE direction —
            the forward-window low for a long. For a short, mirror it before
            calling (`2*close - high_window`) so the distance is still measured
            downward.
        ladder: TOTAL rung count including the entry rung (setting.json LADDER).
        step_bps: adverse move between rungs, in bps.

    Returns:
        Integer Series in [1, ladder], indexed like `close`.
    """
    step_size = float(step_bps) * 1e-4
    # Only ladder-1 rungs are reachable by MOVING; rung 1 fills at entry. LADDER
    # is the TOTAL rung count (`knull/ladder.py:142` ladder_max = self._LADDER,
    # with levels running 1..ladder_max inclusive via the `:177` advance loop),
    # so capping at LADDER here instead would silently redefine LADDER between
    # the tick and kline algos.
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
    # representation. Measured: a 2.0bp dip at 1bp steps evaluated to
    # 1.9999999999999998, so floor() returned 1 rung instead of 2.
    layers = np.floor(distance / step_size + 1e-9).clip(
        0, max_extra).fillna(0).astype(int)
    return pd.Series(1 + layers, index=close_s.index)


def compute_ladder_returns(
    config: Dict,
    df: pd.DataFrame,
    close_col: str,
    low_col: str,
    high_col: str,
    horizon_bars: int = 1,
) -> pd.DataFrame:
    """Compute ladder-adjusted return columns for a single symbol.

    Mirrors the agamotto.research.AgamottoResearch.engineer_features ladder
    logic (both read the step from the required LADDER_BPS key, same clipping,
    same column names),
    generalized to a variable forward horizon so the fill window matches
    the prediction window.

    At horizon_bars == 1 the behaviour is identical to the original
    1-bar shift, preserving native-mode parity (TIME_UNIT == bar
    resolution, e.g. mjolnir.base.5s_1). For boundary-aligned
    experiments (e.g. mjolnir.base.30s_1: 5s bars predicting the next
    30s boundary close) callers should pass
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
    # Rung spacing: REQUIRED top-level key, cross-checked against every
    # EXECUTORS[<venue>] block so the target's rungs are the ones the executor
    # actually fills. Was the module constant LADDER_STEP_BPS = 1.0.
    step_bps = resolve_ladder_bps(config)
    # FEE is required (no fallback): see commit 0500d8fa for rationale.
    # The historical `or 0.0` collapsed any falsy FEE to the default.
    fee_rate = float(config["FEE"]) / 10000.0

    close = df[close_col]
    low_series = df[low_col]
    high_series = df[high_col]

    close_safe = close.replace(0, np.nan)
    # Forward h-bar price return: close[t+h] / close[t] - 1, off close_safe so a
    # ZERO close yields NaN (droppable) rather than +/-inf (which poisons the
    # target and every fit downstream). Identical to `close` on valid data —
    # they differ only where close == 0.
    #
    # Off raw `close` this was `(x - 0)/0 = inf`. It used to be masked by
    # accident: close_safe[t] was NaN there, so the old sizing floored to 0 rungs
    # and `inf * 0` collapsed to NaN. The base rung (2026-08-08) makes size >= 1,
    # so the inf would now survive into the label. Guard it at the source.
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

    # Rungs filled on each side's forward ADVERSE excursion, in [1, LADDER].
    # low_layers = how far price DUG DOWN below cost (long entries); high_layers
    # = how far it RALLIED UP above cost (short entries).
    #
    # 2026-08-08: the base rung is back. "2026-06-20 refinement #1: use n
    # layers, NOT 1 + n — no free base rung" assumed the entry needs price to
    # come to it; the executor fills rung 1 on entry unconditionally (Taker
    # crosses; LadderMaker chases the book). See compute_ladder_multiplier's
    # docstring for the citations and for why this is a copy of agamotto's.
    low_layers = compute_ladder_multiplier(
        close_safe, low_window, ladder, step_bps)
    # Mirror the short's adverse (upward) move about the close so the same
    # downward-measuring helper serves both legs.
    high_layers = compute_ladder_multiplier(
        close_safe, 2.0 * close_safe - high_window, ladder, step_bps)

    # 2026-06-20 refinement #2 (Stan — mark-to-market exit): size each position
    # by its ENTRY-side penetration and mark the WHOLE opened stack at the
    # horizon close, i.e. force-liquidate at close[t+h] whatever did not take
    # profit. Unclosed inventory therefore BOOKS the real directional move at
    # the close (underwater longs in a downtrend / underwater shorts in a bull
    # realize their loss) instead of being silently dropped — the earlier
    # size=min(low,high) gate discarded the carried, mostly-losing, inventory
    # and was optimistically biased. LONG accumulates on the DIP -> size_long =
    # low_layers; SHORT accumulates on the RALLY -> size_short = high_layers.
    # No look-ahead favorable exit is credited: selecting round-trips by
    # min(open,close) and paying them a take-profit would re-introduce a smaller
    # excursion-conditioned artifact, so the exit is purely liquidate-at-close.
    size_long = low_layers
    size_short = high_layers
    # Per-unit UNSIGNED price return on each side's own exit path, BEFORE fee and
    # size. Default ("ladder") marks the whole opened stack at the horizon close
    # (close-to-close), so both sides see the same close-to-close move; fill_mode
    # may override either the size (flat) or the exit price (limit_then_taker),
    # which is why the two are tracked separately.
    #
    # UNSIGNED, i.e. the MARKET move, NOT the trade's P&L: a short is profitable
    # when this is NEGATIVE. This matches agamotto (agamotto/research.py:379-380,
    # `return_{long,short}_raw = price_return * size`) so the shared marvel PnL
    # engine — which books `signal * y_true_raw`, with signal = -1 for a short —
    # is correct for BOTH algos. Until 2026-07-29 mjolnir negated the short here,
    # making its target position-SIGNED while agamotto's stayed unsigned under the
    # SAME column names; the engine then signed mjolnir's a second time and every
    # short leg booked +price_return. See tasks/lessons.md 2026-07-29.
    ret_long_px = price_return
    ret_short_px = price_return

    fill_mode = resolve_fill_mode(config)
    if fill_mode == "flat":
        # Two-way TAKER model — fixed size 1 per bar, filled at the decision
        # (horizon-close) price, no laddered size and no maker rungs. This is
        # what aggressive taker execution actually realizes (precise fill
        # PRICE, fixed SIZE), as opposed to the laddered maker accumulation.
        size_long = 1
        size_short = 1
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
        close_h = close.shift(-horizon_bars)
        close_2h = close.shift(-2 * horizon_bars)
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

    fee_cost = (fee_rate * 2.0) if fee_rate else 0.0
    # Targets are UNSIGNED market returns (the trade direction is applied by the
    # consumer). The round-trip fee is therefore SUBTRACTED for the long and ADDED
    # for the short: a long needs the move above +fee to profit, a short needs it
    # below -fee. Identical to agamotto (agamotto/research.py:370-380
    # `long: price_return - fee_cost`, `short: price_return + fee_cost`) and to
    # features.py's non-ladder path.
    return_long = ((ret_long_px - fee_cost) * size_long).rename("return_long")
    return_short = ((ret_short_px + fee_cost) * size_short).rename("return_short")
    return_long_raw = (ret_long_px * size_long).rename("return_long_raw")
    return_short_raw = (ret_short_px * size_short).rename("return_short_raw")

    return pd.concat(
        [return_long, return_short, return_long_raw, return_short_raw], axis=1)
