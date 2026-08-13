"""MM-realistic kline target — what the live MarketMaker actually earns.

Design: marvel `docs/plans/2026-08-10-mm-realistic-target-design.md` (Phase B) and the
2026-08-10 minute-resolution revision.

WHAT THIS REPLACES. The ladder target (`ladder.py` + `research._compute_ladder_returns`)
scales a **close-to-close** return by a rung multiplier. It never credits the passive entry
fill price — the entire economic point of resting post-only — and marks the exit at the bar
close rather than at the resting rung. That prices a directional taker, not knull's
MarketMaker.

WHY MINUTES, NOT 15m BARS. The first cut of this module resolved the lifecycle from 15m
bars and was structurally wrong. `PASSIVE_SEC 600 + CROSSING_SEC 300 = 900 s` is **exactly
one 15m bar**, so entry, every ladder rung, the resting exit and the crossing all happen
inside bar **T+1**. Reading T+2 and discarding T+1 threw away the half of the window where
a bounce is most likely, and marked the crossing up to a full bar late on a branch already
conditioned on "price did not come back". Measured, that model implied **-29.96 bps/bar =
-28.8% of capital per day** against a live book that was profitable. No market maker loses
28% a day. Minute bars inside T+1 are therefore not a refinement — they are the resolution
the strategy actually lives at.

WHAT THE LIVE EXECUTOR DOES (verified in marvel `knull/`):

  * post-only entry ladder, rungs at `close - k*LADDER_BPS` for a long
    (`knull/ladder.py:186,200-211`);
  * a resting reduce-only close at `avg_cost +/- MM_PROFIT_AIM` bps placed the moment any
    entry rung partial-fills (`knull/market_maker.py:53`);
  * a FLAT signal does NOT cross — resting MM rungs fill passively
    (`knull/base_executor.py:3993-4008`); only a direction flip crosses immediately
    (`:4032-4054`);
  * no taker fallback inside MM placement — a post-only reject is skipped, never crossed
    (`knull/market_maker.py:148,282`);
  * the real taker trigger is a TIMER (`knull/base_executor.py:3977-3983`).

So the payoff is capped at the aim on the maker branch and open-ended on the taker branch.

SIGN CONVENTION — read before touching the short leg. The target columns live in
**price-return space, not PnL space**, because the canonical PnL engine applies the position
sign itself (marvel `gauntlet/evaluate_regimes.py:196-197`:
``sign = 1 if position == 'long' else -1; rets = grp[ret_col].values * sign``). The existing
target obeys this — `return_short_raw = price_return * size_short`, NOT negated. An MM target
returning a short's true PnL would be sign-flipped a SECOND time and would silently invert
the whole short book. Hence:

    return_long_raw  = +mm_pnl_long
    return_short_raw = -mm_pnl_short

`test_mm_target.py` pins it.

FEE. The taker branch absorbs `MM_TAKER_FEE_BPS`; the maker branch pays zero (post-only both
sides). Seven downstream sites re-derive a round trip from `FEE` (marvel
`generate_daily_pnl.py:1168,1365,1541`, `filter_regime_stacks.py:144`,
`evaluate_regimes.py:476`, and dc `research.py:218,252`), so a non-zero `FEE` would DOUBLE
CHARGE. `mm_params` raises rather than letting that happen.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .ladder import ladder_params
from .utils import _timeframe_to_seconds

# The only base timeframe whose MM lifecycle this module resolves. Note the limit is now the
# 15m *signal grid*, not the data: with minute bars the 900 s lifecycle is resolvable at any
# base timeframe, so lifting this is a data-plumbing question rather than a modelling one.
SUPPORTED_TIMEFRAME = "15m"

MINUTE_SECONDS = 60

# Timeframe directory holding the sub-bar data, i.e.
# data/BINANCEFUTURES/1m/<family>/<SYM>/<SYM>_<YYYY-MM>_1m.csv. Synced by
# marvel `gauntlet/sync_klines.sh`.
MINUTE_TIMEFRAME = "1m"

TARGET_MODE_LADDER = "ladder"
TARGET_MODE_MM = "mm"
_TARGET_MODES = (TARGET_MODE_LADDER, TARGET_MODE_MM)


def target_mode(config: Dict) -> str:
    """Which target to build: ``"ladder"`` (close-to-close) or ``"mm"``.

    # DEPRECATED: drop the absent->ladder branch after 2027-01-01. 159 kline settings
    # predate this key; breaking them all at once would strand every experiment. Same
    # precedent and shape as `ladder_params`' LADDER_LONG -> LADDER chain. An UNKNOWN value
    # always raises, so the chain still ends in a fail-fast.
    """
    mode = config.get("TARGET_MODE")
    if mode is None:
        return TARGET_MODE_LADDER
    if mode not in _TARGET_MODES:
        raise ValueError(
            f"TARGET_MODE={mode!r} is not one of {_TARGET_MODES}. "
            f"'{TARGET_MODE_LADDER}' is the close-to-close ladder target; "
            f"'{TARGET_MODE_MM}' is the minute-resolved MM target (15m signal grid, needs "
            f"1m bars — see agamotto/mm_target.py)")
    return str(mode)


def assert_supported_timeframe(config: Dict) -> None:
    """Refuse any signal grid other than 15m. See SUPPORTED_TIMEFRAME."""
    tf = config.get("TIME_UNIT")
    if tf != SUPPORTED_TIMEFRAME:
        raise ValueError(
            f"TARGET_MODE='mm' supports TIME_UNIT={SUPPORTED_TIMEFRAME!r} only, got {tf!r}. "
            f"The MM lifecycle is 900 s = one 15m bar; on other grids the entry bar and the "
            f"patience window do not line up, and this module does not yet handle the "
            f"offset. Lifting it is plumbing, not modelling — the minute bars already carry "
            f"the resolution.")


def mm_params(config: Dict) -> Tuple[float, float, float]:
    """``(MM_PROFIT_AIM, MM_PATIENCE_SEC, MM_TAKER_FEE_BPS)`` — all REQUIRED.

    No defaults anywhere (CLAUDE.md): an aim, a patience or a taker fee that silently
    defaulted would bake an operating point into what is meant to measure one.

    ``FEE`` must be exactly 0 — this target is already net (see the module docstring's FEE
    note). ``MM_PATIENCE_SEC`` must be a whole number of minutes, because the maker window is
    resolved on a minute grid and a partial minute cannot be adjudicated.

    Raises:
        KeyError: when any of the three keys is absent.
        ValueError: when FEE is non-zero, a value is non-finite/negative, or the patience is
            not a whole number of minutes.
    """
    if "FEE" not in config:
        raise KeyError(
            "FEE missing — required even for TARGET_MODE='mm', where it must be exactly 0 "
            "(this target is already net of the taker fee).")
    fee = float(config["FEE"])
    if fee != 0.0:
        raise ValueError(
            f"TARGET_MODE='mm' requires FEE == 0, got {fee}. This target absorbs "
            f"MM_TAKER_FEE_BPS on the taker branch and pays nothing on the maker branch, so "
            f"it is already NET. A non-zero FEE double charges at every downstream "
            f"round-trip site (generate_daily_pnl.py:1168,1365,1541, "
            f"filter_regime_stacks.py:144, evaluate_regimes.py:476, research.py:218,252).")

    out = []
    for key in ("MM_PROFIT_AIM", "MM_PATIENCE_SEC", "MM_TAKER_FEE_BPS"):
        if config.get(key) is None:
            raise KeyError(
                f"{key} is required in setting.json for TARGET_MODE='mm' — no default. "
                f"MM_PROFIT_AIM is the resting exit's distance from the rung VWAP in bps; "
                f"MM_PATIENCE_SEC is PASSIVE_SEC+CROSSING_SEC, when the crossing fires; "
                f"MM_TAKER_FEE_BPS is the ONE-SIDE taker fee, charged on the taker branch "
                f"only.")
        val = float(config[key])
        if not np.isfinite(val) or val < 0:
            raise ValueError(f"{key} must be finite and >= 0, got {val}")
        out.append(val)

    aim, patience, taker_fee = out
    if patience % MINUTE_SECONDS != 0:
        raise ValueError(
            f"MM_PATIENCE_SEC={patience} is not a whole number of minutes. The maker window "
            f"is adjudicated on a 1m grid; a partial minute cannot be resolved without "
            f"assuming the intra-minute path.")
    return aim, patience, taker_fee


def patience_minutes(patience_sec: float) -> int:
    """Number of 1m bars the MM lifecycle spans. Raises on a non-positive window."""
    n = int(round(float(patience_sec) / MINUTE_SECONDS))
    if n <= 0:
        raise ValueError(
            f"MM_PATIENCE_SEC={patience_sec} yields {n} minutes — the position would be "
            f"crossed before any maker fill could be observed, which is not a strategy.")
    return n


def minute_matrices(signal_index: pd.DatetimeIndex, minute_bars: pd.DataFrame,
                    n_minutes: int,
                    bar_seconds: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape 1m OHLC into ``(n_signal_bars, n_minutes)`` matrices covering T+1.

    A 15m bar with open time ``t`` closes at ``t + 15m``; the signal fires there and the
    lifecycle runs over the NEXT ``n_minutes`` minutes, i.e. ``t+15m .. t+15m+(n-1)m``.

    Timestamps are built by broadcasting rather than by positional slicing, so a gap in the
    1m series produces NaN for that minute instead of silently shifting every later bar onto
    the wrong minutes.

    ``bar_seconds`` is REQUIRED rather than inferred from ``signal_index``: inference needs
    two bars, which a single-bar frame does not have, and the signal grid is already pinned
    exactly by ``assert_supported_timeframe``. Guessing it would also be a silent fallback.

    Returns:
        ``(high, low, close)``, each ``(len(signal_index), n_minutes)`` float arrays, NaN
        where the minute is missing.
    """
    if not isinstance(signal_index, pd.DatetimeIndex):
        raise TypeError(f"signal_index must be a DatetimeIndex, got {type(signal_index)!r}")
    for col in ("high", "low", "close"):
        if col not in minute_bars.columns:
            raise KeyError(f"minute_bars is missing required column {col!r}")
    if not isinstance(minute_bars.index, pd.DatetimeIndex):
        raise TypeError("minute_bars must be indexed by a DatetimeIndex")

    if not np.isfinite(bar_seconds) or bar_seconds <= 0:
        raise ValueError(f"bar_seconds must be finite and > 0, got {bar_seconds}")

    # Start of T+1 for every signal bar. Timedelta arithmetic only (CLAUDE.md): never a raw
    # int64 view of a datetime index, whose unit is [us] after a parquet round trip and would
    # read 15 minutes as 0.0009 s.
    bar_delta = pd.Timedelta(seconds=float(bar_seconds))
    starts = signal_index + bar_delta
    offsets = pd.to_timedelta(np.arange(n_minutes) * MINUTE_SECONDS, unit="s")
    wanted = (starts.values[:, None] + offsets.values[None, :]).ravel()

    src = minute_bars[~minute_bars.index.duplicated(keep="last")]
    aligned = src.reindex(pd.DatetimeIndex(wanted))
    shape = (len(signal_index), n_minutes)
    return (aligned["high"].to_numpy(dtype=float).reshape(shape),
            aligned["low"].to_numpy(dtype=float).reshape(shape),
            aligned["close"].to_numpy(dtype=float).reshape(shape))


def resolve_leg(close: pd.Series, high_m: np.ndarray, low_m: np.ndarray,
                close_m: np.ndarray, *, leg: str, ladder: int, ladder_bps: float,
                aim_bps: float, taker_fee_bps: float) -> pd.DataFrame:
    """Walk the minutes of T+1 and resolve each signal bar to a maker or taker branch.

    Fully vectorised: a Python loop over bars would be ~3.2M x 15 iterations.

    The rules, and why each is what it is:

      * Rung 0 fills at the first minute — the base rung needs no adverse move
        (`knull/ladder.py:165-172`, and dc `ladder.py`'s multiplier is ``1 + extra``). The
        timer starts there, which is the EARLIEST possible start and so can only UNDER-count
        maker fills.
      * Rung k>0 fills the first minute whose low reaches ``close * (1 - k*step)``.
      * The exit resting DURING minute j is priced off fills up to the end of minute j-1, and
        is tested against minute j's high. Nothing assumes intra-minute ordering, so a minute
        that both fills a rung and prints through the exit resolves as a fill only, never as
        a same-minute round trip.
      * **Size at a maker exit is the rung count as of j-1, not the final count.** Once the
        reduce-only close fills, the position is flat and the remaining entry rungs are
        cancelled. Using the final size would let a bar bank the aim on rungs it never held,
        and would understate the asymmetry that defines this mode: a quick bounce locks a
        SMALL position, while a ladder that never recovers takes the full loss at a LARGE one.
      * No maker fill by the last minute -> cross at that minute's close, paying
        ``taker_fee_bps`` once.

    Returns:
        DataFrame indexed like ``close`` with ``size``, ``avg_cost``, ``is_maker`` and
        ``pnl`` (fractional, per unit CAPITAL, TRUE sign for this leg). Bars whose minute
        coverage is incomplete are NaN — never guessed.
    """
    if leg not in ("long", "short"):
        raise ValueError(f"leg must be 'long' or 'short', got {leg!r}")
    sign = 1 if leg == "long" else -1
    if int(ladder) < 1:
        raise ValueError(f"ladder must be >= 1, got {ladder}")

    close_s = pd.Series(close).replace(0, np.nan)
    c = close_s.to_numpy(dtype=float)[:, None]              # (n,1)
    step = float(ladder_bps) * 1e-4
    aim_frac = float(aim_bps) * 1e-4
    fee_frac = float(taker_fee_bps) * 1e-4

    # A bar is resolvable only with COMPLETE minute coverage. A partial window would silently
    # shorten the patience for that bar.
    valid = (np.isfinite(high_m).all(axis=1) & np.isfinite(low_m).all(axis=1)
             & np.isfinite(close_m).all(axis=1) & np.isfinite(c[:, 0]))

    # Rungs run AGAINST the position: below the close for a long, above for a short.
    k = np.arange(int(ladder))[None, :]                      # (1,L)
    rungs = c * (1.0 - sign * k * step)                      # (n,L)

    # How far price has run against us by the end of each minute.
    extreme = (np.minimum.accumulate(low_m, axis=1) if sign > 0
               else np.maximum.accumulate(high_m, axis=1))   # (n,M)
    # Rung k (k>=1) is filled once the adverse extreme reaches it. Rung 0 is the base rung.
    reached = ((extreme[:, :, None] <= rungs[:, None, 1:]) if sign > 0
               else (extreme[:, :, None] >= rungs[:, None, 1:]))
    n_filled = 1 + reached.sum(axis=2)                       # (n,M), in [1, ladder]

    avg_cost = c * (1.0 - sign * ((n_filled - 1) / 2.0) * step)   # (n,M)
    exit_px = avg_cost * (1.0 + sign * aim_frac)                  # (n,M)

    # Exit resting during minute j is priced at j-1 and tested against minute j.
    if high_m.shape[1] < 2:
        raise ValueError(
            "need >= 2 minutes in the patience window: with one minute the exit can only be "
            "tested on the same minute it was placed, which assumes the intra-minute path.")
    hit = ((high_m[:, 1:] >= exit_px[:, :-1]) if sign > 0
           else (low_m[:, 1:] <= exit_px[:, :-1]))            # (n,M-1)

    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)          # index into j-1 space; the fill minute is first+1
    rows = np.arange(hit.shape[0])

    # Maker: size and cost frozen at the minute BEFORE the fill.
    size_maker = n_filled[rows, first]
    pnl_maker = size_maker * aim_frac

    # Taker: everything the ladder accumulated, marked at the last minute's close.
    size_taker = n_filled[:, -1]
    avg_taker = avg_cost[:, -1]
    taker_ret = sign * (close_m[:, -1] - avg_taker) / avg_taker
    pnl_taker = size_taker * (taker_ret - fee_frac)

    is_maker = any_hit & valid
    pnl = np.where(any_hit, pnl_maker, pnl_taker)
    size = np.where(any_hit, size_maker, size_taker)
    cost = np.where(any_hit, avg_cost[rows, first], avg_taker)

    return pd.DataFrame(
        {"size": np.where(valid, size, np.nan),
         "avg_cost": np.where(valid, cost, np.nan),
         "is_maker": pd.Series(np.where(valid, is_maker, np.nan), index=close_s.index),
         "pnl": np.where(valid, pnl, np.nan)},
        index=close_s.index,
    )


def compute_mm_target(df: pd.DataFrame, close_col: str, low_col: str, high_col: str,
                      config: Dict,
                      minute_bars: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """MM-realistic replacements for the four ladder target columns.

    Drop-in for `research._compute_ladder_returns`: same index, same four column names,
    different economics. Values are net fractional PnL per unit CAPITAL, in PRICE-RETURN SIGN
    SPACE (see the module's SIGN CONVENTION note).

    ``low_col``/``high_col`` are accepted for signature compatibility with the ladder path but
    are unused: the 15m extremes cannot adjudicate a 900 s lifecycle, which is the defect this
    revision exists to fix.

    Raises:
        ValueError: when ``minute_bars`` is absent. There is deliberately NO fallback to the
            15m-resolved model — it implied -28.8%/day against a profitable live book, and a
            silent downgrade to it is exactly the class of failure CLAUDE.md bans.
    """
    assert_supported_timeframe(config)
    ladder_long, ladder_short, step_bps = ladder_params(config)
    aim_bps, patience_sec, taker_fee_bps = mm_params(config)
    n_minutes = patience_minutes(patience_sec)

    if minute_bars is None or len(minute_bars) == 0:
        raise ValueError(
            "TARGET_MODE='mm' requires 1m bars covering the signal span, and none were "
            "supplied. Sync them with gauntlet/sync_klines.sh (the 1m leg) — there is no "
            "fallback to the 15m-resolved model, which discarded bar T+1 entirely and "
            "implied -28.8% of capital per day against a live book that was profitable.")

    # Exact, not inferred: assert_supported_timeframe has already pinned the grid to 15m.
    high_m, low_m, close_m = minute_matrices(
        df.index, minute_bars, n_minutes,
        _timeframe_to_seconds(SUPPORTED_TIMEFRAME))
    close = df[close_col]

    long_res = resolve_leg(close, high_m, low_m, close_m, leg="long", ladder=ladder_long,
                           ladder_bps=step_bps, aim_bps=aim_bps,
                           taker_fee_bps=taker_fee_bps)
    short_res = resolve_leg(close, high_m, low_m, close_m, leg="short", ladder=ladder_short,
                            ladder_bps=step_bps, aim_bps=aim_bps,
                            taker_fee_bps=taker_fee_bps)

    long_col = long_res["pnl"]
    # NEGATED: the engine multiplies a short leg's target by -1. Returning true short PnL
    # here would invert the entire short book. See SIGN CONVENTION.
    short_col = -short_res["pnl"]

    return pd.DataFrame({
        "return_long": long_col,
        "return_short": short_col,
        "return_long_raw": long_col,
        "return_short_raw": short_col,
    }, index=df.index)
