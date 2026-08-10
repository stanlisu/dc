"""MM-realistic kline target — what the live MarketMaker actually earns.

Design: marvel `docs/plans/2026-08-10-mm-realistic-target-design.md`, Phase B.

WHAT THIS REPLACES. The ladder target (`ladder.py` + `research._compute_ladder_returns`)
scales a **close-to-close** return by a rung multiplier. It never credits the passive
entry fill price — which is the entire economic point of resting post-only — and it marks
the exit at the bar close rather than at the resting rung. That target describes a
directional taker, not `knull`'s MarketMaker.

WHAT THE LIVE EXECUTOR DOES (verified in marvel `knull/`):

  * post-only entry ladder, rungs at `close - k*LADDER_BPS` for a long
    (`knull/ladder.py:186,200-211`);
  * a resting reduce-only close at `avg_cost ± MM_PROFIT_AIM` bps placed the moment any
    entry rung partial-fills (`knull/market_maker.py:53`);
  * a FLAT signal does NOT cross — resting MM rungs are left to fill passively
    (`knull/base_executor.py:3993-4008`); only a direction flip crosses immediately
    (`:4032-4054`);
  * there is no taker fallback inside MM placement — a post-only reject is skipped, never
    crossed (`knull/market_maker.py:148,282`);
  * the real taker trigger is a TIMER: `PASSIVE_SEC` then the crossing phase
    (`knull/base_executor.py:3977-3983`).

So the payoff is capped at the aim on the maker branch and open-ended on the taker branch.

SIGN CONVENTION — read this before touching the short leg. The target columns live in
**price-return space, not PnL space**, because the canonical PnL engine applies the
position sign itself (marvel `gauntlet/evaluate_regimes.py:196-197`:
``sign = 1 if position == 'long' else -1; rets = grp[ret_col].values * sign``). The
existing target obeys this — `return_short_raw = price_return * size_short`, NOT negated.
An MM target that returned a short's true PnL would therefore be sign-flipped a SECOND
time by the engine and silently invert the entire short book. Hence:

    return_long_raw  = +mm_pnl_long
    return_short_raw = -mm_pnl_short

so that `column * sign` recovers the true MM PnL on both legs. `test_mm_target.py`
pins this.

FEE. The taker branch absorbs `MM_TAKER_FEE_BPS` internally, and the maker branch pays
zero (post-only both sides). Seven downstream sites re-derive a round trip from `FEE`
(marvel `generate_daily_pnl.py:1168,1365,1541`, `filter_regime_stacks.py:144`,
`evaluate_regimes.py:476`, and dc `research.py:218,252`), so a non-zero `FEE` would
DOUBLE CHARGE against this target. `mm_params` raises rather than letting that happen —
the design's "fee audit" gate, enforced in code instead of recorded in prose.

TIMEFRAME. 15m only, and excluded rather than deferred elsewhere: `PASSIVE_SEC +
CROSSING_SEC = 900 s` is exactly one 15m bar, so the maker window spans T+1..T+2 and is
readable off the bar grid. On 1h and above the entry fill, the maker window and the
crossing all land inside T+1; resolving them needs an intra-bar path assumption, which
manufactures a free lunch (a full ladder plus a profitable maker exit needs ~11 bps of
range at LADDER_BPS=1 / aim=1, which nearly every bar has).
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .ladder import compute_ladder_multiplier, ladder_params

# The only timeframe whose MM lifecycle is resolvable from its own bars.
SUPPORTED_TIMEFRAME = "15m"

# Target-mode selector values.
TARGET_MODE_LADDER = "ladder"
TARGET_MODE_MM = "mm"
_TARGET_MODES = (TARGET_MODE_LADDER, TARGET_MODE_MM)


def target_mode(config: Dict) -> str:
    """Which target to build: ``"ladder"`` (close-to-close) or ``"mm"``.

    # DEPRECATED: drop the absent->ladder branch after 2027-01-01. 159 kline
    # settings predate this key; breaking them all at once to force an explicit
    # mode would strand every experiment. Same precedent, and the same shape, as
    # `ladder_params`' LADDER_LONG -> LADDER chain. An UNKNOWN value always
    # raises, so the chain still ends in a fail-fast.
    """
    mode = config.get("TARGET_MODE")
    if mode is None:
        return TARGET_MODE_LADDER
    if mode not in _TARGET_MODES:
        raise ValueError(
            f"TARGET_MODE={mode!r} is not one of {_TARGET_MODES}. "
            f"'{TARGET_MODE_LADDER}' is the close-to-close ladder target; "
            f"'{TARGET_MODE_MM}' is the MM-realistic target (15m only, see "
            f"agamotto/mm_target.py)")
    return str(mode)


def assert_supported_timeframe(config: Dict) -> None:
    """Refuse any arm coarser than 15m. See this module's TIMEFRAME note."""
    tf = config.get("TIME_UNIT")
    if tf != SUPPORTED_TIMEFRAME:
        raise ValueError(
            f"TARGET_MODE='mm' supports TIME_UNIT={SUPPORTED_TIMEFRAME!r} only, "
            f"got {tf!r}. 1h/4h/1d are EXCLUDED, not deferred: the whole MM "
            f"lifecycle lands inside one bar there, so any number would be an "
            f"intra-bar path assumption rather than a measurement. Those arms "
            f"need 1m klines, which is a separate data decision.")


def mm_params(config: Dict) -> Tuple[float, float, float]:
    """``(MM_PROFIT_AIM, MM_PATIENCE_SEC, MM_TAKER_FEE_BPS)`` — all REQUIRED.

    No defaults anywhere (CLAUDE.md): an aim, a patience or a taker fee that
    silently defaulted would bake an operating point into what is supposed to be
    a measurement of one.

    ``FEE`` must be exactly 0. The taker branch charges ``MM_TAKER_FEE_BPS``
    itself and the maker branch is post-only on both sides, so this target is
    already net. Every downstream round-trip fee derivation would double charge.

    Raises:
        KeyError: when any of the three keys is absent.
        ValueError: when FEE is non-zero, or a value is non-finite/negative.
    """
    if "FEE" not in config:
        raise KeyError(
            "FEE missing — required even for TARGET_MODE='mm', where it must be "
            "exactly 0 (this target is already net of the taker fee).")
    fee = float(config["FEE"])
    if fee != 0.0:
        raise ValueError(
            f"TARGET_MODE='mm' requires FEE == 0, got {fee}. This target absorbs "
            f"MM_TAKER_FEE_BPS on the taker branch and pays nothing on the maker "
            f"branch, so it is already NET. A non-zero FEE double charges at "
            f"every downstream round-trip site (generate_daily_pnl.py:1168,1365,"
            f"1541, filter_regime_stacks.py:144, evaluate_regimes.py:476, and "
            f"research.py:218,252).")

    out = []
    for key in ("MM_PROFIT_AIM", "MM_PATIENCE_SEC", "MM_TAKER_FEE_BPS"):
        if config.get(key) is None:
            raise KeyError(
                f"{key} is required in setting.json for TARGET_MODE='mm' — no "
                f"default. MM_PROFIT_AIM is the resting exit's distance from the "
                f"rung VWAP in bps; MM_PATIENCE_SEC is PASSIVE_SEC+CROSSING_SEC, "
                f"when the crossing fires; MM_TAKER_FEE_BPS is the ONE-SIDE taker "
                f"fee charged on the taker branch only.")
        val = float(config[key])
        if not np.isfinite(val) or val < 0:
            raise ValueError(f"{key} must be finite and >= 0, got {val}")
        out.append(val)
    return out[0], out[1], out[2]


def bar_seconds(index: pd.DatetimeIndex) -> float:
    """Bar spacing in seconds, via ``total_seconds()``.

    NEVER a raw int64 view of a datetime index (CLAUDE.md): ``.asi8`` returns the
    index's OWN unit, which is ``[us]`` for anything round-tripped through
    parquet — dc `25eb57a` read 5s bars as 0.005s exactly that way.
    """
    if index is None or len(index) < 2:
        raise ValueError("need >= 2 bars to infer the bar spacing")
    deltas = pd.Series(index[1:] - index[:-1]).dt.total_seconds()
    spacing = float(deltas.median())
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"non-positive bar spacing inferred: {spacing}")
    return spacing


def expiry_offset(patience_sec: float, bar_sec: float) -> int:
    """Bar offset from the signal bar T of the bar holding the timer expiry.

    The MM clock starts at the first rung fill, i.e. somewhere INSIDE entry bar
    T+1. We take the EARLIEST possible start — the open of T+1 — which expires
    the timer soonest and can therefore only UNDER-count maker fills. That is the
    design's conservative no-lookahead rule.
    """
    return 1 + int(np.floor(float(patience_sec) / float(bar_sec)))


def resolve_leg(close, low_next, high_next, index, *, leg: str, ladder: int,
                ladder_bps: float, aim_bps: float, patience_sec: float,
                taker_fee_bps: float) -> pd.DataFrame:
    """Resolve every signal bar to a maker or taker branch, for ONE leg.

    Mirrors marvel `gauntlet/mm_fill_census.py::resolve_leg` — the Phase A census
    that measured this surface. The two MUST agree: the census is what says this
    operating point loses 29.96 bps/bar, and a target that priced it differently
    would be measuring a different strategy than the one that was screened. dc
    packages ship standalone and cannot import marvel's `gauntlet`, so the
    agreement is pinned by golden values in `tests/test_mm_target.py` instead of
    by a shared import.

    Returns:
        DataFrame with `size`, `avg_cost`, `is_maker`, `pnl` (fractional PnL per
        unit CAPITAL, TRUE sign for this leg), indexed like `close`. Rows lacking
        the forward bars they need carry NaN `pnl` and are never guessed at.
    """
    if leg not in ("long", "short"):
        raise ValueError(f"leg must be 'long' or 'short', got {leg!r}")
    sign = 1 if leg == "long" else -1

    close_s = pd.Series(close).replace(0, np.nan)
    # compute_ladder_multiplier always measures DOWNWARD, so mirror the short's
    # adverse (upward) extreme about the close before handing it over — exactly
    # as research._compute_ladder_returns does.
    adverse = (pd.Series(low_next) if sign > 0
               else 2.0 * close_s - pd.Series(high_next))
    size = compute_ladder_multiplier(close_s, adverse, ladder, ladder_bps)

    # Rungs fill at close -/+ k*step for k = 0..n, so their VWAP sits n/2 steps
    # against the position.
    n_extra = size - 1
    avg_cost = close_s - sign * close_s * (n_extra / 2.0) * float(ladder_bps) * 1e-4
    exit_px = avg_cost * (1.0 + sign * float(aim_bps) * 1e-4)

    bar_sec = bar_seconds(index)
    exit_off = expiry_offset(patience_sec, bar_sec)

    # Maker credit comes ONLY from bars strictly after the entry bar (offset 1),
    # through the expiry bar. Whatever part of the maker window sits inside T+1
    # is discarded — we never assume the intra-bar path. An empty window means
    # the timer expires inside T+1, which is taker by construction and the honest
    # reading of a patience shorter than one bar.
    window_offsets = list(range(2, exit_off + 1))
    highs = pd.Series(high_next).reset_index(drop=True)
    lows = pd.Series(low_next).reset_index(drop=True)
    # high_next/low_next are already shifted by 1, so offset k means shift(k-1).
    if window_offsets:
        col = highs if sign > 0 else lows
        reach = pd.concat([col.shift(-(k - 1)) for k in window_offsets], axis=1)
        window_extreme = (reach.max(axis=1) if sign > 0 else reach.min(axis=1))
        window_extreme.index = close_s.index
    else:
        window_extreme = pd.Series(np.nan, index=close_s.index)

    close_exit = close_s.shift(-exit_off)
    is_maker = (window_extreme >= exit_px if sign > 0
                else window_extreme <= exit_px)

    taker_ret = sign * (close_exit - avg_cost) / avg_cost
    taker_pnl = size * (taker_ret - float(taker_fee_bps) * 1e-4)
    # Capped upside: the maker branch earns the aim and pays no fee, however far
    # price ran afterwards. This cap against an uncapped taker tail is the whole
    # economics of the mode.
    maker_pnl = size * float(aim_bps) * 1e-4

    pnl = maker_pnl.where(is_maker, taker_pnl)
    # A row is priced only with its entry geometry AND its expiry bar present.
    unresolvable = avg_cost.isna() | close_exit.isna()
    return pd.DataFrame(
        {"size": size, "avg_cost": avg_cost,
         "is_maker": is_maker.where(~unresolvable),
         "pnl": pnl.where(~unresolvable)},
        index=close_s.index,
    )


def compute_mm_target(df: pd.DataFrame, close_col: str, low_col: str,
                      high_col: str, config: Dict) -> pd.DataFrame:
    """MM-realistic replacements for the four ladder target columns.

    Drop-in for `research._compute_ladder_returns`: same index, same four column
    names, different economics. Values are net fractional PnL per unit CAPITAL,
    in PRICE-RETURN SIGN SPACE (see this module's SIGN CONVENTION note) — the
    short columns are negated so the PnL engine's `* -1` recovers true short PnL.

    `return_long` / `return_short` are identical to their `_raw` twins here: the
    fee is already absorbed per branch, so there is no separate fee-adjusted
    variant to build. They are emitted anyway because every downstream consumer
    expects all four names.
    """
    assert_supported_timeframe(config)
    ladder_long, ladder_short, step_bps = ladder_params(config)
    aim_bps, patience_sec, taker_fee_bps = mm_params(config)

    close = df[close_col]
    low_next = df[low_col].shift(-1)
    high_next = df[high_col].shift(-1)

    long_res = resolve_leg(
        close, low_next, high_next, df.index, leg="long", ladder=ladder_long,
        ladder_bps=step_bps, aim_bps=aim_bps, patience_sec=patience_sec,
        taker_fee_bps=taker_fee_bps)
    short_res = resolve_leg(
        close, low_next, high_next, df.index, leg="short", ladder=ladder_short,
        ladder_bps=step_bps, aim_bps=aim_bps, patience_sec=patience_sec,
        taker_fee_bps=taker_fee_bps)

    long_col = long_res["pnl"]
    # NEGATED: the engine multiplies a short leg's target by -1. Returning true
    # short PnL here would invert the entire short book. See SIGN CONVENTION.
    short_col = -short_res["pnl"]

    return pd.DataFrame({
        "return_long": long_col,
        "return_short": short_col,
        "return_long_raw": long_col,
        "return_short_raw": short_col,
    }, index=df.index)
