#!/usr/bin/env bash
# Negative controls for tests/mm_ladder_driver.cpp.
#
# Every rule in mm_ladder.hpp came from a knull incident. A rule with no mutant
# is a rule nobody has checked -- and MM is being built with NO incremental live
# validation, so the offline gate is the only gate there is until the shadow run.
#
# Usage:  bash tests/run_mm_ladder_mutants.sh [<sentinel_stan_code_dir>]
set -uo pipefail

SDK="${1:-/home/stan/sandbox/sentinel/Strategy/ltp_release/ltp_strat_sdk/stan_code}"
SRC="$SDK/mm_ladder.hpp"
DRV="$(cd "$(dirname "$0")" && pwd)/mm_ladder_driver.cpp"
[ -f "$SRC" ] || { echo "no mm_ladder.hpp at $SRC"; exit 2; }

killed=0
survived=0

mutate() {
    local name="$1" old="$2" new="$3"
    rm -rf /tmp/mmmut && mkdir -p /tmp/mmmut
    OLD="$old" NEW="$new" SRC="$SRC" python3 - <<'PYEOF'
import os, pathlib, sys
s = pathlib.Path(os.environ["SRC"]).read_text()
old, new = os.environ["OLD"], os.environ["NEW"]
if old not in s:
    sys.stderr.write("anchor not found\n"); sys.exit(9)
pathlib.Path("/tmp/mmmut/mm_ladder.hpp").write_text(s.replace(old, new, 1))
PYEOF
    if [ $? -ne 0 ]; then
        echo "  ANCHOR MISSING  $name"; survived=$((survived + 1)); return
    fi
    if ! g++ -std=c++17 -ffp-contract=off -I /tmp/mmmut -I "$SDK" "$DRV" \
            -o /tmp/mmmut/drv 2>/dev/null; then
        echo "  DID NOT COMPILE $name"; survived=$((survived + 1)); return
    fi
    if /tmp/mmmut/drv >/dev/null 2>&1; then
        echo "  SURVIVED        $name"; survived=$((survived + 1))
    else
        echo "  killed          $name"; killed=$((killed + 1))
    fi
}

# BASELINE FIRST. A mutation suite is MEANINGLESS if the unmutated driver
# fails: every mutant then "kills" trivially and the run reports a clean sweep
# it did not earn. That happened on 2026-08-26 -- 38/38 killed while the
# baseline was red from a bad test constant. Check it, and refuse.
echo "=== baseline (must pass before any mutant means anything) ==="
if ! g++ -std=c++17 -ffp-contract=off -I "$SDK" "$DRV" -o /tmp/mmbase 2>/dev/null; then
    echo "  BASELINE DID NOT COMPILE -- mutants would be meaningless"; exit 2
fi
if ! /tmp/mmbase >/dev/null 2>&1; then
    echo "  BASELINE FAILS -- fix the driver before trusting any mutant result"
    /tmp/mmbase 2>&1 | grep FAIL | head -5
    exit 2
fi
echo "  baseline passes"
echo
echo "=== mm ladder negative controls ==="

# --- entry -----------------------------------------------------------------
mutate "signal clamp removed (quotes outside the signal price)" \
    "    if (!(aSignalClose > 0.0)) return touch_;" \
    "    if (true) return touch_;"

mutate "rungs walk TOWARD the market instead of away" \
    "        const double raw_ = (aSide > 0) ? aAnchor - i * aTick" \
    "        const double raw_ = (aSide > 0) ? aAnchor + i * aTick"

# --- mm aim ----------------------------------------------------------------
mutate "aim rounds to NEAREST on the long leg (aim becomes an average)" \
    "        const double a_ = snapCeil(aim_, aCfg.price_tick);" \
    "        const double a_ = snapNearest(aim_, aCfg.price_tick);"

mutate "aim rounds to NEAREST on the short leg" \
    "    const double a_ = snapFloor(aim_, aCfg.price_tick);" \
    "    const double a_ = snapNearest(aim_, aCfg.price_tick);"

mutate "aim ignores the touch (post-only would cross)" \
    "        return (a_ > t_) ? a_ : t_;" \
    "        return a_;"

# --- exit ------------------------------------------------------------------
mutate "exit anchor takes the AGGRESSIVE side" \
    "    return (aSide > 0) ? ((touch_ < aBook.last) ? touch_ : aBook.last)" \
    "    return (aSide > 0) ? ((touch_ > aBook.last) ? touch_ : aBook.last)"

mutate "phase B step derived from the cap instead of the hardcoded /3.0" \
    "    const double step_ = aCfg.crossing_sec / 3.0;" \
    "    const double step_ = aCfg.crossing_sec / 2.0;"

mutate "phase B depth not capped at EXIT_DEPTH_MAX" \
    "    if (aCfg.exit_depth_max > 0 && d_ > aCfg.exit_depth_max) {" \
    "    if (false) {"

# --- the matcher -----------------------------------------------------------
mutate "diff rebuilds instead of matching (cancel storm, loses queue position)" \
    "                live_kept_[i] = true;" \
    "                live_kept_[i] = false;"

# --- position cap ----------------------------------------------------------
mutate "cap counts the unfilled remainder instead of the full ordered qty" \
    "        if (r_.active && !r_.cancelling) live_ += r_.qty;" \
    "        if (r_.active && !r_.cancelling) live_ += r_.qty - r_.filled;"

mutate "cap lets the first ladder through by accident (capacity guard gone)" \
    "    if (!(aState.tier_capacity > 0.0)) return false;
    double live_ = 0.0;" \
    "    if (false) return false;
    double live_ = 0.0;"

# --- tier gate -------------------------------------------------------------
mutate "tier gate not conjunctive (fires on the move alone)" \
    "    if (aState.filled_qty < need_ - 1e-9) return false;" \
    "    if (false) return false;"

mutate "tier trigger not sticky (unarms when the market recovers)" \
    "    if (aState.tier_triggered[aState.level]) return true;" \
    "    if (false) return true;"

mutate "tier fires past LADDER" \
    "    if (aState.level >= aCfg.ladder) return false;          // no tiers left" \
    "    if (false) return false;          // no tiers left"

# --- fill accounting -------------------------------------------------------
mutate "late fill on a CANCELLING rung is dropped" \
    "        const double prev_ = s_.filled_qty;" \
    "        if (r_.cancelling) return s_;
        const double prev_ = s_.filled_qty;"

mutate "implausible fill ignored instead of halting" \
    "            s_.halted = true;
            return s_;" \
    "            return s_;"

mutate "fill for an unknown uid is booked anyway" \
    "    if (idx_ < 0 || aResp.uid == 0) return s_;" \
    "    if (idx_ < 0 || aResp.uid == 0) { s_.filled_qty += aResp.qty; return s_; }"

mutate "avg cost takes the last price instead of the weighted mean" \
    "            s_.avg_cost = (prev_ * s_.avg_cost + aResp.qty * aResp.price)" \
    "            s_.avg_cost = 0.0 * (prev_ * s_.avg_cost + aResp.qty * aResp.price)"

mutate "fully filled rung stays in the live set" \
    "        if (r_.filled >= r_.qty - 1e-12) r_.active = false;" \
    "        if (false) r_.active = false;"

mutate "a reject leaves the rung live (phantom resting order)" \
    "        r_.active = false;
        ++s_.rejects;" \
    "        ++s_.rejects;"

# --- the state machine -----------------------------------------------------
mutate "entry timeout ABANDONS a partially filled position (-> FLAT)" \
    "    if (s_.filled_qty > 1e-12) {
        s_.phase = Phase::OPEN;
        return s_;
    }" \
    "    if (false) {
        s_.phase = Phase::OPEN;
        return s_;
    }"

mutate "entry timeout never fires (stuck ENTERING forever)" \
    "    if (aNowSec - s_.entered_at < aCfg.entry_timeout_sec) return s_;" \
    "    if (true) return s_;"

mutate "a fill during ENTERING does not open the position" \
    "        if (s_.phase == Phase::ENTERING) s_.phase = Phase::OPEN;" \
    "        if (false) s_.phase = Phase::OPEN;"

mutate "FLAT patches fields instead of wiping (stale cost basis survives)" \
    "    LadderState s_{};                 // everything default-constructed" \
    "    LadderState s_ = aState;"

mutate "net-zero reducing fill does not go FLAT" \
    "        if (aResp.reduce_only && s_.filled_qty <= 1e-12) return goFlat(s_);" \
    "        if (false) return goFlat(s_);"

mutate "a reducing fill ADDS to the position instead of reducing it" \
    "        s_.filled_qty += aResp.reduce_only ? -aResp.qty : aResp.qty;" \
    "        s_.filled_qty += aResp.qty;"

mutate "exit price folded into the cost basis (aim chases its own fills)" \
    "        if (!aResp.reduce_only && aResp.price > 0.0 && s_.filled_qty > 0.0) {" \
    "        if (aResp.price > 0.0 && s_.filled_qty > 0.0) {"

mutate "tier capacity keeps growing past the first ladder" \
    "    if (s_.phase == Phase::ENTERING && s_.level <= 1) {" \
    "    if (true) {"

mutate "crossing phase starts immediately (no passive window)" \
    "    return (aNowSec - aState.exit_started_at) >= aCfg.passive_sec;" \
    "    return true;"

mutate "phase B elapsed includes the passive window (depth walks too fast)" \
    "    const double e_ = aNowSec - aState.exit_started_at - aCfg.passive_sec;" \
    "    const double e_ = aNowSec - aState.exit_started_at;"

# --- stoploss --------------------------------------------------------------
mutate "stop marked at MID instead of the exit-side touch" \
    "    const double touch_ = (aState.side > 0) ? aBook.bid : aBook.ask;
    if (!(touch_ > 0.0)) return false;

    // A COST BASIS" \
    "    const double touch_ = 0.5 * (aBook.bid + aBook.ask);
    if (!(touch_ > 0.0)) return false;

    // A COST BASIS"

mutate "stop fires on an implausible cost basis (phantom stoploss)" \
    "        if (ratio_ < 0.66 || ratio_ > 1.5) return false;" \
    "        if (false) return false;"

mutate "stop sign inverted (fires on profit)" \
    "    return pnl_ <= -aCfg.stoploss_frac;" \
    "    return pnl_ >= aCfg.stoploss_frac;"

# --- trailing --------------------------------------------------------------
mutate "unknown ATR treated as usable (stop at the touch)" \
    "    if (!(aAtr > 0.0) || !(aCfg.trail_trigger_atr > 0.0)) return false;" \
    "    if (!(aCfg.trail_trigger_atr > 0.0)) return false;"

mutate "trailing peak LATCHES instead of clearing below the trigger" \
    "        s_.trail_armed = false;
        s_.trail_peak = 0.0;
        return s_;" \
    "        return s_;"

mutate "trailing arms below the trigger" \
    "    return move_ >= aCfg.trail_trigger_atr * aAtr;" \
    "    return true;"

mutate "trailing exits on any pullback, ignoring the distance" \
    "    return draw_ >= aCfg.trail_distance_atr * aAtr;" \
    "    return draw_ > 0.0;"

mutate "trailing peak ratchets the WRONG way on a long" \
    "        if (touch_ > s_.trail_peak) s_.trail_peak = touch_;" \
    "        if (touch_ < s_.trail_peak) s_.trail_peak = touch_;"

echo
echo "=== killed: $killed   survived: $survived ==="
[ "$survived" -eq 0 ]
