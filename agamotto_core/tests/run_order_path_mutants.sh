#!/usr/bin/env bash
# Negative controls for tests/order_path_driver.cpp.
#
# A test that has never failed proves nothing. Each mutation below breaks ONE
# behaviour of order_path_math.hpp and the suite must FAIL as a result; a
# mutant that survives means the property is unguarded, not that the code is
# good. Same intent as the --negative flags on run_*_parity.sh.
#
# Usage:  bash tests/run_order_path_mutants.sh [<sentinel_stan_code_dir>]
set -uo pipefail

# No default. A hardcoded path here silently tested whichever copy of the
# header happened to live on ONE machine -- on any other it failed with a
# confusing "no such header" instead of saying what it wanted. Ask for it.
if [ -z "${1:-}" ]; then
    echo "usage: $0 <path-to-stan_code>" >&2
    echo "  e.g. $0 ~/sandbox/sentinel/Strategy/ltp_release/ltp_strat_sdk/stan_code" >&2
    exit 2
fi
SDK="$1"
SRC="$SDK/order_path_math.hpp"
DRV="$(cd "$(dirname "$0")" && pwd)/order_path_driver.cpp"
[ -f "$SRC" ] || { echo "no order_path_math.hpp at $SRC"; exit 2; }

killed=0
survived=0

mutate() {
    local name="$1" old="$2" new="$3"
    rm -rf /tmp/opmut && mkdir -p /tmp/opmut
    OLD="$old" NEW="$new" SRC="$SRC" python3 - <<'PYEOF'
import os, pathlib, sys
s = pathlib.Path(os.environ["SRC"]).read_text()
old, new = os.environ["OLD"], os.environ["NEW"]
if old not in s:
    sys.stderr.write("anchor not found\n"); sys.exit(9)
pathlib.Path("/tmp/opmut/order_path_math.hpp").write_text(s.replace(old, new, 1))
PYEOF
    if [ $? -ne 0 ]; then
        echo "  ANCHOR MISSING  $name"; survived=$((survived + 1)); return
    fi
    if ! g++ -std=c++17 -ffp-contract=off -I /tmp/opmut -I "$SDK" "$DRV" \
            -o /tmp/opmut/drv 2>/dev/null; then
        echo "  DID NOT COMPILE $name"; survived=$((survived + 1)); return
    fi
    if /tmp/opmut/drv >/dev/null 2>&1; then
        echo "  SURVIVED        $name"; survived=$((survived + 1))
    else
        echo "  killed          $name"; killed=$((killed + 1))
    fi
}

echo "=== order path negative controls ==="

mutate "floorToStep rounds up instead of down" \
    "return std::floor(snapped_) * aStep;" \
    "return std::ceil(snapped_) * aStep;"

mutate "net_count cap not applied" \
    "return sign_ * ((mag_ < aMaxRungs) ? mag_ : aMaxRungs);" \
    "return aNet;"

mutate "qty ignores net_count (always 1x CAPITAL)" \
    "aCfg.capital_usd * std::fabs(static_cast<double>(targetNet_)) / mid_," \
    "aCfg.capital_usd * 1.0 / mid_,"

mutate "reverse accepts any value" \
    "if (aCfg.reverse != 1 && aCfg.reverse != -1) {" \
    "if (false) {"

mutate "crossed book accepted" \
    "if (aBook.ask < aBook.bid) return false;" \
    "if (false) return false;"

mutate "unsigned clock underflow reinstated" \
    "if (aNowNs > aBook.recv_ns && (aNowNs - aBook.recv_ns) > aStaleNs) return false;" \
    "if ((aNowNs - aBook.recv_ns) > aStaleNs) return false;"

mutate "reduce_only set on a flip too" \
    "const bool reduceOnly_ = (targetNet_ == 0);" \
    "const bool reduceOnly_ = true;"

mutate "cap clamps instead of halting" \
    "        p_.action = Action::HALT;
        p_.qty = 0.0;
        p_.why = \"position notional would exceed the cap\";
        return p_;" \
    "        // clamp instead of halt"

mutate "cap epsilon removed (boundary self-breach)" \
    "aCfg.max_position_notional_usd * (1.0 + POS_CAP_EPS)" \
    "aCfg.max_position_notional_usd"

mutate "ioc price snaps to nearest, not aggressive" \
    "    const double snapped_ = (aSide > 0) ? std::ceil(ticks_ - 1e-9)
                                        : std::floor(ticks_ + 1e-9);" \
    "    const double snapped_ = std::round(ticks_);"

mutate "stale book ignored" \
    "    if (!bookIsFresh(aBook, aNowNs, aCfg.quote_stale_ns)) {" \
    "    if (false) {"

echo
echo "=== killed: $killed   survived: $survived ==="
echo
echo "KNOWN EQUIVALENT MUTANT (deliberately not listed above):"
echo "  removing the 'already at target within one lot' guard changes NO"
echo "  behaviour -- the lot floor below it enforces the identical condition."
echo "  It is kept for log diagnostics, not for control flow. Listing it as a"
echo "  control would report a test gap that does not exist."

[ "$survived" -eq 0 ]
