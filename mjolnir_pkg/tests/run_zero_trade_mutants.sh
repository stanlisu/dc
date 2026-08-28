#!/usr/bin/env bash
# Negative controls for mjolnir_pkg/tests/test_zero_trade_fill.py.
#
# A test that has never failed proves nothing. Each mutation below breaks ONE
# behaviour of the zero-trade change and the suite MUST fail as a result. A
# mutant that SURVIVES means that property is unguarded, not that the code is
# good. Same intent as agamotto_core/tests/run_order_path_mutants.sh.
#
# The mutation is applied to a COPY of mjolnir_pkg/src, never to the tree, so
# an interrupted run cannot leave a mutated source behind.
#
# Usage:  bash mjolnir_pkg/tests/run_zero_trade_mutants.sh [<python>]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/mjolnir_pkg/src"
TESTS="$ROOT/mjolnir_pkg/tests/test_zero_trade_fill.py"
PY="${1:-python3}"
WORK="${TMPDIR:-/tmp}/mjolnir_zt_mutants"

[ -d "$SRC/mjolnir/core" ] || { echo "no mjolnir/core under $SRC"; exit 2; }

killed=0
survived=0

# mutate <name> <relative-file> <old> <new> [pytest -k selector]
mutate() {
    local name="$1" rel="$2" old="$3" new="$4" sel="${5:-}"
    rm -rf "$WORK" && mkdir -p "$WORK"
    cp -R "$SRC/." "$WORK/"

    OLD="$old" NEW="$new" TARGET="$WORK/$rel" "$PY" - <<'PYEOF'
import os
import pathlib
import sys
p = pathlib.Path(os.environ["TARGET"])
s = p.read_text()
old, new = os.environ["OLD"], os.environ["NEW"]
if old not in s:
    sys.stderr.write("anchor not found\n")
    sys.exit(9)
p.write_text(s.replace(old, new, 1))
PYEOF
    if [ $? -ne 0 ]; then
        echo "  ANCHOR MISSING  $name"
        survived=$((survived + 1))
        return
    fi

    local args=(-q -p no:cacheprovider "$TESTS")
    [ -n "$sel" ] && args+=(-k "$sel")
    if PYTHONPATH="$WORK" "$PY" -m pytest "${args[@]}" >/dev/null 2>&1; then
        echo "  SURVIVED        $name"
        survived=$((survived + 1))
    else
        echo "  killed          $name"
        killed=$((killed + 1))
    fi
}

A="mjolnir/core/aligner.py"
R="mjolnir/core/research.py"
F="mjolnir/core/features.py"

echo "=== zero-trade fill negative controls ==="

# --- the builder flag -------------------------------------------------------
mutate "aligner: fill_zero_trade gets a default" "$A" \
    "*, fill_zero_trade: bool) -> None:" \
    "*, fill_zero_trade: bool = True) -> None:" \
    "has_no_default"

mutate "aligner: fills the price columns unconditionally" "$A" \
    "        if self.fill_zero_trade:
            # Legacy convention" \
    "        if True:
            # Legacy convention" \
    "leaves_ohlc_nan"

mutate "aligner: never fills (identity with True is broken)" "$A" \
    "        if self.fill_zero_trade:
            # Legacy convention" \
    "        if False:
            # Legacy convention" \
    "reproduces_legacy or fills_the_interior_gap"

mutate "aligner: ffill replaced by bfill (identity is broken)" "$A" \
    'agg["close"] = agg["close"].ffill()' \
    'agg["close"] = agg["close"].bfill()' \
    "reproduces_legacy"

mutate "aligner: has_trade is constant True" "$A" \
    'agg["has_trade"] = agg["n_trades"].fillna(0.0) > 0' \
    'agg["has_trade"] = pd.Series(True, index=agg.index)' \
    "has_trade or HasTrade"

mutate "aligner: flow columns left NaN on zero-trade bars" "$A" \
    '        for col in ["volume", "n_trades", "buy_vol", "sell_vol"]:
            agg[col] = agg[col].fillna(0.0)' \
    '        for col in []:
            agg[col] = agg[col].fillna(0.0)' \
    "flow"

# --- the three LEGITIMATE ffills (must never become conditional) ------------
mutate "aligner: book_ticker ffill disabled" "$A" \
    "        last = last.reindex(bar_index).ffill()
        return last" \
    "        last = last.reindex(bar_index)
        return last" \
    "book_ticker_still_ffills"

mutate "aligner: book_snapshot ffill disabled" "$A" \
    'last = snap.groupby("bar").last().reindex(bar_index).ffill()' \
    'last = snap.groupby("bar").last().reindex(bar_index)' \
    "book_snapshot_still_ffills"

mutate "aligner: derivative ffill disabled" "$A" \
    "        last = df_sub.groupby(\"bar\").last()
        last = last.reindex(bar_index).ffill()" \
    "        last = df_sub.groupby(\"bar\").last()
        last = last.reindex(bar_index)" \
    "derivative_ticker_still_ffills"

# --- the load-time guard ----------------------------------------------------
mutate "research: re-fills silently under a no-fill tree" "$R" \
    "    if fill_zero_trade:
        df[\"close\"] = df[\"close\"].ffill()" \
    "    if True:
        df[\"close\"] = df[\"close\"].ffill()" \
    "TestLoadTimePolicy"

mutate "research: load guard is a no-op" "$R" \
    "        _assert_nan_prices_are_zero_trade(df, label)" \
    "        pass" \
    "rejects"

mutate "research: FILL_ZERO_TRADE defaults to true when absent" "$R" \
    '    if "FILL_ZERO_TRADE" not in config:
        raise KeyError(' \
    '    if "FILL_ZERO_TRADE" not in config:
        return True
    if False:
        raise KeyError(' \
    "missing_key_raises"

mutate "research: FILL_ZERO_TRADE accepts a truthy string" "$R" \
    "    if not isinstance(val, bool):" \
    "    if False:" \
    "non_boolean_raises"

# --- the features fill split ------------------------------------------------
mutate "features: blanket fill regardless of the flag" "$F" \
    "        if self.zero_fill_prices:" \
    "        if True:" \
    "TestFeatureFillSplit or TestTwentyOneColumnPoisoning"

mutate "features: no fill at all (flow columns lose their 0.0)" "$F" \
    "            flow_cols = self._zero_fill_columns(feature_cols)" \
    "            flow_cols = []" \
    "flow_columns_are_still_zero_filled"

mutate "features: prices join the zero-fill allowlist" "$F" \
    '        return [c for c in cols if c.startswith(self._ZERO_FILL_PREFIXES)]' \
    '        return list(cols)' \
    "price_columns_are_never_zero_filled or poisoning_absent"

mutate "features: default flips away from the live convention" "$F" \
    "        zero_fill_prices: bool = True," \
    "        zero_fill_prices: bool = False," \
    "default_is_the_legacy_blanket_fill"

rm -rf "$WORK"
echo
echo "killed=$killed survived=$survived"
[ "$survived" -eq 0 ] || exit 1
