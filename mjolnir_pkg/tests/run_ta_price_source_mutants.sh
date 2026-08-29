#!/usr/bin/env bash
# Negative controls for mjolnir_pkg/tests/test_ta_price_source.py.
#
# A test that has never failed proves nothing. Each mutation below breaks ONE
# behaviour of the TA price-source change and the suite MUST fail as a result.
# A mutant that SURVIVES means that property is unguarded, not that the code is
# good. Same intent as run_zero_trade_mutants.sh.
#
# The mutation is applied to a COPY of mjolnir_pkg/src, never to the tree, so
# an interrupted run cannot leave a mutated source behind.
#
# Usage:  bash mjolnir_pkg/tests/run_ta_price_source_mutants.sh [<python>]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/mjolnir_pkg/src"
TESTS="$ROOT/mjolnir_pkg/tests/test_ta_price_source.py"
PY="${1:-python3}"
WORK="${TMPDIR:-/tmp}/mjolnir_taps_mutants"

[ -d "$SRC/mjolnir/core" ] || { echo "no mjolnir/core under $SRC"; exit 2; }

killed=0
survived=0

# mutate <name> <relative-file> <old> <new> [pytest -k selector] [test file]
#
# The 6th arg overrides the test file this mutant is scored against. It exists
# because ta_price_source is CONSUMED outside mjolnir_pkg: agamotto's
# test_scale_free_levels.py drives MjolnirFeatures._compute_price_features
# directly, and one class of mutant (a value readable WITHOUT the constructor)
# is invisible to every test in this package — see the class-attribute mutant
# at the bottom of this file.
mutate() {
    local name="$1" rel="$2" old="$3" new="$4" sel="${5:-}" tests="${6:-$TESTS}"
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

    local args=(-q -p no:cacheprovider "$tests")
    [ -n "$sel" ] && args+=(-k "$sel")
    # agamotto_pkg/src is on the path unconditionally: harmless for mjolnir's
    # own tests, required for the cross-package ones. $WORK comes FIRST so the
    # mutated mjolnir shadows any installed copy.
    if PYTHONPATH="$WORK:$ROOT/agamotto_pkg/src" "$PY" -m pytest "${args[@]}" >/dev/null 2>&1; then
        echo "  SURVIVED        $name"
        survived=$((survived + 1))
    else
        echo "  killed          $name"
        killed=$((killed + 1))
    fi
}

F="mjolnir/core/features.py"
R="mjolnir/core/research.py"

echo "=== TA price-source negative controls ==="

# --- the flag ---------------------------------------------------------------
mutate "features: ta_price_source gets a default" "$F" \
    "        *,
        ta_price_source: str,
    ) -> None:" \
    "        *,
        ta_price_source: str = \"close\",
    ) -> None:" \
    "has_no_default"

mutate "features: any string is accepted as a source" "$F" \
    "        if ta_price_source not in TA_PRICE_SOURCES:" \
    "        if False:" \
    "unknown_source"

# --- the SOURCE: snapshot L0, not book_ticker L1 ----------------------------
mutate "features: book mid reads the book_ticker L1 columns" "$F" \
    'BOOK_MID_SOURCE_COLS: tuple = ("bids_0_price", "asks_0_price")' \
    'BOOK_MID_SOURCE_COLS: tuple = ("bid_price", "ask_price")'

mutate "features: book mid computed from bid_price/ask_price directly" "$F" \
    "    return (df[BOOK_MID_SOURCE_COLS[0]] + df[BOOK_MID_SOURCE_COLS[1]]) / 2" \
    '    return (df["bid_price"] + df["ask_price"]) / 2'

mutate "features: missing book columns fall through instead of raising" "$F" \
    "    missing = [c for c in BOOK_MID_SOURCE_COLS if c not in df.columns]
    if missing:" \
    "    missing = [c for c in BOOK_MID_SOURCE_COLS if c not in df.columns]
    if False:" \
    "missing_column"

# --- the flag actually reaching TA-Lib --------------------------------------
mutate "features: flag ignored, TA always on the trade close" "$F" \
    '        if self.ta_price_source == "close":
            return close, high, low
        mid = book_mid_price(df)' \
    '        if True:
            return close, high, low
        mid = book_mid_price(df)'

mutate "features: high/low kept from the trade OHLC under book_mid" "$F" \
    "        mid = book_mid_price(df)
        return mid, mid, mid" \
    "        mid = book_mid_price(df)
        return mid, high, low" \
    "high_low"

# --- the NaN policy ---------------------------------------------------------
mutate "features: no compaction, book_mid inherits NaN propagation" "$F" \
    '        if self.ta_price_source == "close":
            return None
        keep = close.notna()' \
    '        if True:
            return None
        keep = close.notna()' \
    "TestBookGaps"

mutate "features: compaction applied on the close path too" "$F" \
    '        if self.ta_price_source == "close":
            return None
        keep = close.notna()' \
    '        if False:
            return None
        keep = close.notna()' \
    "does_not_drop_nan_rows"

mutate "features: NaN rows are back-filled instead of dropped" "$F" \
    "            c, h, lo, v = close[keep], high[keep], low[keep], volume[keep]" \
    "            c, h, lo, v = (close.bfill().ffill(), high.bfill().ffill(),
                           low.bfill().ffill(), volume)" \
    "rows_without_a_book_stay_nan"

# --- the scale-free normaliser ----------------------------------------------
mutate "features: scale-free transforms keep normalising by the trade close" "$F" \
    '            sf_src = df if self.ta_price_source == "close" \
                else df.assign(close=ta_close)' \
    "            sf_src = df" \
    "book_mid_makes_the_21_dense"

# --- the TARGET must not move ----------------------------------------------
mutate "features: target repointed at the book mid" "$F" \
    '        target_mid = df.get("mid_price", df.get("close"))' \
    '        target_mid = (book_mid_price(df)
                      if self.ta_price_source == "book_mid"
                      else df.get("mid_price", df.get("close")))' \
    "TestTargetUnchanged"

# --- the setting.json resolver ---------------------------------------------
mutate "research: TA_PRICE_SOURCE silently defaults" "$R" \
    '    if "TA_PRICE_SOURCE" not in config:' \
    "    if False:" \
    "missing_key"

mutate "research: value is not checked against the accepted set" "$R" \
    "    if not isinstance(val, str) or val not in TA_PRICE_SOURCES:" \
    "    if not isinstance(val, str):" \
    "invalid_value"

# --- the default, re-added where no signature shows it ----------------------
#
# The first mutant in this file puts the default back in the __init__
# SIGNATURE, and test_ta_price_source_has_no_default kills it. This one puts
# the same default on the CLASS instead. The signature is untouched, so
# `MjolnirFeatures()` still raises TypeError and every test in THIS package
# passes — the mutant survives all 28 of them. What it silently restores is
# the read path: `cls.__new__(cls)` gets a working `self.ta_price_source`
# again, so any caller that skips __init__ picks up "close" without ever
# stating it. That is exactly the magic default dc #62 deleted.
#
# It is scored against agamotto's suite because that is where the only
# constructor-skipping caller lives (test_scale_free_levels.py — the one that
# turned main red on 2026-08-28, CI run 33146271206).
mutate "features: default re-added as a CLASS attribute, signature untouched" "$F" \
    'class MjolnirFeatures:
    """Compute microstructure' \
    'class MjolnirFeatures:
    ta_price_source = "close"
    """Compute microstructure' \
    "skipped_its_constructor" \
    "$ROOT/agamotto_pkg/tests/test_scale_free_levels.py"

echo
echo "killed=$killed survived=$survived"
[ "$survived" -eq 0 ] || exit 1
