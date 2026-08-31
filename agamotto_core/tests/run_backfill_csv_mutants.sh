#!/usr/bin/env bash
# Negative controls for tests/backfill_csv_driver.cpp.
#
# A test that has never failed proves nothing. Every rule in backfill_csv.hpp is
# one the OLD parser enforced by accident of using std::stod and std::getline;
# the rewrite has to enforce each one deliberately, and a rule with no mutant is
# a rule nobody has checked.
#
# Why that matters more here than it looks: this parser feeds loadBackfill(),
# which turns a parse failure into mHalted. A mutant that makes the parser too
# LOOSE ships bad history into a 700-bar rolling window silently; one that makes
# it too STRICT is a fleet that will not boot. Both directions are represented.
#
# Usage:  bash tests/run_backfill_csv_mutants.sh <sentinel_stan_code_dir> [real_csv_dir]
set -uo pipefail

# No default. A hardcoded path here silently tested whichever copy of the header
# happened to live on ONE machine -- on any other it failed with a confusing
# "no such header" instead of saying what it wanted. Ask for it.
if [ -z "${1:-}" ]; then
    echo "usage: $0 <path-to-stan_code> [dir-of-real-backfill-csvs]" >&2
    echo "  e.g. $0 ~/sandbox/sentinel/Strategy/ltp_release/ltp_strat_sdk/stan_code" >&2
    exit 2
fi
SDK="$1"
REAL="${2:-}"
SRC="$SDK/backfill_csv.hpp"
DRV="$(cd "$(dirname "$0")" && pwd)/backfill_csv_driver.cpp"
[ -f "$SRC" ] || { echo "no backfill_csv.hpp at $SRC"; exit 2; }

# THE BASELINE RUN. Without it a suite that fails on the UNMUTATED header would
# report every mutant as killed and pass with flying colours.
echo "=== baseline (unmutated) ==="
if ! g++ -std=c++17 -Wall -Wextra -I "$SDK" "$DRV" -o /tmp/bfmut_base 2>&1; then
    echo "BASELINE DID NOT COMPILE"; exit 1
fi
if ! /tmp/bfmut_base $REAL; then
    echo "BASELINE FAILS -- fix the parser before reading any mutant result"; exit 1
fi

killed=0
survived=0

mutate() {
    local name="$1" old="$2" new="$3"
    rm -rf /tmp/bfmut && mkdir -p /tmp/bfmut
    OLD="$old" NEW="$new" SRC="$SRC" python3 - <<'PYEOF'
import os, pathlib, sys
s = pathlib.Path(os.environ["SRC"]).read_text()
old, new = os.environ["OLD"], os.environ["NEW"]
if old not in s:
    sys.stderr.write("anchor not found\n"); sys.exit(9)
pathlib.Path("/tmp/bfmut/backfill_csv.hpp").write_text(s.replace(old, new, 1))
PYEOF
    if [ $? -ne 0 ]; then
        echo "  ANCHOR MISSING  $name"; survived=$((survived + 1)); return
    fi
    if ! g++ -std=c++17 -I /tmp/bfmut -I "$SDK" "$DRV" \
            -o /tmp/bfmut/drv 2>/dev/null; then
        echo "  DID NOT COMPILE $name"; survived=$((survived + 1)); return
    fi
    if /tmp/bfmut/drv $REAL >/dev/null 2>&1; then
        echo "  SURVIVED        $name"; survived=$((survived + 1))
    else
        echo "  killed          $name"; killed=$((killed + 1))
    fi
}

echo
echo "=== mutants ==="

# TOO LOOSE. A row with nine fields read as nine values and a zero leaves a bar
# with volume 0 in the rolling window -- a "correction" that causes the defect
# it exists to repair.
mutate "short row accepted (field-count guard neutered)" \
    "		if (nFields < kColumns) {
			err.kind = ParseError::Kind::FIELD_COUNT;" \
    "		if (false) {
			err.kind = ParseError::Kind::FIELD_COUNT;"

# TOO LOOSE. end == f.beg is the ONLY signal that a field held no number at all;
# without it a garbage field silently reads as strtoll's 0.
mutate "integer field with no number accepted (endptr check dropped)" \
    "	const long long v = std::strtoll(f.beg, &end, 10);" \
    "	const long long v = std::strtoll(f.beg, &end, 10);
	if (true) { out = static_cast<int64_t>(v); return true; }"

mutate "double field with no number accepted (endptr check dropped)" \
    "	const double v = std::strtod(f.beg, &end);" \
    "	const double v = std::strtod(f.beg, &end);
	if (true) { out = v; return true; }"

# TOO LOOSE. std::stod threw out_of_range on overflow; strtod merely sets ERANGE
# and hands back HUGE_VAL. Dropping the check turns 1e999 into inf in a price.
mutate "out-of-range value accepted (ERANGE ignored)" \
    "	if (end == f.beg || errno == ERANGE) return false;
	out = static_cast<int64_t>(v);" \
    "	if (end == f.beg) return false;
	out = static_cast<int64_t>(v);"

mutate "out-of-range double accepted (ERANGE ignored)" \
    "	if (end == f.beg || errno == ERANGE) return false;
	out = v;" \
    "	if (end == f.beg) return false;
	out = v;"

# An empty field is the one case where reading in place could quietly succeed by
# running on into the NEXT field's bytes -- ',' stops strtod, but only if the
# zero-length guard has not already been removed.
mutate "empty field accepted (zero-length guard dropped from toF64)" \
    "inline bool toF64(const Field& f, double& out)
{
	if (f.len == 0) return false;" \
    "inline bool toF64(const Field& f, double& out)
{"

# DERIVED FIELDS. bucket_close_ms, aggressor_source and from_backfill are not in
# the file; nothing downstream can tell a wrong one from a right one.
mutate "bucket_close_ms off by one bar (the -1 dropped)" \
    "		    b.bucket_open_ms + static_cast<int64_t>(bar_sec) * 1000 - 1;" \
    "		    b.bucket_open_ms + static_cast<int64_t>(bar_sec) * 1000;"

mutate "backfill bars claim QUOTE_RULE instead of venue-exact" \
    "		b.aggressor_source = KlineBar::AggressorSource::EXACT_MAKER_FLAG;" \
    "		b.aggressor_source = KlineBar::AggressorSource::QUOTE_RULE;"

mutate "from_backfill left unset (a spliced bar looks locally built)" \
    "		b.from_backfill = true;" \
    "		b.from_backfill = false;"

# TOO STRICT, and this is the direction that will not boot. The first line is
# column names; a parser that tries to read it rejects every real file.
mutate "header line parsed instead of discarded" \
    "		if (lineNo == 1 || llen == 0) {" \
    "		if (llen == 0) {"

# TOO STRICT. A blank line becomes a one-field row and halts the run.
mutate "blank lines no longer skipped" \
    "		if (lineNo == 1 || llen == 0) {" \
    "		if (lineNo == 1) {"

# TOO STRICT on CRLF, and silently so: the '\r' rides on the last field.
mutate "CRLF not stripped" \
    "		if (end > pos && data[end - 1] == '\\r') --end;" \
    "		if (false) --end;"

# A file that yields no rows must be an ERROR. Returning true with an empty
# vector reads to every caller as "checked, nothing to fix".
mutate "empty result reported as success" \
    "	if (out.empty()) {
		err.kind = ParseError::Kind::EMPTY;
		return false;
	}" \
    "	if (out.empty()) {
		err.kind = ParseError::Kind::EMPTY;
	}"

# The last row of a file with no trailing newline must not be dropped -- the
# fetcher's atomic rename can land a file either way.
mutate "final row dropped when the file has no trailing newline" \
    "		if (atEnd) break;
	}

	if (out.empty()) {" \
    "		if (atEnd) { out.pop_back(); break; }
	}

	if (out.empty()) {"

# Error line numbers have to survive skipped lines, or an operator is sent to a
# row that is not the broken one.
mutate "line numbers do not count skipped lines" \
    "		++lineNo;" \
    "		if (llen != 0) ++lineNo;"

echo
echo "=== killed: $killed   survived: $survived ==="
[ "$survived" -eq 0 ]
