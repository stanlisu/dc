#!/bin/bash
# The stage-2.1 gate, end to end: emit the pandas golden on the HOST, then diff
# it against the C++ primitives inside the build image.
#
# Two hosts are involved because neither can do both jobs. The build image
# (mjolnir-core-build:latest, rockylinux:8) has ta-lib and gcc 8.5 but NO
# python; the workstation has pandas 2.3.3 but not the toolchain the .so must
# be built with. So the golden crosses the boundary as CSV, written with
# %.17g — which round-trips a double exactly, and which the driver reads with
# strtod for the same reason.
#
#   CAUTION for anyone re-reading these CSVs from Python: pass
#   float_precision="round_trip" to read_csv. The default C parser is NOT
#   exactly rounded and shifts values by ~1 ULP, which shows up as a 1e-11
#   "parity failure" that is entirely the parser.
#
# Usage:
#   tests/run_pdops_parity.sh                  # generate golden + run the gate
#   tests/run_pdops_parity.sh --golden-only    # just regenerate the CSVs
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="mjolnir-core-build:latest"
GOLDEN_DIR="$HERE/build-linux/golden"
DRIVER="$HERE/build-linux/pdops_parity_driver"

GOLDEN_ONLY=0
[ "${1:-}" = "--golden-only" ] && GOLDEN_ONLY=1

PY="${PDOPS_PYTHON:-python3}"
if ! "$PY" -c 'import pandas' >/dev/null 2>&1; then
    # Loud, not skipped: a parity harness that quietly does nothing when its
    # reference is missing reports a green that means nothing.
    echo "FAIL: $PY has no pandas. The golden IS pandas — refusing to run a" >&2
    echo "      parity gate with no reference. Set PDOPS_PYTHON=/path/to/python." >&2
    exit 1
fi

echo "=== emitting the pandas golden ==="
"$PY" "$HERE/tests/pdops_golden.py" --out-dir "$GOLDEN_DIR"

[ "$GOLDEN_ONLY" = "1" ] && exit 0

if [ ! -x "$DRIVER" ]; then
    echo "FAIL: $DRIVER not built. Run ./build_linux.sh first." >&2
    exit 1
fi

echo
echo "=== diffing the C++ primitives against it in $IMAGE ==="
docker run --rm -v "$HERE:/src" -w /src "$IMAGE" \
    ./build-linux/pdops_parity_driver \
    build-linux/golden/pdops_input.csv build-linux/golden/pdops_golden.csv
