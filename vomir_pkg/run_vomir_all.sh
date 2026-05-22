#!/bin/bash
# vomir/run_vomir_all.sh — Run Vomir classifier for all TFs × both universes.
#
# Runs SEQUENTIALLY (LightGBM memory safety).
# window-size 12: 11 months train + 1 month test per rolling window.
#
# Usage:
#   nohup bash vomir/run_vomir_all.sh > vomir/run_vomir_all.log 2>&1 &

set -e
PY="/home/ubuntu/miniconda3/envs/py313/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/agamotto_pkg/src:$ROOT"
cd "$ROOT"

OUTPUT="$ROOT/vomir/output"
WINDOW=12

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*"; }

run_clf() {
    local universe=$1
    local tf=$2
    log "  [vomir classifier] universe=$universe  tf=$tf  window=$WINDOW"
    $PY vomir/run_classifier.py \
        --tf "$tf" \
        --universe "$universe" \
        --window-size "$WINDOW" \
        --output "$OUTPUT" \
        > "$OUTPUT/${universe}_${tf}_classifier.log" 2>&1
    log "  done: $universe/$tf"
}

mkdir -p "$OUTPUT"

log "=== Vomir All — liquid + altcoins, all TFs, window=$WINDOW ==="

log "--- Liquid (pred_agamotto*_1) ---"
run_clf liquid 15m
run_clf liquid 1h
run_clf liquid 4h
run_clf liquid 1d

log "--- Altcoins (pred_agamotto*_2) ---"
run_clf altcoins 15m
run_clf altcoins 1h
run_clf altcoins 4h
run_clf altcoins 1d

log "=== ALL DONE — results in $OUTPUT ==="
