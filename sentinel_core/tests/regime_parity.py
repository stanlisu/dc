#!/usr/bin/env python3
"""Differential parity: C++ applyFilterMask == reference apply_filter_mask.

Compares the boolean mask for every supported regime x position, plus composed
(_and_ / _or_) expressions and the coded-name form, over the same feature panel.

Why this matters more than its size suggests: several predicates are
QUANTILE-based over the whole panel. An off-by-one in the quantile, or the
wrong interpolation, produces a mask that is mostly right — which is exactly the
failure that survives eyeballing and quietly changes which bars trade.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bar_parity import BAR_SEC, TARGET_SEC, gen_events, load_reference  # noqa: E402
from feature_parity import (  # noqa: E402
    assert_reference_used_talib, ref_bars_df, require_reference_talib)

REGIMES = [
    ("trade_imbalance", "long"), ("trade_imbalance", "short"),
    ("deep_book", "long"), ("deep_book", "short"),
    ("wide_spread", "long"), ("tight_spread", "long"),
    ("high_liquidation_pressure", "long"), ("low_liquidation_pressure", "long"),
    ("funding_positive", "long"), ("funding_negative", "long"),
    ("basis_premium", "long"), ("basis_discount", "long"),
    ("pre_funding_settlement", "long"),
    ("ofi_positive", "long"), ("ofi_positive", "short"),
    ("rsi_oversold", "long"), ("rsi_overbought", "long"),
    ("macd_bullish", "long"), ("macd_bearish", "short"),
    ("adx_trend", "long"), ("vol_breakout", "long"), ("high_volume", "long"),
    ("low_volume", "long"), ("high_vol", "long"), ("low_vol", "long"),
    ("mom_positive", "long"), ("mom_positive", "short"),
    # composition + the deployed suffixed form + a coded name
    ("trade_imbalance_and_wide_spread", "long"),
    ("ofi_positive_or_deep_book", "short"),
    ("trade_imbalance_long", "long"),
    ("r068", "short"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-bar", required=True)
    ap.add_argument("--ref-feat", required=True)
    ap.add_argument("--pkg-src", required=True,
                    help="dc/mjolnir_pkg/src — so relative imports in the reference resolve")
    ap.add_argument("--driver", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # Five regimes below (rsi_oversold, rsi_overbought, macd_bullish,
    # macd_bearish, adx_trend) read TA-Lib columns directly, so a reference
    # without TA-Lib grades them against numpy stubs and NaN.
    ta_ver = require_reference_talib()
    print(f"[regime_parity] reference TA-Lib {ta_ver}")

    events = gen_events(seed=args.seed)
    bars = ref_bars_df(load_reference(Path(args.ref_bar)), events)
    feat_mod = load_reference(Path(args.ref_feat))
    fe = feat_mod.MjolnirFeatures(feature_windows=[30, 60, 300, 900],
                                  bar_tf="5s", target_tf="5s")
    panel = fe.compute(bars)
    assert_reference_used_talib(panel, "reference panel")

    # Import as a package member: regime_filters uses relative imports
    # (the codec), so loading it as a standalone file always ImportErrors.
    sys.path.insert(0, args.pkg_src)
    from mjolnir.core import regime_filters as rf  # noqa: PLC0415

    # The C++ gate accepts codes only (real names are refused so they cannot be
    # recovered from the .so). Encode here — the harness legitimately has the map.
    regmap = json.loads((Path(args.pkg_src).parent.parent / "obfuscation" / "map.json")
                        .read_text())["regimes"]

    def enc(expr: str) -> str:
        out = []
        for tok in re.split(r"(_and_|_or_)", expr):
            if tok in ("_and_", "_or_"):
                out.append(tok); continue
            base, suf = tok, ""
            for s2 in ("_long", "_short"):
                if base.endswith(s2):
                    base, suf = base[: -len(s2)], s2
            out.append((regmap.get(base, base)) + suf)
        return "".join(out)

    # Dedupe: with everything coded, an entry written as a real name and one
    # written as its code collapse to the same spec (they were distinct only
    # while tolerant decoding existed). Duplicates would produce duplicate
    # output columns and break the comparison.
    seen, specs, pairs = set(), [], []
    for n, pos in REGIMES:
        key = f"{enc(n)}|{pos}"
        if key in seen:
            continue
        seen.add(key)
        specs.append(key)
        pairs.append((n, pos))
    proc = subprocess.run([args.driver, str(BAR_SEC), str(TARGET_SEC), *specs],
                          capture_output=True, text=True, input="\n".join(events))
    if proc.returncode != 0:
        raise SystemExit(f"driver failed: {proc.stderr}")
    lines = proc.stdout.strip().splitlines()
    cpp = pd.DataFrame([[int(x) for x in ln.split(",")] for ln in lines[1:]],
                       columns=lines[0].split(","))

    bad, checked = [], 0
    for name, pos in pairs:
        key = f"{enc(name)}|{pos}"
        try:
            ref_mask = rf.apply_filter_mask(panel, name, pos).to_numpy(bool)
        except Exception as exc:                      # noqa: BLE001
            print(f"  SKIP {key}: reference raised {type(exc).__name__}: {exc}")
            continue
        got = cpp[key].to_numpy(bool)
        checked += 1
        n_diff = int((ref_mask != got).sum())
        if n_diff:
            bad.append((key, n_diff, int(ref_mask.sum()), int(got.sum())))

    print(f"[regime_parity] rows={len(panel)} regimes compared={checked}")
    if checked == 0:
        # A harness that compares nothing and prints PASS is worse than no
        # harness: it certifies work that was never checked.
        print("=== FAIL: compared 0 regimes — nothing was verified ===")
        return 1
    if checked < len(pairs):
        print(f"=== FAIL: only {checked}/{len(pairs)} regimes compared "
              f"(the rest were skipped, so they are UNVERIFIED) ===")
        return 1
    if bad:
        print(f"=== FAIL: {len(bad)} regimes differ ===")
        for k, d, r, g in bad:
            print(f"   {k}: {d} rows differ (ref true={r}, cpp true={g})")
        return 1
    print(f"=== PASS: all {checked} regime masks identical ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
