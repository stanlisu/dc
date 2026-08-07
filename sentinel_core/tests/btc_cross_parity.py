#!/usr/bin/env python3
"""Differential parity for the anchor (BTC) cross-feature bus.

Builds TWO symbols from two independent event streams, computes the anchor's
panel, and merges its cross-features into the peer — in both C++ and the
reference — then compares every shared column.

Why the two streams are deliberately DIFFERENT: the anchor and peer close bars
at different times, so their bar sets do not coincide. That is what forces the
join to be BY bar_ts. A positional join passes on identical streams and corrupts
silently on real data, which is exactly the bug this test exists to catch.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-bar", required=True)
    ap.add_argument("--ref-feat", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--anchor-seed", type=int, default=7)
    ap.add_argument("--peer-seed", type=int, default=21)
    args = ap.parse_args()

    ta_ver = require_reference_talib()
    print(f"[btc_cross_parity] reference TA-Lib {ta_ver}")

    a_ev = gen_events(seed=args.anchor_seed)
    p_ev = gen_events(seed=args.peer_seed)
    a_path = Path("/tmp/_anchor_ev.csv"); a_path.write_text("\n".join(a_ev))
    p_path = Path("/tmp/_peer_ev.csv");   p_path.write_text("\n".join(p_ev))

    bar_mod = load_reference(Path(args.ref_bar))
    feat_mod = load_reference(Path(args.ref_feat))
    fe = feat_mod.MjolnirFeatures(feature_windows=[30, 60, 300, 900],
                                  bar_tf="5s", target_tf="5s")

    a_panel = fe.compute(ref_bars_df(bar_mod, a_ev))
    p_panel = fe.compute(ref_bars_df(bar_mod, p_ev))
    assert_reference_used_talib(a_panel, "anchor panel")
    assert_reference_used_talib(p_panel, "peer panel")
    # The reference reindexes the anchor onto the peer's index internally.
    ref = fe.add_btc_cross_features(p_panel, a_panel)

    proc = subprocess.run([args.driver, str(a_path), str(p_path)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"driver failed: {proc.stderr[-2000:]}")
    lines = proc.stdout.strip().splitlines()
    cpp = pd.DataFrame([[float(x) for x in ln.split(",")] for ln in lines[1:]],
                       columns=lines[0].split(","))

    mp = json.loads((Path(__file__).resolve().parents[2] / "obfuscation" / "map.json")
                    .read_text())["features"]

    def enc(col: str) -> str:
        m2 = re.match(r"^(.*)_roll(\d+)_(mean|std)$", col)
        if m2 and m2.group(1) in mp:
            return f"{mp[m2.group(1)]}_roll{m2.group(2)}_{m2.group(3)}"
        return mp.get(col, col)

    ref = ref.rename(columns={c: enc(c) for c in ref.columns})

    print(f"[btc_cross] rows ref={len(ref)} cpp={len(cpp)}")
    if len(ref) != len(cpp):
        print("=== FAIL: row count differs ===")
        return 1

    # The cross columns are the point of this test — assert they are actually
    # present, or a green here would mean nothing.
    cross = [enc(c) for c in ("btc_mid_return_lag1", "btc_mid_return_lag4",
                              "btc_book_imbalance_L1", "btc_trade_imbalance",
                              "btc_ofi_L1", "btc_spread_ratio", "btc_liq_directional")]
    missing_cross = [c for c in cross if c not in cpp.columns]
    if missing_cross:
        print(f"=== FAIL: C++ produced no cross-features: {missing_cross} ===")
        return 1

    shared = sorted(set(ref.columns) & set(cpp.columns))
    cross_shared = [c for c in shared if c in cross or
                    any(c.startswith(x + "_roll") for x in cross)]
    print(f"[btc_cross] compared {len(shared)} columns "
          f"({len(cross_shared)} of them cross-features)")
    if len(cross_shared) < 25:
        print(f"=== FAIL: only {len(cross_shared)} cross columns compared "
              f"(7 base + 24 rollings expected) ===")
        return 1

    bad = []
    for c in shared:
        r = pd.to_numeric(ref[c], errors="coerce").to_numpy(float)
        v = cpp[c].to_numpy(float)
        r = np.where(np.isfinite(r), r, 0.0)
        v = np.where(np.isfinite(v), v, 0.0)
        d = np.abs(r - v) / np.maximum(1.0, np.abs(r))
        k = int((d > args.tol).sum())
        if k:
            bad.append((c, k, float(d.max())))

    if bad:
        print(f"=== FAIL: {len(bad)} columns differ ===")
        for c, k, mx in bad[:15]:
            print(f"   {c}: {k} cells, max rel diff {mx:.3e}")
        return 1
    print(f"=== PASS: all {len(shared)} columns identical, "
          f"incl. {len(cross_shared)} cross-features (tol={args.tol}) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
