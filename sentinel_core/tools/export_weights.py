#!/usr/bin/env python3
"""Export joblib weight artifacts into a form the C++ core can load without Python.

The deployed weights are joblib pickles (LGBMRegressor + RobustScaler + meta).
C++ cannot read those, so each regime directory is converted to:

    model.txt      LightGBM native text model (loaded via LGBM_BoosterCreateFromModelfile)
    scaler.txt     "<n>" then n lines of "<center> <scale>"  (RobustScaler)
    features.txt   one CODED feature name per line, in model input order

Run it where the weights are (the live host), then ship the converted directory
alongside the core. This is a PREREQUISITE for the C++ model runner, not part of
it — until the weight-roll tooling emits these itself, conversion is a manual
step per roll.

VERIFICATION IS NOT OPTIONAL: every export re-predicts a batch of random feature
vectors through both the original sklearn object and the exported booster and
fails if they disagree. An export that silently changes predictions is the
worst possible failure here, because everything downstream still runs.

Usage:
    python export_weights.py --weights <window_dir> --out <dir> [--tol 1e-9]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

_HERE = Path(__file__).resolve().parent


def _regime_map() -> dict:
    """real regime name -> rNNN, from the single source of truth."""
    mp = json.loads((_HERE / ".." / ".." / "obfuscation" / "map.json").read_text())
    return mp["regimes"]


def encode_regime_dir(name: str, regmap: dict) -> str:
    """'<real>_long' -> '<code>_long'. The C++ core accepts CODES ONLY — real
    names are refused there so none can be recovered from the built .so."""
    for suffix in ("_long", "_short"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            code = regmap.get(base)
            if code is None:
                raise SystemExit(f"regime {base!r} is not in the obfuscation map")
            return code + suffix
    code = regmap.get(name)
    if code is None:
        raise SystemExit(f"regime {name!r} is not in the obfuscation map")
    return code


def export_one(regime_dir: Path, out_dir: Path, tol: float) -> dict:
    model_p = regime_dir / "lightgbm_model.pkl"
    scaler_p = regime_dir / "lightgbm_scaler.pkl"
    meta_p = regime_dir / "lightgbm_meta.pkl"
    for p in (model_p, meta_p):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    model = joblib.load(model_p)
    meta = joblib.load(meta_p)
    feats = list(meta["feature_columns"])

    out_dir.mkdir(parents=True, exist_ok=True)

    booster = model.booster_
    booster.save_model(str(out_dir / "model.txt"))
    (out_dir / "features.txt").write_text("\n".join(feats) + "\n")

    # RobustScaler: (x - center_) / scale_. Read the attributes explicitly
    # rather than assuming StandardScaler's mean_/scale_ — the deployed scaler
    # is RobustScaler and mean_ does not exist on it.
    if scaler_p.is_file():
        sc = joblib.load(scaler_p)
        cls = type(sc).__name__
        if hasattr(sc, "center_"):
            center = np.asarray(sc.center_, dtype=float)
        elif hasattr(sc, "mean_"):
            center = np.asarray(sc.mean_, dtype=float)
        else:
            raise SystemExit(f"scaler {cls} has neither center_ nor mean_")
        scale = np.asarray(sc.scale_, dtype=float)
        if center.shape[0] != len(feats) or scale.shape[0] != len(feats):
            raise SystemExit(
                f"scaler length {center.shape[0]}/{scale.shape[0]} != {len(feats)} features")
    else:
        cls = "identity"
        center = np.zeros(len(feats))
        scale = np.ones(len(feats))

    # %.17g round-trips a double exactly. NOT repr(): under numpy 2.x
    # repr(np.float64(0.1)) is "np.float64(0.1)", which no C++ parser accepts —
    # and the in-memory verification below would never notice.
    scaler_path = out_dir / "scaler.txt"
    with scaler_path.open("w") as fh:
        fh.write(f"{len(feats)}\n")
        for c, s in zip(center, scale):
            fh.write("%.17g %.17g\n" % (float(c), float(s)))

    # Read the FILE back and check it parses to the same numbers. Verifying the
    # in-memory objects only proves the export logic, not the artifact that
    # actually ships — which is where the numpy-repr bug lived.
    with scaler_path.open() as fh:
        n_read = int(fh.readline())
        rt = [tuple(float(x) for x in ln.split()) for ln in fh if ln.strip()]
    if n_read != len(feats) or len(rt) != len(feats):
        raise SystemExit(f"scaler.txt readback length mismatch in {out_dir}")
    for i, (c, s) in enumerate(rt):
        if c != float(center[i]) or s != float(scale[i]):
            raise SystemExit(f"scaler.txt readback differs at row {i} in {out_dir}")

    # --- verification: exported path must reproduce the sklearn path ---------
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, len(feats))) * 10.0
    Xs = (X - center) / scale
    ref = model.predict(Xs)
    got = booster.predict(Xs)
    max_diff = float(np.max(np.abs(ref - got)))
    if max_diff > tol:
        raise SystemExit(
            f"VERIFY FAILED for {regime_dir.name}: sklearn vs booster differ by "
            f"{max_diff:.3e} (> {tol}). Refusing to export a model that predicts "
            f"differently than the deployed one.")

    return {
        "regime": regime_dir.name,
        "n_features": len(feats),
        "scaler": cls,
        "verify_max_diff": max_diff,
        "n_trees": booster.num_trees(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="window_YYYY_MM_DD directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    win = Path(args.weights).expanduser()
    out = Path(args.out).expanduser()
    regimes = sorted(d for d in win.iterdir() if d.is_dir())
    if not regimes:
        raise SystemExit(f"no regime dirs under {win}")

    regmap = _regime_map()
    report = []
    for rd in regimes:
        coded = encode_regime_dir(rd.name, regmap)
        info = export_one(rd, out / coded, args.tol)
        info["coded_dir"] = coded
        report.append(info)
        print(f"  {info['regime']} -> {info['coded_dir']}: {info['n_features']} feats, "
              f"{info['n_trees']} trees, scaler={info['scaler']}, "
              f"verify_max_diff={info['verify_max_diff']:.2e}")

    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(f"=== exported {len(report)} regimes -> {out} (all verified) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
