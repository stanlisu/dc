#!/usr/bin/env python3
"""
vomir/run_regime_rolling.py — Per-regime-group cluster-then-classify rolling CLI.

Usage:
    PYTHONPATH=agamotto_pkg/src:. python vomir/run_regime_rolling.py --tf 1h

Outputs:
    vomir/output_regime/{universe}/{tf}/classifier_rolling_summary.csv
    vomir/output_regime/{universe}/{tf}/classifier_predictions.csv
    vomir/weights_regime/window_{YYYY_MM}/  ← one artifacts dir per rolling window
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vomir.classifier import VALID_TFS, VALID_UNIVERSES, ClassifierWindow  # noqa: E402
from vomir.regime_classifier import VomirRegimeClassifier  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

_N_GROUPS = 4  # baseline, high_volume, low_volume, vol_breakout


def _parse_int_list(s: str, n: int, name: str):
    parts = s.split(",")
    if len(parts) != n:
        raise argparse.ArgumentTypeError(
            f"--{name} must be {n} comma-separated integers, got: {s!r}"
        )
    try:
        return [int(p.strip()) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--{name} must be integers, got: {s!r}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vomir Regime: per-regime-group cluster-then-classify pipeline"
    )
    p.add_argument("--tf", required=True, choices=VALID_TFS,
                   help="Timeframe (15m, 1h, 4h, 1d)")
    p.add_argument("--universe", default="liquid", choices=VALID_UNIVERSES,
                   help="Symbol universe: liquid or altcoins (default: liquid)")
    p.add_argument("--output", default="vomir/output_regime",
                   help="Output root directory (default: vomir/output_regime/)")
    p.add_argument("--weights-dir", default=None,
                   help="Override default vomir/weights_regime/ for saved artifacts")
    p.add_argument("--data-dir", default=None,
                   help="Marvel repo root (default: parent of this script)")
    p.add_argument("--window-size", type=int, default=12,
                   help="Window size in months: window_size-1 train + 1 test (default: 12)")
    p.add_argument("--k-per-group", default="6,6,4,6",
                   help="Comma-sep KMeans k per group [baseline,high_volume,low_volume,vol_breakout] "
                        "(default: 6,6,4,6)")
    p.add_argument("--pca-per-group", default="10,10,8,10",
                   help="Comma-sep PCA components per group (default: 10,10,8,10)")
    p.add_argument("--label-threshold", type=float, default=0.0,
                   help="Min |return_raw| to be labelled LONG/SHORT (default: 0.0)")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving model artifacts (dry-run evaluation only)")
    p.add_argument("--dump-month", default=None, metavar="YYYY-MM",
                   help="Generate production weights for this deploy month (e.g. 2026-03). "
                        "Trains on the most recent window_size-1 months of data with no holdout.")
    p.add_argument("--no-test", action="store_true",
                   help="Production dump mode: train on all available months, no holdout evaluation. "
                        "Requires --dump-month.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Parse per-group lists
    try:
        k_per_group = _parse_int_list(args.k_per_group, _N_GROUPS, "k-per-group")
    except argparse.ArgumentTypeError as e:
        logger.error("%s", e)
        sys.exit(1)
    try:
        pca_per_group = _parse_int_list(args.pca_per_group, _N_GROUPS, "pca-per-group")
    except argparse.ArgumentTypeError as e:
        logger.error("%s", e)
        sys.exit(1)

    data_dir = os.path.abspath(
        args.data_dir or os.path.join(os.path.dirname(__file__), "..")
    )
    weights_dir = (
        os.path.abspath(args.weights_dir)
        if args.weights_dir
        else os.path.join(data_dir, "vomir", "weights_regime")
    )

    if args.no_test and not args.dump_month:
        logger.error("--no-test requires --dump-month")
        sys.exit(1)

    dump_month = None
    if args.dump_month:
        try:
            parts = args.dump_month.split("-")
            dump_month = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            logger.error("--dump-month must be YYYY-MM, got: %s", args.dump_month)
            sys.exit(1)

    logger.info(
        "data_dir=%s  tf=%s  universe=%s  window_size=%d  "
        "k_per_group=%s  pca_per_group=%s  label_threshold=%.4f%s",
        data_dir, args.tf, args.universe, args.window_size,
        k_per_group, pca_per_group, args.label_threshold,
        f"  dump_month={args.dump_month} no_test={args.no_test}" if dump_month else "",
    )

    clf = VomirRegimeClassifier(
        tf=args.tf,
        data_dir=data_dir,
        universe=args.universe,
        label_threshold=args.label_threshold,
        weights_dir=weights_dir,
        k_per_group=k_per_group,
        pca_per_group=pca_per_group,
    )

    # Production dump mode
    if dump_month and args.no_test:
        df = clf._load_data()
        months = sorted(df.groupby(["year", "month"]).groups.keys())
        if len(months) < args.window_size - 1:
            logger.error(
                "Not enough months (%d) for window_size-1=%d",
                len(months), args.window_size - 1,
            )
            sys.exit(1)
        train_months = months[-(args.window_size - 1):]
        prod_window = ClassifierWindow(
            window_id=1,
            train_months=train_months,
            test_month=dump_month,
            production=True,
        )
        result_w = clf.fit_predict_window(prod_window, save=not args.no_save, no_test=True)
        print(f"\n=== Vomir Regime Production Dump — tf={args.tf}  deploy={args.dump_month} ===")
        print(f"Train period : {prod_window.train_label}")
        print(f"n_train      : {result_w.n_train}")
        print(f"n_long_label : {result_w.n_long_label}")
        print(f"n_short_label: {result_w.n_short_label}")
        print(f"Weights saved: {result_w.weights_path}")
        return

    result = clf.run_rolling(window_size=args.window_size, save=not args.no_save)
    clf.save_results(result, output_dir=os.path.join(data_dir, args.output))

    cols = [
        "window_id", "test_month", "n_long", "n_short",
        "sharpe_long", "sharpe_short", "sharpe_combined",
        "sharpe_baseline_long", "sharpe_baseline_short",
    ]
    print(f"\n=== Vomir Regime Rolling Results — tf={args.tf}  universe={args.universe} ===")
    print(result.summary[cols].to_string(index=False))

    all_sharpes_comb = result.summary["sharpe_combined"]
    print(f"\nMean combined Sharpe across {len(result.summary)} windows: "
          f"{all_sharpes_comb.mean():.4f}  (std {all_sharpes_comb.std():.4f})")

    out_base = os.path.join(data_dir, args.output, args.universe, args.tf)
    print(f"\nOutputs  → {out_base}/")
    if not args.no_save:
        print(f"Artifacts → {weights_dir}/window_*/")


if __name__ == "__main__":
    main()
