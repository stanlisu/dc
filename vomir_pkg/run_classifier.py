#!/usr/bin/env python3
"""
vomir/run_classifier.py — CLI for VomirClassifier rolling cluster-then-classify.

Usage:
    python vomir/run_classifier.py --tf 1h --output vomir/output/ --window-size 6

Outputs (under {output}/{tf}/):
    classifier_predictions.csv        — timestamp, symbol, regime, y_true, y_prob, y_pred
    classifier_rolling_summary.csv    — per window: train_period, test_month, n_pos, n_neg,
                                         sharpe_signal, sharpe_baseline
    classifier_comparison.csv         — overall: signal_sharpe, baseline_sharpe, pct_filtered
"""

import argparse
import logging
import os
import sys

# Allow running from repo root: PYTHONPATH=agamotto_pkg/src:.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vomir.classifier import VomirClassifier, VALID_TFS, VALID_UNIVERSES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vomir: rolling cluster-then-classify pipeline"
    )
    p.add_argument(
        "--tf",
        required=True,
        choices=VALID_TFS,
        help="Timeframe to analyse (15m, 1h, 4h, 1d)",
    )
    p.add_argument(
        "--output",
        default="vomir/output",
        help="Output root directory (default: vomir/output/)",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Marvel repo root (default: parent of this script)",
    )
    p.add_argument(
        "--universe",
        default="liquid",
        choices=VALID_UNIVERSES,
        help="Symbol universe: liquid (_1) or altcoins (_2) (default: liquid)",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=6,
        help="Total window size in months: window_size-1 train + 1 test (default: 6)",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        default=8,
        help="Number of KMeans clusters (default: 8)",
    )
    p.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help="PCA components fed to KMeans (default: 10)",
    )
    p.add_argument(
        "--sharpe-threshold",
        type=float,
        default=0.0,
        help="Clusters with Sharpe > this are labelled 1 (default: 0.0)",
    )
    p.add_argument(
        "--dead-zone-sharpe",
        type=float,
        default=0.02,
        help="Clusters with |Sharpe| <= this are discarded from training (default: 0.02)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
    else:
        # __file__ is vomir/run_classifier.py → parent is repo root
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    logger.info(
        "data_dir=%s  tf=%s  universe=%s  window_size=%d  output=%s",
        data_dir, args.tf, args.universe, args.window_size, args.output,
    )

    clf = VomirClassifier(
        tf=args.tf,
        data_dir=data_dir,
        universe=args.universe,
        n_clusters=args.n_clusters,
        pca_components=args.pca_components,
        sharpe_threshold=args.sharpe_threshold,
        dead_zone_sharpe=args.dead_zone_sharpe,
    )

    result = clf.run_rolling(window_size=args.window_size)
    clf.save_results(result, output_dir=args.output)

    # Human-readable summary
    print(f"\n=== Vomir Classifier tf={args.tf}: Rolling Summary ===")
    print(result.summary.to_string(index=False))
    print("\n=== Overall Comparison ===")
    print(result.comparison.to_string(index=False))
    print(f"\nOutputs written to: {os.path.join(args.output, args.universe, args.tf)}/")


if __name__ == "__main__":
    main()
