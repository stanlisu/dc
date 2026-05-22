#!/usr/bin/env python3
"""
vomir/run_research.py — CLI entry point for VomirResearch.

Usage:
    python vomir/run_research.py --tf 1h --output vomir/output/

Outputs (under {output}/{tf}/):
    pca_variance.csv       — explained variance per component
    pca_loadings.csv       — feature loadings (features × components)
    clusters.csv           — per-row cluster assignment + y + regime + symbol
    cluster_summary.csv    — per-cluster Sharpe, count, top features
    umap_projection.csv    — 2D coords + y + cluster label (subsampled if large)
"""

import argparse
import logging
import os
import sys

import pandas as pd

# Allow running from repo root: PYTHONPATH=agamotto_pkg/src:.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vomir.research import VomirResearch, VALID_TFS, VALID_UNIVERSES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vomir: unsupervised structure discovery in agamotto features"
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
        "--n-components",
        type=int,
        default=20,
        help="Number of PCA components (default: 20)",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        default=8,
        help="Number of KMeans clusters (default: 8)",
    )
    p.add_argument(
        "--pca-for-cluster",
        type=int,
        default=10,
        help="PCA components used as input to KMeans / UMAP (default: 10)",
    )
    p.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15; ignored for t-SNE)",
    )
    return p.parse_args()


def save_outputs(
    tf: str,
    output_root: str,
    results: dict,
) -> None:
    """Write all result DataFrames to CSV files."""
    out_dir = os.path.join(output_root, tf)
    os.makedirs(out_dir, exist_ok=True)

    fm = results["fm"]
    pca = results["pca"]
    cluster = results["cluster"]
    umap_result = results["umap"]
    summary = results["cluster_summary"]

    # 1. PCA variance
    variance_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio))],
        "explained_variance_ratio": pca.explained_variance_ratio,
        "cumulative_variance": pca.cumulative_variance,
    })
    variance_path = os.path.join(out_dir, "pca_variance.csv")
    variance_df.to_csv(variance_path, index=False)
    logger.info("Saved %s", variance_path)

    # 2. PCA loadings
    loadings_path = os.path.join(out_dir, "pca_loadings.csv")
    pca.loadings.to_csv(loadings_path)
    logger.info("Saved %s", loadings_path)

    # 3. Clusters (per-row)
    clusters_df = fm.meta.copy()
    clusters_df["cluster"] = cluster.labels
    clusters_df["y"] = fm.y
    clusters_path = os.path.join(out_dir, "clusters.csv")
    clusters_df.to_csv(clusters_path, index=False)
    logger.info("Saved %s (%d rows)", clusters_path, len(clusters_df))

    # 4. Cluster summary
    summary_path = os.path.join(out_dir, "cluster_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Saved %s", summary_path)

    # 5. UMAP/t-SNE projection
    coords = umap_result.coords
    idx = getattr(results.get("_research"), "_umap_idx", None)

    # Reconstruct corresponding y + cluster labels
    if idx is not None:
        y_proj = fm.y[idx]
        labels_proj = cluster.labels[idx]
        meta_proj = fm.meta.iloc[idx].reset_index(drop=True)
    else:
        y_proj = fm.y
        labels_proj = cluster.labels
        meta_proj = fm.meta.reset_index(drop=True)

    umap_df = meta_proj.copy()
    umap_df["dim1"] = coords[:, 0]
    umap_df["dim2"] = coords[:, 1]
    umap_df["y"] = y_proj
    umap_df["cluster"] = labels_proj
    umap_df["method"] = umap_result.method
    umap_path = os.path.join(out_dir, "umap_projection.csv")
    umap_df.to_csv(umap_path, index=False)
    logger.info("Saved %s (%d rows, method=%s)", umap_path, len(umap_df), umap_result.method)


def main() -> None:
    args = parse_args()

    # Resolve data_dir to marvel repo root
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
    else:
        # __file__ is vomir/run_research.py → parent is repo root
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    logger.info("data_dir=%s  tf=%s  universe=%s  output=%s", data_dir, args.tf, args.universe, args.output)

    research = VomirResearch(tf=args.tf, data_dir=data_dir, universe=args.universe)
    results = research.run_all(
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        pca_for_cluster=args.pca_for_cluster,
        n_neighbors=args.n_neighbors,
    )

    # Attach research object so save_outputs can access _umap_idx
    results["_research"] = research
    output_root = os.path.join(args.output, args.universe)
    save_outputs(tf=args.tf, output_root=output_root, results=results)

    # Print human-readable summary
    summary = results["cluster_summary"]
    print(f"\n=== Vomir tf={args.tf} universe={args.universe}: Cluster Summary ===")
    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {os.path.join(output_root, args.tf)}/")


if __name__ == "__main__":
    main()
