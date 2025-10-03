#!/usr/bin/env python3
"""Compute abundance–specificity TF scores for cardiomyocytes.

Paths below are **hard‑coded** for convenience—edit them to match your
filesystem before running.

Run:
    $ python tf_vcm_ipr.py

Output:
    • ``tf_scores.csv`` – plain CSV (no compression)
"""

from __future__ import annotations

from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

# ───────────────────────────── user‑editable paths ────────────────────────────
ADATA_PATH = Path("/data/rbg/users/seanmurphy/transformer_grn/preprocessing/Global_raw.h5ad")
TF_CSV     = Path("/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/human_tfs_lambert.csv")
TARGET_CTS = ["Atrial Cardiomyocyte", "Ventricular Cardiomyocyte"]
TARGET_NAME = "Cardiomyocytes"
OUT_DIR    = Path("/data/rbg/users/seanmurphy/project_may/data/preprocessing/supplementary")

# ───────────────────────────── core helpers ───────────────────────────────────

def load_tf_subset(adata_path: Path, tf_csv: Path) -> ad.AnnData:
    """Return AnnData containing only TF genes found in *tf_csv*."""
    adata = ad.read_h5ad(adata_path)
    tf_ids = pd.read_csv(tf_csv)["Ensembl ID"].unique()
    return adata[:, adata.var_names.isin(tf_ids)].copy()


def mean_by_cell_type(adata_tf: ad.AnnData, key: str = "cell_type", combine_types: dict = None) -> pd.DataFrame:
    """Sparse‑aware per‑cell‑type means (genes × cell types).
    
    Args:
        adata_tf: AnnData object with TF genes
        key: Column in adata_tf.obs containing cell type information
        combine_types: Dict mapping new group names to lists of cell types to combine
    """
    X = adata_tf.X.tocsr() if not sparse.isspmatrix_csr(adata_tf.X) else adata_tf.X
    codes, cats = adata_tf.obs[key].factorize()

    # Initialize the dataframe with standard cell types
    means = np.empty((X.shape[1], len(cats)), dtype=np.float32)
    for c in range(len(cats)):
        means[:, c] = np.asarray(X[codes == c].mean(axis=0)).ravel()
    
    result_df = pd.DataFrame(means, index=adata_tf.var_names, columns=cats)
    
    # Add combined cell types if specified
    if combine_types:
        for group_name, cell_types in combine_types.items():
            valid_types = [ct for ct in cell_types if ct in result_df.columns]
            if valid_types:
                # Get cells in any of the specified types
                mask = adata_tf.obs[key].isin(valid_types)
                if mask.any():
                    result_df[group_name] = np.asarray(X[mask].mean(axis=0)).ravel()
    
    return result_df


def ipr_and_score(ct_means: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compute IPR, share in *target*, and combined abundance‑specificity score."""
    totals = ct_means.sum(axis=1)
    P      = ct_means.div(totals.replace(0, np.nan), axis=0)
    ipr    = (P**2).sum(axis=1)

    #score = np.log1p(ct_means[target]) * P[target] * ipr

    df = pd.DataFrame({"ipr": ipr})
    df["percentile"] = df["ipr"].rank(method="min", ascending=True, pct=True) * 100
    df.index.name = "gene"
    return df

def save_results(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "inverse_participation_ratio_tf_scores.csv"
    df.reset_index().to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

# ─────────────────────────────── main ────────────────────────────────────────

def main() -> None:
    print("• Loading data …")
    adata_tf = load_tf_subset(ADATA_PATH, TF_CSV)
    print(f"  → {adata_tf.shape[1]} TF genes × {adata_tf.shape[0]} cells")

    print("• Aggregating per cell type …")
    combine_types = {TARGET_NAME: TARGET_CTS}
    ct_means = mean_by_cell_type(adata_tf, combine_types=combine_types)

    print("• Computing scores …")
    results = ipr_and_score(ct_means, TARGET_NAME)

    # add HGNC symbols
    tf_table = pd.read_csv(TF_CSV, usecols=["Ensembl ID", 'HGNC symbol']).drop_duplicates()
    symbol_map = dict(zip(tf_table["Ensembl ID"], tf_table['HGNC symbol']))
    results.insert(0, "symbol", results.index.map(symbol_map))

    print("\nTop 10 TFs:")
    print(results.head(10))

    print("• Saving …")
    save_results(results, OUT_DIR)


if __name__ == "__main__":
    main()
