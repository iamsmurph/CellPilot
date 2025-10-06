#%%
import argparse
import os
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional
"""Compute TF×TF Jaccard on top‑K MACS targets directly from an h5ad matrix.

K is chosen per TF with a variable size within [k_min, k_max].
TFs with fewer than k_min nonzero targets are dropped.
"""


def build_topk_binary_matrix(
    X, gene_names: np.ndarray, k_min: int, k_max: int
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Return (B_csc, sizes, kept_tf_indices) with variable K per TF in [k_min, k_max].

    - B is binary genes×TFs with 1s for top‑K genes per kept TF.
    - Only TFs (columns) having at least k_min nonzero entries are kept.
    - Per TF, K = min(k_max, number of nonzero targets for that TF).
    - Tie‑break is deterministic by gene name (ascending) after sorting by descending value.
    """
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    X_csc = X.tocsc()  # column‑wise access

    n_genes, n_tfs = X_csc.shape
    indptr = X_csc.indptr
    indices = X_csc.indices
    data = X_csc.data

    kept_rows_per_tf: list[np.ndarray] = []
    kept_sizes: list[int] = []
    kept_tf_indices: list[int] = []

    for j in range(n_tfs):
        s, e = indptr[j], indptr[j + 1]
        rows = indices[s:e]
        vals = data[s:e]
        m = vals.size
        if m < k_min:
            # Drop TFs with fewer than k_min nonzero targets
            continue
        k_sel = k_max if m >= k_max else m
        names = gene_names[rows]
        order = np.lexsort((names, -vals))
        if order.size > k_sel:
            order = order[:k_sel]
        sel = rows[order]
        kept_rows_per_tf.append(sel)
        kept_sizes.append(int(sel.size))
        kept_tf_indices.append(j)

    if len(kept_tf_indices) == 0:
        # Return empty matrices/arrays with correct shapes
        B = sparse.csc_matrix((n_genes, 0), dtype=np.uint8)
        return B, np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int64)

    sizes = np.array(kept_sizes, dtype=np.int32)
    nnz = int(sizes.sum())

    B_indptr = np.empty(len(kept_tf_indices) + 1, dtype=np.int64)
    B_indptr[0] = 0
    np.cumsum(sizes, out=B_indptr[1:])
    B_indices = np.empty(nnz, dtype=np.int32)
    pos = 0
    for rows in kept_rows_per_tf:
        m = rows.size
        if m:
            B_indices[pos : pos + m] = np.sort(rows)
        pos += m
    B_data = np.ones(nnz, dtype=np.uint8)
    B = sparse.csc_matrix((B_data, B_indices, B_indptr), shape=(n_genes, len(kept_tf_indices)))
    return B, sizes, np.array(kept_tf_indices, dtype=np.int64)


def jaccard_from_binary(B: sparse.csc_matrix, sizes: np.ndarray) -> np.ndarray:
    """Compute dense TF×TF Jaccard from binary genes×TFs matrix B and column sizes."""
    G = (B.T @ B).toarray().astype(np.float64)  # intersections
    s = sizes.astype(np.float64)
    den = s[:, None] + s[None, :] - G
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc = np.where(den > 0, G / den, 0.0)
    nonempty = s > 0
    jacc[nonempty, nonempty] = 1.0
    return jacc


def main():
    parser = argparse.ArgumentParser(description="Compute TF×TF Jaccard on top‑K MACS targets from an h5ad.")
    parser.add_argument(
        "--h5ad",
        default="/data/rbg/users/ujp/ib/gene_info/chipatlas/chip/target/chip_data_hg38.h5ad",
        help="Path to input h5ad (genes×TFs, TFs in var)",
    )
    parser.add_argument("--k-min", type=int, default=100, help="Minimum K per TF (drop TFs with fewer nonzero targets)")
    parser.add_argument("--k-max", type=int, default=400, help="Maximum K per TF; actual K for a TF is min(k_max, its nonzero targets)")
    parser.add_argument("--k", type=int, default=None, help="DEPRECATED: fixed K. If set, uses K for both k-min and k-max")
    parser.add_argument(
        "--out-jaccard",
        default=None,
        help="Output CSV path for TF×TF Jaccard matrix in long 3‑column format. If omitted, will be auto-named based on input and K settings.",
    )
    args = parser.parse_args()

    # Resolve K settings (support legacy --k)
    if args.k is not None:
        k_min = int(args.k)
        k_max = int(args.k)
    else:
        k_min = int(args.k_min)
        k_max = int(args.k_max)
    if k_min < 1:
        raise SystemExit("k_min must be >= 1")
    if k_max < k_min:
        raise SystemExit("k_max must be >= k_min")

    adata = ad.read_h5ad(args.h5ad)
    gene_names = np.array(adata.obs_names.astype(str))
    tf_names = np.array(adata.var_names.astype(str))
    total_tf_count = int(len(tf_names))

    # Lambert filter: keep only TFs marked as TFs in the Lambert catalog (mandatory)
    LAMBERT_CSV = "/data/rbg/users/seanmurphy/CellPilot/data/datasets/supplementary/human_tfs_lambert.csv"
    lambert = pd.read_csv(LAMBERT_CSV)
    # Robustly find columns
    def find_col(name_snippet: str, fallback: Optional[str] = None) -> str:
        lower = {c.lower(): c for c in lambert.columns}
        for k, v in lower.items():
            if name_snippet.lower() in k:
                return v
        if fallback is not None and fallback in lambert.columns:
            return fallback
        raise KeyError(f"Expected column containing '{name_snippet}' not found in Lambert CSV")

    hgnc_col = None
    for candidate in ["hgnc symbol", "symbol", "gene symbol"]:
        try:
            hgnc_col = find_col(candidate)
            break
        except KeyError:
            continue
    if hgnc_col is None:
        raise KeyError("Could not find HGNC symbol column in Lambert CSV")

    try:
        istf_col = find_col("is tf?")
    except KeyError:
        # Fallback to other assessment fields if needed
        for candidate in ["tfclass considers it a tf?", "cisbp considers it a tf?", "tf assessment"]:
            try:
                istf_col = find_col(candidate)
                break
            except KeyError:
                istf_col = None
        if istf_col is None:
            raise KeyError("Could not find an 'Is TF?' indicator column in Lambert CSV")

    def truthy(x) -> bool:
        s = str(x).strip().lower()
        return s in {"yes", "true", "1", "y", "t"} or s.startswith("yes")

    lambert_symbols_upper = (
        lambert.loc[lambert[istf_col].map(truthy), hgnc_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )
    lambert_set = set(lambert_symbols_upper)

    tf_names_upper = np.char.upper(tf_names.astype(str))
    lambert_mask = np.isin(tf_names_upper, np.array(list(lambert_set)))
    if not lambert_mask.any():
        raise SystemExit("No TFs overlap with Lambert TF list. Aborting.")

    adata = adata[:, lambert_mask]
    tf_names = np.array(adata.var_names.astype(str))
    lambert_tf_count = int(lambert_mask.sum())

    B, sizes, kept_idx = build_topk_binary_matrix(adata.X, gene_names=gene_names, k_min=k_min, k_max=k_max)
    if B.shape[1] == 0:
        raise SystemExit(f"No TFs have at least k_min={k_min} nonzero targets. Nothing to compute.")
    kept_tf_names = tf_names[kept_idx]
    jacc = jaccard_from_binary(B, sizes)
    jaccard_df = pd.DataFrame(jacc, index=kept_tf_names, columns=kept_tf_names)
    # Convert to long 3-column unique unordered pairs (i<=j) and drop self-pairs
    long_df = (jaccard_df.where(np.triu(np.ones_like(jaccard_df, dtype=bool), k=1))
                            .stack()
                            .rename("jaccard")
                            .reset_index()
                            .rename(columns={"level_0":"tf_a","level_1":"tf_b"}))
    long_df = long_df.sort_values("jaccard", ascending=False).reset_index(drop=True)

    # Resolve output path
    if args.out_jaccard is not None and str(args.out_jaccard).strip():
        out_path = args.out_jaccard
    else:
        in_base = os.path.splitext(os.path.basename(args.h5ad))[0]
        if args.k is not None or k_min == k_max:
            k_tag = f"top{args.k if args.k is not None else k_min}"
        else:
            k_tag = f"top{k_min}to{k_max}"
        out_dir = "/data/rbg/users/seanmurphy/CellPilot/data/datasets/supplementary"
        out_filename = f"chip_tf_jaccard_{k_tag}_{in_base}.csv"
        out_path = os.path.join(out_dir, out_filename)

    long_df.to_csv(out_path, index=False)

    print(
        f"Lambert filter: {lambert_tf_count} TFs of {total_tf_count} total.\n"
        f"Kept {len(kept_tf_names)} TFs with ≥{k_min} targets (K per TF ≤ {k_max}).\n"
        f"Wrote {len(long_df)} TF pairs to {out_path}"
    )


if __name__ == "__main__":
    main()

