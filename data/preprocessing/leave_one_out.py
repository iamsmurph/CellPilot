#!/usr/bin/env python3
"""
run_elastic_grn.py

Preprocess an AnnData object and build a gene regulatory network by
fitting an Elastic Net model for each gene in parallel.
"""

import argparse
import gc

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp


from sklearn.linear_model import ElasticNetCV, ElasticNet
from joblib import Parallel, delayed

def qc_filter(a: ad.AnnData, subsample_n: int) -> ad.AnnData:
    """Calculate QC metrics, filter, normalize, log-transform, scale, and subsample."""
    sc.pp.calculate_qc_metrics(a, qc_vars=[], inplace=True)
    sc.pp.filter_cells(a,  min_genes= int(np.percentile(a.obs['n_genes_by_counts'], 1)))
    sc.pp.filter_genes(a,  min_cells= int(np.percentile(a.var['n_cells_by_counts'], 1)))
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.pp.scale(a)
    sc.pp.subsample(a, n_obs=subsample_n, random_state=42)
    return a

def fit_elastic_net(i: int, X: np.ndarray, genes: np.ndarray) -> tuple:
    """
    Fit Elastic Net for gene i (response) against all other genes (predictors).
    Returns (target_gene, coef_vector).
    """
    y = X[:, i]
    X_pred = np.delete(X, i, axis=1)
    
    l1_ratio = 0.5
    a_max  = np.abs(X_pred.T @ y).max() / (len(y) * l1_ratio)
    alpha  = 0.05 * a_max

    model = ElasticNet(
        max_iter=5_000, tol=1e-4,
        alpha=alpha,
        l1_ratio=l1_ratio
    )
    model.fit(X_pred, y)
    coefs = model.coef_
    full_coefs = np.insert(coefs, i, 0.0)

    del X_pred, model
    gc.collect()

    return genes[i], full_coefs

def main(args):
    sc.settings.verbosity = 3
    sc.settings.seed = 42

    adata = ad.read_h5ad(args.input)
    adata = qc_filter(adata, subsample_n=adata.shape[1])

    pth = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/human_tfs_lambert.csv"
    tfs = pd.read_csv(pth, index_col=0)
    unique_tfs = tfs["Ensembl ID"].unique()

    cardiac_tfs_mask = adata.var_names.isin(unique_tfs)
    print(f"Found {np.sum(cardiac_tfs_mask)} TFs in the dataset")

    print("Identifying highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=10000, flavor='seurat')
    
    hvg_mask = adata.var.highly_variable
    union_mask = np.logical_or(hvg_mask, cardiac_tfs_mask)
    
    adata = adata[:, union_mask].copy()

    if sp.issparse(adata.X):
        X = adata.X.A
    else:
        X = np.array(adata.X)

    genes = adata.var_names.values
    n_cells, n_genes = X.shape
    print(f"Data dims after subsampling: {n_cells} cells × {n_genes} genes")

    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fit_elastic_net)(i, X, genes) for i in range(n_genes)
    )

    edges = (
        pd.DataFrame(
            np.vstack([coef for _, coef in results]),
            index=[gene for gene, _ in results],
            columns=genes
        )
        .stack()
        .rename_axis(['target','regulator'])
        .reset_index(name='coef')
        .query('coef != 0')
    )

    edges.to_csv(args.output, index=False)
    print(f"Wrote {len(edges)} edges to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess scRNA-seq and infer GRN via Elastic Net"
    )
    parser.add_argument(
        "-i", "--input",
        default="/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/ventricular_cardiomyocytes.h5ad",
        help="Path to input .h5ad file"
    )
    parser.add_argument(
        "-o", "--output",
        default="/data/rbg/users/seanmurphy/project_may/data/preprocessing/gene_regulatory_edges.csv",
        help="Path for output CSV (target, regulator, coef)"
    )
    parser.add_argument(
        "-j", "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel jobs for model fitting"
    )
    args = parser.parse_args()
    main(args)
