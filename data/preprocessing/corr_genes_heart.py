#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import networkx as nx


def qc_filter(a):
    sc.pp.calculate_qc_metrics(a, inplace=True)
    sc.pp.filter_cells(a, min_genes=int(np.percentile(a.obs['n_genes_by_counts'], 1)))
    sc.pp.filter_genes(a, min_cells=int(np.percentile(a.var['n_cells_by_counts'], 1)))
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    return a


def main(args):
    sc.settings.verbosity = 3

    adata_path = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/ventricular_cardiomyocytes.h5ad"
    adata = ad.read_h5ad(adata_path)
    adata = qc_filter(adata)

    tf_path = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/human_tfs_lambert.csv"
    tfs = pd.read_csv(tf_path, index_col=0)["Ensembl ID"].unique()

    tf_mask = adata.var_names.isin(tfs)
    sc.pp.highly_variable_genes(adata, n_top_genes=10_000, flavor='seurat')
    keep = adata.var.highly_variable.values | tf_mask
    adata = adata[:, keep].copy()

    sc.pp.subsample(adata, n_obs=adata.n_vars, random_state=42)

    X = adata.X.toarray().astype(np.float32)
    n_cells, n_genes = X.shape
    print(f"Computing Spearman on {n_cells:,}×{n_genes:,}…")

    Xr = np.apply_along_axis(rankdata, 0, X)
    Xr -= Xr.mean(axis=0)
    Xr /= Xr.std(axis=0, ddof=1)

    corr = (Xr.T @ Xr) / (n_cells - 1)

    plt.hist(corr.flatten(), bins=100)
    plt.title("Distribution of Spearman correlations")
    plt.xlabel("Correlation coefficient")
    plt.ylabel("Frequency")
    plt.show()

    flat_corr = corr.flatten()
    flat_corr = flat_corr[~np.eye(n_genes, dtype=bool).flatten()]
    flat_corr = flat_corr[~np.isnan(flat_corr)]

    percentile_filter = 0.9
    abs_corr_sorted = np.sort(np.abs(flat_corr))
    threshold = abs_corr_sorted[int(len(abs_corr_sorted) * percentile_filter)]
    print(f"Using correlation threshold: {threshold:.4f} (90th percentile of |corr| values)")

    gene_names = list(adata.var_names)

    adjacency = np.abs(corr) >= threshold

    G = nx.Graph()
    G.add_nodes_from(gene_names)
    idxs = np.transpose(np.triu(adjacency, k=1).nonzero())
    for i, j in idxs:
        G.add_edge(gene_names[i], gene_names[j], weight=float(corr[i, j]))

    print("Computing centrality metrics…")
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector centrality did not converge; skipping.")
        eigen_cent = {node: np.nan for node in G.nodes()}

    tf_records = []
    for tf in tfs:
        if tf in G:
            tf_records.append({
                'TF': tf,
                'degree': degree_cent.get(tf, 0.0),
                'betweenness': betweenness_cent.get(tf, 0.0),
                'eigenvector': eigen_cent.get(tf, np.nan),
            })
    tf_df = pd.DataFrame(tf_records)

    tf_df = tf_df.sort_values('degree', ascending=False).reset_index(drop=True)

    print("Top 20 TFs by degree centrality:")
    print(tf_df.head(20))

    out_csv = "tf_network_centrality.csv"
    tf_df.to_csv(out_csv, index=False)
    print(f"Saved TF centrality table to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF importance via network centrality from scRNA-seq correlation patterns.")
    args = parser.parse_args()
    main(args)
