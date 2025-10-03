#%%
#!/usr/bin/env python3
import scanpy as sc
import anndata as ad
import pandas as pd
import gc
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
# %%
cardiac = ad.read_h5ad(
        "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/ventricular_cardiomyocytes.h5ad"
    )


# %%

def qc_filter(a: ad.AnnData):
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.subsample(a, n_obs=20000, random_state=42)
    return a


def main():
    # Load cardiac data
    cardiac = ad.read_h5ad(
        "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/ventricular_cardiomyocytes.h5ad"
    )
    
    # Load iPSC data
    ipsc = ad.read_h5ad(
        "/data/rbg/shared/projects/grn/nourreddine/KOLF_Pan_Genome_Aggregate_rna_non_targeting.h5ad"
    )
    
    # Alternative iPSC data path
    #ipsc = ad.read_h5ad(
    #    "/data/rbg/users/ujp/ib/cardiac/friedman/friedman.h5ad"
    #)
    #day0_mask = ipsc.obs['dayrep'].str.startswith('Day0')
    #ipsc = ipsc[day0_mask]

    # Load transcription factors
    pth = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/human_tfs_lambert.csv"
    tfs = pd.read_csv(pth)
    
    # Create TF mapping correctly from the dataframe
    tf_ensembl_to_hgnc = dict(zip(tfs["Ensembl ID"], tfs["HGNC symbol"]))
    unique_tfs = set(tfs["Ensembl ID"].dropna().unique())
    
    # Create a comprehensive mapping from cardiac data
    # This assumes cardiac has gene names in a column
    if "gene_name-new" in cardiac.var.columns:
        cardiac_ensembl_to_hgnc = dict(zip(cardiac.var.index, cardiac.var["gene_name-new"]))
    else:
        # If no gene name column exists, we'll only have TF mappings
        cardiac_ensembl_to_hgnc = {}
    
    # Combine mappings (cardiac mapping takes precedence if there are conflicts)
    ensembl_to_hgnc_map = {**tf_ensembl_to_hgnc, **cardiac_ensembl_to_hgnc}

    # Find shared genes and TFs
    shared_genes = set(cardiac.var_names).intersection(set(ipsc.var.gene_ids))
    print(f"Found {len(shared_genes)} shared genes between datasets")

    shared_tfs = shared_genes.intersection(unique_tfs)
    print(f"Found {len(shared_tfs)} shared transcription factors between datasets")
    
    # Use only TFs for the analysis
    shared_genes = shared_tfs

    # Filter datasets
    cardiac = cardiac[:, cardiac.var_names.isin(shared_genes)].copy()
    ipsc = ipsc[:, ipsc.var.gene_ids.isin(shared_genes)].copy()
    
    # IMPORTANT: Make sure iPSC uses gene_ids as var_names for concatenation
    ipsc.var_names = ipsc.var.gene_ids
    ipsc.var_names_make_unique()

    # QC filtering
    cardiac = qc_filter(cardiac)
    ipsc = qc_filter(ipsc)

    # Concatenate datasets
    adata = ad.concat([cardiac, ipsc],
                      label="batch",
                      keys=["cardiomyocyte", "ipsc"],
                      merge="same",
                      index_unique=None)

    del cardiac, ipsc
    gc.collect()
    print(f"Shape of concatenated adata: {adata.shape}")
    print(f"Batch distribution in adata: \n{adata.obs['batch'].value_counts()}")

    # Differential expression analysis
    sc.tl.rank_genes_groups(
            adata,
            groupby="batch",
            groups=["cardiomyocyte"],
            reference="ipsc",
            method="wilcoxon",
            pts=True,
            use_raw=False,
            layer=None
        )

    de_df = sc.get.rank_genes_groups_df(
        adata,
        group="cardiomyocyte",
        pval_cutoff=None,
    )
    de_df = de_df[de_df["pvals_adj"] < 0.05]

    de_df = de_df[de_df['logfoldchanges'] > 0]
    de_df["percentile"] = de_df["logfoldchanges"].rank(pct=True)*100
    
    # Add both Ensembl ID and HGNC symbol columns
    de_df["ensembl_id"] = de_df["names"]  # names column contains Ensembl IDs
    de_df["HGNC symbol"] = de_df["names"].map(ensembl_to_hgnc_map)
    
    # Reorder columns for clarity
    cols = ["ensembl_id", "HGNC symbol"] + [col for col in de_df.columns if col not in ["ensembl_id", "HGNC symbol"]]
    de_df = de_df[cols]
    
    output_path = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary"
    de_df.to_csv(f"{output_path}/de_genes_heart_atlas_with_ensembl.csv", index=False)
    
    # Print summary
    print(f"\nDE analysis complete. Found {len(de_df)} significantly upregulated TFs.")
    print(f"TFs with HGNC symbols mapped: {de_df['HGNC symbol'].notna().sum()}/{len(de_df)}")

if __name__ == "__main__":
    main()