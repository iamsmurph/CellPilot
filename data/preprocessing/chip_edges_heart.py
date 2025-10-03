# %%
import scanpy as sc
import anndata as ad
import pandas as pd
import gc
import numpy as np
import matplotlib.pyplot as plt
import os
#%%

cardiac = ad.read_h5ad(
    "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/ventricular_cardiomyocytes.h5ad"
)
#%%
pth = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/human_tfs_lambert.csv"
tfs = pd.read_csv(pth)
unique_tfs = tfs["Ensembl ID"].unique()
# %%
ensembl_to_hgnc = dict(zip(tfs["Ensembl ID"], tfs["HGNC symbol"]))
hgnc_to_ensembl = dict(zip(tfs["HGNC symbol"], tfs["Ensembl ID"]))
#%%
shared_genes = set(cardiac.var_names).intersection(unique_tfs)
shared_genes_HGNC = [ensembl_to_hgnc[gene] for gene in shared_genes]
#%%
pth = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/chip_all_edges_symbol_hgnc.tsv"
edges = pd.read_csv(pth, sep="\t")
#%%
edges_filtered = edges[edges["source"].isin(shared_genes_HGNC)]
source_counts = edges_filtered["source"].value_counts().to_dict()
source_count_df = pd.DataFrame({
    'gene': list(source_counts.keys()),
    'target_count': list(source_counts.values())
})
source_count_df = source_count_df.sort_values('target_count', ascending=False).reset_index(drop=True)
#%%
total_genes = len(source_count_df)
# Calculate percentile directly without using rank
source_count_df['percentile'] = 100 * (1 - source_count_df.index / total_genes)

print(f"\nTotal DE genes found as sources in edges dataset: {total_genes}")
print(f"Average number of targets per DE gene: {source_count_df['target_count'].mean():.2f}")
print(f"Maximum number of targets: {source_count_df['target_count'].max()} (Gene: {source_count_df.iloc[0]['gene']})")
print(f"Minimum number of targets: {source_count_df['target_count'].min()} (Gene: {source_count_df.iloc[-1]['gene']})")

print("\nCardiac TFs by number of targets:")
print("TF\tNumber of Targets\tPercentile")
print("-" * 50)
cardiac_tfs = ["GATA4", "TBX5", "MEF2C", "HAND2", "HAND1", "NKX2-5"]

for tf in cardiac_tfs:
    tf_data = source_count_df[source_count_df['gene'] == tf]
    if not tf_data.empty:
        count = tf_data['target_count'].values[0]
        percentile = tf_data['percentile'].values[0]
        print(f"{tf}\t{count}\t{percentile:.2f}%")
    else:
        print(f"{tf}\tNot found as source in edges dataset")

#%%
# Save the source count dataframe to a CSV file
output_path = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary"
source_count_df.to_csv(f"{output_path}/chip_edges_heart_source_counts.csv", index=False)
print(f"Source count dataframe saved to {output_path}/chip_edges_heart_source_counts.csv")

# %%
