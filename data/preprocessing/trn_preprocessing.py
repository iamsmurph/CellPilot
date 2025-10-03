# %%
import anndata as ad
import scanpy as sc
import pandas as pd

def qc_filter(a: ad.AnnData):
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.log1p(a)
    return a

data_path = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/ventricular_cardiomyocytes.h5ad"
tf_path = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/human_tfs_lambert.csv"
    
# Load and preprocess the data
cardiac = ad.read_h5ad(data_path)
cardiac = qc_filter(cardiac)

# Load transcription factors
tfs = pd.read_csv(tf_path, index_col=0)
unique_tfs = tfs["Ensembl ID"].unique()

# Split into TFs and non-TFs
cardiac_tfs_mask = cardiac.var_names.isin(unique_tfs)
cardiac_tfs_only = cardiac[:, cardiac_tfs_mask].copy()
cardiac_no_tfs = cardiac[:, ~cardiac_tfs_mask].copy()

# Extract top variable genes
sc.pp.highly_variable_genes(cardiac_no_tfs, n_top_genes=5000)
cardiac_no_tfs = cardiac_no_tfs[:, cardiac_no_tfs.var.highly_variable].copy()

# Create combined dataset
combined_data = ad.concat(
    [cardiac_tfs_only, cardiac_no_tfs],
    axis=1,
    label="batch",
    keys=["TFs", "Non-TFs"],
    merge="same",
)
combined_data.var_names_make_unique()  # ensure unique gene names

# Scale the data
sc.pp.scale(combined_data)

combined_data.write_h5ad("/data/rbg/users/seanmurphy/project_may/data/datasets/train/trn_combined_data.h5ad")

# %%
