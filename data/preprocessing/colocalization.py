#%%
import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

#%%
# Load and preprocess Jaccard pairs data
pth = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/tf_jaccard_pairs.tsv"
df = pd.read_csv(pth, sep="\t")
df = df.dropna()
df['Jaccard_percentile'] = df['Jaccard'].rank(pct=True)

#%%
# Load gold standard pairs
pth = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/tf_co_localization_pairs_gold.csv"
#pth = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/tf_co_localization_pairs_gold_top10.csv"
df_gold = pd.read_csv(pth, index_col=0)

#%%
# Find common pairs between datasets
pairs_df_1 = list(zip(df.TF1, df.TF2))
pairs_df_2 = list(zip(df.TF2, df.TF1))
pairs_df = set(pairs_df_1 + pairs_df_2)
pairs_df_gold = set(zip(df_gold.TF1, df_gold.TF2))
common_pairs = pairs_df_gold.intersection(pairs_df)

#%%
# Get unique genes from common pairs
unique_genes = {gene for pair in common_pairs for gene in pair}

#%%
# Filter dataframe for common pairs
mask = df.apply(lambda row: (row['TF1'], row['TF2']) in common_pairs or (row['TF2'], row['TF1']) in common_pairs, axis=1)
df_common = df[mask].reset_index(drop=True)
#%%
# Plot distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot Jaccard percentiles
ax1.hist(df_common['Jaccard_percentile'], bins=30)
ax1.set_title('Jaccard Percentiles for Common Pairs, CHIP-seq')
ax1.set_xlabel('Percentile')
ax1.set_ylabel('Count')

# Plot raw Jaccard scores
ax2.hist(df['Jaccard'], bins=30, alpha=0.7, label='All Pairs')
for jaccard in df_common['Jaccard']:
    ax2.axvline(x=jaccard, color='red', alpha=0.3)
ax2.set_title('Jaccard Scores Distribution, CHIP-seq')
ax2.set_xlabel('Jaccard Score')
ax2.set_ylabel('Count')
ax2.legend()

plt.tight_layout()
plt.show()

#%%
# Load and process embeddings
pth = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/evo2_7b_tf_transcripts_all_isoforms.pt"
data = torch.load(pth, weights_only=False)
evo_embeddings = {x["refseq"]: x["embedding"] for x in data}

#%%
# Compute cosine similarities
cosine_similarities = []
missing_pairs = []

for _, row in df.iterrows():
    tf1, tf2 = row['TF1'], row['TF2']
    
    if tf1 in evo_embeddings and tf2 in evo_embeddings:
        emb1 = torch.tensor(evo_embeddings[tf1])
        emb2 = torch.tensor(evo_embeddings[tf2])
        cos_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        cosine_similarities.append({'TF1': tf1, 'TF2': tf2, 'cosine_similarity': cos_sim})
    else:
        missing_pairs.append((tf1, tf2))

# Convert to DataFrame and compute percentiles
cos_sim_df = pd.DataFrame(cosine_similarities)
cos_sim_df["sim_percentile"] = cos_sim_df["cosine_similarity"].rank(pct=True)

#%%
# Get similarities for common pairs
common_pairs_similarities = []

for tf1, tf2 in common_pairs:
    mask = ((cos_sim_df['TF1'] == tf1) & (cos_sim_df['TF2'] == tf2)) | \
           ((cos_sim_df['TF1'] == tf2) & (cos_sim_df['TF2'] == tf1))
    
    if mask.any():
        similarity = cos_sim_df[mask]['sim_percentile'].iloc[0]
        common_pairs_similarities.append({
            'TF1': tf1,
            'TF2': tf2,
            'similarity_percentile': similarity
        })

common_pairs_sim_df = pd.DataFrame(common_pairs_similarities)
#%%
# Plot distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot similarity percentiles
ax1.hist(common_pairs_sim_df['similarity_percentile'], bins=30)
ax1.set_title('Similarity Percentiles for Common Pairs, Evo2')
ax1.set_xlabel('Percentile')
ax1.set_ylabel('Count')

# Plot raw cosine similarities
ax2.hist(cos_sim_df['cosine_similarity'], bins=30, alpha=0.7, label='All Pairs')
for similarity in common_pairs_sim_df['similarity_percentile']:
    ax2.axvline(x=similarity, color='red', alpha=0.3)
ax2.set_title('Cosine Similarity Distribution, Evo2')
ax2.set_xlabel('Cosine Similarity')
ax2.set_ylabel('Count')
ax2.legend()

plt.tight_layout()
plt.show()
#%%
# Print summary statistics
print("\nSummary Statistics:")
print("Number of common pairs in df_common ", len(df_common))
print(f"Mean Jaccard percentile for common pairs: {df_common.Jaccard_percentile.mean():.3f}")
print("Number of common pairs in common_pairs_sim_df ", len(common_pairs_sim_df))
print(f"Mean similarity percentile for common pairs: {common_pairs_sim_df['similarity_percentile'].mean():.3f}")

# %%
