#%%
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
pth = "/data/rbg/users/seanmurphy/project_may/data/datasets/supplementary/"

tf_ids = pd.read_csv(pth + "human_tfs_lambert.csv")
tf_mapping = dict(zip(tf_ids["Ensembl ID"], tf_ids["HGNC symbol"]))
#%%
chip_counts = pd.read_csv(pth + "chip_edges_heart_source_counts.csv")
de_genes = pd.read_csv(pth + "de_genes_heart_atlas.csv")
ipr_scores = pd.read_csv(pth + "inverse_participation_ratio_tf_scores.csv")
centrality = pd.read_csv(pth + "tf_network_centrality.csv")
#%%
# Add symbol column to centrality dataframe using the TF column (Ensembl ID)
centrality["symbol"] = centrality["TF"].map(tf_mapping)
centrality = centrality[["TF", "symbol", "degree", "betweenness", "eigenvector"]]
# Calculate percentile for eigenvector centrality (higher values = higher percentile)
centrality["eigenvector_percentile"] = centrality["eigenvector"].rank(ascending=False, method="min", pct=True) * 100
# Sort by eigenvector percentile (highest percentile first)
centrality = centrality.sort_values("eigenvector_percentile", ascending=False)

#%%
cardiac_tfs = ["GATA4", "TBX5", "MEF2C", "HAND2", "HAND1", "NKX2-5"]

# Subset chip_counts by cardiac TFs
chip_subset = chip_counts[chip_counts['gene'].isin(cardiac_tfs)]
print("\nChIP-seq target counts percentiles for cardiac TFs:")
chip_percentiles = []
for tf in cardiac_tfs:
    tf_data = chip_subset[chip_subset['gene'] == tf]
    if not tf_data.empty:
        percentile = tf_data['percentile'].values[0]
        chip_percentiles.append(percentile)
        print(f"{tf}: {percentile:.2f}%")
    else:
        print(f"{tf}: Not found in ChIP-seq data")
if chip_percentiles:
    print(f"Average ChIP-seq percentile for cardiac TFs: {sum(chip_percentiles)/len(chip_percentiles):.2f}%")

# Subset de_genes by cardiac TFs
de_subset = de_genes[de_genes['names'].isin(cardiac_tfs)]
print("\nDifferential expression percentiles for cardiac TFs:")
de_percentiles = []
for tf in cardiac_tfs:
    tf_data = de_subset[de_subset['names'] == tf]
    if not tf_data.empty:
        percentile = tf_data['percentile'].values[0]
        de_percentiles.append(percentile)
        print(f"{tf}: {percentile:.2f}%")
    else:
        print(f"{tf}: Not found in DE data")
if de_percentiles:
    print(f"Average DE percentile for cardiac TFs: {sum(de_percentiles)/len(de_percentiles):.2f}%")

# Subset ipr_scores by cardiac TFs
ipr_subset = ipr_scores[ipr_scores['symbol'].isin(cardiac_tfs)]
print("\nInverse participation ratio percentiles for cardiac TFs:")
ipr_percentiles = []
for tf in cardiac_tfs:
    tf_data = ipr_subset[ipr_subset['symbol'] == tf]
    if not tf_data.empty:
        percentile = tf_data['percentile'].values[0]
        ipr_percentiles.append(percentile)
        print(f"{tf}: {percentile:.2f}%")
    else:
        print(f"{tf}: Not found in IPR data")
if ipr_percentiles:
    print(f"Average IPR percentile for cardiac TFs: {sum(ipr_percentiles)/len(ipr_percentiles):.2f}%")

# Subset centrality by cardiac TFs
centrality_subset = centrality[centrality['symbol'].isin(cardiac_tfs)]
print("\nEigenvector centrality percentiles for cardiac TFs:")
centrality_percentiles = []
for tf in cardiac_tfs:
    tf_data = centrality_subset[centrality_subset['symbol'] == tf]
    if not tf_data.empty:
        percentile = tf_data['eigenvector_percentile'].values[0]
        centrality_percentiles.append(percentile)
        print(f"{tf}: {percentile:.2f}%")
    else:
        print(f"{tf}: Not found in centrality data")
if centrality_percentiles:
    print(f"Average centrality percentile for cardiac TFs: {sum(centrality_percentiles)/len(centrality_percentiles):.2f}%")
#%%
# AGGREGATING PERCENTILES

# Step 1: Prepare DataFrames by standardizing column names for merging
# For chip_counts, 'gene' is the symbol, 'percentile' is the score
df_chip = chip_counts[['gene', 'percentile']].rename(columns={'gene': 'symbol', 'percentile': 'chip_percentile'})

# For de_genes, 'names' is the symbol, 'percentile' is the score
df_de = de_genes[['names', 'percentile']].rename(columns={'names': 'symbol', 'percentile': 'de_percentile'})

# For ipr_scores, 'symbol' is the symbol, 'percentile' is the score
df_ipr = ipr_scores[['symbol', 'percentile']].rename(columns={'percentile': 'ipr_percentile'})

# For centrality, 'symbol' is already the symbol, 'eigenvector_percentile' is the score
df_centrality = centrality[['symbol', 'eigenvector_percentile']]

# List of DataFrames and their associated information
all_dfs_info = [
    (df_chip, "ChIP-seq", "chip_percentile"),
    (df_de, "Differential Expression", "de_percentile"),
    (df_ipr, "Inverse Participation Ratio", "ipr_percentile"),
    (df_centrality, "Eigenvector Centrality", "eigenvector_percentile")
]

# cardiac_tfs is defined earlier in the script and will be used directly.

print("\n--- Aggregated Ranks for Cardiac TFs (All Permutations) ---")

# Define target permutations for top 100 gene extraction
target_permutations_for_top_genes = [
    "Differential Expression",
    "ChIP-seq, Differential Expression, Inverse Participation Ratio",
    "Differential Expression, Inverse Participation Ratio"
]
top_genes_from_selected_permutations = {}

# Variables to track the best permutation for cardiac TFs
best_avg_mean_percentile_cardiac = -1.0
best_permutation_name_cardiac = ""
best_avg_rank_cardiac = float('inf')  # Initialize with a very high number, as lower ranks are better
best_permutation_name_for_rank_cardiac = ""
all_permutation_summaries_list = [] # To store summary data for all permutations

for r_val in range(1, len(all_dfs_info) + 1): # Iterate from 1 to 4 (number of datasets to combine)
    for combo_info in combinations(all_dfs_info, r_val):
        dataset_names = [info[1] for info in combo_info]
        dataset_name_str = ', '.join(dataset_names)
        print(f"\nAggregating data from: {dataset_name_str}")

        if not combo_info:
            continue

        # Initialize merged_df with the first DataFrame in the combination
        first_df_obj, _, first_percentile_col = combo_info[0]
        # Ensure to make a copy to avoid SettingWithCopyWarning on slices later
        current_merged_df = first_df_obj[['symbol', first_percentile_col]].copy()
        
        current_percentile_cols = [first_percentile_col]

        # Merge with subsequent DataFrames in the combination
        for i in range(1, len(combo_info)):
            df_obj, _, percentile_col = combo_info[i]
            # Select only symbol and the specific percentile column before merging
            df_to_merge = df_obj[['symbol', percentile_col]]
            current_merged_df = pd.merge(current_merged_df, df_to_merge, on='symbol', how='inner')
            if current_merged_df.empty: # Optimization: if a merge results in empty, no need to continue for this combo
                break
            current_percentile_cols.append(percentile_col)

        if current_merged_df.empty:
            print(f"No common TFs found for this combination of datasets: {dataset_name_str}.")
            for tf_symbol in cardiac_tfs:
                print(f"{tf_symbol}: Not found (no common TFs in merged set for this combination: {dataset_name_str})")
            continue
            
        # Calculate mean percentile
        if len(current_percentile_cols) == 1:
            # If only one dataset, mean_percentile is just its percentile column
            current_merged_df['mean_percentile'] = current_merged_df[current_percentile_cols[0]]
        else:
            current_merged_df['mean_percentile'] = current_merged_df[current_percentile_cols].mean(axis=1)

        # Rank genes by mean percentile
        current_merged_df['rank'] = current_merged_df['mean_percentile'].rank(ascending=False, method='min')

        # Sort by rank for clarity
        current_merged_df = current_merged_df.sort_values('rank', ascending=True)

        # Extract top 100 genes if this is a targeted permutation
        # total_ranked_tfs_in_combo is len(current_merged_df) and will be used later for cardiac TF summary,
        # but we can capture it here for the top 100 gene summary as well.
        if dataset_name_str in target_permutations_for_top_genes:
            top_genes_list = current_merged_df.head(100)['symbol'].tolist()
            top_genes_from_selected_permutations[dataset_name_str] = {
                "genes": top_genes_list,
                "count": len(top_genes_list),
                "total_ranked_in_permutation": len(current_merged_df) # total TFs in this specific permutation
            }

        # Print ranks for cardiac TFs
        total_ranked_tfs_in_combo = len(current_merged_df)
        found_tfs_ranks_combo = {}
        # Filter for cardiac TFs present in the current merged dataframe
        cardiac_tfs_in_current_merged = current_merged_df[current_merged_df['symbol'].isin(cardiac_tfs)]
        
        for _, row_data in cardiac_tfs_in_current_merged.iterrows():
            found_tfs_ranks_combo[row_data['symbol']] = (int(row_data['rank']), row_data['mean_percentile'])

        for tf_symbol in cardiac_tfs:
            if tf_symbol in found_tfs_ranks_combo:
                rank_val, mean_perc_val = found_tfs_ranks_combo[tf_symbol]
                print(f"{tf_symbol}: Rank {rank_val} out of {total_ranked_tfs_in_combo} (Mean Percentile: {mean_perc_val:.2f})")
            else:
                print(f"{tf_symbol}: Not found in the set of genes shared across these datasets ({dataset_name_str}).")
        
        # Calculate average mean percentile for cardiac TFs in this combination
        cardiac_tf_mean_percentiles_in_combo = [val[1] for val in found_tfs_ranks_combo.values()]
        if cardiac_tf_mean_percentiles_in_combo: # Ensure there are found cardiac TFs to average
            current_avg_mean_percentile_cardiac = sum(cardiac_tf_mean_percentiles_in_combo) / len(cardiac_tf_mean_percentiles_in_combo)
            if current_avg_mean_percentile_cardiac > best_avg_mean_percentile_cardiac:
                best_avg_mean_percentile_cardiac = current_avg_mean_percentile_cardiac
                best_permutation_name_cardiac = dataset_name_str
            
            # Calculate average rank for cardiac TFs in this combination
            cardiac_tf_ranks_in_combo = [val[0] for val in found_tfs_ranks_combo.values()] # val[0] is the rank
            current_avg_rank_cardiac = sum(cardiac_tf_ranks_in_combo) / len(cardiac_tf_ranks_in_combo)
            if current_avg_rank_cardiac < best_avg_rank_cardiac:
                best_avg_rank_cardiac = current_avg_rank_cardiac
                best_permutation_name_for_rank_cardiac = dataset_name_str
        
        # Prepare data for the summary list for this permutation
        perm_summary_avg_mean_p = None
        perm_summary_avg_rank = None
        perm_num_cardiac_tfs_found = 0

        if cardiac_tf_mean_percentiles_in_combo: # Check if any cardiac TFs were processed
            perm_summary_avg_mean_p = current_avg_mean_percentile_cardiac
            perm_summary_avg_rank = current_avg_rank_cardiac
            perm_num_cardiac_tfs_found = len(cardiac_tf_mean_percentiles_in_combo)
        
        # total_ranked_tfs_in_combo is already len(current_merged_df) or 0 if empty
        all_permutation_summaries_list.append({
            "PermutationName": dataset_name_str,
            "AvgMeanPercentile_CardiacTFs": perm_summary_avg_mean_p,
            "AvgRank_CardiacTFs": perm_summary_avg_rank,
            "NumCardiacTFsFound": perm_num_cardiac_tfs_found,
            "TotalTFsRankedInPermutation": total_ranked_tfs_in_combo
        })
#%%
print("\n--- Summary ---")
if best_permutation_name_cardiac:
    print(f"The permutation with the highest average mean percentile for cardiac TFs was: '{best_permutation_name_cardiac}'")
    print(f"Average Mean Percentile for cardiac TFs in this permutation: {best_avg_mean_percentile_cardiac:.2f}%")
else:
    print("Could not determine the best permutation for average mean percentile (perhaps no cardiac TFs were found in any combination).")

if best_permutation_name_for_rank_cardiac:
    print(f"\nThe permutation with the best (lowest) average rank for cardiac TFs was: '{best_permutation_name_for_rank_cardiac}'")
    print(f"Average Rank for cardiac TFs in this permutation: {best_avg_rank_cardiac:.2f}")
else:
    print("Could not determine the best permutation for average rank (perhaps no cardiac TFs were found in any combination).")

# Save all permutation results to a CSV file
#summary_df = pd.DataFrame(all_permutation_summaries_list)
#output_csv_filename = "cardiac_tfs_aggregation_permutations_summary.csv"
## pth is defined at the beginning of the script
#output_csv_path = pth + output_csv_filename 
#summary_df.to_csv(output_csv_path, index=False, float_format='%.2f')
#print(f"\nSummary of all cardiac TF aggregation permutations saved to: {output_csv_path}")

# %%
# Convert the list of summaries to a DataFrame
summary_df = pd.DataFrame(all_permutation_summaries_list)

# Handle cases where 'AvgRank_CardiacTFs' or 'AvgMeanPercentile_CardiacTFs' might be None
# For ranking, we want to ensure that None values (due to no cardiac TFs found) are treated as worst.

# Rank by 'AvgRank_CardiacTFs' (lower is better)
# NaNs will be given the worst rank (largest number)
summary_df['rank_by_avg_rank'] = summary_df['AvgRank_CardiacTFs'].rank(method='min', ascending=True, na_option='bottom')

# Rank by 'AvgMeanPercentile_CardiacTFs' (higher is better)
# NaNs will be given the worst rank (largest number, because ascending=False means higher values get lower rank numbers)
summary_df['rank_by_avg_percentile'] = summary_df['AvgMeanPercentile_CardiacTFs'].rank(method='min', ascending=False, na_option='bottom')

# Calculate global rank as the average of the two ranks
# If one of the ranks is NaN (e.g., due to all values in a column being NaN if NumCardiacTFsFound is 0 for all),
# this could lead to NaN in global_rank. We'll handle this by only averaging available ranks.
# However, with na_option='bottom', ranks should always be numerical.
summary_df['global_rank'] = summary_df[['rank_by_avg_rank', 'rank_by_avg_percentile']].mean(axis=1)

# Sort by the global rank (lower is better)
sorted_summary_df = summary_df.sort_values('global_rank', ascending=True)

print("\n--- Permutations Ranked by Global Score (Avg of Rank by AvgRank and Rank by AvgPercentile) ---")
print("Lower Global Rank is better.")
print(sorted_summary_df[['PermutationName', 'AvgRank_CardiacTFs', 'AvgMeanPercentile_CardiacTFs', 'rank_by_avg_rank', 'rank_by_avg_percentile', 'global_rank', 'NumCardiacTFsFound', 'TotalTFsRankedInPermutation']].to_string())

# Save the detailed ranked summary to a CSV file
output_ranked_csv_filename = "cardiac_tfs_global_ranked_permutations.csv"
output_ranked_csv_path = pth + output_ranked_csv_filename
sorted_summary_df.to_csv(output_ranked_csv_path, index=False, float_format='%.2f')
print(f"\nDetailed ranked summary of all permutations saved to: {output_ranked_csv_path}")

# Display the original summary DF that was previously commented out (optional, but good for comparison)
# summary_df_original = pd.DataFrame(all_permutation_summaries_list) # Re-create if needed, or use from above
# output_csv_filename = "cardiac_tfs_aggregation_permutations_summary.csv"
# output_csv_path = pth + output_csv_filename
# summary_df_original.to_csv(output_csv_path, index=False, float_format='%.2f')
# print(f"\nOriginal summary of all cardiac TF aggregation permutations saved to: {output_csv_path}")
# %%

print("\n\n--- Top Genes from Selected Permutations ---")
for perm_name, data in top_genes_from_selected_permutations.items():
    print(f"\nPermutation: {perm_name}")
    print(f"Number of top genes extracted: {data['count']} (out of {data['total_ranked_in_permutation']} total TFs ranked in this permutation)")
    if data['genes']:
        # Sanitize permutation name for filename
        perm_name_safe = perm_name.replace(', ', '_').replace(' ', '_').replace('-', '_')
        output_filename = f"top_{data['count']}_genes_{perm_name_safe}.csv"
        
        # Create a DataFrame for the top genes with their rank in this specific permutation
        df_top_genes = pd.DataFrame({'symbol': data['genes']})
        df_top_genes['rank_in_permutation'] = range(1, len(data['genes']) + 1)
        
        # Save to CSV
        # Ensure 'pth' is defined; it's usually defined at the top of the script for path prefixes.
        full_output_path = pth + output_filename 
        df_top_genes.to_csv(full_output_path, index=False)
        print(f"Saved top genes to: {full_output_path}")
        
        # Print first few genes to console for quick review
        print("First 10 genes:", ", ".join(data['genes'][:10]) + ("..." if len(data['genes']) > 10 else ""))
    else:
        # This case should ideally not be hit if the permutation was processed and had TFs.
        print("No genes found or permutation did not produce results in the merged set.")
# %%
