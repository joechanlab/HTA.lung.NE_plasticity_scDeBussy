import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from multiprocessing import Pool

# Step 1: Compute Gene Set Scores (Single Computation for All Patients)
def compute_gene_scores(adata, gene_sets, layer="X"):
    for gene_set_name, gene_list in gene_sets.items():
        sc.tl.score_genes(adata, gene_list, score_name=gene_set_name, use_raw=(layer == "raw"))
    return adata

# Step 2: Normalize Scores Per Patient and Gene Set
def normalize_scores(adata, gene_sets, subject_label='subject'):
    patients = adata.obs[subject_label].unique()
    for gene_set in gene_sets:
        # Normalize per patient
        for patient in patients:
            subset = adata.obs[subject_label] == patient
            adata.obs.loc[subset, gene_set] = zscore(adata.obs.loc[subset, gene_set])
            adata.obs[gene_set][subset] = zscore(adata.obs[gene_set][subset])
        # Normalize across all patients for comparability
        adata.obs[gene_set] = zscore(adata.obs[gene_set])
    return adata

# Step 3: Bin Cells by Pseudotime
def bin_pseudotime(adata, num_bins=15, pseudotime='pseudotime'):
    adata.obs['pseudotime_bin'] = pd.qcut(adata.obs[pseudotime], num_bins, labels=False)
    return adata


def compute_median_scores(adata, gene_sets, subject_label='subject'):
    median_scores = {}
    grouped = adata.obs.groupby([subject_label, 'pseudotime_bin'])
    for gene_set in gene_sets:
        median_scores[gene_set] = grouped[gene_set].median()
    return median_scores

# Step 5: Permutation Test
def compute_permutation(args):
    """
    Helper function to compute permutations for a single gene set.
    """
    gene_set_name, random_gene_sets, adata, subject_label, layer = args
    null_scores = []
    for random_genes in random_gene_sets[gene_set_name]:
        sc.tl.score_genes(adata, random_genes, score_name="random_gene_set", use_raw=(layer == "raw"))
        grouped = adata.obs.groupby([subject_label, 'pseudotime_bin'], observed=True)
        median_scores = grouped["random_gene_set"].median()
        null_scores.append(median_scores)
    return gene_set_name, null_scores

def permutation_test(adata, gene_sets, num_permutations=500, subject_label='subject', layer="X"):
    """
    Perform permutation testing by selecting random gene sets of the same size as each given gene set,
    excluding genes already in the original gene set.
    """
    # Precompute filtered genes
    excluded_genes = set()
    for gene_list in gene_sets.values():
        excluded_genes.update(gene_list)
    filtered_genes = [gene for gene in adata.var_names if gene not in excluded_genes]

    # Generate all random gene sets at once
    random_gene_sets = {
        gene_set_name: [np.random.choice(filtered_genes, size=len(gene_list), replace=False)
                        for _ in range(num_permutations)]
        for gene_set_name, gene_list in gene_sets.items()
    }

    # Prepare arguments for multiprocessing
    args = [
        (gene_set_name, random_gene_sets, adata, subject_label, layer)
        for gene_set_name in gene_sets.keys()
    ]

    # Use multiprocessing to parallelize computation with a progress bar
    null_distributions = {}
    with Pool() as pool:
        # Wrap pool.imap_unordered with tqdm to show progress
        for result in tqdm(pool.imap_unordered(compute_permutation, args), total=len(args), desc="Processing permutations"):
            gene_set_name, scores = result
            null_distributions[gene_set_name] = scores

    # Store results in adata
    adata.uns['null_distributions'] = null_distributions
    return null_distributions


# Step 6: Compute Significance
def compute_significance(real_scores, null_distributions, alpha=0.05):
    """
    Compute p-values and determine which patients are significantly enriched in each bin.
    """
    p_values = {}

    for gene_set in real_scores:
        real_values = real_scores[gene_set]
        null_values = np.array(null_distributions[gene_set])
        
        # Compute p-values for each (patient, bin) pair based on null distribution
        p_vals = real_values.apply(lambda x: np.mean(null_values.flatten() > x))
        
        # Correct for multiple testing using FDR correction
        p_corrected = multipletests(p_vals.values.flatten(), method='fdr_bh')[1]
        
        # Store p-values and enrichment flags (significant or not)
        p_values[gene_set] = pd.DataFrame({'pseudotime_bin': real_values.index.get_level_values('pseudotime_bin'),
                                           'patient': real_values.index.get_level_values('subject'),
                                           'p_value': p_corrected,
                                           'flag': (p_corrected < alpha).astype(int)})

    return p_values

# Step 7: compute recurrence score
def compute_patient_proportion(p_values, adata, subject_label="subject"):
    """
    Computes the proportion of patients with significant enrichment in each pseudotime bin.
    """
    patient_counts = adata.obs[subject_label].nunique()  # Total number of patients
    proportions = {}

    for gene_set, flags in p_values.items():
        # Count patients enriched per pseudotime bin
        enriched_counts = flags.loc[:,['pseudotime_bin', 'flag']].groupby('pseudotime_bin').sum()
        proportions[gene_set] = enriched_counts / patient_counts  # Normalize
        
    return proportions

def plot_enrichment_and_proportion(real_scores, proportion_enriched):
    """
    Plots the real enrichment scores and proportion of enriched patients across pseudotime bins.
    """
    for gene_set in real_scores.keys():
        plt.figure(figsize=(12, 5))

        # First subplot: Real enrichment scores (median scores)
        plt.subplot(1, 2, 1)
        mean_scores = real_scores[gene_set].groupby('pseudotime_bin').mean()
        std_scores = real_scores[gene_set].groupby('pseudotime_bin').std()
        bins = mean_scores.index

        plt.plot(bins, mean_scores, marker='o', label='Mean Median Score')
        plt.fill_between(bins, mean_scores - std_scores, mean_scores + std_scores, alpha=0.3)
        plt.xlabel('Pseudotime Bin')
        plt.ylabel('Median Gene Score')
        plt.title(f'Median Gene Score for {gene_set}')
        plt.legend()

        # Second subplot: Proportion of enriched patients
        plt.subplot(1, 2, 2)
        if gene_set in proportion_enriched:
            proportion = proportion_enriched[gene_set]
            plt.plot(proportion.index, proportion.values.flatten(), marker='s', color='r', label='Proportion Enriched')
            plt.xlabel('Pseudotime Bin')
            plt.ylabel('Proportion of Enriched Patients')
            plt.ylim(0, 1)
            plt.title(f'Proportion of Enriched Patients for {gene_set}')
            plt.legend()

        plt.tight_layout()
        plt.show()

# Example Usage
adata = sc.read_h5ad("/scratch/chanj3/wangm10/NSCLC_SCLC-A.h5ad")  # Load your single-cell dataset

# Define gene sets
with open("/scratch/chanj3/wangm10/cellmarker_dict.json", 'r') as f:
    gene_sets = json.load(f)

# Run pipeline
adata = compute_gene_scores(adata, gene_sets)
adata = normalize_scores(adata, gene_sets, subject_label='subject')
adata = bin_pseudotime(adata, num_bins=15, pseudotime='aligned_score')

# Compute real enrichment
real_scores = compute_median_scores(adata, gene_sets)

# Generate null distributions via permutation
null_distributions = permutation_test(adata, gene_sets, num_permutations=100)

# Compute statistical significance
p_values = compute_significance(real_scores, null_distributions, alpha=0.05)

proportion_enriched = compute_patient_proportion(p_values, adata)

print(proportion_enriched)

# Call the function to visualize the results
plot_enrichment_and_proportion(real_scores, proportion_enriched)