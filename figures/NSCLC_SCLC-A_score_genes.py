import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
import json
from tqdm import tqdm

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
def bin_pseudotime(adata, num_bins=10, pseudotime='pseudotime'):
    adata.obs['pseudotime_bin'] = pd.qcut(adata.obs[pseudotime], num_bins, labels=False)
    return adata

# Step 4: Compute Enrichment Scores Per Patient Per Bin
def compute_enrichment(adata, gene_sets, threshold_percentile=95, subject_label='subject'):
    enrichment_scores = {}
    for gene_set in gene_sets:
        T_g = np.percentile(adata.obs[gene_set], threshold_percentile)
        grouped = adata.obs.groupby([subject_label, 'pseudotime_bin'])
        enrichment_scores[gene_set] = grouped.apply(lambda x: np.mean(x[gene_set] > T_g))
    return enrichment_scores

# Step 5: Permutation Test
def permutation_test(adata, gene_sets, num_permutations=1000, subject_label='subject'):
    null_distributions = {gene_set: [] for gene_set in gene_sets}
    
    for _ in tqdm(range(num_permutations), desc="Running permutations"):
        shuffled_adata = adata.copy()
        for patient in adata.obs[subject_label].unique():
            subset = shuffled_adata.obs[subject_label] == patient
            for gene_set in gene_sets:
                shuffled_adata.obs[gene_set][subset] = np.random.permutation(shuffled_adata.obs[gene_set][subset])

        null_scores = compute_enrichment(shuffled_adata, gene_sets)
        for gene_set in gene_sets:
            null_distributions[gene_set].append(null_scores[gene_set])

    adata.uns['null_distributions'] = null_distributions
    return null_distributions

# Step 6: Compute Significance
def compute_significance(real_scores, null_distributions, alpha=0.05):
    """
    Compute p-values and determine which patients are significantly enriched in each bin.
    """
    p_values = {}
    enrichment_flags = {}

    for gene_set in real_scores:
        real_values = real_scores[gene_set]
        null_values = np.array(null_distributions[gene_set])
        
        # Compute p-values for each (patient, bin) pair
        p_vals = real_values.apply(lambda x: np.mean(null_values >= x))
        
        # Correct for multiple testing
        p_corrected = multipletests(p_vals, method='fdr_bh')[1]
        
        # Store p-values
        p_values[gene_set] = pd.DataFrame({'pseudotime_bin': real_values.index, 'p_value': p_corrected})

        # Determine which patients are significantly enriched
        enrichment_flags[gene_set] = pd.DataFrame({'pseudotime_bin': [x[1] for x in real_values.index], 
                                                   'enriched': (p_corrected < alpha).astype(int)})

    return p_values, enrichment_flags

# Step 7: compute recurrence score
def compute_patient_proportion(enrichment_flags, adata, subject_label="subject"):
    """
    Computes the proportion of patients with significant enrichment in each pseudotime bin.
    """
    patient_counts = adata.obs[subject_label].nunique()  # Total number of patients
    proportions = {}

    for gene_set, flags in enrichment_flags.items():
        # Count patients enriched per pseudotime bin
        enriched_counts = flags.groupby('pseudotime_bin').sum()
        proportions[gene_set] = enriched_counts / patient_counts  # Normalize
        
    return proportions

# Example Usage
adata = sc.read_h5ad("/scratch/chanj3/wangm10/NSCLC_SCLC-A.h5ad")  # Load your single-cell dataset

# Define gene sets
with open("/scratch/chanj3/wangm10/cellmarker_dict.json", 'r') as f:
    gene_sets = json.load(f)

# Run pipeline
adata = compute_gene_scores(adata, gene_sets)
adata = normalize_scores(adata, gene_sets, subject_label='subject')
adata = bin_pseudotime(adata, num_bins=10, pseudotime='aligned_score')

# Compute real enrichment
real_scores = compute_enrichment(adata, gene_sets, threshold_percentile=90)

# Generate null distributions via permutation
null_distributions = permutation_test(adata, gene_sets, num_permutations=500)

# Compute statistical significance
p_values, enrichment_flags = compute_significance(real_scores, null_distributions)

proportion_enriched = compute_patient_proportion(enrichment_flags, adata)

print(proportion_enriched)