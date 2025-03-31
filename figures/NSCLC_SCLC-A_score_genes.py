import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from multiprocessing import Pool
from pygam import LinearGAM, s

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
    """
    Bin cells by pseudotime into evenly spaced intervals.
    
    Parameters:
        adata (AnnData): The annotated data matrix.
        num_bins (int): Number of bins to divide pseudotime into.
        pseudotime (str): Column name in `adata.obs` containing pseudotime values.
    
    Returns:
        AnnData: Updated AnnData object with a new column `pseudotime_bin` containing binned pseudotime values.
    """
    # Perform evenly spaced binning using pd.cut
    adata.obs['pseudotime_bin'], bins = pd.cut(adata.obs[pseudotime], num_bins, labels=False, retbins=True)
    
    # Map integer bins back to the midpoint of each bin
    bin_midpoints = (bins[:-1] + bins[1:]) / 2  # Calculate midpoints of bins
    adata.obs['pseudotime_bin'] = adata.obs['pseudotime_bin'].map(lambda x: bin_midpoints[int(x)])
    
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

from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt

def plot_smoothed_mean_median_with_recurrence_heatmap(real_scores, proportion_enriched, figsize=(8, 6), fontsize=12, short_labels=None, color="royalblue", save_path=None, smoothing_method="savgol", window_length=5, polyorder=2, lowess_frac=0.1):
    """
    Creates a plot for smoothed mean median gene scores and recurrence scores across pseudotime bins,
    with a smoothed standard error area.
    
    Parameters:
        real_scores (dict): Dictionary of median gene scores per bin per gene set.
        proportion_enriched (dict): Dictionary of recurrence scores (proportion of enriched patients) per bin per gene set.
        figsize (tuple): Size of the figure.
        fontsize (int): Font size for labels.
        short_labels (dict): Optional dictionary for shorter labels for gene sets.
        color (str): Color for the plots.
        save_path (str): Path to save the figure. If None, the plot is displayed but not saved.
        smoothing_method (str): Smoothing method ("savgol" or "lowess").
        window_length (int): Window length for Savitzky-Golay filter.
        polyorder (int): Polynomial order for Savitzky-Golay filter.
        lowess_frac (float): Fraction of data used for LOWESS smoothing.
    """
    num_gene_sets = len(real_scores)
    global_vmax = 1  # Max value across all recurrence scores

    fig, axes = plt.subplots(num_gene_sets, 1, figsize=figsize, sharex=True, constrained_layout=True,
                             gridspec_kw={'height_ratios': [4] * num_gene_sets})

    if num_gene_sets == 1:
        axes = [axes]  # Ensure axes is always a list

    for ax, (gene_set, median_scores) in zip(axes, real_scores.items()):
        # Extract pseudotime bins and calculate mean and standard error
        grouped = median_scores.groupby('pseudotime_bin')
        mean_scores = grouped.mean()
        std_scores = grouped.std()
        count_scores = grouped.size()
        se_scores = std_scores / np.sqrt(count_scores)  # Standard Error

        bins = mean_scores.index
        patient_proportion = proportion_enriched[gene_set]  # Recurrence scores

        # Apply smoothing to mean median scores and error bounds
        if smoothing_method == "savgol":
            smoothed_mean_scores = savgol_filter(mean_scores.values, window_length=window_length, polyorder=polyorder)
            smoothed_upper_bound = savgol_filter((mean_scores + se_scores).values, window_length=window_length, polyorder=polyorder)
            smoothed_lower_bound = savgol_filter((mean_scores - se_scores).values, window_length=window_length, polyorder=polyorder)
        elif smoothing_method == "lowess":
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed_data_mean = lowess(mean_scores.values, bins.values, frac=lowess_frac)
            smoothed_data_upper = lowess((mean_scores + se_scores).values, bins.values, frac=lowess_frac)
            smoothed_data_lower = lowess((mean_scores - se_scores).values, bins.values, frac=lowess_frac)
            smoothed_mean_scores = smoothed_data_mean[:, 1]
            smoothed_upper_bound = smoothed_data_upper[:, 1]
            smoothed_lower_bound = smoothed_data_lower[:, 1]

        # Plot smoothed mean median score with smoothed standard error area
        ax.plot(bins, smoothed_mean_scores, marker='o', color=color, label='Smoothed Mean Median Score')
        ax.fill_between(bins, smoothed_lower_bound, smoothed_upper_bound, color=color, alpha=0.3, label='Smoothed Standard Error')

        # Set y-axis limits with padding
        y_min = smoothed_lower_bound.min()
        y_max = smoothed_upper_bound.max()
        padding = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - padding, y_max + padding)

        # Style adjustments for the subplot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

        y_label = short_labels.get(gene_set, gene_set) if short_labels else gene_set
        ax.set_ylabel(y_label, fontsize=fontsize, rotation=0, labelpad=50)

        # Add heatmap for recurrence scores below each plot
        ax_heatmap = ax.inset_axes([0, -0.15, 1, 0.1])  # Adjust position as needed
        X = np.linspace(bins.min(), bins.max(), len(patient_proportion) + 1)
        Y = np.array([0, 1])
        Z = patient_proportion.values.reshape(1, -1)

        heatmap = ax_heatmap.pcolormesh(X, Y, Z, cmap='Blues', shading='flat', vmin=0, vmax=global_vmax)

        # Style adjustments for the heatmap subplot
        ax_heatmap.spines["top"].set_visible(False)
        ax_heatmap.spines["right"].set_visible(False)
        ax_heatmap.spines["left"].set_visible(False)
        ax_heatmap.spines["bottom"].set_visible(False)
        ax_heatmap.tick_params(axis="both", which="both", length=0)
        ax_heatmap.set_yticklabels([])
        ax_heatmap.set_xticklabels([])

    # Set x-axis label on the last subplot
    axes[-1].set_xlabel("Aligned Pseudotime", fontsize=fontsize)  
    axes[-1].tick_params(axis='x', pad=15)

    # Add a colorbar for heatmaps
    cbar = fig.colorbar(heatmap, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.08, aspect=40)
    cbar.set_label('Recurrence Score (Proportion Enriched)', fontsize=fontsize)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()



adata = sc.read_h5ad("/scratch/chanj3/wangm10/NSCLC_SCLC-A.h5ad")  # Load your single-cell dataset

#------------------------#
# Cell Marker Gene Sets #
#------------------------#
with open("/scratch/chanj3/wangm10/cellmarker_dict.json", 'r') as f:
    cellmarker_gene_sets = json.load(f)

# Run pipeline
adata = compute_gene_scores(adata, cellmarker_gene_sets)
adata = normalize_scores(adata, cellmarker_gene_sets, subject_label='subject')
adata = bin_pseudotime(adata, num_bins=15, pseudotime='aligned_score')

# Compute real enrichment
cellmarker_real_scores = compute_median_scores(adata, cellmarker_gene_sets)

# Generate null distributions via permutation
cellmarker_null_distributions = permutation_test(adata, cellmarker_gene_sets, num_permutations=100)

# Compute statistical significance
cellmarker_p_values = compute_significance(cellmarker_real_scores, cellmarker_null_distributions, alpha=0.05)

cellmarker_proportion_enriched = compute_patient_proportion(cellmarker_p_values, adata)

# Call the function to visualize the results
plot_enrichment_and_proportion(cellmarker_real_scores, cellmarker_proportion_enriched)

plot_smoothed_mean_median_with_recurrence_heatmap(cellmarker_real_scores, cellmarker_proportion_enriched, 
                                                  figsize=(7, 7), fontsize=15, short_labels=None, color="royalblue", 
                                                  save_path="NSCLC_SCLC-A_score_genes_cellmarker.png")

#------------------------#
# Pathway Gene Sets #
#------------------------#  
with open("/scratch/chanj3/wangm10/pathway_dict.json", 'r') as f:
    pathway_gene_sets = json.load(f)

# Run pipeline
adata = compute_gene_scores(adata, pathway_gene_sets)
adata = normalize_scores(adata, pathway_gene_sets, subject_label='subject')
adata = bin_pseudotime(adata, num_bins=15, pseudotime='aligned_score')

# Compute real enrichment
pathway_real_scores = compute_median_scores(adata, pathway_gene_sets)

# Generate null distributions via permutation
pathway_null_distributions = permutation_test(adata, pathway_gene_sets, num_permutations=100)

# Compute statistical significance
pathway_p_values = compute_significance(pathway_real_scores, pathway_null_distributions, alpha=0.05)

pathway_proportion_enriched = compute_patient_proportion(pathway_p_values, adata)

# Call the function to visualize the results
plot_enrichment_and_proportion(pathway_real_scores, pathway_proportion_enriched)

plot_smoothed_mean_median_with_recurrence_heatmap(pathway_real_scores, pathway_proportion_enriched, 
                                                  figsize=(7, 7), fontsize=15, short_labels=None, color="royalblue", 
                                                  save_path="NSCLC_SCLC-A_score_genes_pathway.png")
