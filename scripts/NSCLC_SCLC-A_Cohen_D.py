import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde, ttest_ind_from_stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from pygam import LinearGAM, s
import anndata
import scanpy as sc
from scdebussy.pp import stratified_downsample
from scdebussy.tl import aligner
import json

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0

def compute_mad_threshold_per_patient(effect_sizes, scale_factor=1.4826, threshold_factor=1.5, percentile_cutoff=90):
    """
    Compute a per-patient threshold using the Median Absolute Deviation (MAD) with stricter filtering.

    Parameters:
        effect_sizes (dict): Dictionary mapping patients to their effect sizes across time bins.
        scale_factor (float): Constant to make MAD comparable to standard deviation (default: 1.4826).
        threshold_factor (float): Multiplier for setting the threshold (increase for stricter cutoff).
        percentile_cutoff (int): Percentile to use for additional filtering (default: 90).

    Returns:
        dict: Dictionary mapping patients to their stricter MAD-based effect size thresholds.
    """
    patient_thresholds = {}

    for patient, patient_effect_sizes in effect_sizes.items():
        if len(patient_effect_sizes) == 0:  # Handle missing or empty data
            patient_thresholds[patient] = 0  # Default threshold
            continue

        median_effect = np.median(patient_effect_sizes)
        mad = scale_factor * np.median(np.abs(patient_effect_sizes - median_effect))
        mad_threshold = median_effect + threshold_factor * mad

        # Add percentile-based strictness
        percentile_threshold = np.percentile(patient_effect_sizes, percentile_cutoff)
        
        # Use the stricter of MAD or percentile cutoff
        patient_thresholds[patient] = max(mad_threshold, percentile_threshold)

    return patient_thresholds

def compute_enrichment_scores(df, gene_set, pseudotime_col, subject_col, num_bins, p_value_threshold):
    """
    Compute enrichment scores for a gene set across pseudotime bins.
    
    Modifications:
    - Uses dynamic effect size thresholding per patient.
    - Applies a Gaussian-weighted rolling window to smooth patient recurrence scores.
    """
    df['pseudotime_bin'] = pd.cut(df[pseudotime_col], bins=num_bins, labels=False)
    
    effect_sizes = {}
    p_values = {}
    patient_effect_sizes = {patient: [] for patient in df[subject_col].unique()}
    
    for b in range(num_bins):
        subset = df[df['pseudotime_bin'] == b]
        patients = subset[subject_col].unique()
        
        effect_sizes[b] = {}
        p_values[b] = {}
        
        for patient in patients:
            patient_data = subset[subset[subject_col] == patient]
            
            gene_set_expr = patient_data[gene_set].values.flatten()
            background_expr = patient_data.drop(columns=gene_set + [subject_col, pseudotime_col, 'pseudotime_bin']).values.flatten()
            
            effect_size = cohens_d(gene_set_expr, background_expr)
            effect_sizes[b][patient] = effect_size
            patient_effect_sizes[patient].append(effect_size)
            
            t_stat, p_value = stats.ttest_ind(gene_set_expr, background_expr)
            p_values[b][patient] = p_value
    
    # Compute dynamic thresholds
    patient_thresholds = compute_mad_threshold_per_patient(patient_effect_sizes)
    
    # Compute enrichment scores
    bin_centers, enrichment_scores, patient_proportions, standard_errors = [], [], [], []
    
    for b in range(num_bins):
        subset = df[df['pseudotime_bin'] == b]
        patients = subset[subject_col].unique()
        
        if len(patients) == 0:
            bin_centers.append(np.nan)
            enrichment_scores.append(0)
            patient_proportions.append(0)
            standard_errors.append(0)
            continue
        
        patient_scores = [effect_sizes[b][patient] for patient in patients]
        patient_p_values = [p_values[b][patient] for patient in patients]
        
        _, adjusted_p_values, _, _ = multipletests(patient_p_values, method='fdr_bh')
        significant_patients = sum((score > patient_thresholds.get(patient)) & (adj_p < p_value_threshold) 
                                    for score, adj_p, patient in zip(patient_scores, adjusted_p_values, patients))
        
        bin_centers.append(df[df['pseudotime_bin'] == b][pseudotime_col].mean())
        enrichment_scores.append(np.mean(patient_scores))
        standard_errors.append(np.std(patient_scores) / np.sqrt(len(patient_scores)))
        patient_proportions.append(significant_patients / len(patients))
    
    # Apply Gaussian-weighted rolling window smoothing
    window_size = 5  # Increased window size for smoother results
    gaussian_weights = np.exp(-0.5 * (np.linspace(-2, 2, window_size) ** 2))
    gaussian_weights /= gaussian_weights.sum()
    smoothed_proportions = np.convolve(patient_proportions, gaussian_weights, mode='same')
    
    enrichment_df = pd.DataFrame({
        'pseudotime': bin_centers,
        'enrichment': enrichment_scores,
        'standard_error': standard_errors,
        'patient_proportion': smoothed_proportions  # More smoothed recurrence score
    }).dropna()
    
    return enrichment_df

def plot_enrichment_results(df, gene_set_name):
    """
    Plots enrichment results for a gene set, including Cohen's D with error bars, GAM fit, and patient proportions.
    
    Parameters:
        df (pd.DataFrame): The input dataframe containing pseudotime, enrichment, standard_error, and patient_proportion.
        gene_set_name (str): Name of the gene set being plotted.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
    
    # GAM fitting on averaged enrichment scores
    gam_enrichment = LinearGAM(s(0, n_splines=10)).fit(df[['pseudotime']], df['enrichment'])
    x_vals = np.linspace(df['pseudotime'].min(), df['pseudotime'].max(), 100)
    y_pred = gam_enrichment.predict(x_vals)
    
    # Plot Cohen's D with error bars and GAM fit
    ax1.errorbar(df['pseudotime'], df['enrichment'], yerr=df['standard_error'], fmt='o', alpha=0.5, label='Binned Averages')
    ax1.plot(x_vals, y_pred, color='black', label='GAM Fit')
    
    # Plot patient proportions
    ax2.plot(df['pseudotime'], df['patient_proportion'], color='blue', label='Patient Proportion')
    
    # Set labels and titles
    ax1.set_ylabel("Effect Size (Cohen's d)")
    ax1.set_title(f"Gene Set Enrichment: {gene_set_name}")
    ax1.legend()
    
    ax2.set_xlabel("Pseudotime")
    ax2.set_ylabel("Proportion of Enriched Patients")
    ax2.legend()
    ax2.set_title(f"Proportion of Patients with Significant Enrichment: {gene_set_name}")
    
    plt.tight_layout()
    plt.show()

def compute_gene_set_enrichment(df, gene_set, gene_set_name, pseudotime_col='aligned_score', subject_col='subject',
                                            num_bins=20, p_value_threshold=0.05):
    gene_set = [gene for gene in gene_set if gene in df.columns]
    results = compute_enrichment_scores(df, gene_set, pseudotime_col, subject_col, num_bins, p_value_threshold)
    plot_enrichment_results(results, gene_set_name)
    return results

def plot_cohens_d_ridge(results, figsize=(8, 6), fontsize=12, short_labels=None, color="royalblue", save_path=None):
    num_gene_sets = len(results)
    global_vmax = max(df['patient_proportion'].max() for df in results.values())

    fig, axes = plt.subplots(num_gene_sets, 1, figsize=figsize, sharex=True, constrained_layout=True, 
                             gridspec_kw={'height_ratios': [4] * num_gene_sets})

    if num_gene_sets == 1:
        axes = [axes]  

    for ax, (gene_set, df) in zip(axes, results.items()):
        x_vals = df['pseudotime']
        y_vals = df['enrichment']
        y_errs = df['standard_error']
        patient_proportion = df['patient_proportion']

        gam = LinearGAM(s(0, n_splines=10)).fit(x_vals, y_vals)
        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_pred = gam.predict(x_smooth)
        # Fit separate GAMs for error bands
        gam_low = LinearGAM(s(0, n_splines=10)).fit(x_vals, y_vals - y_errs)
        gam_high = LinearGAM(s(0, n_splines=10)).fit(x_vals, y_vals + y_errs)

        # Predict smooth confidence interval bounds
        y_err_low = gam_low.predict(x_smooth)
        y_err_high = gam_high.predict(x_smooth)
        
        ax.fill_between(
            x_smooth, y_err_low, y_err_high, color=color, alpha=0.3  # Shaded error band
        )
        ax.plot(x_smooth, y_pred, color=color, lw=2)

        y_min = min((y_vals - y_errs).min(), y_pred.min())
        y_max = max((y_vals + y_errs).max(), y_pred.max())
        padding = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - padding, y_max + padding)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        #ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

        y_label = short_labels.get(gene_set, gene_set) if short_labels else gene_set
        ax.set_ylabel(y_label, fontsize=fontsize, rotation=0, labelpad=50)

        ax_heatmap = ax.inset_axes([0, -0.15, 1, 0.1])  
        X = np.linspace(x_vals.min(), x_vals.max(), len(x_vals) + 1)
        Y = np.array([0, 1])
        Z = patient_proportion.values.reshape(1, -1)

        heatmap = ax_heatmap.pcolormesh(X, Y, Z, cmap='Blues', shading='flat', vmin=0, vmax=global_vmax)

        ax_heatmap.spines["top"].set_visible(False)
        ax_heatmap.spines["right"].set_visible(False)
        ax_heatmap.spines["left"].set_visible(False)
        ax_heatmap.spines["bottom"].set_visible(False)
        ax_heatmap.tick_params(axis="both", which="both", length=0)
        ax_heatmap.set_yticklabels([])
        ax_heatmap.set_xticklabels([])

    axes[-1].set_xlabel("Pseudotime", fontsize=fontsize)  
    axes[-1].tick_params(axis='x', pad=10)

    plt.subplots_adjust(bottom=0.2)  # More space for x-axis labels

    cbar = fig.colorbar(heatmap, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.08, aspect=40)
    cbar.set_label('Patient Proportion', fontsize=fontsize)

    # ---- Custom Colorbar on Top of the First Subplot ---- #
    # Set a color map for the pseudotime < 0.5 (tab:red) and > 0.5 (tab:gold)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["red", "gold"])

    # Insert the custom color gradient for pseudotime within the first subplot
    ax_colorbar = axes[0].inset_axes([0, 1.1, 1, 0.05])  # Location of the colorbar (inside first subplot)
    ax_colorbar.imshow([np.linspace(0, 1, 256)], aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
    ax_colorbar.set_xticks([])
    ax_colorbar.set_xticklabels([])  # Remove x-axis ticks
    ax_colorbar.set_yticks([])  # Remove y-axis ticks
    ax_colorbar.set_xticklabels([])  # Remove x-axis ticks
    ax_colorbar.spines["top"].set_visible(False)
    ax_colorbar.spines["right"].set_visible(False)
    ax_colorbar.spines["left"].set_visible(False)
    ax_colorbar.spines["bottom"].set_visible(False)
    ax_colorbar.text(0.25, 1.05, "NSCLC", ha="center", va="bottom", fontsize=fontsize)
    ax_colorbar.text(0.75, 1.05, "SCLC-A", ha="center", va="bottom", fontsize=fontsize)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def read_gmt(gmt_file):
    gene_sets = {}
    with open(gmt_file, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            gene_set_name = parts[0]
            gene_set = parts[2:]
            gene_sets[gene_set_name] = gene_set
    return gene_sets

#----------------------#
# Load data #
#----------------------#
os.chdir("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/results_all_genes/preprocessing")
df = pd.read_csv('NSCLC_SCLC-A/cellrank.NSCLC_SCLC-A.csv', index_col=0)
counts = pd.read_csv('NSCLC_SCLC-A/counts.NSCLC_SCLC-A.csv', index_col=0)
all_genes = set(counts.columns[2:(counts.shape[1] - 9)])
clusters = ["NSCLC", "SCLC-A"]
downsample = 1500
df = df.groupby('subject').apply(lambda group: stratified_downsample(group, 'score', downsample)).reset_index(drop=True)

aligned_obj = aligner(df=df, 
                            cluster_ordering=clusters, 
                            subject_col='subject', 
                            score_col='score', 
                            cell_id_col='cell_id', 
                            cell_type_col='cell_type',
                            verbose=False)
aligned_obj.align()
df = aligned_obj.df
df_columns_selected = ['subject', 'aligned_score']
df_columns_selected = df_columns_selected + df.columns[11:].values.tolist()
adata = anndata.AnnData(df.iloc[:, 11:], obs=df.iloc[:, :11])
adata.write_h5ad('/scratch/chanj3/wangm10/NSCLC_SCLC-A.h5ad')

#----------------------#
# CellMarker Analysis #
#----------------------#
# get the cellmarker gene set genes
cellmarker_path = "/data1/chanj3/wangm10/gene_sets/CellMarker_2024.txt"
cellmarker_gene_set = read_gmt(cellmarker_path)
gene_sets_of_interest = ['Secretory Cell Lung Human', 'Basal Cell Lung Human', 'Cycling Basal Cell Trachea Mouse', 
                        'Neural Progenitor Cell Embryonic Prefrontal Cortex Human', 'Neuroendocrine Cell Trachea Mouse'] # 'Mesenchymal Stem Cell Undefined Human', 
names = ['Secretory', 'Basal', 'Cycling Basal', 'Neural Progenitor',  "Neuroendocrine"]
name_to_genes_dict = {
    name: cellmarker_gene_set.get(gene_set, [])  # Use .get() to handle missing keys gracefully
    for name, gene_set in zip(names, gene_sets_of_interest)
}
output_file = "/scratch/chanj3/wangm10/cellmarker_dict.json"
with open(output_file, 'w') as f:
    json.dump(name_to_genes_dict, f, indent=4)
# score genes


# Get unique patients
patients = adata.obs['subject'].unique()
n_patients = len(patients)
n_gene_sets = len(gene_sets_of_interest)

# Create subplots: one row per patient, one column per gene set
fig, axes = plt.subplots(nrows=n_patients, ncols=n_gene_sets, figsize=(3 * n_gene_sets, 3 * n_patients), sharex=True, sharey=True)

# If there's only one patient, make axes a 2D array for indexing consistency
if n_patients == 1:
    axes = np.array([axes])

# Iterate through patients
for row_idx, patient in enumerate(patients):
    patient_data = adata[adata.obs['subject'] == patient].copy()
    
    # Iterate through gene sets
    for col_idx, (gene_set, name) in enumerate(zip(gene_sets_of_interest, names)):
        ax = axes[row_idx, col_idx]

        # Score genes
        genes = cellmarker_gene_set[gene_set]
        sc.tl.score_genes(patient_data, genes, ctrl_as_ref=False, ctrl_size=len(genes), n_bins=50, score_name=name)

        # Scatter plot of pseudotime vs gene set score
        ax.scatter(patient_data.obs['aligned_score'], patient_data.obs[name], alpha=0.1, s=10, label="Data")

        # Fit and plot GAM smoothing
        gam = LinearGAM(s(0, n_splines=5)).fit(patient_data.obs['aligned_score'], patient_data.obs[name])
        x_smooth = np.linspace(patient_data.obs['aligned_score'].min(), patient_data.obs['aligned_score'].max(), 100)
        y_pred = gam.predict(x_smooth)
        ax.plot(x_smooth, y_pred, color='black', label='GAM Fit')

        # Titles and labels
        if row_idx == 0:
            ax.set_title(name)
        if col_idx == 0:
            ax.set_ylabel(f"Patient {patient}")

# Adjust layout and show the figure
plt.tight_layout()
plt.show()


cellmarker_short_labels = dict(zip(gene_sets_of_interest, names))
cellmarker_results = {}
# compute the enrichment for each gene set
for gene_set in gene_sets_of_interest:
    genes = cellmarker_gene_set[gene_set]
    enrichment = compute_gene_set_enrichment(df.loc[:,df_columns_selected], genes, gene_set, 
                                                           num_bins=10, p_value_threshold=0.05)
    cellmarker_results[gene_set] = enrichment
plot_cohens_d_ridge(cellmarker_results, short_labels=cellmarker_short_labels, figsize=(3, 5), fontsize=10, save_path="/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/cell_marker_ridge.png")

#----------------------#
# Pathway Analysis #
#----------------------#

# get the pathway gene set genes
with open("/data1/chanj3/morrill1/projects/HTA/data/biological_reference/spectra_gene_sets/Spectra.NE_NonNE.gene_sets.p", "rb") as infile:
    pathway_gene_set = pickle.load(infile)['global']
gene_sets_of_interest = ['all_TNF-via-NFkB_signaling', 'all_IL6-JAK-STAT3_signaling', 'all_PI3K-AKT-mTOR_signaling',  'all_MYC_targets', 'all_DNA-repair']
names = ['NFkB', "JAK-STAT3", "PI3K-AKT", 'MYC',  'DNA repair']
name_to_genes_dict = {
    name: pathway_gene_set.get(gene_set, [])  # Use .get() to handle missing keys gracefully
    for name, gene_set in zip(names, gene_sets_of_interest)
}
output_file = "/scratch/chanj3/wangm10/pathway_dict.json"
with open(output_file, 'w') as f:
    json.dump(name_to_genes_dict, f, indent=4)

pathway_short_labels = dict(zip(gene_sets_of_interest, names))
pathway_results = {}
for gene_set in gene_sets_of_interest:
    genes = pathway_gene_set[gene_set]
    enrichment = compute_gene_set_enrichment(df.loc[:,df_columns_selected], genes, gene_set, 
                                                           num_bins=10, p_value_threshold=0.05)
    pathway_results[gene_set] = enrichment
plot_cohens_d_ridge(pathway_results, short_labels=pathway_short_labels, figsize=(3, 5), fontsize=10, save_path="/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/pathway_ridge.png")



import pickle
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
import scanpy as sc

# Load gene sets
with open("/data1/chanj3/morrill1/projects/HTA/data/biological_reference/spectra_gene_sets/Spectra.NE_NonNE.gene_sets.p", "rb") as infile:
    pathway_gene_set = pickle.load(infile)['global']

# Define gene sets and corresponding names
gene_sets_of_interest = [
    'all_TNF-via-NFkB_signaling', 'all_IL6-JAK-STAT3_signaling', 
    'all_PI3K-AKT-mTOR_signaling',  'all_MYC_targets', 'all_DNA-repair'
]
names = ['NFkB', "JAK-STAT3", "PI3K-AKT", 'MYC', 'DNA repair']

# Get unique patients
patients = adata.obs['subject'].unique()
n_patients = len(patients)
n_gene_sets = len(gene_sets_of_interest)

# Create subplots: one row per patient, one column per gene set
fig, axes = plt.subplots(nrows=n_patients, ncols=n_gene_sets, figsize=(3 * n_gene_sets, 3 * n_patients), sharex=True, sharey=True)

# If there's only one patient, make axes a 2D array for indexing consistency
if n_patients == 1:
    axes = np.array([axes])

# Iterate through patients
for row_idx, patient in enumerate(patients):
    patient_data = adata[adata.obs['subject'] == patient].copy()
    
    # Iterate through gene sets
    for col_idx, (gene_set, name) in enumerate(zip(gene_sets_of_interest, names)):
        ax = axes[row_idx, col_idx]

        # Score genes
        genes = pathway_gene_set[gene_set]
        sc.tl.score_genes(patient_data, genes, ctrl_as_ref=False, ctrl_size=len(genes), n_bins=50, score_name=name)

        # Scatter plot of pseudotime vs gene set score
        ax.scatter(patient_data.obs['aligned_score'], patient_data.obs[name], alpha=0.1, s=10, label="Data")

        # Fit and plot GAM smoothing
        gam = LinearGAM(s(0, n_splines=5)).fit(patient_data.obs['aligned_score'], patient_data.obs[name])
        x_smooth = np.linspace(patient_data.obs['aligned_score'].min(), patient_data.obs['aligned_score'].max(), 100)
        y_pred = gam.predict(x_smooth)
        ax.plot(x_smooth, y_pred, color='black', label='GAM Fit')

        # Titles and labels
        if row_idx == 0:
            ax.set_title(name)
        if col_idx == 0:
            ax.set_ylabel(f"Patient {patient}")

# Adjust layout and show the figure
plt.tight_layout()
plt.show()