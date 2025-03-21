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
from scDeBussy.pp import stratified_downsample
from scDeBussy import aligner

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
    results = {}
    
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
    gam_enrichment = LinearGAM(s(0, n_splines=15)).fit(df[['pseudotime']], df['enrichment'])
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
    """
    Plots Cohen's D with error bars and GAM fit for multiple gene sets as a stacked ridge plot,
    with a heatmap track below each ridge plot representing the patient_proportion values.

    Parameters:
        results (dict): Dictionary mapping gene set names to their enrichment results.
        figsize (tuple): Size of the figure.
        fontsize (int): Font size for labels.
        short_labels (dict, optional): Dictionary mapping full gene set names to shortened labels.
        color (str, optional): Color for all plots (default: "royalblue").
        save_path (str, optional): Path to save the figure. If None, the figure is not saved.
    """
    num_gene_sets = len(results)
    fig, axes = plt.subplots(num_gene_sets, 1, figsize=figsize, sharex=True, constrained_layout=True, 
                             gridspec_kw={'height_ratios': [4] * num_gene_sets})

    if num_gene_sets == 1:  # Ensure axes is always iterable
        axes = [axes]

    # Plot ridge plots and heatmap tracks for each gene set
    for ax, (gene_set, df) in zip(axes, results.items()):
        x_vals = df['pseudotime']
        y_vals = df['enrichment']
        y_errs = df['standard_error']
        patient_proportion = df['patient_proportion']

        # Plot Cohen's D with subtle error bars on the ridge plot
        ax.errorbar(
            x_vals, y_vals, yerr=y_errs, fmt='o', color=color, alpha=0.5, 
            elinewidth=0.8, capsize=2, capthick=0.8
        )

        # GAM fit for the enrichment scores
        gam = LinearGAM(s(0, n_splines=10)).fit(x_vals, y_vals)
        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_pred = gam.predict(x_smooth)
        ax.plot(x_smooth, y_pred, color=color, lw=2)

        # Calculate y-axis limits for the ridge plot, taking error bars into account
        y_min = min((y_vals - y_errs).min(), y_pred.min())  # Minimum of enrichment scores (with error bars) and GAM predictions
        y_max = max((y_vals + y_errs).max(), y_pred.max())  # Maximum of enrichment scores (with error bars) and GAM predictions

        # Add some padding to the y-axis limits
        padding = (y_max - y_min) * 0.1  # 10% padding
        y_min -= padding
        y_max += padding

        # Set y-axis limits for the ridge plot
        ax.set_ylim(y_min, y_max)

        # Remove axis lines, ticks, and labels for the ridge plot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)  # Remove ticks

        # Use shortened label if provided, else use full gene set name
        y_label = short_labels.get(gene_set, gene_set) if short_labels else gene_set
        ax.set_ylabel(y_label, fontsize=fontsize, rotation=0, labelpad=50)
        ax.set_xticklabels([])  # Remove x-axis labels
        ax.set_yticklabels([])  # Remove y-axis labels

        # Create a heatmap track below the ridge plot
        ax_heatmap = ax.inset_axes([0, -0.2, 1, 0.1])  # Place heatmap track below the ridge plot
        X = np.linspace(x_vals.min(), x_vals.max(), len(x_vals) + 1)  # X edges
        Y = np.array([0, 1])  # Y edges (single row heatmap)
        Z = patient_proportion.values.reshape(1, -1)  # Reshape patient_proportion for heatmap

        # Plot the heatmap
        heatmap = ax_heatmap.pcolormesh(X, Y, Z, cmap='Blues', shading='flat')#, vmin=0, vmax=1)

        # Remove axis lines, ticks, and labels for the heatmap track
        ax_heatmap.spines["top"].set_visible(False)
        ax_heatmap.spines["right"].set_visible(False)
        ax_heatmap.spines["left"].set_visible(False)
        ax_heatmap.spines["bottom"].set_visible(False)
        ax_heatmap.tick_params(axis="both", which="both", length=0)  # Remove ticks
        ax_heatmap.set_yticklabels([])  # Remove y-axis labels
        ax_heatmap.set_xticklabels([])  # Remove x-axis labels

    # Add a single colorbar for all heatmap tracks
    #cbar = plt.colorbar(heatmap, ax=axes, orientation='horizontal', pad=0.2)
    #cbar.set_label('Patient Proportion', fontsize=fontsize)

    # Set x-axis label for the last subplot
    axes[-1].set_xlabel("Pseudotime", fontsize=fontsize)

    # Save the figure if save_path is provided
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
# get the cellmarker gene set genes
cellmarker_path = "/data1/chanj3/wangm10/gene_sets/CellMarker_2024.txt"
cellmarker_gene_set = read_gmt(cellmarker_path)
gene_sets_of_interest = ['Secretory Cell Lung Human', 'Basal Cell Lung Human', 'Cycling Basal Cell Trachea Mouse', 
                        'Neural Progenitor Cell Embryonic Prefrontal Cortex Human', 'Neuroendocrine Cell Trachea Mouse'] # 'Mesenchymal Stem Cell Undefined Human', 
names = ['Secretory', 'Basal', 'Cycling Basal', 'Neural Progenitor',  "Neuroendocrine"] 
cellmarker_short_labels = dict(zip(gene_sets_of_interest, names))
cellmarker_results = {}
# compute the enrichment for each gene set
for gene_set in gene_sets_of_interest:
    genes = cellmarker_gene_set[gene_set]
    enrichment = compute_gene_set_enrichment(df.loc[:,df_columns_selected], genes, gene_set, 
                                                           num_bins=10, p_value_threshold=0.05)
    cellmarker_results[gene_set] = enrichment
plot_cohens_d_ridge(cellmarker_results, short_labels=cellmarker_short_labels, figsize=(5, 4), save_path="/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/cell_marker_ridge.png")

# get the pathway gene set genes
with open("/data1/chanj3/morrill1/projects/HTA/data/biological_reference/spectra_gene_sets/Spectra.NE_NonNE.gene_sets.p", "rb") as infile:
    pathway_gene_set = pickle.load(infile)['global']
gene_sets_of_interest = ['all_TNF-via-NFkB_signaling', 'all_IL6-JAK-STAT3_signaling', 'all_PI3K-AKT-mTOR_signaling',  'all_MYC_targets', 'all_DNA-repair']
names = ['NFkB', "JAK-STAT3", "PI3K-AKT", 'MYC',  'DNA repair']
pathway_short_labels = dict(zip(gene_sets_of_interest, names))
pathway_results = {}
for gene_set in gene_sets_of_interest:
    genes = pathway_gene_set[gene_set]
    enrichment = compute_gene_set_enrichment(df.loc[:,df_columns_selected], genes, gene_set, 
                                                           num_bins=10, p_value_threshold=0.05)
    pathway_results[gene_set] = enrichment
plot_cohens_d_ridge(pathway_results, short_labels=pathway_short_labels, figsize=(5, 4), save_path="/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/pathway_ridge.png")
