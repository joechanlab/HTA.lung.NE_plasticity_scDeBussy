import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LinearGAM, s, f
from scdebussy.tl import aligner, gam_smooth_expression
from scdebussy.pp import create_cellrank_probability_df
from scdebussy.pl import plot_sigmoid_fits, plot_summary_curve
import numpy as np
from matplotlib.patches import Patch
import re

path = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/"

n_cols = 4
downsample = 5000
cellrank_obsms = 'palantir_pseudotime_slalom'
clusters = ['NSCLC', 'SCLC-A', 'SCLC-N']
samples = ['RU1083', 'RU1444', 'RU581', 'RU831', 'RU942', 'RU263', 'RU151', 'RU1303','RU1518', 'RU1646']
n_samples = len(samples)
n_rows = (n_samples + n_cols - 1) // n_cols
color_map = {"NSCLC": "gold", "SCLC-A": "tab:red", "SCLC-N": "tab:cyan"}
data = """
RU151,NSCLC_1,SCLC-N_2,SCLC-A,NonNE SCLC_1
RU263,NSCLC_2,SCLC-A_3,SCLC-N_1,SCLC-N_3
RU581,NSCLC,SCLC-A_4,SCLC-A_5,SCLC-N
RU831,NSCLC,SCLC-A_5,SCLC-A_6,SCLC-A_3,SCLC-N
RU942,NSCLC_1,NSCLC_3,SCLC-A_3,SCLC-N_2,SCLC-N_1
RU1042,SCLC-A,SCLC-N_5,SCLC-N_7
RU1083,NSCLC,SCLC-A_3,SCLC-A_1,SCLC-N_4
RU1181,SCLC-A,SCLC-N_6
RU1215,SCLC-A,SCLC-N_2,SCLC-N_3
RU1250,RB1-proficient NSCLC_4,RB1-deficient NSCLC_1,RB1-deficient NSCLC_4,RB1-deficient NSCLC_2
RU1293,NSCLC,SCLC-N_6,SCLC-N_7
RU1303,NSCLC_4,SCLC-A_1,SCLC-A_2
RU1304,SCLC-A,SCLC-N_7,SCLC-N_5
RU1444,NSCLC,SCLC-A_1,SCLC-N_3,UBA52+ SCLC_2
RU1518,NSCLC_1,SCLC-N_8
RU1676,SCLC-A,SCLC-N_7
RU1646,NSCLC,SCLC-N_3
"""
lines = data.strip().split('\n')
result_dict = {}
for line in lines:
    parts = line.split(',')
    key = parts[0]
    values = parts[1:]
    result_dict[key] = values

h5ad_files = [os.path.join(path, x + ".no_cc.hvg_2000.090124.h5ad") if x != 'RU263' else os.path.join(path, x + ".no_cc.hvg_2000.curated.120624.h5ad") for x in samples]
combined_adata, df = create_cellrank_probability_df(h5ad_files, 'cell_type_final2', samples, clusters, 
                                                    cellrank_obsms, 
                                                    cellrank_cols_dict=result_dict, downsample=downsample, need_all_clusters=False)
df['score'] = df.groupby('subject')['score'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
lam = 3
n_splines = 5
df_samples = []
for gene in ['ASCL1', 'NEUROD1']:
    for subject in df.subject.unique():
        df_sample = df[df['subject'] == subject][['score', gene]]
        df_sample.columns = ['score', 'expression']
        df_sample['expression'] = df_sample['expression'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else pd.Series(0, index=x.index)
        )
        gam = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(df_sample.score, df_sample.expression)
        df_sample['smoothed'] = gam.predict(df_sample.score)
        df_sample['gene'] = gene
        df_sample['subject'] = subject
        df_samples.append(df_sample)
df_ascl1_neurod1 = pd.concat(df_samples)
df_ascl1_neurod1.reset_index(drop=True, inplace=True)

to_plot='smoothed'
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), squeeze=False)
axes = axes.flatten()  # Flatten axes array for easier indexing
for i, sample in enumerate(samples):
    ax = axes[i]
    print(sample)
    df_sample = df[df['subject'] == sample]
    df_smoothed = df_ascl1_neurod1[df_ascl1_neurod1['subject'] == sample]
    #df_smoothed = df_smoothed[df_ascl1_neurod1['gene'] == 'ASCL1']

    # if sample == 'RU1303':
    #     df_smoothed = df_smoothed[~np.isin(df_smoothed['gene'], ['NEUROD1'])]
    # elif re.search('RU1518|RU1646', sample):
    #     df_smoothed = df_smoothed[~np.isin(df_smoothed['gene'], ['ASCL1'])]
    if i == 0:
        # sns.scatterplot(x='score', y=to_plot, hue='gene', data=df_smoothed, 
        #             ax=ax, palette={"ASCL1": "tab:red", "NEUROD1": "tab:cyan"}, alpha=0.5)
        sns.lineplot(x='score', y=to_plot, hue='gene', data=df_smoothed, 
                    ax=ax, palette={"ASCL1": "tab:red", "NEUROD1": "tab:cyan"}, alpha=0.5)
                                
        handles, labels = ax.get_legend_handles_labels()  # Retrieve legend info
        ax.legend_.remove()
    else: 
        # sns.scatterplot(x='score', y=to_plot, hue='gene', data=df_smoothed, 
        #             ax=ax, palette={"ASCL1": "tab:red", "NEUROD1": "tab:cyan"}, legend=False, alpha=0.5)
        sns.lineplot(x='score', y=to_plot, hue='gene', data=df_smoothed, 
                    ax=ax, palette={"ASCL1": "tab:red", "NEUROD1": "tab:cyan"}, legend=False, alpha=0.5)
        
    # Add title and labels to the subplot
    expression_max = np.nanmax(df_ascl1_neurod1[to_plot])
    expression_min = np.nanmin(df_ascl1_neurod1[to_plot])
    bar_space = (expression_max - expression_min) * 0.1
    y_max = expression_max + 0.1
    y_min = expression_min - bar_space * 2
    x_min = -0.1
    x_max = 1.1
    cell_types = pd.Categorical(df_sample['cell_type'].unique(), ordered=True, categories=color_map.keys())
    cell_types = cell_types.sort_values()
    n_cell_types = 3

    for k, cell_type in enumerate(cell_types):
        subset = df_sample[df_sample['cell_type'] == cell_type]
        ax.bar(subset['score'], height=bar_space/n_cell_types, bottom=y_min + (n_cell_types-k)/n_cell_types * bar_space, width=0.01, 
            color=color_map[cell_type], edgecolor=None, align='edge', alpha=0.2)
    
    print(y_min, y_max, x_min, x_max)
    ax.set_ylim([y_min, y_max])
    ax.set_xlim([x_min, x_max])
    ax.set_title(f'{sample}', fontsize=20)
    ax.set_xlabel('Pseudotime', fontsize=20)
    ax.set_ylabel('Expression', fontsize=20)

    if i % n_cols != 0:
        ax.set_ylabel('')
    if i < (n_rows - 1) * n_cols:
        ax.set_xlabel('')
    
# Hide any unused subplots if the number of samples is less than rows * cols
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

handles_gene, labels_gene = axes[0].get_legend_handles_labels()  # Get legend info from the first subplot
for legend_handle in handles_gene:
    legend_handle.set_alpha(1)

# Add legend for the barplot marks
handles_bar = [Patch(color=color_map[cell_type], label=cell_type) for cell_type in color_map.keys()]
fig.legend(handles_gene + handles_bar, labels_gene + list(color_map.keys()), loc='lower right', title="", bbox_to_anchor=(0.8, 0.1), frameon=False, fontsize=20)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(f'neurod1_ascl1_lam_{lam}_splines_{n_splines}.png', dpi=300, bbox_inches='tight')
plt.show()

def classify_trend(smoothed_values, pseudotime, poly_order=3, noise_threshold=1e-2, flat_range_threshold=0.1, peak_drop_threshold=0.05):
    """
    Classifies the trend of gene expression along pseudotime into:
    - "increasing": Majority of trend is increasing.
    - "flat": Little to no significant change.
    - "peak": Has a clear peak and then declines.

    Parameters:
    - smoothed_values (array-like): Smoothed expression values.
    - pseudotime (array-like): Corresponding pseudotime values.
    - poly_order (int): Order of the polynomial for fitting.
    - noise_threshold (float): Threshold for ignoring small variations.
    - flat_range_threshold (float): Maximum slope range to classify as "flat".
    - peak_drop_threshold (float): Minimum drop after peak for classification as "peak".

    Returns:
    - str: Trend type ("increasing", "flat", or "peak").
    """

    # Fit a polynomial to the smoothed values
    coeffs = np.polyfit(pseudotime, smoothed_values, poly_order)
    poly_values = np.polyval(coeffs, pseudotime)
    poly_derivative = np.polyder(coeffs)  # First derivative
    derivative_values = np.polyval(poly_derivative, pseudotime)

    # Check for flat trend
    if np.max(derivative_values) - np.min(derivative_values) < flat_range_threshold:
        return "flat"

    # Identify peaks: Where derivative changes from positive to negative
    peak_index = np.argmax(poly_values)
    if peak_index > 0 and peak_index < len(pseudotime) - 1:
        peak_value = poly_values[peak_index]
        end_value = poly_values[-1]
        if (peak_value - end_value) > peak_drop_threshold:
            return "peak"

    # Adjust "increasing" threshold to be stricter
    increasing_fraction = np.mean(derivative_values > noise_threshold)
    if increasing_fraction > 0.75:  # Was 0.6 before, now stricter
        return "increasing"

    return "peak"  # Default to peak if not classified as increasing or flat

# Compute recurrence scores
recurrence_scores_improved = []
trend_types = ["increasing", "flat", "peak"]

for gene in df_ascl1_neurod1['gene'].unique():
    gene_data = df_ascl1_neurod1[df_ascl1_neurod1['gene'] == gene]
    trend_counts = {t: 0 for t in trend_types}

    for subject in gene_data['subject'].unique():
        subject_data = gene_data[gene_data['subject'] == subject]
        trend_type = classify_trend(subject_data['smoothed'].values, subject_data['score'].values)
        trend_counts[trend_type] += 1
        print(f"Gene: {gene}, Subject: {subject}, Trend: {trend_type}")

    total_subjects = len(gene_data['subject'].unique())
    recurrence_scores_improved.append({
        "gene": gene,
        **{t: trend_counts[t] / total_subjects for t in trend_types}
    })

# Create a DataFrame with improved recurrence scores
recurrence_df_improved = pd.DataFrame(recurrence_scores_improved)
print(recurrence_df_improved)
