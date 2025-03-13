import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LinearGAM, s, f
from scDeBussy import aligner, gam_smooth_expression
from scDeBussy.pp import create_cellrank_probability_df
from scDeBussy.pl import plot_sigmoid_fits, plot_summary_curve
import numpy as np
from matplotlib.patches import Patch
import re

path = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/"
lam = 2
n_splines = 4
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
df_ascl1_neurod1, _, _ = gam_smooth_expression(df, ['ASCL1', 'NEUROD1'], score_col='score', n_splines=n_splines, lam=lam)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), squeeze=False)
axes = axes.flatten()  # Flatten axes array for easier indexing

for i, sample in enumerate(samples):
    ax = axes[i]
    print(sample)
    df_sample = df[df['subject'] == sample]
    df_smoothed = df_ascl1_neurod1[df_ascl1_neurod1['subject'] == sample]

    if sample == 'RU1303':
        df_smoothed = df_smoothed[~np.isin(df_smoothed['gene'], ['NEUROD1'])]
    elif re.search('RU1518|RU1646', sample):
        df_smoothed = df_smoothed[~np.isin(df_smoothed['gene'], ['ASCL1'])]
    if i == 0:
        sns.lineplot(x='score', y='smoothed', hue='gene', data=df_smoothed, 
                    ax=ax, palette={"ASCL1": "tab:red", "NEUROD1": "tab:cyan"})
        handles, labels = ax.get_legend_handles_labels()  # Retrieve legend info
        ax.legend_.remove()
    else: 
        sns.lineplot(x='score', y='smoothed', hue='gene', data=df_smoothed, 
                    ax=ax, palette={"ASCL1": "tab:red", "NEUROD1": "tab:cyan"}, legend=False)
        
    # Add title and labels to the subplot
    expression_max = np.nanmax(df_smoothed.smoothed)
    expression_min = np.nanmin(df_smoothed.smoothed)
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
plt.savefig(f'figures/neurod1_ascl1_lam_{lam}_splines_{n_splines}.png', dpi=300, bbox_inches='tight')
plt.show()
