import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pycirclize import Circos
import tempfile
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

tumor_color_map = {
    "NSCLC": "gold",      # Yellow
    "SCLC-A": "tab:red",      # Red
    "SCLC-N": "tab:cyan",      # Cyan
    "SCLC-P": "tab:blue",      # Blue
    "NonNE\nSCLC": "tab:purple",  # Purple
    "pDC": "#004d00",
    "PMN": "#339933",
    "Mφ/Mono": "#66cc66",
    "Mφ/Mono\nCD14": "#99cc99",
    "Mφ/Mono\nCD11c": "#336633",
    "Mast cell": "#003300"
}

sample_groups = {
    "De novo SCLC and ADC": [
        "RU1311", "RU1124", "RU1108", "RU1065",
        "RU871", "RU1322", "RU1144", "RU1066",
        "RU1195", "RU779", "RU1152", "RU426",
        "RU1231", "RU222", "RU1215", "RU1145",
        "RU1360", "RU1287", 
    ],
    "ADC → SCLC": [
        "RU1068", "RU1676", "RU263", "RU942",
        "RU325", "RU1304", "RU1444", "RU151",
        "RU1181", "RU1042", "RU581", "RU1303",
        "RU1305", "RU1250", "RU831", "RU1646",
        "RU226", "RU773", "RU1083", "RU1518",
        "RU1414", "RU1405", "RU1293"
    ]
}

# Create reversed dictionary where sample IDs are keys and their groups are values
sample_to_group = {
    sample_id: group
    for group, samples in sample_groups.items()
    for sample_id in samples
}

def read_tab_delimited_files(file_list):
    dfs = []
    for file in file_list:
        df = pd.read_csv(file, sep='\t')
        dfs.append(df)
    return dfs

def extract_ru_subdir(file_paths):
    ru_dirs = []
    for path in file_paths:
        parts = os.path.normpath(path).split(os.sep)
        ru_subdirs = [part for part in parts if part.startswith('RU')]
        ru_dirs.extend(ru_subdirs)
    return ru_dirs

def group_nsclc(x):
    if x in ['Basal-like_NSCLC', 'Mucinous_NSCLC']:
        return 'NSCLC'
    elif x == "NonNE SCLC":
        return "NSCLC"
    elif x == "Mφ/_Mono":
        return "Mφ/Mono"
    elif x == "Mφ/Mono_CD11c":
        return "Mφ/Mono\nCD11c"
    elif x == "Mφ/Mono_CD14":
        return "Mφ/Mono\nCD14"
    elif x == "Mφ/Mono_CCL":
        return "Mφ/Mono\nCCL"
    elif x == "CD4_EM/Effector":
        return "CD4_EM\nEffector"
    elif x == "CD4_naive/CM":
        return "CD4_naive\nCM"
    elif x == "CD8_TRM/EM":
        return "CD8_TRM\nEM"
    elif x == "Mast_cell":
        return "Mast cell"
    else:
        return x

def get_interaction_type(sender, receiver):
    if sender in tumor_cells and receiver in tumor_cells:
        return "within_tumor"
    elif ((sender in myeloid_cells and receiver in tumor_cells) or
          (sender in tumor_cells and receiver in myeloid_cells)):
        return "myeloid_tumor"
    elif ((sender in lymphoid_cells and receiver in tumor_cells) or
          (sender in tumor_cells and receiver in lymphoid_cells)):
        return "lymphoid_tumor"
    else:
        return "other"

#----------------------------------
base_dir = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/out.individual.120122/"
relevant_file_pattern = base_dir + "/*/cellphonedb.output/degs_analysis_relevant_interactions_042725.txt"
file_list =  glob.glob(relevant_file_pattern)
label_file = "/data1/chanj3/morrill1/projects/HTA/results/figures-tables/factor_analysis/Spectra_HVGs/v7_KEGGAKTgenesetwPHOX2BwPOU3F2wMYCLwMYCN/SCLCNSCLC/light_factor_rescue/subsetpatientsTRUE/basal/comparison_with_other_basal_labels/SpectraScores_basal_cluster_summary.csv"
label_df = pd.read_csv(label_file)
condition_labels = pd.DataFrame(dict(
    patient=extract_ru_subdir(file_list),
    condition=[sample_to_group.get(x, "unknown") for x in extract_ru_subdir(file_list)]
))

#----------------------------------
# compare the 15 subjects containing basal nonNE to rest of the 16 subjects
subject_list = extract_ru_subdir(file_list)
subject_list = list(set(subject_list) & set(condition_labels.patient))
condition_labels.set_index('patient', inplace=True)
condition_labels = condition_labels.loc[subject_list,:] # order the list as in the subject list
file_list = [base_dir + x + "/cellphonedb.output/degs_analysis_relevant_interactions_042725.txt" for x in subject_list]
dfs = read_tab_delimited_files(file_list)
condition_list = condition_labels.condition.values

#----------------------------------
# Prepare for data for chord plot
long_df_list = []
for i, df in enumerate(dfs):
    if condition_list[i] != "unknown":
        print(f"Processing {subject_list[i]}: {condition_list[i]}")
        interacting_pairs = df['interacting_pair']
        df = df.iloc[:,13:]
        df.loc[:,'interacting_pair'] = interacting_pairs
        df.loc[:,'subject'] = subject_list[i]
        df.loc[:,'condition'] = condition_list[i]
        df = df.reset_index(drop=True)
        
        long_df = pd.melt(df, id_vars=['interacting_pair', 'subject', 'condition'], var_name='pair', value_name='value')
        if long_df.shape[0] > 0:
            # Split the 'pair' column into 'sender' and 'receiver'
            long_df[['sender', 'receiver']] = long_df['pair'].str.split('|', expand=True)
            # Optionally, drop the original 'pair' column if not needed
            long_df = long_df.drop(columns=['pair'])
            long_df = long_df[long_df['value'] != 0]
            long_df['sender'] = long_df['sender'].str.split(r'[.]').str[0]
            long_df['receiver'] = long_df['receiver'].str.split(r'[.]').str[0]
            long_df['sender'] = long_df['sender'].apply(group_nsclc)
            long_df['receiver'] = long_df['receiver'].apply(group_nsclc)
            long_df_list.append(long_df)
relevance_long_df = pd.concat(long_df_list, ignore_index=True)
relevance_long_df = relevance_long_df.loc[relevance_long_df.groupby(['subject', 'condition', 'sender', 'receiver', 'interacting_pair'])['value'].idxmax()]
relevance_long_df = relevance_long_df.loc[~(((relevance_long_df.sender == 'NSCLC') & (relevance_long_df.receiver.isin(['SCLC-A', 'SCLC-N']))) | ((relevance_long_df.sender.isin(['SCLC-A', 'SCLC-N'])) & (relevance_long_df.receiver == 'NSCLC')) & (relevance_long_df.condition == 'de_novo_SCLC_LUAD'))]
relevance_long_df.loc[:,'condition'] = pd.Categorical(relevance_long_df['condition'], categories=["De novo SCLC and ADC", "ADC → SCLC"], ordered=True)
# Assuming your DataFrame is named df
pivot = relevance_long_df.pivot_table(
    index=['subject', 'condition', 'sender'],
    columns='receiver',
    values='value',
    aggfunc='sum',
    fill_value=0
)
# Reset index to make subject, condition, sender columns
pivot_reset = pivot.reset_index().rename_axis(None, axis=1)

# Group by condition and sender, then average across subjects
median = pivot_reset.drop(columns=['subject']).groupby(['condition', 'sender']).median()

# Find all unique senders across all conditions
all_senders = median.index.get_level_values('sender').unique().sort_values()

# For each condition, reindex the sender rows to include all senders, filling missing ones with zeros
condition_matrices = {}
for condition in median.index.get_level_values('condition').unique():
    cond_df = median.loc[condition]
    cond_df_reindexed = cond_df.reindex(all_senders, fill_value=0)
    condition_matrices[condition] = cond_df_reindexed

conditions = list(condition_matrices.keys())
matrices = [condition_matrices[cond] for cond in conditions]

fig, axes = plt.subplots(1, len(matrices), figsize=(18, 10), sharey=True)

max_val = max([m.values.max() for m in matrices])

for ax, matrix, cond in zip(axes, matrices, conditions):
    sns.heatmap(matrix, ax=ax, cmap='viridis', cbar=True, vmin=0, vmax=max_val, 
                xticklabels=True, yticklabels=True)
    ax.set_title(cond, fontsize=16)
    ax.set_xlabel('Receiver', fontsize=14)
    ax.set_ylabel('Sender', fontsize=14)
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.show()

#----------------------------------
# Separate the interaction as within tumors, myeloid-tumor, lymphoid tumor
tumor_cells = {"NSCLC", "SCLC-A", "SCLC-N"}
myeloid_cells = {"Mφ/Mono", "Mφ/Mono\nCD14", "Mφ/Mono\nCD11c", "Mφ/Mono\nCCL", "PMN", "Mast cell", "pDC"}
lymphoid_cells = {"CD4_EM\nEffector", "CD4_naive\nCM", "CD4_TRM", "CD8_TRM\nEM", "T_reg", "NK", "ILC", "B_memory"}
stromal_cells = {"Fibroblast", "Endothelial", "Basal", "Ciliated"}
relevance_long_df['interaction_type'] = relevance_long_df.apply(
    lambda row: get_interaction_type(row['sender'], row['receiver']), axis=1
)
interaction_types = ["within_tumor", "myeloid_tumor", "lymphoid_tumor"]

for interaction_type in interaction_types:
    filtered_df = relevance_long_df[relevance_long_df['interaction_type'] == interaction_type]
    if filtered_df.empty:
        print(f"No interactions for {interaction_type}")
        continue

    # Pivot as before
    pivot = filtered_df.pivot_table(
        index=['subject', 'condition', 'sender'],
        columns='receiver',
        values='value',
        aggfunc='sum',
        fill_value=0
    ).reset_index().rename_axis(None, axis=1)
    median = pivot.drop(columns=['subject']).groupby(['condition', 'sender']).median()
    all_senders = median.index.get_level_values('sender').unique().sort_values()

    condition_matrices = {}
    for condition in ["De novo SCLC and ADC", "ADC → SCLC"]:
        cond_df = median.loc[condition]
        cond_df_reindexed = cond_df.reindex(all_senders, fill_value=0)
        condition_matrices[condition] = cond_df_reindexed

    # Chord plot for each condition
    all_nodes = set()
    for matrix_df in condition_matrices.values():
        all_nodes.update(matrix_df.index)
        all_nodes.update(matrix_df.columns)
    all_nodes = sorted(all_nodes)

    cmap = cm.get_cmap('tab20', len(all_nodes))
    node2color = {}
    for node in all_nodes:
        if node in tumor_color_map:
            node2color[node] = tumor_color_map[node]
        else:
            # Use your previous color assignment for non-tumor nodes
            node2color[node] = mcolors.to_hex(cmap(all_nodes.index(node)))

    images = []
    labels = []
    for condition, matrix_df in condition_matrices.items():
        matrix_df = matrix_df.fillna(0)
        matrix_df = matrix_df.reindex(index=all_nodes, columns=all_nodes, fill_value=0)
        matrix_df = matrix_df.loc[(matrix_df != 0).any(axis=1), (matrix_df != 0).any(axis=0)]
        filtered_order = [node for node in all_nodes if node in matrix_df.index and node in matrix_df.columns]
        if matrix_df.empty or len(filtered_order) == 0:
            print(f"Skipping {condition} for {interaction_type}: matrix is empty after filtering.")
            continue

        circos = Circos.chord_diagram(
            matrix_df,
            space=19,
            cmap=node2color,
            label_kws=dict(size=25),
            link_kws=dict(ec="black", lw=0.5, direction=1),
            #order=filtered_order
        )
        tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        circos.savefig(tmpfile.name)
        images.append(Image.open(tmpfile.name))
        labels.append(condition)

    # Plot images side by side
    if images:
        fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
        if len(images) == 1:
            axes = [axes]
        for ax, img, label in zip(axes, images, labels):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{label}", fontsize=21)
        plt.tight_layout()
        fig.savefig(f"{interaction_type}_chord_plot.png", dpi=300, bbox_inches='tight')
        plt.show()


#----------------------------------
# dot plot for Joe's basal-focused grant proposal
def group_nsclc(x):
    if x in ['Basal-like_NSCLC', 'Mucinous_NSCLC']:
        return 'NSCLC'
    elif x == "Mφ/_Mono":
        return "Mφ/Mono"
    elif x == "Mφ/Mono_CD14":
        return "Mφ/Mono CD14"
    elif x == "Mφ/Mono_CD11c":
        return "Mφ/Mono CD11c"
    elif x == "Mast_cell":
        return "Mast cell"
    elif x == "CD4_naive/CM":
        return "CD4 naive/CM"
    elif x == "CD4_EM/Effector":
        return "CD4 EM/Effector"
    elif x == "CD4_TRM":
        return "CD4 TRM"
    elif x == "CD8_TRM/EM":
        return "CD8 TRM/EM"
    elif x == "T_reg":
        return "T reg"
    elif x == "B_memory":
        return "B memory"
    else:
        return x
means_list = [base_dir + x + "/cellphonedb.output/degs_analysis_significant_means_042725.txt" for x in subject_list]
means_dfs = read_tab_delimited_files(means_list)

# Load the means files
long_df_list = []
for i, df in enumerate(means_dfs):
    if condition_list[i] != "unknown":
        filtered_df = df[df['interacting_pair'].str.contains('IL6_|IL6ST|LIF|OSM', case=False, na=False)]
        filtered_df.index = filtered_df['interacting_pair']
        subset_cols = [col for col in filtered_df.columns if len(col.split('|')) > 1 and 'SCLC' in col.split('|')[1]]
        subset_df = filtered_df[subset_cols]
        subset_df = subset_df.fillna(0)
        subset_df = subset_df.loc[:, (subset_df != 0).any(axis=0)]
        subset_df = subset_df.reset_index()
        # Melt the DataFrame to long format
        long_df = pd.melt(subset_df, id_vars=subset_df.columns[0], var_name='pair', value_name='value')
        if long_df.shape[0] > 0:
            # Split the 'pair' column into 'sender' and 'receiver'
            long_df[['sender', 'receiver']] = long_df['pair'].str.split('|', expand=True)
            # Optionally, drop the original 'pair' column if not needed
            long_df = long_df.drop(columns=['pair'])
            # Reorder columns if desired
            long_df = long_df[[subset_df.columns[0], 'sender', 'receiver', 'value']]
            long_df = long_df[long_df['value'] != 0]
            long_df['sender'] = long_df['sender'].str.split(r'[.]').str[0]
            long_df['receiver'] = long_df['receiver'].str.split(r'[.]').str[0]
            long_df['sender'] = long_df['sender'].apply(group_nsclc)
            long_df['receiver'] = long_df['receiver'].apply(group_nsclc)
            long_df.loc[:,'subject'] = subject_list[i]
            long_df.loc[:,'condition'] = condition_list[i]
            long_df_list.append(long_df)
long_df = pd.concat(long_df_list, ignore_index=True)
long_df = long_df.loc[long_df.groupby(['subject', 'condition', 'sender', 'receiver', 'interacting_pair'])['value'].idxmax()]
long_df = long_df.loc[~(((long_df.sender == 'NSCLC') & (long_df.receiver.isin(['SCLC-A', 'SCLC-N']))) | ((long_df.sender.isin(['SCLC-A', 'SCLC-N'])) & (long_df.receiver == 'NSCLC')) & (long_df.condition == 'de_novo_SCLC_LUAD'))]

tumor_cells = {"NSCLC", "SCLC-A", "SCLC-N"}
myeloid_cells = {"Mφ/Mono", "Mφ/Mono CD14", "Mφ/Mono CD11c", "Mφ/Mono CCL", "PMN", "Mast cell", "pDC"}
lymphoid_cells = {"CD4 EM/Effector", "CD4 naive/CM", "CD4 TRM", "CD8 TRM/EM", "T reg", "NK", "ILC", "B memory"}
stromal_cells = {"Fibroblast", "Endothelial", "Basal", "Ciliated"}
long_df['interaction_type'] = long_df.apply(
    lambda row: get_interaction_type(row['sender'], row['receiver']), axis=1
)
long_df = long_df[long_df.interaction_type == 'myeloid_tumor']
pivot = long_df.pivot_table(
    index=['subject', 'condition', 'sender', 'interacting_pair'],
    columns='receiver',
    values='value',
    fill_value=0
)
# Reset index to make subject, condition, sender columns
pivot_reset = pivot.reset_index().rename_axis(None, axis=1)
pivot_reset = pd.melt(pivot_reset, id_vars=['subject', 'condition', 'sender', 'interacting_pair'], var_name='receiver', value_name='value')
pivot_reset['sender_receiver_pair'] = pivot_reset['sender'] + '→' + pivot_reset['receiver']

# Group by condition and sender, then average across subjects
median = pivot_reset.drop(columns=['subject']).groupby(['condition', 'sender', 'receiver', 'sender_receiver_pair', 'interacting_pair']).median().reset_index()

# Set up FacetGrid
receiver_order = sorted(median['receiver'].unique())
median['receiver'] = pd.Categorical(median['receiver'], categories=receiver_order, ordered=True)
median = median.sort_values('receiver')

# Set condition order
median['condition'] = pd.Categorical(median['condition'], 
                                   categories=["De novo SCLC and ADC", "ADC → SCLC"],
                                   ordered=True)

# Optionally, create a categorical for sender_receiver_pair that follows receiver order
median['sender_receiver_pair'] = median['sender'] + '→' + median['receiver'].astype(str)
sender_receiver_order = median.sort_values('receiver')['sender_receiver_pair'].unique()
median['sender_receiver_pair'] = pd.Categorical(median['sender_receiver_pair'], categories=sender_receiver_order, ordered=True)
median = median[median.value != 0]
median['interacting_pair'] = median['interacting_pair'].replace(
    {'CD1B_complex_IL6ST': 'CD1B_IL6ST'}
)

tumor_color_map = {
    "NSCLC": "gold",      # Yellow
    "SCLC-A": "tab:red",      # Red
    "SCLC-N": "tab:cyan",      # Cyan
    "SCLC-P": "tab:blue",      # Blue
    "NonNE SCLC": "tab:purple"}

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

conditions = median['condition'].cat.categories

# Compute width ratios based on unique sender_receiver_pair counts
width_ratios = [
    median[median['condition'] == cond]['sender_receiver_pair'].nunique()
    for cond in conditions
]
width_ratios = np.array(width_ratios, dtype=float)
width_ratios /= width_ratios.min()

# Fix interacting_pair order for y-axis
from scipy.cluster.hierarchy import linkage, leaves_list
import pandas as pd

# Pivot to matrix form

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Choose condition
condition = "ADC → SCLC"

# Filter and sort data
condition_data = median[median['condition'] == condition].copy()
pivot = condition_data.pivot_table(
    index='interacting_pair',
    columns='sender_receiver_pair',
    values='value',
    aggfunc='mean',
    fill_value=0
)
linkage_rows = linkage(pivot, method='average', metric='euclidean')
interacting_order = pivot.index[leaves_list(linkage_rows)]

# Update categorical order in original DataFrame
condition_data['interacting_pair'] = pd.Categorical(
    condition_data['interacting_pair'],
    categories=interacting_order,
    ordered=True
)

# Get sender-receiver pairs
all_pairs = condition_data['sender_receiver_pair'].unique()

# Create full grid
full_index = pd.MultiIndex.from_product(
    [all_pairs, interacting_order],
    names=['sender_receiver_pair', 'interacting_pair']
)
condition_data = condition_data.set_index(['sender_receiver_pair', 'interacting_pair'])
condition_data = condition_data.reindex(full_index).reset_index()
condition_data['value'] = condition_data['value'].fillna(0)

# Remove all-zero columns and rows
nonzero_senders = (
    condition_data.groupby('sender_receiver_pair')['value'].sum().reset_index().query("value > 0")
)
filtered = condition_data.merge(nonzero_senders[['sender_receiver_pair']], on='sender_receiver_pair')

nonzero_interactions = (
    filtered.groupby('interacting_pair')['value'].sum().reset_index().query("value > 0")
)
filtered = filtered.merge(nonzero_interactions[['interacting_pair']], on='interacting_pair')

# Plotting
fig = plt.figure(figsize=(1.5, 3))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0])

sc = ax.scatter(
    filtered['interacting_pair'],
    filtered['sender_receiver_pair'],
    s=filtered['value'] * 50,
    c=filtered['value'],
    cmap='viridis',
    alpha=0.7,
    edgecolors='w',
    linewidth=0.5,
    vmax=1.5,
)
ax.margins(x=0.15, y=0.08)

# Reverse y-axis
ax.invert_yaxis()

# Move x-axis ticks to top and rotate
ax.xaxis.tick_top()
plt.setp(ax.get_xticklabels(), rotation=45, ha='left')

# Set axis labels
ax.set_xlabel('Ligand Receptor', fontsize=12, fontstyle='normal', fontweight='bold', labelpad=3)
ax.xaxis.set_label_position('top')
ax.set_ylabel('Sender → Receiver', fontsize=12, fontstyle='normal', fontweight='bold')

# Adjust spacing
fig.subplots_adjust(left=0.25, right=0.95, top=0.8, bottom=0.1)

# Colorbar - move further left
cbar_ax = fig.add_axes([-0.5, 1, 0.3, 0.03])  # [left, bottom, width, height]
cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cb.set_label('Expression')
plt.savefig('Joe_basal_dotplot.png', dpi=300, bbox_inches='tight')
plt.show()
