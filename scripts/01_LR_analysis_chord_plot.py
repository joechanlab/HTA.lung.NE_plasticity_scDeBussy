import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
from pycirclize import Circos
import tempfile
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from constants import sample_to_group, tumor_color_map

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

def get_interaction_type(sender, receiver):
    if sender in tumor_cells and receiver in tumor_cells:
        return "within_tumor"
    elif ((sender in myeloid_cells and receiver in tumor_cells) or
          (sender in tumor_cells and receiver in myeloid_cells)):
        return "myeloid_tumor"
    elif ((sender in lymphoid_cells and receiver in tumor_cells) or
          (sender in tumor_cells and receiver in lymphoid_cells)):
        return "lymphoid_tumor"
    elif ((sender in stromal_cells and receiver in tumor_cells) or
           (sender in tumor_cells and receiver in stromal_cells)):
        return "stromal_tumor"
    else:
        return "other"

#----------------------------------
# relevant interactions
base_dir = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/out.individual.120122/"
relevant_file_pattern = base_dir + "/*/cellphonedb.output/degs_analysis_relevant_interactions_042725.txt"
file_list =  glob.glob(relevant_file_pattern)
subject_list = extract_ru_subdir(file_list)
condition_labels = pd.DataFrame(dict(
    patient=subject_list,
    condition=[sample_to_group.get(x, "unknown") for x in subject_list]
))
condition_labels.set_index('patient', inplace=True)
condition_list = condition_labels.condition.values
dfs = read_tab_delimited_files(file_list)

#----------------------------------
# Data for Fisher's exact test
# To find out the differences in ligand-receptor interactions between transformed and control groups
# Filter for tumors only
def group_nsclc(x):
    if x in ['Basal-like_NSCLC', 'Mucinous_NSCLC']:
        return 'NSCLC'
    elif x == "Mφ/_Mono":
        return "Mφ/Mono"
    else:
        return x

long_df_list = []
for i, df in enumerate(dfs):
    if condition_list[i] != "unknown":
        print(f"Processing {subject_list[i]}: {condition_list[i]}")
        interacting_pairs = df['interacting_pair']
        classification = df['classification']
        df = df.iloc[:,13:]
        df.loc[:,'interacting_pair'] = interacting_pairs
        df.loc[:,'classification'] = classification
        df.loc[:,'subject'] = subject_list[i]
        df.loc[:,'condition'] = condition_list[i]
        df = df.reset_index(drop=True)
        
        long_df = pd.melt(df, id_vars=['interacting_pair', 'classification', 'subject', 'condition'], var_name='pair', value_name='value')
        if long_df.shape[0] > 0:
            # Split the 'pair' column into 'sender' and 'receiver'
            long_df[['sender', 'receiver']] = long_df['pair'].str.split('|', expand=True)
            # Optionally, drop the original 'pair' column if not needed
            long_df = long_df.drop(columns=['pair'])
            #long_df = long_df[long_df['value'] != 0]
            long_df['sender'] = long_df['sender'].str.split(r'[.]').str[0]
            long_df['receiver'] = long_df['receiver'].str.split(r'[.]').str[0]
            long_df['sender'] = long_df['sender'].apply(group_nsclc)
            long_df['receiver'] = long_df['receiver'].apply(group_nsclc)
            long_df_list.append(long_df)
relevance_long_df = pd.concat(long_df_list, ignore_index=True)
relevance_long_df = relevance_long_df.loc[relevance_long_df.groupby(['subject', 'condition', 'sender', 'receiver', 'interacting_pair'])['value'].idxmax()]
relevance_long_df = relevance_long_df.loc[~(((relevance_long_df.sender == 'NSCLC') & (relevance_long_df.receiver.isin(['SCLC-A', 'SCLC-N']))) | ((relevance_long_df.sender.isin(['SCLC-A', 'SCLC-N'])) & (relevance_long_df.receiver == 'NSCLC')) & (relevance_long_df.condition == 'de_novo_SCLC_LUAD'))]

tumor_cells = {"NSCLC", "SCLC-A", "SCLC-N", "NonNE SCLC"}
myeloid_cells = {"Mφ/Mono", "Mφ/Mono_CD14", "Mφ/Mono_CD11c", "Mφ/Mono_CCL", "PMN", "Mast cell", "pDC"}
lymphoid_cells = {"CD4_EM_Effector", "CD4_naive_CM", "CD4_TRM", "CD8_TRM_EM", "T_reg", "NK", "ILC", "B_memory"}
stromal_cells = {"Fibroblast", "Endothelial", "Basal", "Ciliated"}
relevance_long_df['interaction_type'] = relevance_long_df.apply(
    lambda row: get_interaction_type(row['sender'], row['receiver']), axis=1
)
relevance_long_df.to_csv("../data/processed/relevance_long_df.csv", index=False)
# combine the information from the relevance as well
within_tumor_relevance = relevance_long_df[relevance_long_df.interaction_type == "within_tumor"].drop(columns=['interaction_type'])
within_tumor_relevance = within_tumor_relevance.loc[within_tumor_relevance.value != 0]
within_tumor_relevance.to_csv("../data/processed/within_tumor_relevance.csv", index=False)

#----------------------------------
# Prepare for data for chord plot
# def group_nsclc(x):
#     if x in ['Basal-like_NSCLC', 'Mucinous_NSCLC']:
#         return 'NSCLC'
#     elif x == "NonNE SCLC":
#         return "NonNE\nSCLC"
#     elif x == "Mφ/_Mono":
#         return "Mφ/Mono"
#     elif x == "Mφ/Mono_CD11c":
#         return "Mφ/Mono\nCD11c"
#     elif x == "Mφ/Mono_CD14":
#         return "Mφ/Mono\nCD14"
#     elif x == "Mφ/Mono_CCL":
#         return "Mφ/Mono\nCCL"
#     elif x == "CD4_EM/Effector":
#         return "CD4_EM\nEffector"
#     elif x == "CD4_naive/CM":
#         return "CD4_naive\nCM"
#     elif x == "CD8_TRM/EM":
#         return "CD8_TRM\nEM"
#     elif x == "Mast_cell":
#         return "Mast cell"
#     else:
#         return x

# long_df_list = []
# for i, df in enumerate(dfs):
#     if condition_list[i] != "unknown":
#         print(f"Processing {subject_list[i]}: {condition_list[i]}")
#         interacting_pairs = df['interacting_pair']
#         df = df.iloc[:,13:]
#         df.loc[:,'interacting_pair'] = interacting_pairs
#         df.loc[:,'subject'] = subject_list[i]
#         df.loc[:,'condition'] = condition_list[i]
#         df = df.reset_index(drop=True)
        
#         long_df = pd.melt(df, id_vars=['interacting_pair', 'subject', 'condition'], var_name='pair', value_name='value')
#         if long_df.shape[0] > 0:
#             # Split the 'pair' column into 'sender' and 'receiver'
#             long_df[['sender', 'receiver']] = long_df['pair'].str.split('|', expand=True)
#             # Optionally, drop the original 'pair' column if not needed
#             long_df = long_df.drop(columns=['pair'])
#             long_df = long_df[long_df['value'] != 0]
#             long_df['sender'] = long_df['sender'].str.split(r'[.]').str[0]
#             long_df['receiver'] = long_df['receiver'].str.split(r'[.]').str[0]
#             long_df['sender'] = long_df['sender'].apply(group_nsclc)
#             long_df['receiver'] = long_df['receiver'].apply(group_nsclc)
#             long_df_list.append(long_df)
# relevance_long_df = pd.concat(long_df_list, ignore_index=True)
# relevance_long_df = relevance_long_df.loc[relevance_long_df.groupby(['subject', 'condition', 'sender', 'receiver', 'interacting_pair'])['value'].idxmax()]
# relevance_long_df = relevance_long_df.loc[~(((relevance_long_df.sender == 'NSCLC') & (relevance_long_df.receiver.isin(['SCLC-A', 'SCLC-N']))) | ((relevance_long_df.sender.isin(['SCLC-A', 'SCLC-N'])) & (relevance_long_df.receiver == 'NSCLC')) & (relevance_long_df.condition == 'de_novo_SCLC_LUAD'))]
# relevance_long_df.loc[:,'condition'] = pd.Categorical(relevance_long_df['condition'], categories=["De novo SCLC and ADC", "ADC → SCLC"], ordered=True)

# #----------------------------------
# # Separate the interaction as within tumors, myeloid-tumor, lymphoid tumor
# tumor_cells = {"NSCLC", "SCLC-A", "SCLC-N", "NonNE\nSCLC"}
# myeloid_cells = {"Mφ/Mono", "Mφ/Mono\nCD14", "Mφ/Mono\nCD11c", "Mφ/Mono\nCCL", "PMN", "Mast cell", "pDC"}
# lymphoid_cells = {"CD4_EM\nEffector", "CD4_naive\nCM", "CD4_TRM", "CD8_TRM\nEM", "T_reg", "NK", "ILC", "B_memory"}
# stromal_cells = {"Fibroblast", "Endothelial", "Basal", "Ciliated"}
# relevance_long_df['interaction_type'] = relevance_long_df.apply(
#     lambda row: get_interaction_type(row['sender'], row['receiver']), axis=1
# )
# interaction_types = ["within_tumor", "myeloid_tumor", "lymphoid_tumor", "stromal_tumor"]

# for interaction_type in interaction_types:
#     filtered_df = relevance_long_df[relevance_long_df['interaction_type'] == interaction_type]
#     if filtered_df.empty:
#         print(f"No interactions for {interaction_type}")
#         continue

#     # Pivot as before
#     pivot = filtered_df.pivot_table(
#         index=['subject', 'condition', 'sender'],
#         columns='receiver',
#         values='value',
#         aggfunc='sum',
#         fill_value=0
#     ).reset_index().rename_axis(None, axis=1)
#     median = pivot.drop(columns=['subject']).groupby(['condition', 'sender']).median()
#     all_senders = median.index.get_level_values('sender').unique().sort_values()

#     condition_matrices = {}
#     for condition in ["De novo SCLC and ADC", "ADC → SCLC"]:
#         cond_df = median.loc[condition]
#         cond_df_reindexed = cond_df.reindex(all_senders, fill_value=0)
#         condition_matrices[condition] = cond_df_reindexed

#     # Chord plot for each condition
#     all_nodes = set()
#     for matrix_df in condition_matrices.values():
#         all_nodes.update(matrix_df.index)
#         all_nodes.update(matrix_df.columns)
#     all_nodes = sorted(all_nodes)

#     cmap = cm.get_cmap('tab20', len(all_nodes))
#     node2color = {}
#     for node in all_nodes:
#         if node in tumor_color_map:
#             node2color[node] = tumor_color_map[node]
#         else:
#             node2color[node] = mcolors.to_hex(cmap(all_nodes.index(node)))

#     images = []
#     labels = []
#     for condition, matrix_df in condition_matrices.items():
#         matrix_df = matrix_df.fillna(0)
#         matrix_df = matrix_df.reindex(index=all_nodes, columns=all_nodes, fill_value=0)
#         matrix_df = matrix_df.loc[(matrix_df != 0).any(axis=1), (matrix_df != 0).any(axis=0)]
#         filtered_order = [node for node in all_nodes if node in matrix_df.index and node in matrix_df.columns]
#         if matrix_df.empty or len(filtered_order) == 0:
#             print(f"Skipping {condition} for {interaction_type}: matrix is empty after filtering.")
#             continue

#         circos = Circos.chord_diagram(
#             matrix_df,
#             space=19,
#             cmap=node2color,
#             label_kws=dict(size=25),
#             link_kws=dict(ec="black", lw=0.5, direction=1),
#             #order=filtered_order
#         )
#         tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
#         circos.savefig(tmpfile.name)
#         images.append(Image.open(tmpfile.name))
#         labels.append(condition)

#     # Plot images side by side
#     if images:
#         fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
#         if len(images) == 1:
#             axes = [axes]
#         for ax, img, label in zip(axes, images, labels):
#             ax.imshow(img)
#             ax.axis('off')
#             ax.set_title(f"{label}", fontsize=21)
#         plt.tight_layout()
#         fig.savefig(f"../results/figures/{interaction_type}_chord_plot.png", dpi=300, bbox_inches='tight')
#         plt.show()

# #------------------------------------------
# cell_sign_file = base_dir + "/*/cellphonedb.output/degs_analysis_CellSign_active_interactions_deconvoluted_042725.txt"
# cell_sign_file_list = glob.glob(cell_sign_file)
# cell_sign_dfs = read_tab_delimited_files(cell_sign_file_list)
# cell_sign_long_df_list = []
# cell_sign_subject_list = extract_ru_subdir(cell_sign_file_list)
# cell_sign_condition_labels = pd.DataFrame(dict(
#     patient=cell_sign_subject_list,
#     condition=[sample_to_group.get(x, "unknown") for x in cell_sign_subject_list]
# ))
# cell_sign_condition_list = cell_sign_condition_labels.condition.values

# for i, df in enumerate(cell_sign_dfs):
#     if cell_sign_condition_list[i] != "unknown":
#         interacting_pairs = df['interacting_pair']
#         df.loc[:,'interacting_pair'] = interacting_pairs
#         df.loc[:,'subject'] = subject_list[i]
#         df = df.reset_index(drop=True)
#         df[['sender', 'receiver']] = df['celltype_pairs'].str.split('|', expand=True)
#         df = df.drop(columns=['celltype_pairs'])
#         df['sender'] = df['sender'].str.split(r'[.]').str[0]
#         df['receiver'] = df['receiver'].str.split(r'[.]').str[0]
#         df['sender'] = df['sender'].apply(group_nsclc)
#         df['receiver'] = df['receiver'].apply(group_nsclc)
#         cell_sign_long_df_list.append(df)
# cell_sign_long_df = pd.concat(cell_sign_long_df_list, ignore_index=True)
# cell_sign_long_df.to_csv('../data/processed/cell_sign_long_df_detailed.csv', index=False)
# cell_sign_long_df = cell_sign_long_df.loc[:,['interacting_pair', 'subject', 'sender', 'receiver', 'active_TF']].drop_duplicates()
# cell_sign_long_df.to_csv("../data/processed/cell_sign_long_df.csv", index=False)

