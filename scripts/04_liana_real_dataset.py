from utils.liana import directional_rank
from utils._aggregate import _robust_rank_aggregate
from utils.data_helpers import summarize_lr_interactions, classify_interaction, group_cell_type
from utils.plotting_helpers import plot_interaction_heatmap, create_tf_activation_legend
from utils.constants import sample_to_group, sample_groups

import pandas as pd
import glob
import numpy as np
import scanpy as sc
import statsmodels.formula.api as smf
sc.set_figure_params(fontsize=16)
import seaborn as sns
sns.set_style("ticks")

base_dir = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/out.individual.120122/"
liana_file_pattern = base_dir + "/*/liana_res/liana_res.*.csv"
liana_files = glob.glob(liana_file_pattern)
liana_dfs = []
for file in liana_files:
    sample_name = file.split('/')[-1].split('.')[1]
    #Skip samples ending with b or c (b/c Joe ran cellphonedb on bioreplicate level)
    if sample_name.endswith(('b', 'c')):
       continue
    df = pd.read_csv(file)
    df['subject'] = file.split('/')[-3]
    df['sample'] = sample_name[:-1] if sample_name.endswith('a') else sample_name
    liana_dfs.append(df)

liana_df = pd.concat(liana_dfs)
liana_df = liana_df.iloc[:,1:]

print("\nSelected samples and their counts:")
print(liana_df['sample'].value_counts())

liana_df.drop_duplicates(inplace=True)

#--------------------------------
# replace the cellphonedb p_val with relevance and add the TF field ranking
relevance_file = '../data/processed/cellphonedb_v5_relevance.csv'
relevance_df = pd.read_csv(relevance_file)
relevance_df = relevance_df.loc[:,['source', 'target', 'interacting_pair', 'ligand_complex', 'receptor_complex', 'classification', 'subject', 'sample', 'relevance']]
relevance_df.drop_duplicates(inplace=True)

cell_sign_file = '../data/processed/cellphonedb_v5_cellsign.csv'
cell_sign_df = pd.read_csv(cell_sign_file)
cell_sign_df = cell_sign_df.loc[:,['source', 'target', 'ligand_complex', 'receptor_complex', 'subject', 'sample', 'cellphonedb_cellsign', 'active_TF']]
cell_sign_df.drop_duplicates(inplace=True)

interaction_scores_file = '../data/processed/cellphonedb_v5_interaction_scores.csv'
interaction_scores_df = pd.read_csv(interaction_scores_file)
interaction_scores_df = interaction_scores_df.loc[:,['source', 'target', 'interacting_pair', 'ligand_complex', 'receptor_complex', 'classification', 'subject', 'sample', 'interaction_score']]
interaction_scores_df.drop_duplicates(inplace=True)

cellphonedb_df = pd.merge(interaction_scores_df, relevance_df, on=['source', 'target', 'interacting_pair', 'ligand_complex', 'receptor_complex', 'classification', 'subject', 'sample'], how='left')
cellphonedb_df = pd.merge(cellphonedb_df, cell_sign_df, on=['source', 'target', 'ligand_complex', 'receptor_complex', 'subject', 'sample'], how='left')

liana_df = liana_df.merge(cellphonedb_df, on=['source', 'target', 'ligand_complex', 'receptor_complex', 'subject', 'sample'], how='left')
# rename cellphonedb_cellsign to CellSign_active
liana_df.rename(columns={'cellphonedb_cellsign': 'CellSign_active'}, inplace=True)
liana_df.loc[:,'CellSign_active'] = liana_df.loc[:,'CellSign_active'].fillna(0)
liana_df.loc[:,'relevance'] = liana_df.loc[:,'relevance'].fillna(0)
liana_df.loc[:,'interaction_score'] = liana_df.loc[:,'interaction_score'].fillna(0)

#--------------------------------
# compute the updated specificity rank per sample
spec_cols = ['cellphone_pvals',  'spec_weight', 'scaled_weight', 'lr_logfc', 'relevance', 'CellSign_active']
spec_ascending = [True, False, False, False, False, False]
mag_cols = ['lr_means', 'expr_prod', 'lrscore', 'interaction_score']
mag_ascending = [False, True, False, False]

# Initialize a new column for computed specificity ranks
liana_df['computed_specificity_rank'] = np.nan
liana_df['computed_magnitude_rank'] = np.nan
# Compute specificity ranks for each sample
for sample in liana_df['sample'].unique():
    sample_mask = liana_df['sample'] == sample
    sample_data = liana_df[sample_mask]
    
    # Compute ranks for this sample
    spec_rmat = directional_rank(sample_data, spec_cols, spec_ascending)
    spec_pvals = _robust_rank_aggregate(spec_rmat.to_numpy())
    
    mag_rmat = directional_rank(sample_data, mag_cols, mag_ascending)
    mag_pvals = _robust_rank_aggregate(mag_rmat.to_numpy())
    # Update the computed specificity rank for this sample
    liana_df.loc[sample_mask, 'computed_specificity_rank'] = spec_pvals
    liana_df.loc[sample_mask, 'computed_magnitude_rank'] = mag_pvals

# Compare specificity ranks
print("\nSpecificity rank comparison:")
print("Original vs Computed correlation:", np.corrcoef(liana_df["specificity_rank"], liana_df["computed_specificity_rank"])[0,1])
print("\nFirst 5 rows comparison:")
comparison_df = pd.DataFrame({
    'original': liana_df["specificity_rank"].head(5),
    'computed': liana_df["computed_specificity_rank"].head(5)
})
print(comparison_df)

print("\nMagnitude rank comparison:")
print("Original vs Computed correlation:", np.corrcoef(liana_df["magnitude_rank"], liana_df["computed_magnitude_rank"])[0,1])
print("\nFirst 5 rows comparison:")
comparison_df = pd.DataFrame({
    'original': liana_df["magnitude_rank"].head(5),
    'computed': liana_df["computed_magnitude_rank"].head(5)
})
print(comparison_df)
#--------------------------------
# Summarize the interactions
liana_df.loc[:,'condition'] = liana_df.loc[:,'subject'].map(sample_to_group)
# rename source and sender and target as receiver
liana_df.rename(columns={'source': 'receiver', 'target': 'sender'}, inplace=True)
# remove rows with interacting pair as nan
liana_df = liana_df[liana_df['interacting_pair'].notna()]
liana_df.loc[:,'interaction_type'] = liana_df.apply(lambda row: classify_interaction(row['sender'], row['receiver']), axis=1)
liana_df = liana_df[liana_df['interaction_type'] != 'other']
liana_df.sort_values(by=['computed_magnitude_rank', 'computed_specificity_rank'], ascending=True)
liana_df.to_csv("../results/tables/liana_df.csv", index=False)
liana_df = pd.read_csv("../results/tables/liana_df.csv")

def summarize_lr_interactions(relevance_df, sample_groups, score_column='computed_specificity_rank', threshold=0.05):
    """
    Summarize ligand-receptor interactions across conditions using a binary confidence threshold
    on specificity or magnitude score, and test differences using Fisher's exact test.

    Parameters:
    -----------
    relevance_df : pandas.DataFrame
        DataFrame containing LIANA outputs with sample-level scores.

    sample_groups : dict
        Dictionary with condition names as keys and lists of patient IDs as values.

    score_column : str
        Name of the score column to use for confidence thresholding (e.g., 'computed_specificity_rank').

    threshold : float
        Threshold below which scores are considered confident.

    Returns:
    --------
    pandas.DataFrame
        Summary statistics per ligand-receptor interaction.
    """

    from scipy.stats import fisher_exact
    import numpy as np
    import pandas as pd

    total_patients = {cond: len(ids) for cond, ids in sample_groups.items()}
    relevance_df['sender_receiver_pair'] = relevance_df['sender'] + '→' + relevance_df['receiver']

    subject_agg = []
    for (interacting_pair, classification, sender_receiver_pair, subject), group in relevance_df.groupby([
        'interacting_pair', 'classification', 'sender_receiver_pair', 'subject']):

        active_tfs = group['active_TF'].dropna().unique()
        active_tfs_str = ';'.join(active_tfs) if len(active_tfs) > 0 else ''

        score_vals = group[score_column].dropna()
        confident_score = (score_vals < threshold).any() if not score_vals.empty else False

        subject_row = {
            'interacting_pair': interacting_pair,
            'classification': classification,
            'sender_receiver_pair': sender_receiver_pair,
            'subject': subject,
            'sender': group['sender'].iloc[0],
            'receiver': group['receiver'].iloc[0],
            'condition': group['condition'].iloc[0],
            'has_confident_score': confident_score,
            'active_TFs': active_tfs_str
        }
        subject_agg.append(subject_row)

    subject_df = pd.DataFrame(subject_agg)

    summary = []
    for (interacting_pair, sender_receiver_pair), group in subject_df.groupby(['interacting_pair', 'sender_receiver_pair']):
        stats = {}
        for condition in total_patients:
            condition_data = group[group['condition'] == condition]

            freq = len(condition_data[condition_data['has_confident_score']]) / total_patients[condition]
            confident_subjects = condition_data[condition_data['has_confident_score']]['subject'].tolist()

            stats[condition] = {
                'confidence_freq': freq,
                'n_subjects': len(condition_data),
                'confident_subjects': ';'.join(confident_subjects),
                'n_confident': len(confident_subjects)
            }
        # Fisher's exact test
        if 'De novo SCLC and ADC' in stats and 'ADC → SCLC' in stats:
            contingency_table = np.array([
                [stats['De novo SCLC and ADC']['n_confident'], total_patients['De novo SCLC and ADC'] - stats['De novo SCLC and ADC']['n_confident']],
                [stats['ADC → SCLC']['n_confident'], total_patients['ADC → SCLC'] - stats['ADC → SCLC']['n_confident']]
            ])
            try:
                fisher_pval = fisher_exact(contingency_table)[1]
            except:
                fisher_pval = np.nan
        else:
            fisher_pval = np.nan

        interaction_type = classify_interaction(group['sender'].iloc[0], group['receiver'].iloc[0])
        all_tfs = group['active_TFs'].unique()
        active_tfs = ';'.join([tf for tf in all_tfs if tf != ''])

        summary_row = {
            'interacting_pair': interacting_pair,
            'classification': group['classification'].iloc[0],
            'sender_receiver_pair': sender_receiver_pair,
            'sender': group['sender'].iloc[0],
            'receiver': group['receiver'].iloc[0],
            'interaction_type': interaction_type,
            'De novo SCLC and ADC_confidence_freq': stats['De novo SCLC and ADC']['confidence_freq'],
            'ADC → SCLC_confidence_freq': stats['ADC → SCLC']['confidence_freq'],
            'De novo SCLC and ADC_n_subjects': stats['De novo SCLC and ADC']['n_subjects'],
            'ADC → SCLC_n_subjects': stats['ADC → SCLC']['n_subjects'],
            'De novo SCLC and ADC_confident_subjects': stats['De novo SCLC and ADC']['confident_subjects'],
            'ADC → SCLC_confident_subjects': stats['ADC → SCLC']['confident_subjects'],
            'active_TFs': active_tfs,
            'fisher_pval': fisher_pval
        }
        summary.append(summary_row)

    summary_df = pd.DataFrame(summary).sort_values('fisher_pval')
    return summary_df

summary = summarize_lr_interactions(liana_df, sample_groups)

# Calculate absolute difference in relevance frequency between conditions
summary['freq_diff'] = summary['ADC → SCLC_confidence_freq'] - summary['De novo SCLC and ADC_confidence_freq']
summary.to_csv("../results/tables/liana_interaction_summary.csv", index=False)

summary = pd.read_csv("../results/tables/liana_interaction_summary.csv")
# Analyze each interaction type
interaction_types = ['within_tumor', 'tumor_immune', 'tumor_stromal']

# within_tumor
within_tumor_ordering = [
    "NSCLC",
    "SCLC-A",
    "SCLC-N",
]
row_order = ['APP_CD74', 'NCAM1_FGFR1','CCL28_CCR10','SEMA4D_PLXNB1', 
        'EFNA5_EPHB2', 'EFNB3_EPHB2',  'EFNB1_EPHB1', 'EFNB1_EPHB2', 'EFNB3_EPHB4',
        'GAS6_TYRO3', 
        'JAG1_NOTCH1', 'JAG2_NOTCH1', 'DLL1_NOTCH2', 'CNTN1_NOTCH1',
       'WNT2B_FZD7_LRP5', 'WNT2B_FZD7_LRP6', 'WNT4_FZD6_LRP5',
        'WNT4_FZD3_LRP5', 'WNT2B_FZD6_LRP6',
       'WNT2B_FZD3_LRP5', 'WNT2B_FZD6_LRP5', 'CADM3_CADM1']
type_summary = summary[summary['interaction_type'] == 'within_tumor']
type_summary['active_TFs'] = type_summary['active_TFs'].fillna('')
plot_interaction_heatmap(
        type_summary,
        output_file=f'../results/figures/liana_within_tumor_interactions_heatmap_sender.png',
        top_n=30,
        pval_threshold=0.1,
        figsize=(10, 8),
        row_order=row_order,
        facet_by='sender',
        facet_order=within_tumor_ordering
    )
plot_interaction_heatmap(
    type_summary,
    output_file=f'../results/figures/liana_within_tumor_interactions_heatmap_sender_no_TF_names.png',
    top_n=30,
    pval_threshold=0.1,
    figsize=(10, 8),
    row_order=row_order,
    facet_by='sender',
    facet_order=within_tumor_ordering,
    show_TF_names=False
)

# tumor_stromal
tumor_stromal_ordering = [
    "SCLC-N",
    "Endothelial",
    "Fibroblast"
]
row_order = ['TNFSF12_TNFRSF12A', 'GAS6_TYRO3', 'SEMA4C_PLXNB2', 
 'EFNB1_EPHB2', 'EFNB2_EPHB2','EFNB1_EPHA4',  'EFNB1_EPHB4',
 'APP_CD74', 
'VEGFA_NRP2', 'VEGFB_NRP1', 'VEGFA_NRP1', 
'PGF_NRP1', 'PGF_NRP2', 
 'JAG1_NOTCH1','JAG1_NOTCH2','JAG1_NOTCH3']
type_summary = summary[summary['interaction_type'] == 'tumor_stromal']
type_summary['active_TFs'] = type_summary['active_TFs'].fillna('')
plot_interaction_heatmap(
    type_summary,
    output_file=f'../results/figures/liana_tumor_stromal_interactions_heatmap_sender.png',
    top_n=30,
    pval_threshold=0.1,
    figsize=(10, 8),
    row_order=row_order,
    facet_by='sender',
    facet_order=tumor_stromal_ordering
)
plot_interaction_heatmap(
    type_summary,
    output_file=f'../results/figures/liana_tumor_stromal_interactions_heatmap_sender_no_TF_names.png',
    top_n=30,
    pval_threshold=0.1,
    figsize=(10, 8),
    row_order=row_order,
    facet_by='sender',
    facet_order=tumor_stromal_ordering,
    show_TF_names=False
)


# tumor_immune
tumor_immune_ordering = [
    "SCLC-A",
    "SCLC-N",
]
row_order = ['TNFSF14_TNFRSF14',  'BTLA_TNFRSF14', 'TNFSF12_TNFRSF12A', 'TNFSF14_LTBR',
        'VEGFA_NRP1', 'VEGFB_NRP1', 'TGFB1_TGFBR3',
       'SEMA4D_PLXNB1','SEMA4D_PLXNB2', 
       'APP_CD74', 'CDH1_KLRG1',
       'DLK1_NOTCH2', 'DLL3_NOTCH2', 'DLK1_NOTCH1', 'DLL3_NOTCH1', 'DLL1_NOTCH1', 
       'PODXL2_SELL','IGF2_IGF2R',
       'NMU_NMUR1']
type_summary = summary[summary['interaction_type'] == 'tumor_immune']
type_summary['active_TFs'] = type_summary['active_TFs'].fillna('')
plot_interaction_heatmap(
    type_summary,
    output_file=f'../results/figures/liana_tumor_immune_interactions_heatmap_sender.png',
    top_n=30,
    pval_threshold=0.1,
    figsize=(10, 8),
    row_order=row_order,
    facet_by='sender',
    facet_order=tumor_immune_ordering
)
plot_interaction_heatmap(
    type_summary,
    output_file=f'../results/figures/liana_tumor_immune_interactions_heatmap_sender_no_TF_names.png',
    top_n=30,
    pval_threshold=0.1,
    figsize=(10, 8),
    row_order=row_order,
    facet_by='sender',
    facet_order=tumor_immune_ordering,
    show_TF_names=False
)

plot_interaction_heatmap(
        type_summary,
        output_file=f'../results/figures/liana_tumor_immune_interactions_heatmap_receiver.png',
        top_n=30,
        pval_threshold=0.1,
        figsize=(10, 8),
        row_order=row_order,
        facet_by='receiver',
        facet_order=tumor_immune_ordering
    )
plot_interaction_heatmap(
    type_summary,
    output_file=f'../results/figures/liana_tumor_immune_interactions_heatmap_receiver_no_TF_names.png',
    top_n=30,
    pval_threshold=0.1,
    figsize=(10, 8),
    row_order=row_order,
    facet_by='receiver',
    facet_order=tumor_immune_ordering,
    show_TF_names=False
)

create_tf_activation_legend(output_file='../results/figures/liana_tf_activation_legend.png')