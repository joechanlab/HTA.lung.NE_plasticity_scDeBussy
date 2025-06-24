import seaborn as sns
from matplotlib.colors import to_hex
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import fdrcorrection_twostage
from joblib import Parallel, delayed
from tqdm import tqdm
import statsmodels.formula.api as smf
from utils.data_helpers import sample_groups
from utils.plotting_helpers import plot_interaction_heatmap
import scanpy as sc
sc.set_figure_params(fontsize=15)
sns.set_style("ticks")
def generate_class_palette(classifications, base_palette="tab20"):
    """
    Generate a consistent color palette for classification categories.

    Parameters:
    - classifications: list or set of unique classification names.
    - base_palette: seaborn palette name to draw colors from.

    Returns:
    - dict mapping classification -> color (hex).
    """
    unique_classes = sorted(set(classifications))
    num_classes = len(unique_classes)
    palette_colors = sns.color_palette(base_palette, num_classes)
    
    class_palette = {cls: to_hex(color) for cls, color in zip(unique_classes, palette_colors)}
    
    return class_palette


def aggregate_subjects(relevance_df, score_column, threshold):
    relevance_df['active_TFs'] = relevance_df['active_TF'].fillna('').astype(str)
    relevance_df['is_confident'] = (relevance_df[score_column] < threshold).astype(int)
    relevance_df = relevance_df.loc[relevance_df.is_confident == 1,:]
    
    agg = relevance_df.groupby([
        'interacting_pair', 'classification', 'sender_receiver_pair', 'subject'
    ]).agg({
        'sender': 'first',
        'receiver': 'first',
        'condition': 'first',
        'tissue': 'first',
        'chemo': 'first',
        'IO': 'first',
        'TKI': 'first',
        'active_TFs': lambda x: ';'.join(filter(None, x.unique())),
        'is_confident': 'max'  # If any sample is confident → mark subject as confident
    }).reset_index()
    
    return agg

def prefilter_pairs(subject_df, total_patients, min_confident=5, fisher_pval_cutoff=0.2):
    results = []
    grouped = subject_df.groupby(['interacting_pair', 'sender_receiver_pair'])
    for (interacting_pair, sender_receiver_pair), group in grouped:
        n_A = group[group['condition'] == 'De novo SCLC and ADC']['is_confident'].sum()
        n_B = group[group['condition'] == 'ADC → SCLC']['is_confident'].sum()
        contingency = np.array([
            [n_A, total_patients['De novo SCLC and ADC'] - n_A],
            [n_B, total_patients['ADC → SCLC'] - n_B]
        ])
        try:
            fisher_p = fisher_exact(contingency)[1]
        except:
            fisher_p = 1.0
        
        freq_A = n_A / total_patients['De novo SCLC and ADC']
        freq_B = n_B / total_patients['ADC → SCLC']
        freq_diff = freq_B - freq_A
        total_confident = n_A + n_B
        
        keep = (total_confident >= min_confident) and (fisher_p <= fisher_pval_cutoff)
        results.append({
            'interacting_pair': interacting_pair,
            'sender_receiver_pair': sender_receiver_pair,
            'n_confident_A': n_A,
            'n_confident_B': n_B,
            'freq_A': freq_A,
            'freq_B': freq_B,
            'freq_diff': freq_diff,
            'fisher_pval': fisher_p,
            'pass_prefilter': keep
        })
    return pd.DataFrame(results)

def process_group(interacting_pair, sender_receiver_pair, group):
    try:
        group = group.copy()
        for col in ['condition', 'tissue', 'chemo', 'IO', 'TKI']:
            if col == 'condition':
                group[col] = pd.Categorical(group[col], categories=['De novo SCLC and ADC', 'ADC → SCLC'])
            else:
                group[col] = pd.Categorical(group[col])
        formula = "is_confident ~ condition + tissue + chemo + IO + TKI"
        model = smf.ols(formula, data=group)
        result = model.fit()
        pval = result.pvalues.get('condition[T.ADC → SCLC]', np.nan)
    except Exception as e:
        pval = np.nan
    return {
        'interacting_pair': interacting_pair,
        'sender_receiver_pair': sender_receiver_pair,
        'glmm_pval': pval
    }

def summarize_lr_interactions(relevance_df, metadata_df, sample_groups, score_column='computed_specificity_rank', 
                              sample_column='sample_name', threshold=0.2, n_jobs=4, min_confident=5, fisher_pval_cutoff=0.2):

    relevance_df['sender_receiver_pair'] = relevance_df['sender'] + '→' + relevance_df['receiver']
    relevance_df = relevance_df.merge(metadata_df, on=['sample', 'subject'], how='left')
    print(f"Input shape: {relevance_df.shape}")
    
    subject_df = aggregate_subjects(relevance_df, score_column, threshold=threshold)
    total_patients = {cond: len(ids) for cond, ids in sample_groups.items()}
    to_test_df = prefilter_pairs(subject_df, total_patients, min_confident, fisher_pval_cutoff)
    to_test_pairs = set(zip(to_test_df[to_test_df['pass_prefilter']]['interacting_pair'], 
                             to_test_df[to_test_df['pass_prefilter']]['sender_receiver_pair']))
    relevance_df['pair_key'] =  list(zip(relevance_df['interacting_pair'], relevance_df['sender_receiver_pair']))
    filtered_df = relevance_df.loc[relevance_df['pair_key'].isin(to_test_pairs), :]
    print(f"Number of pairs to test after prefilter by Fisher's exact test with pval cutoff {fisher_pval_cutoff}: {len(to_test_pairs)}")
    
    grouped = filtered_df.groupby(['interacting_pair', 'sender_receiver_pair'])
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_group)(name[0], name[1], group)
        for name, group in tqdm(grouped, total=len(grouped))
    )
    summary_df = pd.DataFrame(results)
    summary_df['glmm_fdr'] = np.nan
    mask = ~summary_df['glmm_pval'].isna()
    if mask.sum() > 0:
        _, qvals, _, _ = fdrcorrection_twostage(summary_df.loc[mask, 'glmm_pval'], method='bh')
        summary_df.loc[mask, 'glmm_fdr'] = qvals
    summary_df = summary_df.merge(to_test_df.drop(columns=['pass_prefilter']), on=['interacting_pair', 'sender_receiver_pair'], how='left')
    return summary_df

#--------------------------------
liana_df = pd.read_csv("/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/results/tables/liana_df.csv")
metadata_df = pd.read_csv('/data1/chanj3/HTA.lung.NE_plasticity.120122/ref/metadata_for_limma.csv')
metadata_df = metadata_df.rename(columns={'batch': 'sample',
                                        'patient': 'subject'})
metadata_df = metadata_df.loc[:, ['sample', 'subject', 'tissue', 'chemo', 'IO', 'TKI']]
print(liana_df.shape)
summary = summarize_lr_interactions(liana_df, metadata_df, sample_groups, threshold=0.5, 
                                    sample_column='sample', min_confident=5, fisher_pval_cutoff=0.5)
summary = summary.dropna(subset=['glmm_pval'])
information = liana_df.loc[:, ['interacting_pair', 'sender_receiver_pair', 'interaction_type', 'classification', 'active_TF']]
information['active_TF'] = information['active_TF'].fillna('')
information = information.rename(columns={'active_TF': 'active_TFs'})
information = information.groupby(['interacting_pair', 'sender_receiver_pair', 'interaction_type', 'classification']).agg({'active_TFs': lambda x: ';'.join(filter(None, x.unique()))})
information = information.reset_index().drop_duplicates()
summary = summary.merge(information, on=['interacting_pair', 'sender_receiver_pair'], how='left')
summary.to_csv("/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/results/tables/liana_df_glmm.csv", index=False)

#--------------------------------
# plot within tumor interactions
standard_class_palette = generate_class_palette(summary['classification'].unique())
top_n = 100
glmm_fdr_threshold = 0.2
within_tumor_ordering = [
    "NSCLC",
    "SCLC-A",
    "SCLC-N",
]
interaction_type = 'within_tumor'
type_summary = summary[summary['interaction_type'] == interaction_type]
type_summary['active_TFs'] = type_summary['active_TFs'].fillna('')
row_order = ['VEGFA_NRP1','APP_TNFRSF21', 'APP_CD74', 
'JAG1_NOTCH2','CNTN1_NOTCH2', 'DLK1_NOTCH2', 'DLL1_NOTCH2','DLL3_NOTCH2', 
 'DLK1_NOTCH1','DLL3_NOTCH1', 'DLL1_NOTCH1',     'JAG2_NOTCH1', 'JAG1_NOTCH1', 'PGF_NRP1',
 'WNT4_FZD1_LRP5','WNT4_FZD5_LRP5','WNT4_FZD6_LRP5',  'WNT4_FZD3_LRP5','SEMA4D_PLXNB1', 
        'EFNA5_EPHB2', 'EFNB1_EPHB2', 'EFNB1_EPHB4',
       'EFNB3_EPHB2', 'EFNB3_EPHB4',  'EFNB1_EPHB2',
       ]
plot_interaction_heatmap(
    type_summary,
    standard_class_palette,
    output_file=f'../results/figures/liana_glmm_within_tumor_interactions_heatmap_sender_no_TF_names.png',
    top_n=top_n,
    pval_threshold=glmm_fdr_threshold,
    figsize=(10, 8),
    row_order=row_order,
    facet_by='sender',
    facet_order=within_tumor_ordering,
    show_TF_names=False,
    pval_column='glmm_fdr'
)

#--------------------------------
# plot tumor stromal interactions
interaction_type = 'tumor_stromal'
tumor_stromal_ordering = [
    "SCLC-A",
    "SCLC-N",
    "Endothelial",
    "Fibroblast",
]
type_summary = summary[summary['interaction_type'] == interaction_type]
type_summary['active_TFs'] = type_summary['active_TFs'].fillna('')
row_order = ['VEGFA_NRP1', 'VEGFA_NRP2', 'VEGFB_NRP1', 'PGF_NRP1', 'PGF_NRP2',
         'APP_CD74', 'CXCL12_CXCR4', 'TNFSF12_TNFRSF12A',
         'EFNB1_EPHB2', 'EFNB2_EPHB2', 'EFNB1_EPHB4',
       'JAG1_NOTCH1',  'JAG1_NOTCH3', 'JAG1_NOTCH2',
       'JAG2_NOTCH1', 'JAG2_NOTCH2', 
       'SEMA4C_PLXNB2', 'SEMA4D_PLXNB2', 
       ]
plot_interaction_heatmap(
    type_summary,
    standard_class_palette,
    output_file=f'../results/figures/liana_glmm_tumor_stromal_interactions_heatmap_sender_no_TF_names.png',
    top_n=top_n,
    pval_threshold=glmm_fdr_threshold,
    figsize=(10, 8),
    row_order=row_order,
    facet_by='sender',
    facet_order=tumor_stromal_ordering,
    show_TF_names=False,
    pval_column='glmm_fdr'
)
#--------------------------------
# plot tumor immune interactions
interaction_type = 'tumor_immune'
tumor_immune_ordering = [
    "SCLC-A",
    "SCLC-N",
]
type_summary = summary[summary['interaction_type'] == interaction_type]
type_summary['active_TFs'] = type_summary['active_TFs'].fillna('')
row_order = ['NECTIN2_TIGIT', 'DLL3_NOTCH2', 'DLL4_NOTCH1', 'DLL1_NOTCH2', 'DLL1_NOTCH3',
    'JAG2_NOTCH1','JAG1_NOTCH1', 'JAG1_NOTCH2', 'JAG1_NOTCH3',  'DLK1_NOTCH2', 'CNTN1_NOTCH2', 'JAG2_NOTCH3', 'DLK1_NOTCH1', 
    'SEMA4D_PLXNB2',
    'TNFSF12_TNFRSF12A', 'TNF_TNFRSF1A', 'TNF_TNFRSF1B','TNFSF14_LTBR', 
     'CXCL14_CXCR4', 'APP_CD74',
       'PODXL2_SELL','CDH1_KLRG1', 
       ]
plot_interaction_heatmap(
    type_summary,
    standard_class_palette,
    output_file=f'../results/figures/liana_glmm_tumor_immune_interactions_heatmap_sender_no_TF_names.png',
    top_n=top_n,
    pval_threshold=glmm_fdr_threshold,
    figsize=(10, 8),
    row_order=row_order,
    facet_by='sender',
    facet_order=tumor_immune_ordering,
    show_TF_names=False,
    pval_column='glmm_fdr'
)
plot_interaction_heatmap(
    type_summary,
    standard_class_palette,
    output_file=f'../results/figures/liana_glmm_tumor_immune_interactions_heatmap_receiver_no_TF_names.png',
    top_n=top_n,
    pval_threshold=glmm_fdr_threshold,
    figsize=(10, 8),
    show_TF_names=False,
    row_order=row_order,
    facet_by='receiver',
    facet_order=tumor_immune_ordering,
    pval_column='glmm_fdr'
)