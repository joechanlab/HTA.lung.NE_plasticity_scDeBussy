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

    class_palette = {
        cls: to_hex(color) for cls, color in zip(unique_classes, palette_colors)
    }

    return class_palette

def aggregate_subjects(relevance_df, score_column):
    # Aggregate bio or technical replicates
    relevance_df['transformed_score'] = -np.log10(
        relevance_df[score_column].clip(lower=1e-10)
    )
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
        'transformed_score': 'mean'
    }).reset_index()
    return agg

def prefilter_group_for_mixedlm(
    group_df,
    min_subjects_per_condition=3,
    min_total_subjects=6,
    min_variance=1e-3
):
    """
    Prefilter individual ligand-receptor groups before mixed model fitting.
    """
    group_valid = group_df.dropna(subset=['transformed_score'])

    total_subjects = group_valid['subject'].nunique()
    if total_subjects < min_total_subjects:
        return False

    variance = group_valid['transformed_score'].var()
    if variance is None or variance < min_variance:
        return False

    condition_counts = group_valid.groupby('condition')['subject'].nunique()
    if any(condition_counts < min_subjects_per_condition):
        return False

    return True

def process_group(interacting_pair, sender_receiver_pair, group, min_subjects_per_condition, min_total_subjects, min_variance):
    try:
        group = group.copy()
        group = group.dropna(subset=['transformed_score'])

        # Apply prefilter before model fitting
        if not prefilter_group_for_mixedlm(group, min_subjects_per_condition, min_total_subjects, min_variance):
            return {
                'interacting_pair': interacting_pair,
                'sender_receiver_pair': sender_receiver_pair,
                'glmm_pval': pd.NA,
                'mean_diff': pd.NA
            }

        # Compute mean difference directly
        mean_adc = group[group['condition'] == 'ADC → SCLC']['transformed_score'].mean()
        mean_de_novo = group[group['condition'] == 'De novo SCLC and ADC']['transformed_score'].mean()
        mean_diff = mean_adc - mean_de_novo

        for col in ['condition', 'tissue', 'chemo', 'IO', 'TKI']:
            if col == 'condition':
                group[col] = pd.Categorical(
                    group[col], categories=['De novo SCLC and ADC', 'ADC → SCLC']
                )
            else:
                group[col] = pd.Categorical(group[col])

        formula = "transformed_score ~ condition + tissue + chemo + IO + TKI"
        model = smf.mixedlm(formula, data=group, groups=group['subject'])
        result = model.fit()
        pval = result.pvalues.get('condition[T.ADC → SCLC]', pd.NA)
    except Exception:
        pval = pd.NA
        mean_diff = pd.NA

    return {
        'interacting_pair': interacting_pair,
        'sender_receiver_pair': sender_receiver_pair,
        'glmm_pval': pval,
        'mean_diff': mean_diff
    }

def compute_frequency_difference(subject_df, sample_groups, threshold_value=2.0):
    """
    Adds frequency difference column (analogous to old freq_diff) to summary dataframe.

    Parameters:
    - subject_df: aggregated subject-level dataframe (after aggregation step)
    - sample_groups: dictionary of {condition: list of subject IDs}
    - threshold_value: threshold on transformed_score to consider confident interaction

    Returns:
    - DataFrame with additional freq_diff column per (interacting_pair, sender_receiver_pair)
    """

    subject_df = subject_df.copy()
    subject_df['is_confident'] = (subject_df['transformed_score'] >= threshold_value).astype(int)

    results = []
    grouped = subject_df.groupby(['interacting_pair', 'sender_receiver_pair'])
    for (interacting_pair, sender_receiver_pair), group in grouped:
        n_A = group[group['condition'] == 'De novo SCLC and ADC']['is_confident'].sum()
        total_A = len(sample_groups['De novo SCLC and ADC'])
        freq_A = n_A / total_A if total_A > 0 else np.nan

        n_B = group[group['condition'] == 'ADC → SCLC']['is_confident'].sum()
        total_B = len(sample_groups['ADC → SCLC'])
        freq_B = n_B / total_B if total_B > 0 else np.nan

        freq_diff = freq_B - freq_A

        results.append({
            'interacting_pair': interacting_pair,
            'sender_receiver_pair': sender_receiver_pair,
            'freq_A': freq_A,
            'freq_B': freq_B,
            'freq_diff': freq_diff
        })

    return pd.DataFrame(results)

def summarize_lr_interactions(
    relevance_df,
    metadata_df,
    sample_groups,
    score_column='computed_specificity_rank',
    threshold_value=2.0,
    n_jobs=4,
    min_subjects_per_condition=3,
    min_total_subjects=6,
    min_variance=1e-3
):
    relevance_df['sender_receiver_pair'] = (
        relevance_df['sender'] + '→' + relevance_df['receiver']
    )
    relevance_df['active_TFs'] = relevance_df['active_TF'].fillna('').astype(str)
    relevance_df = relevance_df.merge(
        metadata_df, on=['sample', 'subject'], how='left'
    )
    print(f"Input shape: {relevance_df.shape}")

    subject_df = aggregate_subjects(relevance_df, score_column)
    frequency_df = compute_frequency_difference(subject_df, sample_groups, threshold_value=threshold_value)
    grouped = subject_df.groupby(['interacting_pair', 'sender_receiver_pair'])
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_group)(name[0], name[1], group, min_subjects_per_condition, min_total_subjects, min_variance)
        for name, group in tqdm(grouped, total=len(grouped))
    )
    
    summary_df = pd.DataFrame(results)
    summary_df['glmm_fdr'] = np.nan
    mask = ~summary_df['glmm_pval'].isna()
    if mask.sum() > 0:
        _, qvals, _, _ = fdrcorrection_twostage(
            summary_df.loc[mask, 'glmm_pval'], method='bh'
        )
        summary_df.loc[mask, 'glmm_fdr'] = qvals
    summary_df = summary_df.merge(
        frequency_df,
        on=['interacting_pair', 'sender_receiver_pair'],
        how='left'
    )
    return summary_df

# --------------------------------
liana_df = pd.read_csv(
    "/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/results/tables/liana_df.csv"
)
metadata_df = pd.read_csv(
    '/data1/chanj3/HTA.lung.NE_plasticity.120122/ref/metadata_for_limma.csv'
)
metadata_df = metadata_df.rename(columns={'batch': 'sample', 'patient': 'subject'})
metadata_df = metadata_df.loc[:, [
    'sample', 'subject', 'tissue', 'chemo', 'IO', 'TKI']]

min_subjects_per_condition=2
min_total_subjects=8
min_variance=1e-3
threshold_value=-np.log10(0.2)

summary = summarize_lr_interactions(
    liana_df,
    metadata_df,
    sample_groups,
    threshold_value=threshold_value,
    min_subjects_per_condition=min_subjects_per_condition,
    min_total_subjects=min_total_subjects,
    min_variance=min_variance
)
summary = summary.dropna(subset=['glmm_pval'])

information = liana_df.loc[:, [
    'interacting_pair', 'sender_receiver_pair', 'interaction_type',
    'classification', 'active_TF']]
information['active_TF'] = information['active_TF'].fillna('')
information = information.rename(columns={'active_TF': 'active_TFs'})
information = information.groupby([
    'interacting_pair', 'sender_receiver_pair', 'interaction_type', 'classification'
]).agg({
    'active_TFs': lambda x: ';'.join(filter(None, x.unique()))
})
information = information.reset_index().drop_duplicates()
summary = summary.merge(
    information,
    on=['interacting_pair', 'sender_receiver_pair'],
    how='left'
)
summary.to_csv(
    "/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/results/tables/liana_df_glmm.csv",
    index=False
)

#--------------------------------
# plot within tumor interactions
top_n = 20
glmm_fdr_threshold = 0.2
summary = summary[summary['glmm_fdr'] < glmm_fdr_threshold]
standard_class_palette = generate_class_palette(summary['classification'].unique())
within_tumor_ordering = [
    "NSCLC",
    "SCLC-A",
    "SCLC-N",
]
interaction_type = 'within_tumor'
type_summary = summary[summary['interaction_type'] == interaction_type]
plot_interaction_heatmap(
    type_summary,
    standard_class_palette,
    output_file=None, #f'../results/figures/liana_glmm_within_tumor_interactions_heatmap_sender_no_TF_names.png',
    top_n=top_n,
    pval_threshold=glmm_fdr_threshold,
    figsize=(10, 8),
    row_order=None, #row_order,
    facet_by='sender',
    facet_order=within_tumor_ordering,
    show_TF_names=False,
    pval_column='glmm_fdr',
    ranking_diff_column='freq_diff',
    viz_diff_column='freq_diff'
)

#--------------------------------
# plot tumor immune interactions
interaction_type = 'tumor_immune'
tumor_immune_ordering = [
    "SCLC-A",
    "SCLC-N",
]
type_summary = summary[summary['interaction_type'] == interaction_type]
plot_interaction_heatmap(
    type_summary,
    standard_class_palette,
    output_file=None, #f'../results/figures/liana_glmm_tumor_immune_interactions_heatmap_sender_no_TF_names.png',
    top_n=top_n,
    pval_threshold=glmm_fdr_threshold,
    figsize=(10, 8),
    row_order=None, #row_order,
    facet_by='sender',
    facet_order=tumor_immune_ordering,
    show_TF_names=False,
    pval_column='glmm_fdr',
    ranking_diff_column='freq_diff',
    viz_diff_column='freq_diff'
)
plot_interaction_heatmap(
    type_summary,
    standard_class_palette,
    output_file=None, #f'../results/figures/liana_glmm_tumor_immune_interactions_heatmap_receiver_no_TF_names.png',
    top_n=top_n,
    pval_threshold=glmm_fdr_threshold,
    figsize=(10, 8),
    show_TF_names=False,
    row_order=None, #row_order,
    facet_by='receiver',
    facet_order=tumor_immune_ordering,
    pval_column='glmm_fdr',
    ranking_diff_column='freq_diff',
    viz_diff_column='freq_diff'
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
if len(type_summary) > 0:
    plot_interaction_heatmap(
        type_summary,
        standard_class_palette,
        output_file=None, #f'../results/figures/liana_glmm_tumor_stromal_interactions_heatmap_sender_no_TF_names.png',
        top_n=top_n,
        pval_threshold=glmm_fdr_threshold,
        figsize=(10, 8),
        row_order=None, #row_order,
        facet_by='sender',
        facet_order=tumor_stromal_ordering,
        show_TF_names=False,
        pval_column='glmm_fdr',
        ranking_diff_column='freq_diff',
        viz_diff_column='mean_diff'
    )
