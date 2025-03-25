import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scdebussy.tl import process_gene_data, perform_sliding_window_enrichr_analysis, analyze_and_plot_enrichment, enrichr
from scdebussy.pl import fit_kde, plot_kde_density_ridge, plot_kde_heatmap, plot_kshape_clustering, plot_summary_curve
import gseapy as gp
import seaborn as sns
from matplotlib.colors import LogNorm
import pickle
import pandas as pd

import pandas as pd
import numpy as np

def compute_patient_fraction(summary_df, df_right, col_names, smoothing_window=5, quantile_threshold=0.75, min_timepoints=2):
    """
    Compute a recurrence score for each (gene, gene set) pair, outputting both actual and smoothed scores.

    Parameters:
    - summary_df: DataFrame with ['subject', 'aligned_score', 'expression', 'gene']
    - df_right: Series mapping genes to gene sets
    - col_names: List of gene sets of interest
    - smoothing_window: Rolling mean window for smoothing
    - quantile_threshold: Quantile to determine the expression threshold per gene
    - min_timepoints: Minimum pseudotime points where a gene must be expressed to count for a patient

    Returns:
    - actual_df: DataFrame with genes as index and gene sets as columns, containing actual recurrence scores.
    - smoothed_df: DataFrame with smoothed recurrence scores for visualization.
    """
    # Expand df_right so each gene maps to multiple gene sets
    gene_to_sets = df_right.dropna().str.split(';').explode().reset_index()
    gene_to_sets.columns = ['gene', 'gene_set']

    # Filter for relevant gene sets
    gene_to_sets = gene_to_sets[gene_to_sets['gene_set'].isin(col_names)]

    # Filter summary_df to include only genes in df_right
    filtered_df = summary_df[summary_df['gene'].isin(df_right.index)]

    # Compute expression threshold per gene (75th percentile instead of median)
    gene_thresholds = filtered_df.groupby('gene')['expression'].quantile(quantile_threshold)

    # Assign threshold-based expression detection per gene
    filtered_df['threshold'] = filtered_df['gene'].map(gene_thresholds)
    filtered_df['expressed'] = filtered_df['expression'] > filtered_df['threshold']

    # Require expression in at least `min_timepoints` pseudotime points for a patient to count
    patient_counts = (
        filtered_df.groupby(['gene', 'subject'])['expressed']
        .sum()  # Count number of time points where expression exceeds threshold
        .ge(min_timepoints)  # Keep only patients where gene is expressed in at least `min_timepoints`
        .groupby('gene')
        .mean()  # Fraction of patients expressing the gene
    )

    # Merge patient fractions with gene sets
    result_df = gene_to_sets.merge(patient_counts, on='gene', how='left')

    # Pivot to get genes as rows and gene sets as columns
    actual_df = result_df.pivot(index='gene', columns='gene_set', values='expressed')

    # Ensure all gene sets are present as columns
    actual_df = actual_df.reindex(columns=col_names, fill_value=0)

    # Ensure the output index matches df_right.index exactly
    actual_df = actual_df.reindex(df_right.index, fill_value=0)

    # Apply smoothing across genes for visualization
    smoothed_df = actual_df.rolling(window=smoothing_window, min_periods=1, axis=0).mean()

    return actual_df, smoothed_df

# input
gene_set_path = "/data1/chanj3/HTA.lung.NE_plasticity.120122/ref/Spectra.NE_NonNE.gene_sets.120524.p"
spectra_global_gene_sets = pickle.load(open(gene_set_path, "rb"))['global']

os.chdir("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/results_all_genes/scDeBussy")
summary_df_path = "NSCLC_SCLC-A/NSCLC_SCLC-A_summary_df.csv"
aggregated_curves_path = "NSCLC_SCLC-A/NSCLC_SCLC-A_aggregated_curves.csv"
scores_df_path = "NSCLC_SCLC-A/NSCLC_SCLC-A_scores_df.csv"
hvg_df_path = "NSCLC_SCLC-A/hvg_genes_NSCLC_SCLC-A.txt"
cluster_ordering = "NSCLC_SCLC-A"
weight = 0.5
summary_df = pd.read_csv(summary_df_path, index_col=0)
gene_curves = pd.read_csv(aggregated_curves_path, index_col=0)
scores_df = pd.read_csv(scores_df_path, index_col=0)
hvg_df = pd.read_csv(hvg_df_path, sep="\t")
hvg_df.columns = ["gene", "rank", "batches"]
clusters = cluster_ordering.split("_")
color_map = {'NSCLC': 'gold',
             'SCLC-A': 'tab:red',
             'SCLC-N': 'tab:cyan'}
colors = [color_map[x] for x in clusters]
scores_df = scores_df.merge(hvg_df, on='gene')
scores_df.sort_values(['GCV'], ascending=[True]).merge(hvg_df, on='gene').head(n=50)

df = gene_curves.iloc[:,1:].T

plot_summary_curve(summary_df, gene_curves, scores_df, 
                   ['TACSTD2', 'EZH2', 'DLL3'], 
                   fig_size=(2.5, 1.5), pt_alpha=0.05)


sorted_gene_curve, row_colors, col_colors, categories = process_gene_data(scores_df, gene_curves, colors, [0.5],
                                                                            n_clusters = 5, n_init=1, MI_threshold=1.2,
                                                                            GCV_threshold=0.04, AIC_threshold=2e6, hierarchical=True,  
                                                                            label_names=['Early', 'Early', "Middle", 'Middle', 'Late'], weight=weight)
print(sorted_gene_curve.shape)
plot_kshape_clustering(sorted_gene_curve, categories, ['Early', 'Middle', 'Late'], alpha=0.03)

# sliding window plot (cell type)
ordered_genes = sorted_gene_curve.index.tolist()
window_size = 150
stride = 50
gene_set_library = "CellMarker_2024"
immune_related_gene_sets = ['Immune Cell', 'T Cell', "T Helper", 'Regulatory T', 'B Cell', 'NK Cell', 'Dendritic Cell', 'Macrophage', 'Neutrophil', 'Eosinophil', 'Basophil', 'Monocyte', 'Mast Cell', 'Myeloid', 'Lymphoid', 'Phagocyte', 'Antigen', 'Plasma']
results = perform_sliding_window_enrichr_analysis(ordered_genes, window_size, stride, gene_set_library)
analyze_and_plot_enrichment(results, exclude_gene_sets=immune_related_gene_sets, save_path="/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/cell_type_enriched_pathway.png")

gene_set_library = "/data1/chanj3/HTA.lung.NE_plasticity.120122/ref/curated.small.100124.gmt"
or_threshold=10
results = perform_sliding_window_enrichr_analysis(ordered_genes, window_size, stride, gene_set_library)
analyze_and_plot_enrichment(results, or_threshold=or_threshold, exclude_gene_sets=immune_related_gene_sets, save_path="/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/pathway_enriched_pathway.png")

# Cell type gene set annotation
gene_info = pd.DataFrame({'gene': sorted_gene_curve.index, 'category': categories})
gene_sets = ['CellMarker_2024']
results = pd.DataFrame()

for category in gene_info.category.unique():
    gene_list = gene_info.gene[gene_info.category == category]
    results_category = enrichr(gene_list, gene_sets)
    results_category.loc[:,'category'] = category
    results = pd.concat([results, results_category])
    
results.loc[:,'overlap'] = results.Overlap.apply(lambda x: int(x.split('/')[0]))
results = results[(results['Adjusted P-value'] < 0.05) & (results['overlap'] > 10)]
results = results.drop_duplicates(subset="Genes")
pd.set_option('display.max_rows', None)
gene_sets = results.sort_values(by=["category", "Odds Ratio"], ascending = [True, False]).reset_index(drop=True)
gene_sets.to_csv("NSCLC_SCLC-A_enrichR.csv")
gene_sets.category = pd.Categorical(gene_sets.category, ordered=True, categories=['Early', 'Middle', 'Late'])
gene_sets.groupby('category').apply(lambda x: x.sort_values(by='Combined Score', ascending=False).head(20)).loc[:,['Term', 'category', 'Odds Ratio', 'Adjusted P-value', 'Overlap', 'Genes']].reset_index(drop=True)

genes = gene_sets['Genes'].str.split(';').explode().str.strip().unique()
gene_to_terms = {}
# Cell type gene sets
for index, row in gene_sets.iterrows():
    term = row['Term']
    genes = row['Genes'].split(';')
    combined_score = row['Combined Score']
    
    for gene in genes:
        if gene not in gene_to_terms:
            gene_to_terms[gene] = {'term': term}
        else:
            gene_to_terms[gene]['term'] = gene_to_terms[gene]['term'] + ';' + term

# Iterate through the spectra_global_gene_sets and add the gene sets to the gene_to_terms dictionary
for gene_set in spectra_global_gene_sets:
    genes = spectra_global_gene_sets[gene_set]
    for gene in genes:
        if gene not in gene_to_terms:
            gene_to_terms[gene] = {'term': gene_set}
        else:
            gene_to_terms[gene]['term'] = gene_to_terms[gene]['term'] + ';' + gene_set

final_gene_to_term = {gene: info['term'] for gene, info in gene_to_terms.items()}

sorted_gene_curve_with_annot = sorted_gene_curve.copy()
sorted_gene_curve_with_annot.loc[:,'gene_sets'] = sorted_gene_curve_with_annot.index.map(final_gene_to_term)
sorted_gene_curve_with_annot.loc[:,'cluster'] = row_colors.values

cell_type_colors = {
    'NSCLC': 'gold', 'SCLC-A': 'tab:red'
}
cell_types = col_colors.map(lambda x: 'NSCLC' if x == 0 else 'SCLC-A')
clusters = row_colors
cluster_colors = {}
for i, category in enumerate(gene_info['category'].unique()):
    cluster_colors[category] = plt.get_cmap('Dark2')(i / len(gene_info['category'].unique()))

df_right = sorted_gene_curve_with_annot['gene_sets']
df_right.index = sorted_gene_curve.index
df_left = sorted_gene_curve_with_annot['cluster']
df_left.index = sorted_gene_curve.index

density=None
left_annotation_columns=None
plot_kde_heatmap(cluster_colors, cell_types, cell_type_colors, sorted_gene_curve, df_left, density, figsize=(3,8), left_annotation_columns=left_annotation_columns, 
                 vmin=-3, vmax=3, save_path=f"/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/NSCLC_SCLC-A_heatmap_weight_{weight}.png")

# Cell type
names = ['Secretory', 'Basal', 'Cycling Basal', 
         'Neural Progenitor',  "Neuroendocrine"] 
col_names = ['Secretory Cell Lung Human', 'Basal Cell Lung Human',
             'Cycling Basal Cell Trachea Mouse', 'Neural Progenitor Cell Embryonic Prefrontal Cortex Human', 'Neuroendocrine Cell Trachea Mouse'] #'Mesenchymal Stem Cell Undefined Human', 
cmap_names = ['early', 'early', 'early', 'middle', 'middle', 'late', 'late']
density = fit_kde(sorted_gene_curve, df_right, col_names, bandwidth=200)
scores_actual, scores_smoothed = compute_patient_fraction(summary_df, df_right, col_names, smoothing_window=50)
scores_actual.columns = names
scores_smoothed.columns = names
plot_kde_density_ridge(density.copy(), 
                       clusters = clusters, 
                       scores = scores_actual,
                       cluster_colors = cluster_colors,
                       name_map = dict(zip(col_names, names)), 
                       cell_type_order=names, figsize=(6, 4), fontsize=10, save_path='/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/NSCLC_SCLC-A_cell_type.png')
plot_kde_density_ridge(density.copy(), 
                       clusters = clusters, 
                       scores = scores_smoothed,
                       cluster_colors = cluster_colors,
                       name_map = dict(zip(col_names, names)), 
                       cell_type_order=names, figsize=(6, 4), fontsize=10, save_path='/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/NSCLC_SCLC-A_cell_type_smoothed.png')


# Global pathway
names = ['NFkB', "JAK-STAT3", "PI3K-AKT", 'MYC',  'DNA repair']
col_names = ['all_TNF-via-NFkB_signaling', 'all_IL6-JAK-STAT3_signaling', 'all_PI3K-AKT-mTOR_signaling',  'all_MYC_targets', 'all_DNA-repair']
density = fit_kde(sorted_gene_curve, df_right, col_names, bandwidth=200)
# randomly simulate the scores for each gene
scores_actual, scores_smoothed = compute_patient_fraction(summary_df, df_right, col_names, smoothing_window=50)
scores_actual.columns = names
scores_smoothed.columns = names
plot_kde_density_ridge(density.copy(), 
                       clusters = clusters, 
                       scores = scores_actual,
                       cluster_colors = cluster_colors,
                       name_map = dict(zip(col_names, names)), 
                       cell_type_order=names, figsize=(6, 4), fontsize=10, save_path='/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/NSCLC_SCLC-A_global_pathway.png')
plot_kde_density_ridge(density.copy(), 
                       clusters = clusters, 
                       scores = scores_smoothed,
                       cluster_colors = cluster_colors,
                       name_map = dict(zip(col_names, names)), 
                       cell_type_order=names, figsize=(6, 4), fontsize=10, save_path='/home/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/NSCLC_SCLC-A_global_pathway_smoothed.png')
