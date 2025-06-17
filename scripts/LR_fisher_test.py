import os
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
from constants import sample_groups, sample_to_group, tumor_color_map

FILTER_CONTROL = False

def perform_fisher_test(relevance_long_df, cell_sign_long_df=None, cond1="De novo SCLC and ADC", cond2="ADC → SCLC", alternative='two-sided'):
    """
    Perform Fisher's exact test on ligand-receptor interactions between two conditions.
    
    Parameters:
    -----------
    relevance_long_df : pandas.DataFrame
        DataFrame containing the relevance data with columns:
        - interacting_pair
        - sender
        - receiver
        - condition
        - value
        - subject
    cell_sign_long_df : pandas.DataFrame, optional
        DataFrame containing TF activity information with columns:
        - interacting_pair
        - subject
        - sender
        - receiver
        - active_TF
    cond1, cond2 : str
        Names of the two conditions to compare
    alternative : str
        The alternative hypothesis to test. Options:
        - 'two-sided': Test for any difference between conditions (default)
        - 'greater': Test if cond2 has higher frequency than cond1
        - 'less': Test if cond2 has lower frequency than cond1
        
    Returns:
    --------
    pandas.DataFrame
        Results of the Fisher's exact test with columns:
        - interacting_pair
        - sender_receiver_pair
        - sender
        - receiver
        - odds_ratio
        - p_value
        - p_adj
        - cond1_present
        - cond1_absent
        - cond2_present
        - cond2_absent
        - cond1_frequency
        - cond2_frequency
        - active_TFs (if cell_sign_long_df provided)
        - TF_activity_frequency (if cell_sign_long_df provided)
    """
    # Get total number of samples per condition
    total_samples = {
        cond1: len(relevance_long_df[relevance_long_df['condition'] == cond1]['subject'].unique()),
        cond2: len(relevance_long_df[relevance_long_df['condition'] == cond2]['subject'].unique())
    }
    
    print(f"Total samples in {cond1}: {total_samples[cond1]}")
    print(f"Total samples in {cond2}: {total_samples[cond2]}")

    # Use relevance_long_df to create proper contingency tables
    fisher_results = []
    for (interacting_pair, sender, receiver), group in relevance_long_df.groupby(['interacting_pair', 'sender', 'receiver']):
        # Count samples with interaction (value=1) for each condition
        cond1_present = group[(group['condition'] == cond1) & (group['value'] == 1)]['subject'].nunique()
        cond2_present = group[(group['condition'] == cond2) & (group['value'] == 1)]['subject'].nunique()
        
        # Calculate absent counts (total samples - present samples)
        cond1_absent = total_samples[cond1] - cond1_present
        cond2_absent = total_samples[cond2] - cond2_present
        
        # Create contingency table
        contingency = pd.DataFrame({
            0: [cond1_absent, cond2_absent],
            1: [cond1_present, cond2_present]
        }, index=[cond1, cond2])
        
        # Perform Fisher's exact test with specified alternative
        odds_ratio, p_value = fisher_exact(contingency, alternative=alternative)
        
        # Get the sender-receiver pair
        sender_receiver_pair = f"{sender}→{receiver}"
        
        # Calculate frequencies
        cond1_freq = cond1_present / total_samples[cond1]
        cond2_freq = cond2_present / total_samples[cond2]
        
        # Create column names with actual condition names
        present_col1 = f"{cond1}_present"
        absent_col1 = f"{cond1}_absent"
        present_col2 = f"{cond2}_present"
        absent_col2 = f"{cond2}_absent"
        freq_col1 = f"{cond1}_frequency"
        freq_col2 = f"{cond2}_frequency"
        
        result_dict = {
            'interacting_pair': interacting_pair,
            'sender_receiver_pair': sender_receiver_pair,
            'sender': sender,
            'receiver': receiver,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            present_col1: cond1_present,
            absent_col1: cond1_absent,
            present_col2: cond2_present,
            absent_col2: cond2_absent,
            freq_col1: cond1_freq,
            freq_col2: cond2_freq
        }
        
        # Add TF activity information if available
        if cell_sign_long_df is not None:
            # Filter cell_sign_long_df for this interaction
            tf_info = cell_sign_long_df[
                (cell_sign_long_df['interacting_pair'] == interacting_pair) &
                (cell_sign_long_df['sender'] == sender) &
                (cell_sign_long_df['receiver'] == receiver)
            ]
            
            if not tf_info.empty:
                # Get unique TFs
                active_tfs = tf_info['active_TF'].unique()
                result_dict['active_TFs'] = ';'.join(active_tfs)
                
                # Calculate TF activity frequency (proportion of samples with TF activity)
                tf_activity_freq = len(tf_info['subject'].unique()) / len(relevance_long_df['subject'].unique())
                result_dict['TF_activity_frequency'] = tf_activity_freq
            else:
                result_dict['active_TFs'] = 'None'
                result_dict['TF_activity_frequency'] = 0.0
        
        fisher_results.append(result_dict)

    # Convert results to DataFrame
    fisher_df = pd.DataFrame(fisher_results)

    # Adjust p-values for multiple testing within each sender-receiver pair
    fisher_df['p_adj'] = np.nan  # Initialize column
    for sender_receiver in fisher_df['sender_receiver_pair'].unique():
        # Get indices for this sender-receiver pair
        mask = fisher_df['sender_receiver_pair'] == sender_receiver
        # Perform multiple testing correction only on this subset
        fisher_df.loc[mask, 'p_adj'] = multipletests(
            fisher_df.loc[mask, 'p_value'], 
            method='fdr_bh'
        )[1]
    
    # Sort by original p-value for easier interpretation
    fisher_df = fisher_df.sort_values('p_value')

    return fisher_df

def plot_significant_interactions(fisher_df, sig_threshold=0.05, pvalue_column='p_adj', value_column='odds_ratio', output_file='significant_lr_interactions_heatmap.png', top_n=30, figsize=(10, 7)):
    """
    Create a heatmap of significant ligand-receptor interactions.
    If no significant hits are found, use top N pairs based on p-value.
    
    Parameters:
    -----------
    fisher_df : pandas.DataFrame
        Results from perform_fisher_test
    sig_threshold : float
        Significance threshold for p-values
    pvalue_column : str
        Column name to use for filtering significant interactions (default: 'p_adj')
    value_column : str
        Column to use for heatmap values ('odds_ratio' or 'p_value')
    output_file : str
        Path to save the heatmap
    top_n : int
        Number of top interactions to plot when no significant hits are found (default: 30)
    """
    # Validate input
    if pvalue_column not in fisher_df.columns:
        raise ValueError(f"Column '{pvalue_column}' not found in DataFrame. Available columns: {fisher_df.columns.tolist()}")
    if value_column not in fisher_df.columns:
        raise ValueError(f"Column '{value_column}' not found in DataFrame. Available columns: {fisher_df.columns.tolist()}")
    
    # Filter for significant results
    sig_fisher = fisher_df[fisher_df[pvalue_column] < sig_threshold].copy()
    
    # If no significant results, use top N based on p-value
    if sig_fisher.empty:
        print(f"No significant interactions found using {pvalue_column} < {sig_threshold}!")
        print(f"Using top {top_n} interactions based on p-value instead.")
        sig_fisher = fisher_df.nsmallest(top_n, pvalue_column).copy()
    
    # Sort by the specified p-value column
    sig_fisher = sig_fisher.sort_values(pvalue_column)
    
    # Save results
    sig_fisher.to_csv('significant_lr_interactions.csv', index=False)
    
    # Print summary of interactions per sender-receiver pair
    print(f"\nInteractions per sender-receiver pair:")
    summary = sig_fisher.groupby('sender_receiver_pair').size()
    print(summary)
    
    # Print summary of TF activity in interactions
    if 'active_TFs' in sig_fisher.columns:
        print("\nTF activity in interactions:")
        tf_summary = sig_fisher[sig_fisher['active_TFs'] != 'None'].groupby('active_TFs').size()
        print(tf_summary)
    
    # Prepare data for heatmap
    try:
        if value_column == 'odds_ratio':
            # Handle infinite odds ratios
            sig_fisher['plot_value'] = sig_fisher[value_column].replace([np.inf, -np.inf], np.nan)
            max_finite = sig_fisher['plot_value'].max()
            sig_fisher['plot_value'] = sig_fisher['plot_value'].fillna(max_finite * 2)
            vmax = 10  # Cap at 10 for better visualization
            center = 1
            cmap = 'RdBu_r'
            cbar_label = 'Odds Ratio'
        else:  # p-value
            sig_fisher['plot_value'] = -np.log10(sig_fisher[value_column])  # Transform p-values to -log10
            vmax = sig_fisher['plot_value'].max()
            center = None
            cmap = 'Spectral_r'  # Yellow to Orange to Red
            cbar_label = '-log10(p-value)'
        
        # Create TF annotation
        if 'active_TFs' in sig_fisher.columns:
            # Create annotation with actual TF names
            sig_fisher['tf_annotation'] = sig_fisher['active_TFs'].apply(
                lambda x: x if x != 'None' else ''
            )
        else:
            sig_fisher['tf_annotation'] = ''
        
        heatmap_data = sig_fisher.pivot_table(
            index='interacting_pair',
            columns='sender_receiver_pair',
            values='plot_value',
            fill_value=0
        )
        
        # Create TF annotation matrix
        tf_annot_matrix = sig_fisher.pivot_table(
            index='interacting_pair',
            columns='sender_receiver_pair',
            values='tf_annotation',
            aggfunc=lambda x: ';'.join(set(filter(None, x))) if any(x) else '',
            fill_value=''
        )
        
        # Check if we have data to plot
        if heatmap_data.empty:
            print("No data available for heatmap after pivoting!")
            return
            
        print("\nHeatmap data shape:", heatmap_data.shape)
        print("Number of unique interacting pairs:", len(heatmap_data.index))
        print("Number of unique sender-receiver pairs:", len(heatmap_data.columns))
        
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        
        # Cluster rows (interacting pairs)
        row_linkage = linkage(pdist(heatmap_data.fillna(0)), method='average')
        row_order = heatmap_data.index[leaves_list(row_linkage)]
        
        # Cluster columns (sender-receiver pairs)
        col_linkage = linkage(pdist(heatmap_data.fillna(0).T), method='average')
        col_order = heatmap_data.columns[leaves_list(col_linkage)]
        
        # Reorder the data
        heatmap_data = heatmap_data.loc[row_order, col_order]
        tf_annot_matrix = tf_annot_matrix.loc[row_order, col_order]
        
        # Create figure with larger size to accommodate TF names
        plt.figure(figsize=figsize)
        
        # Create heatmap with adjusted annotation parameters
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            center=center,
            vmin=0,
            vmax=vmax,
            cbar_kws={'label': cbar_label},
            xticklabels=True,
            yticklabels=True,
            annot=tf_annot_matrix,  # Add TF annotations
            fmt='',  # Empty format string since we're using custom annotations
            annot_kws={
                'color': 'black',
                'weight': 'bold',
                'size': 8,  # Adjust font size for TF names
                'rotation': 0  # Keep TF names horizontal
            },
            mask=heatmap_data == 0  # Mask cells with value 0 to show as white
        )
        
        # Update title based on whether we're using significant or top N interactions
        if len(sig_fisher) == top_n and not any(sig_fisher[pvalue_column] < sig_threshold):
            title = f'Top {top_n} Ligand-Receptor Interactions\n(Based on {pvalue_column})'
        else:
            title = f'Significant Differential Ligand-Receptor Interactions\n(Fisher\'s Exact Test, {pvalue_column} < {sig_threshold})'
        
        #plt.title(title)
        plt.ylabel("Ligand-Receptor Pairs")
        plt.xlabel("Sender-Receiver Pairs")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save heatmap
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print information about infinite odds ratios if using odds ratio
        if value_column == 'odds_ratio':
            inf_count = (sig_fisher[value_column] == np.inf).sum()
            if inf_count > 0:
                print(f"\nNote: {inf_count} interactions had infinite odds ratios and were capped at {vmax}")
                print("These interactions had zero counts in one condition and non-zero in the other.")
        
        # Print TF activity summary
        if 'active_TFs' in sig_fisher.columns:
            tf_active_count = (sig_fisher['active_TFs'] != 'None').sum()
            print(f"\nTF Activity Summary:")
            print(f"Total interactions with TF activity: {tf_active_count}")
            print("TF activity by sender-receiver pair:")
            tf_by_pair = sig_fisher[sig_fisher['active_TFs'] != 'None'].groupby('sender_receiver_pair')['active_TFs'].apply(lambda x: ';'.join(set(x)))
            print(tf_by_pair)
        
    except Exception as e:
        print(f"Error creating heatmap: {str(e)}")
        print("\nDebug information:")
        print("DataFrame columns:", sig_fisher.columns.tolist())
        print("Number of rows:", len(sig_fisher))
        print("Sample of data:")
        print(sig_fisher.head())
        raise



if __name__ == "__main__":
    # Example usage:
    # 1. Load your relevance_long_df
    relevance_long_df = pd.read_csv('../data/processed/within_tumor_relevance.csv')
    cell_sign_long_df = pd.read_csv('../data/processed/cell_sign_long_df.csv')
    #----------------------------------
    # Filter for NonNE SCLC interactions
    nonne_sclc_interactions = relevance_long_df[
        (relevance_long_df['sender'].isin(['NonNE SCLC', 'NSCLC'])) | 
        (relevance_long_df['receiver'].isin(['NonNE SCLC', 'NSCLC']))
    ]
    nonne_sclc_interactions['sender'][nonne_sclc_interactions['sender'] == 'NonNE SCLC'] = 'NSCLC'
    nonne_sclc_interactions['receiver'][nonne_sclc_interactions['receiver'] == 'NonNE SCLC'] = 'NSCLC'
    nonne_sclc_interactions = nonne_sclc_interactions.merge(cell_sign_long_df, on=['interacting_pair', "subject", 'sender', 'receiver'], how='left')
    
    # Count number of patients per interaction and get patient list
    interaction_counts = nonne_sclc_interactions.groupby(
        ['interacting_pair', 'sender', 'receiver']
    ).agg(
        num_patients=('subject', 'nunique'),
        patient_list=('subject', lambda x: ', '.join(sorted(x.unique()))),
        active_TFs=('active_TF', lambda x: ';'.join(set(x[x.notna()].astype(str))) if not x.isna().all() else np.nan)
    ).reset_index()
    # Calculate percentage of subjects in ADC → SCLC group
    interaction_counts['percent_transformed'] = interaction_counts['patient_list'].apply(
        lambda x: len([patient for patient in x.split(', ') if patient in sample_groups['ADC → SCLC']]) / len(x.split(', ')) * 100
    )
    # Filter for interactions occurring in more than one patient
    frequent_interactions = interaction_counts[interaction_counts['num_patients'] > 1]
    
    # Print summary of frequent interactions
    print("\nFrequent NonNE SCLC interactions (occurring in >1 patient):")
    print(f"Total number of frequent interactions: {len(frequent_interactions)}")
    
    # Rank interactions where NonNE SCLC is sender
    print("\nFrequent interactions where NSCLC is sender:")
    sender_ranked = frequent_interactions[frequent_interactions['sender'].isin(['NonNE SCLC', 'NSCLC'])].sort_values(
        'num_patients', ascending=False
    )
    sender_ranked = sender_ranked.loc[sender_ranked.sender != sender_ranked.receiver,:]
    print(sender_ranked.head())   
    
    # Rank interactions where NonNE SCLC is receiver
    print("\nFrequent interactions where NonNE SCLC is receiver:")
    receiver_ranked = frequent_interactions[frequent_interactions['receiver'].isin(['NonNE SCLC', 'NSCLC'])].sort_values(
        'num_patients', ascending=False
    )
    receiver_ranked = receiver_ranked.loc[receiver_ranked.sender != receiver_ranked.receiver,:]
    print(receiver_ranked.head())
    
    # Save ranked interactions
    with pd.ExcelWriter('nonne_sclc_frequent_interactions.xlsx') as writer:
        sender_ranked.to_excel(writer, sheet_name='NSCLC as Sender', index=False)
        receiver_ranked.to_excel(writer, sheet_name='NSCLC as Receiver', index=False)





    #----------------------------------
    # Create dotplots for the four subjects with NonNE SCLC
    nonne_sclc_subjects = ['RU1068', 'RU151', 'RU1518', 'RU325']
    
    # Define base directory and cell type column
    base_dir = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/out.individual.120122/"
    cell_type_col = 'cell_type_final'  # Specify the cell type column name
    
    # Initialize combined adata
    combined_adata = None
    
    # Load and combine adatas for the four subjects
    
    adata = sc.read_h5ad("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/scvi_latent150/adata.scvi.all_genes.final_subset.062124.h5ad")
    adata = adata[adata.obs['patient'].isin(nonne_sclc_subjects)]
    adata.obs['cell_type_patient'] = adata.obs[cell_type_col].astype(str) + "_" + adata.obs['patient'].astype(str)
    
    # Get unique genes from the frequent interactions
    unique_ligands = np.unique(list(dict.fromkeys(sender_ranked['interacting_pair'].str.split('_').str[0])) + list(dict.fromkeys(receiver_ranked['interacting_pair'].str.split('_').str[0])))
    unique_receptors = np.unique(list(dict.fromkeys(receiver_ranked['interacting_pair'].str.split('_').str[1])) + list(dict.fromkeys(sender_ranked['interacting_pair'].str.split('_').str[1])))
    
    # Filter genes to only include those present in adata
    unique_ligands = [gene for gene in unique_ligands if gene in adata.var_names]
    unique_receptors = [gene for gene in unique_receptors if gene in adata.var_names]
    
    print(f"\nNumber of ligands found in adata: {len(unique_ligands)}")
    print(f"Number of receptors found in adata: {len(unique_receptors)}")
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get all cell types involved in interactions
    all_sender_cells = ['NonNE SCLC', "SCLC-N"]
    all_receiver_cells = ['NonNE SCLC', "SCLC-N"]
    
    # Plot sender cells with ligand genes
    sender_mask = adata.obs[cell_type_col].isin(all_sender_cells)
    sc.pl.dotplot(
        adata[sender_mask],
        var_names=unique_ligands,
        groupby='cell_type_patient',
        ax=ax1,
        show=False
    )
    ax1.set_title('Ligand Expression in Sender Cells')
    
    # Plot receiver cells with receptor genes
    receiver_mask = adata.obs[cell_type_col].isin(all_receiver_cells)
    sc.pl.dotplot(
        adata[receiver_mask],
        var_names=unique_receptors,
        groupby='cell_type_patient',
        ax=ax2,
        show=False
    )
    ax2.set_title('Receptor Expression in Receiver Cells')
    
    plt.tight_layout()
    plt.savefig('nonne_sclc_subjects_dotplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    #----------------------------------
    if FILTER_CONTROL:
        patients_filtered = ["RU1195", "RU1215"]
        patients_filtered_2 = relevance_long_df.groupby('condition')['subject'].unique()['ADC → SCLC']
        patients_filtered.extend(patients_filtered_2)
        relevance_long_df = relevance_long_df.loc[relevance_long_df['subject'].isin(patients_filtered)]
    relevance_long_df = relevance_long_df.loc[relevance_long_df.value != 0,:]

    cell_sign_long_df = cell_sign_long_df.loc[cell_sign_long_df['subject'].isin(relevance_long_df.subject.unique())]

    fisher_results = perform_fisher_test(relevance_long_df, cell_sign_long_df, alternative='greater')
    fisher_results.to_csv(f'fisher_exact_test_results_{"FILTER_CONTROL" if FILTER_CONTROL else "ALL"}.csv', index=False)
    
    # Print top 10 lowest p-value interactions per sender-receiver pair
    print("\nTop 10 lowest p-value interactions per sender-receiver pair:")
    top10_per_group = fisher_results.sort_values(['sender_receiver_pair', 'p_value']).groupby('sender_receiver_pair').head(10)
    print(top10_per_group[['interacting_pair', 'sender_receiver_pair', 'ADC → SCLC_frequency', 'De novo SCLC and ADC_frequency', 'active_TFs', 'odds_ratio', 'p_value']].to_string())
    
    # 4. Plot results using different value columns
    # Using p-values
    plot_significant_interactions(
        fisher_results, 
        pvalue_column='p_value',
        value_column='p_value',
        output_file=f'../results/figures/significant_lr_interactions_heatmap_pval_{"FILTER_CONTROL" if FILTER_CONTROL else "ALL"}.png',
        top_n=30,
        figsize=(9,8)
    )

 
 
 
 
 
 
    #----------------------------------
    # Verification on adata
    # # Load the adata and create validation plots
    # base_dir = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/out.individual.120122/"
    # cell_type_col = 'cell_type_fine'  # Specify the cell type column name
    
    # # Initialize combined adata
    # combined_adata = None
    
    # # Load and combine adatas
    # for patient in relevance_long_df['subject'].unique():
    #     adata = sc.read_h5ad(os.path.join(base_dir, f"{patient}/adata.{patient}.h5ad"))
    #     # Filter for relevant cell types
    #     cell_types = list(set(senders + receivers))
    #     adata = adata[adata.obs[cell_type_col].isin(cell_types), :]
    #     adata.obs['condition'] = sample_to_group[patient]
    #     # Add patient information to cell type labels
    #     adata.obs['cell_type_patient'] = [f"{ct}_{patient}" for ct in adata.obs[cell_type_col]]
        
    #     if combined_adata is None:
    #         combined_adata = adata
    #     else:
    #         combined_adata = combined_adata.concatenate(adata)
    # # Create validation plots using combined adata
    # plot_lr_validation(
    #     combined_adata,
    #     ligands=ligands,
    #     receptors=receptors,
    #     senders=senders,
    #     receivers=receivers,
    #     cell_type_col=cell_type_col,
    #     output_prefix='lr_validation_combined'
    # )

    # verification on UMAPs
    # cell_sign_long_df_detailed = pd.read_csv('cell_sign_long_df_detailed.csv')
    # relevance_long_df = relevance_long_df.loc[relevance_long_df.value != 0,:]
    # cell_sign_long_df_detailed = cell_sign_long_df_detailed.merge(relevance_long_df, on=['interacting_pair', 'subject', 'sender', 'receiver'], how='inner')
    # significant_lr = fisher_results.loc[(fisher_results['p_value'] < 0.05) & (fisher_results.active_TFs != 'None') & (fisher_results.sender != fisher_results.receiver),
    #                                     ["interacting_pair", "sender", "receiver", "sender_receiver_pair"]].drop_duplicates()
    # cell_sign_long_df_detailed = cell_sign_long_df_detailed.merge(significant_lr, on=['interacting_pair', 'sender', 'receiver'], how='inner') # only 73 rows left 
    # cell_sign_long_df_detailed = cell_sign_long_df_detailed.loc[:, ["interacting_pair", "gene_a", "gene_b", "active_TF", "sender", "receiver", "subject"]].drop_duplicates() # 40 rows left
    # cell_sign_long_df_detailed.to_csv('cell_sign_long_df_detailed.csv', index=False)
 
    # Load the adata and plot UMAP of corresponding genes and cell types
#     base_dir = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/"
#     unique_combinations = cell_sign_long_df_detailed[['gene_a', 'gene_b']].drop_duplicates()
    
#     for _, row in unique_combinations.iterrows():
#         print(row)
#         gene_a = row['gene_a']
#         gene_b = row['gene_b']
        
#         # Get all patients that have this combination
#         if pd.isna(gene_a):
#             relevant_patients = cell_sign_long_df_detailed[
#                 (cell_sign_long_df_detailed['gene_b'] == gene_b)
#             ]['subject'].unique()
#         else:
#             relevant_patients = cell_sign_long_df_detailed[
#                 (cell_sign_long_df_detailed['gene_a'] == gene_a) &
#                 (cell_sign_long_df_detailed['gene_b'] == gene_b)
#             ]['subject'].unique()
        
#         # Create a figure with subplots for each patient
#         n_patients = len(relevant_patients)
#         # Adjust number of columns based on whether gene_a is NaN
#         n_genes = 2 if pd.isna(gene_a) else 3  # gene_b and cell_type_fine, plus gene_a if not NaN
        
#         # Create a figure with n_patients rows and n_genes columns
#         fig, axes = plt.subplots(n_patients, n_genes, figsize=(3*n_genes, 3*n_patients))
#         if n_patients == 1:
#             axes = axes.reshape(1, -1)
        
#         # Plot UMAP for each patient
#         for i, patient in enumerate(relevant_patients):
#             # Load and filter adata for this patient
#             adata = sc.read_h5ad(os.path.join(base_dir, f"{patient}.no_cc.hvg_2000.090124.h5ad"))
#             cell_types = ['SCLC-A', 'SCLC-N']
#             adata = adata[adata.obs['cell_type_final2'].isin(cell_types), :]
            
#             # Get all sender-receiver pairs for this patient and gene combination
#             if pd.isna(gene_a):
#                 patient_sr_pairs = cell_sign_long_df_detailed[
#                     (cell_sign_long_df_detailed['subject'] == patient) &
#                     (cell_sign_long_df_detailed['gene_b'] == gene_b)
#                 ][['sender', 'receiver']].drop_duplicates()
#             else:
#                 patient_sr_pairs = cell_sign_long_df_detailed[
#                     (cell_sign_long_df_detailed['subject'] == patient) &
#                     (cell_sign_long_df_detailed['gene_a'] == gene_a) &
#                     (cell_sign_long_df_detailed['gene_b'] == gene_b)
#                 ][['sender', 'receiver']].drop_duplicates()
            
#             # Create patient-specific title with all sender-receiver information
#             sr_pairs_str = '; '.join([f"{sr['sender']}→{sr['receiver']}" for _, sr in patient_sr_pairs.iterrows()])
#             patient_title = f"{patient}\n{gene_a if not pd.isna(gene_a) else 'NaN'}-{gene_b}\n({sr_pairs_str})"
            
#             # Plot each gene
#             genes_to_plot = []
#             if not pd.isna(gene_a):
#                 genes_to_plot.append(gene_a)
#             genes_to_plot.extend([gene_b, 'cell_type_final2'])
            
#             for j, gene in enumerate(genes_to_plot):
#                 ax = axes[i, j]
#                 if gene == 'cell_type_final2':
#                     # Get unique cell types in this dataset
#                     unique_cell_types = adata.obs['cell_type_final2'].unique()
                    
#                     sc.pl.umap(
#                         adata,
#                         color=gene,
#                         use_raw=False,
#                         palette=tumor_color_map,
#                         title=f"{patient_title}\n{gene}" if j == 0 else gene,
#                         ax=ax,
#                         show=False,
#                         legend_loc='on data' if len(unique_cell_types) <= 5 else 'right margin'
#                     )
#                 else:
#                     sc.pl.umap(
#                         adata,
#                         color=gene,
#                         use_raw=False,
#                         cmap="Spectral_r",
#                         title=f"{patient_title}\n{gene}" if j == 0 else gene,
#                         ax=ax,
#                         sort_order=False,
#                         show=False
#                     )
        
#         plt.suptitle(f"Gene Pair: {gene_a if not pd.isna(gene_a) else 'NaN'}-{gene_b}", y=1.02, fontsize=12)
#         plt.tight_layout()
#         plt.savefig(f"umap_{gene_a if not pd.isna(gene_a) else 'NaN'}_{gene_b}_by_patient.png", 
#                    dpi=300, bbox_inches='tight')
#         plt.close()

# # Make a dot plot to compare gene expression across cell types
# print("\nCreating dot plot for gene expression across cell types...")

# # Get unique combinations of gene_a, gene_b
# unique_combinations = cell_sign_long_df_detailed[['gene_a', 'gene_b']].drop_duplicates()

# for _, row in unique_combinations.iterrows():
#     gene_a = row['gene_a']
#     gene_b = row['gene_b']
    
#     # Get all patients that have this combination
#     if pd.isna(gene_a):
#         relevant_patients = cell_sign_long_df_detailed[
#             (cell_sign_long_df_detailed['gene_b'] == gene_b)
#         ]['subject'].unique()
#     else:
#         relevant_patients = cell_sign_long_df_detailed[
#             (cell_sign_long_df_detailed['gene_a'] == gene_a) &
#             (cell_sign_long_df_detailed['gene_b'] == gene_b)
#         ]['subject'].unique()
    
#     # Create a combined adata for all relevant patients
#     combined_adata = None
#     for patient in relevant_patients:
#         adata = sc.read_h5ad(os.path.join(base_dir, f"{patient}.no_cc.hvg_2000.090124.h5ad"))
#         cell_types = ['SCLC-A', 'SCLC-N']
#         adata = adata[adata.obs['cell_type_final2'].isin(cell_types), :]
        
#         # Add patient information to cell type labels
#         adata.obs['cell_type_patient'] = [x + f'_{patient}' for x in adata.obs['cell_type_final2']]
        
#         if combined_adata is None:
#             combined_adata = adata
#         else:
#             combined_adata = combined_adata.concatenate(adata)
    
#     # Get genes to plot
#     genes_to_plot = []
#     if not pd.isna(gene_a):
#         genes_to_plot.append(gene_a)
#     genes_to_plot.append(gene_b)
    
#     # Create dot plot
#     n_patients = len(relevant_patients)
#     plt.figure(figsize=(4 * n_patients, len(genes_to_plot) * 0.5 + 1))
#     sc.pl.dotplot(
#         combined_adata,
#         var_names=genes_to_plot,
#         groupby='cell_type_patient',
#         title=f"{gene_a if not pd.isna(gene_a) else 'NaN'}-{gene_b}",
#         save=f"dotplot_{gene_a if not pd.isna(gene_a) else 'NaN'}_{gene_b}.png",
#         show=False
#     )
#     plt.close()