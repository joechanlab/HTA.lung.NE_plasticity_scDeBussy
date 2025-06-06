import pandas as pd
import glob
from utils.data_helpers import group_cell_type, summarize_lr_interactions
from utils.plotting_helpers import plot_interaction_heatmap
from constants import sample_to_group

# Example usage:
if __name__ == "__main__":
    #--------------------------------
    # Relevance matrix
    base_dir = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/out.individual.120122/"
    relevant_file_pattern = base_dir + "/*/cellphonedb.output/degs_analysis_relevant_interactions_042725.txt"
    file_list =  glob.glob(relevant_file_pattern)

    dfs = []
    for file in file_list:
        df = pd.read_csv(file, sep='\t')
        patient = file.split('/')[-3]
        interacting_pairs = df['interacting_pair']
        classification = df['classification']
        df = df.iloc[:,13:]
        df.loc[:,'interacting_pair'] = interacting_pairs
        df.loc[:,'classification'] = classification
        df.loc[:,'subject'] = patient
        df = df.reset_index(drop=True)
        df = pd.melt(df, id_vars=['interacting_pair', 'classification', 'subject'], var_name='pair', value_name='relevance')
        df = df.loc[df.relevance != 0]
        df[['sender', 'receiver']] = df['pair'].str.split('|', expand=True)
        df['sample'] = df['sender'].str.split(r'[.]').str[1]
        df['sender'] = df['sender'].str.split(r'[.]').str[0]
        df['receiver'] = df['receiver'].str.split(r'[.]').str[0]
        df = df.drop(columns=['pair'])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv("../data/processed/degs_analysis_relevant_interactions.txt", sep='\t', index=False)

    #--------------------------------
    # Interaction scores
    relevant_file_pattern = base_dir + "/*/cellphonedb.output/degs_analysis_interaction_scores_042725.txt"
    file_list =  glob.glob(relevant_file_pattern)

    dfs = []
    for file in file_list:
        df = pd.read_csv(file, sep='\t')
        patient = file.split('/')[-3]
        interacting_pairs = df['interacting_pair']
        df = df.iloc[:,13:]
        df.loc[:,'interacting_pair'] = interacting_pairs
        df.loc[:,'subject'] = patient
        df = df.reset_index(drop=True)
        df = pd.melt(df, id_vars=['interacting_pair', 'subject'], var_name='pair', value_name='interaction_score')
        df = df.loc[df.interaction_score != 0]
        df[['sender', 'receiver']] = df['pair'].str.split('|', expand=True)
        df['sample'] = df['sender'].str.split(r'[.]').str[1]
        df['sender'] = df['sender'].str.split(r'[.]').str[0].apply(group_cell_type)
        df['receiver'] = df['receiver'].str.split(r'[.]').str[0].apply(group_cell_type)
        df = df.drop(columns=['pair'])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("../data/processed/degs_analysis_interaction_scores.txt", sep='\t', index=False)

    #--------------------------------
    # CellSign information
    relevant_file_pattern = base_dir + "/*/cellphonedb.output/degs_analysis_CellSign_active_interactions_042725.txt"
    file_list =  glob.glob(relevant_file_pattern)

    dfs = []
    for file in file_list:
        df = pd.read_csv(file, sep='\t')
        patient = file.split('/')[-3]
        interacting_pairs = df['interacting_pair']
        df = df.iloc[:,13:]
        df.loc[:,'interacting_pair'] = interacting_pairs
        df.loc[:,'subject'] = patient
        df = df.reset_index(drop=True)
        df = pd.melt(df, id_vars=['interacting_pair', 'subject'], var_name='pair', value_name='CellSign_active')
        df = df.loc[df.CellSign_active != 0]
        df[['sender', 'receiver']] = df['pair'].str.split('|', expand=True)
        df['sample'] = df['sender'].str.split(r'[.]').str[1]
        df['sender'] = df['sender'].str.split(r'[.]').str[0].apply(group_cell_type)
        df['receiver'] = df['receiver'].str.split(r'[.]').str[0].apply(group_cell_type)
        df = df.drop(columns=['pair'])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("../data/processed/degs_analysis_CellSign_active_interactions.txt", sep='\t', index=False)

    #--------------------------------
    # Active TF information
    relevant_file_pattern = base_dir + "/*/cellphonedb.output/degs_analysis_CellSign_active_interactions_deconvoluted_042725.txt"
    file_list =  glob.glob(relevant_file_pattern)

    dfs = []
    for file in file_list:
        df = pd.read_csv(file, sep='\t')
        patient = file.split('/')[-3]
        df = df.loc[:,["interacting_pair", "celltype_pairs", "active_TF", "active_celltype"]]
        df.loc[:,'subject'] = patient
        df = df.reset_index(drop=True)
        df[['sender', 'receiver']] = df['celltype_pairs'].str.split('|', expand=True)
        df['sample'] = df['sender'].str.split(r'[.]').str[1]
        df['sender'] = df['sender'].str.split(r'[.]').str[0].apply(group_cell_type)
        df['receiver'] = df['receiver'].str.split(r'[.]').str[0].apply(group_cell_type)
        df['active_celltype'] = df['active_celltype'].str.split(r'[.]').str[0]
        df = df.drop(columns=['celltype_pairs'])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("../data/processed/degs_analysis_CellSign_active_interactions_deconvoluted.txt", sep='\t', index=False)

    #--------------------------------
    # Combine the information together
    relevance_df = pd.read_csv("../data/processed/degs_analysis_relevant_interactions.txt", sep='\t')
    expression_df = pd.read_csv("../data/processed/degs_analysis_interaction_scores.txt", sep='\t')
    cellsign_df = pd.read_csv("../data/processed/degs_analysis_CellSign_active_interactions.txt", sep='\t')
    deconvoluted_df = pd.read_csv("../data/processed/degs_analysis_CellSign_active_interactions_deconvoluted.txt", sep='\t')

    relevance_df = relevance_df.merge(expression_df, on=['interacting_pair', 'subject', 'sender', 'receiver', 'sample'], how='left')
    relevance_df = relevance_df.merge(cellsign_df, on=['interacting_pair', 'subject', 'sender', 'receiver', 'sample'], how='left')
    relevance_df = relevance_df.merge(deconvoluted_df, on=['interacting_pair', 'subject', 'sender', 'receiver', 'sample'], how='left')
    
    subject_list = relevance_df['subject'].unique()
    condition_labels = pd.DataFrame(dict(
        subject=subject_list,
        condition=[sample_to_group.get(x, "unselected") for x in subject_list]
    ))
    relevance_df = relevance_df.merge(condition_labels, on='subject', how='left')
    relevance_df = relevance_df.loc[relevance_df.condition != "unselected"]
    relevance_df = relevance_df.loc[~relevance_df.interaction_score.isna()]
    relevance_df.to_csv("../data/processed/degs_analysis_relevant_interactions_with_scores.txt", sep='\t', index=False)

    # Load the relevance data
    relevance_df = pd.read_csv("../data/processed/degs_analysis_relevant_interactions_with_scores.txt", sep='\t')
    
    # Generate summary
    summary = summarize_lr_interactions(relevance_df)
    
    # Calculate absolute difference in relevance frequency between conditions
    summary['relevance_freq_diff'] = abs(summary['ADC → SCLC_relevance_freq'] - summary['De novo SCLC and ADC_relevance_freq'])
    
    # Save full summary to file
    summary.to_csv("../results/tables/lr_interaction_summary.txt", sep='\t', index=False)
    
    # Analyze each interaction type
    interaction_types = ['within_tumor', 'tumor_immune', 'tumor_stromal']
    
    within_tumor_ordering = [
    "SCLC-A",
    "SCLC-N"
    ]
    tumor_immune_ordering = [
        "SCLC-N",
        "SCLC-A",
        "NSCLC",
        "T_reg",
        "CD4_naive/CM"
    ]

    tumor_stromal_ordering = [
        "SCLC-A",
        "SCLC-N",
        "Endothelial",
        "Fibroblast"
    ]
    col_ordering = {
        'within_tumor': within_tumor_ordering,
        'tumor_immune': tumor_immune_ordering,
        'tumor_stromal': tumor_stromal_ordering
    }
    for interaction_type in interaction_types:
        # Get data for this interaction type
        type_summary = summary[summary['interaction_type'] == interaction_type].sort_values('relevance_freq_diff', ascending=False)
        
        # Save to file
        output_file = f"../results/tables/lr_interaction_summary_{interaction_type}.txt"
        type_summary.to_csv(output_file, sep='\t', index=False)
        
        # Create heatmap
        plot_interaction_heatmap(
            type_summary,
            output_file=f'../results/figures/{interaction_type}_interactions_heatmap_sender.png',
            top_n=30,
            pval_threshold=1,
            figsize=(12.5, 8),
            facet_by='sender',
            facet_order=col_ordering[interaction_type]
        )
