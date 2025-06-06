import pandas as pd
from constants import sample_groups
import numpy as np
from scipy.stats import fisher_exact

def group_cell_type(x):
    if x in ['Basal-like_NSCLC', 'Mucinous_NSCLC', 'NonNE SCLC']:
        return 'NSCLC'
    elif x == "Mφ/_Mono":
        return "Mφ/Mono"
    else:
        return x

def classify_interaction(sender, receiver):
    """
    Classify the interaction type based on sender and receiver cell types.
    
    Parameters:
    -----------
    sender : str
        Sender cell type
    receiver : str
        Receiver cell type
        
    Returns:
    --------
    str
        Classification of the interaction as:
        - 'within_tumor': both sender and receiver are tumor cells
        - 'tumor_immune': one is tumor and one is immune
        - 'stromal_immune': one is stromal and one is immune
    """
    # Define cell type groups
    tumor_cells = ["NSCLC", "SCLC-A", "SCLC-N", "NonNE SCLC"]
    immune_cells = ["Mφ/Mono", "Mφ/Mono_CD14", "Mφ/Mono_CD11c", "Mφ/Mono_CCL", "PMN", "Mast_cell", "pDC", 
                    "CD4_EM/Effector", "CD4_naive/CM", "CD4_TRM", "CD8_TRM/EM", "T_reg", "NK", "ILC", "B_memory"]
    stromal_cells = ['Fibroblast', 'Endothelial', "Basal", "Ciliated"]
    
    # Check if both are tumor cells
    if sender in tumor_cells and receiver in tumor_cells:
        return 'within_tumor'
    
    # Check if one is tumor and one is immune
    if (sender in tumor_cells and receiver in immune_cells) or \
       (sender in immune_cells and receiver in tumor_cells):
        return 'tumor_immune'
    
    # Check if one is stromal and one is immune
    if (sender in stromal_cells and receiver in tumor_cells) or \
       (sender in tumor_cells and receiver in stromal_cells):
        return 'tumor_stromal'
    
    # Default case
    return 'other'

def summarize_lr_interactions(relevance_df):
    """
    Summarize ligand-receptor interactions across conditions, including relevance, CellSign activity,
    and interaction scores. Handles missing patients by using total patient counts from sample_groups.
    Properly aggregates multiple samples per subject before calculating condition-level statistics.
    
    Parameters:
    -----------
    relevance_df : pandas.DataFrame
        DataFrame containing the relevance data with columns:
        - interacting_pair
        - classification
        - sender
        - receiver
        - condition
        - relevance
        - interaction_score
        - CellSign_active
        - subject
        - sample
        
    Returns:
    --------
    pandas.DataFrame
        Summary statistics for each interaction with columns:
        - interacting_pair
        - classification
        - sender_receiver_pair
        - sender
        - receiver
        - interaction_type
        - De novo SCLC and ADC_relevance_freq
        - ADC → SCLC_relevance_freq
        - De novo SCLC and ADC_cellsign_freq
        - ADC → SCLC_cellsign_freq
        - De novo SCLC and ADC_median_interaction
        - ADC → SCLC_median_interaction
        - active_TFs
        - fisher_pval
    """
    # Get total number of patients per condition
    total_patients = {
        'De novo SCLC and ADC': len(sample_groups['De novo SCLC and ADC']),
        'ADC → SCLC': len(sample_groups['ADC → SCLC'])
    }
    
    # Create sender-receiver pair column
    relevance_df['sender_receiver_pair'] = relevance_df['sender'] + '→' + relevance_df['receiver']
    
    # First aggregate data at the subject level
    subject_agg = []
    for (interacting_pair, classification, sender_receiver_pair, subject), group in relevance_df.groupby(['interacting_pair', 'classification', 'sender_receiver_pair', 'subject']):
        # For each subject, aggregate their samples
        # Get unique active TFs, excluding None values
        active_tfs = group['active_TF'].dropna().unique()
        active_tfs_str = ';'.join(active_tfs) if len(active_tfs) > 0 else ''
        
        subject_row = {
            'interacting_pair': interacting_pair,
            'classification': classification,
            'sender_receiver_pair': sender_receiver_pair,
            'subject': subject,
            'sender': group['sender'].iloc[0],
            'receiver': group['receiver'].iloc[0],
            'condition': group['condition'].iloc[0],
            # Consider a subject as having relevance if any of their samples have relevance
            'has_relevance': group['relevance'].max() > 0,
            # Consider a subject as having CellSign active if any of their samples are active
            'has_cellsign': group['CellSign_active'].max() > 0,
            # Use median interaction score across all samples for this subject
            'median_interaction': group['interaction_score'].median(),
            # Get unique active TFs across all samples, excluding None
            'active_TFs': active_tfs_str
        }
        subject_agg.append(subject_row)
    
    # Convert to DataFrame
    subject_df = pd.DataFrame(subject_agg)
    
    # Now calculate condition-level statistics
    summary = []
    for (interacting_pair, sender_receiver_pair), group in subject_df.groupby(['interacting_pair', 'sender_receiver_pair']):
        # Calculate statistics for each condition
        stats = {}
        for condition in total_patients.keys():
            condition_data = group[group['condition'] == condition]
            
            # Calculate frequencies using total patients as denominator
            relevance_freq = len(condition_data[condition_data['has_relevance']]) / total_patients[condition]
            cellsign_freq = len(condition_data[condition_data['has_cellsign']]) / total_patients[condition]
            
            # Calculate median interaction score across subjects
            median_interaction = condition_data['median_interaction'].median() if len(condition_data) > 0 else 0
            
            # Get lists of subjects with relevance and CellSign activity
            relevant_subjects = condition_data[condition_data['has_relevance']]['subject'].tolist()
            cellsign_subjects = condition_data[condition_data['has_cellsign']]['subject'].tolist()
            
            stats[condition] = {
                'relevance_freq': relevance_freq,
                'cellsign_freq': cellsign_freq,
                'median_interaction': median_interaction,
                'n_subjects': len(condition_data),
                'relevant_subjects': ';'.join(relevant_subjects),
                'cellsign_subjects': ';'.join(cellsign_subjects),
                'n_relevant': len(relevant_subjects)
            }
        
        # Perform Fisher's exact test
        contingency_table = np.array([
            [stats['De novo SCLC and ADC']['n_relevant'], 
             total_patients['De novo SCLC and ADC'] - stats['De novo SCLC and ADC']['n_relevant']],
            [stats['ADC → SCLC']['n_relevant'], 
             total_patients['ADC → SCLC'] - stats['ADC → SCLC']['n_relevant']]
        ])
        fisher_pval = fisher_exact(contingency_table)[1]
        
        # Classify the interaction type
        interaction_type = classify_interaction(group['sender'].iloc[0], group['receiver'].iloc[0])
        
        # Get unique active TFs across all subjects, excluding empty strings
        all_tfs = group['active_TFs'].unique()
        active_tfs = ';'.join([tf for tf in all_tfs if tf != ''])
        
        # Create summary row
        summary_row = {
            'interacting_pair': interacting_pair,
            'classification': group['classification'].iloc[0],
            'sender_receiver_pair': sender_receiver_pair,
            'sender': group['sender'].iloc[0],
            'receiver': group['receiver'].iloc[0],
            'interaction_type': interaction_type,
            'De novo SCLC and ADC_relevance_freq': stats['De novo SCLC and ADC']['relevance_freq'],
            'ADC → SCLC_relevance_freq': stats['ADC → SCLC']['relevance_freq'],
            'De novo SCLC and ADC_cellsign_freq': stats['De novo SCLC and ADC']['cellsign_freq'],
            'ADC → SCLC_cellsign_freq': stats['ADC → SCLC']['cellsign_freq'],
            'De novo SCLC and ADC_median_interaction': stats['De novo SCLC and ADC']['median_interaction'],
            'ADC → SCLC_median_interaction': stats['ADC → SCLC']['median_interaction'],
            'active_TFs': active_tfs,
            'De novo SCLC and ADC_n_subjects': stats['De novo SCLC and ADC']['n_subjects'],
            'ADC → SCLC_n_subjects': stats['ADC → SCLC']['n_subjects'],
            'De novo SCLC and ADC_total_subjects': total_patients['De novo SCLC and ADC'],
            'ADC → SCLC_total_subjects': total_patients['ADC → SCLC'],
            'De novo SCLC and ADC_relevant_subjects': stats['De novo SCLC and ADC']['relevant_subjects'],
            'ADC → SCLC_relevant_subjects': stats['ADC → SCLC']['relevant_subjects'],
            'De novo SCLC and ADC_cellsign_subjects': stats['De novo SCLC and ADC']['cellsign_subjects'],
            'ADC → SCLC_cellsign_subjects': stats['ADC → SCLC']['cellsign_subjects'],
            'fisher_pval': fisher_pval
        }
        summary.append(summary_row)
    
    # Convert to DataFrame and sort by p-value
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('fisher_pval')
    
    return summary_df

def analyze_interactions(summary_df, interaction_type, min_relevance_freq=0.1, min_diff=0.2, min_subjects=3):
    """
    Filter and analyze interactions based on multiple criteria.
    
    Parameters:
    -----------
    summary_df : pandas.DataFrame
        Summary DataFrame from summarize_lr_interactions
    interaction_type : str
        Type of interaction to analyze ('within_tumor', 'tumor_immune', 'tumor_stromal')
    min_relevance_freq : float
        Minimum relevance frequency in either condition
    min_diff : float
        Minimum difference in relevance frequency between conditions
    min_subjects : int
        Minimum number of subjects in either condition
        
    Returns:
    --------
    pandas.DataFrame
        Filtered and analyzed interactions
    """
    # Filter by interaction type
    filtered = summary_df[summary_df['interaction_type'] == interaction_type].copy()
    
    # Add maximum relevance frequency across conditions
    filtered['max_relevance_freq'] = filtered[['De novo SCLC and ADC_relevance_freq', 
                                             'ADC → SCLC_relevance_freq']].max(axis=1)
    
    # Add direction of difference
    filtered['relevance_freq_direction'] = (filtered['ADC → SCLC_relevance_freq'] - 
                                          filtered['De novo SCLC and ADC_relevance_freq'])
    
    # Apply filters
    filtered = filtered[
        (filtered['max_relevance_freq'] >= min_relevance_freq) &
        (filtered['relevance_freq_diff'] >= min_diff) &
        ((filtered['De novo SCLC and ADC_n_subjects'] >= min_subjects) |
         (filtered['ADC → SCLC_n_subjects'] >= min_subjects))
    ]
    
    # Sort by absolute difference
    filtered = filtered.sort_values('relevance_freq_diff', ascending=False)
    
    return filtered

