import pandas as pd
import numpy as np
from constants import sample_groups

relevance_long_df = pd.read_csv('../data/processed/within_tumor_relevance.csv')
cell_sign_long_df = pd.read_csv('../data/processed/cell_sign_long_df.csv')

nonne_sclc_interactions = relevance_long_df[
(relevance_long_df['sender'].isin(['NonNE SCLC', 'NSCLC'])) | 
(relevance_long_df['receiver'].isin(['NonNE SCLC', 'NSCLC']))
]
nonne_sclc_interactions['sender'][nonne_sclc_interactions['sender'] == 'NonNE SCLC'] = 'NSCLC'
nonne_sclc_interactions['receiver'][nonne_sclc_interactions['receiver'] == 'NonNE SCLC'] = 'NSCLC'
nonne_sclc_interactions = nonne_sclc_interactions.merge(cell_sign_long_df, on=['interacting_pair', "subject", 'sender', 'receiver'], how='left')

# Count number of patients per interaction and get patient list
interaction_counts = nonne_sclc_interactions.groupby(
['interacting_pair', 'classification', 'sender', 'receiver']
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
with pd.ExcelWriter('../results/tables/nonne_sclc_frequent_interactions.xlsx') as writer:
    sender_ranked.to_excel(writer, sheet_name='NSCLC as Sender', index=False)
    receiver_ranked.to_excel(writer, sheet_name='NSCLC as Receiver', index=False)


