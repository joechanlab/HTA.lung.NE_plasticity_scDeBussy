import anndata as ad
import os
import pandas as pd
import ktplotspy as kpy
import glob

sample_groups = {
    "De novo SCLC and ADC": [
        "RU1311", "RU1124", "RU1108", "RU1065",
        "RU871", "RU1322", "RU1144", "RU1066",
        "RU1195", "RU779", "RU1152", "RU426",
        "RU1231", "RU222", "RU1215", "RU1145",
        "RU1360", "RU1287", # 18
    ],
    "ADC → SCLC": [
        "RU1068", "RU1676", "RU263", "RU942",
        "RU325", "RU1304", "RU1444", "RU151",
        "RU1181", "RU1042", "RU581", "RU1303",
        "RU1305", "RU1250", "RU831", "RU1646",
        "RU226", "RU773", "RU1083", "RU1518",
        "RU1414", "RU1405", "RU1293" # 23
    ]
}
sample_to_group = {
    sample_id: group
    for group, samples in sample_groups.items()
    for sample_id in samples
}

base_dir_pattern = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/out.individual.120122/*/cellphonedb.output"
interacting_pairs = ['JAG1_NOTCH1', 'DLL3_NOTCH2', 'EFNB2_EPHB2', 'DLK1_NOTCH1']
cell_type_senders = ['SCLC-A', 'SCLC-A', 'SCLC-N', 'SCLC-N']
cell_type_receivers = ['SCLC-N', 'SCLC-N', 'SCLC-A', 'SCLC-A']

# Get all matching directories
matching_dirs = glob.glob(base_dir_pattern)

for i in range(len(interacting_pairs)):
    interacting_pair = interacting_pairs[i]
    cell_type_sender = cell_type_senders[i]
    cell_type_receiver = cell_type_receivers[i]
    print(f"Processing {interacting_pair} ({cell_type_sender} -> {cell_type_receiver})")
    # Initialize empty list to store means dataframes
    all_means = []

    # Process each directory
    for dir_path in matching_dirs:
        # Extract subject from path
        subject = os.path.basename(os.path.dirname(dir_path))
        
        # Read means file
        means_file = os.path.join(dir_path, "degs_analysis_relevant_interactions_042725.txt")
        if os.path.exists(means_file):
            means = pd.read_csv(means_file, sep='\t')
            # Filter for interacting pair
            means = means[means['interacting_pair'] == interacting_pair]
            
            # Find the column that matches SCLC-A|SCLC-N pattern for this subject
            target_col = None
            for col in means.columns[14:]:  # Skip first 13 columns
                if col == 'subject':  # Skip the subject column
                    continue
                sender, receiver = col.split('|')
                if sender.startswith(cell_type_sender) and receiver.startswith(cell_type_receiver):
                    target_col = col
                    break
            
            if target_col is not None:
                # Extract only the relevant columns and interaction value
                means_subset = means[['interacting_pair', target_col]].copy()
                means_subset['subject'] = subject
                means_subset['interaction_value'] = means_subset[target_col]
                means_subset = means_subset[['interacting_pair', 'subject', 'interaction_value']]
                all_means.append(means_subset)

    # Combine all means dataframes
    if all_means:
        means = pd.concat(all_means, ignore_index=True)
        # Add group column based on sample_to_group dictionary
        means['group'] = means['subject'].map(sample_to_group)
        means = means.sort_values(['group', 'interaction_value'])
        print(f"Found data for {len(all_means)} subjects")
        print("Subjects:", means['subject'].unique())
        print("\nInteraction values:")
        print(means)
    else:
        print("No means data found in any directory")
