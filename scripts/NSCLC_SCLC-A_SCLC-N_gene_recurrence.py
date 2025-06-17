# Use the GAM to define the recurrence of genes in NSCLC and SCLC
import os
import pandas as pd
from scdebussy.pp import create_cellrank_probability_df
from sklearn.preprocessing import MinMaxScaler
from pygam import LinearGAM, f, s
import matplotlib.pyplot as plt

# Load the data
path = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/"
lam = 2
n_splines = 4
n_cols = 4
downsample = 1000
cellrank_obsms = 'palantir_pseudotime_slalom'
clusters = ['NSCLC', 'SCLC-A', 'SCLC-N']
samples = ['RU1083', 'RU1444', 'RU581', 'RU831', 'RU942', 'RU263', 'RU151', 'RU1303','RU1518', 'RU1646']
n_samples = len(samples)
n_rows = (n_samples + n_cols - 1) // n_cols
color_map = {"NSCLC": "gold", "SCLC-A": "tab:red", "SCLC-N": "tab:cyan"}
data = """
RU151,NSCLC_1,SCLC-N_2,SCLC-A,NonNE SCLC_1
RU263,NSCLC_2,SCLC-A_3,SCLC-N_1,SCLC-N_3
RU581,NSCLC,SCLC-A_4,SCLC-A_5,SCLC-N
RU831,NSCLC,SCLC-A_5,SCLC-A_6,SCLC-A_3,SCLC-N
RU942,NSCLC_1,NSCLC_3,SCLC-A_3,SCLC-N_2,SCLC-N_1
RU1042,SCLC-A,SCLC-N_5,SCLC-N_7
RU1083,NSCLC,SCLC-A_3,SCLC-A_1,SCLC-N_4
RU1181,SCLC-A,SCLC-N_6
RU1215,SCLC-A,SCLC-N_2,SCLC-N_3
RU1250,RB1-proficient NSCLC_4,RB1-deficient NSCLC_1,RB1-deficient NSCLC_4,RB1-deficient NSCLC_2
RU1293,NSCLC,SCLC-N_6,SCLC-N_7
RU1303,NSCLC_4,SCLC-A_1,SCLC-A_2
RU1304,SCLC-A,SCLC-N_7,SCLC-N_5
RU1444,NSCLC,SCLC-A_1,SCLC-N_3,UBA52+ SCLC_2
RU1518,NSCLC_1,SCLC-N_8
RU1676,SCLC-A,SCLC-N_7
RU1646,NSCLC,SCLC-N_3
"""
lines = data.strip().split('\n')
result_dict = {}
for line in lines:
    parts = line.split(',')
    key = parts[0]
    values = parts[1:]
    result_dict[key] = values

h5ad_files = [os.path.join(path, x + ".no_cc.hvg_2000.090124.h5ad") if x != 'RU263' else os.path.join(path, x + ".no_cc.hvg_2000.curated.120624.h5ad") for x in samples]
combined_adata, df = create_cellrank_probability_df(h5ad_files, 'cell_type_final2', samples, clusters, 
                                                    cellrank_obsms, 
                                                    cellrank_cols_dict=result_dict, downsample=downsample, need_all_clusters=False)
df['score'] = df.groupby('subject', observed=True)['score'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
df.to_csv("/scratch/chanj3/wangm10/NSCLC_SCLC-A_SCLC-N.csv", index=False)
# compute the GAM
gene = 'ASCL1'
score_col = 'score'
df_gene = df[['subject', score_col, gene]]
df_gene.columns = ['subject', score_col, "expression"]
df_gene["expression"] = df_gene.groupby("subject")["expression"].transform(
        lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten()
    )
df_gene.sort_values(by=score_col, inplace=True)
x = df_gene[[score_col, "subject"]].copy()
x["subject"] = x["subject"].astype("category").cat.codes
y = df_gene["expression"].values

# Define and fit the GAM with subject as a factor
n_splines = 4
lam = 2
gam = LinearGAM(s(0, n_splines=n_splines, lam=lam) + f(1)).fit(x, y)
y_pred = gam.predict(x)

# Compute deviance residuals for each sample
deviance_residuals = gam.deviance_residuals(x, y)

subject_ids = df_gene['subject']  # Use original subject names for clarity
summary_per_patient = pd.DataFrame({
    'subject': subject_ids,
    'deviance_residuals': deviance_residuals
})
size_factors = summary_per_patient.groupby('subject')['deviance_residuals'].count() / len(summary_per_patient)
summary_per_patient = summary_per_patient.groupby('subject').median()
summary_per_patient.reset_index(inplace=True)
summary_per_patient['size_factor'] = summary_per_patient['subject'].map(size_factors)
summary_per_patient['normalized_deviance'] = summary_per_patient['deviance_residuals'] / summary_per_patient['size_factor']

# Plot the summarized deviance residuals per patient
plt.figure(figsize=(6, 5))
plt.bar(summary_per_patient['subject'], summary_per_patient['normalized_deviance'], color='skyblue')
plt.xlabel('Patient')
plt.ylabel('Mean Deviance Residuals')
plt.title('Mean Deviance Residuals per Patient')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a scatter plot of gene expression vs aligned score, colored by patient
unique_subjects = df['subject'].unique()
plt.figure(figsize=(6, 5))

for subject in unique_subjects:
    # Filter data for the current patient
    subject_mask = df['subject'] == subject
    aligned_scores = df.loc[subject_mask, score_col]
    expressions = df_gene.loc[subject_mask, 'expression']
    
    # Generate grid for smooth curve fitting
    XX = gam.generate_X_grid(term=0)  # Generate grid for aligned_score
    
    # Predict fitted values for the current patient
    subject_code = x.loc[subject_mask, "subject"].iloc[0]  # Get the categorical code for the subject
    XX[:, 1] = subject_code  # Set the subject code in the grid
    
    fitted_values = gam.predict(XX)
    
    # Plot raw data (scatter points)
    plt.scatter(aligned_scores, expressions, label=f'{subject}', alpha=0.1)
    
    # Plot fitted curve (smooth line)
    plt.plot(XX[:, 0], fitted_values, label=f'{subject}', linestyle='--')

# Customize plot appearance
plt.xlabel('Aligned Score')
plt.ylabel('Gene Expression')
plt.title(gene)
plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

## Subplot version
n_subjects = len(unique_subjects)
fig, axes = plt.subplots(nrows=n_subjects, ncols=1, figsize=(8, 4 * n_subjects), sharex=True)

if n_subjects == 1:  # If there's only one patient, axes is not iterable
    axes = [axes]

for ax, subject in zip(axes, unique_subjects):
    # Filter data for the current patient
    subject_mask = df['subject'] == subject
    aligned_scores = df.loc[subject_mask, score_col]
    expressions = df_gene.loc[subject_mask, 'expression']

    # Generate grid for smooth curve fitting
    XX = gam.generate_X_grid(term=0)  # Generate grid for aligned_score
    
    # Predict fitted values for the current patient
    subject_code = x.loc[subject_mask, "subject"].iloc[0]  # Get the categorical code for the subject
    XX[:, 1] = subject_code  # Set the subject code in the grid
    
    fitted_values = gam.predict(XX)
    
    # Plot raw data (scatter points)
    ax.scatter(aligned_scores, expressions, label=f'{subject} Raw Data', alpha=0.6)
    
    # Plot fitted curve (smooth line)
    ax.plot(XX[:, 0], fitted_values, label=f'{subject} Fitted Curve', linestyle='--', color='red')
    
    # Customize each subplot
    ax.set_title(f'Patient: {subject}')
    ax.set_xlabel('Aligned Score')
    ax.set_ylabel('Gene Expression')
    ax.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()