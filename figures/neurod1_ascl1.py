import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LinearGAM, s, f
from scdebussy.tl import aligner, gam_smooth_expression
from scdebussy.pp import create_cellrank_probability_df
from scdebussy.pl import plot_sigmoid_fits, plot_summary_curve
import numpy as np
from matplotlib.patches import Patch
import re

path = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/individual.090124/"

n_cols = 4
downsample = 5000
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
df['score'] = df.groupby('subject')['score'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
lam = 3
n_splines = 5
df_samples = []
for gene in ['ASCL1', 'NEUROD1']:
    for subject in df.subject.unique():
        df_sample = df[df['subject'] == subject][['score', gene]]
        df_sample.columns = ['score', 'expression']
        df_sample['expression'] = df_sample['expression'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else pd.Series(0, index=x.index)
        )
        gam = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(df_sample.score, df_sample.expression)
        df_sample['smoothed'] = gam.predict(df_sample.score)
        df_sample['gene'] = gene
        df_sample['subject'] = subject
        df_samples.append(df_sample)
df_ascl1_neurod1 = pd.concat(df_samples)
df_ascl1_neurod1.reset_index(drop=True, inplace=True)

to_plot='smoothed'
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), squeeze=False)
axes = axes.flatten()  # Flatten axes array for easier indexing
for i, sample in enumerate(samples):
    ax = axes[i]
    print(sample)
    df_sample = df[df['subject'] == sample]
    df_smoothed = df_ascl1_neurod1[df_ascl1_neurod1['subject'] == sample]
    
    if i == 0:
        sns.lineplot(x='score', y=to_plot, hue='gene', data=df_smoothed, 
                    ax=ax, palette={"ASCL1": "tab:red", "NEUROD1": "tab:cyan"}, alpha=0.5)
                                
        handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()
    else: 
        sns.lineplot(x='score', y=to_plot, hue='gene', data=df_smoothed, 
                    ax=ax, palette={"ASCL1": "tab:red", "NEUROD1": "tab:cyan"}, legend=False, alpha=0.5)
        
    # Add title and labels to the subplot
    expression_max = np.nanmax(df_ascl1_neurod1[to_plot])
    expression_min = np.nanmin(df_ascl1_neurod1[to_plot])
    bar_space = (expression_max - expression_min) * 0.1
    y_max = expression_max + 0.1
    y_min = expression_min - bar_space * 2
    x_min = -0.1
    x_max = 1.1
    cell_types = pd.Categorical(df_sample['cell_type'].unique(), ordered=True, categories=color_map.keys())
    cell_types = cell_types.sort_values()
    n_cell_types = 3

    for k, cell_type in enumerate(cell_types):
        subset = df_sample[df_sample['cell_type'] == cell_type]
        ax.bar(subset['score'], height=bar_space/n_cell_types, bottom=y_min + (n_cell_types-k)/n_cell_types * bar_space, width=0.01, 
            color=color_map[cell_type], edgecolor=None, align='edge', alpha=0.2)
    
    print(y_min, y_max, x_min, x_max)
    ax.set_ylim([y_min, y_max])
    ax.set_xlim([x_min, x_max])
    ax.set_title(f'{sample}', fontsize=20)
    ax.set_xlabel('Pseudotime', fontsize=20)
    ax.set_ylabel('Expression', fontsize=20)

    if i % n_cols != 0:
        ax.set_ylabel('')
    if i < (n_rows - 1) * n_cols:
        ax.set_xlabel('')
    
# Hide any unused subplots if the number of samples is less than rows * cols
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

handles_gene, labels_gene = axes[0].get_legend_handles_labels()  # Get legend info from the first subplot
for legend_handle in handles_gene:
    legend_handle.set_alpha(1)

# Add legend for the barplot marks
handles_bar = [Patch(color=color_map[cell_type], label=cell_type) for cell_type in color_map.keys()]
fig.legend(handles_gene + handles_bar, labels_gene + list(color_map.keys()), loc='lower right', title="", bbox_to_anchor=(0.8, 0.1), frameon=False, fontsize=20)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(f'neurod1_ascl1_lam_{lam}_splines_{n_splines}.png', dpi=300, bbox_inches='tight')
plt.show()

#--------------
# Compare individual trends with aggregate trend
import numpy as np
import pandas as pd
from pygam import LinearGAM, s, f
import matplotlib.pyplot as plt

def angular_similarity(x, y):
    # Convert time series into vectors
    x_diff = np.diff(x)
    y_diff = np.diff(y)
    
    # Normalize vectors
    x_norm = x_diff / np.linalg.norm(x_diff)
    y_norm = y_diff / np.linalg.norm(y_diff)
    
    # Compute cosine similarity between vector directions
    similarity = np.dot(x_norm, y_norm)
    return similarity


lam = 3
n_splines = 6

for gene in ['ASCL1', 'NEUROD1']:
    print(gene)
    
    # Encode subject as categorical variable for aggregate GAM fitting
    df['subject'] = df['subject'].astype('category').cat.codes
    gam_aggregate = LinearGAM(s(0, n_splines=n_splines, lam=lam) + f(1)).fit(df[['score', 'subject']], df[gene])
    
    # Evaluate trends at evenly spaced pseudotime points
    pseudotime_points = np.linspace(0, 1, 100)
    patient_map = dict(zip(df.subject.unique(), samples))
    
    # Compute angular similarities for each subject
    distances = {}
    
    # Plot individual trends for each sample in separate panels
    n_cols = 4
    n_samples = len(df.subject.unique())
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, subject in enumerate(df.subject.unique()):
        ax = axes[i]
        
        # Fit individual GAM for the subject
        gam_individual = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(
            df[df['subject'] == subject]['score'], 
            df[df['subject'] == subject][gene]
        )
        individual_trend = gam_individual.predict(pseudotime_points)
        
        # Predict aggregate trend for the same subject
        aggregate_trend = gam_aggregate.predict(np.column_stack((pseudotime_points, [subject] * len(pseudotime_points))))
        
        # Compute angular similarity
        distance = angular_similarity(individual_trend, aggregate_trend)
        distances[subject] = distance
        
        # Plot individual trend
        ax.plot(pseudotime_points, individual_trend, label=f'{patient_map[subject]} (Individual)', color='blue', alpha=0.7)
        
        # Plot aggregate trend on the same panel
        ax.plot(pseudotime_points, aggregate_trend, label='Aggregate Trend', color='red', linestyle='--', linewidth=2)
        
        # Annotate panel title with gene name and angular similarity
        ax.set_title(f'{patient_map[subject]} ({gene})\nAngular Sim: {distance:.2f}', fontsize=10)
    
    # Hide unused subplots if there are fewer samples than rows * cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.supxlabel('Pseudotime', fontsize=12)
    fig.supylabel('Expression', fontsize=12)
    fig.suptitle(f'Individual Trends with Aggregate Trend for {gene}', fontsize=16)
    
    # Add a legend to one of the subplots (e.g., the first one)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    
    plt.tight_layout()
    plt.show()
    
    # Save distances to a DataFrame and print it
    distances_df = pd.DataFrame.from_dict(distances, orient='index', columns=['Angular Similarity'])
    distances_df['Sample'] = distances_df.index.map(patient_map)
    
    print(distances_df)

#-------
# Investigate how often ASCL1 comes before NERUOD1
import numpy as np
import pandas as pd
from pygam import LinearGAM, s

# Function to compute the first derivative
def compute_derivative(y, x):
    return np.gradient(y, x)

# Function to find the first increasing time point
def find_first_increasing(pseudotime, trend):
    derivative = compute_derivative(trend, pseudotime)
    increasing_points = np.where(derivative > 0)[0]
    return pseudotime[increasing_points[0]] if len(increasing_points) > 0 else None

# Function to determine if ASCL1 increases before NEUROD1
def analyze_increase_order(pseudotime, ascl1_trend, neurod1_trend):
    # Identify points where the trends start increasing
    first_ascl1 = find_first_increasing(pseudotime, ascl1_trend)
    first_neurod1 = find_first_increasing(pseudotime, neurod1_trend)
    
    # Handle cases where one or both trends are flat
    if first_ascl1 is None and first_neurod1 is None:
        order = "Both Flat"
    elif first_ascl1 is None:
        order = "NEUROD1_before_ASCL1 (ASCL1 Flat)"
    elif first_neurod1 is None:
        order = "ASCL1_before_NEUROD1 (NEUROD1 Flat)"
    elif first_ascl1 < first_neurod1:
        order = "ASCL1_before_NEUROD1"
    elif first_neurod1 < first_ascl1:
        order = "NEUROD1_before_ASCL1"
    else:
        order = "Simultaneous"
    
    return {"order": order, "first_ascl1": first_ascl1, "first_neurod1": first_neurod1}

# Parameters for GAM fitting
lam = 3
n_splines = 6

# Example dataset structure: df contains pseudotime and gene expression per patient
# Columns: ['score', 'ASCL1', 'NEUROD1', 'subject']
results = []

for patient in df['subject'].unique():
    # Filter data for the current patient
    df_patient = df[df['subject'] == patient]
    
    # Fit GAM models for ASCL1 and NEUROD1
    gam_ascl1 = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(df_patient['score'], df_patient['ASCL1'])
    gam_neurod1 = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(df_patient['score'], df_patient['NEUROD1'])
    
    # Predict smoothed trends over pseudotime
    pseudotime_points = np.linspace(df_patient['score'].min(), df_patient['score'].max(), 100)
    ascl1_trend = gam_ascl1.predict(pseudotime_points)
    neurod1_trend = gam_neurod1.predict(pseudotime_points)
    
    # Analyze increase order
    result = analyze_increase_order(pseudotime_points, ascl1_trend, neurod1_trend)
    result['patient'] = patient
    results.append(result)

# Convert results to DataFrame and summarize
results_df = pd.DataFrame(results)

# Count occurrences of each case
summary = results_df['order'].value_counts()
print("Summary of Events:")
print(summary)

# Display detailed results per patient
print("Detailed Results:")
print(results_df)


# Plotting
import matplotlib.pyplot as plt

# Group data points by their coordinates (first_ascl1, first_neurod1)
results_df['coordinates'] = list(zip(results_df['first_ascl1'], results_df['first_neurod1']))
grouped = results_df.groupby('coordinates')['patient'].apply(', '.join).reset_index()
grouped.columns = ['coordinates', 'patients']

# Split coordinates back into first_ascl1 and first_neurod1 for plotting
grouped['first_ascl1'] = grouped['coordinates'].apply(lambda x: x[0])
grouped['first_neurod1'] = grouped['coordinates'].apply(lambda x: x[1])

# Plotting with concatenated annotations
plt.figure(figsize=(8, 6))

# Scatter plot of unique coordinates
for _, row in grouped.iterrows():
    x = row['first_ascl1']
    y = row['first_neurod1']
    
    # Plot the point
    plt.scatter(x, y, s=150, alpha=0.8)
    
    # Annotate the point with concatenated patient IDs
    annotation_x = x + 0.02  # Offset for annotation
    annotation_y = y + 0.02
    plt.text(annotation_x, annotation_y, row['patients'], fontsize=10, color='blue')
    
    # Draw a line pointing to the annotation
    plt.plot([x, annotation_x], [y, annotation_y], color='black', linestyle='-', linewidth=0.5)

# Add diagonal line for simultaneous expression
plt.plot([0, 0.5], [0, 0.5], color='gray', linestyle='--', label='Simultaneous Increase')

# Add labels and legend
plt.xlabel('First ASCL1 Increase (Pseudotime)', fontsize=12)
plt.ylabel('First NEUROD1 Increase (Pseudotime)', fontsize=12)
plt.title('First Increasing Time Points of ASCL1 vs NEUROD1', fontsize=14)
plt.grid(alpha=0.3)

# Adjust axes limits and show plot
plt.xlim(-0.05, 0.5)  # Set x-axis maximum limit to 0.5
plt.ylim(-0.05, 0.5)  # Set y-axis maximum limit to 0.5
plt.tight_layout()
plt.show()