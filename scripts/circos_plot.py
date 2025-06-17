import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from PIL import Image
import tempfile
from pycirclize import Circos

def create_circos_plot(relevance_df, interaction_type=None, output_file=None):
    """
    Create a circos plot for ligand-receptor interactions.
    
    Parameters:
    -----------
    relevance_df : pandas.DataFrame
        DataFrame containing the relevance data with columns:
        - subject
        - condition
        - sender
        - receiver
        - value
    interaction_type : str, optional
        Type of interaction to filter for ('within_tumor', 'myeloid_tumor', 'lymphoid_tumor')
    output_file : str, optional
        Path to save the output plot. If None, plot will be displayed but not saved.
        
    Returns:
    --------
    None
        Creates and displays/saves the circos plot
    """
    # Define cell type groups
    tumor_cells = {"NSCLC", "SCLC-A", "SCLC-N"}
    myeloid_cells = {"Mφ/Mono", "Mφ/Mono\nCD14", "Mφ/Mono\nCD11c", "Mφ/Mono\nCCL", "PMN", "Mast cell", "pDC"}
    lymphoid_cells = {"CD4_EM\nEffector", "CD4_naive\nCM", "CD4_TRM", "CD8_TRM\nEM", "T_reg", "NK", "ILC", "B_memory"}
    stromal_cells = {"Fibroblast", "Endothelial", "Basal", "Ciliated"}
    
    # Define tumor color map
    tumor_color_map = {
        "NSCLC": "gold",      # Yellow
        "SCLC-A": "tab:red",      # Red
        "SCLC-N": "tab:cyan",      # Cyan
        "SCLC-P": "tab:blue",      # Blue
        "NonNE SCLC": "tab:purple",  # Purple
    }
    
    def get_interaction_type(sender, receiver):
        """Helper function to determine interaction type"""
        if sender in tumor_cells and receiver in tumor_cells:
            return "within_tumor"
        elif (sender in myeloid_cells and receiver in tumor_cells) or (sender in tumor_cells and receiver in myeloid_cells):
            return "myeloid_tumor"
        elif (sender in lymphoid_cells and receiver in tumor_cells) or (sender in tumor_cells and receiver in lymphoid_cells):
            return "lymphoid_tumor"
        return "other"
    
    # Add interaction type if not already present
    if 'interaction_type' not in relevance_df.columns:
        relevance_df['interaction_type'] = relevance_df.apply(
            lambda row: get_interaction_type(row['sender'], row['receiver']), axis=1
        )
    
    # Filter by interaction type if specified
    if interaction_type:
        relevance_df = relevance_df[relevance_df['interaction_type'] == interaction_type]
        if relevance_df.empty:
            print(f"No interactions found for type: {interaction_type}")
            return
    
    # Create pivot table
    pivot = relevance_df.pivot_table(
        index=['subject', 'condition', 'sender'],
        columns='receiver',
        values='value',
        aggfunc='sum',
        fill_value=0
    ).reset_index().rename_axis(None, axis=1)
    
    # Calculate median values
    median = pivot.drop(columns=['subject']).groupby(['condition', 'sender']).median()
    all_senders = median.index.get_level_values('sender').unique().sort_values()
    
    # Create condition matrices
    condition_matrices = {}
    for condition in ["De novo SCLC and ADC", "ADC → SCLC"]:
        if condition in median.index.get_level_values('condition'):
            cond_df = median.loc[condition]
            cond_df_reindexed = cond_df.reindex(all_senders, fill_value=0)
            condition_matrices[condition] = cond_df_reindexed
    
    # Get all unique nodes
    all_nodes = set()
    for matrix_df in condition_matrices.values():
        all_nodes.update(matrix_df.index)
        all_nodes.update(matrix_df.columns)
    all_nodes = sorted(all_nodes)
    
    # Create color mapping
    cmap = cm.get_cmap('tab20', len(all_nodes))
    node2color = {}
    for node in all_nodes:
        if node in tumor_color_map:
            node2color[node] = tumor_color_map[node]
        else:
            node2color[node] = mcolors.to_hex(cmap(all_nodes.index(node)))
    
    # Create plots for each condition
    images = []
    labels = []
    for condition, matrix_df in condition_matrices.items():
        matrix_df = matrix_df.fillna(0)
        matrix_df = matrix_df.reindex(index=all_nodes, columns=all_nodes, fill_value=0)
        matrix_df = matrix_df.loc[(matrix_df != 0).any(axis=1), (matrix_df != 0).any(axis=0)]
        filtered_order = [node for node in all_nodes if node in matrix_df.index and node in matrix_df.columns]
        
        if matrix_df.empty or len(filtered_order) == 0:
            print(f"Skipping {condition}: matrix is empty after filtering.")
            continue
        
        # Create circos plot
        circos = Circos.chord_diagram(
            matrix_df,
            space=19,
            cmap=node2color,
            label_kws=dict(size=25),
            link_kws=dict(ec="black", lw=0.5, direction=1),
        )
        
        # Save to temporary file
        tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        circos.savefig(tmpfile.name)
        images.append(Image.open(tmpfile.name))
        labels.append(condition)
    
    # Create final figure
    if images:
        fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
        if len(images) == 1:
            axes = [axes]
        
        for ax, img, label in zip(axes, images, labels):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{label}", fontsize=21)
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No valid data to plot")

