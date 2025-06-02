
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import networkx as nx

def plot_lr_validation(adata, ligands, receptors, senders, receivers, cell_type_col='cell_type_fine', save_file=None):
    """
    Create validation plots for ligand-receptor expression patterns.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix containing gene expression data
    ligands : list
        List of ligand genes to plot
    receptors : list
        List of receptor genes to plot
    senders : list
        List of sender cell types
    receivers : list
        List of receiver cell types
    cell_type_col : str
        Name of the column in adata.obs containing cell type information
    output_prefix : str
        Prefix for output files
    """
    
    # Get unique conditions and create a color map
    conditions = adata.obs['condition'].unique()
    
    # Create ordering based on condition
    cell_type_order = []
    for cond in conditions:
        # Get cell types for this condition
        cond_cell_types = adata.obs[adata.obs['condition'] == cond]['cell_type_patient'].unique()
        # Sort within condition by cell type
        cond_cell_types = sorted(cond_cell_types)
        cell_type_order.extend(cond_cell_types)
    
    # Set the category order in the AnnData object
    adata.obs['cell_type_patient'] = pd.Categorical(
        adata.obs['cell_type_patient'],
        categories=cell_type_order,
        ordered=True
    )
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    
    # Get unique genes
    unique_ligands = list(dict.fromkeys(ligands))  # Remove duplicates while preserving order
    unique_receptors = list(dict.fromkeys(receptors))  # Remove duplicates while preserving order
    
    # Filter genes to only include those present in adata
    unique_ligands = [gene for gene in unique_ligands if gene in adata.var_names]
    unique_receptors = [gene for gene in unique_receptors if gene in adata.var_names]
    
    print(f"\nNumber of ligands found in adata: {len(unique_ligands)}")
    print(f"Number of receptors found in adata: {len(unique_receptors)}")
    
    # Plot sender cells with ligand genes
    sender_mask = adata.obs[cell_type_col].isin(senders)
    sc.pl.dotplot(
        adata[sender_mask],
        var_names=unique_ligands,
        groupby='cell_type_patient',
        title='Ligand Expression in Sender Cells',
        ax=ax1,
        show=False
    )
    
    # Plot receiver cells with receptor genes
    receiver_mask = adata.obs[cell_type_col].isin(receivers)
    sc.pl.dotplot(
        adata[receiver_mask],
        var_names=unique_receptors,
        groupby='cell_type_patient',
        title='Receptor Expression in Receiver Cells',
        ax=ax2,
        show=False
    )
    
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(f'{save_file}_dotplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def split_interacting_pair(pair):
    if '_by' in pair:
        # Find index after the '_byXXX_' section
        parts = pair.split('_')
        for i, part in enumerate(parts):
            if part.startswith('by') and i + 1 < len(parts):
                split_idx = i  # ligand ends here
                break
        ligand = '_'.join(parts[:split_idx + 1])
        receptor = '_'.join(parts[split_idx + 1:])
    else:
        ligand, receptor = pair.split('_', 1)
    return ligand, receptor


def plot_lr_network(
    dataframe,
    top_n=20,
    weight_column='num_patients',
    title='Ligand–Receptor Network',
    figsize=(10, 10),
    layout='spring',
    layout_k=1.0,
    save_path=None
):
    """
    Plot a ligand–receptor network graph from a dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame with 'interacting_pair' and a weight column.
    top_n : int
        Number of top ligand–receptor pairs to plot.
    weight_column : str
        Column name to use for edge weights.
    title : str
        Plot title.
    figsize : tuple
        Size of the plot, e.g., (width, height).
    layout : str
        Layout type: 'spring', 'circular', or 'kamada'.
    layout_k : float
        For spring layout, controls node spread (larger = more spread out).
    save_path : str or None
        If given, save the plot to this file path.
    """

    # Select top N interactions
    top_pairs = dataframe.head(top_n)

    # Initialize directed graph
    G = nx.DiGraph()

    # Add nodes and edges
    for _, row in top_pairs.iterrows():
        ligand, receptor = split_interacting_pair(row['interacting_pair'])
        G.add_node(ligand, type='ligand')
        G.add_node(receptor, type='receptor')
        G.add_edge(ligand, receptor, weight=row[weight_column])

    # Set node colors
    node_colors = [
        'skyblue' if G.nodes[node]['type'] == 'ligand' else 'salmon'
        for node in G.nodes()
    ]

    # Set edge widths
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]

    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=layout_k, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada':
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError("Unsupported layout type. Use 'spring', 'circular', or 'kamada'.")

    # Plot
    plt.figure(figsize=figsize)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=800,
        edge_color='gray',
        width=edge_widths,
        arrowsize=20,
        font_size=10
    )

    plt.title(title, fontsize=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()