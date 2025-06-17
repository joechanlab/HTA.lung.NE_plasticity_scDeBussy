import math
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import networkx as nx
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_interaction_heatmap(type_summary, output_file='interaction_heatmap.png', top_n=30, figsize=(12, 6),
                              row_order=None, pval_threshold=None, 
                              facet_by=None, facet_order=None):
    

    if pval_threshold is not None:
        type_summary = type_summary[type_summary['fisher_pval'] < pval_threshold].copy()

    top_interactions = type_summary.head(top_n).copy()
    top_interactions[['sender', 'receiver']] = top_interactions['sender_receiver_pair'].str.split('→', expand=True)
    classification_map = top_interactions.set_index('interacting_pair')['classification'].to_dict()
    unique_classes = sorted(set(classification_map.values()))
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    class_to_color = {cls: distinct_colors[i % len(distinct_colors)] for i, cls in enumerate(unique_classes)}
    
    if facet_by in ['sender', 'receiver']:
        grouped = dict(tuple(top_interactions.groupby(facet_by)))
        if facet_order is None:
            facet_order = sorted(grouped.keys())
        facet_groups = {k: grouped[k] for k in facet_order if k in grouped}
    else:
        facet_groups = {'All': top_interactions}

    if row_order is None:
        all_ordered_rows = []
        used_rows = set()
        for _, group in facet_groups.items():
            for col in group['sender_receiver_pair'].unique():
                subset = group[group['sender_receiver_pair'] == col]
                nonzero = subset['interacting_pair'].drop_duplicates().tolist()
                for r in nonzero:
                    if r not in used_rows:
                        all_ordered_rows.append(r)
                        used_rows.add(r)
        row_order = all_ordered_rows

    # Width ratios
    facet_col_counts = [len(g['receiver' if facet_by == 'sender' else 'sender'].unique())
                        for g in facet_groups.values()]
    total_cols = sum(facet_col_counts)
    width_ratios = [count / total_cols for count in facet_col_counts]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, len(facet_groups), width_ratios=width_ratios)
    axs = [fig.add_subplot(gs[0, i]) for i in range(len(facet_groups))]

    heatmap_refs = []  # to keep a reference for colorbar scaling

    for i, (ax, (facet_val, group)) in enumerate(zip(axs, facet_groups.items())):
        opposite = 'receiver' if facet_by == 'sender' else 'sender'
        group['facet_column'] = group[opposite]

        heatmap_data = group.pivot(index='interacting_pair', columns='facet_column', values='freq_diff').fillna(0)
        tf_annot_matrix = group.pivot_table(
            index='interacting_pair',
            columns='facet_column',
            values='active_TFs',
            aggfunc=lambda x: ';'.join(set(filter(None, x))) if any(x) else '',
            fill_value=''
        )

        heatmap_data = heatmap_data.reindex(row_order)
        tf_annot_matrix = tf_annot_matrix.reindex(row_order)

        hm = sns.heatmap(
            heatmap_data,
            cmap='RdBu_r',
            center=0,
            ax=ax,
            annot=tf_annot_matrix,
            fmt='',
            annot_kws={'color': 'black', 'weight': 'bold', 'size': 10, 'rotation': 0},
            cbar=False,  # 🔁 No individual colorbars
            mask=heatmap_data == 0,
            yticklabels=True if i == 0 else False
        )
        heatmap_refs.append(hm)

        # Light border around each facet
        for side in ["left", "right", "top", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_edgecolor("lightgray")
            ax.spines[side].set_linewidth(1)

        if i == 0:
            for label in ax.get_yticklabels():
                cls = classification_map.get(label.get_text(), 'Unclassified')
                label.set_color(class_to_color.get(cls, 'black'))
        ax.tick_params(axis='y', length=0)
        ax.set_title(f"{facet_val}", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    # Shared colorbar appended to last facet
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r')
    sm.set_array([-1, 1])  # Dummy range, just for colorbar scale
    fig.colorbar(sm, cax=cax, label='Frequency Difference\n(ADC → SCLC - De novo)')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, label=cls, marker='s', linestyle='None', markersize=10)
        for cls, color in class_to_color.items()
    ]
    fig.legend(handles=legend_elements, title='Classification', bbox_to_anchor=(1.01, 1), loc='upper left')

    # Axis titles
    fig.text(0.5, 0.04, f"{'Receiver' if facet_by == 'sender' else 'Sender'}", ha='center', fontsize=14)
    
    # Align facet label above y axis
    renderer = fig.canvas.get_renderer()
    bbox = axs[0].get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    fig.text(bbox.x0, 0.96, f"{facet_by.capitalize()}:", ha='left', fontsize=15)

    fig.tight_layout(rect=[0, 0.06, 0.9, 0.95])
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Faceted heatmap saved to {output_file}")

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


def plot_lr_network_unweighted(
    dataframe,
    top_n=20,
    weight_column='num_patients',
    title='Ligand–Receptor Network',
    figsize=(10, 10),
    layout='kamada',
    layout_k=1.0,
    layout_seed=42,
    save_path=None
):
    """
    Plot a hierarchical ligand–receptor network graph with classification nodes.
    Each connected component is laid out separately and shifted to improve clarity.
    """

    top_pairs = dataframe.head(top_n)
    G = nx.DiGraph()

    unique_classifications = sorted(set(top_pairs['classification']))
    for cls in unique_classifications:
        G.add_node(cls, type='classification', classification=cls)

    for _, row in top_pairs.iterrows():
        ligand, receptor = split_interacting_pair(row['interacting_pair'])
        cls = row['classification']
        for node, ntype in [(ligand, 'ligand'), (receptor, 'receptor')]:
            if node not in G:
                G.add_node(node, type=ntype, classification=cls)
        G.add_edge(cls, ligand)
        G.add_edge(cls, receptor)
        G.add_edge(ligand, receptor)

    classification_colors = {
        cls: plt.cm.Set3(i / len(unique_classifications))
        for i, cls in enumerate(unique_classifications)
    }

    node_colors, node_edge_colors = [], []
    for node in G.nodes():
        cls = G.nodes[node]['classification']
        if G.nodes[node]['type'] == 'classification':
            node_colors.append('white')  # Hollow fill
            node_edge_colors.append(classification_colors[cls])  # Colored border
        else:
            node_colors.append(classification_colors[cls])  # Solid fill
            node_edge_colors.append('black')  # Optional: black border for contrast

    # ---- Layout: separate per connected component ----
    components = list(nx.connected_components(G.to_undirected()))
    pos = {}

    # Grid layout parameters
    n_components = len(components)
    n_cols = math.ceil(math.sqrt(n_components))
    n_rows = math.ceil(n_components / n_cols)
    grid_spacing = 4.0  # adjust to avoid overlap

    component_positions = {}
    component_index = 0

    for row in range(n_rows):
        for col in range(n_cols):
            if component_index >= n_components:
                break
            component_positions[component_index] = (col * grid_spacing, -row * grid_spacing)
            component_index += 1

    # Position each component with grid offsets
    pos = {}
    for i, comp in enumerate(components):
        subgraph = G.subgraph(comp)

        if layout == 'kamada':
            sub_pos = nx.kamada_kawai_layout(subgraph, scale=1.0)
        elif layout == 'spring':
            sub_pos = nx.spring_layout(subgraph, k=layout_k, seed=layout_seed)
        elif layout == 'circular':
            sub_pos = nx.circular_layout(subgraph)
        elif layout == 'spectral':
            sub_pos = nx.spectral_layout(subgraph)
        elif layout == 'bipartite':
            for node in subgraph.nodes():
                subgraph.nodes[node]['subset'] = 0 if G.nodes[node]['type'] == 'classification' else 1
            sub_pos = nx.multipartite_layout(subgraph, subset_key='subset')
        else:
            raise ValueError("Unsupported layout type.")

        # Spread classification nodes outward
        center_x = np.mean([x for x, y in sub_pos.values()])
        center_y = np.mean([y for x, y in sub_pos.values()])
        for node in subgraph.nodes():
            if G.nodes[node]['type'] == 'classification':
                x, y = sub_pos[node]
                dx, dy = x - center_x, y - center_y
                norm = np.hypot(dx, dy) + 1e-3
                sub_pos[node] = (center_x + dx / norm * 1.5, center_y + dy / norm * 1.5)

        # Shift subgraph to grid cell
        grid_x, grid_y = component_positions[i]
        for node, (x, y) in sub_pos.items():
            pos[node] = (x + grid_x, y + grid_y)

    # ---- Plotting ----
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Draw edges
    directed_edges = []
    undirected_edges = []
    for u, v in G.edges():
        if G.nodes[u]['type'] == 'classification' or G.nodes[v]['type'] == 'classification':
            undirected_edges.append((u, v))
        else:
            directed_edges.append((u, v))

    for u, v in undirected_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], linestyle='--', color='gray', alpha=0.6, linewidth=1.0)

    for u, v in directed_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        ax.arrow(x1, y1, dx, dy,
                 length_includes_head=True,
                 head_width=0.05, head_length=0.08,
                 fc='gray', ec='gray', alpha=0.8,
                 linewidth=2, zorder=0)

    node_sizes = [1000 if G.nodes[n]['type'] == 'classification' else 400 for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edge_colors,
        linewidths=2,
        alpha=0.8
    )

    label_pos = {
        n: (x * 1.1, y * 1.1) if G.nodes[n]['type'] == 'classification' else (x, y + 0.05)
        for n, (x, y) in pos.items()
    }
    nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=classification_colors[cls],
                   markersize=15, label=cls)
        for cls in unique_classifications
    ] + [
        plt.Line2D([0], [0], linestyle='--', color='gray', label='Undirected (classification → L/R)'),
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Directed (ligand → receptor)', marker='>', markevery=[1])
    ]
    ax.legend(handles=legend_elements, title='Legend', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_rank_histograms(liana_df, figsize=(12, 5), save_path=None):
    """
    Plot side-by-side histograms of specificity and magnitude ranks.
    
    Parameters:
    -----------
    liana_df : pandas.DataFrame
        DataFrame containing computed_specificity_rank and computed_magnitude_rank columns
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot. If None, plot will be displayed.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot specificity rank histogram
    ax1.hist(liana_df['computed_specificity_rank'], bins=50, color='#1f77b4', alpha=0.7)
    ax1.set_title('Specificity Rank Distribution')
    ax1.set_xlabel('Specificity Rank')
    ax1.set_ylabel('Count')
    
    # Plot magnitude rank histogram
    ax2.hist(liana_df['computed_magnitude_rank'], bins=50, color='#ff7f0e', alpha=0.7)
    ax2.set_title('Magnitude Rank Distribution')
    ax2.set_xlabel('Magnitude Rank')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()