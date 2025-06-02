
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

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

