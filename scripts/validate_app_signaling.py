import scanpy as sc
import os
from constants import sample_to_group
from utils.plotting_helpers import plot_lr_validation

# Load the adata and create validation plots
base_dir = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/out.individual.120122/"
cell_type_col = 'cell_type_fine'
combined_adata = None

patient_list = ["RU1068", "RU1083", "RU1144", "RU1444",  "RU151", "RU263", "RU325", "RU581", "RU831"]
senders = ['SCLC-N', 'SCLC-A']
receivers = ['NSCLC', 'NonNE SCLC']
ligands = ['APP']
receptors = ['SORL1', 'TNFRSF21', 'CD74']

for patient in patient_list:
    adata = sc.read_h5ad(os.path.join(base_dir, f"{patient}/adata.{patient}.h5ad"))
    # Filter for relevant cell types
    cell_types = list(set(senders + receivers))
    adata = adata[adata.obs[cell_type_col].isin(cell_types), :]
    # Add patient information to cell type labels
    adata.obs['cell_type_patient'] = [f"{ct}_{patient}" for ct in adata.obs[cell_type_col]]
    
    if combined_adata is None:
        combined_adata = adata
    else:
        combined_adata = combined_adata.concatenate(adata)

combined_adata.obs['condition'] = combined_adata.obs['cell_type_patient'].apply(lambda x: sample_to_group[x.split('_')[1]])

# Create validation plots using combined adata
plot_lr_validation(
    combined_adata,
    ligands=ligands,
    receptors=receptors,
    senders=senders,
    receivers=receivers,
    cell_type_col=cell_type_col,
    save_file=f"../results/figures/APP_signaling_validation"
)

