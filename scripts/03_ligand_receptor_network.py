from utils.plotting_helpers import plot_lr_network, plot_lr_network_unweighted
import pandas as pd

# Within tumors
# NSCLC → SCLC
df = pd.read_excel('../results/tables/nonne_sclc_frequent_interactions.xlsx', sheet_name='NSCLC as Sender')
plot_lr_network_unweighted(df, top_n=50, title='NSCLC → SCLC Ligand–Receptor Network', layout='spring', layout_k = 30, layout_seed=10, figsize=(15, 8))#, save_path='../results/figures/NSCLC_SCLC_LR_network.png')

# SCLC → NSCLC
df = pd.read_excel('../results/tables/nonne_sclc_frequent_interactions.xlsx', sheet_name='NSCLC as Receiver')
plot_lr_network_unweighted(df, top_n=200, title='SCLC → NSCLC Ligand–Receptor Network', layout='spring', layout_k = 30, layout_seed=10, figsize=(25, 8))#, save_path='../results/figures/SCLC_NSCLC_LR_network.png')

