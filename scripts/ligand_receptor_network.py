from utils.plotting_helpers import plot_lr_network
import pandas as pd

# Within tumors
# NSCLC → SCLC
df = pd.read_excel('../results/tables/nonne_sclc_frequent_interactions.xlsx', sheet_name='NSCLC as Sender')
plot_lr_network(df, top_n=50, title='NSCLC → SCLC Ligand–Receptor Network', layout='spring', figsize=(6, 6))#, save_path='../results/figures/NSCLC_SCLC_LR_network.png')

# SCLC → NSCLC
df = pd.read_excel('../results/tables/nonne_sclc_frequent_interactions.xlsx', sheet_name='NSCLC as Receiver')
plot_lr_network(df, top_n=50, title='SCLC → NSCLC Ligand–Receptor Network', layout='spring', figsize=(8, 8))#, save_path='../results/figures/SCLC_NSCLC_LR_network.png')

