# read cellphonedb results and save as csv
import os
import pandas as pd
import glob
from utils.plotting_helpers import split_interacting_pair

def read_tab_delimited_files(file_list):
    dfs = []
    for file in file_list:
        df = pd.read_csv(file, sep='\t')
        dfs.append(df)
    return dfs

def extract_ru_subdir(file_paths):
    ru_dirs = []
    for path in file_paths:
        parts = os.path.normpath(path).split(os.sep)
        ru_subdirs = [part for part in parts if part.startswith('RU')]
        ru_dirs.extend(ru_subdirs)
    return ru_dirs

#----------------------------------
# relevant interactions
base_dir = "/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.combined/out.individual.120122/"
# relevant_file_pattern = base_dir + "/*/cellphonedb.output/degs_analysis_relevant_interactions_042725.txt"
# file_list =  glob.glob(relevant_file_pattern)
# subject_list = extract_ru_subdir(file_list)
# dfs = read_tab_delimited_files(file_list)

# long_df_list = []
# for i, df in enumerate(dfs):
#         interacting_pairs = df['interacting_pair']
#         classification = df['classification']
#         df = df.iloc[:,13:]
#         df.loc[:,'interacting_pair'] = interacting_pairs
#         df.loc[:,'classification'] = classification
#         df.loc[:,'subject'] = subject_list[i]
#         df = df.reset_index(drop=True)
#         long_df = pd.melt(df, id_vars=['interacting_pair', 'subject', 'classification'], var_name='pair', value_name='relevance')
#         long_df[['source', 'target']] = long_df['pair'].str.split('|', expand=True)
#         long_df = long_df.drop(columns=['pair'])
#         long_df['sample'] = long_df['source'].str.split(r'[.]').str[1]
#         long_df['source'] = long_df['source'].str.split(r'[.]').str[0]
#         long_df['target'] = long_df['target'].str.split(r'[.]').str[0]
#         long_df.loc[:,'ligand_complex'] = long_df['interacting_pair'].apply(lambda x: split_interacting_pair(x)[0])
#         long_df.loc[:,'receptor_complex'] = long_df['interacting_pair'].apply(lambda x: split_interacting_pair(x)[1])
#         long_df_list.append(long_df)
# relevance_long_df = pd.concat(long_df_list, ignore_index=True)
# relevance_long_df.to_csv('../data/processed/cellphonedb_v5_relevance.csv', index=False)

#----------------------------------
# TF accessibility
# cell_sign_file = base_dir + "/*/cellphonedb.output/degs_analysis_CellSign_active_interactions_deconvoluted_042725.txt"
# cell_sign_file_list = glob.glob(cell_sign_file)
# cell_sign_dfs = read_tab_delimited_files(cell_sign_file_list)
# cell_sign_long_df_list = []
# cell_sign_subject_list = extract_ru_subdir(cell_sign_file_list)

# for i, df in enumerate(cell_sign_dfs):
#         df = df.loc[:,['interacting_pair', 'celltype_pairs', 'active_TF']]
#         df.loc[:,'subject'] = cell_sign_subject_list[i]
#         df = df.reset_index(drop=True)
#         df[['source', 'target']] = df['celltype_pairs'].str.split('|', expand=True)
#         df = df.drop(columns=['celltype_pairs'])
#         df['sample'] = df['source'].str.split(r'[.]').str[1]
#         df['source'] = df['source'].str.split(r'[.]').str[0]
#         df['target'] = df['target'].str.split(r'[.]').str[0]
#         df.loc[:,'ligand_complex'] = df['interacting_pair'].apply(lambda x: split_interacting_pair(x)[0])
#         df.loc[:,'receptor_complex'] = df['interacting_pair'].apply(lambda x: split_interacting_pair(x)[1])
#         df.loc[:,'cellphonedb_cellsign'] = 1
#         cell_sign_long_df_list.append(df)
# cell_sign_long_df = pd.concat(cell_sign_long_df_list, ignore_index=True)
# cell_sign_long_df.to_csv('../data/processed/cellphonedb_v5_cellsign.csv', index=False)

# #----------------------------------
# interaction scores
interaction_scores_file = base_dir + "/*/cellphonedb.output/degs_analysis_interaction_scores_042725.txt"
interaction_scores_file_list = glob.glob(interaction_scores_file)
interaction_scores_dfs = read_tab_delimited_files(interaction_scores_file_list)
interaction_scores_long_df_list = []
interaction_scores_subject_list = extract_ru_subdir(interaction_scores_file_list)

for i, df in enumerate(interaction_scores_dfs):
    interacting_pairs = df['interacting_pair']
    classification = df['classification']
    df = df.iloc[:,13:]
    df.loc[:,'interacting_pair'] = interacting_pairs
    df.loc[:,'classification'] = classification
    df.loc[:,'subject'] = interaction_scores_subject_list[i]
    df = df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    long_df = pd.melt(df, id_vars=['interacting_pair', 'subject', 'classification'], var_name='pair', value_name='interaction_score')
    long_df[['source', 'target']] = long_df['pair'].str.split('|', expand=True)
    long_df = long_df.drop(columns=['pair'])
    long_df['sample'] = long_df['source'].str.split(r'[.]').str[1]
    long_df['source'] = long_df['source'].str.split(r'[.]').str[0]
    long_df['target'] = long_df['target'].str.split(r'[.]').str[0]
    long_df.loc[:,'ligand_complex'] = long_df['interacting_pair'].apply(lambda x: split_interacting_pair(x)[0])
    long_df.loc[:,'receptor_complex'] = long_df['interacting_pair'].apply(lambda x: split_interacting_pair(x)[1])
    interaction_scores_long_df_list.append(long_df)
interaction_scores_long_df = pd.concat(interaction_scores_long_df_list, ignore_index=True)
interaction_scores_long_df.to_csv('../data/processed/cellphonedb_v5_interaction_scores.csv', index=False)
