import scanpy as sc
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.barycenters import softdtw_barycenter
os.chdir("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/results_all_genes/preprocessing/NSCLC_SCLC-A/")

# get some example data
df = pd.read_csv("cellrank.NSCLC_SCLC-A.csv", index_col=0)
df = df.sort_values(by=["subject", "score"], ascending=[True, True])
hvg_genes = ["ASCL1", "TACSTD2", "DLL3"]
df_genes = df.loc[:, ["subject", "score"] + hvg_genes]

# downsample cells to 1000 each randomly with seed
df_genes = df_genes.groupby("subject").apply(lambda x: x.sample(n=500, random_state=1)).reset_index(drop=True)
df_genes = df_genes.sort_values(by=["subject", "score"], ascending=[True, True])
subject_arrays = np.array([df_genes[df_genes['subject'] == subject].sort_values('score').iloc[:, 2:].values.tolist()
                 for subject in df_genes['subject'].unique()], dtype=object)

# Soft-DTW on all data
start_time = time.time()
barycenter = softdtw_barycenter(subject_arrays, gamma=1, weights=None, max_iter=100, tol=1e-3, init=None)
print("--- %s seconds ---" % (time.time() - start_time))

# plot the barycenter
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(barycenter.shape[1]):
    ax.plot(barycenter[:,i], label=hvg_genes[i])
ax.legend()
plt.show()


# Interpolate the expression (https://github.com/lindsaysmoore/cellAlign/blob/master/R/interWeights.R)
# Define the inter_weights function
def inter_weights(exp_data, traj_cond, win_sz, num_pts):
    # Remove NAs from the trajectory
    if np.isnan(traj_cond).any():
        exp_data = exp_data.loc[~np.isnan(traj_cond)]
        traj_cond = traj_cond[~np.isnan(traj_cond)]
    
    # Generate equally-spaced points along the trajectory
    traj_val_new_pts = np.linspace(min(traj_cond), max(traj_cond), num_pts)
    
    # Calculate weighted averages for each new point
    val_new_pts = np.array([
        np.sum(exp_data.values * np.exp(-((traj_cond - traj_pt) ** 2) / (win_sz ** 2))[:, np.newaxis], axis=0) / 
        np.sum(np.exp(-((traj_cond - traj_pt) ** 2) / (win_sz ** 2)))
        for traj_pt in traj_val_new_pts
    ])
    
    # Find closest interpolated point for each real data point
    closest_int = np.array([np.argmin(np.abs(traj_val_new_pts - traj_val)) for traj_val in traj_cond])
    
    # Calculate the error of the smoothing function
    err_per_gene = np.abs(exp_data.values - val_new_pts[closest_int])
    
    # Interpolate the error at each interpolated point
    err_interpolated = np.array([
        np.sum(err_per_gene * np.exp(-((traj_cond - traj_pt) ** 2) / (win_sz ** 2))[:, np.newaxis], axis=0) / 
        np.sum(np.exp(-((traj_cond - traj_pt) ** 2) / (win_sz ** 2)))
        for traj_pt in traj_val_new_pts
    ])
    
    return val_new_pts, err_interpolated, traj_val_new_pts

# Apply interpolation to each subject
subject_arrays = []
for subject in df_genes['subject'].unique():
    subject_data = df_genes[df_genes['subject'] == subject]
    traj_cond = subject_data['score'].values
    exp_data = subject_data[hvg_genes]
    val_new_pts, _, traj_val_new_pts = inter_weights(exp_data, traj_cond, win_sz=0.1, num_pts=10)
    subject_arrays.append(val_new_pts)

subject_arrays = np.array(subject_arrays, dtype=object)

# Soft-DTW on all data
start_time = time.time()
barycenter = softdtw_barycenter(subject_arrays, gamma=1, weights=None, max_iter=100, tol=1e-3, init=None)
print("--- %s seconds ---" % (time.time() - start_time))

# plot the barycenter
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(barycenter.shape[1]):
    ax.plot(barycenter[:,i], label=hvg_genes[i])
ax.legend()
plt.show()