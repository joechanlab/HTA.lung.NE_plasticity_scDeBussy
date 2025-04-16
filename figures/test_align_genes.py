import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.barycenters import softdtw_barycenter, dtw_barycenter_averaging
from tslearn.metrics import soft_dtw_alignment

class scDeBussy:
    def __init__(self, win_sz=0.3, num_pts=30, gamma=10):
        self.win_sz = win_sz
        self.num_pts = num_pts
        self.gamma = gamma
        self.barycenter = None
        self.subject_arrays = None
        self.traj_val_new_pts = None
        self.all_genes = None
        self.subjects = None

    def inter_weights(self, exp_data, traj_cond, win_sz=0.3, num_pts=30):
        """
        Interpolate gene expression data independently per gene in a concise and efficient manner.

        Parameters:
            exp_data (pd.DataFrame): Gene expression data (rows: samples, columns: genes).
            traj_cond (np.ndarray): Pseudotime values for each sample.
            win_sz (float): Window size for Gaussian weighting.
            num_pts (int): Number of equally spaced pseudotime points for interpolation.

        Returns:
            tuple: Interpolated values (val_new_pts), new pseudotime points (traj_val_new_pts).
        """
        # Remove NAs from the trajectory
        if (pd.isna(traj_cond).any()):
            valid_idx = ~pd.isna(traj_cond)
            exp_data = exp_data.iloc[valid_idx, :]
            traj_cond = traj_cond[valid_idx]

        # Generate equally spaced pseudotime points
        traj_val_new_pts = np.linspace(min(traj_cond), max(traj_cond), num_pts)

        # Compute pairwise distances between traj_cond and traj_val_new_pts
        dist_matrix = traj_cond[:, np.newaxis] - traj_val_new_pts[np.newaxis, :]  # Shape: (samples, new_pts)

        # Compute Gaussian weights
        weights = np.exp(-(dist_matrix**2) / (win_sz**2))  # Shape: (samples, new_pts)
        weights /= np.sum(weights, axis=0, keepdims=True)  # Normalize weights along samples

        # Perform weighted sum for all genes at once using matrix multiplication
        val_new_pts = np.dot(weights.T, exp_data.values)  # Shape: (new_pts, genes)

        return val_new_pts, traj_val_new_pts

    def prepare_data(self, df, genes, subject_col='subject', score_col='score', n_samples=500, random_state=1):
        """
        Prepare data with gene filtering based on low expression and coefficient of variation (CV).

        Parameters:
            df (pd.DataFrame): Input DataFrame containing gene expression data.
            genes (list): List of gene names to include.
            subject_col (str): Column name for subject identifiers.
            score_col (str): Column name for pseudotime/trajectory scores.
            n_samples (int): Maximum number of samples per subject.
            random_state (int): Random seed for reproducibility.
            low_expression_threshold (float): Minimum mean expression threshold for filtering.
            cv_threshold (float): Maximum CV threshold for filtering.

        Returns:
            self: Updated scDeBussy instance with filtered and interpolated data.
        """
        # Extract relevant columns
        df_genes = df.loc[:, [subject_col, score_col] + genes]
        
        # Sample data within each subject group
        def safe_sample(group):
            return group.sample(n=min(n_samples, len(group)), random_state=random_state)
        
        df_genes = df_genes.groupby(subject_col).apply(safe_sample).reset_index(drop=True)
        df_genes = df_genes.sort_values(by=[subject_col, score_col], ascending=[True, True])
        
        # Interpolate data for each subject
        subject_arrays = []
        subjects = df_genes[subject_col].unique()
        for subject in subjects:
            subject_data = df_genes[df_genes[subject_col] == subject]
            traj_cond = subject_data[score_col].values
            exp_data = subject_data[genes]
            
            # Clip extreme values based on percentiles
            lower_bound = exp_data.quantile(0.01)  # 1st percentile
            upper_bound = exp_data.quantile(0.99)  # 99th percentile
            exp_data_clipped = exp_data.clip(lower=lower_bound, upper=upper_bound, axis=1)
            
            # Apply min-max scaling after clipping
            exp_data_scaled = (exp_data_clipped - exp_data_clipped.min()) / (exp_data_clipped.max() - exp_data_clipped.min())
            exp_data_scaled = exp_data_scaled.fillna(0)  # Handle cases where max == min

            val_new_pts, traj_val_new_pts = self.inter_weights(exp_data_scaled, traj_cond, win_sz=self.win_sz, num_pts=self.num_pts)
            subject_arrays.append(val_new_pts)
        
        self.subject_arrays = np.array(subject_arrays, dtype=object)
        self.all_genes = genes
        self.traj_val_new_pts = traj_val_new_pts
        self.subjects = subjects
        return self

    def compute_barycenter(self, method='soft_dtw', max_iter=100, tol=1e-3):
        if self.subject_arrays is None:
            raise ValueError("Data not prepared. Call prepare_data first.")
            
        if method == 'soft_dtw':
            self.barycenter = softdtw_barycenter(
                self.subject_arrays, 
                gamma=self.gamma,
                max_iter=max_iter,
                tol=tol
            )
        elif method == 'dba':
            self.barycenter = dtw_barycenter_averaging(
                self.subject_arrays,
                max_iter=max_iter,
                tol=tol
            )
        return self

    def plot_original_expression(self, df_genes, genes, score_col='score', subject_col='subject'):
        fig, ax = plt.subplots(1, len(genes), figsize=(5*len(genes), 5))
        for i in range(len(genes)):
            for subject in df_genes[subject_col].unique():
                mask = df_genes[subject_col] == subject
                ax[i].scatter(
                    df_genes.loc[mask, score_col],
                    df_genes.loc[mask, genes[i]],
                    s=0.1,
                    label=subject
                )
            ax[i].legend()
            ax[i].set_title(genes[i])
        plt.show()

    def plot_interpolated_expression(self, genes):
        """
        Plot interpolated expression levels for each gene with a single color legend for subjects.
        """
        gene_indices = [self.all_genes.index(gene) for gene in genes]
        fig, ax = plt.subplots(1, len(genes), figsize=(5 * len(genes), 5))
        
        # Generate unique colors for each subject
        num_subjects = self.subject_arrays.shape[0]
        colors = plt.cm.tab10(np.linspace(0, 1, num_subjects))  # Use a colormap (e.g., tab10)
        
        # Collect handles and labels for the legend
        handles = []
        labels = []
        
        for i, gene_idx in enumerate(gene_indices):
            for j in range(num_subjects):
                # Plot each subject's data with its assigned color
                line, = ax[i].plot(self.traj_val_new_pts, self.subject_arrays[j][:, gene_idx], 
                                color=colors[j], label=f'Subject {j}')
                
                # Collect one handle-label pair per subject (only once)
                if i == 0:  # Only collect on the first subplot to avoid duplicates
                    handles.append(line)
                    labels.append(f'Subject {j}')
            
            ax[i].set_title(genes[i])
            ax[i].set_xlabel('Pseudotime')
            ax[i].set_ylabel('Expression Level')
        
        # Add a single legend to the right of all subplots
        fig.legend(handles, labels, title="Subjects", loc='center right', bbox_to_anchor=(1.02, 0.5))
        
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space on the right for the legend
        plt.show()

    def plot_barycenter_comparison(self, genes):
        """
        Plot the barycenter and individual time series, using mapped pseudotime as the x-axis.

        Parameters:
            genes (list): List of genes to plot.
            all_genes (list, optional): Full list of genes for indexing.

        Requires:
            self.mapped_pseudotimes: Mapped pseudotimes for each subject (computed via map_pseudotime_to_barycenter).
        """
        if self.mapped_pseudotimes is None:
            raise ValueError("Mapped pseudotimes have not been computed. Run map_pseudotime_to_barycenter first.")
        
        idx_genes = np.where(np.isin(self.all_genes, genes))[0]
        
        fig, ax = plt.subplots(1, len(genes), figsize=(5 * len(genes), 5))
        
        for i, idx in enumerate(idx_genes):
            for j in range(self.subject_arrays.shape[0]):
                # Use mapped pseudotime for x-axis
                mapped_pseudotime = self.mapped_pseudotimes[j]
                ax[i].plot(mapped_pseudotime, self.subject_arrays[j][:, idx], color='gray', alpha=0.5)
            
            # Plot barycenter using uniform pseudotime (as it is already aligned)
            ax[i].plot(range(self.barycenter.shape[0]), self.barycenter[:, idx], label="Barycenter", color='red')
            
            ax[i].legend()
            ax[i].set_title(genes[i])
            ax[i].set_xlabel("Mapped Pseudotime")
            ax[i].set_ylabel("Expression Level")
        
        plt.tight_layout()
        plt.show()

    def map_pseudotime_to_barycenter(self):
        """
        Map original pseudotime for each time series onto the same time axis as the barycenter using Soft-DTW alignment.

        Parameters:
            gamma (float): Smoothing parameter for Soft-DTW alignment.

        Saves:
            self.mapped_pseudotimes: List of mapped pseudotimes for each input time series.
        """
        if self.barycenter is None or self.subject_arrays is None:
            raise ValueError("Barycenter or subject arrays are not defined. Please run prepare_data and compute_barycenter first.")
        
        self.mapped_pseudotimes = []  # Reset mapped pseudotimes
        self.sample_distances = []

        for idx, ts in enumerate(self.subject_arrays):
            # Compute Soft-DTW alignment matrix
            alignment_matrix, _ = soft_dtw_alignment(ts, self.barycenter, gamma=self.gamma)
            
            # Extract mapping from alignment matrix
            mapped_pseudotime = np.argmax(alignment_matrix, axis=1)  # Most aligned barycenter time point
            # Append mapped pseudotime
            self.mapped_pseudotimes.append(mapped_pseudotime)

    def get_mapped_pseudotimes(self):
        """
        Retrieve the mapped pseudotimes.

        Returns:
            list of np.ndarray: Mapped pseudotimes for each subject.
        """
        if self.mapped_pseudotimes is None:
            raise ValueError("Mapped pseudotimes have not been computed. Run map_pseudotime_to_barycenter first.")
        
        return self.mapped_pseudotimes

    def compute_sample_distances_to_barycenter(self):
        """
        Compute the Euclidean distance of each sample's aligned gene trajectory to the barycenter for each gene,
        using previously computed mapped pseudotimes.

        Returns:
            list of np.ndarray: A list where each element is a distance matrix (samples x genes) for each subject.
                                Rows correspond to samples, and columns correspond to genes.
        """
        if self.barycenter is None or self.subject_arrays is None:
            raise ValueError("Barycenter or subject arrays are not defined. Run prepare_data and compute_barycenter first.")
        if self.mapped_pseudotimes is None:
            raise ValueError("Mapped pseudotimes are not available. Run map_pseudotime_to_barycenter first.")

        distances_per_subject = []

        for subject_idx, (subject_array, mapped_pseudotime) in enumerate(zip(self.subject_arrays, self.mapped_pseudotimes)):
            # Map each sample's pseudotime to the corresponding barycenter values
            mapped_barycenter = self.barycenter[mapped_pseudotime, :]  # Shape: (num_samples, num_genes)

            # Compute Euclidean distances between each sample and its mapped barycenter values for each gene
            distances = np.abs(subject_array - mapped_barycenter)  # Element-wise absolute difference
            distances_per_subject.append(distances)
        
        self.distance_matrices = distances_per_subject
        return distances_per_subject

    def compute_average_distance_to_barycenter(self):
        """
        Compute the average Euclidean distance to the barycenter per patient per gene.

        Returns:
            list of np.ndarray: A list where each element is an array of shape (num_genes,)
                                containing the average distance for each gene for a given subject.
        """
        # Ensure distances are computed
        if not hasattr(self, 'distance_matrices') or self.distance_matrices is None:
            self.compute_sample_distances_to_barycenter()
        
        average_distances_per_subject = []

        # Iterate over each subject's distance matrix
        for distances in self.distance_matrices:
            # Compute mean across rows (samples) for each column (gene)
            average_distances = np.mean(distances, axis=0)  # Shape: (num_genes,)
            average_distances_per_subject.append(average_distances)
        
        return average_distances_per_subject

    def plot_distance_matrix(self, genes):
        """
        Plot the distance matrix per gene as panels. Each panel corresponds to a gene,
        and individual samples are plotted as separate lines for each subject.

        Parameters:
            genes (list): List of gene names to plot.

        Returns:
            None
        """
        # Check if distance matrices are available
        if not hasattr(self, "distance_matrices") or self.distance_matrices is None:
            raise ValueError("Distance matrices are not available. Run compute_sample_distances_to_barycenter first.")

        # Ensure genes exist in self.all_genes
        if self.all_genes is None:
            raise ValueError("Gene list (self.all_genes) is not defined. Run prepare_data first.")
        
        gene_indices = [self.all_genes.index(gene) for gene in genes]  # Map gene names to indices
        num_genes = len(genes)
        num_subjects = len(self.distance_matrices)

        # Create default subject names if not already defined
        subject_names = [f"Subject {i + 1}" for i in range(num_subjects)]

        # Create subplots: one panel per gene
        fig, axes = plt.subplots(1, num_genes, figsize=(5 * num_genes, 5), sharey=True)

        # Ensure axes is iterable (even if there's only one gene)
        if num_genes == 1:
            axes = [axes]

        # Loop through each gene to create a panel
        for g_idx, (gene_idx, gene_name) in enumerate(zip(gene_indices, genes)):
            ax = axes[g_idx]

            # Plot distances for each subject and sample for this gene
            for s_idx, distances in enumerate(self.distance_matrices):
                # Extract distances for the current gene across all samples
                gene_distances = distances[:, gene_idx]  # Shape: (num_samples,)
                
                # Plot individual sample lines
                ax.plot(
                    range(len(gene_distances)), 
                    gene_distances, 
                    label=subject_names[s_idx], 
                    alpha=0.7
                )

            # Customize each subplot
            ax.set_title(f"Gene: {gene_name}")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Distance to Barycenter")
            ax.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Test the class on minimal example
    df = pd.read_csv("/scratch/chanj3/wangm10/NSCLC_SCLC-A_SCLC-N.csv", index_col=0)
    df.reset_index(inplace=True)
    benchmark_genes = ["ASCL1", "NEUROD1"]#, "TACSTD2", "DLL3"]

    hvg_genes = pd.read_csv("/data1/chanj3/HTA.lung.NE_plasticity.120122/fresh.SCLC_NSCLC.062124/results_all_genes/preprocessing/NSCLC_SCLC-A_SCLC-N/hvg_genes_NSCLC_SCLC-A_SCLC-N.txt",
                            delimiter='\t')
    genes = benchmark_genes #hvg_genes.iloc[:,0].values.tolist()

    # Initialize analyzer
    analyzer = scDeBussy(win_sz=0.1, num_pts=20, gamma=1)

    # Plot original expression
    df_subset = df.loc[:, ["subject", "score"] + genes]
    analyzer.plot_original_expression(df_subset, benchmark_genes)

    # Prepare data and compute barycenter
    analyzer.prepare_data(df_subset, genes)
    analyzer.compute_barycenter(method='soft_dtw')

    # Plot interpolated expression and barycenter
    analyzer.plot_interpolated_expression(benchmark_genes)

    analyzer.map_pseudotime_to_barycenter()
    analyzer.plot_barycenter_comparison(benchmark_genes)

    analyzer.compute_sample_distances_to_barycenter()
    analyzer.plot_distance_matrix(benchmark_genes)