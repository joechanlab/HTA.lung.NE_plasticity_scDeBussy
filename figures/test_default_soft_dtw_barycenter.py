import numpy as np
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import dtw_path, soft_dtw
from tslearn.backend import instantiate_backend

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.misc import electrocardiogram

def compute_weights_from_barycenter(X, gamma=1.0, be=None, n_init=3, temperature=0.1):
    """
    Compute weights based on distance to preliminary barycenter.
    
    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset
    gamma : float
        SoftDTW regularization parameter
    gap_penalty : float
        Gap penalty for alignments
    be : backend object
        Computation backend
    n_init : int
        Number of iterations to compute preliminary barycenter
    temperature : float
        Controls weight distribution sharpness
        
    Returns
    -------
    weights : array, shape=(n_ts,)
        Normalized weights (higher for closer sequences)
    """
    be = instantiate_backend(be, X)
    
    # Compute preliminary barycenter with few iterations
    barycenter = softdtw_barycenter(
        X, 
        gamma=gamma,
        weights=None,  # Uniform weights for preliminary barycenter
        max_iter=n_init
    )
    
    # Compute distances to barycenter
    similarities = be.zeros(len(X))
    for i in range(len(X)):
        sim = soft_dtw(X[i], barycenter, gamma=gamma)
        similarities[i] = sim
    similarities = similarities - np.max(similarities)
    
    # Convert similarity to weights using softmax
    weights = be.exp(similarities / temperature)
    weights = weights / be.sum(weights)  # Normalize
    
    return weights, barycenter


def get_barycenter_mappings(barycenter, time_series, gap_penalty=1.0, distance_threshold=1.0):
    """
    Get the mapping between the barycenter and individual time series.
    
    Parameters:
    -----------
    barycenter : array-like
        The barycenter time series
    time_series : list of array-like
        List of time series to map to the barycenter
    gap_penalty : float, optional
        Penalty for gaps in the alignment
    distance_threshold : float, optional
        Threshold for applying gap penalty
        
    Returns:
    --------
    mappings : list of tuples
        List of (barycenter_indices, time_series_indices) for each time series
    """
    mappings = []
    
    for i, ts in enumerate(time_series):
        # Use dtw_path directly with the two sequences
        path, _ = dtw_path(barycenter, ts)
        
        # Convert path to barycenter and time series indices
        bary_indices = [p[0] for p in path]
        ts_indices = [p[1] for p in path]
        
        mappings.append((bary_indices, ts_indices))
    
    return mappings

def plot_barycenter_mappings(barycenter, time_series, mappings, figsize=(12, 4)):
    """
    Plot the time series with the barycenter and their mappings.
    
    Parameters:
    -----------
    barycenter : array-like
        The barycenter time series
    time_series : list of array-like
        List of time series
    mappings : list of tuples
        List of (barycenter_indices, time_series_indices) for each time series
    figsize : tuple, optional
        Figure size
    """
    n_series = len(time_series)
    n_dims = barycenter.shape[1]

    fig, axes = plt.subplots(n_dims, n_series, figsize=figsize, sharex='col')
    
    # If there's only one time series, axes will be a single object, not an array
    if n_series == 1:
        axes = axes.reshape(-1, 1)
    if n_dims == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each time series with the barycenter and its mapping
    for dim in range(n_dims):
        for i, (ts, (bary_indices, ts_indices)) in enumerate(zip(time_series, mappings)):
            ax = axes[dim, i]
            ts_arr_dim = np.asarray(ts)
            barycenter_dim = np.asarray(barycenter)
            
            if n_dims > 1:
                ts_arr_dim = ts_arr_dim[:, dim]
                barycenter_dim = barycenter_dim[:, dim]
            
            # Plot the time series
            ax.plot(ts_arr_dim, label=f'Time series {i+1} (dim {dim + 1})', color=f'C{i}', linewidth=2)
            
            # Plot the barycenter
            ax.plot(barycenter_dim, label='Barycenter', color='red', linewidth=2, linestyle='--')
            
            # Plot the mapping
            for b_idx, t_idx in zip(bary_indices, ts_indices):
                # Skip if either index is None (gap)
                if b_idx is not None and t_idx is not None:
                    x_coords = [t_idx, b_idx]
                    
                    y_coords = [ts_arr_dim[t_idx], barycenter_dim[b_idx] if n_dims > 1 else barycenter_dim[b_idx][0]]
                    
                    ax.plot(x_coords, y_coords, color='gray', linestyle='-', alpha=0.3)
            if dim == 0:
                ax.set_title(f'Time series {i+1}')
            if i == 0:
                ax.set_ylabel(f'Dimension {dim + 1}')
            ax.legend()
    
    plt.tight_layout()
    plt.show()
    return fig

if __name__ == "__main__":
    # Test case 1: 
    gamma = 0.5
    X = [[1, 2, 3], [1, 0, 3], [0, 2, 4]]
    barycenter = softdtw_barycenter(X, gamma=gamma, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = get_barycenter_mappings(barycenter, X)
    fig = plot_barycenter_mappings(barycenter, X, mappings)

    # Test case 2: 
    X = [
        [1, 2, 3, 4],     # Longer series
        [1, 3],           # Short series (should insert gaps)
        [2, 3, 4]         # Medium series
    ]
    barycenter = softdtw_barycenter(X, gamma=gamma, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = get_barycenter_mappings(barycenter, X)
    fig = plot_barycenter_mappings(barycenter, X, mappings)

    # Test case 3:
    t = np.linspace(0, 2*np.pi, 20)
    X = [
        np.sin(t) + np.random.normal(0, 0.1, len(t)),  # Noisy sine
        np.sin(t + 0.5*np.pi) + np.random.normal(0, 0.1, len(t)),  # Phase-shifted
        np.sin(t[:15])   # Shorter version (tests gaps)
    ]
    barycenter = softdtw_barycenter(X, gamma=gamma, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = get_barycenter_mappings(barycenter, X)
    fig = plot_barycenter_mappings(barycenter, X, mappings)

    # Test case 4:
    X = [
        [1, 2, 3, 4, 5],
        [1, 2, 100, 4, 5],  # Extreme outlier
        [1, 2, 3, 4, 5]
    ]
    barycenter = softdtw_barycenter(X, gamma=gamma, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = get_barycenter_mappings(barycenter, X)
    fig = plot_barycenter_mappings(barycenter, X, mappings)

    # Test case 5: Multiple dimensional time series
    X = [
        np.array([[1, 10], [2, 20], [3, 30]]),  # (x, y) coordinates
        np.array([[1, 10], [3, 30]]),           # Missing middle point
        np.array([[2, 20], [3, 30], [4, 40]])
        ]
    barycenter = softdtw_barycenter(X, gamma=gamma, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = get_barycenter_mappings(barycenter, X)
    fig = plot_barycenter_mappings(barycenter, X, mappings)

    # Test case 7: Real-world example ECG Beats 
    ecg = electrocardiogram()[1000:1200]  # Sample ECG data
    X = [
        ecg[20:100],     # Partial heartbeat
        ecg[10:110],     # Longer segment
        ecg[30:90]       # Shorter segment
    ]
    barycenter = softdtw_barycenter(X, gamma=gamma, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = get_barycenter_mappings(barycenter, X)
    fig = plot_barycenter_mappings(barycenter, X, mappings, figsize=(12, 4))