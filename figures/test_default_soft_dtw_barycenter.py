import numpy as np
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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

def plot_barycenter_mappings(barycenter, time_series, mappings, figsize=(8, 8)):
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
    fig, axes = plt.subplots(n_series, 1, figsize=figsize, sharex=True)
    
    # If there's only one time series, axes will be a single object, not an array
    if n_series == 1:
        axes = [axes]
    
    # Plot each time series with the barycenter and its mapping
    for i, (ts, (bary_indices, ts_indices)) in enumerate(zip(time_series, mappings)):
        ax = axes[i]
        
        # Plot the time series
        ax.plot(ts, label=f'Time series {i+1}', color=f'C{i}', linewidth=2)
        
        # Plot the barycenter
        ax.plot(barycenter, label='Barycenter', color='red', linewidth=2, linestyle='--')
        
        # Plot the mapping
        for b_idx, t_idx in zip(bary_indices, ts_indices):
            # Create x and y coordinates for the line
            x_coords = [t_idx, b_idx]
            y_coords = [float(ts[t_idx]), float(barycenter[b_idx])]
            ax.plot(x_coords, y_coords, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_title(f'Time series {i+1} with barycenter mapping')
        ax.legend()
    
    plt.tight_layout()
    return fig

# Example usage:
x = np.arange(0, 10)
scaler = MinMaxScaler()
time_series = [[-2*(x - 5) ** 2 + 100 for x in x],
               [-(x - 10) ** 2 + 100 for x in x],
               [-(x - 3) ** 2 + 100 for x in x]]
time_series = [scaler.fit_transform(np.array(ts).reshape(-1, 1)).flatten() for ts in time_series]

barycenter = softdtw_barycenter(time_series, max_iter=5)

# Get the mappings between barycenter and time series
mappings = get_barycenter_mappings(barycenter, time_series)

# Plot the barycenter and time series with their mappings
fig = plot_barycenter_mappings(barycenter, time_series, mappings)
plt.savefig('barycenter_mappings.png', dpi=300, bbox_inches='tight')
plt.show()

# Alternative visualization: plot all time series and barycenter on the same plot
# with connections showing the mappings
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(barycenter, label='Barycenter', color='red', linewidth=2)

# Plot each time series and its mapping to the barycenter
for i, (ts, (bary_indices, ts_indices)) in enumerate(zip(time_series, mappings)):
    ax.plot(ts, label=f'Time series {i+1}', color=f'C{i}', alpha=0.7)
    
    # Plot a subset of the mappings to avoid overcrowding
    step = max(1, len(bary_indices) // 20)  # Show at most 20 connections
    for b_idx, t_idx in zip(bary_indices[::step], ts_indices[::step]):
        # Create x and y coordinates for the line
        x_coords = [t_idx, b_idx]
        y_coords = [float(ts[t_idx]), float(barycenter[b_idx])]
        ax.plot(x_coords, y_coords, color='gray', linestyle='--', alpha=0.2)

ax.set_title('Barycenter and Time Series with Mappings')
ax.legend()
plt.tight_layout()
plt.savefig('barycenter_mappings_combined.png', dpi=300, bbox_inches='tight')
plt.show()