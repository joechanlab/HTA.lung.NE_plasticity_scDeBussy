import numpy as np
from scipy.optimize import minimize

from tslearn.utils import to_time_series_dataset, check_equal_size, \
    to_time_series
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.metrics import SquaredEuclidean
from tslearn.barycenters.utils import _set_weights
from tslearn.barycenters.euclidean import euclidean_barycenter
from tslearn.backend import instantiate_backend
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram

def soft_dtw_with_gaps(ts1, ts2, gamma=1.0, gap_penalty=1.0, be=None, verbose=False):
    """
    Modified SoftDTW with insertion/deletion operations using:
    - Affine gap penalties (gap_extend = 0.2 * gap_penalty)
    - Automatic gap penalty calculation option
    - Stable softmin using logsumexp trick
    - Probability-weighted path tracking
    
    Parameters
    ----------
    ts1 : array-like
        First time series
    ts2 : array-like
        Second time series
    gamma : float
        Softmin parameter
    gap_penalty : float or "auto"
        Penalty for opening gaps ("auto" calculates based on sequence length)
    be : Backend object
        Computation backend
    verbose : bool
        Whether to print debug information
    """
    be = instantiate_backend(be, ts1, ts2)
    ts1 = be.array(ts1)
    ts2 = be.array(ts2)
    
    # Compute distance matrix
    D = SquaredEuclidean(ts1, ts2, be=be).compute()
    m, n = be.shape(D)
    
    if verbose:
        print("Distance matrix D:")
        print(D)
    
    # Initialize matrices
    R = be.full((m + 2, n + 2), be.inf)  # Cost matrix with padding
    P = [[None] * (n + 2) for _ in range(m + 2)]  # Path probabilities
    
    # Track whether previous move was a gap (for affine penalty)
    from_deletion = be.zeros((m + 2, n + 2), dtype=bool)
    from_insertion = be.zeros((m + 2, n + 2), dtype=bool)
    
    # Set gap extension penalty (20% of gap_penalty)
    gap_extend = 0.5 * gap_penalty
    
    # Initialize boundaries
    R[0, 0] = 0
    P[0][0] = be.array([0., 0., 0.])  # [match_prob, del_prob, ins_prob]
    
    # First column (deletions)
    for i in range(1, m + 1):
        if i == 1:
            R[i, 0] = gap_penalty
        else:
            R[i, 0] = R[i-1, 0] + (gap_extend if from_deletion[i-1, 0] else gap_penalty)
        P[i][0] = be.array([0., 1., 0.])  # Deletion
        from_deletion[i, 0] = True
    
    # First row (insertions)
    for j in range(1, n + 1):
        if j == 1:
            R[0, j] = gap_penalty
        else:
            R[0, j] = R[0, j-1] + (gap_extend if from_insertion[0, j-1] else gap_penalty)
        P[0][j] = be.array([0., 0., 1.])  # Insertion
        from_insertion[0, j] = True
    
    # Main recursion with stable softmin
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Compute raw costs
            match_cost = R[i-1, j-1] + D[i-1, j-1]
            
            # Affine deletion cost
            del_cost = R[i-1, j] + (gap_extend if from_deletion[i-1, j] else gap_penalty)
            
            # Affine insertion cost
            ins_cost = R[i, j-1] + (gap_extend if from_insertion[i, j-1] else gap_penalty)
            
            # Stable softmin computation
            costs = be.array([match_cost, del_cost, ins_cost])
            max_cost = be.max(-costs / gamma)
            shifted_exps = be.exp(-costs / gamma - max_cost)
            sum_exp = be.sum(shifted_exps)
            
            # Update cost matrix
            R[i, j] = -gamma * (be.log(sum_exp) + max_cost)
            
            # Compute path probabilities
            probs = shifted_exps / (sum_exp + 1e-16)  # Avoid division by zero
            P[i][j] = probs
            
            # Track most probable move for gap extension
            if be.argmax(probs) == 1:  # Deletion
                from_deletion[i, j] = True
            elif be.argmax(probs) == 2:  # Insertion
                from_insertion[i, j] = True
    
    if verbose:
        print("Cost matrix R:")
        print(R[1:m+1, 1:n+1])
        print("Path probabilities (last row):")
        for j in range(n+1):
            print(f"P[{m},{j}] = {P[m][j]}")
    
    # Backtrack to find alignment (using most probable path)
    alignment = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (P[i][j] is not None) and be.argmax(P[i][j]) == 0:
            alignment.append((i-1, j-1))  # Match
            i -= 1
            j -= 1
        elif i > 0 and (P[i][j] is None or be.argmax(P[i][j]) == 1):
            alignment.append((i-1, None))  # Deletion
            i -= 1
        elif j > 0:
            alignment.append((None, j-1))  # Insertion
            j -= 1
    
    return R[m, n], alignment[::-1]

def length_based_penalty(sequence, max_len=20):
    return 0.5 * (len(sequence) / max_len)

def compute_weights_from_barycenter(X, gamma=1.0, be=None, gap_penalty=1.0, n_init=3, temperature=0.1):
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
    barycenter, _ = softdtw_barycenter_with_gaps(
        X, 
        gamma=gamma,
        gap_penalties="auto",
        weights=None,  # Uniform weights for preliminary barycenter
        max_iter=n_init
    )
    
    # Compute distances to barycenter
    distances = be.zeros(len(X))
    for i in range(len(X)):
        cost, _ = soft_dtw_with_gaps(X[i], barycenter, gamma=gamma, 
                                   gap_penalty=gap_penalty)
        distances[i] = cost
    
    # Convert distances to weights using softmax
    weights = be.exp(-distances / temperature)
    weights = weights / be.sum(weights)  # Normalize
    
    return weights, barycenter

def softdtw_barycenter_with_gaps(X, gamma=1.0, gap_penalties="auto", weights=None, method="L-BFGS-B", 
                                tol=1e-3, max_iter=50, init=None, verbose=False):
    """Compute barycenter (time series averaging) under the soft-DTW with gaps geometry.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.
    gamma: float
        Regularization parameter.
        Lower is less smoothed (closer to true DTW).
    gap_penalty: float
        Penalty for insertion/deletion operations.
    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.
    method: string
        Optimization method, passed to `scipy.optimize.minimize`.
        Default: L-BFGS.
    tol: float
        Tolerance of the method used.
    max_iter: int
        Maximum number of iterations.
    init: array or None (default: None)
        Initial barycenter to start from for the optimization process.
        If `None`, euclidean barycenter is used as a starting point.

    Returns
    -------
    np.array of shape (bsz, d) where `bsz` is the size of the `init` array
            if provided or `sz` otherwise
        Soft-DTW barycenter of the provided time series dataset.
    """
    X_ = to_time_series_dataset(X)
    weights = _set_weights(weights, X_.shape[0])
    if gap_penalties is None:
        gap_penalties = [1.0] * len(X)
    elif gap_penalties == "auto":
        # Compute penalties based on sequence features
        gap_penalties = [
            length_based_penalty(ts, X_.shape[1])
            for i, ts in enumerate(X)
        ]
    
    if init is None:
        if check_equal_size(X_):
            barycenter = euclidean_barycenter(X_, weights)
        else:
            resampled_X = TimeSeriesResampler(sz=X_.shape[1]).fit_transform(X_)
            barycenter = euclidean_barycenter(resampled_X, weights)
    else:
        barycenter = init
    alignment_maps = []

    if max_iter > 0:
        X_ = [to_time_series(d, remove_nans=True) for d in X_]

        def f(Z):
            obj, grad, paths = _softdtw_func_with_gaps(Z, X_, weights, barycenter, gamma, gap_penalties, verbose=verbose)
            alignment_maps.append(paths)
            return obj, grad

        # The function works with vectors so we need to vectorize barycenter
        res = minimize(f, barycenter.ravel(), method=method, jac=True, tol=tol,
                      options=dict(maxiter=max_iter, disp=False))
        final_barycenter = res.x.reshape(barycenter.shape)
    else:
        final_barycenter = barycenter
    final_alignments = alignment_maps[-1] if alignment_maps else []

    if verbose:
        for P in final_alignments:
            plot_alignment_path(P)
    return final_barycenter, final_alignments

def _softdtw_func_with_gaps(Z, X, weights, barycenter, gamma, gap_penalties, verbose=False):
    """Compute objective value and gradient for soft-DTW with gaps barycenter."""
    Z = Z.reshape(barycenter.shape)
    G = np.zeros_like(Z)
    obj = 0
    alignment_maps = []

    for i in range(len(X)):
        gap_open = gap_penalties[i] if gap_penalties is not None else 1.0
        gap_extend = 0.4 * gap_open
        # Reuse the distance matrix computation from soft_dtw_with_gaps
        Z_clean = np.nan_to_num(Z, nan=0.0, posinf=1e10, neginf=-1e10)
        X_clean = np.nan_to_num(X[i], nan=0.0, posinf=1e10, neginf=-1e10)
        
        D = SquaredEuclidean(Z_clean, X_clean).compute()
        m, n = D.shape
        if verbose:
            print("Distance Matrix D: ")
            print(D)

        # Initialize matrices
        R = np.zeros((m + 1, n + 1))  # Cost matrix
        E = np.zeros((m + 1, n + 1))  # Gradient matrix
        P = [[None] * (n + 1) for _ in range(m + 1)]  # Path matrix (1: match, 2: deletion, 3: insertion)
        
        # track if previous move was a gap
        from_deletion = np.zeros((m + 1, n + 1), dtype=bool)
        from_insertion = np.zeros((m + 1, n + 1), dtype=bool)

        # Initialize first row and column with gap penalties
        for j in range(1, m + 1):
            if j == 1: R[j, 0] = gap_open
            else:
                if from_deletion[j-1, 0]:
                    R[j, 0] = R[j-1, 0] + gap_extend
                else:
                    R[j, 0] = R[j-1, 0] + gap_open
            P[j][0] = np.array([0, 1, 0]) # deletion
            from_deletion[j, 0] = True

        for j in range(1, n + 1):
            if j == 1: R[0, j] = gap_open
            else:
                if from_insertion[0, j-1]:
                    R[0, j] = R[0, j-1] + gap_extend
                else:
                    R[0, j] = R[0, j-1] + gap_open
            from_insertion[0, j] = True
            P[0][j] = np.array([0, 0, 1]) # insertion
        P[0][0] = np.array([0, 0, 0]) # start point
        
        # Forward pass - reuse the same logic as in soft_dtw_with_gaps
        outlier_threshold = 2.0 * np.std(D)
        for j in range(1, m + 1):
            for k in range(1, n + 1):
                dist = D[j-1, k-1]
                if dist > outlier_threshold:
                    match_cost = R[j-1, k-1] + 10 * dist
                else:
                    match_cost = R[j-1, k-1] + dist
                
                if from_deletion[j-1, k]:
                    delete_cost = R[j-1, k] + gap_extend
                else:
                    delete_cost = R[j-1, k] + gap_open
                
                if from_insertion[j, k-1]:
                    insert_cost = R[j, k-1] + gap_extend
                else:
                    insert_cost = R[j, k-1] + gap_open
                
                costs = np.array([match_cost, delete_cost, insert_cost])

                max_cost = np.max(-costs / gamma)
                sum_exp = np.sum(np.exp(-costs / gamma - max_cost))
                R[j, k] = -gamma * (np.log(sum_exp) + max_cost)

                costs = np.array([match_cost, delete_cost, insert_cost])
                max_term = np.max(-costs / gamma)
                shifted_exps = np.exp(-costs / gamma - max_term)
                probs = shifted_exps / (shifted_exps.sum() + 1e-16) # prevent 0/0
                P[j][k] = probs # probability of match

                if np.argmax(probs) == 1:
                    from_deletion[j, k] = True
                elif np.argmax(probs) == 2:
                    from_insertion[j, k] = True
                
                if verbose: print(f"Path Matrix P at (j={j}, k={k}):\n{P}")

        if verbose:
            print("Cost Matrix R (after forward pass):")
            print(R)
        # Backward pass for gradient - keep the original implementation
        E[m, n] = 1
        E_rows = m + 1
        E_cols = n + 1
        for j in range(E_rows-2, -1, -1):
            for k in range(E_cols-2, -1, -1):
                if j + 1 < E_rows and k + 1 < E_cols:
                    match_term = (R[j+1, k+1] - R[j, k] - D[j, k])/gamma
                    match_term = np.clip(match_term, -20, 20)
                    match_exp_term = np.exp(-match_term)
                    match_grad = E[j+1, k+1] * match_exp_term * P[j+1][k+1][0]

                    if from_deletion[j+1, k]:
                        gap_penalty = gap_extend
                    else:
                        gap_penalty = gap_open
                    
                    delete_term = (R[j+1, k] - R[j, k] - gap_penalty)/gamma
                    delete_term = np.clip(delete_term, -20, 20)
                    delete_exp_term = np.exp(-delete_term)
                    delete_grad = E[j+1, k] * delete_exp_term * P[j+1][k][1]

                    insert_term = (R[j, k+1] - R[j, k] - gap_penalty)/gamma
                    insert_term = np.clip(insert_term, -20, 20)
                    insert_exp_term = np.exp(-insert_term)
                    insert_grad = E[j, k+1] * insert_exp_term * P[j][k+1][2]

                    E[j, k] = match_grad + delete_grad + insert_grad
                elif j + 1 < E_rows:
                    if from_deletion[j+1, k]:
                        gap_penalty = gap_extend
                    else:
                        gap_penalty = gap_open

                    delete_term = (R[j+1, k] - R[j, k] - gap_penalty)/gamma
                    delete_term = np.clip(delete_term, -20, 20)
                    delete_exp_term = np.exp(-delete_term)
                    E[j, k] = E[j+1, k] * delete_exp_term * P[j+1][k][1]
                elif k + 1 < E_cols:
                    if from_insertion[j, k+1]:
                        gap_penalty = gap_extend
                    else:
                        gap_penalty = gap_open
                    
                    insert_term = (R[j, k+1] - R[j, k] - gap_penalty)/gamma
                    insert_term = np.clip(insert_term, -20, 20)
                    insert_exp_term = np.exp(-insert_term)
                    E[j, k] = E[j, k+1] * insert_exp_term * P[j][k+1][2]
                else:
                    E[j, k] = 0  # Only for (0,0)
        if verbose:
            print("Gradient Matrix E (after backward pass):")
            print(E)
        
        alignment_maps.append(P)
        # Compute gradient with respect to Z
        G_tmp = np.zeros_like(Z)
        for j in range(m):
            for k in range(n):
                if j < m and k < n: # consider only matching pairs
                    G_tmp[j] += E[j+1, k+1] * P[j+1][k+1][0] * (Z[j] - X[i][k])
                elif k >= len(X[i]) - 3:  # If Near edges to allow borrow info
                    borrow_factor = 0.5
                    G_tmp[j] += E[j+1, k+1] * P[j+1][k+1][0] * (Z[j] - X[i][k]) * (1 - borrow_factor)

            if verbose:
                plt.plot(G_tmp, label=f"TS {j} Gradient")
                plt.legend()
                plt.show()
        
        G += weights[i] * G_tmp
        obj += weights[i] * R[m, n]

        if verbose:
            print("Gradient Vector (G):")
            print(G)
    
    return obj, G.ravel(), alignment_maps

def extract_mappings(alignment_maps, X, barycenter):
    """Convert probability matrices (P) into barycenter↔time_series mappings."""
    mappings = []
    bary_len = len(barycenter)
    
    for i, P in enumerate(alignment_maps):
        ts = np.asarray(X[i])
        ts_len = len(ts)
        
        bary_indices = []
        ts_indices = []
        j, k = bary_len, ts_len  # Start from the end of the cost matrix
        
        while j > 0 or k > 0:
            if j == 0:
                # Insertion (gap in barycenter)
                ts_indices.append(k-1)
                bary_indices.append(None)
                k -= 1
            elif k == 0:
                # Deletion (gap in time series)
                bary_indices.append(j-1)
                ts_indices.append(None)
                j -= 1
            else:
                # Use probabilities to choose the most likely operation
                probs = P[j][k]
                op = np.argmax(probs)
                if op == 0:  # Match
                    bary_indices.append(j-1)
                    ts_indices.append(k-1)
                    j -= 1
                    k -= 1
                elif op == 1:  # Delete
                    bary_indices.append(j-1)
                    ts_indices.append(None)
                    j -= 1
                else:  # Insert
                    bary_indices.append(None)
                    ts_indices.append(k-1)
                    k -= 1
        
        # Reverse to chronological order
        mappings.append((bary_indices[::-1], ts_indices[::-1]))
    
    return mappings


def plot_alignment_path(P):
    plt.imshow(P, cmap='viridis')
    plt.colorbar()
    plt.title("Alignment Path Matrix (P)")
    plt.xlabel("Time Series Index")
    plt.ylabel("Barycenter Index")
    plt.show()

def get_final_alignments(final_barycenter, X, gamma, gap_penalty):
    """Recompute alignments using optimized barycenter."""
    final_alignments = []
    for ts in X:
        _, path = soft_dtw_with_gaps(final_barycenter, ts, gamma=gamma, gap_penalty=gap_penalty)
        final_alignments.append(path)
    return final_alignments

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

# Test the function with proper error handling
if __name__ == "__main__":
    # Test case 1: simple outlier
    gamma = 0.5
    gap_penalty = 1.0
    X = [[1, 2, 3], [1, 0, 3], [0, 2, 4]]
    final_barycenter, final_alignments = softdtw_barycenter_with_gaps(X, gamma=gamma, gap_penalty=gap_penalty, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = extract_mappings(final_alignments, X, final_barycenter)
    fig = plot_barycenter_mappings(final_barycenter, X, mappings)

    # Test case 2: variable lengths
    X = [
        [1, 2, 3, 4],     # Longer series
        [1, 3],           # Short series (should insert gaps)
        [2, 3, 4]         # Medium series
    ]
    gap_penalty = 2.0
    final_barycenter, final_alignments = softdtw_barycenter_with_gaps(X, gamma=gamma, gap_penalty=gap_penalty, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = extract_mappings(final_alignments, X, final_barycenter)
    fig = plot_barycenter_mappings(final_barycenter, X, mappings)

    # Test case 3: Noisy sinusoids
    t = np.linspace(0, 2*np.pi, 20)
    X = [
        np.sin(t) + np.random.normal(0, 0.1, len(t)),  # Noisy sine
        np.sin(t + 0.5*np.pi) + np.random.normal(0, 0.1, len(t)),  # Phase-shifted
        np.sin(t[:15])   # Shorter version (tests gaps)
    ]
    gap_penalty = 0.5    # Low penalty to allow gaps
    final_barycenter, final_alignments = softdtw_barycenter_with_gaps(X, gamma=gamma, gap_penalty=gap_penalty, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = extract_mappings(final_alignments, X, final_barycenter)
    fig = plot_barycenter_mappings(final_barycenter, X, mappings)

    # Test case 4: extreme outlier test
    X = [
        [1, 2, 3, 4, 5],
        [1, 2, 100, 4, 5],  # Extreme outlier
        [1, 2, 3, 4, 5]
    ]
    gap_penalty = 0.1    # Low penalty: should gap the outlier
    #gap_penalty = 50.0 # High penalty: should match the outlier
    final_barycenter, final_alignments = softdtw_barycenter_with_gaps(X, gamma=gamma, gap_penalty=gap_penalty, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = extract_mappings(final_alignments, X, final_barycenter)
    fig = plot_barycenter_mappings(final_barycenter, X, mappings)

    # Test case 5: Multiple dimensional time series
    X = [
        np.array([[1, 10], [2, 20], [3, 30]]),  # (x, y) coordinates
        np.array([[1, 10], [3, 30]]),           # Missing middle point
        np.array([[2, 20], [3, 30], [4, 40]])
        ]
    gap_penalty = 10
    final_barycenter, final_alignments = softdtw_barycenter_with_gaps(X, gamma=gamma, gap_penalty=gap_penalty, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = extract_mappings(final_alignments, X, final_barycenter)
    fig = plot_barycenter_mappings(final_barycenter, X, mappings)

    # Test case 6: Real-world example ECG Beats 
    ecg = electrocardiogram()[1000:1200]  # Sample ECG data
    X = [
        ecg[20:100],     # Partial heartbeat
        ecg[10:110],     # Longer segment
        ecg[30:90]       # Shorter segment
    ]
    gamma = 0.5
    gap_penalty = 0.5    # ECG tolerates small misalignments
    final_barycenter, final_alignments = softdtw_barycenter_with_gaps(X, gamma=gamma, gap_penalty=gap_penalty, weights=None, method="L-BFGS-B", 
                                    tol=1e-3, max_iter=50, init=None)
    mappings = extract_mappings(final_alignments, X, final_barycenter)
    fig = plot_barycenter_mappings(final_barycenter, X, mappings, figsize=(12, 4))

    # Test case 7: Single cell data