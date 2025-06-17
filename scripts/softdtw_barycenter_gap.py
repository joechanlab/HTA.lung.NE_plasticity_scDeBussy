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

class ShiftAwareDistance:
    def __init__(self, ts1, ts2, max_shift=3, shift_weight=0.1, be=None):
        self.ts1 = ts1
        self.ts2 = ts2
        self.max_shift = max_shift
        self.shift_weight = shift_weight  # Penalty for larger shifts
        self.be = instantiate_backend(be, ts1, ts2)
        
        if len(self.be.shape(self.ts1)) == 1:
            self.ts1 = self.be.reshape(self.ts1, (-1, 1))
        if len(self.be.shape(self.ts2)) == 1:
            self.ts2 = self.be.reshape(self.ts2, (-1, 1))
    
    def compute(self):
        m = self.be.shape(self.ts1)[0]
        n = self.be.shape(self.ts2)[0]
        D = self.be.full((m, n), self.be.inf)
        S = self.be.zeros((m, n))  # Track best shifts
        
        for i in range(m):
            for j in range(n):
                min_dist = self.be.inf
                best_shift = 0
                for s in range(-self.max_shift, self.max_shift + 1):
                    if 0 <= j + s < n:
                        diff = self.ts1[i] - self.ts2[j + s]
                        dist = self.be.sum(diff ** 2) + self.shift_weight * abs(s)
                        if dist < min_dist:
                            min_dist = dist
                            best_shift = s
                D[i, j] = min_dist
                S[i, j] = best_shift  # Store optimal shift
        return D, S  # Return both distance and shift matrices

def soft_dtw_with_gaps(ts1, ts2, gamma=1.0, gap_penalty=1.0, max_shift=10, be=None, verbose=False):
    """
    Modified SoftDTW with insertion/deletion operations using:
    - Affine gap penalties (gap_extend = 0.5 * gap_penalty)
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
    
    # Get both distance and shift matrices
    D, S = ShiftAwareDistance(ts1, ts2, max_shift=max_shift, be=be).compute()
    m, n = be.shape(D)
    
    # Initialize matrices
    R = be.full((m + 2, n + 2), be.inf)
    P = [[None] * (n + 2) for _ in range(m + 2)]
    from_deletion = be.zeros((m + 2, n + 2), dtype=bool)
    from_insertion = be.zeros((m + 2, n + 2), dtype=bool)
    gap_extend = 0.8 * gap_penalty

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
            # Incorporate shift information into match cost
            shift_cost = abs(S[i-1, j-1]) * (gap_extend / max_shift)
            match_cost = R[i-1, j-1] + D[i-1, j-1] + shift_cost
            
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

def compute_weights_from_barycenter(X, gamma=1.0, be=None, gap_penalty=1.0, max_shift=10, n_init=3, temperature=0.1):
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
                                   gap_penalty=gap_penalty, max_shift=max_shift)
        distances[i] = cost
    
    # Convert distances to weights using softmax
    weights = be.exp(-distances / temperature)
    weights = weights / be.sum(weights)  # Normalize
    
    return weights, barycenter

class ReliableSquaredEuclidean:
    """Distance metric that ignores unreliable genes, fully compatible with your DTW functions"""
    def __init__(self, unreliable_masks):
        self.unreliable_masks = unreliable_masks
        self.Z = None
        self.X_i = None
    
    def __call__(self, Z, X_i):
        """
        Store the input matrices and return self for chaining.
        Matches the interface expected by your DTW functions.
        """
        self.Z = Z
        self.X_i = X_i
        return self
    
    def compute(self):
        """
        Compute the distance matrix while ignoring unreliable genes.
        """
        if self.Z is None or self.X_i is None:
            raise ValueError("Input matrices not set. Call the distance object first.")
        
        # Find which sample this is (by comparing to our masks)
        sample_idx = self._find_sample_index(self.X_i)
        mask = self.unreliable_masks[sample_idx] if sample_idx is not None else None
        
        # Create masked versions
        Z_masked = self.Z.copy()
        X_masked = self.X_i.copy()
        if mask is not None:
            Z_masked[:, mask] = 0
            X_masked[:, mask] = 0
        
        # Compute squared Euclidean distance
        D = np.zeros((Z_masked.shape[0], X_masked.shape[0]))
        for i in range(Z_masked.shape[0]):
            for j in range(X_masked.shape[0]):
                diff = Z_masked[i] - X_masked[j]
                D[i,j] = np.sum(diff**2)
        
        # Scale by fraction of reliable genes
        if mask is not None:
            n_genes = Z_masked.shape[1]
            n_reliable = np.sum(~mask)
            if n_reliable > 0:
                D *= (n_genes / n_reliable)
        
        return D
    
    def _find_sample_index(self, X_i):
        """Helper to find which sample this is in our dataset"""
        # Compare shape and first few values to identify the sample
        for idx, mask in enumerate(self.unreliable_masks):
            if X_i.shape[1] == len(mask):  # Compare number of genes
                return idx
        return None

def reliable_softdtw_barycenter(X, gamma=1.0, gap_penalties="auto", 
                               variance_threshold=0.01, min_expression=0.01,
                               **kwargs):
    """
    Compute barycenter using only reliable genes.
    Fully compatible with your existing functions.
    
    Args:
        X: Input time series (samples × timepoints × genes)
        gamma: Soft-DTW regularization parameter
        gap_penalties: Gap penalty specification
        variance_threshold: Minimum gene variance threshold
        min_expression: Minimum mean expression threshold
        **kwargs: Additional arguments for softdtw_barycenter_with_gaps
        
    Returns:
        barycenter: Computed barycenter
        alignments: Alignment paths
        reliable_counts: Number of reliable genes per sample
    """
    # Detect unreliable genes
    X = to_time_series_dataset(X)
    unreliable_masks = detect_unreliable_genes(X, variance_threshold, min_expression)
    reliable_counts = [np.sum(~mask) for mask in unreliable_masks]
    
    # Create distance metric instance
    distance_metric = ReliableSquaredEuclidean(unreliable_masks)
    
    # Compute barycenter using your existing function
    barycenter, alignments = softdtw_barycenter_with_gaps(
        X, 
        gamma=gamma,
        gap_penalties=gap_penalties,
        distance=distance_metric,  # Pass our modified distance metric
        **kwargs
    )
    
    return barycenter, alignments, unreliable_masks, reliable_counts

def detect_unreliable_genes(X, variance_threshold=0.01, min_expression=0.01, relative_var_threshold=0.01):
    """
    Identify genes that are poorly correlated with others in each sample.
    
    Parameters:
        X: numpy array of shape (n_samples, n_timepoints, n_genes)
        variance_threshold: Minimum variance to consider a gene reliable
        min_expression: Minimum mean expression to consider a gene potentially reliable
        
    Returns:
        List of boolean masks (one per sample) indicating unreliable genes (True = unreliable)
    """
    X = np.asarray(X, dtype=np.float64)
    unreliable_masks = []
    
    for sample in X:
        with np.errstate(invalid='ignore'):
            mean_expr = np.nanmean(sample, axis=0)
            variance = np.nanvar(sample, axis=0)
            
            # Calculate relative variance (variance/mean)
            relative_var = np.zeros_like(variance)
            non_zero = mean_expr > 0
            relative_var[non_zero] = variance[non_zero] / mean_expr[non_zero]
            
            low_expr = (mean_expr < min_expression) | np.isnan(mean_expr)
            low_var = (variance < variance_threshold) | np.isnan(variance)
            low_rel_var = (relative_var < relative_var_threshold)  # Additional threshold
            
            # Combine criteria with OR
            unreliable = low_expr | low_var | low_rel_var
            
            # Ensure we keep at least some genes
            if np.all(unreliable):
                unreliable = low_expr  # Fall back to just expression threshold
        
        unreliable_masks.append(unreliable)
    
    return unreliable_masks

def softdtw_barycenter_with_gaps(X, gamma=1.0, gap_penalties="auto", weights=None, distance=SquaredEuclidean, method="L-BFGS-B", 
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
            obj, grad, paths = _softdtw_func_with_gaps(Z, X_, weights, barycenter, gamma, gap_penalties, distance, verbose=verbose)
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

def _softdtw_func_with_gaps(Z, X, weights, barycenter, gamma, gap_penalties, distance, verbose=False):
    """Compute objective value and gradient for soft-DTW with gaps barycenter."""
    Z = Z.reshape(barycenter.shape)
    G = np.zeros_like(Z)
    obj = 0
    alignment_maps = []

    for i in range(len(X)):
        # Reuse the distance matrix computation from soft_dtw_with_gaps
        Z_clean = np.nan_to_num(Z, nan=0.0, posinf=1e10, neginf=-1e10)
        X_clean = np.nan_to_num(X[i], nan=0.0, posinf=1e10, neginf=-1e10)
        gap_open = gap_penalties[i] if gap_penalties is not None else 1.0
        gap_extend = gap_open * 0.8
        D = distance(Z_clean, X_clean).compute()
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
        for j in range(1, m + 1):
            for k in range(1, n + 1):
                dist = D[j-1, k-1]
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
                    G_tmp[j] += E[j+1, k+1] * P[j+1][k+1][0] * 2 * (Z[j] - X[i][k])

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
                probs = P[j][k]
                # Use probability-weighted path selection
                if probs[0] > 0.6:  # Strong match preference
                    bary_indices.append(j-1)
                    ts_indices.append(k-1)
                    j -= 1
                    k -= 1
                elif probs[1] > probs[2]:  # Prefer deletion
                    bary_indices.append(j-1)
                    ts_indices.append(None)
                    j -= 1
                else:  # Prefer insertion
                    bary_indices.append(None)
                    ts_indices.append(k-1)
                    k -= 1
        
        # Reverse to chronological order
        mappings.append((bary_indices[::-1], ts_indices[::-1]))
    
    return mappings

def extract_mappings_mask(alignment_maps, X, barycenter, unreliable_masks):
    """Convert probability matrices (P) into gene-first alignment mappings.
    
    Args:
        alignment_maps: List of path probability matrices from soft-DTW
        X: Original time series data (n_samples × n_timepoints × n_genes)
        barycenter: Computed barycenter (n_timepoints × n_genes)
        unreliable_masks: List of boolean masks (True=unreliable) per sample
        
    Returns:
        dict containing:
            - 'by_gene': {
                  gene_index: [
                      (sample_0_bary_indices, sample_0_ts_indices),
                      (sample_1_bary_indices, sample_1_ts_indices),
                      ...
                  ]
              }
            - 'reliability': ndarray (n_samples × n_genes)
    """
    n_samples = len(X)
    n_genes = len(X[0][0])
    bary_len = len(barycenter)
    
    # Initialize output structure
    by_gene = {g: [] for g in range(n_genes)}
    reliability = np.zeros((n_samples, n_genes))
    
    for sample_idx, P in enumerate(alignment_maps):
        ts = np.asarray(X[sample_idx])
        ts_len = len(ts)
        mask = unreliable_masks[sample_idx]
        
        for gene_idx in range(n_genes):
            if mask[gene_idx]:  # Unreliable gene - all gaps
                dummy_length = max(bary_len, ts_len)
                by_gene[gene_idx].append((
                    [None]*dummy_length,
                    [None]*dummy_length
                ))
                reliability[sample_idx, gene_idx] = 0
            else:  # Reliable gene
                bary_indices, ts_indices = [], []
                j, k = bary_len, ts_len
                scores = []
                
                while j > 0 or k > 0:
                    if j == 0:  # Insertion
                        ts_indices.append(k-1)
                        bary_indices.append(None)
                        k -= 1
                    elif k == 0:  # Deletion
                        bary_indices.append(j-1)
                        ts_indices.append(None)
                        j -= 1
                    else:
                        probs = P[j][k]
                        op = np.argmax(probs)
                        
                        if op == 0:  # Match
                            score = (barycenter[j-1,gene_idx] * 
                                   ts[k-1,gene_idx] * 
                                   probs[0])
                            bary_indices.append(j-1)
                            ts_indices.append(k-1)
                            scores.append(score)
                            j -= 1
                            k -= 1
                        elif op == 1:  # Deletion
                            bary_indices.append(j-1)
                            ts_indices.append(None)
                            scores.append(0)
                            j -= 1
                        else:  # Insertion
                            bary_indices.append(None)
                            ts_indices.append(k-1)
                            scores.append(0)
                            k -= 1
                
                # Store reversed alignment
                by_gene[gene_idx].append((
                    bary_indices[::-1],
                    ts_indices[::-1]
                ))
                reliability[sample_idx, gene_idx] = (
                    np.sum(scores) / (len(scores) + 1e-8)
                )
    
    return {
        'by_gene': by_gene,
        'reliability': reliability
    }

def _create_global_alignment(gene_alignments, mask):
    """Create consensus global alignment from gene-level alignments."""
    # Get first reliable gene's alignment as template
    reliable_genes = [g for g in gene_alignments if not mask[g]]
    if not reliable_genes:
        return ([], [])  # No reliable genes
    
    template_gene = reliable_genes[0]
    bary_template, ts_template = gene_alignments[template_gene]
    
    return (bary_template, ts_template)

def extract_gene_specific_alignments(mappings, unreliable_masks, X):
    """
    Convert whole-cell alignments into gene-specific alignments.
    
    Parameters:
        mappings: Original alignments from extract_mappings() 
                 (list of (bary_indices, ts_indices) tuples)
        unreliable_masks: List of boolean masks from detect_unreliable_genes()
        X: Input data (samples × timepoints × genes)
        
    Returns:
        List of lists: For each sample, a list of (gene_bary, gene_ts) tuples for each gene
    """
    n_samples = len(mappings)
    n_genes = X.shape[2]
    gene_alignments = []
    
    for sample_idx in range(n_samples):
        # Get original whole-cell alignment
        bary_indices, ts_indices = mappings[sample_idx]
        mask = unreliable_masks[sample_idx]
        
        # For each gene, extract its specific alignment path
        sample_gene_alignments = []
        for gene_idx in range(n_genes):
            if mask[gene_idx]:
                # Unreliable gene - all gaps
                gene_bary = [None] * len(bary_indices)
                gene_ts = [None] * len(ts_indices)
            else:
                # Reliable gene - copy original alignment
                gene_bary = bary_indices.copy()
                gene_ts = ts_indices.copy()
            
            sample_gene_alignments.append((gene_bary, gene_ts))
        
        gene_alignments.append(sample_gene_alignments)
    
    return gene_alignments

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