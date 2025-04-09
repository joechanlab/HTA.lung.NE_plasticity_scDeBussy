import numpy
from scipy.optimize import minimize

from tslearn.utils import to_time_series_dataset, check_equal_size, \
    to_time_series
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.metrics import SquaredEuclidean
from tslearn.barycenters.utils import _set_weights
from tslearn.barycenters.euclidean import euclidean_barycenter

def softdtw_barycenter_with_gaps(X, gamma=1.0, gap_penalty=1.0, weights=None, method="L-BFGS-B", 
                                tol=1e-3, max_iter=50, init=None):
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
    numpy.array of shape (bsz, d) where `bsz` is the size of the `init` array
            if provided or `sz` otherwise
        Soft-DTW barycenter of the provided time series dataset.
    """
    X_ = to_time_series_dataset(X)
    weights = _set_weights(weights, X_.shape[0])
    
    if init is None:
        if check_equal_size(X_):
            barycenter = euclidean_barycenter(X_, weights)
        else:
            resampled_X = TimeSeriesResampler(sz=X_.shape[1]).fit_transform(X_)
            barycenter = euclidean_barycenter(resampled_X, weights)
    else:
        barycenter = init

    if max_iter > 0:
        X_ = [to_time_series(d, remove_nans=True) for d in X_]

        def f(Z):
            return _softdtw_func_with_gaps(Z, X_, weights, barycenter, gamma, gap_penalty)

        # The function works with vectors so we need to vectorize barycenter
        res = minimize(f, barycenter.ravel(), method=method, jac=True, tol=tol,
                      options=dict(maxiter=max_iter, disp=False))
        return res.x.reshape(barycenter.shape)
    else:
        return barycenter

def _softdtw_func_with_gaps(Z, X, weights, barycenter, gamma, gap_penalty):
    """Compute objective value and gradient for soft-DTW with gaps barycenter."""
    Z = Z.reshape(barycenter.shape)
    G = numpy.zeros_like(Z)
    obj = 0

    for i in range(len(X)):
        # Compute distance matrix
        D = SquaredEuclidean(Z, X[i]).compute()
        m, n = D.shape

        # Initialize matrices
        R = numpy.zeros((m + 1, n + 1))
        E = numpy.zeros((m + 1, n + 1))

        # Initialize first row and column with gap penalties
        for j in range(1, m + 1):
            R[j, 0] = j * gap_penalty
        for j in range(1, n + 1):
            R[0, j] = j * gap_penalty

        # Forward pass
        for j in range(1, m + 1):
            for k in range(1, n + 1):
                match_cost = R[j-1, k-1] + D[j-1, k-1]
                delete_cost = R[j-1, k] + gap_penalty
                insert_cost = R[j, k-1] + gap_penalty
                
                costs = numpy.array([match_cost, delete_cost, insert_cost])
                R[j, k] = gamma * numpy.log(numpy.sum(numpy.exp(-costs/gamma)))

        # Backward pass for gradient
        E[m, n] = 1
        for j in range(m, 0, -1):
            for k in range(n, 0, -1):
                if j < m and k < n:
                    match_grad = E[j+1, k+1] * numpy.exp(-(R[j+1, k+1] - R[j, k] - D[j, k])/gamma)
                    delete_grad = E[j+1, k] * numpy.exp(-(R[j+1, k] - R[j, k] - gap_penalty)/gamma)
                    insert_grad = E[j, k+1] * numpy.exp(-(R[j, k+1] - R[j, k] - gap_penalty)/gamma)
                    
                    E[j, k] = match_grad + delete_grad + insert_grad
                elif j < m:
                    E[j, k] = E[j+1, k] * numpy.exp(-(R[j+1, k] - R[j, k] - gap_penalty)/gamma)
                elif k < n:  # Changed from else to elif k < n
                    E[j, k] = E[j, k+1] * numpy.exp(-(R[j, k+1] - R[j, k] - gap_penalty)/gamma)
                else:
                    E[j, k] = 0  # Added else case for completeness

        # Compute gradient with respect to Z
        G_tmp = numpy.zeros_like(Z)
        for j in range(m):
            for k in range(n):
                if j < m and k < n:  # Added bounds checking
                    G_tmp[j] += E[j+1, k+1] * (Z[j] - X[i][k])

        G += weights[i] * G_tmp
        obj += weights[i] * R[m, n]

    return obj, G.ravel()

# Test the function with proper error handling
if __name__ == "__main__":
    # Create some example time series
    X = [
        [1, 2, 3, 4],
        [1, 2, 4, 5, 6],
        [1, 2, 3, 4, 5]
    ]
    
    # Convert to numpy arrays and ensure proper shapes
    X = [numpy.array(x).reshape(-1, 1) for x in X]
    
    # Compute barycenter with different gap penalties
    gap_penalties = [0.5, 1.0, 2.0, 5.0]
    gammas = [0.1, 0.5, 1.0]
    
    try:
        for gap_penalty in gap_penalties:
            for gamma in gammas:
                print(f"\nTrying gap_penalty: {gap_penalty}, gamma: {gamma}")
                barycenter = softdtw_barycenter_with_gaps(
                    X, 
                    gamma=gamma, 
                    gap_penalty=gap_penalty,
                    max_iter=100
                )
                print("Barycenter:", barycenter.flatten())
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Stack trace:", traceback.format_exc())