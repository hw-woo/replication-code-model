# prctile_weighted.m (from MATLAB to PYTHON)

import numpy as np

def prctile_weighted(X, p, w=None):
    """
    Computes weighted percentiles of a dataset.

    Parameters:
        X (numpy array): 1D array of data values.
        p (numpy array): 1D array of percentiles to compute (values between 0 and 100).
        w (numpy array, optional): 1D array of weights corresponding to `X`. If not provided, equal weights are assumed.

    Returns:
        numpy array: The weighted percentiles corresponding to `p`.
    """
    if w is None:
        w = np.ones_like(X)

    # Remove NaNs
    valid = ~np.isnan(X)
    X = X[valid]
    w = w[valid]

    # Sort X and weights
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    w_sorted = w[sorted_indices]

    # Compute cumulative weights
    W_cumsum = np.cumsum(w_sorted) / np.sum(w_sorted)

    # Compute percentiles
    x_p = np.empty_like(p, dtype=float)
    for j, P in enumerate(p):
        P_normalized = P / 100.0
        if P_normalized == 1.0:
            i = len(W_cumsum) - 1
        else:
            i = np.searchsorted(W_cumsum, P_normalized, side="right")
        
        if i > 0 and W_cumsum[i - 1] == P_normalized:
            x_p[j] = (X_sorted[i - 1] + X_sorted[i]) / 2
        else:
            x_p[j] = X_sorted[i]

    return x_p
