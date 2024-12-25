# prc_rank.m (from MATLAB to PYTHON)

import numpy as np

def prc_rank(value, X, W=None):
    """
    Calculates the percentile rank of `value` within the dataset `X` using weights `W`.

    Parameters:
        value (numpy array): A 1D array of values to evaluate, size (1, k).
        X (numpy array): A 1D array of the dataset values.
        W (numpy array, optional): A 1D array of weights for the dataset values. If not provided, equal weights are assumed.

    Returns:
        numpy array: Percentile ranks of `value` in `X`.
    """
    if value.ndim != 1:
        raise ValueError("value must be a 1D array with size (k,), where k is the number of values to evaluate.")

    if W is None:
        W = np.ones_like(X)

    # Ensure that X and W are valid by removing NaNs
    valid = ~np.isnan(X)
    X = X[valid]
    W = W[valid]

    # Calculate the sum of weights
    n = np.sum(W)

    # Calculate the lower and upper percentiles
    p_lower = np.sum((X <= value[:, None]) * W, axis=1) / n * 100
    p_upper = (n - np.sum((X >= value[:, None]) * W, axis=1)) / n * 100

    # Calculate the final percentile rank
    p = (p_lower + p_upper) / 2

    return p
