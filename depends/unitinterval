# unitinterval.m (from MATLAB to PYTHON)

import numpy as np

def unitinterval(x, dim=0):
    """
    Rescale input array x to the [0, 1] interval along the specified dimension.

    Parameters:
        x (numpy.ndarray): Input array to be rescaled.
        dim (int): Dimension along which to rescale. Default is 0.

    Returns:
        numpy.ndarray: Rescaled array with values in the [0, 1] interval.
    """
    x_min = np.min(x, axis=dim, keepdims=True)
    x_max = np.max(x, axis=dim, keepdims=True)
    y = (x - x_min) / (x_max - x_min)
    return y
