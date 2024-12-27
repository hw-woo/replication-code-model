# unpack_xb.m (from MATLAB to PYTHON)

import numpy as np

def unpack_xb(xb, param, A_j=None, beta_k=None):
    """
    Unpack the parameter vector `xb` into individual components for race-specific parameters.

    Parameters:
        xb (numpy.ndarray): Parameter vector for blacks to unpack.
        param (list): List containing parameter details [I, J, K, Y, Ktheta (optional)].
        A_j (numpy.ndarray, optional): Occupation fixed effects (length J). Default is NaN.
        beta_k (numpy.ndarray, optional): Task prices (length K). Default is NaN.

    Returns:
        tuple: Contains the following unpacked parameters:
            - Xb: Combined parameter vector for `mom_fun` (if `A_j` and `beta_k` are provided).
            - A_j_b: Adjusted occupation fixed effects for blacks.
            - A_gap: Gap in occupation preferences.
            - AH_gap: Gap in home preferences.
            - DE_k: Race-specific pecuniary barriers (length K).
            - gamma_k: Race-specific non-pecuniary barriers (length K).
    """
    J = param[1]
    K = param[2]

    # Default values for A_j and beta_k
    if A_j is None:
        A_j = np.full((1, J), np.nan)
    if beta_k is None:
        beta_k = np.full((1, K), np.nan)

    # Parse the delimiters
    delim_b = np.cumsum([1, 1, K, K])  # A_gap, A_H_gap, DE_k, gamma_k

    # Ensure `xb` is a 2D array with rows corresponding to years
    if xb.ndim == 1:
        xb = xb.reshape(1, -1)

    # Unpack components
    A_gap = xb[:, delim_b[0] - 1]
    AH_gap = xb[:, delim_b[1] - 1]
    A_j_b = np.hstack([A_j[:, :1] + AH_gap[:, np.newaxis], A_j[:, 1:] + A_gap[:, np.newaxis]])
    DE_k = xb[:, delim_b[1]:delim_b[2]]
    gamma_k = xb[:, delim_b[2]:delim_b[3]]

    # Construct the combined parameter vector `Xb` if `A_j` and `beta_k` are provided
    if A_j is not None and beta_k is not None:
        Xb = np.hstack([A_j_b, beta_k, DE_k, gamma_k])
    else:
        Xb = None

    return Xb, A_j_b, A_gap, AH_gap, DE_k, gamma_k
