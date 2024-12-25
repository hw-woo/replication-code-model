# repackage_D.m (from MATLAB to PYTHON)

import numpy as np
from scipy.linalg import block_diag

def repackage_D(D, param, dtheta=False, dpsi=False):
    """
    Repackages yearly Jacobians returned by mom_fun into the combined one used by mom_fun_wrapper.

    Parameters:
        D (numpy.ndarray): 3D array of derivatives with dimensions (M, N, Y).
        param (list or numpy.ndarray): Array of parameter settings [I, J, K, Y, (optional) Ktheta].
        dtheta (bool): If True, include derivatives with respect to theta.
        dpsi (bool): If True, include derivatives with respect to psi.

    Returns:
        numpy.ndarray: Repackaged derivatives.
    """
    J = int(param[1])  # Number of occupations
    K = int(param[2])  # Number of skill types
    M, N, Y = D.shape  # Dimensions of the input D (moments, parameters, periods)
    
    Ktheta = int(param[4]) if len(param) > 4 else 1  # Theta dimensions

    if N == J + K + Ktheta + 1:  # mom_type == 'level'
        # Calculate delimiters for slicing
        delim = np.cumsum([1, J - 1, K, dtheta * Ktheta, dpsi])

        # Extract derivatives
        D_dAH = D[:, :delim[0], :]  # Derivative w.r.t. AH
        D_dA_j = D[:, delim[0]:delim[1], :]  # Derivative w.r.t. A_j (excluding H)
        D_dbeta = D[:, delim[1]:delim[2], :]  # Derivative w.r.t. beta
        D_dtheta = D[:, delim[2]:delim[3], :] if dtheta else np.zeros((M, 0, Y))
        D_dpsi = D[:, delim[3]:delim[4], :] if dpsi else np.zeros((M, 0, Y))

        # Extract A_j fixed effects and reshape
        D_dA_j_fe = D_dA_j[:, 1:, :]  # Exclude the first column (H)
        D_dA_j_fe = D_dA_j_fe.transpose(0, 2, 1).reshape(M * Y, -1)

        # Time fixed effects
        D_dA_t_fe_cell = [np.sum(D_dAH[:, :, i] + D_dA_j[:, :, i], axis=1).reshape(M, 1) for i in range(Y)]
        D_dA_t_fe = block_diag(*D_dA_t_fe_cell)

        # Combine AH and beta into block diagonal
        D_rest_cell = [np.hstack((D_dAH[:, :, i], D_dbeta[:, :, i])) for i in range(Y)]
        D_rest = block_diag(*D_rest_cell)

        # Combine all pieces
        D_new = np.hstack([
            D_dA_j_fe,
            D_dA_t_fe,
            D_rest,
            D_dtheta.transpose(0, 2, 1).reshape(-1, Ktheta),
            D_dpsi.ravel()[:, None]
        ])

    elif N in [2 + K + K, 1]:  # mom_type == 'gap' or derivative w.r.t. Lb
        # Create block diagonal matrix
        D_cell = [D[:, :, i] for i in range(Y)]
        D_new = block_diag(*D_cell)

    else:
        raise ValueError("Invalid dimensions of input D or parameters.")

    return D_new


### Example input data ###
# D = np.random.rand(5, 10, 3)  # (M, N, Y)
# param = [100, 5, 3, 3]  # Example parameters (I, J, K, Y)
# dtheta = True
# dpsi = False

### Repackage derivatives ###
# D_new = repackage_D(D, param, dtheta, dpsi)
# print(D_new.shape)  # Output shape depends on input
