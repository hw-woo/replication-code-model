# unpack_x.m (from MATLAB to PYTHON)

import numpy as np

def unpack_x(x, param, notheta=False, nopsi=False):
    """
    Parse out the parameter vector `x` into individual components.

    Parameters:
        x (numpy.ndarray): Parameter vector to unpack.
        param (list): List containing parameter details [I, J, K, Y, Ktheta (optional)].
        notheta (bool): If True, theta is not included in the output.
        nopsi (bool): If True, psi is not included in the output.

    Returns:
        tuple: Contains the following unpacked parameters:
            - X: Combined parameter vector for `mom_fun`.
            - A_jt: Y x J matrix for occupation parameters.
            - beta_kt: Y x K matrix for task prices.
            - delta_eta_kt: Y x K matrix for pecuniary barriers.
            - delta_eta_np_kt: Y x K matrix for non-pecuniary barriers.
            - theta: Frechet shape parameters (1 x K or empty if `notheta=True`).
            - psi: Frechet shape parameter for occupational preferences (scalar or empty if `nopsi=True`).
            - A_j_fe: Primitives for occupation fixed effects (vector).
            - A_t_fe: Time fixed effects (Y x 1).
            - A_Ht: Home preference (Y x 1).
    """
    J = param[1]
    K = param[2]
    Y = param[3]
    Ktheta = param[4] if len(param) > 4 else 1

    # Parse psi
    if not nopsi:
        psi = x[-1]
        x = x[:-1]
    else:
        psi = None

    # Parse theta
    if not notheta:
        theta = x[-Ktheta:]
        x = x[:-Ktheta]
    else:
        theta = None

    # Parse remaining components
    delim = np.cumsum([J - 2, Y, Y * (1 + K), Y * K * 2])
    A_j_fe = x[:delim[0]]  # Occupation fixed effects
    A_t_fe = x[delim[0]:delim[1]]  # Time fixed effects (Y x 1)
    A_t_fe = A_t_fe.reshape(-1, 1)

    x_rest = x[delim[1]:delim[2]].reshape(Y, -1)  # Y x (1+K)
    A_Ht = x_rest[:, 0]  # Home preference (Y x 1)
    beta_kt = x_rest[:, 1:1 + K]  # Task prices (Y x K)

    if len(x) == delim[2]:
        delta_eta_kt = np.zeros((Y, K))  # Default: delta_eta_k = 0
        delta_eta_np_kt = np.zeros((Y, K))  # Default: delta_eta_np_k = 0
    else:
        delta_eta_kt_temp = x[delim[2]:delim[3]].reshape(Y, -1)  # Y x 2K
        delta_eta_kt = delta_eta_kt_temp[:, :K]  # Pecuniary barriers (Y x K)
        delta_eta_np_kt = delta_eta_kt_temp[:, K:]  # Non-pecuniary barriers (Y x K)

    # Calculate A_jt
    A_jt = np.hstack([A_Ht.reshape(-1, 1) + A_t_fe, np.column_stack([np.zeros(Y), A_j_fe]) + A_t_fe])

    # Construct the parameter vector X for mom_fun
    X = np.hstack([
        A_jt,
        beta_kt,
        delta_eta_kt,
        delta_eta_np_kt,
        np.tile(np.hstack([theta, psi]), (Y, 1)) if theta is not None and psi is not None else np.zeros((Y, 0))
    ])

    return X, A_jt, beta_kt, delta_eta_kt, delta_eta_np_kt, theta, psi, A_j_fe, A_t_fe, A_Ht
