# mom_fun_wrapper.m (from MATLAB to PYTHON)
import numpy as np
from mom_fun import mom_fun
from unpack_x import unpack_x 
from repackage_D import repackage_D

def mom_fun_wrapper(x, mom_type, mom, tau_jk, z_scores, phi_ik_temp, weights, param, dtheta=False, dpsi=False):
    """
    Wrapper function for mom_fun. Parses out the parameter vector x and evaluates the moments by year.
    """
    J = param[1]  # number of occupations (including home sector)
    K = param[2]  # number of skill types
    Y = param[3]  # number of periods
    Ktheta = param[4] if len(param) > 4 else 1  # Ktheta = K if theta differs by k

    # Unpack the parameter vector x
    X = unpack_x(x, param)

    # Determine the size of moments and parameters
    if mom_type == 'level':
        M = (J - 1) + (J - 1) + 1 + K + K  # # of moments
        N = J + K + Ktheta + 1  # # of parameters
    elif mom_type in {'gap', 'gap_alt'}:
        M = 1 + K + K + 1  # # of moments
        N = 1 + 1 + K + K  # # of parameters
    else:
        M = 1
        N = 1

    # Pre-allocate storage for results
    fval_mat = np.full((M, Y), np.nan)
    deriv_mat = np.full((M, N, Y), np.nan)
    mom_hat_mat = np.full((M, Y), np.nan)
    D_mom_hat_mat = np.full((M, N, Y), np.nan)

    mom_str = []
    D_mom_str = []

    # Run the moment function for each year
    for i in range(Y):
        result = mom_fun(
            X[i], mom_type, mom[i], tau_jk[:, :, i], z_scores[:, :, i], phi_ik_temp, weights[i], param, dtheta, dpsi
        )

        # Store the results
        fval_mat[:, i] = result[0]
        if len(result) > 1:
            deriv_mat[:, :, i] = result[1]
        if len(result) > 2:
            mom_hat_mat[:, i] = result[2]
        if len(result) > 3:
            D_mom_hat_mat[:, :, i] = result[3]
        if len(result) > 4:
            mom_str.append(result[4])
        if len(result) > 5:
            D_mom_str.append(result[5])

    # Repackage results
    valid = ~np.isnan(fval_mat) & (np.any(deriv_mat != 0, axis=1))
    fval = fval_mat[valid]

    deriv = None
    if deriv_mat.any():
        deriv = repackage_D(deriv_mat, param, dtheta, dpsi)
        deriv = deriv[valid.flatten(), :]

    mom_hat = mom_hat_mat

    D_mom_hat = None
    if D_mom_hat_mat.any():
        D_mom_hat = repackage_D(D_mom_hat_mat, param, dtheta, dpsi)

    D_mom_str_repack = {}
    if D_mom_str:
        fields = D_mom_str[0].keys()
        for field in fields:
            dim = np.array(D_mom_str[0][field]).shape
            if len(dim) == 2 and (dim[1] == J + K + dtheta * Ktheta + dpsi or dim[1] in {K + 1, 1}):
                D_mat = np.array([d[field] for d in D_mom_str])
                D_mat_new = repackage_D(D_mat, param, dtheta, dpsi)
                D_mom_str_repack[field] = D_mat_new

    return fval, deriv, mom_hat, D_mom_hat, mom_str, D_mom_str, D_mom_str_repack
