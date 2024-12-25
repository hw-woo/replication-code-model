# mom_fun.m (from MATLAB to PYTHON)
import numpy as np

def mom_fun(x, mom_type, mom, tau_jk, z_scores, phi_ik_temp, weights, param, dtheta=False, dpsi=False):
    """
    Evaluate moments given the parameter vector x and calculate the distances from data counterparts.

    Parameters:
    x : numpy.ndarray
        Parameter vector.
    mom_type : str
        Moment type ('level' or 'gap').
    mom : dict
        Moments from data.
    tau_jk : numpy.ndarray
        Task requirements, rescaled to [0,1].
    z_scores : numpy.ndarray
        Task requirements, in z-score units.
    phi_ik_temp : numpy.ndarray
        Frechet draws with shape 1, used for skills.
    weights : dict
        Weights for moments.
    param : list or numpy.ndarray
        Miscellaneous parameters.
    dtheta : bool, optional
        Calculate the derivative wrt theta (default is False).
    dpsi : bool, optional
        Calculate the derivative wrt psi (default is False).

    Returns:
    fval : numpy.ndarray
        Weighted distances between model moments and data moments.
    deriv : numpy.ndarray
        Derivatives of the moments, if requested.
    mom_hat : numpy.ndarray
        Predicted moments.
    """

    # Parse parameters
    I, J, K = param[:3]  # Number of skill draws, occupations, and skill types
    Ktheta = param[4] if len(param) > 4 else 1

    # Parse parameter vector x
    delim = np.cumsum([J, K, K, K, Ktheta, 1])
    A_j = x[:delim[0]]  # Occupation returns (1 x J)
    beta_k = x[delim[0]:delim[1]]  # Task prices (1 x K)
    delta_eta_k = x[delim[1]:delim[2]]  # Race-specific barriers (pecuniary)
    gamma_k = x[delim[2]:delim[3]]  # Race-specific barriers (non-pecuniary)
    theta_k = x[delim[3]:delim[4]]  # Frechet shape parameter
    psi_d = x[delim[4]]  # Frechet shape parameter for occupational preferences

    # Adjust Frechet draws for skill distribution
    phi_ik = phi_ik_temp ** (1.0 / theta_k)

    # Compute log wages and employment
    w_ij = A_j + (beta_k * (phi_ik + delta_eta_k)) @ tau_jk.T  # Log wage (I x J)
    w_ij_np = A_j + (beta_k * (phi_ik + delta_eta_k + gamma_k)) @ tau_jk.T
    w_ij_np -= np.max(w_ij_np)  # Normalize for numerical stability

    # Fraction of workers sorting into each occupation
    l_ij = np.exp(psi_d * w_ij_np) / np.sum(np.exp(psi_d * w_ij_np), axis=1, keepdims=True)

    # Employment and average log wage in each occupation
    l_j = np.mean(l_ij, axis=0)  # Employment (J x 1)
    w_j = np.sum(l_ij * w_ij, axis=0) / l_j  # Average log wage (J x 1)
    w_j[0] = np.nan  # Wage undefined for home sector

    # Employment share of each occupation
    rho_j = l_j[1:] / (1 - l_j[0])

    # Calculate moments: levels
    if mom_type == 'level':
        # Aggregate task content
        tc = rho_j @ z_scores[1:, :]

        # Aggregate task prices (weighted regression)
        valid = ~np.isnan(w_j)
        X_tp = np.hstack((np.ones((np.sum(valid), 1)), z_scores[valid, :]))  # Regressors
        W_tp = l_j[valid]  # Weights
        XWX_tp = X_tp.T @ (W_tp[:, None] * X_tp)
        tp_temp = np.linalg.solve(XWX_tp, X_tp.T @ (W_tp * w_j[valid]))
        tp = tp_temp[1:].T

        # Aggregate wage (mean log wage)
        agg_wage = rho_j @ w_j[1:]

        # Construct model moments
        mom_hat = np.hstack([
            w_j[1:],  # Log wages
            np.log(rho_j),  # Log employment shares
            np.log(l_j[0]),  # Log home sector share
            tc,  # Task contents
            tp,  # Task prices
        ])

        # Data moments
        mom_data = np.hstack([
            mom['ln_w'],
            mom['ln_emp'],
            mom['ln_l_H'],
            mom['tc'],
            mom['tp'],
        ])

        # Weights
        W_vec = np.hstack([
            weights['ln_w'],
            weights['ln_emp'],
            weights['ln_l_H'],
            weights['tc'],
            weights['tp'],
        ])

    elif mom_type == 'gap':
        # Calculate gaps for 'gap' moments
        ln_l_H_gap = np.log(l_j[0]) - mom['ln_l_H']
        tc_gap = tc - mom['tc']
        tp_gap = tp - mom['tp']
        wage_gap = agg_wage - mom['agg_wage']

        # Construct model moments
        mom_hat = np.hstack([
            ln_l_H_gap,
            tc_gap,
            tp_gap,
            wage_gap,
        ])

        # Data moments
        mom_data = np.hstack([
            mom['ln_l_H_gap'],
            mom['tc_gap'],
            mom['tp_gap'],
            mom['wage_gap'],
        ])

        # Weights
        W_vec = np.hstack([
            weights['ln_l_H_gap'],
            weights['tc_gap'],
            weights['tp_gap'],
            weights['wage_gap'],
        ])

    else:
        raise ValueError("Invalid mom_type. Must be 'level' or 'gap'.")

    # Calculate distances
    diff = mom_hat - mom_data
    diff[np.isnan(mom_data)] = 0  # Handle missing moments

    # Weighted distances
    fval = np.sqrt(W_vec) * diff

    # Optionally calculate derivatives (not included in this basic implementation)
    deriv = None  # Placeholder if needed

    return fval, deriv, mom_hat
