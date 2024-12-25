# decompose_trends_helper.m (from MATLAB to PYTHON)
import numpy as np
from mom_fun import mom_fun

# Helper function to interpolate between two values based on t
def interpolate(value_start, value_end, t):
    return value_start + (value_end - value_start) * t

def decompose_trends_helper(t, x0_white, x1_white, x0_black, x1_black, tau_jk0, tau_jk1, z_scores0, z_scores1, LFb0, LFb1, phi_ik_temp, param):
    """
    Decomposes trends by evaluating derivatives of the moment function at time t between pre- and post-periods.

    Parameters:
        t (float): Time parameter (0 <= t <= 1), where 0 is the pre-period and 1 is the post-period.
        x0_white (np.array): Parameter vector for whites in pre-period.
        x1_white (np.array): Parameter vector for whites in post-period.
        x0_black (np.array): Parameter vector for blacks in pre-period.
        x1_black (np.array): Parameter vector for blacks in post-period.
        tau_jk0 (np.array): Rescaled tasks for pre-period.
        tau_jk1 (np.array): Rescaled tasks for post-period.
        z_scores0 (np.array): Task requirements in pre-period.
        z_scores1 (np.array): Task requirements in post-period.
        LFb0 (float): Size of Black labor force in pre-period.
        LFb1 (float): Size of Black labor force in post-period.
        phi_ik_temp (np.array): Frechet draws for skills.
        param (np.array): Miscellaneous parameters.

    Returns:
        D (np.array): Stacked derivatives of the moment function.
        mom_str_white (dict): Moments structure for whites.
        mom_str_black (dict): Moments structure for blacks.
    """
    J, K = int(param[1]), int(param[2])  # Number of occupations and skill types

    # Interpolate values for time t
    x_white = interpolate(x0_white, x1_white, t)
    x_black = interpolate(x0_black, x1_black, t)
    LFb = interpolate(LFb0, LFb1, t)
    tau_jk = interpolate(tau_jk0, tau_jk1, t)
    z_scores = interpolate(z_scores0, z_scores1, t)

    # Evaluate derivatives for whites
    _, _, _, _, mom_str_white, D_mom_str_white, D_mom_str_white_dtau_H = mom_fun(x_white, 'level', None, tau_jk, z_scores, phi_ik_temp, None, param)

    # Prepare moments for blacks
    mom_black = {
        'w_j_white': mom_str_white['w_j'],
        'l_j_white': mom_str_white['l_j'],
        'tp_white': mom_str_white['tp'],
        'agg_wage_white': mom_str_white['agg_wage'],
        'LFb': LFb,
        'Dl_j_white': D_mom_str_white['Dl_j'],
        'Dl_j_dtau_H_white': D_mom_str_white_dtau_H['Dl_j']
    }

    # Evaluate derivatives for blacks
    _, _, _, _, mom_str_black, D_mom_str_black, D_mom_str_black_dtau_H = mom_fun(x_black, 'gap', mom_black, tau_jk, z_scores, phi_ik_temp, None, param)

    # Parse out parameter dimensions
    delim_temp = [J, K, K, K]
    delim = np.cumsum(delim_temp)

    # Compute wage gaps derivatives
    dw_gap_dA = D_mom_str_black['D_agg_wage'][:, :delim[0]] - D_mom_str_white['D_agg_wage'][:, :delim[0]]
    dw_gap_dbeta = D_mom_str_black['D_agg_wage'][:, delim[0]:delim[1]] - D_mom_str_white['D_agg_wage'][:, delim[0]:delim[1]]
    dw_gap_dDE = D_mom_str_black['D_agg_wage'][:, delim[1]:delim[2]]
    dw_gap_dgamma = D_mom_str_black['D_agg_wage'][:, delim[2]:delim[3]]
    dw_gap_dtau_H = D_mom_str_black_dtau_H['D_agg_wage'] - D_mom_str_white_dtau_H['D_agg_wage']

    # Compute task price gaps derivatives
    dtp_gap_dA = D_mom_str_black['D_tp'][:, :delim[0]] - D_mom_str_white['D_tp'][:, :delim[0]]
    dtp_gap_dbeta = D_mom_str_black['D_tp'][:, delim[0]:delim[1]] - D_mom_str_white['D_tp'][:, delim[0]:delim[1]]
    dtp_gap_dDE = D_mom_str_black['D_tp'][:, delim[1]:delim[2]]
    dtp_gap_dgamma = D_mom_str_black['D_tp'][:, delim[2]:delim[3]]
    dtp_gap_dtau_H = D_mom_str_black_dtau_H['D_tp'] - D_mom_str_white_dtau_H['D_tp']

    # Compute task content gaps derivatives
    dtc_gap_dA = D_mom_str_black['D_tc_gap_dwhite'][:, :delim[0]]
    dtc_gap_dbeta = D_mom_str_black['D_tc_gap_dwhite'][:, delim[0]:delim[1]]
    dtc_gap_dDE = D_mom_str_black['D_tc_gap'][:, delim[1]:delim[2]]
    dtc_gap_dgamma = D_mom_str_black['D_tc_gap'][:, delim[2]:delim[3]]
    dtc_gap_dtau_H = D_mom_str_black_dtau_H['D_tc_gap']

    # Stack all derivatives
    D_w = np.hstack([dw_gap_dA, dw_gap_dbeta, dw_gap_dDE, dw_gap_dgamma, dw_gap_dtau_H])
    D_tp = np.hstack([dtp_gap_dA, dtp_gap_dbeta, dtp_gap_dDE, dtp_gap_dgamma, dtp_gap_dtau_H])
    D_tc = np.hstack([dtc_gap_dA, dtc_gap_dbeta, dtc_gap_dDE, dtc_gap_dgamma, dtc_gap_dtau_H])

    D = np.vstack([D_w, D_tp, D_tc])

    return D, mom_str_white, mom_str_black
