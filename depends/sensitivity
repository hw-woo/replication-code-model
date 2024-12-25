# sensitivity.m (from MATLAB to PYTHON)

import numpy as np
import pandas as pd

def sensitivity(
    dmw_dxw, dmb_dxb, dmb_dxw,
    mom_hat_white, mom_hat_black, 
    mom_data_white, mom_data_black, 
    W_w, W_b, Y, 
    select_xw, select_xb, 
    parnames, momnames, xw, xb
):
    """
    Sensitivity analysis to calculate the sensitivity of parameters to moments.

    Parameters:
        dmw_dxw (numpy.ndarray): Jacobian of moments (white) with respect to parameters (white).
        dmb_dxb (numpy.ndarray): Jacobian of moments (black) with respect to parameters (black).
        dmb_dxw (numpy.ndarray): Jacobian of moments (black) with respect to parameters (white).
        mom_hat_white (numpy.ndarray): Model moments (white).
        mom_hat_black (numpy.ndarray): Model moments (black).
        mom_data_white (numpy.ndarray): Data moments (white).
        mom_data_black (numpy.ndarray): Data moments (black).
        W_w (numpy.ndarray): Weight matrix (white).
        W_b (numpy.ndarray): Weight matrix (black).
        Y (int): Number of time periods.
        select_xw (numpy.ndarray): Boolean array for selected white parameters.
        select_xb (numpy.ndarray): Boolean array for selected black parameters.
        parnames (list): List of parameter names.
        momnames (list): List of moment names.
        xw (numpy.ndarray): Parameter estimates for white.
        xb (numpy.ndarray): Parameter estimates for black.

    Returns:
        pandas.DataFrame: Sensitivity table.
    """
    # Number of moments per year
    n_mom_w = mom_hat_white.shape[0]
    n_mom_b = mom_hat_black.shape[0]

    # Residuals
    error_w = mom_hat_white - mom_data_white
    error_b = mom_hat_black - mom_data_black
    error_w[np.isnan(error_w)] = 0
    error_b[np.isnan(error_b)] = 0

    # First-order conditions in optimization
    foc_w = (error_w.flatten() * W_w.flatten()) @ dmw_dxw
    foc_b = (error_b.flatten() * W_b.flatten()) @ dmb_dxb

    # Select parameters in the interior of the parameter space
    interior_w = (np.abs(foc_w) < 2e-6) & select_xw
    interior_b = (np.abs(foc_b) < 2e-6) & select_xb

    dmw_dxw_temp = dmw_dxw[:, interior_w]
    dmb_dxw_temp = dmb_dxw[:, interior_w]
    dmb_dxb_temp = dmb_dxb[:, interior_b]

    # Sensitivity analysis
    sensitivity_white = np.full((len(foc_w), n_mom_w * Y + n_mom_b * Y), np.nan)
    sensitivity_black = np.full((len(foc_b), n_mom_w * Y + n_mom_b * Y), np.nan)

    if interior_w.any():
        sensitivity_white[interior_w, :] = np.linalg.solve(
            (dmw_dxw_temp * W_w.flatten()) @ dmw_dxw_temp.T,
            (dmw_dxw_temp * W_w.flatten()) @ np.hstack([np.eye(n_mom_w * Y), np.zeros((n_mom_w * Y, n_mom_b * Y))])
        )

    if interior_b.any():
        sensitivity_black[interior_b, :] = np.linalg.solve(
            (dmb_dxb_temp * W_b.flatten()) @ dmb_dxb_temp.T,
            (dmb_dxb_temp * W_b.flatten()) @ np.hstack([
                -dmb_dxw_temp @ sensitivity_white[interior_w, :n_mom_w * Y],
                np.eye(n_mom_b * Y)
            ])
        )

    # Prepare outputs
    x = np.concatenate([xw.flatten(), xb.flatten()])
    sensitivity_combined = np.vstack([sensitivity_white, sensitivity_black])
    result = np.round(np.hstack([x[:, None], sensitivity_combined]), 4)

    # Create a pandas DataFrame
    Tsens = pd.DataFrame(result, index=parnames, columns=["estimate"] + momnames)

    return Tsens

### Example inputs ###
# dmw_dxw = np.random.rand(10, 5)
# dmb_dxb = np.random.rand(10, 5)
# dmb_dxw = np.random.rand(10, 5)
# mom_hat_white = np.random.rand(10, 3)
# mom_hat_black = np.random.rand(10, 3)
# mom_data_white = np.random.rand(10, 3)
# mom_data_black = np.random.rand(10, 3)
# W_w = np.random.rand(10, 3)
# W_b = np.random.rand(10, 3)
# Y = 3
# select_xw = np.array([True, False, True, True, False])
# select_xb = np.array([True, True, False, False, True])
# parnames = ["param1", "param2", "param3", "param4", "param5"]
# momnames = ["moment1", "moment2", "moment3"]
# xw = np.random.rand(5)
# xb = np.random.rand(5)

### Perform sensitivity analysis ###
# Tsens = sensitivity(
#     dmw_dxw, dmb_dxb, dmb_dxw,
#     mom_hat_white, mom_hat_black,
#     mom_data_white, mom_data_black,
#     W_w, W_b, Y,
#     select_xw, select_xb,
#     parnames, momnames, xw, xb
# )

# print(Tsens)

