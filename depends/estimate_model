# estimate_model.m (from MATLAB to PYTHON)

import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.optimize import Bounds
from pathlib import Path

from mom_fun_wrapper import mom_fun_wrapper
from unpack_x import unpack_x
from unpack_xb import unpack_xb
from unitinterval import unitinterval


def estimate_model(savedir_root, psi_d, theta=0, seed=0, n_points=6):
    """
    Estimates parameters of the model using a minimum-distance estimator.

    Parameters:
        savedir_root (str): Root directory for saving results.
        psi_d (float): Value of psi.
        theta (float): Value of theta. If 0, the code will estimate it.
        seed (int): Seed for random number generation.
        n_points (int): Number of initial points to try in optimization.

    Returns:
        None
    """
    estimate_theta = theta == 0

    # Set directory for saving results
    if estimate_theta:
        print(f"\nESTIMATION (Estimate theta): psi_d = {psi_d:.1f}")
        savedir = f"{savedir_root}/baseline_psi_d_{psi_d:.1f}/"
    else:
        print(f"\nESTIMATION (Fix theta): psi_d = {psi_d:.1f}, theta = {theta:.1f}")
        savedir = f"{savedir_root}/fixtheta_psi_d_{psi_d:.1f}_theta_{theta:.1f}/"

    # Create necessary directories
    os.makedirs(savedir, exist_ok=True)

    # Data years
    yrs = [1960, 1970, 1980, 1990, 2000, 2012, 2018]
    Y = len(yrs)

    # Load data moments
    data_mom = pd.read_csv("../output/acs/moments_all.csv", index_col=0)
    data_occ = pd.read_csv("../output/acs/moments_ztasks_broad.csv")

    task_types = ["cont", "abs", "man", "rout"]
    z_scores_emp = data_occ[[f"ztask_{t}" for t in task_types]].values
    z_scores_emp = np.repeat(z_scores_emp[:, :, np.newaxis], Y, axis=2)

    # Only home sector task requirements change over time
    z_scores_home = data_mom[[f"tau_H_{t}" for t in task_types]].values.T
    z_scores_home = np.expand_dims(z_scores_home, axis=0)
    z_scores = np.concatenate([z_scores_home, z_scores_emp], axis=0)

    # Rescale to [0, 1]
    tau_jk = unitinterval(z_scores, dim=(0, 2))

    # Number of occupations, skill types, and skill draws
    J, K, _ = z_scores.shape
    I = 12 ** K
    np.random.seed(seed)
    phi_ik_temp = (-np.log(np.random.rand(I, K))) ** -1

    # Parameter settings
    param = [I, J, K, Y]
    settings = {"param": param, "z_scores": z_scores, "yrs": yrs}

    # Save settings
    temp_dir = Path(savedir) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    np.save(temp_dir / "settings.npy", settings)

    print("ESTIMATING RACE-NEUTRAL PARAMETERS")

    # Construct moments for whites
    mom_white = []
    weights_white = []

    for i in range(Y):
        mom_white.append({
            "ln_w": data_mom.loc[str(yrs[i]), [f"ln_w_{j}" for j in range(1, J)]].values,
            "ln_emp": data_mom.loc[str(yrs[i]), [f"ln_emp_{j}" for j in range(1, J)]].values,
            "ln_l_H": data_mom.loc[str(yrs[i]), "ln_l_H"],
            "tc": data_mom.loc[str(yrs[i]), [f"tc_{t}" for t in task_types]].values,
            "tp": data_mom.loc[str(yrs[i]), [f"tp_{t}" for t in task_types]].values,
        })

        weights_white.append({
            "ln_w": np.ones(J - 1) / (J - 1) / Y,
            "ln_emp": np.ones(J - 1) / (J - 1) / Y,
            "ln_l_H": 1,
            "tc": np.ones(K),
            "tp": np.ones(K) * 25,
        })

    # Save moments
    np.save(temp_dir / "mom_white.npy", mom_white)
    np.save(temp_dir / "weights_white.npy", weights_white)

    # Define bounds and initial values
    lb = np.concatenate([
        -10 * np.ones(J - 2),
        np.zeros(Y),
        np.tile(np.concatenate([[0], np.zeros(K)]), Y),
    ])
    ub = np.concatenate([
        10 * np.ones(J - 2),
        20 * np.ones(Y),
        np.tile(np.concatenate([[20], 10 * np.ones(K)]), Y),
    ])
    x0 = np.concatenate([
        np.zeros(J - 2),
        np.zeros(Y),
        np.tile(np.concatenate([[0], 0.5 * np.ones(K)]), Y),
    ])

    if estimate_theta:
        lb = np.append(lb, 2)
        ub = np.append(ub, 10)
        x0 = np.append(x0, 4)

    # Define the objective function
    if estimate_theta:
        fun_white = lambda x: mom_fun_wrapper(np.append(x, psi_d), 'level', mom_white, tau_jk, z_scores, phi_ik_temp, weights_white, param, True, False)
    else:
        fun_white = lambda x: mom_fun_wrapper(np.append(x, [theta, psi_d]), 'level', mom_white, tau_jk, z_scores, phi_ik_temp, weights_white, param, False, False)

    # Perform optimization
    bounds = Bounds(lb, ub)
    result = least_squares(fun_white, x0, bounds=bounds, xtol=1e-8, ftol=1e-8, gtol=1e-8, verbose=2)

    # Save results
    np.save(temp_dir / "x_sol_white.npy", result.x)

    # Parse results
    if estimate_theta:
        _, A_j, beta_k, _, _, theta = unpack_x(result.x, param, nopsi=True)
    else:
        _, A_j, beta_k = unpack_x(result.x, param, nopsi=True, notheta=True)

    # Proceed with Step 2 (Black-White Gaps)...

    # (The remaining code follows a similar structure for estimating race-specific parameters)
