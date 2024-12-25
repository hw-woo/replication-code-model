# decompose_trends.m (from MATLAB to PYTHON)

import os
import numpy as np
import pandas as pd
from scipy.integrate import quad
from pathlib import Path

from unpack_x import unpack_x
from unpack_xb import unpack_xb
from unitinterval import unitinterval
from decompose_trends_helper import decompose_trends_helper

def decompose_trends(savedir_root, psi_d, theta=0, seed=0):
    """
    Decomposes trends in moments into components explained by different parameters.

    Parameters:
        savedir_root (str): Root directory for saving results.
        psi_d (float): Psi parameter.
        theta (float): Theta parameter. If 0, it will be estimated.
        seed (int): Seed for random number generation.

    Returns:
        None
    """
    estimate_theta = (theta == 0)

    if estimate_theta:
        print(f"\nDECOMPOSE TRENDS (Estimate theta): psi_d = {psi_d:.1f}")
        savedir = f"{savedir_root}/baseline_psi_d_{psi_d:.1f}/"
    else:
        print(f"\nDECOMPOSE TRENDS (Fix theta): psi_d = {psi_d:.1f}, theta = {theta:.1f}")
        savedir = f"{savedir_root}/fixtheta_psi_d_{psi_d:.1f}_theta_{theta:.1f}/"

    regions = ['all', 'south', 'nonsouth']

    for region in regions:
        print(f"\nREGION: {region.upper()}")
        savedir2 = f"{savedir}/region_{region}/"
        os.makedirs(savedir2, exist_ok=True)

        # Load settings
        settings = np.load(f"{savedir}/temp/settings.npy", allow_pickle=True).item()
        param = settings['param']
        I, J, K, Y = param

        z_scores = settings['z_scores']
        tau_jk = unitinterval(z_scores, dim=(0, 2))
        yrs = settings['yrs']

        # Generate skills
        np.random.seed(seed)
        phi_ik_temp = (-np.log(np.random.rand(I, K))) ** -1

        A_j_b = np.full((Y, J), np.nan)
        DE_bk = np.zeros((Y, K))
        gamma_bk = np.zeros((Y, K))
        A_gap = np.full(Y, np.nan)
        AH_gap = np.full(Y, np.nan)
        LFb = np.full(Y, np.nan)

        # Load results for Whites
        results_white = np.loadtxt(f"{savedir}/temp/white/x_sol_white.csv", delimiter=",")

        if estimate_theta:
            x_sol_white = np.append(results_white[1:], psi_d)
            _, A_j_w, beta_k, _, _, theta, psi = unpack_x(x_sol_white, param)
        else:
            x_sol_white = np.append(results_white[1:], [theta, psi_d])
            _, A_j_w, beta_k, _, _, _, psi = unpack_x(x_sol_white, param)

        # Load results for Blacks
        for i, year in enumerate(yrs):
            savedir_black = f"{savedir}/temp/black_region_{region}/{year}/"
            results_black = np.loadtxt(f"{savedir_black}/x_sol_black.csv", delimiter=",")
            _, A_j_b[i, :], A_gap[i], AH_gap[i], DE_bk[i, :], gamma_bk[i, :] = unpack_xb(
                results_black[1:], param, A_j_w[i, :], beta_k[i, :]
            )

            mom_black = np.load(f"{savedir_black}/mom_black.npy", allow_pickle=True).item()
            LFb[i] = mom_black['LFb']

        # Decompose trends
        phi_ik = phi_ik_temp ** (1 / theta)
        tau_H = tau_jk[0, :, :].T

        y0, y1 = np.arange(Y - 1), np.arange(1, Y)

        x0_white = np.hstack([
            A_j_w[y0, :], beta_k[y0, :], np.zeros((Y - 1, K)), np.zeros((Y - 1, K)),
            np.ones((Y - 1, 1)), psi * np.ones((Y - 1, 1))
        ])
        x1_white = np.hstack([
            A_j_w[y1, :], beta_k[y1, :], np.zeros((Y - 1, K)), np.zeros((Y - 1, K)),
            np.ones((Y - 1, 1)), psi * np.ones((Y - 1, 1))
        ])
        x0_black = np.hstack([
            A_j_b[y0, :], beta_k[y0, :], DE_bk[y0, :], gamma_bk[y0, :],
            np.ones((Y - 1, 1)), psi * np.ones((Y - 1, 1))
        ])
        x1_black = np.hstack([
            A_j_b[y1, :], beta_k[y1, :], DE_bk[y1, :], gamma_bk[y1, :],
            np.ones((Y - 1, 1)), psi * np.ones((Y - 1, 1))
        ])

        tau_jk0, tau_jk1 = tau_jk[:, :, y0], tau_jk[:, :, y1]
        z_scores0, z_scores1 = z_scores[:, :, y0], z_scores[:, :, y1]
        LFb0, LFb1 = LFb[y0], LFb[y1]

        D_all = []

        for i in range(Y - 1):
            def D_fun(t):
                return decompose_trends_helper(
                    t, x0_white[i], x1_white[i], x0_black[i], x1_black[i],
                    tau_jk0[:, :, i], tau_jk1[:, :, i],
                    z_scores0[:, :, i], z_scores1[:, :, i],
                    LFb0[i], LFb1[i], phi_ik, param
                )

            result, _ = quad(D_fun, 0, 1, epsabs=1e-4, epsrel=1e-4)
            D_all.append(result)

        # Save decomposed trends
        decomposed_path = Path(savedir2) / "decompose_trends"
        decomposed_path.mkdir(parents=True, exist_ok=True)
        np.save(decomposed_path / "D_all.npy", np.array(D_all))

    print("Decomposition complete.")
