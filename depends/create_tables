# create_tables.m (from MATLAB to PYTHON)

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import gamma
from sensitivity import sensitivity
from unpack_x import unpack_x
from unpack_xb import unpack_xb
from mom_fun_wrapper import mom_fun_wrapper
from prc_rank import prc_rank
from prctile_weighted import prctile_weighted

def create_tables(savedir_root, psi_d, theta=0, seed=0):
    estimate_theta = theta == 0
    np.random.seed(seed)

    if estimate_theta:
        print(f"\nCREATE TABLES (Estimate theta): psi_d = {psi_d:.1f}\n")
        savedir = os.path.join(savedir_root, f"baseline_psi_d_{psi_d:.1f}/")
    else:
        print(f"\nCREATE TABLES (Fix theta): psi_d = {psi_d:.1f}, theta = {theta:.1f}\n")
        savedir = os.path.join(savedir_root, f"fixtheta_psi_d_{psi_d:.1f}_theta_{theta:.1f}/")

    regions = ['all', 'south', 'nonsouth']

    for region in regions:
        print(f"\nREGION: {region.upper()}\n")
        savedir2 = os.path.join(savedir, f"region_{region}/")
        Path(savedir2).mkdir(parents=True, exist_ok=True)

        savedir_temp = os.path.join(savedir, "temp/")
        settings = np.load(os.path.join(savedir_temp, "settings.npz"), allow_pickle=True)['settings'].item()

        param = settings['param']
        I, J, K, Y = param
        z_scores = settings['z_scores']
        tau_jk = (z_scores - z_scores.min(axis=(1, 2), keepdims=True)) / (
            z_scores.max(axis=(1, 2), keepdims=True) - z_scores.min(axis=(1, 2), keepdims=True))
        yrs = settings['yrs']

        phi_ik_temp = (-np.log(np.random.rand(I, K))) ** -1

        f_val_b = np.full(Y, np.nan)
        A_j_b = np.full((Y, J), np.nan)
        DE_bk = np.zeros((Y, K))
        gamma_bk = np.zeros((Y, K))
        A_gap = np.full(Y, np.nan)
        A_H_gap = np.full(Y, np.nan)

        parnames_fe = [f"A_fe_j{i}" for i in range(2, J)] + ["A_fe_t"]
        parnames_w = ["AH"] + [f"beta_{t}" for t in ["cont", "abs", "man", "rout"]]
        parnames_b = ["A_gap", "AH_gap"] + [f"DE_{t}" for t in ["cont", "abs", "man", "rout"]] + [f"gamma_{t}" for t in ["cont", "abs", "man", "rout"]]
        parnames = parnames_fe + parnames_w + parnames_b + ["theta", "psi"]

        savedir_white = os.path.join(savedir, "temp/white/")
        results_white = np.loadtxt(os.path.join(savedir_white, "x_sol_white.csv"), delimiter=",")

        if estimate_theta:
            x_sol_white = np.append(results_white[1:], psi_d)
            _, A_j_w, beta_k, _, _, theta, psi, A_j_fe, A_t_fe, A_Ht = unpack_x(x_sol_white, param)
        else:
            x_sol_white = np.append(results_white[1:], [theta, psi_d])
            _, A_j_w, beta_k, _, _, _, psi, A_j_fe, A_t_fe, A_Ht = unpack_x(x_sol_white, param)

        mom_white = np.load(os.path.join(savedir_white, "mom_white.npy"), allow_pickle=True)
        weights_white = np.load(os.path.join(savedir_white, "weights_white.npy"), allow_pickle=True)

        _, _, mom_hat_white, dmw_dxw, mom_str_white, D_mom_str_white = mom_fun_wrapper(
            x_sol_white, 'level', mom_white, tau_jk, z_scores, phi_ik_temp, weights_white, param, compute_gradients=True
        )

        x_sol_black = np.full((len(parnames_b), Y), np.nan)
        for i in range(Y):
            savedir_black = os.path.join(savedir, f"temp/black_region_{region}/{yrs[i]}/")
            results_black = np.loadtxt(os.path.join(savedir_black, "x_sol_black.csv"), delimiter=",")
            f_val_b[i] = results_black[0]
            x_sol_black[:, i] = results_black[1:]

            _, A_j_b[i, :], A_gap[i], A_H_gap[i], DE_bk[i, :], gamma_bk[i, :] = unpack_xb(x_sol_black[:, i], param, A_j_w[i, :], beta_k[i, :])

        # Construct parameter estimates table
        estimates = np.hstack([
            np.tile(A_j_fe, (Y, 1)),
            A_t_fe[:, None],
            A_Ht[:, None],
            beta_k,
            A_gap[:, None],
            A_H_gap[:, None],
            DE_bk,
            gamma_bk,
            np.vstack([[theta, psi_d], np.full((Y-1, 2), np.nan)])
        ])

        T_param = pd.DataFrame(estimates, index=yrs, columns=parnames)
        T_param.to_csv(os.path.join(savedir2, "0_estimates.csv"))

        # Save results for sensitivity analysis
        Tsens = sensitivity(
            dmw_dxw, None, None, mom_hat_white, None, None, None,
            None, None, Y, None, None, parnames, None, x_sol_white, x_sol_black
        )
        Tsens.to_csv(os.path.join(savedir2, "sensitivity_analysis.csv"))
