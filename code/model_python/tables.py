# tables.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from utils import unitinterval, unpack_x, unpack_xb
from estimation import mom_fun_wrapper, decompose_trends_helper

def create_tables(savedir_root: str, psi_d: float, theta: float = 0, 
                 options: Dict = None) -> None:
    """
    Create tables of parameter estimates, moments, and sensitivity matrices.
    """
    if options is None:
        options = {'seed': 0}
    
    estimate_theta = theta == 0
    seed = options['seed']
    
    # Set paths
    if estimate_theta:
        print(f'\nCREATE TABLES (Estimate theta): psi_d = {psi_d:.1f}\n')
        savedir = Path(savedir_root) / f'baseline_psi_d_{psi_d:.1f}'
    else:
        print(f'\nCREATE TABLES (Fix theta): psi_d = {psi_d:.1f}, theta = {theta:.1f}\n')
        savedir = Path(savedir_root) / f'fixtheta_psi_d_{psi_d:.1f}_theta_{theta:.1f}'

    # Process regions
    regions = ['all', 'south', 'nonsouth']
    
    for r in regions:
        print(f'\nREGION: {r.upper()}\n')
        savedir2 = savedir / f'region_{r}'
        savedir2.mkdir(parents=True, exist_ok=True)

        # Load settings
        settings = np.load(savedir / 'temp' / 'settings.npy', allow_pickle=True).item()
        param = settings['param']
        I, J, K, Y = param
        
        # Task types and data years
        task_types = ['cont', 'abs', 'man', 'rout']
        z_scores = settings['z_scores']
        tau_jk = unitinterval(z_scores, dim=1)
        years = settings['years']

        # Draw skills
        np.random.seed(seed)
        phi_ik_temp = (-np.log(np.random.random((I, K))))**-1

        # Pre-allocate arrays
        f_val_b = np.full(Y, np.nan)
        A_j_b = np.full((Y, J), np.nan)
        DE_bk = np.zeros((Y, K))
        gamma_bk = np.zeros((Y, K))
        A_gap = np.full(Y, np.nan)
        A_H_gap = np.full(Y, np.nan)

        # Define parameter and moment names
        parnames_fe = [f'A_fe_j{j}' for j in range(2, J)]
        parnames_fe.append('A_fe_t')
        
        parnames_w = ['AH']
        parnames_w.extend([f'beta_{t}' for t in task_types])
        
        parnames_b = ['A_gap', 'AH_gap']
        parnames_b.extend([f'DE_{t}' for t in task_types])
        parnames_b.extend([f'gamma_{t}' for t in task_types])
        
        parnames = parnames_fe + parnames_w + parnames_b + ['theta', 'psi']

        # Load white results
        results_white = np.load(savedir / 'temp/white/x_sol_white.npy')
        if estimate_theta:
            x_sol_white = np.concatenate([results_white[1:], [psi_d]])
            X_w = unpack_x(x_sol_white, param)[0]
        else:
            x_sol_white = np.concatenate([results_white[1:], [theta, psi_d]])
            X_w = unpack_x(x_sol_white, param, options={'notheta': True})[0]

        # Load moments and weights for whites
        mom_white = np.load(savedir / 'temp/white/mom_white.npy', allow_pickle=True).item()
        weights_white = np.load(savedir / 'temp/white/weights_white.npy', allow_pickle=True).item()
        
        # Evaluate moments
        mom_hat_white, dmw_dxw = mom_fun_wrapper(x_sol_white, 'level', mom_white, tau_jk,
                                                z_scores, phi_ik_temp, weights_white, param, 
                                                estimate_theta, True)[:2]

        # Load and process black results
        x_sol_black = np.full((len(parnames_b), Y), np.nan)
        
        for i in range(Y):
            black_dir = savedir / f'temp/black_region_{r}/{years[i]}'
            results_black = np.load(black_dir / 'x_sol_black.npy')
            
            f_val_b[i] = results_black[0]
            x_sol_black[:, i] = results_black[1:]
            
            X_b = unpack_xb(x_sol_black[:, i], param, X_w[i, :J], X_w[i, J:J+K])[0]
            A_j_b[i] = X_b[:J]
            A_gap[i] = x_sol_black[0, i]
            A_H_gap[i] = x_sol_black[1, i]
            DE_bk[i] = x_sol_black[2:2+K, i]
            gamma_bk[i] = x_sol_black[2+K:2+2*K, i]

            # Load black moments
            mom_black = np.load(black_dir / 'mom_black.npy', allow_pickle=True).item()
            weights_black = np.load(black_dir / 'weights_black.npy', allow_pickle=True).item()

        # Create parameter estimates table
        estimates = np.column_stack([
            np.tile(X_w[:J-2], (Y, 1)),  # A_j_fe
            X_w[:, -2:],                  # A_t_fe
            X_w[:, J:J+K],               # beta_k
            A_gap, A_H_gap, DE_bk, gamma_bk,
            np.array([theta, psi_d] if estimate_theta else [np.nan, np.nan])
        ])

        T_param = pd.DataFrame(estimates, index=years, columns=parnames)
        T_param.index.name = 'year'

        # Create moments table
        momnames_w = [f'ln_w_{j}' for j in range(1, J)]
        momnames_w.extend([f'ln_emp_{j}' for j in range(1, J)])
        momnames_w.extend(['ln_l_H'])
        momnames_w.extend([f'tc_{t}' for t in task_types])
        momnames_w.extend([f'tp_{t}' for t in task_types])

        momnames_b = ['ln_l_H_gap']
        momnames_b.extend([f'tc_gap_{t}' for t in task_types])
        momnames_b.extend([f'tp_gap_{t}' for t in task_types])
        momnames_b.append('wage_gap')

        T_mom = pd.DataFrame(np.vstack([mom_hat_white, mom_hat_black]).T,
                           index=years, columns=momnames_w + momnames_b)
        T_mom.index.name = 'year'

        # Save tables
        T_param.to_csv(savedir2 / '0_estimates.csv')
        T_param[['theta', 'psi']].to_csv(savedir2 / '0_estimates_summary.csv')
        T_mom.to_csv(savedir2 / '1_moments.csv')
        T_mom[momnames_w[:3] + momnames_b].to_csv(savedir2 / '1_moments_summary.csv')

        # Calculate and save additional moments
        calculate_additional_moments(savedir2, T_param, T_mom, param, years, task_types)

def save_tables(savedir_root: str, psi_d: float, theta: float = 0) -> None:
    """
    Save LaTeX tables for the paper.
    """
    estimate_theta = theta == 0
    print('\nSAVE TEX TABLES\n')
    
    # Set paths
    if estimate_theta:
        savedir = Path(savedir_root) / f'baseline_psi_d_{psi_d:.1f}'
    else:
        savedir = Path(savedir_root) / f'fixtheta_psi_d_{psi_d:.1f}_theta_{theta:.1f}'
        
    savedir2 = savedir / 'region_all'
    
    # Load settings and data
    settings = np.load(savedir / 'temp/settings.npy', allow_pickle=True).item()
    task_types = ['cont', 'abs', 'man', 'rout']
    years = settings['years']
    Y = len(years)
    
    # Load results
    T_param = pd.read_csv(savedir2 / '0_estimates.csv', index_col=0)
    Tsens_FT_white = pd.read_csv(savedir2 / 'sensitivity/table_1_sens_white_fixtheta.csv', 
                                index_col=0)
    Tsens_white = pd.read_csv(savedir2 / 'sensitivity/table_0_sens_white.csv', index_col=0)
    
    # Create Table 2
    create_table_2(T_param, task_types, years)
    
    # Create Tables R7-R10
    create_table_R7(Tsens_FT_white, years)
    create_table_R8(Tsens_FT_white, years)
    create_table_R9(Tsens_white, years)
    create_table_R10(savedir2, years, task_types)

def decompose_trends(savedir_root: str, psi_d: float, theta: float = 0, 
                    options: Dict = None) -> None:
    """
    Decompose trends in moments into components explained by different parameters.
    """
    if options is None:
        options = {'seed': 0}
        
    estimate_theta = theta == 0
    seed = options['seed']
    
    # Set paths
    if estimate_theta:
        print(f'\nDECOMPOSE TRENDS (Estimate theta): psi_d = {psi_d:.1f}\n')
        savedir = Path(savedir_root) / f'baseline_psi_d_{psi_d:.1f}'
    else:
        print(f'\nDECOMPOSE TRENDS (Fix theta): psi_d = {psi_d:.1f}, theta = {theta:.1f}\n')
        savedir = Path(savedir_root) / f'fixtheta_psi_d_{psi_d:.1f}_theta_{theta:.1f}'

    # Process regions
    regions = ['all', 'south', 'nonsouth']
    
    for r in regions:
        print(f'\nREGION: {r.upper()}\n')
        savedir2 = savedir / f'region_{r}'
        savedir2.mkdir(parents=True, exist_ok=True)

        # Load settings
        settings = np.load(savedir / 'temp/settings.npy', allow_pickle=True).item()
        param = settings['param']
        years = settings['years']
        Y = len(years)

        # Initialize arrays for decomposition
        D_list = []
        
        # Process each year pair
        for i in range(Y-1):
            y0, y1 = years[i], years[i+1]
            
            # Load data for both years
            data0 = load_year_data(savedir, r, y0)
            data1 = load_year_data(savedir, r, y1)
            
            # Calculate decomposition
            D = calculate_decomposition(data0, data1, param)
            D_list.append(D)
            
        # Create and save decomposition tables
        save_decomposition_tables(D_list, years, savedir2)

def create_table_2(T_param: pd.DataFrame, task_types: List[str], years: List[int]) -> None:
    """
    Create and save Table 2 in LaTeX format.
    
    Args:
        T_param: Parameter estimates DataFrame
        task_types: List of task types
        years: List of years
    """
    output_path = Path('../figures/table_2.tex')
    
    with output_path.open('w') as f:
        # Write table header
        f.write('\\begin{tabular}{lcccccccccccccc} \n')
        f.write('\\hline\\hline \n')
        f.write('&& ' + ' & '.join(map(str, years)) + ' \\\\ \n')
        f.write('\\hline \n& & \\\\ \n')
        
        # Write race neutral betas
        f.write('\\underline{Race Neutral $\\beta_{kt}$\'s}\\\\ \n')
        for t in task_types:
            values = T_param[f'beta_{t}'].values
            f.write(f'$\\beta_{{{t},t}}$ && ' + 
                   ' & '.join([f'{v:.2f}' for v in values]) + ' \\\\ \n')
        
        f.write('\\\\ \n')
        
        # Write racial barriers
        f.write('\\underline{Additional Racial Barriers}\\\\ \n')
        # Combined barriers
        values = T_param['DEG_rout'].values
        f.write('$Routine:(\\eta_{kt}+\\delta_{kt}+\\gamma_{kt})$ && ' + 
               ' & '.join([f'{v:.2f}' for v in values]) + ' \\\\ \n')
        # Gamma only
        values = T_param['gamma_rout'].values
        f.write('$Routine:\\gamma_{kt}$ && ' + 
               ' & '.join([f'{v:.2f}' for v in values]) + ' \\\\ \n')
        
        f.write('\\\\ \n')
        
        # Write A gaps
        for param, label in [('A_gap', 'A^b_t'), ('AH_gap', 'A^b_{Ht}')]:
            values = T_param[param].values
            f.write(f'${label}$ && ' + 
                   ' & '.join([f'{v:.2f}' for v in values]) + ' \\\\ \n')
        
        f.write('\\\\ \n\\hline \n\\end{tabular}\n')
        
        # Write note
        theta = T_param.iloc[0]['theta']
        note = (
            '\\vspace{1pt}\n\\begin{minipage}{16cm} \n'
            '\\footnotesize Note: Table shows model estimates of the change in aggregate task prices, '
            'the $\\beta_{kt}$\'s, as well as the various other race-specific driving forces. '
            f'The model also estimates $\\theta={theta:.2f}$. '
            'Key task-specific racial barriers for \\textit{Contact} and \\textit{Abstract} tasks are '
            'graphically illustrated in Figure \\ref{fig:task_race_barriers}. \n'
            '\\end{minipage}'
        )
        f.write(note)

def create_table_R7(Tsens_FT_white: pd.DataFrame, years: List[int]) -> None:
    """
    Create and save Table R7 in LaTeX format.
    
    Args:
        Tsens_FT_white: Fixed theta sensitivity analysis results
        years: List of years
    """
    output_path = Path('../figures/table_R7.tex')
    
    with output_path.open('w') as f:
        # Write table header
        f.write('\\begin{tabular}{l|c|ccccccc|ccccccc} \n')
        f.write('\\hline\\hline \n')
        f.write('& & \\multicolumn{7}{c}{Task Content, Contact} & '
                '\\multicolumn{7}{c}{Task Content, Abstract} \\\\ \n')
        years_str = ' & '.join(map(str, years))
        f.write(f'& Est. & {years_str} & {years_str} \\\\ \n')
        f.write('\\hline \n')
        
        # Write beta differences from mean
        for year in years:
            beta_cont = Tsens_FT_white[f'beta_cont_{year}']
            beta_abs = Tsens_FT_white[f'beta_abs_{year}']
            f.write(f'$\\beta_{{Contact, {year}}}-\\bar{{\\beta}}_{{Contact}}$ && ' +
                   ' & '.join([f'{v:.2f}' for v in beta_cont]) + ' \\\\ \n')
            f.write(f'$\\beta_{{Abstract, {year}}}-\\bar{{\\beta}}_{{Abstract}}$ && ' +
                   ' & '.join([f'{v:.2f}' for v in beta_abs]) + ' \\\\ \n')
        
        # Write means and task coefficients
        for param in ['beta_cont_avg', 'beta_abs_avg', 'a_cont', 'a_abs']:
            values = Tsens_FT_white.loc[param]
            label = {
                'beta_cont_avg': '\\bar{\\beta}_{Contact}',
                'beta_abs_avg': '\\bar{\\beta}_{Abstract}',
                'a_cont': 'a_{Contact}',
                'a_abs': 'a_{Abstract}'
            }[param]
            f.write(f'${label}$ && ' +
                   ' & '.join([f'{v:.2f}' for v in values]) + ' \\\\ \n')
        
        f.write('\\hline \n\\end{tabular}\n')

def create_table_R8(Tsens_FT_white: pd.DataFrame, years: List[int]) -> None:
    """
    Create and save Table R8 in LaTeX format.
    
    Args:
        Tsens_FT_white: Fixed theta sensitivity analysis results
        years: List of years
    """
    output_path = Path('../figures/table_R8.tex')
    
    with output_path.open('w') as f:
        # Write table header
        f.write('\\begin{tabular}{l|c|ccccccc|ccccccc} \n')
        f.write('\\hline\\hline \n')
        f.write('& & \\multicolumn{7}{c}{Task Premium, Contact} & '
                '\\multicolumn{7}{c}{Task Premium, Abstract} \\\\ \n')
        years_str = ' & '.join(map(str, years))
        f.write(f'& Est. & {years_str} & {years_str} \\\\ \n')
        f.write('\\hline \n')
        
        # Write regression coefficients
        for year in years:
            tp_cont = Tsens_FT_white[f'tp_cont_{year}']
            tp_abs = Tsens_FT_white[f'tp_abs_{year}']
            f.write(f'$\\beta_{{Contact, {year}}}$ && ' +
                   ' & '.join([f'{v:.2f}' for v in tp_cont]) + ' \\\\ \n')
            f.write(f'$\\beta_{{Abstract, {year}}}$ && ' +
                   ' & '.join([f'{v:.2f}' for v in tp_abs]) + ' \\\\ \n')
        
        f.write('\\hline \n\\end{tabular}\n')

def create_table_R9(Tsens_white: pd.DataFrame, years: List[int]) -> None:
    """
    Create and save Table R9 in LaTeX format.
    
    Args:
        Tsens_white: Sensitivity analysis results
        years: List of years
    """
    output_path = Path('../figures/table_R9.tex')
    
    with output_path.open('w') as f:
        # Panel A: Task Content
        f.write('\\begin{tabular}{l|c|ccccccc|ccccccc} \n')
        f.write('\\hline\\hline \n')
        f.write('\\multicolumn{2}{l}{Panel A: }& '
                '\\multicolumn{7}{c}{Task Content, Contact} & '
                '\\multicolumn{7}{c}{Task Content, Abstract} \\\\ \n')
        years_str = ' & '.join(map(str, years))
        f.write(f'& Est. & {years_str} & {years_str} \\\\ \n')
        f.write('\\hline \n')
        
        # Write theta sensitivity for task content
        theta_tc_cont = Tsens_white[['tc_cont_' + str(y) for y in years]]
        theta_tc_abs = Tsens_white[['tc_abs_' + str(y) for y in years]]
        values = pd.concat([theta_tc_cont, theta_tc_abs], axis=1).iloc[0]
        f.write('$\\theta$ && ' + 
               ' & '.join([f'{v:.2f}' for v in values]) + ' \\\\ \n')
        f.write('\\hline \n\\end{tabular}\n')
        
        # Panel B: Task Premium
        f.write('\n\\begin{tabular}{l|c|ccccccc|ccccccc} \n')
        f.write('\\hline\\hline \n')
        f.write('\\multicolumn{2}{l}{Panel B: }& '
                '\\multicolumn{7}{c}{Task Premium, Contact} & '
                '\\multicolumn{7}{c}{Task Premium, Abstract} \\\\ \n')
        f.write(f'& Est. & {years_str} & {years_str} \\\\ \n')
        f.write('\\hline \n')
        
        # Write theta sensitivity for task premium
        theta_tp_cont = Tsens_white[['tp_cont_' + str(y) for y in years]]
        theta_tp_abs = Tsens_white[['tp_abs_' + str(y) for y in years]]
        values = pd.concat([theta_tp_cont, theta_tp_abs], axis=1).iloc[0]
        f.write('$\\theta$ && ' + 
               ' & '.join([f'{v:.2f}' for v in values]) + ' \\\\ \n')
        f.write('\\hline \n\\end{tabular}\n')

def create_table_R10(savedir2: Path, years: List[int], task_types: List[str]) -> None:
    """
    Create and save Table R10 in LaTeX format.
    
    Args:
        savedir2: Directory for saving results
        years: List of years
        task_types: List of task types
    """
    output_path = Path('../figures/table_R10.tex')
    
    with output_path.open('w') as f:
        # Write table header
        f.write('\\begin{tabular}{l|c|cccccc} \n')
        f.write('\\hline\\hline \n')
        f.write('& & \\multicolumn{6}{c}{Gaps in:} \\\\ \n')
        
        moment_cols = ['ln_l_H_gap', 'tc_gap_cont', 'tc_gap_abs', 
                      'tp_gap_cont', 'tp_gap_abs', 'wage_gap']
        
        # Process each year
        for year in years:
            # Load sensitivity data
            sens_path = savedir2 / 'sensitivity' / f'sens_{year}.csv'
            Tsens_black = pd.read_csv(sens_path, index_col=0)
            
            # Write year header
            f.write('\\hline \n')
            f.write(f'{year} & Est. & Home Share & TC, Cont. & TC, Abst. & '
                   'TP, Cont. & TP, Abst. & Agg. Wage \\\\ \n')
            
            # Write parameter sensitivities
            params_to_write = [
                ('AH_gap', '$A^b_H$'),
                ('DE_cont', '$\\delta+\\eta$, Contact'),
                ('DE_abs', '$\\delta+\\eta$, Abstract'),
                ('gamma_cont', '$\\gamma$, Contact'),
                ('gamma_abs', '$\\gamma$, Abstract'),
                ('A_gap', '$A^b$')
            ]
            
            for param, label in params_to_write:
                if param in Tsens_black.index:
                    values = Tsens_black.loc[param, moment_cols]
                    value_str = ' & '.join([f'{v:.2f}' for v in values])
                    f.write(f'{label} & {Tsens_black.loc[param, "estimate"]:.2f} & {value_str} \\\\ \n')
        
        f.write('\\hline \n\\end{tabular}\n')

def calculate_additional_moments(savedir2: Path, T_param: pd.DataFrame, T_mom: pd.DataFrame,
                              param: List[int], years: List[int], 
                              task_types: List[str]) -> None:
    """
    Calculate and save additional untargeted moments.
    
    Args:
        savedir2: Directory for saving results
        T_param: Parameter estimates DataFrame
        T_mom: Model moments DataFrame
        param: Model parameters [I, J, K, Y]
        years: List of years
        task_types: List of task types
    """
    I, J, K = param[:3]
    
    # Initialize additional moments table
    othernames = ['wage_gap_cond', 'rank_gap_p50', 'rank_gap_p90']
    othernames.extend([f'phi_gap_{t}' for t in task_types])
    othernames.append('e_Ls')
    
    T_other = pd.DataFrame(np.nan, index=years, columns=othernames)
    
    for year in years:
        # Extract occupation-level data
        w_wj = T_mom.loc[year, [f'ln_w_{j}' for j in range(1, J)]].values
        l_wj = np.exp(T_mom.loc[year, [f'ln_emp_{j}' for j in range(1, J)]].values)
        w_bj = w_wj.copy()
        l_bj = l_wj * T_param.loc[year, 'LFb']
        
        # Calculate conditional wage gap
        X = np.column_stack([
            np.ones(2*(J-1)),
            np.concatenate([np.zeros(J-1), np.ones(J-1)])
        ])
        W = np.concatenate([l_wj, l_bj])
        y = np.concatenate([w_wj, w_bj])
        
        valid = ~np.isnan(y)
        XtW = (X[valid].T * W[valid])
        beta = np.linalg.solve(XtW @ X[valid], XtW @ y[valid])
        T_other.loc[year, 'wage_gap_cond'] = beta[1]
        
        # Calculate rank gaps
        ranks = calculate_wage_ranks(w_wj, w_bj, l_wj, l_bj)
        T_other.loc[year, ['rank_gap_p50', 'rank_gap_p90']] = ranks
        
        # Calculate skill gaps
        phi_gaps = calculate_skill_gaps(param, T_param, year, task_types)
        for t, gap in zip(task_types, phi_gaps):
            T_other.loc[year, f'phi_gap_{t}'] = gap
        
        # Calculate labor supply elasticity
        T_other.loc[year, 'e_Ls'] = calculate_labor_elasticity(
            T_param.loc[year, 'psi'],
            np.exp(T_mom.loc[year, 'ln_l_H'])
        )
    
    # Save additional moments
    T_other.to_csv(savedir2 / '2_other_moments.csv')

def calculate_wage_ranks(w_wj: np.ndarray, w_bj: np.ndarray, 
                        l_wj: np.ndarray, l_bj: np.ndarray) -> np.ndarray:
    """
    Calculate wage rank gaps at p50 and p90.
    
    Args:
        w_wj: White wages by occupation
        w_bj: Black wages by occupation
        l_wj: White employment shares
        l_bj: Black employment shares
        
    Returns:
        Array containing p50 and p90 rank gaps
    """
    # Calculate weights
    w_wj = w_wj[~np.isnan(w_wj)]
    w_bj = w_bj[~np.isnan(w_bj)]
    l_wj = l_wj[~np.isnan(w_wj)] / np.sum(l_wj[~np.isnan(w_wj)])
    l_bj = l_bj[~np.isnan(w_bj)] / np.sum(l_bj[~np.isnan(w_bj)])
    
    # Calculate black percentiles
    p_vals = [50, 90]
    w_p_b = [np.percentile(w_bj, p) for p in p_vals]
    
    # Calculate ranks in white distribution
    gaps = []
    for p, w_p in zip(p_vals, w_p_b):
        rank_w = 100 * np.sum(l_wj[w_wj <= w_p])
        gaps.append(rank_w - p)
    
    return np.array(gaps)

def calculate_skill_gaps(param: List[int], T_param: pd.DataFrame, 
                        year: int, task_types: List[str]) -> np.ndarray:
    """
    Calculate skill gaps by task type.
    
    Args:
        param: Model parameters [I, J, K, Y]
        T_param: Parameter estimates DataFrame
        year: Year for calculation
        task_types: List of task types
        
    Returns:
        Array of skill gaps for each task type
    """
    theta = T_param.loc[year, 'theta']
    
    # Calculate distributional parameters
    phi_mean = gamma(1 - 1/theta)
    phi_std = np.sqrt(gamma(1 - 2/theta) - phi_mean**2)
    
    # Calculate standardized gaps
    gaps = []
    for t in task_types:
        beta = T_param.loc[year, f'beta_{t}']
        gamma_val = T_param.loc[year, f'gamma_{t}']
        gap = (beta + gamma_val - phi_mean) / phi_std
        gaps.append(gap)
    
    return np.array(gaps)

def calculate_labor_elasticity(psi: float, l_H: float) -> float:
    """
    Calculate labor supply elasticity.
    
    Args:
        psi: Labor supply parameter
        l_H: Home sector share
        
    Returns:
        Labor supply elasticity
    """
    return psi * (1 - l_H) * (1 - (1 - l_H))

def save_cumulative_changes(D_list: List[np.ndarray], years: List[int],
                          decomp_dir: Path, base_year: int) -> None:
    """
    Save tables of cumulative changes.
    
    Args:
        D_list: List of decomposition arrays
        years: List of years
        decomp_dir: Directory for saving results
        base_year: Base year for cumulative changes
    """
    # Determine components to save
    components = ['wage_gap', 'TC_gap_cont', 'TC_gap_abs', 'TC_gap_rout',
                 'TP_gap_cont', 'TP_gap_abs', 'TP_gap_rout']
    
    # Create cumulative changes tables
    for comp in components:
        # Extract relevant decomposition
        D_comp = [D[components.index(comp)] for D in D_list]
        
        # Calculate cumulative changes
        if base_year == 1960:
            cum_changes = np.cumsum(D_comp, axis=0)
            suffix = 'cum60'
        else:  # base_year == 1980
            base_idx = years.index(1980)
            cum_changes = np.cumsum(D_comp, axis=0)
            cum_changes = cum_changes - cum_changes[base_idx]
            suffix = 'cum80'
        
        # Create and save table
        df = pd.DataFrame(cum_changes, 
                         index=years[1:],
                         columns=['total', 'beta_all', 'race_barriers', 'DE_all', 'gamma_all'])
        df.to_csv(decomp_dir / f'dc_{comp}_{suffix}.csv')
