# utils.py

import numpy as np
from scipy import linalg
from scipy.special import gamma
from typing import Tuple, List, Optional, Union
import pandas as pd

def unitinterval(x: np.ndarray, dim: int = 1) -> np.ndarray:
    """
    Normalize values to [0,1] interval along specified dimension.
    
    Args:
        x: Input array
        dim: Dimension along which to normalize (default=1)
    
    Returns:
        Normalized array with values in [0,1]
    """
    x_min = np.min(x, axis=dim, keepdims=True)
    x_max = np.max(x, axis=dim, keepdims=True)
    return (x - x_min) / (x_max - x_min)

def prc_rank(value: np.ndarray, X: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate percentile rank.
    
    Args:
        value: Values to evaluate (1 x k array)
        X: Reference data
        W: Optional weights
    
    Returns:
        Percentile ranks
    """
    if value.shape[0] != 1:
        raise ValueError('value must be a vector with size 1 x k')
        
    if W is None:
        W = np.ones_like(X)
        
    valid = ~np.isnan(X)
    X = X[valid]
    W = W[valid]
    
    n = np.sum(W)
    p_lower = np.sum((X <= value) * W, axis=0) / n * 100
    p_upper = (n - np.sum((X >= value) * W, axis=0)) / n * 100
    
    return (p_lower + p_upper) / 2

def prctile_weighted(X: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate weighted percentiles.
    
    Args:
        X: Input array
        p: Percentiles to compute
        w: Optional weights
        
    Returns:
        Weighted percentile values
    """
    x_p = np.full(p.shape, np.nan)
    
    if w is None:
        w = np.ones_like(X)
        
    valid = ~np.isnan(X)
    X = X[valid]
    w = w[valid]
    
    idx = np.argsort(X)
    X_i = X[idx]
    w_i = w[idx]
    W_i = np.cumsum(w_i) / np.sum(w_i)
    
    for j in range(len(p)):
        P = p[j] / 100
        i = np.where(W_i > P)[0]
        if len(i) > 0:
            i = i[0]
            if P == 1:
                i = len(W_i) - 1
            if i > 0 and W_i[i-1] == P:
                x_p[j] = (X_i[i-1] + X_i[i]) / 2
            else:
                x_p[j] = X_i[i]
    
    return x_p

def repackage_D(D: np.ndarray, param: List[int], dtheta: bool = False, dpsi: bool = False) -> np.ndarray:
    """
    Repackage yearly Jacobians.
    
    Args:
        D: Input derivatives array
        param: Model parameters [I, J, K, Y]
        dtheta: Calculate theta derivative
        dpsi: Calculate psi derivative
        
    Returns:
        Repackaged derivatives
    """
    J = param[1]  # number of occupations
    K = param[2]  # number of skill types 
    M = D.shape[0]  # number of moments
    N = D.shape[1]  # number of parameters
    Y = D.shape[2]  # number of periods
    
    Ktheta = param[4] if len(param) > 4 else 1
    
    if N == J + K + Ktheta + 1:  # momtype == 'level'
        delim = np.cumsum([1, J-1, K, dtheta*Ktheta, dpsi])
        
        D_dAH = D[:, :delim[0], :]
        D_dA_j = D[:, delim[0]:delim[1], :]
        D_dbeta = D[:, delim[1]:delim[2], :]
        D_dtheta = D[:, delim[2]:delim[3], :]
        D_dpsi = D[:, delim[3]:delim[4], :]
        
        # Process occupation fixed effects
        D_dA_j_fe = D_dA_j[:, 1:, :]
        D_dA_j_fe = np.transpose(D_dA_j_fe, (0, 2, 1)).reshape(M*Y, -1)
        
        # Process time fixed effects
        D_dA_t_fe_list = []
        for y in range(Y):
            D_sum = np.sum([D_dAH[:,:,y], D_dA_j[:,:,y]], axis=1)
            D_dA_t_fe_list.append(D_sum)
        D_dA_t_fe = linalg.block_diag(*D_dA_t_fe_list)
        
        # Process remaining terms
        D_rest_list = []
        for y in range(Y):
            D_rest = np.hstack([D_dAH[:,:,y], D_dbeta[:,:,y]])
            D_rest_list.append(D_rest)
        D_rest = linalg.block_diag(*D_rest_list)
        
        D_dtheta_reshaped = np.transpose(D_dtheta, (0, 2, 1)).reshape(-1, Ktheta)
        
        return np.hstack([
            D_dA_j_fe,
            D_dA_t_fe, 
            D_rest,
            D_dtheta_reshaped,
            D_dpsi.flatten()[:, np.newaxis]
        ])
        
    elif N == 2 + K + K or N == 1:  # momtype == 'gap' or derivative w.r.t. Lb
        D_list = [D[:,:,y] for y in range(Y)]
        return linalg.block_diag(*D_list)
    
    return None

def unpack_x(x: Union[np.ndarray, List], param: List[int], options: dict = None) -> Tuple:
    """
    Unpack parameter vector x.
    
    Args:
        x: Parameter vector
        param: Model parameters [I, J, K, Y]
        options: Optional settings for unpacking
        
    Returns:
        Tuple of unpacked parameters
    """
    if options is None:
        options = {'notheta': False, 'nopsi': False}
        
    # Convert x to numpy array if it's a list
    x = np.array(x)
    
    J = param[1]  # occupations
    K = param[2]  # skill types
    Y = param[3]  # periods
    Ktheta = param[4] if len(param) > 4 else 1
    
    x = x.copy()  # avoid modifying input
    
    # Extract psi and theta if needed
    if not options.get('nopsi'):
        psi = x[-1]
        x = x[:-1]
    else:
        psi = None
        
    if not options.get('notheta'):
        theta = x[-Ktheta:]
        x = x[:-Ktheta]
    else:
        theta = None
    
    # Parse remaining parameters
    delim = np.cumsum([J-2, Y, Y*(1+K), Y*K*2])
    
    A_j_fe = x[:delim[0]]
    A_t_fe = x[delim[0]:delim[1]]
    x_rest = x[delim[1]:delim[2]]
    x_rest = x_rest.reshape(-1, 1+K).T
    
    A_Ht = x_rest[0]
    beta_kt = x_rest[1:K+1]
    
    if len(x) == delim[2]:
        delta_eta_kt = np.zeros((Y, K))
        delta_eta_np_kt = np.zeros((Y, K))
    else:
        x_de = x[delim[2]:delim[3]].reshape(-1, K).T
        delta_eta_kt = x_de[:K]
        delta_eta_np_kt = x_de[K:]
        
    # Construct A_jt
    A_jt = np.column_stack([
        A_Ht + A_t_fe,
        np.tile([0] + list(A_j_fe), (Y, 1)) + A_t_fe[:, np.newaxis]
    ])
    
    # Construct final parameters column
    if theta is not None and psi is not None:
        # Create a Y x 2 array with repeated values
        final_params = np.zeros((Y, 2))
        final_params[:, 0] = theta
        final_params[:, 1] = psi
    else:
        final_params = np.zeros((Y, 0))  # Empty array with correct dimensions
    
    # Construct full parameter matrix X
    X = np.column_stack([
        A_jt,
        beta_kt.T,
        delta_eta_kt,
        delta_eta_np_kt,
        final_params
    ])
    
    return X, A_jt, beta_kt.T, delta_eta_kt, delta_eta_np_kt, theta, psi, A_j_fe, A_t_fe, A_Ht

def unpack_xb(xb: np.ndarray, param: List[int], A_j: Optional[np.ndarray] = None, 
              beta_k: Optional[np.ndarray] = None) -> Tuple:
    """
    Unpack black parameter vector xb.
    
    Args:
        xb: Parameter vector for blacks
        param: Model parameters
        A_j: Optional A_j parameters
        beta_k: Optional beta_k parameters
        
    Returns:
        Tuple of unpacked parameters
    """
    K = param[2]  # skill types
    
    delim_b = np.cumsum([1, 1, K, K])
    
    if xb.shape[0] == delim_b[-1]:
        xb = xb.T
        
    A_gap = xb[:, :delim_b[0]]
    AH_gap = xb[:, delim_b[0]:delim_b[1]]
    DE_k = xb[:, delim_b[1]:delim_b[2]]
    gamma_k = xb[:, delim_b[2]:delim_b[3]]
    
    if A_j is not None:
        A_j_b = np.column_stack([
            A_j[:, 0] + AH_gap,
            A_j[:, 1:]
        ]) + A_gap
        
        if beta_k is not None:
            Xb = np.column_stack([A_j_b, beta_k, DE_k, gamma_k])
        else:
            Xb = None
    else:
        A_j_b = None
        Xb = None
        
    return Xb, A_j_b, A_gap, AH_gap, DE_k, gamma_k
