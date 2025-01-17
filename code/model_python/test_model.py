# test_model.py
import pandas as pd
from pathlib import Path
import numpy as np
import sys
import os

# Import model utilities
from utils import unitinterval, prc_rank, prctile_weighted

# Import Matlab-related modules conditionally
def get_octave():
    from oct2py import Oct2Py
    return Oct2Py()

def get_matlab_bridge():
    from matlab_bridge import estimate_model_matlab
    return estimate_model_matlab

def test_data_loading():
    print("\n=== Testing Data Loading ===")
    try:
        # Set base directory
        BASE_DIR = Path(r"D:/Research/RA_EmmanuelYimfor/AER_taskdiscrimination/replication/")
        
        # Load moments_all.csv
        moments_path = BASE_DIR / 'output/acs/moments_all.csv'
        moments_all = pd.read_csv(moments_path)
        print("✓ moments_all.csv loaded successfully")
        print(f"  - Shape: {moments_all.shape}")
        print(f"  - Years: {moments_all['year_graph'].values}")
        
        # Load moments_ztasks_broad.csv
        tasks_path = BASE_DIR / 'output/acs/moments_ztasks_broad.csv'
        tasks = pd.read_csv(tasks_path)
        print("✓ moments_ztasks_broad.csv loaded successfully")
        print(f"  - Shape: {tasks.shape}")
        print(f"  - Task types: {[col for col in tasks.columns if 'ztask_' in col]}")
        
        return moments_all, tasks
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None

def test_utils(moments_all):
    print("\n=== Testing Utility Functions ===")
    try:
        # Test unitinterval
        print("\nTesting unitinterval function:")
        x = np.array([[1, 2, 3], [4, 5, 6]])
        normalized = unitinterval(x)
        print("✓ unitinterval function working")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {normalized.shape}")
        print(f"  - Output values:\n{normalized}")

        # Test prc_rank
        print("\nTesting prc_rank function:")
        value = np.array([[50]])  # value to find rank for
        X = np.arange(100)  # reference distribution
        rank = prc_rank(value, X)
        print("✓ prc_rank function working")
        print(f"  - Rank of {value[0][0]}: {rank}")

        # Test prctile_weighted
        print("\nTesting prctile_weighted function:")
        X = np.random.normal(size=1000)
        p = np.array([25, 50, 75])  # percentiles to compute
        percentiles = prctile_weighted(X, p)
        print("✓ prctile_weighted function working")
        print(f"  - Percentiles {p}: {percentiles}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in utils: {e}")
        print(f"  Details: {str(e)}")
        return False

def test_estimation():
    print("\n=== Testing Model Estimation ===")
    try:
        # Test settings
        psi_d = 4.5
        BASE_DIR = Path(r"D:/Research/RA_EmmanuelYimfor/AER_taskdiscrimination/replication")
        savedir_root = BASE_DIR / 'output/model/test'
        test_params = {
            'n_points': 2,
            'seed': 0
        }

        print("Testing with minimal settings...")
        
        # Load and display data for verification
        data_path = BASE_DIR / 'output/acs/moments_all.csv'
        data_mom = pd.read_csv(data_path)
        print("\nData loaded:")
        print("Columns:", data_mom.columns.tolist())
        print("year_graph values:", data_mom['year_graph'].values)
        
        # Set index
        data_mom = data_mom.set_index('year_graph')
        print("\nIndex after setting:", data_mom.index.values)
        
        # Get Matlab bridge after environment is set up
        estimate_model_matlab = get_matlab_bridge()
        
        # Run Matlab-based estimation
        estimate_model_matlab(savedir_root=savedir_root, psi_d=psi_d, 
                            test_mode=True, test_params=test_params)
        print("✓ Basic estimation test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in estimation: {e}")
        import traceback
        print(traceback.format_exc())
        return False
