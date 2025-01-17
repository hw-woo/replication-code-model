# main.py

import os
import time
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from estimation import estimate_model
from tables import create_tables, decompose_trends, save_tables

def main(savedir_root='../output/model_python/', psi_d=4.5, 
         test_mode=False, test_params=None):
    """
    Main entry point for model estimation and analysis.
    
    Args:
        savedir_root: Output directory path
        psi_d: Model parameter
        test_mode: If True, run with reduced data/iterations
        test_params: Dictionary of test parameters
    """
    start_time = time.time()
    
    # Settings
    workdir = Path().absolute()
    os.chdir(workdir)
    
    # Create output directory if it doesn't exist
    savedir_root = Path(savedir_root)
    savedir_root.mkdir(parents=True, exist_ok=True)
    
    try:
        n_cores = mp.cpu_count()
        print(f"\nInitializing parallel pool with {n_cores} workers")
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            if test_mode and test_params:
                estimate_model(savedir_root, psi_d, options=test_params)
            else:
                # Regular execution
                estimate_model(savedir_root, psi_d)
                create_tables(savedir_root, psi_d)
                decompose_trends(savedir_root, psi_d)
                save_tables(savedir_root, psi_d)
    
    finally:
        total_time = time.time() - start_time
        print(f'\nTOTAL RUN-TIME: {total_time/60:.0f} min')

if __name__ == '__main__':
    main()