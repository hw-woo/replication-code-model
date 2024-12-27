# m0_model_run_all.m (from MATLAB to PYTHON)

import os
import time
from pathlib import Path
from multiprocessing import Pool

# Import necessary functions from the depends folder
from depends.estimate_model import estimate_model
from depends.create_tables import create_tables
from depends.decompose_trends import decompose_trends
from depends.save_tables import save_tables

# Settings
workdir = Path(r'C:/Users/kazuatsu/Desktop/ChicagoResearch/TaskBasedDiscrimination/replication/code')
savedir_root = workdir / '../output/model/'
os.chdir(workdir)

# Disable figure visibility
os.environ['MPLBACKEND'] = 'Agg'

def main():
    start_time = time.time()

    # Baseline
    psi_d = 4.5
    print(f"\nRunning baseline with psi_d = {psi_d}\n")
    estimate_model(savedir_root, psi_d)
    create_tables(savedir_root, psi_d)
    decompose_trends(savedir_root, psi_d)
    save_tables(savedir_root, psi_d)

    # Robustness: alternate psi's
    psi_vec = [3.5, 5.5]
    for psi_d in psi_vec:
        print(f"\nRunning robustness check with psi_d = {psi_d}\n")
        estimate_model(savedir_root, psi_d)
        create_tables(savedir_root, psi_d)
        decompose_trends(savedir_root, psi_d)

    # Robustness: alternate theta's
    psi_d = 4.5  # Fix psi at 4.5
    theta_vec = [2.8, 4.5]
    for theta in theta_vec:
        print(f"\nRunning robustness check with psi_d = {psi_d}, theta = {theta}\n")
        estimate_model(savedir_root, psi_d, theta=theta)
        create_tables(savedir_root, psi_d, theta=theta)
        decompose_trends(savedir_root, psi_d, theta=theta)

    # Print overall runtime
    total_time = time.time() - start_time
    print(f"\nTOTAL RUN-TIME: {total_time / 60:.0f} min\n")

if __name__ == "__main__":
    main()
