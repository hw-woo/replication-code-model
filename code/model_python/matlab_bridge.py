# matlab_bridge.py
from oct2py import Oct2Py
import numpy as np
import pandas as pd
import os
from pathlib import Path

def setup_octave_environment(base_path):
    """Setup Octave environment with correct paths"""
    try:
        # Initialize Octave
        oc = Oct2Py(logger=None)  # Disable logging for cleaner output
        
        # Add all necessary Matlab paths
        paths = [
            os.path.join(base_path, 'model_matlab'),
            os.path.join(base_path, 'model_matlab/depends'),
            os.path.join(base_path, 'model_matlab/depends/mom_fun'),
            os.path.join(base_path, 'model_matlab/depends/utilities')
        ]
        
        # Convert paths to use forward slashes and add them
        for path in paths:
            path_str = str(Path(path)).replace('\\', '/')
            if os.path.exists(path):
                oc.addpath(path_str)
                print(f"Added Matlab path: {path_str}")
            else:
                print(f"Warning: Path does not exist: {path_str}")
                
        # Change current directory to model_matlab path
        model_matlab_path = str(Path(base_path) / 'model_matlab').replace('\\', '/')
        oc.eval(f"cd('{model_matlab_path}')")
        print(f"Changed Octave working directory to: {model_matlab_path}")
        
        return oc
        
    except Exception as e:
        print(f"Error setting up Octave environment: {str(e)}")
        print("Current environment variables:")
        print(f"PATH: {os.environ['PATH']}")
        print(f"OCTAVE_EXECUTABLE: {os.environ.get('OCTAVE_EXECUTABLE', 'Not set')}")
        raise

class MatlabBridge:
    def __init__(self):
        # Initialize Octave
        self.base_path = r"D:/Research/RA_EmmanuelYimfor/AER_taskdiscrimination/replication/code"
        self.oc = setup_octave_environment(self.base_path)
        # Run initial Matlab setup
        print("Running initial Matlab setup...")
        self.oc.eval('m0_model_run_all')
        
    def run_estimation(self, savedir_root: str, psi_d: float, test_mode: bool = False, 
                      test_params: dict = None):
        """Run the Matlab estimation code"""
        try:
            # Convert paths to absolute paths
            savedir_root = str(Path(savedir_root).absolute())
            
            # Create command string
            if test_mode and test_params is not None:
                # Convert test_params to Matlab struct
                params_str = "{" + ",".join(f"'{k}',{v}" for k,v in test_params.items()) + "}"
                cmd = f"estimate_model('{savedir_root}', {psi_d}, 0, {params_str})"
            else:
                cmd = f"estimate_model('{savedir_root}', {psi_d})"
            
            print(f"Executing Matlab command: {cmd}")
            
            # Run the Matlab code
            self.oc.eval(cmd)
            
            print("Matlab estimation completed successfully")
            return True
            
        except Exception as e:
            print(f"Error running Matlab estimation: {str(e)}")
            return False
        
    def close(self):
        """Clean up Octave instance"""
        try:
            self.oc.exit()
        except:
            pass


def estimate_model_matlab(savedir_root='../output/model_python/', psi_d=4.5, 
                        test_mode=False, test_params=None):
    """
    Main entry point for Matlab-based estimation.
    
    Args:
        savedir_root: Output directory path
        psi_d: Model parameter
        test_mode: If True, run with reduced data/iterations
        test_params: Dictionary of test parameters
    """
    bridge = None
    try:
        # Create output directory if it doesn't exist
        savedir_root = Path(savedir_root)
        savedir_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize Matlab bridge
        print("Initializing Matlab bridge...")
        bridge = MatlabBridge()
        
        # Run estimation
        success = bridge.run_estimation(savedir_root, psi_d, test_mode, test_params)
        
        if success:
            print("Model estimation completed successfully")
        else:
            print("Model estimation failed")
            
    except Exception as e:
        print(f"Error in estimate_model_matlab: {str(e)}")
        raise
    
    finally:
        if bridge:
            bridge.close()
