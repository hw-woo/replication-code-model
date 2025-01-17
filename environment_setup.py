# environment_setup.py
import os
from pathlib import Path

def setup_octave_path():
    """Setup Octave path in environment variables"""
    octave_base = Path(r"C:/Program Files/GNU Octave/Octave-9.3.0")
    octave_bin = octave_base / "mingw64" / "bin"
    octave_cli = octave_bin / "octave-cli.exe" # Octave-9.3.0 (CLI).exe or octave-cli.exe
    
    # Convert to string and use forward slashes
    octave_bin_str = str(octave_bin).replace('\\', '/')
    octave_cli_str = str(octave_cli).replace('\\', '/')
    
    # Set environment variables
    os.environ['PATH'] = str(octave_bin) + os.pathsep + os.environ['PATH']
    os.environ['OCTAVE_EXECUTABLE'] = str(octave_cli)
    
    # For debugging
    print("Octave paths set:")
    print(f"  PATH addition: {octave_bin_str}")
    print(f"  OCTAVE_EXECUTABLE: {octave_cli_str}")