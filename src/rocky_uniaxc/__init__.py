"""
pyuniaxc: Tools for setting up multiscale uniaxial compression

This package provides tools for setting up and analysing
multiscale uniaxial compression.

Author: Abhirup Roy
"""

__version__ = "0.1"
__author__ = "Abhirup Roy"
__all__ = [
    "launch_sweep",
    "launch_ofat",
    "analyse",
    "externals",
    "pyrocky",
]


from .doe.sweep import launch_sweep
from .doe.ofat import launch_ofat
# from .doe.sobol import launch_sobol

# from .doe import med
from . import sweep_analysis as analyse
from . import externals
from . import pyrocky

import pathlib as _pathlib

# Auto-detect Rocky executable path at import time
ROCKY_EXE_PATH = pyrocky.find_rocky_exe()
HEADLESS = True
BACKEND = "pyrocky"


def set_rocky_exe_path(path: str) -> None:
    """Set the path to the Rocky executable for the rocky_uniaxc package.
    This allows users to specify the path to the Rocky executable if it is not in a standard location or not found automatically. This allows the pyrocky API to use the specified executable for running simulations.
        Args:
            path (str): The file path to the Rocky executable.
        Raises:
            FileNotFoundError: If the specified path does not point to a valid file.
        Usage:
            set_rocky_exe_path("/path/to/rocky/executable")
        Note:
            Ensure that the provided path is correct and that the Rocky executable is accessible at that location.
    """

    if not _pathlib.Path(path).is_file():
        raise FileNotFoundError(f"Specified Rocky executable not found at: {path}")

    global ROCKY_EXE_PATH
    ROCKY_EXE_PATH = path


def set_headless_mode(headless: bool) -> None:
    """Set the headless mode for Rocky simulations in the rocky_uniaxc package.
    This function allows users to specify whether Rocky simulations should run in headless mode (without a graphical user interface) or with a GUI. This setting will affect how the pyrocky API launches Rocky for simulations.
        Args:
            headless (bool): If True, Rocky will run in headless mode. If False, Rocky will launch with its GUI.
        Usage:
            set_headless_mode(True)  # Run Rocky in headless mode
            set_headless_mode(False) # Run Rocky with GUI
        Note:
            Running in headless mode is more suitable for batch processing or running on servers without display capabilities, while running with the GUI can be useful for interactive use and debugging.
    """
    global HEADLESS
    HEADLESS = headless


def set_backend(backend: str) -> None:
    """Set the backend for Rocky simulations in the rocky_uniaxc package.
    This function allows users to specify which backend to use for running Rocky simulations. The backend determines how the package interacts with the Rocky executable. Accepted values are "pyrocky" or "rocky_prepost".
        Args:
            backend (str): The backend to use: "pyrocky" or "rocky_prepost".
        Usage:
            set_backend("pyrocky")  # Use the pyrocky backend for simulations
        Note:
            Ensure that the specified backend is supported and properly configured in the rocky_uniaxc package. Using an unsupported backend will result in an error when attempting to run simulations.
            All non-simulation utilities use pyrocky regardless of this setting
    """
    if backend not in ["pyrocky", "rocky_prepost"]:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends are 'pyrocky' and 'rocky_prepost'."
        )
    global BACKEND
    BACKEND = backend
