"""Tools for setting up and running multiscale DEM simulations in Ansys Rocky.

This package provides general case-setup and Rocky API utilities (mesh-free
job scheduling, particle shape wrappers, VTK/STL export, and pyrocky session
management) shared across DEM test models. Model-specific workflows (e.g.
uniaxial compression, shear cell) live under :mod:`rocky_digtools.models`.

Author:
    Abhirup Roy
"""

__version__ = "0.1"
__author__ = "Abhirup Roy"
__all__ = [
    "externals",
    "pyrocky",
    "models",
    "utils",
    "RockyScheduler",
]


HEADLESS = True
ROCKY_EXE_PATH = None

import pathlib as _pathlib
from . import utils
from .utils import RockyScheduler
from . import externals
from . import pyrocky
from . import models

# Auto-detect Rocky executable path at import time
ROCKY_EXE_PATH = pyrocky.find_rocky_exe()


def set_rocky_exe_path(path: str) -> None:
    """Set the path to the Rocky executable for the rocky_digtools package.

    Allows users to specify the path to the Rocky executable if it is not in a
    standard location or not found automatically. The pyrocky API will use the
    specified executable for running simulations.

    Args:
        path: The file path to the Rocky executable.

    Raises:
        FileNotFoundError: If the specified path does not point to a valid file.

    Example:
        >>> set_rocky_exe_path("/path/to/rocky/executable")
    """

    if not _pathlib.Path(path).is_file():
        raise FileNotFoundError(f"Specified Rocky executable not found at: {path}")

    global ROCKY_EXE_PATH
    ROCKY_EXE_PATH = path


def set_headless_mode(headless: bool) -> None:
    """Set the headless mode for Rocky simulations.

    Controls whether Rocky simulations run in headless mode (without a GUI) or
    with the graphical interface. This setting affects how the pyrocky API
    launches Rocky.

    Args:
        headless: If ``True``, Rocky runs in headless mode. If ``False``,
            Rocky launches with its GUI.

    Example:
        >>> set_headless_mode(True)   # batch processing / servers
        >>> set_headless_mode(False)  # interactive use / debugging
    """
    global HEADLESS
    HEADLESS = headless
