"""Uniaxial compression model.

Provides utilities for configuring, launching, and post-processing
uniaxial compression experiments using Ansys Rocky DEM. Supports full
parameter sweeps, one-factor-at-a-time (OFAT) designs, and pyrocky-based
simulation workflows.
"""

__all__ = [
    "launch_sweep",
    "launch_ofat",
    "analyse",
    "Settings",
    "UniaxialCompressionSimulation",
    "create_meshes",
    "set_backend",
]

BACKEND = "pyrocky"

from .doe import launch_sweep, launch_ofat
from . import sweep_analysis as analyse
from .simulation import Settings, UniaxialCompressionSimulation
from .compr_meshgen import create_meshes


def set_backend(backend: str) -> None:
    """Set the backend used to generate uniaxial compression case scripts.

    Args:
        backend: The backend to use. Must be ``"pyrocky"`` or
            ``"rocky_prepost"``.

    Raises:
        ValueError: If an unsupported backend is specified.

    Example:
        >>> set_backend("pyrocky")
    """
    if backend not in ["pyrocky", "rocky_prepost"]:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends are 'pyrocky' and 'rocky_prepost'."
        )
    global BACKEND
    BACKEND = backend
