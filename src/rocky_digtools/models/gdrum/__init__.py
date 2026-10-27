"""Rotating granular drum with the same volume as a cube of side box_len."""

from .doe import launch_ofat, launch_sweep
from .drum_meshgen import create_meshes, drum_dimensions
from .simulation import GranularDrumSimulation, Settings

BACKEND = "pyrocky"

__all__ = [
    "Settings",
    "GranularDrumSimulation",
    "create_meshes",
    "drum_dimensions",
    "launch_sweep",
    "launch_ofat",
    "set_backend",
]


def set_backend(backend):
    """Select pyrocky or rocky_prepost for subsequent DOE launches."""
    if backend not in ("pyrocky", "rocky_prepost"):
        raise ValueError("backend must be 'pyrocky' or 'rocky_prepost'")
    global BACKEND
    BACKEND = backend
