"""Shear-cell model.

Provides utilities for configuring and launching shear-cell experiments
using Ansys Rocky DEM. Supports full parameter sweeps and
one-factor-at-a-time (OFAT) designs via the shared DOE engine.
"""

__all__ = [
    "launch_sweep",
    "launch_ofat",
    "launch_calibration",
    "create_meshes",
    "set_backend",
]

BACKEND = "pyrocky"

from .doe import launch_sweep, launch_ofat
from .shcell_meshgen import create_meshes


def launch_calibration(*args, **kwargs):
    """Launch ACCES calibration for shear-cell yield-locus matching."""
    from .calibration import launch_calibration as _launch_calibration

    return _launch_calibration(*args, **kwargs)


def set_backend(backend: str) -> None:
    """Set the backend used to generate shear-cell case scripts.

    Args:
        backend: The backend to use. Must be ``"pyrocky"`` or
            ``"rocky_prepost"``.

    Raises:
        ValueError: If an unsupported backend is specified.

    Example:
        >>> set_backend("rocky_prepost")
    """
    if backend not in ["pyrocky", "rocky_prepost"]:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends are 'pyrocky' and 'rocky_prepost'."
        )
    global BACKEND
    BACKEND = backend
