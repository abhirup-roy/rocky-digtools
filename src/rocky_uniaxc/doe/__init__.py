"""Design of Experiments (DOE) sub-package for rocky_uniaxc.

Provides shared constants for sweep and OFAT experiment workflows. Job
submission is handled by :class:`rocky_uniaxc.schedulers.RockyScheduler`.
"""

import pathlib

shapes_module_path = str(
    (pathlib.Path(__file__).parent.parent / "particles_shapes.py").resolve()
)
