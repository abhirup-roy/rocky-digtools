"""Design of Experiments (DOE) sub-package for the uniaxial compression model.

Provides shared constants for sweep and OFAT experiment workflows. Job
submission is handled by :class:`rocky_digtools.utils.RockyScheduler`.
"""

import pathlib

shapes_module_path = str(
    (pathlib.Path(__file__).parent.parent.parent.parent / "particles_shapes.py").resolve()
)
