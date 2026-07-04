"""Design of Experiments (DOE) engine shared across rocky_digtools models.

Provides a generic parameter schema (:mod:`.schema`), case-preparation
plumbing (:mod:`.runtime`), and full-factorial / OFAT sweep engines
(:mod:`.sweep`, :mod:`.ofat`). Each model (uniaxial compression, shear
cell, ...) supplies a :class:`~.schema.ParamSchema` describing its
parameter extensions and a :class:`~.runtime.ModelRuntime` describing its
case-runner, template, and mesh-generation hooks, then calls
:func:`~.sweep.launch_sweep` / :func:`~.ofat.launch_ofat` with them.
"""

import pathlib

shapes_module_path = str(
    (pathlib.Path(__file__).parent.parent.parent / "particles_shapes.py").resolve()
)

from .schema import (
    ShapeConfig,
    SimParams,
    ParamSchema,
    iter_params,
    iter_ofat,
    get_unique_box_lens,
)
from .runtime import (
    ModelRuntime,
    case_directory,
    prepare_case,
    script_context_from_params,
    load_template,
)
from .sweep import launch_sweep
from .ofat import launch_ofat

__all__ = [
    "shapes_module_path",
    "ShapeConfig",
    "SimParams",
    "ParamSchema",
    "iter_params",
    "iter_ofat",
    "get_unique_box_lens",
    "ModelRuntime",
    "case_directory",
    "prepare_case",
    "script_context_from_params",
    "load_template",
    "launch_sweep",
    "launch_ofat",
]
