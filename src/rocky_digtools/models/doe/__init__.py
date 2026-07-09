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

from .ofat import launch_ofat
from .runtime import (
    ModelRuntime,
    load_template,
    prepare_case,
    script_context_from_params,
)
from .schema import (
    ParamSchema,
    ShapeConfig,
    SimParams,
    get_unique_box_lens,
    iter_ofat,
    iter_params,
)
from .sweep import launch_sweep

__all__ = [
    "shapes_module_path",
    "ShapeConfig",
    "SimParams",
    "ParamSchema",
    "iter_params",
    "iter_ofat",
    "get_unique_box_lens",
    "ModelRuntime",
    "prepare_case",
    "script_context_from_params",
    "load_template",
    "launch_sweep",
    "launch_ofat",
]
