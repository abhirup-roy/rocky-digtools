"""Shear-cell specialisation of the generic DOE engine.

Declares the shear-specific parameter schema (the shear-protocol fields added
on top of the common particle/interaction/contact-model fields) and the
:class:`~rocky_digtools.models.doe.runtime.ModelRuntime` (case-runner
module, Jinja2 template, shear timings, mesh generation) needed by the
shared sweep/OFAT engines in :mod:`rocky_digtools.models.doe` to generate
shear-cell cases.
"""

from typing import Optional

from ..doe import ModelRuntime, ParamSchema
from ..doe import launch_ofat as _generic_launch_ofat
from ..doe import launch_sweep as _generic_launch_sweep
from .shcell_meshgen import create_meshes

SHEARCELL_SCHEMA = ParamSchema(
    extra_experiment_fields=(
        "t_settle",
        "t_compression",
        "sigma_pre",
        "n_shear_points",
        "n_procs",
        "neighbour_search",
        "t_shear",
        "shear_vel",
    ),
    extra_ranges={
        "t_settle": (0, None),
        "t_compression": (0, None),
        "sigma_pre": (0, None),
        "n_shear_points": (1, None),
        "n_procs": (1, None),
        "t_shear": (0, None),
        "shear_vel": (0, None),
    },
)


def _settings_extra(script_context: dict) -> dict:
    """Shear-protocol timings and config for the pyrocky ``settings.json``."""
    return {
        "t_settle": script_context["T_SETTLE"],
        "t_compression": script_context["T_COMPRESSION"],
        "sigma_pre": script_context["SIGMA_PRE"],
        "n_shear_points": script_context["N_SHEAR_POINTS"],
        "n_procs": script_context["NPROCS"],
        "neighbour_search": script_context["NEIGHBOUR_SEARCH"],
        "t_shear": script_context["T_SHEAR"],
        "shear_vel": script_context["SHEAR_VEL"],
    }


def _mesh_kwargs(cases: list[dict]) -> dict:
    """Use the greatest configured shear travel for a shared box mesh."""
    longest = max(cases, key=lambda case: case["t_shear"] * case["shear_vel"])
    return {"t_shear": longest["t_shear"], "shear_vel": longest["shear_vel"]}


SHEARCELL_RUNTIME = ModelRuntime(
    case_runner_module="rocky_digtools.models.shearcell.case_runner",
    template_package="rocky_digtools.models.shearcell",
    template_name="template_shear.py",
    script_filename="script_shear.py",
    extra_key_map={
        "t_settle": "T_SETTLE",
        "t_compression": "T_COMPRESSION",
        "sigma_pre": "SIGMA_PRE",
        "n_shear_points": "N_SHEAR_POINTS",
        "n_procs": "NPROCS",
        "neighbour_search": "NEIGHBOUR_SEARCH",
        "t_shear": "T_SHEAR",
        "shear_vel": "SHEAR_VEL",
    },
    settings_extra=_settings_extra,
    create_meshes=create_meshes,
    mesh_kwargs=_mesh_kwargs,
)


def launch_sweep(
    sweep_name: str,
    scheduler,
    json_path: str,
    meshdir: str = "meshes",
    template_dir: Optional[str] = None,
    autolaunch: bool = True,
    target: str = "GPU",
    backend: Optional[str] = None,
) -> None:
    """Generate and launch a full-factorial shear-cell sweep.

    Thin wrapper around :func:`rocky_digtools.models.doe.launch_sweep` bound
    to the shear-cell :data:`SHEARCELL_SCHEMA` and :data:`SHEARCELL_RUNTIME`.
    See that function for full documentation of the parameters.

    Args:
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.
            Defaults to the shearcell package-level :data:`~.BACKEND` setting.
    """
    if backend is None:
        from . import BACKEND

        backend = BACKEND

    _generic_launch_sweep(
        sweep_name=sweep_name,
        scheduler=scheduler,
        json_path=json_path,
        schema=SHEARCELL_SCHEMA,
        runtime=SHEARCELL_RUNTIME,
        meshdir=meshdir,
        template_dir=template_dir,
        autolaunch=autolaunch,
        target=target,
        backend=backend,
    )


def launch_ofat(
    sweep_name: str,
    scheduler,
    ofat_values: dict[str, list | str],
    n_points: int,
    json_path: str,
    autolaunch: bool = True,
    target: str = "CPU",
    backend: Optional[str] = None,
    template_dir: Optional[str] = None,
) -> None:
    """Launch a One-Factor-at-a-Time shear-cell experiment block.

    Thin wrapper around :func:`rocky_digtools.models.doe.launch_ofat` bound
    to the shear-cell :data:`SHEARCELL_SCHEMA` and :data:`SHEARCELL_RUNTIME`.
    See that function for full documentation of the parameters.

    Args:
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.
            Defaults to the shearcell package-level :data:`~.BACKEND` setting.
    """
    if backend is None:
        from . import BACKEND

        backend = BACKEND

    _generic_launch_ofat(
        sweep_name=sweep_name,
        scheduler=scheduler,
        ofat_values=ofat_values,
        n_points=n_points,
        json_path=json_path,
        schema=SHEARCELL_SCHEMA,
        runtime=SHEARCELL_RUNTIME,
        autolaunch=autolaunch,
        target=target,
        backend=backend,
        template_dir=template_dir,
    )
