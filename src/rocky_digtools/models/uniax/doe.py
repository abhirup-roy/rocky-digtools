"""Uniaxial-compression specialisation of the generic DOE engine.

Declares the compression-specific parameter schema (adds ``p_compress`` on
top of the common particle/interaction/contact-model fields) and the
:class:`~rocky_digtools.models.doe.runtime.ModelRuntime` (case-runner
module, Jinja2 template, compression timings, mesh generation) needed by
the shared sweep/OFAT engines in :mod:`rocky_digtools.models.doe` to
generate uniaxial compression cases.
"""

from typing import Optional

from ..doe import ModelRuntime, ParamSchema
from ..doe import launch_ofat as _generic_launch_ofat
from ..doe import launch_sweep as _generic_launch_sweep
from .compr_meshgen import create_meshes

UNIAX_SCHEMA = ParamSchema(
    extra_experiment_fields=("p_compress",),
    extra_ranges={"p_compress": (0, None)},
)


def _settings_extra(script_context: dict) -> dict:
    """Compression timings and pressure for the pyrocky ``settings.json``."""
    return {
        "t_fill": 1.0,
        "t_settle": 0.5,
        "t_compress": 2.0,
        "p_compress": script_context["P_COMPRESS"],
    }


UNIAX_RUNTIME = ModelRuntime(
    case_runner_module="rocky_digtools.models.uniax.case_runner",
    template_package="rocky_digtools.models.uniax",
    template_name="template_uniax.py",
    script_filename="script_uniax.py",
    extra_key_map={"p_compress": "P_COMPRESS"},
    settings_extra=_settings_extra,
    create_meshes=create_meshes,
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
    """Generate and launch a full-factorial uniaxial compression sweep.

    Thin wrapper around :func:`rocky_digtools.models.doe.launch_sweep` bound
    to the uniaxial compression :data:`UNIAX_SCHEMA` and :data:`UNIAX_RUNTIME`.
    See that function for full documentation of the parameters.

    Args:
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.
            Defaults to the uniax package-level :data:`~.BACKEND` setting.
    """
    if backend is None:
        from . import BACKEND

        backend = BACKEND

    _generic_launch_sweep(
        sweep_name=sweep_name,
        scheduler=scheduler,
        json_path=json_path,
        schema=UNIAX_SCHEMA,
        runtime=UNIAX_RUNTIME,
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
    """Launch a One-Factor-at-a-Time uniaxial compression experiment block.

    Thin wrapper around :func:`rocky_digtools.models.doe.launch_ofat` bound
    to the uniaxial compression :data:`UNIAX_SCHEMA` and :data:`UNIAX_RUNTIME`.
    See that function for full documentation of the parameters.

    Args:
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.
            Defaults to the uniax package-level :data:`~.BACKEND` setting.
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
        schema=UNIAX_SCHEMA,
        runtime=UNIAX_RUNTIME,
        autolaunch=autolaunch,
        target=target,
        backend=backend,
        template_dir=template_dir,
    )
