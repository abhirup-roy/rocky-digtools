"""Rotating-drum specialisation of the shared sweep and OFAT engines."""

from ..doe import ModelRuntime, ParamSchema
from ..doe import launch_ofat as _launch_ofat, launch_sweep as _launch_sweep
from .drum_meshgen import create_meshes, drum_dimensions

GDRUM_SCHEMA = ParamSchema(
    extra_experiment_fields=("aspect_ratio", "rpm"),
    extra_ranges={"aspect_ratio": (0, None), "rpm": (0, None)},
)


def _settings_extra(context):
    drum_dimensions(context["L_BOX"], context["ASPECT_RATIO"])
    from .simulation import validate_rpm

    validate_rpm(context["RPM"])
    return {
        "aspect_ratio": context["ASPECT_RATIO"],
        "rpm": context["RPM"],
        "t_fill": 1.0,
        "t_settle": 0.5,
        "n_revolutions": 5.0,
        "fill_fraction": 0.5,
        "packing_fraction": 0.6,
    }


def _mesh_kwargs(cases):
    from .simulation import validate_rpm

    for case in cases:
        drum_dimensions(case["box_len"], case["aspect_ratio"])
        validate_rpm(case["rpm"])
    return {"aspect_ratios": sorted({case["aspect_ratio"] for case in cases})}


GDRUM_RUNTIME = ModelRuntime(
    case_runner_module="rocky_digtools.models.gdrum.case_runner",
    template_package="rocky_digtools.models.gdrum",
    template_name="template_gdrum.py",
    script_filename="script_gdrum.py",
    extra_key_map={"aspect_ratio": "ASPECT_RATIO", "rpm": "RPM"},
    settings_extra=_settings_extra,
    create_meshes=create_meshes,
    mesh_kwargs=_mesh_kwargs,
)


def launch_sweep(
    sweep_name,
    scheduler,
    json_path,
    meshdir="meshes",
    template_dir=None,
    autolaunch=True,
    target="GPU",
    backend=None,
):
    """Generate a full-factorial drum sweep; JSON values are lists."""
    from . import BACKEND

    return _launch_sweep(
        sweep_name=sweep_name,
        scheduler=scheduler,
        json_path=json_path,
        schema=GDRUM_SCHEMA,
        runtime=GDRUM_RUNTIME,
        meshdir=meshdir,
        template_dir=template_dir,
        autolaunch=autolaunch,
        target=target,
        backend=BACKEND if backend is None else backend,
    )


def launch_ofat(
    sweep_name,
    scheduler,
    ofat_values,
    n_points,
    json_path,
    autolaunch=True,
    target="CPU",
    backend=None,
    template_dir=None,
):
    """Generate a drum OFAT design; base JSON values are scalars."""
    from . import BACKEND

    return _launch_ofat(
        sweep_name=sweep_name,
        scheduler=scheduler,
        ofat_values=ofat_values,
        n_points=n_points,
        json_path=json_path,
        schema=GDRUM_SCHEMA,
        runtime=GDRUM_RUNTIME,
        autolaunch=autolaunch,
        target=target,
        backend=BACKEND if backend is None else backend,
        template_dir=template_dir,
    )
