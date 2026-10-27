"""Rotating-drum specialisation of the shared sweep and OFAT engines."""

from dataclasses import replace
import json
import math
from pathlib import Path

from ..doe import ModelRuntime, ParamSchema
from ..doe import launch_ofat as _launch_ofat, launch_sweep as _launch_sweep
from .drum_meshgen import create_meshes, drum_dimensions

GDRUM_SCHEMA = ParamSchema(
    extra_experiment_fields=("aspect_ratio", "rpm"),
    extra_ranges={"aspect_ratio": (0, None), "rpm": (0, None)},
)
GDRUM_LENGTH_SCHEMA = ParamSchema(
    extra_experiment_fields=("length", "rpm"),
    extra_ranges={"length": (0, None), "rpm": (0, None)},
)


def _schema_for_settings(experiment_settings):
    geometry_fields = tuple(
        name
        for name in ("drum_radius", "length", "aspect_ratio")
        if experiment_settings.get(name) is not None
    )
    has_box = experiment_settings.get("box_len") is not None
    if len(geometry_fields) != (1 if has_box else 2):
        raise ValueError(
            "Provide exactly two of drum_radius, length and aspect_ratio without box_len; "
            "or box_len and exactly one geometry value"
        )
    if geometry_fields == ("aspect_ratio",) and has_box:
        return GDRUM_SCHEMA
    if geometry_fields == ("length",) and has_box:
        return GDRUM_LENGTH_SCHEMA
    return ParamSchema(
        extra_experiment_fields=geometry_fields + ("rpm",),
        extra_ranges={name: (0, None) for name in geometry_fields + ("rpm",)},
        defaults={} if has_box else {"box_len": None},
        normalise_values=None if has_box else _normalise_values,
    )


def _normalise_values(values):
    radius, length = drum_dimensions(
        values.get("box_len"),
        values.get("aspect_ratio"),
        length=values.get("length"),
        radius=values.get("drum_radius"),
    )
    return {**values, "box_len": (math.pi * radius**2 * length) ** (1 / 3)}


def _design_config(json_path):
    schema = _schema_for_settings(
        json.loads(Path(json_path).read_text())["experiment_settings"]
    )
    runtime = replace(
        GDRUM_RUNTIME,
        extra_key_map={name: name.upper() for name in schema.extra_experiment_fields},
    )
    return schema, runtime


def _settings_extra(context):
    geometry = {
        name: context[name.upper()]
        for name in ("aspect_ratio", "length", "drum_radius")
        if name.upper() in context
    }
    if "drum_radius" in geometry:
        geometry["radius"] = geometry.pop("drum_radius")
    box_len = context["L_BOX"] if len(geometry) == 1 else None
    drum_dimensions(box_len, **geometry)
    from .simulation import validate_rpm

    validate_rpm(context["RPM"])
    return {
        **geometry,
        "particle_box_len": box_len,
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
        validate_rpm(case["rpm"])
    fields = [
        name
        for name in ("drum_radius", "length", "aspect_ratio")
        if cases[0].get(name) is not None
    ]
    if len(fields) == 2 or fields == ["drum_radius"]:
        geometries = []
        for case in cases:
            geometry = {
                "radius" if name == "drum_radius" else name: case[name]
                for name in fields
            }
            geometry["box_len"] = case["box_len"] if len(fields) == 1 else None
            drum_dimensions(**geometry)
            if geometry not in geometries:
                geometries.append(geometry)
        return {"geometries": geometries}
    for case in cases:
        drum_dimensions(
            case["box_len"], case.get("aspect_ratio"), length=case.get("length")
        )
    geometry_key = "length" if cases[0].get("length") is not None else "aspect_ratio"
    mesh_key = "lengths" if geometry_key == "length" else "aspect_ratios"
    return {mesh_key: sorted({case[geometry_key] for case in cases})}


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

    schema, runtime = _design_config(json_path)
    return _launch_sweep(
        sweep_name=sweep_name,
        scheduler=scheduler,
        json_path=json_path,
        schema=schema,
        runtime=runtime,
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

    schema, runtime = _design_config(json_path)
    if schema.normalise_values and "box_len" in ofat_values.get("parameters", []):
        raise ValueError(
            "box_len is derived from the drum geometry and cannot be varied in this mode"
        )
    return _launch_ofat(
        sweep_name=sweep_name,
        scheduler=scheduler,
        ofat_values=ofat_values,
        n_points=n_points,
        json_path=json_path,
        schema=schema,
        runtime=runtime,
        autolaunch=autolaunch,
        target=target,
        backend=BACKEND if backend is None else backend,
        template_dir=template_dir,
    )
