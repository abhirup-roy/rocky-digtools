"""Case-preparation plumbing shared by the generic sweep/OFAT engines.

Provides directory management, script-context construction, and backend
dispatch (Jinja2 ``rocky_prepost`` templates vs. generated ``pyrocky``
launcher scripts). The parts that differ per model — where the case runner
lives, which Jinja2 template to render, and how to translate a script
context into a ``settings.json`` payload — are supplied by a
:class:`ModelRuntime` instance.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import jinja2

from .schema import SimParams

logger = logging.getLogger(__name__)


@dataclass
class ModelRuntime:
    """Model-specific hooks needed to prepare and launch a DOE case.

    Attributes:
        case_runner_module: Dotted module path invoked as
            ``python -m <case_runner_module> settings.json`` by the
            generated ``pyrocky``-backend launcher script.
        template_package: Dotted package name passed to
            :class:`jinja2.PackageLoader` to locate the model's
            ``rocky_prepost`` template (e.g.
            ``"rocky_digtools.models.uniax"``).
        template_name: Filename of the model's Jinja2 template within
            ``template_package``'s ``templates/`` directory.
        script_filename: Filename of the generated per-case launcher script
            (e.g. ``"script_uniax.py"``).
        extra_key_map: Maps :class:`~.schema.SimParams` ``extra`` field names
            to the script-context keys used in Jinja2 templates (e.g.
            ``{"p_compress": "P_COMPRESS"}``).
        settings_extra: Given the script context dict, returns the
            model-specific fields to merge into the ``settings.json``
            payload for the ``pyrocky`` backend (e.g. compression timings).
        create_meshes: The model's mesh-generation function, called as
            ``create_meshes(size, meshsize=..., out_dir=...)``.
    """

    case_runner_module: str
    template_package: str
    template_name: str
    script_filename: str
    extra_key_map: dict[str, str]
    settings_extra: Callable[[dict], dict]
    create_meshes: Callable[..., None]


@contextmanager
def case_directory(
    sweep_name: str | Path, case_idx: int, meshdir: str = "meshes"
) -> Iterator[Path]:
    """Context manager for creating and managing a case directory.

    Creates the following directory structure::

        sweep_name/case_<case_idx>/
            plots/
            <meshdir>/

    Args:
        sweep_name: Name or path of the sweep directory.
        case_idx: Index of the case.
        meshdir: Name of the mesh subdirectory.

    Yields:
        pathlib.Path: Path to the created case directory.
    """
    sweep_path = Path(sweep_name)
    case_path = sweep_path / f"case_{case_idx}"

    (case_path / "plots").mkdir(parents=True, exist_ok=True)
    (case_path / meshdir).mkdir(parents=True, exist_ok=True)

    yield case_path


def script_context_from_params(
    params: SimParams,
    target: str,
    meshdir: str = "meshes",
    extra_key_map: Optional[dict[str, str]] = None,
) -> dict:
    """Build a script context dictionary from a :class:`SimParams` instance.

    Args:
        params: Simulation parameters.
        target: Processor target (``"CPU"``, ``"GPU"``, etc.).
        meshdir: Name of the mesh subdirectory.
        extra_key_map: Maps ``params.extra`` field names to script-context
            keys (see :attr:`ModelRuntime.extra_key_map`).

    Returns:
        Dictionary of template variables for script rendering.
    """
    rolling_fric = params.fric_rolling_pp if params.rolling != "none" else 0

    ctx = {
        "RADIUS_P": params.radius,
        "DENSITY_P": params.density,
        "POISSON_P": params.poisson,
        "YOUNGMOD_P": params.youngmod,
        "DYNAMIC_FRICTION_PP": params.fric_dyn_pp,
        "STATIC_FRICTION_PP": params.fric_stat_pp,
        "COR_PP": params.cor_pp,
        "TANG_STIFF_RATIO_PP": params.tang_stiff_ratio_pp,
        "SURF_EN_PP": params.surf_en_pp,
        "DYNAMIC_FRICTION_PW": params.fric_dyn_pw,
        "STATIC_FRICTION_PW": params.fric_stat_pw,
        "COR_PW": params.cor_pw,
        "TANG_STIFF_RATIO_PW": params.tang_stiff_ratio_pw,
        "SURF_EN_PW": params.surf_en_pw,
        "L_BOX": params.box_len,
        "NORMAL_MODEL": params.normal,
        "TANG_MODEL": params.tangential,
        "ROLLING_MODEL": params.rolling,
        "ADH_MODEL": params.adhesion,
        "SHAPE": params.shape.name,
        "VERT_AR": params.shape.vert_ar,
        "HORIZ_AR": params.shape.horiz_ar,
        "N_CORNERS": int(params.shape.n_corners),
        "SQ_DEGREE": params.shape.sq_degree,
        "PARTICLE_PATH": params.shape.particle_path,
        "SMOOTHNESS": params.shape.smoothness,
        "XPU": target,
        "MESH_DIR": meshdir,
        "ROLLING_FRICTION": rolling_fric,
    }

    if extra_key_map:
        for field_name, key in extra_key_map.items():
            ctx[key] = params.extra[field_name]

    return ctx


def render_pyrocky_script(
    case_dir: str | Path,
    script_context: dict,
    runtime: ModelRuntime,
    meshdir: str = "meshes",
    mesh_path: Optional[str | Path] = None,
) -> None:
    """Render a pyrocky simulation case.

    Dumps simulation settings to a ``settings.json`` file and creates a small
    launcher script that invokes the model's case runner.

    Args:
        case_dir: Path to the case directory.
        script_context: Dictionary containing script template variables.
        runtime: The model's :class:`ModelRuntime`.
        meshdir: Name of the mesh subdirectory (used only when
            ``mesh_path`` is ``None``).
        mesh_path: Absolute path to the mesh directory. When provided this
            takes precedence over ``meshdir``, allowing callers to point
            cases at a shared pre-generated mesh directory.
    """
    case_dir = Path(case_dir)
    if mesh_path is None:
        mesh_path = os.path.abspath(case_dir / meshdir)
    else:
        mesh_path = str(os.path.abspath(mesh_path))

    settings_dict = {
        "particle_box_len": script_context["L_BOX"],
        "p_radius": script_context["RADIUS_P"],
        "p_density": script_context["DENSITY_P"],
        "p_youngmod": script_context["YOUNGMOD_P"],
        "p_poisson": script_context["POISSON_P"],
        "fric_dyn_pp": script_context["DYNAMIC_FRICTION_PP"],
        "fric_stat_pp": script_context["STATIC_FRICTION_PP"],
        "cor_pp": script_context["COR_PP"],
        "tang_stiff_ratio_pp": script_context["TANG_STIFF_RATIO_PP"],
        "surf_en_pp": script_context["SURF_EN_PP"],
        "fric_dyn_pw": script_context["DYNAMIC_FRICTION_PW"],
        "fric_stat_pw": script_context["STATIC_FRICTION_PW"],
        "cor_pw": script_context["COR_PW"],
        "tang_stiff_ratio_pw": script_context["TANG_STIFF_RATIO_PW"],
        "surf_en_pw": script_context["SURF_EN_PW"],
        "normal_force_model": script_context["NORMAL_MODEL"].strip('"'),
        "tangential_force_model": script_context["TANG_MODEL"].strip('"'),
        "adhesion_model": script_context["ADH_MODEL"].strip('"'),
        "rolling_fric": script_context.get("ROLLING_FRICTION", 0.0),
        "rolling_model": script_context["ROLLING_MODEL"].strip('"'),
        "processor": script_context["XPU"].strip('"'),
        "mesh_dir": mesh_path,
        "shape_name": script_context["SHAPE"].strip('"'),
        "vert_ar": script_context["VERT_AR"],
        "horiz_ar": script_context["HORIZ_AR"],
        "n_corners": script_context["N_CORNERS"],
        "sq_degree": script_context["SQ_DEGREE"],
        "particle_path": script_context["PARTICLE_PATH"],
        "smoothness": script_context["SMOOTHNESS"],
    }
    settings_dict.update(runtime.settings_extra(script_context))

    settings_path = case_dir / "settings.json"
    with open(settings_path, "w") as f:
        json.dump(settings_dict, f, indent=4)

    script_content = f"""import sys
import subprocess
from pathlib import Path

# Run the single runner module
subprocess.run(
    [sys.executable, "-m", "{runtime.case_runner_module}", "settings.json"],
    check=True,
)
"""
    (case_dir / runtime.script_filename).write_text(script_content)


def prepare_case(
    case_dir: Path,
    script_context: dict,
    backend: str,
    runtime: ModelRuntime,
    rocky_template: Optional[jinja2.Template] = None,
    mesh_path: Optional[str | Path] = None,
) -> None:
    """Write a simulation script to the case directory.

    Args:
        case_dir: Path to the case directory.
        script_context: Script context dictionary for template rendering.
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.
        runtime: The model's :class:`ModelRuntime`.
        rocky_template: Jinja2 template instance. Required when
            ``backend="rocky_prepost"``.
        mesh_path: Absolute path to the mesh directory (pyrocky backend
            only). When provided, the generated ``settings.json`` points
            at this directory instead of deriving a per-case path.

    Raises:
        ValueError: If ``backend="rocky_prepost"`` and no template is
            provided, or if the backend string is unrecognised.
    """
    script_path = case_dir / runtime.script_filename

    if backend == "rocky_prepost":
        if rocky_template is None:
            raise ValueError("rocky_template required for rocky_prepost backend")
        rendered = rocky_template.render(script_context)
        script_path.write_text(rendered)
    elif backend == "pyrocky":
        render_pyrocky_script(case_dir, script_context, runtime, mesh_path=mesh_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    logger.debug("Script written to %s", script_path)


def load_template(runtime: ModelRuntime, template_dir: Optional[str | os.PathLike]):
    """Load the model's Jinja2 ``rocky_prepost`` template.

    Args:
        runtime: The model's :class:`ModelRuntime`.
        template_dir: Optional path to a directory with a custom template
            named ``runtime.template_name``. Defaults to the model's
            built-in templates directory.

    Returns:
        The loaded :class:`jinja2.Template`.

    Raises:
        FileNotFoundError: If ``template_dir`` does not exist.
    """
    if template_dir:
        template_dir = Path(template_dir).resolve()
        if not template_dir.exists():
            raise FileNotFoundError(f"Directory {template_dir} does not exist.")
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)), undefined=jinja2.StrictUndefined
        )
    else:
        env = jinja2.Environment(
            loader=jinja2.PackageLoader(runtime.template_package, "templates"),
            undefined=jinja2.StrictUndefined,
        )
    return env.get_template(runtime.template_name)
