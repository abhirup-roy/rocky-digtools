"""One-Factor-at-a-Time (OFAT) experiment setup and execution.

Generates OFAT experiment designs from a JSON base configuration, creates case
directories with simulation scripts and SLURM submission files, and optionally
launches the jobs. This engine is model-agnostic: callers supply a
:class:`~rocky_digtools.models.doe.schema.ParamSchema` and
:class:`~rocky_digtools.models.doe.runtime.ModelRuntime` describing the
model-specific parameters and case-preparation hooks.
"""

import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ...utils import RockyScheduler
from . import shapes_module_path
from .runtime import ModelRuntime, load_template, prepare_case
from .schema import ParamSchema, iter_ofat

# Fixed script-context keys for the common parameter set (mirrors
# script_context_from_params, but built from a flat OFAT experiment dict
# rather than a SimParams instance).
_COMMON_KEY_MAP = {
    "radius": "RADIUS_P",
    "density": "DENSITY_P",
    "poisson": "POISSON_P",
    "youngmod": "YOUNGMOD_P",
    "fric_dyn_pp": "DYNAMIC_FRICTION_PP",
    "fric_stat_pp": "STATIC_FRICTION_PP",
    "cor_pp": "COR_PP",
    "fric_dyn_pw": "DYNAMIC_FRICTION_PW",
    "fric_stat_pw": "STATIC_FRICTION_PW",
    "cor_pw": "COR_PW",
    "box_len": "L_BOX",
    "normal": "NORMAL_MODEL",
    "tangential": "TANG_MODEL",
    "rolling": "ROLLING_MODEL",
    "adhesion": "ADH_MODEL",
    "shape": "SHAPE",
}


def launch_ofat(
    sweep_name: str,
    scheduler: RockyScheduler,
    ofat_values: dict[str, list | str],
    n_points: int,
    json_path: str,
    schema: ParamSchema,
    runtime: ModelRuntime,
    autolaunch: bool = True,
    target: str = "CPU",
    backend: str = "pyrocky",
    template_dir: Optional[str] = None,
) -> None:
    """Launch a One-Factor-at-a-Time (OFAT) experiment block.

    Generates the necessary case directories, input scripts, and SLURM
    submission scripts for a series of OFAT experiments based on the provided
    configuration.

    Example::

        ofat_values = {
            "parameters": ["cor_pp", "fric_dyn_pp"],
            "test_range": [(0.1, 0.5), (0.2, 0.8)],
            "hold_values": ["m", "l"],
        }
        launch_ofat(
            sweep_name="ofat_sweep",
            scheduler=scheduler,
            ofat_values=ofat_values,
            n_points=5,
            json_path="config.json",
            schema=schema,
            runtime=runtime,
        )

    Args:
        sweep_name: Name of the OFAT experiment block, used for directory
            naming.
        scheduler: :class:`~rocky_digtools.utils.RockyScheduler` describing
            the SLURM configuration for each case.
        ofat_values: Dictionary specifying the OFAT design. Must contain
            ``"parameters"`` (list of names), ``"test_range"`` (list of
            ``(min, max)`` tuples), and ``"hold_values"`` (list of ``"h"``,
            ``"l"``, or ``"m"`` strategies).
        n_points: Number of test points to generate for each factor.
        json_path: Path to the JSON configuration file with base parameters.
        schema: The model's :class:`~rocky_digtools.models.doe.schema.ParamSchema`.
        runtime: The model's :class:`~rocky_digtools.models.doe.runtime.ModelRuntime`.
        autolaunch: If ``True``, automatically submit the SLURM jobs after
            setup. Defaults to ``True``.
        target: Compute target — ``"CPU"`` or ``"GPU"``. Defaults to
            ``"CPU"``.
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.
            Defaults to ``"pyrocky"``.
        template_dir: Optional path to a directory with a custom Jinja2
            template.

    Raises:
        ValueError: If an unsupported backend or target is specified.
        FileNotFoundError: If ``template_dir`` does not exist.
        NotImplementedError: If ``target="MULTI_GPU"`` is requested.
    """
    if backend not in ["rocky_prepost", "pyrocky"]:
        raise ValueError("backend must be 'rocky_prepost' or 'pyrocky'")
    elif backend == "pyrocky":
        scheduler.run_command = (
            f"{sys.executable} -m {runtime.case_runner_module} settings.json"
        )

    target = target.upper()
    if target not in ["CPU", "GPU", "MULTI_GPU"]:
        raise ValueError("Select from 'CPU', 'GPU', 'MULTI_GPU'")
    elif target == "MULTI_GPU":
        raise NotImplementedError("Multi GPU use not validated yet")

    target_quoted = f'"{target}"'

    rocky_template = load_template(runtime, template_dir) if backend == "rocky_prepost" else None

    experiments_df, base_dict = iter_ofat(
        json_path=str(json_path),
        ofat_values=ofat_values,
        n_points=n_points,
        schema=schema,
    )

    total_cases = len(experiments_df)
    vars_list = experiments_df.columns.tolist()

    sweep_path = Path(sweep_name)
    sweep_path.mkdir(parents=True, exist_ok=True)

    case_dirs = []
    for i in range(total_cases):
        case_dirs.append(sweep_path / f"case_{i}")

    if "box_len" in experiments_df.columns:
        unique_sizes = set(experiments_df["box_len"])
    elif "box_len" in base_dict.keys():
        unique_sizes = [base_dict["box_len"]]
    else:
        raise ValueError(
            "No box length parameter found in experiments or base dictionary. "
            "Debugging required"
        )

    size_to_mesh_dir = {}
    for size in tqdm(unique_sizes, desc="Generating meshes", unit="mesh"):
        shared_mesh_dir = sweep_path / f"meshes_{size}"
        shared_mesh_dir.mkdir(parents=True, exist_ok=True)
        runtime.create_meshes(size, meshsize=0.01, out_dir=str(shared_mesh_dir))
        size_to_mesh_dir[size] = shared_mesh_dir

    for i, row in tqdm(
        experiments_df.iterrows(),
        total=total_cases,
        desc="Preparing OFAT cases",
        unit="case",
    ):
        case_dir = case_dirs[i]
        case_dir.mkdir(parents=True, exist_ok=True)

        exp_dict = {var: row[var] for var in vars_list}
        exp_dict.update(base_dict)

        script_contxt = {
            script_key: exp_dict[field_name]
            for field_name, script_key in _COMMON_KEY_MAP.items()
        }
        for field_name, script_key in runtime.extra_key_map.items():
            script_contxt[script_key] = exp_dict[field_name]

        script_contxt.update(
            {
                "VERT_AR": exp_dict.get("vert_ar"),
                "HORIZ_AR": exp_dict.get("horiz_ar"),
                "N_CORNERS": int(exp_dict.get("n_corners", 8)),
                "SQ_DEGREE": exp_dict.get("sq_degree"),
                "PARTICLE_PATH": exp_dict.get("particle_path"),
                "SMOOTHNESS": exp_dict.get("smoothness", 0.5),
                "XPU": target_quoted,
                "MESH_DIR": str(size_to_mesh_dir[exp_dict["box_len"]]),
                "SHAPES_MODULE_PATH": shapes_module_path,
            }
        )

        if exp_dict["rolling"] != "none":
            script_contxt["ROLLING_FRICTION"] = exp_dict["fric_rolling_pp"]
        else:
            script_contxt["ROLLING_FRICTION"] = 0

        prepare_case(
            case_dir,
            script_contxt,
            backend,
            runtime,
            rocky_template,
            mesh_path=size_to_mesh_dir[exp_dict["box_len"]],
        )

        scheduler.generate(case_dir)

    tqdm.write(f"\nOFAT experiments:\n{experiments_df}")

    if autolaunch:
        scheduler.launch_all([str(d) for d in case_dirs])
