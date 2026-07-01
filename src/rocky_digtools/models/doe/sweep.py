"""Full-factorial parameter sweep generation and execution.

Reads a JSON configuration specifying parameter ranges, computes all
combinations via the Cartesian product, generates case directories, mesh
files, simulation scripts, and SLURM submission scripts, and optionally
launches the jobs. This engine is model-agnostic: callers supply a
:class:`~rocky_digtools.models.doe.schema.ParamSchema` and
:class:`~rocky_digtools.models.doe.runtime.ModelRuntime` describing the
model-specific parameters and case-preparation hooks.
"""

import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from . import shapes_module_path
from .runtime import ModelRuntime, case_directory, prepare_case, script_context_from_params, load_template
from .schema import ParamSchema, get_unique_box_lens, iter_params
from ...utils import RockyScheduler


def launch_sweep(
    sweep_name: str,
    scheduler: RockyScheduler,
    json_path: str,
    schema: ParamSchema,
    runtime: ModelRuntime,
    meshdir: str = "meshes",
    template_dir: Optional[str] = None,
    autolaunch: bool = True,
    target: str = "GPU",
    backend: str = "pyrocky",
):
    """Generate and launch a full-factorial parameter sweep.

    Reads parameter ranges from a JSON configuration, computes all
    combinations, creates case directories with simulation scripts and SLURM
    submission files, and optionally submits the jobs.

    Args:
        sweep_name: Title of the sweep, used as the root directory name.
        scheduler: :class:`~rocky_digtools.utils.RockyScheduler` describing
            the SLURM configuration for each case.
        json_path: Path to the JSON configuration file defining parameter
            ranges.
        schema: The model's :class:`~rocky_digtools.models.doe.schema.ParamSchema`.
        runtime: The model's :class:`~rocky_digtools.models.doe.runtime.ModelRuntime`.
        meshdir: Name of the mesh subdirectory inside each case. Defaults to
            ``"meshes"``.
        template_dir: Optional path to a directory containing a custom
            Jinja2 template. Defaults to the model's built-in templates.
        autolaunch: Whether to automatically submit SLURM jobs after setup.
            Defaults to ``True``.
        target: Compute target — ``"CPU"`` or ``"GPU"``. Defaults to
            ``"GPU"``.
        backend: Simulation backend — ``"rocky_prepost"`` or ``"pyrocky"``.
            Defaults to ``"pyrocky"``.

    Raises:
        ValueError: If an unsupported backend or target is specified.
        FileNotFoundError: If ``template_dir`` does not exist.
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

    rocky_template = load_template(runtime, template_dir)

    all_params = list(iter_params(json_path, schema))
    total_cases = len(all_params)

    sweep_path = Path(sweep_name)
    sweep_path.mkdir(exist_ok=True)

    case_dirs = []
    for i in range(total_cases):
        case_dirs.append(sweep_path / f"case_{i}")

    unique_sizes = get_unique_box_lens(all_params)

    size_to_mesh_dir = {}
    for size in tqdm(unique_sizes, desc="Generating meshes", unit="mesh"):
        shared_mesh_dir = sweep_path / f"meshes_{size}"
        shared_mesh_dir.mkdir(parents=True, exist_ok=True)
        runtime.create_meshes(size, meshsize=0.01, out_dir=str(shared_mesh_dir))
        size_to_mesh_dir[size] = shared_mesh_dir

    for i, params in tqdm(
        enumerate(all_params),
        total=total_cases,
        desc="Preparing cases",
        unit="case",
    ):
        case_dir = case_dirs[i]

        with case_directory(sweep_path, i, meshdir):
            pass

        script_contxt = script_context_from_params(
            params, target_quoted, meshdir, extra_key_map=runtime.extra_key_map
        )
        script_contxt["SHAPES_MODULE_PATH"] = shapes_module_path
        prepare_case(
            case_dir,
            script_contxt,
            backend,
            runtime,
            rocky_template,
            mesh_path=size_to_mesh_dir[params.box_len],
        )

        scheduler.generate(case_dir)

    tqdm.write(f"\nAll cases:\n{all_params}")

    if autolaunch:
        scheduler.launch_all([str(d) for d in case_dirs])
