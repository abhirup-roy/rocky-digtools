"""ACCES calibration helpers for shear-cell simulations."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...utils import cd
from ..doe import ShapeConfig, SimParams, script_context_from_params
from ..doe.runtime import render_pyrocky_script
from .doe import SHEARCELL_RUNTIME, SHEARCELL_SCHEMA
from .simulation import aggregate_results

PENALTY_ERROR = 1.0e30

_FIELD_PATHS = {
    "radius": ("particle_properties", "radius"),
    "density": ("particle_properties", "density"),
    "poisson": ("particle_properties", "poisson"),
    "youngmod": ("particle_properties", "youngmod"),
    "fric_dyn_pp": ("inseractions", "pp", "fric_dyn"),
    "fric_stat_pp": ("inseractions", "pp", "fric_stat"),
    "fric_rolling_pp": ("inseractions", "pp", "fric_rolling"),
    "cor_pp": ("inseractions", "pp", "cor"),
    "fric_dyn_pw": ("inseractions", "pw", "fric_dyn"),
    "fric_stat_pw": ("inseractions", "pw", "fric_stat"),
    "cor_pw": ("inseractions", "pw", "cor"),
    "box_len": ("experim_settings", "box_len"),
    "normal": ("contact_model", "normal"),
    "tangential": ("contact_model", "tangential"),
    "rolling": ("contact_model", "rolling"),
    "adhesion": ("contact_model", "adhesion"),
    **{
        field: ("experim_settings", field)
        for field in SHEARCELL_SCHEMA.extra_experim_fields
    },
}


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    target = data
    for key in path:
        target = target[key]
    return target


def _reject_lists(config: dict[str, Any]) -> None:
    for field, path in _FIELD_PATHS.items():
        value = _get_nested(config, path)
        if isinstance(value, list):
            raise ValueError(
                f"Calibration base config must use scalar values; {field!r} is a list."
            )
    if isinstance(config["shape"], list):
        raise ValueError("Calibration base config must use one shape object, not a list.")


def _with_overrides(
    base_config: dict[str, Any], parameter_values: dict[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    for name, value in parameter_values.items():
        if name not in _FIELD_PATHS:
            raise ValueError(f"Unsupported shearcell calibration parameter: {name}")
        _set_nested(config, _FIELD_PATHS[name], value)
    _reject_lists(config)
    return config


def _sim_params_from_config(config: dict[str, Any]) -> SimParams:
    pp = config["inseractions"]["pp"]
    pw = config["inseractions"]["pw"]
    props = config["particle_properties"]
    exp = config["experim_settings"]
    contact = config["contact_model"]
    return SimParams(
        radius=props["radius"],
        density=props["density"],
        poisson=props["poisson"],
        youngmod=props["youngmod"],
        fric_dyn_pp=pp["fric_dyn"],
        fric_stat_pp=pp["fric_stat"],
        fric_rolling_pp=pp["fric_rolling"],
        cor_pp=pp["cor"],
        fric_dyn_pw=pw["fric_dyn"],
        fric_stat_pw=pw["fric_stat"],
        cor_pw=pw["cor"],
        box_len=exp["box_len"],
        normal=contact["normal"],
        tangential=contact["tangential"],
        rolling=contact["rolling"],
        adhesion=contact["adhesion"],
        shape=ShapeConfig.from_dict(config["shape"]),
        extra={field: exp[field] for field in SHEARCELL_SCHEMA.extra_experim_fields},
    )


def prepare_candidate_settings(
    base_json: str | Path,
    candidate_dir: str | Path,
    parameter_values: dict[str, Any],
    target: str = "CPU",
) -> Path:
    """Write one shear-cell candidate's ``settings.json``."""
    candidate_dir = Path(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    config = _with_overrides(_read_json(base_json), parameter_values)
    params = _sim_params_from_config(config)
    ctx = script_context_from_params(
        params,
        f'"{target.upper()}"',
        extra_key_map=SHEARCELL_RUNTIME.extra_key_map,
    )
    render_pyrocky_script(candidate_dir, ctx, SHEARCELL_RUNTIME)
    return candidate_dir / "settings.json"


def yield_locus_error(
    target_yield_locus: str | Path,
    sigma: np.ndarray,
    tau: np.ndarray,
) -> float:
    """Return mean squared relative error against a target yield locus CSV."""
    target = pd.read_csv(target_yield_locus)
    if set(["sigma", "tau"]) - set(target.columns):
        raise ValueError("Target yield locus CSV must contain 'sigma' and 'tau' columns.")

    target_sigma = target["sigma"].to_numpy(dtype=float)
    target_tau = target["tau"].to_numpy(dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)

    order = np.argsort(sigma)
    sigma = sigma[order]
    tau = tau[order]

    if target_sigma.min() < sigma.min() or target_sigma.max() > sigma.max():
        raise ValueError("Target yield locus sigma range is outside simulated range.")

    sim_tau = np.interp(target_sigma, sigma, tau)
    scale = np.maximum(np.abs(target_tau), 1.0)
    return float(np.mean(((sim_tau - target_tau) / scale) ** 2))


def wait_for_shearcell_metrics(
    candidate_dir: str | Path,
    poll_interval: float = 60,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Poll shear-cell aggregation until all shear points are complete."""
    candidate_dir = Path(candidate_dir)
    started = time.monotonic()
    while True:
        metrics = aggregate_results(candidate_dir)
        if metrics is not None:
            return metrics
        if timeout is not None and time.monotonic() - started >= timeout:
            raise TimeoutError(f"Timed out waiting for shearcell results in {candidate_dir}")
        time.sleep(poll_interval)


def evaluate_candidate(
    parameters,
    access_id: int,
    base_json: str,
    work_dir: str,
    target_yield_locus: str,
    poll_interval: float = 60,
    timeout: float | None = None,
    penalty: float = PENALTY_ERROR,
) -> float:
    """Run one ACCES candidate and return its scalar calibration error."""
    candidate_dir = Path(work_dir) / f"candidate_{int(access_id)}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    try:
        values = parameters["value"].to_dict()
        settings_path = prepare_candidate_settings(base_json, candidate_dir, values)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "rocky_digtools.models.shearcell.case_runner",
                str(settings_path),
            ],
            check=True,
            cwd=str(candidate_dir),
        )
        wait_for_shearcell_metrics(candidate_dir, poll_interval, timeout)

        outputs_dir = candidate_dir / "pyoutputs"
        sigma = np.load(outputs_dir / "sigma.npy")
        tau = np.load(outputs_dir / "shear_stresses.npy")
        return yield_locus_error(target_yield_locus, sigma, tau)
    except Exception as exc:
        (candidate_dir / "calibration_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        return float(penalty)


def _normalise_free_parameters(
    free_parameters: dict[str, dict[str, float]],
) -> tuple[list[str], list[float], list[float], list[float], list[float]]:
    names = list(free_parameters)
    minimums = []
    maximums = []
    values = []
    sigmas = []
    for name in names:
        spec = free_parameters[name]
        if name not in _FIELD_PATHS:
            raise ValueError(f"Unsupported shearcell calibration parameter: {name}")
        if "min" not in spec or "max" not in spec:
            raise ValueError(f"Free parameter {name!r} must define 'min' and 'max'.")
        lo = float(spec["min"])
        hi = float(spec["max"])
        if hi <= lo:
            raise ValueError(f"Free parameter {name!r} must have max > min.")
        minimums.append(lo)
        maximums.append(hi)
        values.append(float(spec.get("value", (lo + hi) / 2)))
        sigmas.append(float(spec.get("sigma", 0.4 * (hi - lo))))
    return names, minimums, maximums, values, sigmas


def _render_access_script(
    base_json: str | Path,
    target_yield_locus: str | Path,
    work_dir: str | Path,
    free_parameters: dict[str, dict[str, float]],
    poll_interval: float,
    timeout: float | None,
) -> str:
    names, minimums, maximums, values, sigmas = _normalise_free_parameters(
        free_parameters
    )
    return f'''"""Generated ACCES shear-cell calibration script."""

# ACCESS PARAMETERS START
import coexist
parameters = coexist.create_parameters(
    variables={names!r},
    minimums={minimums!r},
    maximums={maximums!r},
    values={values!r},
    sigma={sigmas!r},
)
access_id = 0
# ACCESS PARAMETERS END

from rocky_digtools.models.shearcell.calibration import evaluate_candidate

error = evaluate_candidate(
    parameters,
    access_id=access_id,
    base_json={str(Path(base_json).resolve())!r},
    work_dir={str(Path(work_dir).resolve())!r},
    target_yield_locus={str(Path(target_yield_locus).resolve())!r},
    poll_interval={float(poll_interval)!r},
    timeout={timeout!r},
)
'''


def launch_calibration(
    calibration_name: str,
    json_path: str,
    target_yield_locus: str,
    free_parameters: dict[str, dict[str, float]],
    num_solutions: int = 8,
    target_sigma: float = 0.1,
    random_seed: int | None = None,
    poll_interval: float = 60,
    timeout: float | None = None,
    access_scheduler=None,
):
    """Launch ACCES calibration for shear-cell yield-locus matching."""
    try:
        import coexist
    except ImportError as exc:
        raise ImportError(
            "Shearcell calibration requires the optional 'coexist' dependency. "
            "Install it with `uv sync --extra calibration`."
        ) from exc

    calibration_dir = Path(calibration_name).resolve()
    calibration_dir.mkdir(parents=True, exist_ok=True)
    work_dir = calibration_dir / "candidates"
    work_dir.mkdir(exist_ok=True)

    script_path = calibration_dir / "access_shearcell.py"
    script_path.write_text(
        _render_access_script(
            base_json=json_path,
            target_yield_locus=target_yield_locus,
            work_dir=work_dir,
            free_parameters=free_parameters,
            poll_interval=poll_interval,
            timeout=timeout,
        )
    )

    with cd(calibration_dir):
        access = (
            coexist.Access(script_path.name)
            if access_scheduler is None
            else coexist.Access(script_path.name, access_scheduler)
        )
        return access.learn(
            num_solutions=num_solutions,
            target_sigma=target_sigma,
            random_seed=random_seed,
        )
