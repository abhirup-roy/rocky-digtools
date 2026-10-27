import builtins
import json

import numpy as np
import pytest

from rocky_digtools.models.shearcell.calibration import (
    _render_access_script,
    prepare_candidate_settings,
    yield_locus_error,
)


def _base_shearcell_config():
    return {
        "shape": {
            "name": "polyhedron",
            "vert_ar": 1,
            "horiz_ar": 1,
            "n_corners": 10,
            "sq_degree": 2.0,
        },
        "particle_properties": {
            "radius": 150e-6,
            "density": 2700,
            "poisson": 0.25,
            "youngmod": 5e6,
        },
        "interactions": {
            "pp": {
                "fric_dyn": 0.7,
                "fric_stat": 0.3,
                "fric_rolling": 0.1,
                "cor": 0.4,
            },
            "pw": {
                "fric_dyn": 0.7,
                "fric_stat": 0.3,
                "fric_rolling": 0.1,
                "cor": 0.4,
            },
        },
        "experiment_settings": {
            "box_len": 0.0025,
            "t_settle": 0.5,
            "t_compression": 2.0,
            "sigma_pre": 15e3,
            "n_shear_points": 5,
            "n_procs": 20,
            "neighbour_search": "BVH",
            "t_shear": 5.0,
            "shear_vel": 0.01,
        },
        "contact_model": {
            "normal": "linear_hysteresis",
            "tangential": "coulomb_limit",
            "rolling": "none",
            "adhesion": "none",
        },
    }


def test_prepare_candidate_settings_applies_overrides(tmp_path):
    base_json = tmp_path / "base.json"
    base_json.write_text(json.dumps(_base_shearcell_config()))

    settings_path = prepare_candidate_settings(
        base_json,
        tmp_path / "candidate_0",
        {"fric_dyn_pp": 0.8, "sigma_pre": 20_000.0},
    )

    data = json.loads(settings_path.read_text())
    assert data["fric_dyn_pp"] == 0.8
    assert data["sigma_pre"] == 20_000.0
    assert data["p_radius"] == 150e-6
    assert data["processor"] == "CPU"


def test_yield_locus_error_interpolates_target_points(tmp_path):
    target = tmp_path / "target.csv"
    target.write_text("sigma,tau\n20,10\n30,15\n")

    error = yield_locus_error(
        target,
        sigma=np.array([10.0, 20.0, 30.0]),
        tau=np.array([5.0, 10.0, 15.0]),
    )

    assert error == pytest.approx(0.0)


def test_yield_locus_error_rejects_missing_columns(tmp_path):
    target = tmp_path / "target.csv"
    target.write_text("normal,stress\n20,10\n")

    with pytest.raises(ValueError, match="sigma.*tau"):
        yield_locus_error(target, np.array([20.0]), np.array([10.0]))


def test_launch_calibration_reports_missing_coexist(monkeypatch, tmp_path):
    from rocky_digtools.models.shearcell.calibration import launch_calibration

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "coexist":
            raise ImportError("missing coexist")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="uv sync --extra calibration"):
        launch_calibration(
            calibration_name=str(tmp_path / "cal"),
            json_path=str(tmp_path / "base.json"),
            target_yield_locus=str(tmp_path / "target.csv"),
            free_parameters={"fric_dyn_pp": {"min": 0.1, "max": 1.0}},
        )


def test_generated_access_script_contains_required_block(tmp_path):
    script = _render_access_script(
        base_json=tmp_path / "base.json",
        target_yield_locus=tmp_path / "target.csv",
        work_dir=tmp_path / "candidates",
        free_parameters={
            "fric_dyn_pp": {"min": 0.1, "max": 1.0, "value": 0.5},
            "sigma_pre": {"min": 5_000.0, "max": 25_000.0},
        },
        poll_interval=5,
        timeout=60,
    )

    assert "# ACCESS PARAMETERS START" in script
    assert "# ACCESS PARAMETERS END" in script
    assert "parameters = coexist.create_parameters" in script
    assert "access_id = 0" in script
    assert "work_dir=" in script
    assert "error = evaluate_candidate" in script
