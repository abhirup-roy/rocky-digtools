"""Unit tests verifying the backend functions that prepare directories and configuration templates."""

import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, patch

from rocky_digtools.models.doe import (
    ShapeConfig,
    SimParams,
    case_directory,
    get_unique_box_lens,
    iter_params,
    prepare_case,
    script_context_from_params,
)
from rocky_digtools.models.uniax.doe import UNIAX_RUNTIME, UNIAX_SCHEMA
from rocky_digtools.models import PyrockySimulation
from rocky_digtools.models.uniax.simulation import (
    Settings,
    UniaxialCompressionSimulation,
)


def _script_context(params, target="GPU"):
    return script_context_from_params(
        params, target, extra_key_map=UNIAX_RUNTIME.extra_key_map
    )


class TestShapeConfig:
    def test_defaults(self):
        sc = ShapeConfig()
        assert sc.name == "sphere"
        assert sc.vert_ar == 1.0
        assert sc.horiz_ar == 1.0
        assert sc.n_corners == 6
        assert sc.sq_degree == 2.0
        assert sc.particle_path == ""
        assert sc.smoothness == 0.5

    def test_from_dict_partial(self):
        sc = ShapeConfig.from_dict({"name": "polyhedron"})
        assert sc.name == "polyhedron"
        assert sc.vert_ar == 1.0  # default

    def test_from_dict_full(self):
        d = {
            "name": "custom_polyhedron",
            "vert_ar": 2.0,
            "horiz_ar": 1.5,
            "n_corners": 20,
            "sq_degree": 5.0,
            "particle_path": "/path/to.stl",
            "smoothness": 0.8,
        }
        sc = ShapeConfig.from_dict(d)
        assert sc.name == "custom_polyhedron"
        assert sc.vert_ar == 2.0
        assert sc.smoothness == 0.8


class TestSimParams:
    def test_common_fields(self, sample_sim_params):
        assert sample_sim_params.radius == 0.001
        assert sample_sim_params.box_len == 0.01
        assert sample_sim_params.shape.name == "polyhedron"

    def test_extra_fields(self, sample_sim_params):
        assert sample_sim_params.extra == {"p_compress": 1000.0}

    def test_extra_defaults_empty(self):
        sp = SimParams(
            radius=0.001,
            density=2700,
            poisson=0.25,
            youngmod=5e6,
            fric_dyn_pp=0.5,
            fric_stat_pp=0.3,
            fric_rolling_pp=0.1,
            cor_pp=0.9,
            fric_dyn_pw=0.5,
            fric_stat_pw=0.3,
            cor_pw=0.9,
            box_len=0.01,
            normal="linear_hysteresis",
            tangential="coulomb_limit",
            rolling="none",
            adhesion="none",
        )
        assert sp.extra == {}


class TestCaseDirectory:
    def test_creates_dirs(self, tmp_path):
        sweep_dir = tmp_path / "sweep_test"
        with case_directory(str(sweep_dir), 0) as case_dir:
            assert (Path(case_dir) / "plots").is_dir()
            assert (Path(case_dir) / "meshes").is_dir()

    def test_custom_meshdir(self, tmp_path):
        sweep_dir = tmp_path / "sweep_test"
        with case_directory(str(sweep_dir), 1, meshdir="custom_mesh") as case_dir:
            assert (Path(case_dir) / "custom_mesh").is_dir()


class TestScriptContextFromParams:
    def test_all_keys_present(self, sample_sim_params):
        ctx = _script_context(sample_sim_params)
        expected_keys = {
            "RADIUS_P",
            "DENSITY_P",
            "POISSON_P",
            "YOUNGMOD_P",
            "DYNAMIC_FRICTION_PP",
            "STATIC_FRICTION_PP",
            "COR_PP",
            "SURF_EN_PP",
            "DYNAMIC_FRICTION_PW",
            "STATIC_FRICTION_PW",
            "COR_PW",
            "SURF_EN_PW",
            "L_BOX",
            "P_COMPRESS",
            "NORMAL_MODEL",
            "TANG_MODEL",
            "ROLLING_MODEL",
            "ADH_MODEL",
            "SHAPE",
            "VERT_AR",
            "HORIZ_AR",
            "N_CORNERS",
            "SQ_DEGREE",
            "PARTICLE_PATH",
            "SMOOTHNESS",
            "XPU",
            "MESH_DIR",
            "ROLLING_FRICTION",
        }
        assert expected_keys.issubset(set(ctx.keys()))

    def test_no_extra_key_map_omits_extras(self, sample_sim_params):
        ctx = script_context_from_params(sample_sim_params, "GPU")
        assert "P_COMPRESS" not in ctx

    def test_rolling_fric_zero_when_none(self, sample_sim_params):
        sample_sim_params.rolling = "none"
        ctx = _script_context(sample_sim_params)
        assert ctx["ROLLING_FRICTION"] == 0

    def test_rolling_fric_nonzero(self, sample_sim_params):
        sample_sim_params.rolling = "type_a"
        ctx = _script_context(sample_sim_params)
        assert ctx["ROLLING_FRICTION"] == sample_sim_params.fric_rolling_pp


class TestGetUniqueBoxLens:
    def test_unique(self, sample_sim_params):
        p2 = SimParams(
            radius=0.002,
            density=2700,
            poisson=0.25,
            youngmod=5e6,
            fric_dyn_pp=0.5,
            fric_stat_pp=0.3,
            fric_rolling_pp=0.1,
            cor_pp=0.9,
            fric_dyn_pw=0.5,
            fric_stat_pw=0.3,
            cor_pw=0.9,
            box_len=0.02,
            normal="linear_hysteresis",
            tangential="coulomb_limit",
            rolling="none",
            adhesion="none",
            extra={"p_compress": 1000.0},
        )
        result = get_unique_box_lens([sample_sim_params, p2])
        assert result == {0.01, 0.02}

    def test_all_same(self, sample_sim_params):
        result = get_unique_box_lens([sample_sim_params, sample_sim_params])
        assert result == {0.01}


class TestPrepareCase:
    def test_jkr_flows_to_rocky_and_contact_reporting(
        self, tmp_path, sweep_json
    ):
        import json

        config = json.loads(Path(sweep_json).read_text())
        config["interactions"]["pp"]["surf_en"] = [0.12]
        config["interactions"]["pw"]["surf_en"] = [0.34]
        config["contact_model"]["adhesion"] = ["JKR"]
        config["shape"][0]["n_corners"] = 10
        Path(sweep_json).write_text(json.dumps(config))
        params = iter_params(sweep_json, UNIAX_SCHEMA)[0]

        case_dir = tmp_path / "case_0"
        (case_dir / "meshes").mkdir(parents=True)
        prepare_case(
            case_dir,
            _script_context(params),
            backend="pyrocky",
            runtime=UNIAX_RUNTIME,
        )

        data = json.loads((case_dir / "settings.json").read_text())
        settings = Settings.from_dict({**data, "project_dir": str(case_dir)})

        pp_interaction = MagicMock()
        pw_interaction = MagicMock()
        interactions = MagicMock()
        interactions.GetMaterialsInteraction.side_effect = [
            pp_interaction,
            pw_interaction,
        ]
        study = MagicMock()
        study.GetMaterialsInteractionCollection.return_value = interactions
        simulation = SimpleNamespace(
            settings=settings,
            _study=study,
            _materials={"particle_mat": MagicMock(), "wall_mat": MagicMock()},
            _ser=lambda proxy: proxy,
        )

        PyrockySimulation.load_interactions(simulation)
        PyrockySimulation.sim_physics(simulation)
        UniaxialCompressionSimulation.load_modules(simulation)

        pp_interaction.SetSurfaceEnergy.assert_called_once_with(0.12, "J/m2")
        pw_interaction.SetSurfaceEnergy.assert_called_once_with(0.34, "J/m2")
        study.GetPhysics.return_value.SetAdhesionModel.assert_called_once_with("JKR")
        study.GetContactData.return_value.EnableIncludeAdhesiveContacts.assert_called_once_with()

    def test_pyrocky_backend(self, tmp_path, sample_sim_params):
        case_dir = tmp_path / "case_0"
        case_dir.mkdir()
        (case_dir / "meshes").mkdir()
        ctx = _script_context(sample_sim_params)
        prepare_case(case_dir, ctx, backend="pyrocky", runtime=UNIAX_RUNTIME)

        # Verify settings.json
        import json

        settings_path = case_dir / "settings.json"
        assert settings_path.exists()
        with open(settings_path) as f:
            data = json.load(f)
            assert data["p_radius"] == sample_sim_params.radius
            assert data["p_compress"] == sample_sim_params.extra["p_compress"]

        # Verify wrapper script
        script_path = case_dir / "script_uniax.py"
        assert script_path.exists()
        content = script_path.read_text()
        assert "rocky_digtools.models.uniax.case_runner" in content

    def test_rocky_prepost_no_template(self, tmp_path, sample_sim_params):
        case_dir = tmp_path / "case_0"
        case_dir.mkdir()
        ctx = _script_context(sample_sim_params)
        with pytest.raises(ValueError, match="rocky_template required"):
            prepare_case(case_dir, ctx, backend="rocky_prepost", runtime=UNIAX_RUNTIME)

    def test_rocky_prepost_with_template(self, tmp_path, sample_sim_params):

        case_dir = tmp_path / "case_0"
        case_dir.mkdir()
        ctx = _script_context(sample_sim_params)
        template = MagicMock()
        template.render.return_value = "# rendered script"
        prepare_case(
            case_dir,
            ctx,
            backend="rocky_prepost",
            runtime=UNIAX_RUNTIME,
            rocky_template=template,
        )
        assert (case_dir / "script_uniax.py").read_text() == "# rendered script"

    def test_invalid_backend(self, tmp_path, sample_sim_params):
        case_dir = tmp_path / "case_0"
        case_dir.mkdir()
        ctx = _script_context(sample_sim_params)
        with pytest.raises(ValueError, match="Unknown backend"):
            prepare_case(case_dir, ctx, backend="invalid", runtime=UNIAX_RUNTIME)

    def test_case_runner(self, tmp_path, fake_rocky_on_path, sample_sim_params):
        case_dir = tmp_path / "case_0"
        case_dir.mkdir()

        # Generate settings.json via the same DOE pipeline that real runs use
        mesh_dir = tmp_path / "meshes"
        mesh_dir.mkdir()
        ctx = _script_context(sample_sim_params)
        prepare_case(
            case_dir, ctx, backend="pyrocky", runtime=UNIAX_RUNTIME, mesh_path=mesh_dir
        )

        settings_path = case_dir / "settings.json"

        argv = sys.argv.copy()
        try:
            sys.argv = ["rocky_digtools.models.uniax.case_runner", str(settings_path)]
            from rocky_digtools.models.uniax import case_runner

            with (
                patch.object(
                    case_runner.UniaxialCompressionSimulation,
                    "setup",
                    return_value=None,
                ),
                patch.object(
                    case_runner.UniaxialCompressionSimulation,
                    "execute",
                    return_value=None,
                ) as mock_execute,
            ):
                case_runner.main()
                mock_execute.assert_called_once()
        finally:
            sys.argv = argv
