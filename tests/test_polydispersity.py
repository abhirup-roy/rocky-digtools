import json
from pathlib import Path
from unittest.mock import MagicMock

from rocky_digtools.models.doe import iter_params, prepare_case, script_context_from_params
from rocky_digtools.models.uniax.doe import UNIAX_RUNTIME, UNIAX_SCHEMA
from rocky_digtools.models.uniax.simulation import Settings
from rocky_digtools.particles_shapes import Sphere


def test_polydisperse_radius_flows_from_json_to_rocky(tmp_path, sweep_json):
    sweep_json = Path(sweep_json)
    config = json.loads(sweep_json.read_text())
    config["particle_properties"]["radius"] = {
        "0.0001": 0.2,
        "0.00015": 0.6,
        "0.0002": 0.2,
    }
    config["shape"][0]["n_corners"] = 10
    sweep_json.write_text(json.dumps(config))

    params = iter_params(str(sweep_json), UNIAX_SCHEMA)[0]
    assert params.radius == {0.0001: 0.2, 0.00015: 0.6, 0.0002: 0.2}

    case_dir = tmp_path / "case"
    (case_dir / "meshes").mkdir(parents=True)
    context = script_context_from_params(
        params, "CPU", extra_key_map=UNIAX_RUNTIME.extra_key_map
    )
    prepare_case(case_dir, context, backend="pyrocky", runtime=UNIAX_RUNTIME)
    data = json.loads((case_dir / "settings.json").read_text())
    settings = Settings.from_dict({**data, "project_dir": str(case_dir)})

    particle = MagicMock()
    entries = [MagicMock() for _ in range(3)]
    particle.GetSizeDistributionList.return_value.New.side_effect = entries
    Sphere(settings.p_radius).particle2rocky(particle, MagicMock())

    sizes = [entry.SetSize.call_args.args[0] for entry in entries]
    cumulative = [entry.SetCumulativePercentage.call_args.args[0] for entry in entries]
    assert sizes == [0.0002, 0.00015, 0.0001]
    assert cumulative == [100, 80, 20]
