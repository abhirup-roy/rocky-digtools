"""Run one drum case: python -m rocky_digtools.models.gdrum.case_runner settings.json."""

import argparse
import json
from pathlib import Path

from .simulation import Settings, GranularDrumSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("settings", type=Path)
    path = parser.parse_args().settings.resolve()
    data = json.loads(path.read_text())
    data["project_dir"] = str(path.parent)
    if data.get("mesh_dir") and not Path(data["mesh_dir"]).is_absolute():
        data["mesh_dir"] = str(path.parent / data["mesh_dir"])
    GranularDrumSimulation(Settings.from_dict(data)).execute()


if __name__ == "__main__":
    main()
