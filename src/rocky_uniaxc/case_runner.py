import sys
from pathlib import Path
from rocky_uniaxc.pyrocky.uniax import Settings, UniaxialCompressionSimulation


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m rocky_uniaxc.case_runner path/to/settings.json")
        sys.exit(1)

    settings_path = Path(sys.argv[1]).resolve()
    project_dir = settings_path.parent

    settings = Settings.from_json(settings_path, project_dir=project_dir)

    sim = UniaxialCompressionSimulation(settings)
    sim.execute()


if __name__ == "__main__":
    main()
