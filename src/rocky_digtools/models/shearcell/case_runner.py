"""CLI entry point for running a single shear-cell case.

Reads a ``settings.json`` file, constructs a
:class:`~rocky_digtools.models.shearcell.simulation.ShearCellSimulation`,
runs the pre-shear phase, spawns the shear-point restart cases, and submits
them as a SLURM array job.  Each array task runs
:func:`~rocky_digtools.models.shearcell.simulation.run_shear_point`; once
all tasks complete,
:func:`~rocky_digtools.models.shearcell.simulation.aggregate_results`
fits the Mohr circles and dumps the flow metrics.
"""

import json
import sys
from pathlib import Path

from .simulation import Settings, ShearCellSimulation


def main():
    """Parse command-line arguments and run a single shear-cell case.

    Expects a single argument: the path to a ``settings.json`` file.

    Example::

        python -m rocky_digtools.models.shearcell.case_runner path/to/settings.json
    """
    if len(sys.argv) < 2:
        print(
            "Usage: python -m rocky_digtools.models.shearcell.case_runner "
            "path/to/settings.json"
        )
        sys.exit(1)

    settings_path = Path(sys.argv[1]).resolve()
    project_dir = settings_path.parent

    with open(settings_path) as f:
        data = json.load(f)
    data["project_dir"] = str(project_dir)
    settings = Settings.from_dict(data)

    sim = ShearCellSimulation(settings)
    sim.execute()


if __name__ == "__main__":
    main()
