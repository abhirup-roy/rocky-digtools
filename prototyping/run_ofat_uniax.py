"""Example: launch a One-Factor-at-a-Time (OFAT) uniaxial compression study.

Starting from the base configuration in ``json/ofat_base.json``, this varies
the particle-particle coefficient of restitution (``cor_pp``) and the applied
compression pressure (``p_compress``) independently, holding every other
parameter at its base value, and submits the resulting cases via SLURM.

Run with::

    python prototyping/run_ofat_uniax.py
"""

from pathlib import Path

from rocky_digtools.models.uniax import launch_ofat
from rocky_digtools.utils import RockyScheduler

BASE_JSON = Path(__file__).parent.resolve() / "json" / "ofat_base_uniax.json"

# 1. Define the cluster scheduler settings (e.g., BlueBear CPU)
scheduler = RockyScheduler.bb_cpu(ncpus=20, run_days=3)

# 2. Describe the OFAT design: which parameters to vary, over what range,
#    and what to hold the other levels of each factor at ("h"igh, "l"ow,
#    or "m"id — the base value stays fixed across every other factor).
ofat_values = {
    "parameters": ["cor_pp", "p_compress"],
    "test_range": [(0.1, 0.9), (5e3, 25e3)],
    "hold_values": ["m", "m"],
}

# 3. Launch the OFAT block
launch_ofat(
    sweep_name="ofat_uniax_example",
    scheduler=scheduler,
    ofat_values=ofat_values,
    n_points=5,
    json_path=str(BASE_JSON),
    autolaunch=True,
    target="CPU",
    backend="rocky_prepost",
)
