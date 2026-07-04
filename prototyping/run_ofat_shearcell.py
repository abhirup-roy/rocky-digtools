from pathlib import Path

from rocky_digtools.models.shearcell import launch_ofat
from rocky_digtools.utils import RockyScheduler

BASE_JSON = Path(__file__).parent.resolve() / "json" / "ofat_base_shearcell.json"

# 1. Define the cluster scheduler settings (e.g., BlueBear CPU)
scheduler = RockyScheduler.bb_cpu(ncpus=20, run_days=3)

# 2. Describe the OFAT design: which parameters to vary, over what range,
#    and what to hold the other levels of each factor at ("high, "low,
#    or "mid" — the base value stays fixed across every other factor).
ofat_values = {
    "parameters": ["cor_pp", "sigma_pre"],
    "test_range": [(0.1, 0.9), (5e3, 25e3)],
    "hold_values": ["mid", "low"],
}

# 3. Launch the OFAT block
launch_ofat(
    sweep_name="ofat_shearcell_example",
    scheduler=scheduler,
    ofat_values=ofat_values,
    n_points=3,
    json_path=str(BASE_JSON),
    autolaunch=True,
    target="CPU",
    backend="rocky_prepost",
)
