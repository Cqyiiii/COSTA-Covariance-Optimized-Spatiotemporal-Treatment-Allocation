from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from costa.naming import prepared_dir_name, simulation_dir_name
from costa.config import read_yaml_config, flatten_config


def test_auto_names_adapt_to_design_scale_and_kronecker_dim():
    assert prepared_dir_name("retailrocket", 15000, 4, 10) == "retailrocket_top15k_b4_k10"
    assert simulation_dir_name(
        "data/processed/retailrocket_top15k_b4_k10",
        0.5,
        500,
        400,
        d_item=32,
        d_time=3,
        designs=["COSTA"],
    ) == "retailrocket_top15k_b4_k10_p0p5_mc500_steps400_di32_dt3_costa"
    assert simulation_dir_name(
        "data/processed/retailrocket_top15k_b4_k10",
        0.5,
        500,
        400,
        seed=123,
        d_item=32,
        d_time=3,
        designs=["COSTA"],
    ) == "retailrocket_top15k_b4_k10_p0p5_mc500_steps400_di32_dt3_costa"


def test_config_is_self_contained_and_simulation_output_root_wins():
    cfg = read_yaml_config(ROOT / "configs" / "sample_debug.yaml")
    flat = flatten_config(cfg)
    assert flat["dataset"] == "sample"
    assert flat["output_root"] == "outputs/sim"
    assert flat["d_time"] == "B-1"
