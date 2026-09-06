from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_inference_validation_revised26.py"
SPEC = importlib.util.spec_from_file_location("plot_inference_validation_revised26", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(method: str) -> dict[str, object]:
    return {
        "dataset": "RetailRocket",
        "design": "COSTA",
        "internal_design": "COSTA",
        "method": method,
        "coverage_design_center": 0.95,
        "coverage_raw_gate": 0.90,
        "coverage_bias_corrected_gate": 0.95,
        "avg_width": 0.25,
        "se_to_mc_sd": 1.01,
        "design_z_mean": 0.01,
        "design_z_sd": 1.02,
        "design_z_q025": -1.96,
        "design_z_q50": 0.0,
        "design_z_q975": 1.96,
        "variance_truncation_rate": 0.0,
    }


def test_generated_latex_tables_have_valid_row_terminators(tmp_path: Path) -> None:
    df = pd.DataFrame([_row(method) for method in MODULE.MAIN_METHODS])
    main_path = tmp_path / "main.tex"
    studentized_path = tmp_path / "studentized.tex"
    MODULE.write_main_table(df, main_path)
    MODULE.write_studentized_table(df, studentized_path)

    for path in (main_path, studentized_path):
        text = path.read_text(encoding="utf-8")
        assert "\\toprule" in text
        assert "\\bottomrule" in text
        header = next(line for line in text.splitlines() if "Variance method" in line)
        assert header.endswith(r"\\")
