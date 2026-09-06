#!/usr/bin/env python3
"""Create manuscript-ready revised26 inference tables and four figure panels."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "legacy_global_abs": "Legacy",
    "heterogeneous_local": "Heterogeneous",
    "design_aware_signed": "DA signed",
    "design_aware_positive": "DA positive",
    "plugin_full_diagnostic": "Plugin full (diagnostic)",
}
MAIN_METHODS = [
    "legacy_global_abs",
    "heterogeneous_local",
    "design_aware_signed",
    "design_aware_positive",
]
DEFAULT_INTERNAL_DESIGNS = ["COSTA", "COSTA_IC_lam1"]
DATASET_LABELS = {
    "retailrocket": "RetailRocket",
    "retail_rocket": "RetailRocket",
    "movielens": "MovieLens",
    "movielens10m": "MovieLens",
}
DATASET_ABBR = {"RetailRocket": "RR", "MovieLens": "ML"}
DATASET_ORDER = {"RetailRocket": 0, "MovieLens": 1}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def load_summaries(input_root: Path) -> pd.DataFrame:
    files = sorted(input_root.rglob("inference_revised26_summary.csv"))
    if not files:
        raise FileNotFoundError(f"No inference_revised26_summary.csv below {input_root}")
    frames = [pd.read_csv(path).assign(source_file=str(path)) for path in files]
    return pd.concat(frames, ignore_index=True, sort=False)


def escape_tex(value: object) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def select_main(
    df: pd.DataFrame,
    *,
    p: float,
    model: str,
    internal_designs: list[str],
) -> pd.DataFrame:
    required = {
        "dataset", "p", "model", "method", "internal_design", "design", "ci_level",
        "coverage_design_center", "coverage_raw_gate", "coverage_bias_corrected_gate",
        "avg_width", "se_to_mc_sd", "design_z_mean", "design_z_sd",
        "design_z_q025", "design_z_q50", "design_z_q975", "variance_truncation_rate",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Summary file is missing required columns: {missing}")
    main = df[
        np.isclose(df["p"].astype(float), float(p))
        & (df["model"].astype(str) == str(model))
        & df["method"].isin(MAIN_METHODS)
        & df["internal_design"].isin(internal_designs)
    ].copy()
    if main.empty:
        raise RuntimeError("No rows match the requested p/model/design filter")
    main["dataset"] = main["dataset"].astype(str).map(
        lambda value: DATASET_LABELS.get(value.lower(), value)
    )
    main["dataset_order"] = main["dataset"].map(DATASET_ORDER).fillna(99)
    main["method_order"] = main["method"].map({m: i for i, m in enumerate(MAIN_METHODS)})
    main["design_order"] = main["internal_design"].map({d: i for i, d in enumerate(internal_designs)})
    main = main.sort_values(["dataset_order", "design_order", "method_order"])

    expected = len(main["dataset"].unique()) * len(internal_designs) * len(MAIN_METHODS)
    if len(main) != expected:
        counts = main.groupby(["dataset", "internal_design", "method"]).size()
        raise RuntimeError(
            "The selected results do not contain exactly one row per dataset/design/method. "
            f"Expected {expected} rows, found {len(main)}. Counts: {counts.to_dict()}"
        )
    return main


def write_main_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}lllccccc@{}}",
        r"\toprule",
        r"Dataset & Design & Variance method & DC cov. & GATE cov. & BC cov. & Width & SE/SD \\",
        r"\midrule",
    ]
    previous_dataset: str | None = None
    previous_design: tuple[str, str] | None = None
    for row in df.itertuples(index=False):
        dataset = str(row.dataset)
        design = str(row.design)
        group = (dataset, design)
        if previous_design is not None and group != previous_design:
            lines.append(r"\addlinespace[2pt]")
        dataset_cell = escape_tex(dataset) if dataset != previous_dataset else ""
        design_cell = escape_tex(design) if group != previous_design else ""
        lines.append(
            f"{dataset_cell} & {design_cell} & {escape_tex(METHOD_LABELS[row.method])} & "
            f"{row.coverage_design_center:.3f} & {row.coverage_raw_gate:.3f} & "
            f"{row.coverage_bias_corrected_gate:.3f} & {row.avg_width:.3f} & "
            f"{row.se_to_mc_sd:.3f} \\\\"
        )
        previous_dataset = dataset
        previous_design = group
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_studentized_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}lllrrrrrr@{}}",
        r"\toprule",
        r"Dataset & Design & Variance method & Mean & SD & $q_{.025}$ & $q_{.50}$ & $q_{.975}$ & Trunc. \\",
        r"\midrule",
    ]
    previous_dataset: str | None = None
    previous_design: tuple[str, str] | None = None
    for row in df.itertuples(index=False):
        dataset = str(row.dataset)
        design = str(row.design)
        group = (dataset, design)
        if previous_design is not None and group != previous_design:
            lines.append(r"\addlinespace[2pt]")
        dataset_cell = escape_tex(dataset) if dataset != previous_dataset else ""
        design_cell = escape_tex(design) if group != previous_design else ""
        lines.append(
            f"{dataset_cell} & {design_cell} & {escape_tex(METHOD_LABELS[row.method])} & "
            f"{row.design_z_mean:.3f} & {row.design_z_sd:.3f} & "
            f"{row.design_z_q025:.3f} & {row.design_z_q50:.3f} & {row.design_z_q975:.3f} & "
            f"{row.variance_truncation_rate:.3f} \\\\"
        )
        previous_dataset = dataset
        previous_design = group
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _design_display(internal_design: str, display_design: str) -> str:
    if internal_design == "COSTA":
        return "COSTA"
    if internal_design.startswith("COSTA_IC"):
        return "COSTA-LP"
    return display_design


def _series_label(dataset: str, internal_design: str, display_design: str) -> str:
    dataset_part = DATASET_ABBR.get(dataset, dataset)
    return f"{dataset_part}-{_design_display(internal_design, display_design)}"


def write_metric_figure(
    df: pd.DataFrame,
    *,
    column: str,
    ylabel: str,
    path: Path,
    nominal: float | None = None,
    reference_one: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(MAIN_METHODS))
    fig, ax = plt.subplots(figsize=(6.2, 4.25))
    for (dataset, internal_design), group in df.groupby(["dataset", "internal_design"], sort=False):
        group = group.sort_values("method_order")
        label = _series_label(dataset, internal_design, str(group["design"].iloc[0]))
        ax.plot(x, group[column].astype(float), marker="o", label=label)
    if nominal is not None:
        ax.axhline(float(nominal), linestyle="--", linewidth=1.0, label=f"Nominal {nominal:.2f}")
    if reference_one:
        ax.axhline(1.0, linestyle="--", linewidth=1.0, label="Reference 1")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in MAIN_METHODS], rotation=22, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="outputs/inference_revised26")
    parser.add_argument(
        "--manuscript-dir",
        required=True,
        help="Directory containing the revised26 LaTeX source, figures/, and tables/.",
    )
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--model", default="linear")
    parser.add_argument(
        "--internal-designs",
        default=",".join(DEFAULT_INTERNAL_DESIGNS),
        help="Comma-separated internal designs included in the manuscript report.",
    )
    args = parser.parse_args()

    all_data = load_summaries(resolve(args.input_root))
    main_data = select_main(
        all_data,
        p=float(args.p),
        model=str(args.model),
        internal_designs=parse_csv(args.internal_designs),
    )
    manuscript = resolve(args.manuscript_dir)
    tables = manuscript / "tables"
    figures = manuscript / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    write_main_table(main_data, tables / "tab_inference_validation_revised26.tex")
    write_studentized_table(main_data, tables / "tab_inference_studentized_revised26.tex")
    nominal = float(main_data["ci_level"].iloc[0])
    write_metric_figure(
        main_data,
        column="coverage_design_center",
        ylabel="Design-centered coverage",
        path=figures / "fig_inference_design_coverage_revised26.pdf",
        nominal=nominal,
        ylim=(0.0, 1.02),
    )
    write_metric_figure(
        main_data,
        column="coverage_raw_gate",
        ylabel="Raw-GATE coverage",
        path=figures / "fig_inference_gate_coverage_revised26.pdf",
        nominal=nominal,
        ylim=(0.0, 1.02),
    )
    write_metric_figure(
        main_data,
        column="avg_width",
        ylabel="Average interval width",
        path=figures / "fig_inference_width_revised26.pdf",
    )
    write_metric_figure(
        main_data,
        column="se_to_mc_sd",
        ylabel="Average estimated SE / Monte Carlo SD",
        path=figures / "fig_inference_se_ratio_revised26.pdf",
        reference_one=True,
    )
    print(f"Wrote manuscript inference artifacts under {manuscript}")


if __name__ == "__main__":
    main()
