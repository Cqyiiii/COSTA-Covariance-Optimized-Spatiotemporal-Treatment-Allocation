#!/usr/bin/env python
"""Plot design-aware CLT variance calibration outputs.

Reads ``outputs/clt_designaware/**/designaware_summary.csv`` and creates a few
figures comparing ordinary PG-HAC, design-MC plug-in, and design-aware PG-HAC
standard errors / coverage.  All paths are relative by default for double-blind
friendly artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def resolve_under_root(p: str | Path) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    if q.exists():
        return q.resolve()
    return (ROOT / q).resolve()


def load_summaries(input_root: Path) -> pd.DataFrame:
    files = sorted(input_root.glob("**/designaware_summary.csv"))
    if not files:
        raise FileNotFoundError(f"No designaware_summary.csv files found under {input_root}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_run"] = str(f.parent.relative_to(ROOT)) if f.parent.is_relative_to(ROOT) else f.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="outputs/clt_designaware")
    ap.add_argument("--out-dir", default="outputs/figures/clt_designaware")
    ap.add_argument("--estimator", default="weighted_ht")
    args = ap.parse_args()

    input_root = resolve_under_root(args.input_root)
    out_dir = resolve_under_root(args.out_dir)
    df = load_summaries(input_root)
    if "estimator" in df.columns:
        df = df[df["estimator"] == args.estimator].copy()
    if df.empty:
        raise RuntimeError("No rows after filtering.")

    # Long format for SE calibration.
    se_long = df.melt(
        id_vars=[c for c in ["dataset", "p", "model", "design", "std", "rmse"] if c in df.columns],
        value_vars=[c for c in ["avg_pg_hac_se", "avg_da_se", "avg_design_mc_se"] if c in df.columns],
        var_name="se_type",
        value_name="se",
    )
    se_long["se_type"] = se_long["se_type"].replace({
        "avg_pg_hac_se": "PG-HAC",
        "avg_da_se": "Design-aware PG-HAC",
        "avg_design_mc_se": "Design-MC plug-in",
    })
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=se_long, x="std", y="se", hue="se_type", style="design", s=80)
    lim = max(float(se_long["std"].max()), float(se_long["se"].max())) * 1.05
    plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    plt.xlabel("Monte Carlo empirical std")
    plt.ylabel("Estimated standard error")
    plt.title("SE calibration: estimated SE vs MC std")
    savefig(out_dir / "designaware_se_calibration.pdf")

    ratio_cols = [c for c in ["pg_hac_se_to_mc_std", "da_hac_se_to_mc_std", "design_mc_se_to_mc_std"] if c in df.columns]
    ratio = df.melt(
        id_vars=[c for c in ["dataset", "p", "model", "design"] if c in df.columns],
        value_vars=ratio_cols,
        var_name="ratio_type",
        value_name="se_to_mc_std",
    )
    ratio["ratio_type"] = ratio["ratio_type"].replace({
        "pg_hac_se_to_mc_std": "PG-HAC / MC std",
        "da_hac_se_to_mc_std": "Design-aware / MC std",
        "design_mc_se_to_mc_std": "Design-MC / MC std",
    })
    plt.figure(figsize=(11, 5))
    sns.barplot(data=ratio, x="model", y="se_to_mc_std", hue="ratio_type")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.ylabel("SE / MC std")
    plt.xlabel("Potential outcome model")
    plt.title("Variance calibration ratios")
    savefig(out_dir / "designaware_se_ratio_by_model.pdf")

    cov_cols = [c for c in ["pg_ci_coverage", "pg_biasaware_coverage", "da_ci_coverage", "da_biasaware_coverage", "design_mc_ci_coverage"] if c in df.columns]
    cov = df.melt(
        id_vars=[c for c in ["dataset", "p", "model", "design"] if c in df.columns],
        value_vars=cov_cols,
        var_name="ci_type",
        value_name="coverage",
    )
    cov["ci_type"] = cov["ci_type"].replace({
        "pg_ci_coverage": "PG-HAC",
        "pg_biasaware_coverage": "PG-HAC + bias",
        "da_ci_coverage": "Design-aware",
        "da_biasaware_coverage": "Design-aware + bias",
        "design_mc_ci_coverage": "Design-MC plug-in",
    })
    plt.figure(figsize=(12, 5))
    sns.barplot(data=cov, x="model", y="coverage", hue="ci_type")
    plt.axhline(0.90, linestyle="--", linewidth=1)
    plt.ylim(0, 1.05)
    plt.ylabel("Empirical coverage")
    plt.xlabel("Potential outcome model")
    plt.title("90% CI coverage comparison")
    savefig(out_dir / "designaware_ci_coverage.pdf")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="abs_bias", y="std", hue="design", style="p", s=90)
    plt.xlabel("|bias|")
    plt.ylabel("Monte Carlo std")
    plt.title("Point estimation bias-std tradeoff")
    savefig(out_dir / "designaware_bias_std_tradeoff.pdf")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "designaware_summary_tidy.csv", index=False)
    print(f"Saved design-aware figures to {out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir}")


if __name__ == "__main__":
    main()
