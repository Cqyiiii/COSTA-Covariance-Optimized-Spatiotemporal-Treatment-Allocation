#!/usr/bin/env python
"""Plot CLT / CI validation outputs.

The script only uses relative paths by default, so generated notebooks/figures do
not reveal local user directories.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(p)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Plot COSTA CLT validation results.")
    ap.add_argument("--input-root", default="outputs/clt")
    ap.add_argument("--out-dir", default="outputs/figures/clt")
    ap.add_argument("--dataset", default=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = ROOT / input_root
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.glob("**/clt_summary.csv"))
    if args.dataset:
        files = [f for f in files if args.dataset.lower() in str(f).lower()]
    if not files:
        raise FileNotFoundError(f"No clt_summary.csv found under {rel(input_root)}")
    summary = pd.concat([pd.read_csv(f).assign(source=rel(f.parent)) for f in files], ignore_index=True)
    summary.to_csv(out_dir / "clt_summary_tidy.csv", index=False)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    # 1. Bias/std tradeoff.
    s = summary.copy()
    s["abs_bias"] = s["bias"].abs()
    g = sns.relplot(
        data=s, x="abs_bias", y="std", hue="design", style="p",
        col="model", row="dataset" if "dataset" in s.columns and s["dataset"].nunique() > 1 else None,
        kind="scatter", s=80, height=3.2, aspect=1.2,
    )
    g.set_axis_labels("|bias|", "Monte Carlo std")
    g.fig.suptitle("CLT validation: bias--std tradeoff", y=1.02)
    g.savefig(out_dir / "clt_bias_std_tradeoff.pdf", dpi=180, bbox_inches="tight")
    plt.close(g.fig)

    # 2. Coverage by p/design/model.
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=s, x="p", y="ci_coverage", hue="design", style="model", marker="o", ax=ax)
    ax.axhline(float(s["ci_level"].iloc[0]) if "ci_level" in s else 0.90, linestyle="--", linewidth=1)
    ax.set_ylabel("Empirical 90% CI coverage")
    ax.set_xlabel("Treatment probability p")
    ax.set_title("Product-graph HAC CI coverage")
    fig.tight_layout()
    fig.savefig(out_dir / "clt_ci_coverage.pdf", dpi=180)
    plt.close(fig)

    # 3. MC oracle interval width vs average HAC interval width.
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = s.melt(
        id_vars=["dataset", "p", "model", "design"],
        value_vars=["mc_interval_width", "avg_ci_halfwidth"],
        var_name="interval_metric", value_name="value",
    )
    # Halfwidth should be comparable to half of the MC interval width.
    plot_df.loc[plot_df["interval_metric"] == "mc_interval_width", "value"] /= 2.0
    plot_df["interval_metric"] = plot_df["interval_metric"].replace({"mc_interval_width": "MC oracle half-width", "avg_ci_halfwidth": "Avg HAC half-width"})
    sns.barplot(data=plot_df, x="design", y="value", hue="interval_metric", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("90% half-width")
    ax.set_title("HAC CI half-width vs Monte Carlo oracle half-width")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "clt_interval_width_comparison.pdf", dpi=180)
    plt.close(fig)

    # 4. Locality diagnostic: standard COSTA vs COSTA-IC.
    loc_cols = [c for c in ["far_product_mean_abs_R", "far_item_mean_abs_R", "far_time_mean_abs_R"] if c in s.columns]
    if loc_cols:
        loc = s.melt(id_vars=["dataset", "p", "model", "design"], value_vars=loc_cols, var_name="locality_metric", value_name="value")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(data=loc, x="locality_metric", y="value", hue="design", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Mean absolute normalized covariance")
        ax.set_title("Inference locality diagnostics")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(out_dir / "clt_locality_diagnostics.pdf", dpi=180)
        plt.close(fig)

    # 5. Standardized sampling distribution histograms from draw files.
    draw_files = sorted(input_root.glob("**/clt_draws.csv"))
    if args.dataset:
        draw_files = [f for f in draw_files if args.dataset.lower() in str(f).lower()]
    if draw_files:
        draws = pd.concat([pd.read_csv(f).assign(source=rel(f.parent)) for f in draw_files], ignore_index=True)
        draws = draws.merge(summary[["dataset", "p", "model", "design", "mean_est", "std"]], on=["dataset", "p", "model", "design"], how="left")
        draws["z_mc"] = (draws["weighted_ht"] - draws["mean_est"]) / draws["std"].replace(0, np.nan)
        # Keep the plot readable by focusing on the first model if many are present.
        first_model = sorted(draws["model"].dropna().unique())[0]
        sub = draws[draws["model"] == first_model].copy()
        g = sns.displot(data=sub, x="z_mc", hue="design", col="p", kind="hist", stat="density", common_norm=False, height=3.2, aspect=1.2, bins=35)
        xs = np.linspace(-4, 4, 300)
        for ax in g.axes.flat:
            ax.plot(xs, np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi), color="black", linewidth=1)
            ax.set_xlim(-4, 4)
        g.set_axis_labels("MC-standardized estimate", "density")
        g.fig.suptitle(f"Sampling distribution vs standard normal ({first_model})", y=1.02)
        g.savefig(out_dir / "clt_standardized_histograms.pdf", dpi=180, bbox_inches="tight")
        plt.close(g.fig)



    # 6. COSTA-IC lambda-local search: post-hoc RMSE and coverage.
    search_files = sorted(input_root.glob("**/costa_ic_lambda_search.csv"))
    if args.dataset:
        search_files = [f for f in search_files if args.dataset.lower() in str(f).lower()]
    if search_files:
        search = pd.concat([pd.read_csv(f).assign(source=rel(f.parent)) for f in search_files], ignore_index=True)
        search.to_csv(out_dir / "clt_lambda_search_tidy.csv", index=False)
        if {"lambda_local", "rmse", "ci_coverage"}.issubset(search.columns):
            search = search.copy()
            search["lambda_local"] = search["lambda_local"].astype(float)
            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            sns.lineplot(data=search, x="lambda_local", y="rmse", hue="p", style="model", marker="o", ax=ax)
            ax.set_xscale("log")
            ax.set_xlabel("COSTA-IC locality penalty $\\lambda_{loc}$")
            ax.set_ylabel("RMSE")
            ax.set_title("COSTA-IC lambda search: RMSE")
            fig.tight_layout()
            fig.savefig(out_dir / "clt_lambda_search_rmse.pdf", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            sns.lineplot(data=search, x="lambda_local", y="ci_coverage", hue="p", style="model", marker="o", ax=ax)
            target = float(search["ci_level"].iloc[0]) if "ci_level" in search.columns else 0.90
            ax.axhline(target, linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set_xlabel("COSTA-IC locality penalty $\\lambda_{loc}$")
            ax.set_ylabel("Empirical 90% CI coverage")
            ax.set_title("COSTA-IC lambda search: HAC coverage")
            fig.tight_layout()
            fig.savefig(out_dir / "clt_lambda_search_coverage.pdf", dpi=180)
            plt.close(fig)

    print(f"Saved CLT figures to {rel(out_dir)}")


if __name__ == "__main__":
    main()
