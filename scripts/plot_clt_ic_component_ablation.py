#!/usr/bin/env python
"""Plot the focused COSTA-IC component ablation results."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

VARIANT_ORDER = [
    "COSTA_IC",
    "COSTA_IC_no_balance",
    "COSTA_IC_no_switch",
    "COSTA_IC_no_balance_no_switch",
]


def relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot COSTA-IC component ablation outputs.")
    ap.add_argument("--input-root", default="outputs/clt_ic_ablation")
    ap.add_argument("--out-dir", default="outputs/figures/clt_ic_ablation")
    ap.add_argument("--dataset", default=None, help="Optional dataset substring filter, e.g. retailrocket or movielens.")
    ap.add_argument("--estimator", default="weighted_ht")
    args = ap.parse_args()

    input_root = (ROOT / args.input_root).resolve() if not Path(args.input_root).is_absolute() else Path(args.input_root)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for f in input_root.glob("*/ic_component_summary.csv"):
        if args.dataset and args.dataset.lower() not in f.parent.name.lower():
            continue
        df = pd.read_csv(f)
        df["run_folder"] = f.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No ic_component_summary.csv files found under {relpath(input_root)}")
    data = pd.concat(frames, ignore_index=True)
    if "estimator" in data.columns:
        data = data[data["estimator"] == args.estimator].copy()
    data["abs_bias"] = data["bias"].abs()
    data["variant"] = pd.Categorical(data["variant"], categories=VARIANT_ORDER, ordered=True)
    data = data.sort_values(["dataset", "p", "model", "variant"])
    data.to_csv(out_dir / "clt_ic_component_summary_tidy.csv", index=False)

    sns.set_theme(style="whitegrid")

    # RMSE by variant.
    g = sns.catplot(
        data=data,
        x="variant",
        y="rmse",
        hue="p",
        col="model",
        row="dataset" if data["dataset"].nunique() > 1 else None,
        kind="bar",
        sharey=False,
        height=3.2,
        aspect=1.6,
    )
    g.set_xticklabels(rotation=25, ha="right")
    g.set_axis_labels("COSTA-IC variant", "RMSE")
    g.fig.tight_layout()
    g.fig.savefig(out_dir / "ic_component_rmse.pdf", dpi=180)
    plt.close(g.fig)

    # Coverage by variant.
    g = sns.catplot(
        data=data,
        x="variant",
        y="ci_coverage",
        hue="p",
        col="model",
        row="dataset" if data["dataset"].nunique() > 1 else None,
        kind="bar",
        sharey=True,
        height=3.2,
        aspect=1.6,
    )
    for ax in g.axes.flat:
        ax.axhline(0.90, linestyle="--", linewidth=1)
        ax.set_ylim(0, 1.05)
    g.set_xticklabels(rotation=25, ha="right")
    g.set_axis_labels("COSTA-IC variant", "90% HAC CI coverage")
    g.fig.tight_layout()
    g.fig.savefig(out_dir / "ic_component_coverage.pdf", dpi=180)
    plt.close(g.fig)

    # Bias-std tradeoff.
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x="abs_bias", y="std", hue="variant", style="p", s=70)
    plt.xlabel("|bias|")
    plt.ylabel("std")
    plt.tight_layout()
    plt.savefig(out_dir / "ic_component_bias_std_tradeoff.pdf", dpi=180)
    plt.close()

    print(f"Saved COSTA-IC component ablation figures to {relpath(out_dir)}")


if __name__ == "__main__":
    main()
