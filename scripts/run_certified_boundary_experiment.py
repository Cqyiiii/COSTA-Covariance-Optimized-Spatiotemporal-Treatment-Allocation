#!/usr/bin/env python3
"""Data-free path experiment illustrating IC-0 versus raw-GATE IC-1.

A path is partitioned into fixed contiguous clusters. Cluster assignments are
independent Bernoulli, so the assignment has finite-range dependence and a
standard centered CLT route. Positive one-hop interference creates a constant
raw-GATE centering bias, while exact design-bias correction restores causal
coverage. This experiment illustrates the paper's boundary theorem; it is not a
certificate for the default fixed-rank COSTA sampler and it does not use the
RetailRocket or MovieLens data.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(value: str) -> list[int]:
    out = [int(x.strip()) for x in value.replace(";", ",").split(",") if x.strip()]
    if not out or any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers.")
    return out


def generate_estimates(
    n: int,
    cluster_size: int,
    p: float,
    beta: float,
    theta: float,
    draws: int,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n % cluster_size != 0:
        raise ValueError(f"n={n} must be divisible by cluster_size={cluster_size}")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    n_clusters = n // cluster_size
    v = p * (1.0 - p)
    pieces: list[np.ndarray] = []
    remaining = int(draws)
    while remaining:
        m = min(remaining, int(batch_size))
        cluster_z = (rng.random((m, n_clusters)) < p).astype(np.float64)
        z = np.repeat(cluster_z, cluster_size, axis=1)
        psi = (z - p) / v
        y = beta * z
        y[:, :-1] += theta * z[:, 1:]
        pieces.append(np.mean(psi * y, axis=1))
        remaining -= m
    return np.concatenate(pieces)


def exact_centers(n: int, cluster_size: int, beta: float, theta: float) -> tuple[float, float, float]:
    target = beta + theta * (n - 1.0) / n
    same_cluster_edges = n - n // cluster_size
    design_mean = beta + theta * same_cluster_edges / n
    bias = design_mean - target
    return target, design_mean, bias


def run_one(
    n: int,
    cluster_size: int,
    p: float,
    beta: float,
    theta: float,
    mc: int,
    design_mc: int,
    batch_size: int,
    alpha: float,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    variance_rng = np.random.default_rng(seed + 10_000 + n)
    sampling_rng = np.random.default_rng(seed + 20_000 + n)
    calibration = generate_estimates(n, cluster_size, p, beta, theta, design_mc, batch_size, variance_rng)
    variance = float(np.var(calibration, ddof=1))
    se = float(np.sqrt(variance))
    if not np.isfinite(se) or se <= 0:
        raise RuntimeError(f"Nonpositive calibrated standard error for n={n}")

    estimates = generate_estimates(n, cluster_size, p, beta, theta, mc, batch_size, sampling_rng)
    target, design_mean, bias = exact_centers(n, cluster_size, beta, theta)
    zcrit = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    centered_z = (estimates - design_mean) / se
    raw_z = (estimates - target) / se
    debiased = estimates - bias
    debiased_z = (debiased - target) / se

    centered_coverage = float(np.mean(np.abs(centered_z) <= zcrit))
    raw_coverage = float(np.mean(np.abs(raw_z) <= zcrit))
    debiased_coverage = float(np.mean(np.abs(debiased_z) <= zcrit))
    ks = float(kstest(centered_z, norm.cdf).statistic)

    summary: dict[str, float | int] = {
        "n": int(n),
        "cluster_size": int(cluster_size),
        "n_clusters": int(n // cluster_size),
        "p": float(p),
        "beta": float(beta),
        "theta": float(theta),
        "mc": int(mc),
        "design_mc": int(design_mc),
        "target_gate": float(target),
        "design_expectation": float(design_mean),
        "exact_design_bias": float(bias),
        "calibrated_variance": variance,
        "calibrated_se": se,
        "abs_standardized_bias": float(abs(bias) / se),
        "centered_coverage": centered_coverage,
        "raw_gate_coverage": raw_coverage,
        "exact_bias_corrected_coverage": debiased_coverage,
        "centered_ks_distance": ks,
        "centered_z_mean": float(np.mean(centered_z)),
        "centered_z_sd": float(np.std(centered_z, ddof=1)),
    }
    draws = pd.DataFrame(
        {
            "n": int(n),
            "estimate": estimates,
            "centered_z": centered_z,
            "raw_gate_z": raw_z,
            "exact_bias_corrected_z": debiased_z,
        }
    )
    return summary, draws


def pattern_checks(summary: pd.DataFrame, alpha: float) -> dict[str, bool | float]:
    nominal = 1.0 - alpha
    largest = summary.sort_values("n").iloc[-1]
    ordered = summary.sort_values("n")
    standardized = ordered["abs_standardized_bias"].to_numpy(float)
    checks: dict[str, bool | float] = {
        "nominal_coverage": nominal,
        "centered_largest_within_0.035": bool(abs(float(largest.centered_coverage) - nominal) <= 0.035),
        "debiased_largest_within_0.035": bool(abs(float(largest.exact_bias_corrected_coverage) - nominal) <= 0.035),
        "raw_largest_at_least_0.10_below_nominal": bool(float(largest.raw_gate_coverage) <= nominal - 0.10),
        "standardized_bias_increases": bool(np.all(np.diff(standardized) > 0)),
        "largest_centered_KS_at_most_0.08": bool(float(largest.centered_ks_distance) <= 0.08),
    }
    checks["all_pattern_checks_pass"] = bool(all(v for k, v in checks.items() if isinstance(v, bool)))
    return checks


def make_figures(summary: pd.DataFrame, out_dir: Path, alpha: float) -> None:
    nominal = 1.0 - alpha
    ordered = summary.sort_values("n")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(ordered.n, ordered.centered_coverage, marker="o", label="Design-centered")
    ax.plot(ordered.n, ordered.raw_gate_coverage, marker="s", label="Raw GATE")
    ax.plot(ordered.n, ordered.exact_bias_corrected_coverage, marker="^", label="Exact bias-corrected")
    ax.axhline(nominal, linewidth=1.0, linestyle="--", label=f"Nominal {nominal:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Path size N")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.0, 1.01)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "boundary_coverage.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "boundary_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(ordered.n, ordered.abs_standardized_bias, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Path size N")
    ax.set_ylabel("|design bias| / standard error")
    fig.tight_layout()
    fig.savefig(out_dir / "boundary_standardized_bias.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "boundary_standardized_bias.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-values", type=parse_int_list, default=parse_int_list("128,256,512,1024"))
    ap.add_argument("--cluster-size", type=int, default=4)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--theta", type=float, default=0.2)
    ap.add_argument("--mc", type=int, default=5000)
    ap.add_argument("--design-mc", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--out", default="outputs/certified_boundary")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--assert-pattern", action="store_true")
    args = ap.parse_args()

    n_values = list(args.n_values)
    mc = int(args.mc)
    design_mc = int(args.design_mc)
    batch_size = int(args.batch_size)
    if args.quick:
        n_values = [128, 256]
        mc = min(mc, 800)
        design_mc = min(design_mc, 3000)
        batch_size = min(batch_size, 250)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float | int]] = []
    draw_frames: list[pd.DataFrame] = []
    for n in n_values:
        result, draws = run_one(
            n=n,
            cluster_size=args.cluster_size,
            p=args.p,
            beta=args.beta,
            theta=args.theta,
            mc=mc,
            design_mc=design_mc,
            batch_size=batch_size,
            alpha=args.alpha,
            seed=args.seed,
        )
        summaries.append(result)
        draw_frames.append(draws)
        print(
            f"N={n}: centered={result['centered_coverage']:.3f}, "
            f"raw={result['raw_gate_coverage']:.3f}, "
            f"debiased={result['exact_bias_corrected_coverage']:.3f}, "
            f"|bias|/SE={result['abs_standardized_bias']:.2f}"
        )

    summary = pd.DataFrame(summaries).sort_values("n")
    all_draws = pd.concat(draw_frames, ignore_index=True)
    summary.to_csv(out_dir / "boundary_summary.csv", index=False)
    with gzip.open(out_dir / "boundary_draws.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        all_draws.to_csv(handle, index=False)

    checks = pattern_checks(summary, args.alpha)
    config = vars(args).copy()
    config.update({"effective_n_values": n_values, "effective_mc": mc, "effective_design_mc": design_mc})
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "acceptance_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "INTERPRETATION.md").write_text(
        "# Interpretation\n\n"
        "This data-free experiment illustrates the paper's route-specific boundary. "
        "The fixed-cluster assignment supplies a finite-range, design-centered CLT route. "
        "With nonvanishing positive local interference, the raw-GATE bias is constant while "
        "the standard error shrinks, so raw-GATE coverage deteriorates. Exact bias correction "
        "restores causal centering. The experiment does not certify the default fixed-rank COSTA "
        "or COSTA-LocalPenalty samplers and should not be reported as evidence for their IC-2 validity.\n",
        encoding="utf-8",
    )
    make_figures(summary, out_dir, args.alpha)
    print(json.dumps(checks, indent=2, sort_keys=True))
    if args.assert_pattern and not bool(checks["all_pattern_checks_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
