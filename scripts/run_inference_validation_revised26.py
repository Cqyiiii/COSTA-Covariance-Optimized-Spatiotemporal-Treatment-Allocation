#!/usr/bin/env python3
"""Supplementary inference validation for the revised COSTA manuscript.

The primary specification uses the linear outcome model, one fixed
pre-randomization baseline adjustment, and independent own-cell noise. It compares:

1. the legacy globally demeaned absolute product-graph HAC;
2. signed local HAC centered by independently calibrated heterogeneous means;
3. that local HAC plus a signed design-aware covariance tail; and
4. that local HAC plus the positive part of the design-aware tail.

The calibration streams are independent of the main Monte Carlo stream.  The
script reports coverage around the design expectation, raw-GATE coverage, and
coverage after model-based bias correction.  Existing point-estimation/RMSE
experiments do not need to be rerun.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from tqdm.auto import tqdm

from costa.ablation import base_data_args, prepared_path
from costa.config import flatten_config, read_yaml_config
from costa.designs import OptimizedDesign, optimize_costa, optimize_costa_ic, sample_design
from costa.inference import locality_diagnostics
from costa.inference_revised26 import (
    ProductGraphPairs,
    calibrate_mean,
    calibrate_tail,
    global_mean_residuals,
    interval,
    stabilized_contributions,
)
from costa.naming import format_count, read_json, write_json
from costa.outcomes import (
    OutcomeParameters,
    canonical_model_name,
    deterministic_mean,
    generate_observed_outcomes,
    make_alpha_adjustment,
    make_baseline,
    true_tau,
)

MODEL_SEED_ID = {"linear": 1, "nonlinear": 2, "demand": 3, "interaction_second_hop": 4}
STREAM_ID = {
    "main_assignment": 1,
    "main_outcome": 2,
    "main_baseline": 3,
    "mean_assignment": 4,
    "tail_assignment": 5,
    "tail_outcome": 6,
}
METHOD_ORDER = [
    "legacy_global_abs",
    "heterogeneous_local",
    "design_aware_signed",
    "design_aware_positive",
    "plugin_full_diagnostic",
]


def make_seed(base_seed: int, model_id: int, design_id: int, rep: int, stream: str) -> int:
    if stream not in STREAM_ID:
        raise ValueError(f"Unknown seed stream: {stream}")
    seq = np.random.SeedSequence(
        [int(base_seed), int(model_id), int(design_id), int(rep), int(STREAM_ID[stream])]
    )
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def parse_list(value: str | None, cast=str) -> list[Any]:
    if value is None:
        return []
    return [cast(x.strip()) for x in str(value).replace(";", ",").split(",") if x.strip()]


def resolve_under_root(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (ROOT / p).resolve()


def relpath(value: str | Path) -> str:
    p = Path(value)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(p)


def get_default(flat: dict[str, Any], key: str, default: Any) -> Any:
    return flat.get(key, flat.get(key.replace("_", "-"), default))


def load_prepared(prepared_dir: str | Path):
    prepared_dir = resolve_under_root(prepared_dir)
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared directory not found: {prepared_dir}")
    sparse_path = prepared_dir / "competition_A.npz"
    dense_path = prepared_dir / "competition_A.npy"
    if sparse_path.exists():
        A = sparse.load_npz(sparse_path).tocsr().astype(np.float32)
    elif dense_path.exists():
        A = sparse.csr_matrix(np.load(dense_path)).astype(np.float32)
    else:
        raise FileNotFoundError(f"Cannot find competition_A.npz or competition_A.npy in {prepared_dir}")
    weights = np.load(prepared_dir / "item_block_weight.npy")
    metadata = read_json(prepared_dir / "metadata.json", default={})
    return A, weights, metadata, prepared_dir


def resolve_d_time(raw: Any, n_blocks: int) -> int:
    key = str(raw).strip().lower().replace("_", "-")
    if key in {"", "none", "auto", "b-1", "bminus1", "b-minus-1"}:
        return max(1, int(n_blocks) - 1)
    return int(float(raw))


def parse_ic_lambda(name: str) -> float:
    if name in {"COSTA_IC", "COSTA-LocalPenalty", "COSTA_LP"}:
        return 1.0
    marker = "lam"
    if marker not in name:
        return 1.0
    raw = name.split(marker, 1)[1].replace("p", ".")
    return float(raw)


def canonical_design_name(name: str) -> str:
    aliases = {
        "independent": "independent",
        "bernoulli": "independent",
        "rbsd": "rbsd_approx",
        "rbsd_approx": "rbsd_approx",
        "costa": "COSTA",
        "COSTA": "COSTA",
        "costa_ic": "COSTA_IC_lam1",
        "costa-lp": "COSTA_IC_lam1",
        "COSTA_IC": "COSTA_IC_lam1",
        "COSTA-LocalPenalty": "COSTA_IC_lam1",
        "COSTA_LP": "COSTA_IC_lam1",
    }
    if name in aliases:
        return aliases[name]
    lower = name.lower()
    if lower.startswith("costa_ic_lam") or lower.startswith("costa-ic-lam"):
        return "COSTA_IC_lam" + name.lower().split("lam", 1)[1]
    if name.startswith("COSTA_IC_lam"):
        return name
    raise ValueError(f"Unknown design: {name}")


def make_params(
    flat: dict[str, Any],
    alpha: np.ndarray,
    beta: np.ndarray,
    model: str,
    noise_sd: float | None,
    item_shock_sd: float,
    block_shock_sd: float,
) -> OutcomeParameters:
    return OutcomeParameters(
        alpha=alpha,
        beta=beta,
        gamma=float(get_default(flat, "gamma", 0.08)),
        eta=float(get_default(flat, "eta", 0.10)),
        noise_sd=float(get_default(flat, "noise_sd", 0.50) if noise_sd is None else noise_sd),
        model=model,
        item_shock_sd=float(item_shock_sd),
        block_shock_sd=float(block_shock_sd),
        nonlinear_saturation_scale=float(get_default(flat, "nonlinear_saturation_scale", 1.5)),
        interaction_scale=float(get_default(flat, "interaction_scale", 0.5)),
        nonlinear_lag2_scale=float(get_default(flat, "nonlinear_lag2_scale", 0.5)),
        second_hop_scale=float(get_default(flat, "second_hop_scale", 0.25)),
        demand_substitution_scale=float(get_default(flat, "demand_substitution_scale", 1.0)),
        demand_treated_buffer=float(get_default(flat, "demand_treated_buffer", 0.4)),
        demand_temporal_scale=float(get_default(flat, "demand_temporal_scale", 3.0)),
        demand_cross_temporal_scale=float(get_default(flat, "demand_cross_temporal_scale", 1.0)),
        demand_saturation_scale=float(get_default(flat, "demand_saturation_scale", 0.25)),
    )


def build_designs(
    names: list[str],
    A,
    p: float,
    n_blocks: int,
    d_item: int,
    d_time: int,
    steps: int,
    lr: float,
    flat: dict[str, Any],
    n_far_pairs: int,
    locality_time_gap: int,
    seed: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for design_id, raw_name in enumerate(names):
        name = canonical_design_name(raw_name)
        if name == "independent":
            design = OptimizedDesign(name="independent", p=float(p))
            label = "Independent"
            lam = np.nan
        elif name == "rbsd_approx":
            design = OptimizedDesign(name="rbsd_approx", p=float(p))
            label = "RBSD"
            lam = np.nan
        elif name == "COSTA":
            design = optimize_costa(
                A,
                p=float(p),
                n_blocks=int(n_blocks),
                d_item=int(d_item),
                d_time=int(d_time),
                steps=int(steps),
                lr=float(lr),
                w_comp=float(get_default(flat, "w_comp", 1.0)),
                w_temp=float(get_default(flat, "w_temp", 1.0)),
                lambda_var=float(get_default(flat, "lambda_var", 0.05)),
                lambda_balance=float(get_default(flat, "lambda_balance", 0.05)),
                switch_min=float(get_default(flat, "switch_min", 0.10)),
                seed=int(seed),
                monitor_every=max(1, int(steps) // 40),
                show_progress=show_progress,
            )
            label = "COSTA"
            lam = np.nan
        elif name.startswith("COSTA_IC_lam"):
            lam = parse_ic_lambda(name)
            design = optimize_costa_ic(
                A,
                p=float(p),
                n_blocks=int(n_blocks),
                d_item=int(d_item),
                d_time=int(d_time),
                steps=int(steps),
                lr=float(lr),
                w_comp=float(get_default(flat, "w_comp", 1.0)),
                w_temp=float(get_default(flat, "w_temp", 1.0)),
                lambda_var=float(get_default(flat, "lambda_var", 0.05)),
                lambda_balance=float(get_default(flat, "lambda_balance", 0.05)),
                switch_min=float(get_default(flat, "switch_min", 0.10)),
                lambda_local=float(lam),
                n_far_pairs=int(n_far_pairs),
                locality_time_gap=int(locality_time_gap),
                seed=int(seed),
                monitor_every=max(1, int(steps) // 40),
                show_progress=show_progress,
            )
            design.name = name
            label = "COSTA-LocalPenalty" if abs(float(lam) - 1.0) < 1e-12 else f"COSTA-LocalPenalty({lam:g})"
        else:  # pragma: no cover
            raise AssertionError(name)
        loc = locality_diagnostics(design, p=float(p), n_pairs=min(int(n_far_pairs), 50_000), seed=int(seed + 17))
        result.append({
            "design_id": design_id,
            "internal_name": name,
            "design": design,
            "label": label,
            "lambda_local": lam,
            "locality": loc,
        })
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/retailrocket_10k.yaml")
    ap.add_argument("--prepared-dir", default=None)
    ap.add_argument("--n-items", type=int, default=10_000)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--p-values", default="0.5")
    ap.add_argument("--models", default="linear")
    ap.add_argument("--designs", default="independent,rbsd_approx,COSTA,COSTA_IC_lam1")
    ap.add_argument("--mc", type=int, default=1000)
    ap.add_argument("--mean-calibration-mc", type=int, default=500)
    ap.add_argument("--tail-calibration-mc", type=int, default=500)
    ap.add_argument("--pair-batch", type=int, default=10_000,
                    help="Pair batch for tail covariance; 10k limits peak memory at N=10k.")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--d-item", type=int, default=None)
    ap.add_argument("--d-time", default="auto")
    ap.add_argument("--n-far-pairs", type=int, default=50_000)
    ap.add_argument("--locality-time-gap", type=int, default=2)
    ap.add_argument("--burn-in", type=int, default=1)
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument("--noise-sd", type=float, default=None)
    ap.add_argument(
        "--tail-outcome-mode", choices=["conditional_mean", "oracle_draw"],
        default="conditional_mean",
        help="Primary run uses the pre-randomization conditional-mean plug-in; oracle_draw is a simulation benchmark.",
    )
    ap.add_argument("--item-shock-sd", type=float, default=0.0,
                    help="Primary theory-aligned run uses 0; nonzero values are robustness only.")
    ap.add_argument("--block-shock-sd", type=float, default=0.0,
                    help="Primary theory-aligned run uses 0; nonzero values are robustness only.")
    ap.add_argument("--alpha-hat-item-sd", type=float, default=0.0)
    ap.add_argument("--alpha-hat-block-sd", type=float, default=0.0)
    ap.add_argument("--alpha-hat-cell-sd", type=float, default=0.0)
    ap.add_argument("--weight-floor-frac", type=float, default=None)
    ap.add_argument("--out-root", default="outputs/inference_revised26")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="Use small Monte Carlo sizes for an end-to-end smoke run.")
    return ap


def summarize_draws(draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "p", "model", "design", "internal_design", "method", "ci_level"]
    for key, g in draws.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, key))
        est = g["tau_hat"].to_numpy(float)
        mc_sd = float(np.std(est, ddof=1)) if len(est) > 1 else np.nan
        se = g["se"].to_numpy(float)
        z_design = g["z_design_center"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        z_gate = g["z_raw_gate"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        z_bc = g["z_bias_corrected"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)

        def quantiles(x: np.ndarray) -> tuple[float, float, float]:
            if x.size == 0:
                return np.nan, np.nan, np.nan
            return tuple(float(v) for v in np.quantile(x, [0.025, 0.5, 0.975]))

        dz = quantiles(z_design)
        gz = quantiles(z_gate)
        bz = quantiles(z_bc)
        row.update({
            "mc": int(len(g)),
            "tau_gate": float(g["tau_gate"].iloc[0]),
            "design_center": float(g["design_center"].iloc[0]),
            "design_bias": float(g["design_bias"].iloc[0]),
            "mean_estimate": float(np.mean(est)),
            "mc_sd": mc_sd,
            "avg_se": float(np.mean(se)),
            "se_to_mc_sd": float(np.mean(se) / mc_sd) if np.isfinite(mc_sd) and mc_sd > 0 else np.nan,
            "avg_width": float(np.mean(g["width"])),
            "coverage_design_center": float(np.mean(g["covers_design_center"])),
            "coverage_raw_gate": float(np.mean(g["covers_raw_gate"])),
            "coverage_bias_corrected_gate": float(np.mean(g["covers_bias_corrected_gate"])),
            "coverage_design_center_mc_se": float(
                np.sqrt(np.mean(g["covers_design_center"]) * (1.0 - np.mean(g["covers_design_center"])) / len(g))
            ),
            "variance_truncation_rate": float(np.mean(g["variance_truncated"])),
            "avg_variance_raw": float(np.mean(g["variance_raw"])),
            "design_z_mean": float(np.mean(z_design)) if z_design.size else np.nan,
            "design_z_sd": float(np.std(z_design, ddof=1)) if z_design.size > 1 else np.nan,
            "design_z_q025": dz[0], "design_z_q50": dz[1], "design_z_q975": dz[2],
            "gate_z_q025": gz[0], "gate_z_q50": gz[1], "gate_z_q975": gz[2],
            "bias_corrected_z_q025": bz[0], "bias_corrected_z_q50": bz[1], "bias_corrected_z_q975": bz[2],
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    out["method_order"] = out["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    return out.sort_values(["dataset", "p", "model", "design", "method_order"]).drop(columns="method_order")


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.mc = min(int(args.mc), 80)
        args.mean_calibration_mc = min(int(args.mean_calibration_mc), 60)
        args.tail_calibration_mc = min(int(args.tail_calibration_mc), 60)
        args.steps = min(int(args.steps or 40), 40)
        args.n_far_pairs = min(int(args.n_far_pairs), 5_000)

    cfg_path = resolve_under_root(args.config)
    cfg = read_yaml_config(cfg_path)
    flat = flatten_config(cfg)
    base = base_data_args(ROOT, str(Path(args.config)))
    prepared_dir = args.prepared_dir
    if prepared_dir is None:
        prepared_dir = prepared_path(
            ROOT, base, n_items=int(args.n_items), n_blocks=int(args.n_blocks), top_k=int(args.top_k)
        )
    A, weights_raw, metadata, prepared_dir = load_prepared(prepared_dir)
    n, B = weights_raw.shape
    burn_in = int(args.burn_in)
    if burn_in >= B:
        raise ValueError("burn_in must be less than the number of blocks")

    d_item = int(args.d_item if args.d_item is not None else get_default(flat, "d_item", 16))
    d_time = resolve_d_time(args.d_time, B)
    steps = int(args.steps if args.steps is not None else get_default(flat, "steps", 400))
    lr = float(args.lr if args.lr is not None else get_default(flat, "lr", 0.05))
    seed = int(args.seed if args.seed is not None else get_default(flat, "seed", 123))
    p_values = parse_list(args.p_values, float)
    models = [canonical_model_name(x) for x in parse_list(args.models, str)]
    design_names = parse_list(args.designs, str)
    zcrit = float(norm.ppf(0.5 + float(args.ci_level) / 2.0))

    dataset = str(metadata.get("dataset", base.get("dataset", Path(prepared_dir).name.split("_top")[0])))
    out_dir = resolve_under_root(
        Path(args.out_root)
        / f"{dataset}_top{format_count(n)}_b{B}_k{metadata.get('top_k', args.top_k)}_mc{args.mc}_mean{args.mean_calibration_mc}_tail{args.tail_calibration_mc}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha, beta = make_baseline(
        weights_raw,
        seed=seed,
        beta_base=float(get_default(flat, "beta_base", 0.08)),
        beta_pop_scale=float(get_default(flat, "beta_pop_scale", 0.04)),
        beta_noise_sd=float(get_default(flat, "beta_noise_sd", 0.12)),
        alpha_traffic_scale=float(get_default(flat, "alpha_traffic_scale", 0.40)),
    )
    W = weights_raw.astype(np.float64)
    floor_frac = float(
        get_default(flat, "weight_floor_frac", 0.05) if args.weight_floor_frac is None else args.weight_floor_frac
    )
    if np.any(W > 0):
        W = W + floor_frac * float(np.mean(W[W > 0]))
    else:
        W = np.ones_like(W)

    # The adjustment is part of the pre-randomization environment and is held
    # fixed over the randomization repetitions and across design comparisons.
    fixed_baseline_seed = make_seed(seed, 0, 0, 0, "main_baseline")
    alpha_hat = make_alpha_adjustment(
        alpha,
        seed=fixed_baseline_seed,
        item_sd=float(args.alpha_hat_item_sd),
        block_sd=float(args.alpha_hat_block_sd),
        cell_sd=float(args.alpha_hat_cell_sd),
    )

    pairs = ProductGraphPairs.from_graph(A, n_blocks=B, burn_in=burn_in)
    all_draws: list[dict[str, Any]] = []
    all_calibration: list[dict[str, Any]] = []
    all_designs: list[dict[str, Any]] = []

    for p_index, p in enumerate(p_values):
        print(f"Building designs for p={p:g} ...")
        designs = build_designs(
            design_names,
            A=A,
            p=p,
            n_blocks=B,
            d_item=d_item,
            d_time=d_time,
            steps=steps,
            lr=lr,
            flat=flat,
            n_far_pairs=int(args.n_far_pairs),
            locality_time_gap=int(args.locality_time_gap),
            seed=seed + 1009 * p_index,
            show_progress=not args.no_progress,
        )
        for entry in designs:
            row = {
                "dataset": dataset,
                "p": p,
                "design": entry["label"],
                "internal_design": entry["internal_name"],
                "lambda_local": entry["lambda_local"],
            }
            row.update(entry["locality"])
            all_designs.append(row)

        for model in models:
            model_id = MODEL_SEED_ID.get(model, 99)
            params = make_params(
                flat,
                alpha=alpha,
                beta=beta,
                model=model,
                noise_sd=args.noise_sd,
                item_shock_sd=float(args.item_shock_sd),
                block_shock_sd=float(args.block_shock_sd),
            )
            tau_gate = float(true_tau(A, params, weights=W, burn_in=burn_in))

            for entry in designs:
                design = entry["design"]
                design_id = int(entry["design_id"] + 100 * p_index)
                label = entry["label"]
                print(f"Calibrating {dataset}, p={p:g}, model={model}, design={label} ...")

                def draw_mean(assignment_seed: int) -> np.ndarray:
                    Z = sample_design(design, n, B, p, seed=int(assignment_seed))
                    Y_mean = deterministic_mean(Z, A, params)
                    _, Q, _ = stabilized_contributions(
                        Z, Y_mean, alpha_hat, p=p, weights=W, burn_in=burn_in
                    )
                    return Q

                def draw_tail(assignment_seed: int, outcome_seed: int) -> np.ndarray:
                    Z = sample_design(design, n, B, p, seed=int(assignment_seed))
                    if args.tail_outcome_mode == "oracle_draw":
                        Y_plugin = generate_observed_outcomes(Z, A, params, seed=int(outcome_seed))
                    else:
                        Y_plugin = deterministic_mean(Z, A, params)
                    _, Q, _ = stabilized_contributions(
                        Z, Y_plugin, alpha_hat, p=p, weights=W, burn_in=burn_in
                    )
                    return Q

                mean_seed = make_seed(seed, model_id, design_id, 0, "mean_assignment")
                tail_assignment_seed = make_seed(seed, model_id, design_id, 0, "tail_assignment")
                tail_outcome_seed = make_seed(seed, model_id, design_id, 0, "tail_outcome")
                mean_cal = calibrate_mean(
                    draw_mean,
                    n_draws=int(args.mean_calibration_mc),
                    seed=mean_seed,
                )
                tail_cal = calibrate_tail(
                    draw_tail,
                    pairs=pairs,
                    n_draws=int(args.tail_calibration_mc),
                    assignment_seed=tail_assignment_seed,
                    outcome_seed=tail_outcome_seed,
                    pair_batch=int(args.pair_batch),
                )
                design_bias = float(mean_cal.design_center - tau_gate)
                all_calibration.append({
                    "dataset": dataset,
                    "p": p,
                    "model": model,
                    "design": label,
                    "internal_design": entry["internal_name"],
                    "lambda_local": entry["lambda_local"],
                    "tau_gate": tau_gate,
                    "design_center": mean_cal.design_center,
                    "design_bias": design_bias,
                    "mean_calibration_mc": mean_cal.n_draws,
                    "mean_calibration_mc_se": mean_cal.mc_se_design_center,
                    "tail_calibration_mc": tail_cal.n_draws,
                    "design_mc_full_variance": tail_cal.full_variance,
                    "design_mc_local_variance": tail_cal.local_variance,
                    "design_mc_signed_tail": tail_cal.signed_tail,
                    "design_mc_positive_tail": tail_cal.positive_tail,
                    "design_mc_full_se": tail_cal.full_se,
                })

                iterator = range(int(args.mc))
                if not args.no_progress:
                    iterator = tqdm(iterator, desc=f"Inference MC {label} {model}", unit="rep", leave=False)
                for rep in iterator:
                    assignment_seed = make_seed(seed, model_id, 0, rep, "main_assignment")
                    outcome_seed = make_seed(seed, model_id, 0, rep, "main_outcome")
                    baseline_seed = fixed_baseline_seed
                    Z = sample_design(design, n, B, p, seed=assignment_seed)
                    Y = generate_observed_outcomes(Z, A, params, seed=outcome_seed)
                    tau_hat, Q, omega = stabilized_contributions(
                        Z, Y, alpha_hat, p=p, weights=W, burn_in=burn_in
                    )

                    legacy_resid = global_mean_residuals(Q, omega, tau_hat)
                    legacy = pairs.estimator_variance(legacy_resid, mode="absolute")
                    hetero_resid = Q - mean_cal.qbar
                    hetero = pairs.estimator_variance(hetero_resid, mode="signed")

                    method_variances = {
                        "legacy_global_abs": float(legacy["variance_raw"]),
                        "heterogeneous_local": float(hetero["variance_raw"]),
                        "design_aware_signed": float(hetero["variance_raw"] + tail_cal.signed_tail),
                        "design_aware_positive": float(hetero["variance_raw"] + tail_cal.positive_tail),
                        "plugin_full_diagnostic": float(tail_cal.full_variance),
                    }
                    tau_bias_corrected = float(tau_hat - design_bias)
                    for method, variance_raw in method_variances.items():
                        low, high, se, truncated = interval(tau_hat, variance_raw, zcrit)
                        bc_low, bc_high, _, _ = interval(tau_bias_corrected, variance_raw, zcrit)
                        width = float(high - low)
                        z_design = (tau_hat - mean_cal.design_center) / se if se > 0 else np.nan
                        z_gate = (tau_hat - tau_gate) / se if se > 0 else np.nan
                        z_bc = (tau_bias_corrected - tau_gate) / se if se > 0 else np.nan
                        all_draws.append({
                            "dataset": dataset,
                            "p": p,
                            "model": model,
                            "design": label,
                            "internal_design": entry["internal_name"],
                            "lambda_local": entry["lambda_local"],
                            "rep": rep,
                            "method": method,
                            "ci_level": float(args.ci_level),
                            "tau_gate": tau_gate,
                            "design_center": mean_cal.design_center,
                            "design_bias": design_bias,
                            "tau_hat": tau_hat,
                            "tau_bias_corrected": tau_bias_corrected,
                            "variance_raw": variance_raw,
                            "variance": max(variance_raw, 0.0),
                            "variance_truncated": truncated,
                            "se": se,
                            "width": width,
                            "covers_design_center": bool(low <= mean_cal.design_center <= high),
                            "covers_raw_gate": bool(low <= tau_gate <= high),
                            "covers_bias_corrected_gate": bool(bc_low <= tau_gate <= bc_high),
                            "z_design_center": z_design,
                            "z_raw_gate": z_gate,
                            "z_bias_corrected": z_bc,
                            "assignment_seed": assignment_seed,
                            "outcome_seed": outcome_seed,
                            "baseline_seed": baseline_seed,
                        })

    draws = pd.DataFrame(all_draws)
    calibration = pd.DataFrame(all_calibration)
    design_diagnostics = pd.DataFrame(all_designs)
    summary = summarize_draws(draws)
    calibration_for_merge = calibration[[
        "dataset", "p", "model", "design", "internal_design",
        "mean_calibration_mc_se", "design_mc_full_variance",
        "design_mc_local_variance", "design_mc_signed_tail", "design_mc_positive_tail",
    ]].copy()
    summary = summary.merge(
        calibration_for_merge,
        on=["dataset", "p", "model", "design", "internal_design"],
        how="left",
        validate="many_to_one",
    )
    summary["mean_calibration_se_to_mc_sd"] = (
        summary["mean_calibration_mc_se"] / summary["mc_sd"].replace(0.0, np.nan)
    )

    with gzip.open(out_dir / "inference_revised26_draws.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        draws.to_csv(handle, index=False)
    summary.to_csv(out_dir / "inference_revised26_summary.csv", index=False)
    calibration.to_csv(out_dir / "inference_revised26_calibration.csv", index=False)
    design_diagnostics.to_csv(out_dir / "inference_revised26_design_diagnostics.csv", index=False)

    run_config = vars(args).copy()
    run_config.update({
        "script": "scripts/run_inference_validation_revised26.py",
        "config": relpath(cfg_path),
        "prepared_dir": relpath(prepared_dir),
        "out_dir": relpath(out_dir),
        "dataset": dataset,
        "N": int(n),
        "B": int(B),
        "top_k": int(metadata.get("top_k", args.top_k)),
        "d_item": d_item,
        "d_time": d_time,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "calibration_independent_of_main_mc": True,
        "fixed_baseline_seed": int(fixed_baseline_seed),
        "primary_scope": "linear DGP; fixed pre-randomization adjustment; independent own-cell noise",
        "tail_plugin_scope": (
            "pre-randomization conditional-mean surface"
            if args.tail_outcome_mode == "conditional_mean"
            else "oracle semi-synthetic outcome draw"
        ),
    })
    write_json(out_dir / "inference_revised26_run_config.json", run_config)
    print(summary[[
        "dataset", "design", "method", "coverage_design_center", "coverage_raw_gate",
        "coverage_bias_corrected_gate", "avg_width", "se_to_mc_sd"
    ]].to_string(index=False))
    print(f"Saved revised inference outputs to {relpath(out_dir)}")


if __name__ == "__main__":
    main()
