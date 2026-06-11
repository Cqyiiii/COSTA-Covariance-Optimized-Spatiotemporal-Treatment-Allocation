#!/usr/bin/env python
"""Run a focused ablation of COSTA-IC regularization components.

The script compares four variants of inference-compatible COSTA under the same
focused settings used by the CLT validation script:

    1. COSTA_IC: locality + soft balance + switching guardrail;
    2. COSTA_IC_no_balance: locality + switching guardrail;
    3. COSTA_IC_no_switch: locality + soft balance;
    4. COSTA_IC_no_balance_no_switch: locality only.

For each p and variant, the script searches over lambda_local using the same
proxy criterion as run_clt_validation.py, then evaluates the selected candidate
using paired Monte Carlo repetitions across variants.
"""
from __future__ import annotations

import argparse
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
from costa.designs import optimize_costa_ic, sample_design
from costa.inference import (
    design_bias_components,
    ht_contributions,
    locality_diagnostics,
    product_graph_hac_variance,
)
from costa.naming import format_count, read_json, write_json
from costa.outcomes import (
    OutcomeParameters,
    canonical_model_name,
    generate_observed_outcomes,
    make_alpha_adjustment,
    make_baseline,
    true_tau,
)
from costa.estimators import difference_in_means, hajek_difference, weighted_ht

MODEL_SEED_ID = {"linear": 1, "nonlinear": 2, "demand": 3, "interaction_second_hop": 4}
SEED_STREAM_ID = {"assignment": 1, "outcome": 2, "baseline": 3}
DEFAULT_LAMBDA_LOCAL_GRID = "0.01,0.05,0.1,0.2,0.5,2.0,5.0"

VARIANTS = [
    {
        "variant": "COSTA_IC",
        "lambda_balance_multiplier": 1.0,
        "lambda_switch": 10.0,
        "description": "locality + soft balance + switching guardrail",
    },
    {
        "variant": "COSTA_IC_no_balance",
        "lambda_balance_multiplier": 0.0,
        "lambda_switch": 10.0,
        "description": "locality + switching guardrail; soft-balance proxy removed",
    },
    {
        "variant": "COSTA_IC_no_switch",
        "lambda_balance_multiplier": 1.0,
        "lambda_switch": 0.0,
        "description": "locality + soft balance; switching guardrail removed",
    },
    {
        "variant": "COSTA_IC_no_balance_no_switch",
        "lambda_balance_multiplier": 0.0,
        "lambda_switch": 0.0,
        "description": "locality only; soft-balance proxy and switching guardrail removed",
    },
]


def make_seed(base_seed: int, model_id: int, rep: int, stream: str) -> int:
    seq = np.random.SeedSequence([int(base_seed), int(model_id), int(rep), SEED_STREAM_ID[stream]])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def parse_list(s: str | None, cast=str) -> list[Any]:
    if s is None:
        return []
    return [cast(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def resolve_under_root(p: str | Path) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    if q.exists():
        return q.resolve()
    return (ROOT / q).resolve()


def relpath(p: str | Path) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(p)


def load_prepared(prepared_dir: str | Path):
    prepared_dir = resolve_under_root(prepared_dir)
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared directory not found: {prepared_dir}")
    if (prepared_dir / "competition_A.npz").exists():
        A = sparse.load_npz(prepared_dir / "competition_A.npz").tocsr().astype(np.float32)
    elif (prepared_dir / "competition_A.npy").exists():
        A = sparse.csr_matrix(np.load(prepared_dir / "competition_A.npy")).astype(np.float32)
    else:
        raise FileNotFoundError(f"Cannot find competition_A.npz or competition_A.npy in {prepared_dir}")
    weights = np.load(prepared_dir / "item_block_weight.npy")
    meta = read_json(prepared_dir / "metadata.json", default={})
    return A, weights, meta, prepared_dir


def get_default(cfg: dict[str, Any], key: str, default: Any) -> Any:
    flat = flatten_config(cfg)
    return flat.get(key, flat.get(key.replace("_", "-"), default))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="COSTA-IC component ablation: balance and switching guardrail.")
    ap.add_argument("--config", "--base-config", dest="config", default="configs/retailrocket_10k.yaml")
    ap.add_argument("--prepared-dir", default=None)
    ap.add_argument("--n-items", type=int, default=10000)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--p-values", default="0.2,0.5")
    ap.add_argument("--models", default="linear,nonlinear,demand")
    ap.add_argument("--mc", type=int, default=500)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--d-item", type=int, default=None)
    ap.add_argument("--d-time", default="auto")
    ap.add_argument("--lambda-local-grid", default=DEFAULT_LAMBDA_LOCAL_GRID)
    ap.add_argument("--ic-select-weight", type=float, default=1.0)
    ap.add_argument("--ic-target-far-product", type=float, default=None)
    ap.add_argument("--n-far-pairs", type=int, default=50000)
    ap.add_argument("--ci-level", type=float, default=0.90)
    ap.add_argument("--variance-mode", choices=["conservative", "signed"], default="conservative")
    ap.add_argument("--out-root", default="outputs/clt_ic_ablation")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--monitor-every", type=int, default=10)
    ap.add_argument("--no-progress", action="store_true")
    return ap


def make_params(flat: dict[str, Any], alpha: np.ndarray, beta: np.ndarray, model: str) -> OutcomeParameters:
    return OutcomeParameters(
        alpha=alpha,
        beta=beta,
        gamma=float(flat.get("gamma", 0.08)),
        eta=float(flat.get("eta", 0.10)),
        noise_sd=float(flat.get("noise_sd", 0.50)),
        model=model,
        item_shock_sd=float(flat.get("item_shock_sd", 1.00)),
        block_shock_sd=float(flat.get("block_shock_sd", 0.20)),
        nonlinear_saturation_scale=float(flat.get("nonlinear_saturation_scale", 1.5)),
        interaction_scale=float(flat.get("interaction_scale", 0.5)),
        nonlinear_lag2_scale=float(flat.get("nonlinear_lag2_scale", 0.5)),
        second_hop_scale=float(flat.get("second_hop_scale", 0.25)),
        demand_substitution_scale=float(flat.get("demand_substitution_scale", 1.0)),
        demand_treated_buffer=float(flat.get("demand_treated_buffer", 0.4)),
        demand_temporal_scale=float(flat.get("demand_temporal_scale", 3.0)),
        demand_cross_temporal_scale=float(flat.get("demand_cross_temporal_scale", 1.0)),
        demand_saturation_scale=float(flat.get("demand_saturation_scale", 0.25)),
    )


def init_metric_store() -> dict[str, list[float]]:
    return {
        "weighted_ht": [], "hajek": [], "dim": [],
        "ci_lower": [], "ci_upper": [], "ci_lower_biasaware": [], "ci_upper_biasaware": [], "se": [],
    }


def summarize_store(vals_dict: dict[str, list[float]], tau: float, ci_level: float, zcrit: float) -> dict[str, float]:
    vals = np.asarray(vals_dict["weighted_ht"], dtype=float)
    if vals.size == 0:
        return {}
    qlo, qhi = np.quantile(vals, [(1.0 - ci_level) / 2.0, 1.0 - (1.0 - ci_level) / 2.0])
    mean_est = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return {
        "mean_est": mean_est,
        "bias": float(mean_est - tau),
        "abs_bias": float(abs(mean_est - tau)),
        "std": std,
        "rmse": float(np.sqrt(np.mean((vals - tau) ** 2))),
        "mc_q05": float(qlo),
        "mc_q95": float(qhi),
        "mc_interval_width": float(qhi - qlo),
        "avg_pg_hac_se": float(np.mean(vals_dict["se"])),
        "median_pg_hac_se": float(np.median(vals_dict["se"])),
        "avg_ci_halfwidth": float(zcrit * np.mean(vals_dict["se"])),
        "ci_coverage": float(np.mean([(lo <= tau <= hi) for lo, hi in zip(vals_dict["ci_lower"], vals_dict["ci_upper"])])),
        "ci_biasaware_coverage": float(np.mean([(lo <= tau <= hi) for lo, hi in zip(vals_dict["ci_lower_biasaware"], vals_dict["ci_upper_biasaware"])])),
    }


def main() -> None:
    args = build_parser().parse_args()
    cfg_path = resolve_under_root(args.config)
    cfg = read_yaml_config(cfg_path)
    base = base_data_args(ROOT, str(Path(args.config)))
    prepared_dir = args.prepared_dir
    if prepared_dir is None:
        prepared_dir = prepared_path(ROOT, base, n_items=args.n_items, n_blocks=args.n_blocks, top_k=args.top_k)
    A, weights_raw, meta, prepared_dir = load_prepared(prepared_dir)
    n, b = weights_raw.shape
    d_time = max(1, b - 1) if str(args.d_time).lower() in {"auto", "b-1", "bminus1"} else int(args.d_time)
    d_item = int(args.d_item if args.d_item is not None else get_default(cfg, "d_item", 16))
    steps = int(args.steps if args.steps is not None else get_default(cfg, "steps", 400))
    lr = float(args.lr if args.lr is not None else get_default(cfg, "lr", 0.05))
    seed = int(args.seed if args.seed is not None else get_default(cfg, "seed", 123))
    p_values = parse_list(args.p_values, float)
    models = [canonical_model_name(x) for x in parse_list(args.models, str)]
    lambda_grid = parse_list(args.lambda_local_grid, float)
    zcrit = float(norm.ppf(0.5 + float(args.ci_level) / 2.0))

    dataset = str(meta.get("dataset", base.get("dataset", Path(prepared_dir).name.split("_top")[0])))
    out_dir = resolve_under_root(Path(args.out_root) / f"{dataset}_top{format_count(n)}_b{b}_k{meta.get('top_k', args.top_k)}_mc{args.mc}_di{d_item}_dt{d_time}")
    out_dir.mkdir(parents=True, exist_ok=True)

    flat = flatten_config(cfg)
    alpha, beta = make_baseline(
        weights_raw,
        seed=seed,
        beta_base=float(flat.get("beta_base", 0.08)),
        beta_pop_scale=float(flat.get("beta_pop_scale", 0.04)),
        beta_noise_sd=float(flat.get("beta_noise_sd", 0.12)),
        alpha_traffic_scale=float(flat.get("alpha_traffic_scale", 0.40)),
    )
    W = weights_raw.astype(float)
    if np.any(W > 0):
        W = W + float(flat.get("weight_floor_frac", 0.05)) * np.mean(W[W > 0])
    else:
        W = np.ones_like(W)

    run_config = {
        "script": "scripts/run_clt_ic_component_ablation.py",
        "config": str(Path(args.config)),
        "prepared_dir": relpath(prepared_dir),
        "out_dir": relpath(out_dir),
        "dataset": dataset,
        "N": int(n), "B": int(b), "top_k": int(meta.get("top_k", args.top_k)),
        "p_values": p_values, "models": models, "mc": int(args.mc), "ci_level": float(args.ci_level),
        "steps": steps, "lr": lr, "d_item": d_item, "d_time": d_time,
        "lambda_local_grid": lambda_grid,
        "variants": [{k: v for k, v in spec.items() if k != "description"} | {"description": spec["description"]} for spec in VARIANTS],
        "base_seed": seed,
    }
    write_json(out_dir / "ic_component_ablation_run_config.json", run_config)

    all_draws: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    all_traj: list[pd.DataFrame] = []

    for p in p_values:
        selected_by_variant: dict[str, dict[str, Any]] = {}
        for spec in VARIANTS:
            variant = spec["variant"]
            print(f"Optimizing {variant} for p={p}...")
            candidates = []
            for lam in lambda_grid:
                lambda_balance = float(flat.get("lambda_balance", 0.05)) * float(spec["lambda_balance_multiplier"])
                cand = optimize_costa_ic(
                    A, p=p, n_blocks=b, d_item=d_item, d_time=d_time, steps=steps, lr=lr,
                    w_comp=float(flat.get("w_comp", 1.0)), w_temp=float(flat.get("w_temp", 1.0)),
                    lambda_var=float(flat.get("lambda_var", 0.05)),
                    lambda_balance=lambda_balance,
                    switch_min=float(flat.get("switch_min", 0.10)),
                    lambda_switch=float(spec["lambda_switch"]),
                    lambda_local=float(lam), n_far_pairs=int(args.n_far_pairs), seed=seed,
                    monitor_every=max(1, int(args.monitor_every)), show_progress=not args.no_progress,
                )
                cand.name = "COSTA_IC"
                loc = locality_diagnostics(cand, p=p, n_pairs=min(args.n_far_pairs, 50000), seed=seed + 11)
                last = dict(cand.objective_history[-1]) if cand.objective_history else {}
                score = float(last.get("mse_objective", np.inf)) + float(args.ic_select_weight) * float(loc.get("far_product_mean_abs_R", np.inf))
                if args.ic_target_far_product is not None and loc.get("far_product_mean_abs_R", np.inf) <= float(args.ic_target_far_product):
                    score = float(last.get("mse_objective", np.inf))
                candidates.append({"score": score, "design": cand, "lambda_local": float(lam), "loc": loc, "last": last, "lambda_balance": lambda_balance})
                search_rows.append({
                    "dataset": dataset, "p": p, "variant": variant, "lambda_local": float(lam),
                    "lambda_balance": lambda_balance, "lambda_switch": float(spec["lambda_switch"]),
                    "selection_score": score,
                    "final_mse_objective": float(last.get("mse_objective", np.nan)),
                    "regularized_loss": float(last.get("regularized_loss", np.nan)),
                    "balance_proxy": float(last.get("balance_proxy", np.nan)),
                    "switch_guard": float(last.get("switch_guard", np.nan)),
                    "far_product_mean_abs_R": float(loc.get("far_product_mean_abs_R", np.nan)),
                    "far_item_mean_abs_R": float(loc.get("far_item_mean_abs_R", np.nan)),
                    "far_time_mean_abs_R": float(loc.get("far_time_mean_abs_R", np.nan)),
                })
                if cand.objective_history:
                    tmp = pd.DataFrame(cand.objective_history)
                    tmp["p"] = p
                    tmp["variant"] = variant
                    tmp["candidate"] = f"{variant}:lambda_local={lam:g}"
                    all_traj.append(tmp)
            candidates.sort(key=lambda x: x["score"])
            selected_by_variant[variant] = candidates[0]
            print(f"Selected {variant} lambda_local={candidates[0]['lambda_local']:g} for p={p}.")

        for row in search_rows:
            if row["p"] == p:
                sel = selected_by_variant.get(row["variant"])
                row["selected_by_proxy"] = bool(sel is not None and np.isclose(float(row["lambda_local"]), float(sel["lambda_local"])))

        for model in models:
            model_id = MODEL_SEED_ID.get(model, 9)
            params = make_params(flat, alpha, beta, model)
            tau = true_tau(A, params, weights=W, burn_in=1)
            eval_entries = []
            for spec in VARIANTS:
                variant = spec["variant"]
                sel = selected_by_variant[variant]
                const = design_bias_components(sel["design"], A, W, p=p, gamma=params.gamma, eta=params.eta, burn_in=1)
                eval_entries.append({
                    "variant": variant,
                    "description": spec["description"],
                    "design": sel["design"],
                    "lambda_local": float(sel["lambda_local"]),
                    "lambda_balance": float(sel["lambda_balance"]),
                    "lambda_switch": float(spec["lambda_switch"]),
                    "loc": sel["loc"],
                    "last": sel["last"],
                    "const": const,
                    "store": init_metric_store(),
                })

            iterator = range(int(args.mc))
            if not args.no_progress:
                iterator = tqdm(iterator, desc=f"IC component MC p={p} model={model}", unit="rep", leave=False)
            for rep in iterator:
                assignment_seed = make_seed(seed, model_id, rep, "assignment")
                outcome_seed = make_seed(seed, model_id, rep, "outcome")
                baseline_seed = make_seed(seed, model_id, rep, "baseline")
                alpha_hat = make_alpha_adjustment(
                    alpha, seed=baseline_seed,
                    item_sd=float(flat.get("alpha_hat_item_sd", 0.90)),
                    block_sd=float(flat.get("alpha_hat_block_sd", 0.25)),
                    cell_sd=float(flat.get("alpha_hat_cell_sd", 0.45)),
                )
                for entry in eval_entries:
                    design = entry["design"]
                    Z = sample_design(design, n, b, p, seed=assignment_seed)
                    Y = generate_observed_outcomes(Z, A, params, seed=outcome_seed)
                    ht = weighted_ht(Z, Y, alpha=alpha_hat, p=p, weights=W, burn_in=1)
                    hj = hajek_difference(Z, Y, alpha=alpha_hat, weights=W, burn_in=1)
                    dm = difference_in_means(Z, Y, alpha=alpha_hat, burn_in=1)
                    ht2, xi, Wtot = ht_contributions(Z, Y, alpha_hat, p=p, weights=W, burn_in=1)
                    vh = product_graph_hac_variance(xi, A, Wtot, burn_in=1, mode=args.variance_mode)
                    se = float(vh["se"])
                    bias_bound = float(entry["const"].get("bias_bound", 0.0))
                    ci_lower = float(ht2 - zcrit * se)
                    ci_upper = float(ht2 + zcrit * se)
                    ci_lower_ba = float(ht2 - zcrit * se - bias_bound)
                    ci_upper_ba = float(ht2 + zcrit * se + bias_bound)
                    store = entry["store"]
                    store["weighted_ht"].append(float(ht))
                    store["hajek"].append(float(hj))
                    store["dim"].append(float(dm))
                    store["ci_lower"].append(ci_lower)
                    store["ci_upper"].append(ci_upper)
                    store["ci_lower_biasaware"].append(ci_lower_ba)
                    store["ci_upper_biasaware"].append(ci_upper_ba)
                    store["se"].append(se)
                    all_draws.append({
                        "dataset": dataset, "p": p, "model": model, "model_seed_id": model_id,
                        "rep": rep, "variant": entry["variant"], "lambda_local": entry["lambda_local"],
                        "lambda_balance": entry["lambda_balance"], "lambda_switch": entry["lambda_switch"],
                        "tau_true": float(tau), "weighted_ht": float(ht), "hajek": float(hj), "dim": float(dm),
                        "weighted_ht_error": float(ht - tau), "pg_hac_se": se,
                        "ci_lower": ci_lower, "ci_upper": ci_upper,
                        "ci_lower_biasaware": ci_lower_ba, "ci_upper_biasaware": ci_upper_ba,
                        "ci_covers_tau": bool(ci_lower <= tau <= ci_upper),
                        "ci_biasaware_covers_tau": bool(ci_lower_ba <= tau <= ci_upper_ba),
                        "base_seed": seed, "assignment_seed": assignment_seed, "outcome_seed": outcome_seed, "baseline_seed": baseline_seed,
                    })

            for entry in eval_entries:
                summ = summarize_store(entry["store"], tau=tau, ci_level=float(args.ci_level), zcrit=zcrit)
                row = {
                    "dataset": dataset, "p": p, "model": model, "variant": entry["variant"],
                    "description": entry["description"], "estimator": "weighted_ht", "tau_true": float(tau), **summ,
                    "lambda_local": entry["lambda_local"], "lambda_balance": entry["lambda_balance"], "lambda_switch": entry["lambda_switch"],
                    "bias_bound": float(entry["const"].get("bias_bound", np.nan)),
                    "design_linear_bias_formula": float(entry["const"].get("bias_linear", np.nan)),
                    "N": n, "B": b, "top_k": int(meta.get("top_k", args.top_k)), "mc": int(args.mc),
                    "d_item": d_item, "d_time": d_time, "ci_level": float(args.ci_level),
                }
                row.update(entry["loc"])
                row.update({f"opt_{k}": v for k, v in entry["last"].items() if k in {"mse_objective", "regularized_loss", "balance_proxy", "switch_guard", "locality_proxy"}})
                all_summary.append(row)

    draws = pd.DataFrame(all_draws)
    summary = pd.DataFrame(all_summary)
    search = pd.DataFrame(search_rows)
    traj = pd.concat(all_traj, ignore_index=True) if all_traj else pd.DataFrame()

    if not summary.empty:
        order = {spec["variant"]: i for i, spec in enumerate(VARIANTS)}
        summary["variant_order"] = summary["variant"].map(order).fillna(99)
        summary = summary.sort_values(["dataset", "p", "model", "variant_order"]).drop(columns=["variant_order"])
    if not search.empty:
        search = search.sort_values(["dataset", "p", "variant", "lambda_local"])

    draws.to_csv(out_dir / "ic_component_draws.csv", index=False)
    summary.to_csv(out_dir / "ic_component_summary.csv", index=False)
    search.to_csv(out_dir / "ic_component_lambda_search.csv", index=False)
    if not traj.empty:
        traj.to_csv(out_dir / "ic_component_optimization_trajectory.csv", index=False)

    display_cols = ["dataset", "p", "model", "variant", "lambda_local", "mean_est", "bias", "std", "rmse", "ci_coverage", "ci_biasaware_coverage"]
    print(summary[display_cols])
    print(f"Saved COSTA-IC component ablation outputs to {relpath(out_dir)}")


if __name__ == "__main__":
    main()
