#!/usr/bin/env python
"""CLT validation with design-aware product-graph HAC.

This script is a focused add-on to ``run_clt_validation.py``.  It compares
standard COSTA and COSTA-IC under the same focused settings used for the CLT
illustration, but reports both:

1. the original product-graph HAC standard error; and
2. a design-aware product-graph HAC standard error that augments the local HAC
   with a design-implied far-dependence correction.

The design-aware correction uses the fact that COSTA fixes the randomization
sampler before the experiment.  For each optimized design we draw assignments
from that sampler, construct score draws

    S_s(a) = w_a psi_a(Z_a^{(s)}) / W_N,

and use them to estimate the full fixed-outcome design variance and the local
product-graph part.  For each Monte Carlo repetition, with observed residualized
outcome X_a = Y_a^{obs} - alpha_hat_a, we compute

    V_design_full(X)  = Var_s[ sum_a S_s(a) X_a ],
    V_design_local(X) = local product-graph covariance part implied by S_s,

then form the tail-corrected estimator

    V_DA-PG = V_PG-HAC + [V_design_full - V_design_local]_+.

This is a design-assisted / model-light diagnostic: it uses the observed
residualized outcome as a fixed-outcome plug-in and the known COSTA sampler to
estimate the nonlocal design variance tail that ordinary PG-HAC omits.
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
from costa.designs import optimize_costa, optimize_costa_ic, sample_design
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
SEED_STREAM_ID = {"assignment": 1, "outcome": 2, "baseline": 3, "design_mc": 4}
DEFAULT_LAMBDA_LOCAL_GRID = "0.01,0.05,0.1,0.2,0.5,2.0,5.0"


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
    ap = argparse.ArgumentParser(description="CLT validation with original PG-HAC and design-aware PG-HAC.")
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
    ap.add_argument("--design-mc", type=int, default=200, help="Number of assignment draws for design-aware variance calibration.")
    ap.add_argument("--design-mc-batch", type=int, default=50, help="Batch size used to build score draws.")
    ap.add_argument("--ci-level", type=float, default=0.90)
    ap.add_argument("--variance-mode", choices=["conservative", "signed"], default="conservative")
    ap.add_argument("--da-method", choices=["tail", "max", "design_mc"], default="tail")
    ap.add_argument("--out-root", default="outputs/clt_designaware")
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


def _undirected_support_edges(A) -> tuple[np.ndarray, np.ndarray]:
    S = ((A != 0).astype(np.int8) + (A.T != 0).astype(np.int8)).tocoo()
    rows = S.row.astype(np.int64)
    cols = S.col.astype(np.int64)
    mask = rows < cols
    return rows[mask], cols[mask]


class DesignAwareCalibrator:
    """Precompute assignment-score draws and local design covariance coefficients."""

    def __init__(
        self,
        design,
        A,
        weights: np.ndarray,
        p: float,
        burn_in: int,
        n_draws: int,
        seed: int,
        batch_size: int = 50,
        show_progress: bool = False,
    ) -> None:
        self.design = design
        self.p = float(p)
        self.burn_in = int(burn_in)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.n, self.B = self.weights.shape
        self.B_eval = self.B - self.burn_in
        self.W_eval = float(np.sum(self.weights[:, self.burn_in:]) + 1e-12)
        self.n_draws = int(n_draws)
        self.score_draws = self._build_score_draws(seed=seed, batch_size=batch_size, show_progress=show_progress)
        self.centered_scores = self.score_draws - self.score_draws.mean(axis=0, keepdims=True)
        self.ddof = max(1, self.score_draws.shape[0] - 1)
        self.diag_cov = np.var(self.score_draws, axis=0, ddof=1).astype(np.float64)
        self.edge_i, self.edge_j = _undirected_support_edges(A)
        self.comp_pairs, self.comp_cov = self._precompute_comp_covariances()
        self.temp_pairs, self.temp_cov = self._precompute_temp_covariances()

    def idx(self, i: np.ndarray | int, t: np.ndarray | int) -> np.ndarray | int:
        return np.asarray(i) * self.B_eval + (np.asarray(t) - self.burn_in)

    def _build_score_draws(self, seed: int, batch_size: int, show_progress: bool) -> np.ndarray:
        q = 1.0 - self.p
        v = self.p * q
        m = self.n * self.B_eval
        scores = np.empty((self.n_draws, m), dtype=np.float32)
        iterator = range(self.n_draws)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Design MC scores {self.design.name}", unit="draw", leave=False)
        for s in iterator:
            Z = sample_design(self.design, self.n, self.B, self.p, seed=int(seed + 7919 * s))
            psi = (Z[:, self.burn_in:].astype(np.float32) - self.p) / v
            sc = (self.weights[:, self.burn_in:].astype(np.float32) / self.W_eval) * psi
            scores[s, :] = sc.reshape(-1)
        return scores

    def _cov_for_pairs(self, idx1: np.ndarray, idx2: np.ndarray, batch: int = 200000) -> np.ndarray:
        idx1 = np.asarray(idx1, dtype=np.int64)
        idx2 = np.asarray(idx2, dtype=np.int64)
        out = np.empty(len(idx1), dtype=np.float64)
        C = self.centered_scores
        for start in range(0, len(idx1), batch):
            end = min(start + batch, len(idx1))
            prod = C[:, idx1[start:end]] * C[:, idx2[start:end]]
            out[start:end] = np.sum(prod, axis=0) / self.ddof
        return out

    def _precompute_comp_covariances(self) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        if len(self.edge_i) == 0:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)), np.zeros(0, dtype=np.float64)
        left = []
        right = []
        for t in range(self.burn_in, self.B):
            left.append(self.idx(self.edge_i, t))
            right.append(self.idx(self.edge_j, t))
        idx1 = np.concatenate(left).astype(np.int64)
        idx2 = np.concatenate(right).astype(np.int64)
        cov = self._cov_for_pairs(idx1, idx2)
        return (idx1, idx2), cov

    def _precompute_temp_covariances(self) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        if self.B <= self.burn_in + 1:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)), np.zeros(0, dtype=np.float64)
        left = []
        right = []
        items = np.arange(self.n, dtype=np.int64)
        for t in range(self.burn_in + 1, self.B):
            left.append(self.idx(items, t))
            right.append(self.idx(items, t - 1))
        idx1 = np.concatenate(left).astype(np.int64)
        idx2 = np.concatenate(right).astype(np.int64)
        cov = self._cov_for_pairs(idx1, idx2)
        return (idx1, idx2), cov

    def full_design_variance(self, X_eval: np.ndarray) -> float:
        x = np.asarray(X_eval, dtype=np.float64).reshape(-1)
        vals = self.score_draws.astype(np.float64) @ x
        return float(np.var(vals, ddof=1)) if vals.size > 1 else 0.0

    def local_design_variance(self, X_eval: np.ndarray) -> float:
        x = np.asarray(X_eval, dtype=np.float64).reshape(-1)
        val = float(np.sum(self.diag_cov * x * x))
        idx1, idx2 = self.comp_pairs
        if len(idx1):
            val += float(2.0 * np.sum(self.comp_cov * x[idx1] * x[idx2]))
        idx1, idx2 = self.temp_pairs
        if len(idx1):
            val += float(2.0 * np.sum(self.temp_cov * x[idx1] * x[idx2]))
        return float(val)

    def designaware_variance(self, X_eval: np.ndarray, pg_var: float, method: str = "tail") -> dict[str, float]:
        v_full = self.full_design_variance(X_eval)
        v_local = self.local_design_variance(X_eval)
        tail = max(float(v_full - v_local), 0.0)
        v_tail = max(float(pg_var) + tail, 0.0)
        v_max = max(float(pg_var), float(v_full), 0.0)
        if method == "tail":
            chosen = v_tail
        elif method == "max":
            chosen = v_max
        elif method == "design_mc":
            chosen = max(float(v_full), 0.0)
        else:
            raise ValueError(f"Unknown DA method: {method}")
        return {
            "design_mc_variance": float(max(v_full, 0.0)),
            "design_mc_local_variance": float(v_local),
            "design_mc_tail_variance": float(tail),
            "da_tail_variance": float(v_tail),
            "da_max_variance": float(v_max),
            "da_variance": float(chosen),
            "design_mc_se": float(np.sqrt(max(v_full, 0.0))),
            "design_mc_local_se": float(np.sqrt(max(v_local, 0.0))),
            "design_mc_tail_se": float(np.sqrt(max(tail, 0.0))),
            "da_tail_se": float(np.sqrt(max(v_tail, 0.0))),
            "da_max_se": float(np.sqrt(max(v_max, 0.0))),
            "da_se": float(np.sqrt(max(chosen, 0.0))),
        }


def init_store() -> dict[str, list[float]]:
    keys = [
        "weighted_ht", "hajek", "dim",
        "pg_se", "design_mc_se", "design_mc_local_se", "design_mc_tail_se", "da_tail_se", "da_max_se", "da_se",
        "pg_ci_lower", "pg_ci_upper", "pg_ba_lower", "pg_ba_upper",
        "da_ci_lower", "da_ci_upper", "da_ba_lower", "da_ba_upper",
        "da_max_ci_lower", "da_max_ci_upper", "design_mc_ci_lower", "design_mc_ci_upper",
    ]
    return {k: [] for k in keys}


def summarize(vals: dict[str, list[float]], tau: float, ci_level: float, zcrit: float) -> dict[str, float]:
    est = np.asarray(vals["weighted_ht"], dtype=float)
    qlo, qhi = np.quantile(est, [(1.0 - ci_level) / 2.0, 1.0 - (1.0 - ci_level) / 2.0])
    mean_est = float(est.mean())
    std = float(est.std(ddof=1)) if est.size > 1 else 0.0
    def cov(lo_key: str, hi_key: str) -> float:
        return float(np.mean([(lo <= tau <= hi) for lo, hi in zip(vals[lo_key], vals[hi_key])]))
    return {
        "mean_est": mean_est,
        "bias": float(mean_est - tau),
        "abs_bias": float(abs(mean_est - tau)),
        "std": std,
        "rmse": float(np.sqrt(np.mean((est - tau) ** 2))),
        "mc_q05": float(qlo),
        "mc_q95": float(qhi),
        "mc_interval_width": float(qhi - qlo),
        "mc_normal_halfwidth": float(zcrit * std),
        "avg_pg_hac_se": float(np.mean(vals["pg_se"])),
        "avg_design_mc_se": float(np.mean(vals["design_mc_se"])),
        "avg_design_mc_local_se": float(np.mean(vals["design_mc_local_se"])),
        "avg_design_mc_tail_se": float(np.mean(vals["design_mc_tail_se"])),
        "avg_da_tail_se": float(np.mean(vals["da_tail_se"])),
        "avg_da_max_se": float(np.mean(vals["da_max_se"])),
        "avg_da_se": float(np.mean(vals["da_se"])),
        "pg_hac_halfwidth": float(zcrit * np.mean(vals["pg_se"])),
        "design_mc_halfwidth": float(zcrit * np.mean(vals["design_mc_se"])),
        "da_hac_halfwidth": float(zcrit * np.mean(vals["da_se"])),
        "pg_hac_se_to_mc_std": float(np.mean(vals["pg_se"]) / (std + 1e-12)),
        "da_hac_se_to_mc_std": float(np.mean(vals["da_se"]) / (std + 1e-12)),
        "design_mc_se_to_mc_std": float(np.mean(vals["design_mc_se"]) / (std + 1e-12)),
        "pg_ci_coverage": cov("pg_ci_lower", "pg_ci_upper"),
        "pg_biasaware_coverage": cov("pg_ba_lower", "pg_ba_upper"),
        "da_ci_coverage": cov("da_ci_lower", "da_ci_upper"),
        "da_biasaware_coverage": cov("da_ba_lower", "da_ba_upper"),
        "da_max_ci_coverage": cov("da_max_ci_lower", "da_max_ci_upper"),
        "design_mc_ci_coverage": cov("design_mc_ci_lower", "design_mc_ci_upper"),
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
    n, B = weights_raw.shape
    d_time = max(1, B - 1) if str(args.d_time).lower() in {"auto", "b-1", "bminus1"} else int(args.d_time)
    d_item = int(args.d_item if args.d_item is not None else get_default(cfg, "d_item", 16))
    steps = int(args.steps if args.steps is not None else get_default(cfg, "steps", 400))
    lr = float(args.lr if args.lr is not None else get_default(cfg, "lr", 0.05))
    seed = int(args.seed if args.seed is not None else get_default(cfg, "seed", 123))
    p_values = parse_list(args.p_values, float)
    models = [canonical_model_name(x) for x in parse_list(args.models, str)]
    lambda_grid = parse_list(args.lambda_local_grid, float)
    zcrit = float(norm.ppf(0.5 + float(args.ci_level) / 2.0))
    flat = flatten_config(cfg)

    dataset = str(meta.get("dataset", base.get("dataset", Path(prepared_dir).name.split("_top")[0])))
    out_dir = resolve_under_root(Path(args.out_root) / f"{dataset}_top{format_count(n)}_b{B}_k{meta.get('top_k', args.top_k)}_mc{args.mc}_designmc{args.design_mc}_di{d_item}_dt{d_time}")
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "script": "scripts/run_clt_designaware_variance.py",
        "config": str(Path(args.config)),
        "prepared_dir": relpath(prepared_dir),
        "out_dir": relpath(out_dir),
        "dataset": dataset,
        "N": int(n),
        "B": int(B),
        "top_k": int(meta.get("top_k", args.top_k)),
        "p_values": p_values,
        "models": models,
        "mc": int(args.mc),
        "design_mc": int(args.design_mc),
        "ci_level": float(args.ci_level),
        "variance_mode": args.variance_mode,
        "da_method": args.da_method,
        "steps": steps,
        "lr": lr,
        "d_item": d_item,
        "d_time": d_time,
        "lambda_local_grid": lambda_grid,
        "ic_select_weight": float(args.ic_select_weight),
        "ic_target_far_product": args.ic_target_far_product,
        "n_far_pairs": int(args.n_far_pairs),
        "base_seed": seed,
    }
    write_json(out_dir / "designaware_run_config.json", run_config)

    all_summary: list[dict[str, Any]] = []
    all_draws: list[dict[str, Any]] = []
    all_search: list[dict[str, Any]] = []
    all_traj: list[pd.DataFrame] = []

    for p in p_values:
        print(f"Optimizing standard COSTA for p={p}...")
        costa = optimize_costa(
            A, p=p, n_blocks=B, d_item=d_item, d_time=d_time, steps=steps, lr=lr,
            w_comp=float(flat.get("w_comp", 1.0)), w_temp=float(flat.get("w_temp", 1.0)),
            lambda_var=float(flat.get("lambda_var", 0.05)), lambda_balance=float(flat.get("lambda_balance", 0.05)),
            switch_min=float(flat.get("switch_min", 0.10)), seed=seed,
            monitor_every=max(1, int(args.monitor_every)), show_progress=not args.no_progress,
        )
        if costa.objective_history:
            tmp = pd.DataFrame(costa.objective_history); tmp["p"] = p; tmp["candidate"] = "standard"; all_traj.append(tmp)

        ic_candidates = []
        for lam in lambda_grid:
            print(f"Optimizing COSTA-IC for p={p}, lambda_local={lam}...")
            cand = optimize_costa_ic(
                A, p=p, n_blocks=B, d_item=d_item, d_time=d_time, steps=steps, lr=lr,
                w_comp=float(flat.get("w_comp", 1.0)), w_temp=float(flat.get("w_temp", 1.0)),
                lambda_var=float(flat.get("lambda_var", 0.05)), lambda_balance=float(flat.get("lambda_balance", 0.05)),
                switch_min=float(flat.get("switch_min", 0.10)), lambda_local=float(lam),
                n_far_pairs=int(args.n_far_pairs), seed=seed,
                monitor_every=max(1, int(args.monitor_every)), show_progress=not args.no_progress,
            )
            cand.name = "COSTA_IC"
            loc = locality_diagnostics(cand, p=p, n_pairs=min(args.n_far_pairs, 50000), seed=seed + 11)
            last = dict(cand.objective_history[-1]) if cand.objective_history else {}
            score = float(last.get("mse_objective", np.inf)) + float(args.ic_select_weight) * float(loc.get("far_product_mean_abs_R", np.inf))
            if args.ic_target_far_product is not None and loc.get("far_product_mean_abs_R", np.inf) <= float(args.ic_target_far_product):
                score = float(last.get("mse_objective", np.inf))
            ic_candidates.append({"score": score, "design": cand, "lambda_local": float(lam), "loc": loc, "last": last})
            if cand.objective_history:
                tmp = pd.DataFrame(cand.objective_history); tmp["p"] = p; tmp["candidate"] = f"lambda_local={lam:g}"; all_traj.append(tmp)
        ic_candidates.sort(key=lambda x: x["score"])
        selected = ic_candidates[0]
        print(f"Selected COSTA-IC lambda_local={selected['lambda_local']:g} for p={p}.")

        designs = [
            {"design": costa, "name": "COSTA", "lambda_local": np.nan, "loc": locality_diagnostics(costa, p=p, n_pairs=min(args.n_far_pairs, 50000), seed=seed + 123), "last": dict(costa.objective_history[-1]) if costa.objective_history else {}},
            {"design": selected["design"], "name": "COSTA_IC", "lambda_local": selected["lambda_local"], "loc": selected["loc"], "last": selected["last"]},
        ]

        print(f"Precomputing design-aware score draws for p={p}...")
        calibrators = []
        for entry in designs:
            cal_seed = make_seed(seed, 0, int(round(1000 * float(p))) + (0 if entry["name"] == "COSTA" else 1), "design_mc")
            calibrators.append(DesignAwareCalibrator(
                entry["design"], A, W, p=float(p), burn_in=1, n_draws=int(args.design_mc),
                seed=cal_seed, batch_size=int(args.design_mc_batch), show_progress=not args.no_progress,
            ))

        for model in models:
            model_id = MODEL_SEED_ID.get(model, 9)
            params = make_params(flat, alpha, beta, model)
            tau = true_tau(A, params, weights=W, burn_in=1)
            stores = [init_store(), init_store()]
            consts = [design_bias_components(d["design"], A, W, p=p, gamma=params.gamma, eta=params.eta, burn_in=1) for d in designs]

            iterator = range(int(args.mc))
            if not args.no_progress:
                iterator = tqdm(iterator, desc=f"DA-HAC MC p={p} model={model}", unit="rep", leave=False)
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
                for j, entry in enumerate(designs):
                    design = entry["design"]
                    Z = sample_design(design, n, B, p, seed=assignment_seed)
                    Y = generate_observed_outcomes(Z, A, params, seed=outcome_seed)
                    ht = weighted_ht(Z, Y, alpha=alpha_hat, p=p, weights=W, burn_in=1)
                    hj = hajek_difference(Z, Y, alpha=alpha_hat, weights=W, burn_in=1)
                    dm = difference_in_means(Z, Y, alpha=alpha_hat, burn_in=1)
                    ht2, xi, Wtot = ht_contributions(Z, Y, alpha_hat, p=p, weights=W, burn_in=1)
                    vh = product_graph_hac_variance(xi, A, Wtot, burn_in=1, mode=args.variance_mode)
                    pg_var = float(vh["variance"])
                    pg_se = float(vh["se"])
                    X_eval = (Y[:, 1:] - alpha_hat[:, 1:]).astype(np.float64)
                    da = calibrators[j].designaware_variance(X_eval, pg_var=pg_var, method=args.da_method)
                    da_se = float(da["da_se"])
                    bias_bound = float(consts[j].get("bias_bound", 0.0))

                    pg_low = float(ht2 - zcrit * pg_se); pg_hi = float(ht2 + zcrit * pg_se)
                    pg_ba_low = float(pg_low - bias_bound); pg_ba_hi = float(pg_hi + bias_bound)
                    da_low = float(ht2 - zcrit * da_se); da_hi = float(ht2 + zcrit * da_se)
                    da_ba_low = float(da_low - bias_bound); da_ba_hi = float(da_hi + bias_bound)
                    da_max_low = float(ht2 - zcrit * da["da_max_se"]); da_max_hi = float(ht2 + zcrit * da["da_max_se"])
                    dmc_low = float(ht2 - zcrit * da["design_mc_se"]); dmc_hi = float(ht2 + zcrit * da["design_mc_se"])

                    s = stores[j]
                    s["weighted_ht"].append(float(ht)); s["hajek"].append(float(hj)); s["dim"].append(float(dm))
                    s["pg_se"].append(pg_se)
                    for key in ["design_mc_se", "design_mc_local_se", "design_mc_tail_se", "da_tail_se", "da_max_se", "da_se"]:
                        s[key].append(float(da[key]))
                    s["pg_ci_lower"].append(pg_low); s["pg_ci_upper"].append(pg_hi)
                    s["pg_ba_lower"].append(pg_ba_low); s["pg_ba_upper"].append(pg_ba_hi)
                    s["da_ci_lower"].append(da_low); s["da_ci_upper"].append(da_hi)
                    s["da_ba_lower"].append(da_ba_low); s["da_ba_upper"].append(da_ba_hi)
                    s["da_max_ci_lower"].append(da_max_low); s["da_max_ci_upper"].append(da_max_hi)
                    s["design_mc_ci_lower"].append(dmc_low); s["design_mc_ci_upper"].append(dmc_hi)

                    all_draws.append({
                        "dataset": dataset, "p": p, "model": model, "design": entry["name"], "lambda_local": entry["lambda_local"],
                        "rep": rep, "tau_true": float(tau), "weighted_ht": float(ht), "hajek": float(hj), "dim": float(dm),
                        "weighted_ht_error": float(ht - tau), "pg_hac_se": pg_se, "design_mc_se": float(da["design_mc_se"]),
                        "da_tail_se": float(da["da_tail_se"]), "da_max_se": float(da["da_max_se"]), "da_hac_se": da_se,
                        "pg_ci_covers_tau": bool(pg_low <= tau <= pg_hi), "pg_biasaware_covers_tau": bool(pg_ba_low <= tau <= pg_ba_hi),
                        "da_ci_covers_tau": bool(da_low <= tau <= da_hi), "da_biasaware_covers_tau": bool(da_ba_low <= tau <= da_ba_hi),
                        "design_mc_variance": float(da["design_mc_variance"]), "design_mc_local_variance": float(da["design_mc_local_variance"]),
                        "design_mc_tail_variance": float(da["design_mc_tail_variance"]), "bias_bound": bias_bound,
                        "assignment_seed": assignment_seed, "outcome_seed": outcome_seed, "baseline_seed": baseline_seed,
                    })

            for j, entry in enumerate(designs):
                summ = summarize(stores[j], tau=tau, ci_level=float(args.ci_level), zcrit=zcrit)
                row = {
                    "dataset": dataset, "p": p, "model": model, "design": entry["name"], "estimator": "weighted_ht", "tau_true": float(tau),
                    "lambda_local": entry["lambda_local"], "N": n, "B": B, "top_k": int(meta.get("top_k", args.top_k)),
                    "mc": int(args.mc), "design_mc": int(args.design_mc), "d_item": d_item, "d_time": d_time,
                    "ci_level": float(args.ci_level), "da_method": args.da_method,
                    "bias_bound": float(consts[j].get("bias_bound", np.nan)),
                    "design_linear_bias_formula": float(consts[j].get("bias_linear", np.nan)),
                }
                row.update(entry.get("loc", {})); row.update(summ)
                all_summary.append(row)

        # compact search diagnostics for IC candidates, using optimizer/locality only
        for cand in ic_candidates:
            all_search.append({
                "dataset": dataset, "p": p, "lambda_local": cand["lambda_local"], "selection_score": cand["score"],
                "selected": bool(cand is selected),
                "final_mse_objective": float(cand["last"].get("mse_objective", np.nan)),
                "far_product_mean_abs_R": float(cand["loc"].get("far_product_mean_abs_R", np.nan)),
                "far_time_mean_abs_R": float(cand["loc"].get("far_time_mean_abs_R", np.nan)),
                "far_item_mean_abs_R": float(cand["loc"].get("far_item_mean_abs_R", np.nan)),
            })

    summary = pd.DataFrame(all_summary)
    draws = pd.DataFrame(all_draws)
    search = pd.DataFrame(all_search)
    traj = pd.concat(all_traj, ignore_index=True) if all_traj else pd.DataFrame()

    if not summary.empty:
        summary["design_order"] = summary["design"].map({"COSTA": 1, "COSTA_IC": 2}).fillna(99)
        summary = summary.sort_values(["dataset", "p", "model", "design_order"]).drop(columns=["design_order"])
    draws.to_csv(out_dir / "designaware_draws.csv", index=False)
    summary.to_csv(out_dir / "designaware_summary.csv", index=False)
    search.to_csv(out_dir / "designaware_lambda_search.csv", index=False)
    if not traj.empty:
        traj.to_csv(out_dir / "designaware_optimization_trajectory.csv", index=False)

    cols = ["dataset", "p", "model", "design", "std", "rmse", "avg_pg_hac_se", "avg_da_se", "avg_design_mc_se", "pg_ci_coverage", "da_ci_coverage", "da_biasaware_coverage"]
    print(summary[cols])
    print(f"Saved design-aware CLT outputs to {relpath(out_dir)}")


if __name__ == "__main__":
    main()
