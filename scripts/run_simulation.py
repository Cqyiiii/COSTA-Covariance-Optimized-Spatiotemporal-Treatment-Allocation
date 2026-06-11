#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

# Conservative thread defaults keep the prototype reliable in small containers.
# Override these environment variables externally for large production runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm.auto import tqdm

from costa.config import set_argparse_defaults_from_config
from costa.designs import (
    OptimizedDesign,
    make_cluster_leiden_design,
    optimize_static_ocd,
    optimize_costa,
    optimize_costa_ic,
    optimize_cluster_costa_no_kro,
    optimize_cluster_costa_kro,
    sample_design,
)
from costa.outcomes import (
    make_baseline,
    make_alpha_adjustment,
    OutcomeParameters,
    generate_observed_outcomes,
    true_tau,
    canonical_model_name,
)
from costa.estimators import weighted_ht, hajek_difference, difference_in_means
from costa.naming import read_json, write_json, simulation_dir_name


MODEL_ORDER = ["linear", "nonlinear", "demand"]
DESIGN_ORDER = [
    "independent", "rbsd_approx", "cluster_leiden", "static_graph_ocd", "COSTA",
    "COSTA_cluster_no_kro", "COSTA_cluster_kro",
    "COSTA_IC_lam0.2", "COSTA_IC_lam1", "COSTA_IC_lam5",
]
ESTIMATOR_ORDER = ["weighted_ht", "hajek", "dim"]

MODEL_SEED_ID = {"linear": 1, "nonlinear": 2, "demand": 3, "interaction_second_hop": 4}
SEED_STREAM_ID = {"assignment": 1, "outcome": 2, "baseline": 3}


def make_seed(base_seed: int, model_id: int, rep: int, stream: str) -> int:
    """Deterministic paired Monte Carlo seed.

    The seed varies with the potential-outcome model and repetition, but not
    with the experimental design. Thus, all designs within the same
    ``(model, rep)`` face the same assignment random source, outcome shocks,
    and baseline-adjustment error. This implements a paired/common-random-
    numbers comparison while remaining fully reproducible from ``base_seed``.
    """
    if stream not in SEED_STREAM_ID:
        raise ValueError(f"Unknown seed stream: {stream}")
    seq = np.random.SeedSequence([int(base_seed), int(model_id), int(rep), SEED_STREAM_ID[stream]])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _resolve_under_root(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    # Prefer a path relative to current working directory if it already exists.
    if p.exists():
        return p.resolve()
    return (ROOT / p).resolve()


def load_prepared(prepared_dir: str | Path) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, Any]]:
    prepared_dir = _resolve_under_root(prepared_dir)
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared directory not found: {prepared_dir}")
    sparse_path = prepared_dir / "competition_A.npz"
    dense_path = prepared_dir / "competition_A.npy"
    if sparse_path.exists():
        A = sparse.load_npz(sparse_path).tocsr().astype(np.float32)
    elif dense_path.exists():
        # Backward compatibility with earlier packages. Convert immediately to CSR;
        # repeated dense A @ Z multiplications were the main reason 10k-item runs
        # could become unexpectedly slow.
        A = sparse.csr_matrix(np.load(dense_path)).astype(np.float32)
    else:
        raise FileNotFoundError(f"Cannot find competition_A.npz or competition_A.npy in {prepared_dir}")
    weights = np.load(prepared_dir / "item_block_weight.npy")
    metadata = read_json(prepared_dir / "metadata.json", default={})
    metadata.setdefault("resolved_prepared_dir", str(prepared_dir))
    metadata.setdefault("graph_nnz", int(A.nnz))
    metadata.setdefault("graph_density", float(A.nnz / max(A.shape[0] * A.shape[1], 1)))
    return A, weights, metadata


def assignment_diagnostics(Z: np.ndarray) -> dict[str, float]:
    Zf = Z.astype(float)
    block_rates = Zf.mean(axis=0)
    item_rates = Zf.mean(axis=1)
    lag_switch = float(np.mean(Z[:, 1:] != Z[:, :-1])) if Z.shape[1] > 1 else float("nan")
    return {
        "treatment_rate": float(Zf.mean()),
        "lag_switch_rate": lag_switch,
        "block_rate_min": float(block_rates.min()),
        "block_rate_max": float(block_rates.max()),
        "block_rate_sd": float(block_rates.std(ddof=0)),
        "item_rate_mean": float(item_rates.mean()),
        "item_rate_sd": float(item_rates.std(ddof=0)),
    }


def flatten_objective_histories(designs: list[OptimizedDesign]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for design in designs:
        for entry in design.objective_history or []:
            if isinstance(entry, dict):
                row = dict(entry)
                row.setdefault("design", design.name)
                rows.append(row)
            else:
                rows.append({
                    "design": design.name,
                    "step": len(rows),
                    "mse_objective": float(entry),
                    "regularized_loss": float(entry),
                })
    return pd.DataFrame(rows)


def write_objective_outputs(
    designs: list[OptimizedDesign],
    out_dir: Path,
    make_plot: bool = True,
    monitor_repetition: int = 0,
    append: bool = False,
) -> None:
    """Save optimizer monitoring artifacts.

    In append mode, existing trajectories are preserved and rows for newly run
    designs are replaced. This is useful when adding extra baselines to an
    already completed ablation folder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = flatten_objective_histories(designs)
    if traj.empty:
        return
    traj["monitor_repetition"] = int(monitor_repetition)
    if append and (out_dir / "optimization_trajectory.csv").exists():
        old = pd.read_csv(out_dir / "optimization_trajectory.csv")
        new_designs = set(traj["design"].astype(str).unique())
        if "design" in old.columns:
            old = old[~old["design"].astype(str).isin(new_designs)]
        traj = pd.concat([old, traj], ignore_index=True, sort=False)
    traj.to_csv(out_dir / "optimization_trajectory.csv", index=False)
    traj.to_csv(out_dir / f"optimization_trajectory_rep{int(monitor_repetition)}.csv", index=False)
    wide = traj.pivot_table(index="step", columns="design", values="mse_objective", aggfunc="last").reset_index()
    wide.to_csv(out_dir / "optimization_mse_trajectory_wide.csv", index=False)
    summary = (
        traj.groupby("design", as_index=False)
        .agg(
            first_step=("step", "min"),
            last_step=("step", "max"),
            initial_mse_objective=("mse_objective", "first"),
            final_mse_objective=("mse_objective", "last"),
            best_mse_objective=("mse_objective", "min"),
            initial_regularized_loss=("regularized_loss", "first"),
            final_regularized_loss=("regularized_loss", "last"),
        )
    )
    summary.to_csv(out_dir / "optimization_summary.csv", index=False)
    obj_payload = {d.name: d.objective_history for d in designs if d.objective_history is not None}
    if append and (out_dir / "design_objectives.json").exists():
        try:
            with open(out_dir / "design_objectives.json", "r", encoding="utf-8") as f:
                old_payload = json.load(f)
            if isinstance(old_payload, dict):
                old_payload.update(obj_payload)
                obj_payload = old_payload
        except Exception:
            pass
    with open(out_dir / "design_objectives.json", "w", encoding="utf-8") as f:
        json.dump(obj_payload, f, indent=2, allow_nan=True)
    if not make_plot:
        return
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for design_name, group in traj.groupby("design"):
            group = group.sort_values("step")
            ax.plot(group["step"], group["mse_objective"], label=design_name)
        ax.set_xlabel("optimization step")
        ax.set_ylabel("MSE-style objective")
        ax.set_title("OCD vs COSTA optimization trajectory")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "optimization_trajectory.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        with open(out_dir / "optimization_plot_warning.txt", "w", encoding="utf-8") as warn:
            warn.write(f"Could not write optimization_trajectory.pdf: {exc}\n")


def parse_models(s) -> list[str]:
    if isinstance(s, (list, tuple)):
        parts = [str(x).strip() for x in s if str(x).strip()]
    else:
        parts = [x.strip() for x in str(s).replace(";", ",").split(",") if x.strip()]
    if not parts:
        parts = MODEL_ORDER
    out = []
    for m in parts:
        cm = canonical_model_name(m)
        if cm == "interaction_second_hop":
            cm = "demand"
        if cm not in out:
            out.append(cm)
    return out


def parse_designs(s) -> list[str]:
    if isinstance(s, (list, tuple)):
        parts = [str(x).strip() for x in s if str(x).strip()]
    else:
        parts = [x.strip() for x in str(s).replace(";", ",").split(",") if x.strip()]
    aliases = {
        "bernoulli": "independent",
        "independent": "independent",
        "rbsd": "rbsd_approx",
        "rbsd_approx": "rbsd_approx",
        "cluster": "cluster_leiden",
        "cluster_leiden": "cluster_leiden",
        "leiden": "cluster_leiden",
        "ocd": "static_graph_ocd",
        "static_ocd": "static_graph_ocd",
        "static_graph_ocd": "static_graph_ocd",
        "costa": "COSTA",
        "COSTA": "COSTA",
        "costa_ic": "COSTA_IC_lam1",
        "COSTA_IC": "COSTA_IC_lam1",
        "costa-ic": "COSTA_IC_lam1",
        "costa_ic_lam0.2": "COSTA_IC_lam0.2",
        "COSTA_IC_lam0.2": "COSTA_IC_lam0.2",
        "costa-ic-lam0.2": "COSTA_IC_lam0.2",
        "costa_ic_lam0p2": "COSTA_IC_lam0.2",
        "COSTA_IC_lam0p2": "COSTA_IC_lam0.2",
        "costa_ic_lam1": "COSTA_IC_lam1",
        "COSTA_IC_lam1": "COSTA_IC_lam1",
        "costa-ic-lam1": "COSTA_IC_lam1",
        "costa_ic_lam5": "COSTA_IC_lam5",
        "COSTA_IC_lam5": "COSTA_IC_lam5",
        "costa-ic-lam5": "COSTA_IC_lam5",
        "costa_cluster_no_kro": "COSTA_cluster_no_kro",
        "COSTA_cluster_no_kro": "COSTA_cluster_no_kro",
        "costa-cluster-no-kro": "COSTA_cluster_no_kro",
        "cluster_costa_no_kro": "COSTA_cluster_no_kro",
        "costa_cluster_kro": "COSTA_cluster_kro",
        "COSTA_cluster_kro": "COSTA_cluster_kro",
        "costa-cluster-kro": "COSTA_cluster_kro",
        "cluster_costa_kro": "COSTA_cluster_kro",
    }
    if not parts:
        parts = DESIGN_ORDER
    out = []
    for d in parts:
        key = aliases.get(d, aliases.get(d.lower()))
        if key is None:
            raise ValueError(f"Unknown design: {d}")
        if key not in out:
            out.append(key)
    return out



def format_lambda_label(value: float) -> str:
    """Return the design-name suffix used for COSTA-IC lambda values."""
    v = float(value)
    if abs(v - round(v)) < 1e-12:
        return str(int(round(v)))
    # Keep decimal notation in summary tables as requested, e.g. lam0.2.
    return (f"{v:.6g}").replace("-", "m")


def parse_float_list(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value]
    else:
        parts = [x.strip() for x in str(value).replace(";", ",").split(",")]
    out = []
    for part in parts:
        if part:
            out.append(float(part))
    return out


def costa_ic_lambda_from_design(name: str, default: float | None = None) -> float:
    """Parse COSTA_IC_lam{lambda} design names."""
    if name == "COSTA_IC":
        return 1.0 if default is None else float(default)
    prefix = "COSTA_IC_lam"
    if not name.startswith(prefix):
        raise ValueError(f"Not a COSTA-IC lambda design: {name}")
    raw = name[len(prefix):].replace("p", ".")
    return float(raw)


def resolve_d_time(raw: Any, n_blocks: int) -> int:
    """Resolve COSTA temporal latent dimension.

    The default is B-1, matching the number of adjacent one-lag temporal edges
    among B blocks. Accepted automatic values are None, "auto", "B-1", and
    "b_minus_1". Integer strings are also accepted.
    """
    if raw is None:
        return max(1, int(n_blocks) - 1)
    if isinstance(raw, str):
        key = raw.strip().lower().replace("_", "-")
        if key in {"", "none", "null", "auto", "b-1", "bminus1", "b-minus-1", "b-minus-1", "bminus-1"}:
            return max(1, int(n_blocks) - 1)
        return int(float(raw))
    return int(raw)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run COSTA semi-synthetic Monte Carlo simulations.")
    ap.add_argument("--config", default=None, help="Optional YAML config. CLI flags override config values.")
    ap.add_argument("--prepared-dir", default="data/processed/sample_top120_b4_k10")
    ap.add_argument(
        "--out", "--out-dir", dest="out_dir", default=None,
        help="Simulation output directory. If omitted, uses outputs/sim/{prepared_name}_p{p}_mc{mc}_steps{steps}_di{d_item}_dt{d_time}_{optional_design_tag} under the package root.",
    )
    ap.add_argument("--output-root", default="outputs/sim")
    ap.add_argument("--models", default="linear,nonlinear,demand", help="Comma-separated PO models: linear, nonlinear, demand.")
    ap.add_argument("--designs", default="independent,rbsd_approx,cluster_leiden,static_graph_ocd,COSTA,COSTA_cluster_no_kro,COSTA_cluster_kro,COSTA_IC_lam0.2,COSTA_IC_lam1,COSTA_IC_lam5", help="Comma-separated designs to run.")
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--mc", type=int, default=50)

    # Potential-outcome and noise parameters.
    ap.add_argument("--gamma", type=float, default=0.08)
    ap.add_argument("--eta", type=float, default=0.10)
    ap.add_argument("--noise-sd", type=float, default=0.50)
    ap.add_argument("--item-shock-sd", type=float, default=1.00)
    ap.add_argument("--block-shock-sd", type=float, default=0.20)
    ap.add_argument("--beta-base", type=float, default=0.08)
    ap.add_argument("--beta-pop-scale", type=float, default=0.04)
    ap.add_argument("--beta-noise-sd", type=float, default=0.12)
    ap.add_argument("--alpha-traffic-scale", type=float, default=0.4)
    ap.add_argument("--alpha-hat-item-sd", type=float, default=0.90)
    ap.add_argument("--alpha-hat-block-sd", type=float, default=0.25)
    ap.add_argument("--alpha-hat-cell-sd", type=float, default=0.45)
    ap.add_argument("--weight-floor-frac", type=float, default=0.05)

    # Misspecification parameters.
    ap.add_argument("--nonlinear-saturation-scale", dest="nonlinear_saturation_scale", type=float, default=1.5)
    ap.add_argument("--interaction-scale", "--nonlinear-interaction-scale", dest="interaction_scale", type=float, default=0.5)
    ap.add_argument("--nonlinear-lag2-scale", "--second-lag-scale", dest="nonlinear_lag2_scale", type=float, default=0.5)
    ap.add_argument("--second-hop-scale", type=float, default=0.25)
    ap.add_argument("--demand-substitution-scale", "--demand-competition-scale", dest="demand_substitution_scale", type=float, default=1.0)
    ap.add_argument("--demand-treated-buffer", "--demand-interaction-scale", dest="demand_treated_buffer", type=float, default=0.4)
    ap.add_argument("--demand-temporal-scale", type=float, default=3.0)
    ap.add_argument("--demand-cross-temporal-scale", "--demand-lag-competition-scale", dest="demand_cross_temporal_scale", type=float, default=1.0)
    ap.add_argument("--demand-saturation-scale", type=float, default=0.25)

    # Optimization parameters.
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--d-item", "--d-spatial", dest="d_item", type=int, default=16, help="COSTA/static-OCD item/spatial latent dimension d_I. The full COSTA cell latent dimension is d_item*d_time.")
    ap.add_argument("--d-time", "--d-temporal", dest="d_time", default="auto", help="COSTA time-block latent dimension d_T. Use auto/B-1 to set d_T=B-1.")
    ap.add_argument("--lambda-var", type=float, default=0.05)
    ap.add_argument("--lambda-balance", type=float, default=0.05)
    ap.add_argument("--w-comp", type=float, default=1.0)
    ap.add_argument("--w-temp", type=float, default=1.0)
    ap.add_argument("--switch-min", type=float, default=0.10)
    ap.add_argument("--costa-ic-lambdas", default="0.2,1,5", help="Comma-separated locality penalties used when COSTA_IC_lam* designs are selected.")
    ap.add_argument("--costa-ic-n-far-pairs", type=int, default=50000, help="Number of sampled far pairs for COSTA-IC locality penalty.")
    ap.add_argument("--cluster-resolution", type=float, default=1.0, help="Leiden/Louvain resolution for the cluster-level baseline.")
    ap.add_argument("--cluster-fallback", default="louvain", choices=["louvain", "greedy", "none"], help="Fallback if leidenalg is unavailable.")
    ap.add_argument("--monitor-every", type=int, default=1, help="Record optimizer metrics every this many steps.")
    ap.add_argument("--monitor-repetition", type=int, default=0, help="Save assignment diagnostics for this Monte Carlo repetition.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    ap.add_argument("--no-plot", action="store_true", help="Skip optimization_trajectory.pdf; CSV trajectories are still written.")
    ap.add_argument("--append-results", action="store_true", help="Append/replace the selected designs in an existing output directory instead of overwriting the summary files. Use with --out-dir for incremental baselines.")
    ap.add_argument("--seed", type=int, default=123)
    return ap


def sort_summary(res: pd.DataFrame) -> pd.DataFrame:
    if res.empty:
        return res
    model_order = {m: i for i, m in enumerate(MODEL_ORDER + ["interaction_second_hop"])}
    design_order = {d: i for i, d in enumerate(DESIGN_ORDER)}
    estimator_order = {e: i for i, e in enumerate(ESTIMATOR_ORDER)}
    res = res.copy()
    res["model_order"] = res["model"].map(model_order).fillna(999).astype(int)
    res["estimator_order"] = res["estimator"].map(estimator_order).fillna(999).astype(int)
    res["design_order"] = res["design"].map(design_order).fillna(999).astype(int)
    res = res.sort_values(["model_order", "estimator_order", "design_order"]).reset_index(drop=True)
    return res.drop(columns=["model_order", "estimator_order", "design_order"])


def write_wide_summary(res: pd.DataFrame, out_dir: Path) -> None:
    if res.empty:
        return
    wide = res.pivot_table(
        index=["model", "estimator", "tau_true"],
        columns="design",
        values=["mean_est", "bias", "std", "rmse", "mse"],
        aggfunc="first",
    )
    wide.columns = [f"{design}_{metric}" for metric, design in wide.columns]
    wide = wide.reset_index()
    model_order = {m: i for i, m in enumerate(MODEL_ORDER + ["interaction_second_hop"])}
    estimator_order = {e: i for i, e in enumerate(ESTIMATOR_ORDER)}
    wide["model_order"] = wide["model"].map(model_order).fillna(999).astype(int)
    wide["estimator_order"] = wide["estimator"].map(estimator_order).fillna(999).astype(int)
    wide = wide.sort_values(["model_order", "estimator_order"]).drop(columns=["model_order", "estimator_order"])
    wide.to_csv(out_dir / "simulation_summary_wide_by_model.csv", index=False)


def write_output_manifest(out_dir: Path) -> None:
    files = sorted([p for p in out_dir.iterdir() if p.is_file()])
    with (out_dir / "output_manifest.txt").open("w", encoding="utf-8") as f:
        for p in files:
            f.write(f"{p.name}\n")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()

    ap = build_parser()
    config_values = set_argparse_defaults_from_config(ap, known.config)
    args = ap.parse_args()

    A, weights, metadata = load_prepared(args.prepared_dir)
    n, b = weights.shape
    models = parse_models(args.models)
    selected_design_names = parse_designs(args.designs)
    d_time_eff = resolve_d_time(args.d_time, b)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = (ROOT / out_dir).resolve()
    else:
        out_dir = (ROOT / Path(args.output_root) / simulation_dir_name(
            args.prepared_dir, args.p, args.mc, args.steps,
            d_item=args.d_item, d_time=d_time_eff, designs=selected_design_names,
        )).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "prepared_dir": str(_resolve_under_root(args.prepared_dir)),
        "out_dir": str(out_dir),
        "models": models,
        "designs": selected_design_names,
        "p": args.p,
        "mc": args.mc,
        "gamma": args.gamma,
        "eta": args.eta,
        "noise_sd": args.noise_sd,
        "item_shock_sd": args.item_shock_sd,
        "block_shock_sd": args.block_shock_sd,
        "beta_base": args.beta_base,
        "beta_pop_scale": args.beta_pop_scale,
        "beta_noise_sd": args.beta_noise_sd,
        "alpha_traffic_scale": args.alpha_traffic_scale,
        "alpha_hat_item_sd": args.alpha_hat_item_sd,
        "alpha_hat_block_sd": args.alpha_hat_block_sd,
        "alpha_hat_cell_sd": args.alpha_hat_cell_sd,
        "weight_floor_frac": args.weight_floor_frac,
        "nonlinear_saturation_scale": args.nonlinear_saturation_scale,
        "interaction_scale": args.interaction_scale,
        "nonlinear_lag2_scale": args.nonlinear_lag2_scale,
        "second_hop_scale": args.second_hop_scale,
        "demand_substitution_scale": args.demand_substitution_scale,
        "demand_treated_buffer": args.demand_treated_buffer,
        "demand_temporal_scale": args.demand_temporal_scale,
        "demand_cross_temporal_scale": args.demand_cross_temporal_scale,
        "demand_saturation_scale": args.demand_saturation_scale,
        "steps": args.steps,
        "lr": args.lr,
        "d_item": args.d_item,
        "d_time": int(d_time_eff),
        "d_time_raw": args.d_time,
        "d_effective_kronecker": int(args.d_item * d_time_eff),
        "lambda_var": args.lambda_var,
        "lambda_balance": args.lambda_balance,
        "w_comp": args.w_comp,
        "w_temp": args.w_temp,
        "switch_min": args.switch_min,
        "costa_ic_lambdas": [float(x) for x in parse_float_list(args.costa_ic_lambdas)],
        "costa_ic_n_far_pairs": int(args.costa_ic_n_far_pairs),
        "cluster_resolution": args.cluster_resolution,
        "cluster_fallback": args.cluster_fallback,
        "monitor_every": args.monitor_every,
        "monitor_repetition": args.monitor_repetition,
        "no_plot": args.no_plot,
        "append_results": bool(args.append_results),
        "base_seed": args.seed,
        "seed_policy": "paired_by_model_and_repetition; design is not included in Monte Carlo seeds",
        "seed_streams": SEED_STREAM_ID,
        "model_seed_id": MODEL_SEED_ID,
        "config": str(args.config) if args.config else None,
        "config_values": config_values,
        "prepared_metadata": metadata,
    }
    if args.append_results and (out_dir / "run_config.json").exists():
        old_cfg = read_json(out_dir / "run_config.json", default={})
        old_designs = old_cfg.get("designs", []) if isinstance(old_cfg.get("designs", []), list) else []
        merged_designs = []
        for name in list(old_designs) + list(selected_design_names):
            if name not in merged_designs:
                merged_designs.append(name)
        append_history = list(old_cfg.get("append_history", [])) if isinstance(old_cfg.get("append_history", []), list) else []
        append_history.append({"designs": selected_design_names, "p": args.p, "mc": args.mc, "steps": args.steps, "d_item": args.d_item, "d_time": int(d_time_eff)})
        merged_cfg = dict(old_cfg)
        merged_cfg.update(run_config)
        merged_cfg["designs"] = merged_designs
        merged_cfg["append_designs_latest"] = selected_design_names
        merged_cfg["append_history"] = append_history
        run_config = merged_cfg
    write_json(out_dir / "run_config.json", run_config)

    alpha, beta = make_baseline(
        weights,
        seed=args.seed,
        beta_base=args.beta_base,
        beta_pop_scale=args.beta_pop_scale,
        beta_noise_sd=args.beta_noise_sd,
        alpha_traffic_scale=args.alpha_traffic_scale,
    )
    W = weights.astype(float)
    if np.any(W > 0):
        W = W + args.weight_floor_frac * np.mean(W[W > 0])
    else:
        W = np.ones_like(W)

    design_lookup: dict[str, OptimizedDesign] = {}
    optimized_for_monitoring: list[OptimizedDesign] = []

    cluster_needed = any(name in selected_design_names for name in {"cluster_leiden", "COSTA_cluster_no_kro", "COSTA_cluster_kro"})

    if cluster_needed:
        print("Clustering item competition graph for cluster-level designs...")
        cluster_design = make_cluster_leiden_design(
            A,
            p=args.p,
            resolution=args.cluster_resolution,
            seed=args.seed,
            fallback=args.cluster_fallback,
        )
        design_lookup["cluster_leiden"] = cluster_design
        if cluster_design.cluster_labels is not None:
            pd.DataFrame({"item_idx": np.arange(len(cluster_design.cluster_labels)), "cluster": cluster_design.cluster_labels}).to_csv(
                out_dir / "cluster_leiden_assignments.csv", index=False
            )
        write_json(out_dir / "cluster_leiden_summary.json", cluster_design.cluster_info or {})
        print(f"Cluster designs: {cluster_design.cluster_info}")

    if "static_graph_ocd" in selected_design_names:
        print("Optimizing static graph OCD...")
        static = optimize_static_ocd(
            A,
            p=args.p,
            d_item=args.d_item,
            steps=args.steps,
            lr=args.lr,
            lambda_var=args.lambda_var,
            lambda_balance=args.lambda_balance,
            seed=args.seed,
            monitor_every=args.monitor_every,
            show_progress=not args.no_progress,
        )
        design_lookup["static_graph_ocd"] = static
        optimized_for_monitoring.append(static)

    if "COSTA" in selected_design_names:
        print("Optimizing COSTA...")
        costa = optimize_costa(
            A,
            p=args.p,
            n_blocks=b,
            d_item=args.d_item,
            d_time=d_time_eff,
            steps=args.steps,
            lr=args.lr,
            w_comp=args.w_comp,
            w_temp=args.w_temp,
            lambda_var=args.lambda_var,
            lambda_balance=args.lambda_balance,
            switch_min=args.switch_min,
            seed=args.seed,
            monitor_every=args.monitor_every,
            show_progress=not args.no_progress,
        )
        design_lookup["COSTA"] = costa
        optimized_for_monitoring.append(costa)

    costa_ic_names = [name for name in selected_design_names if name == "COSTA_IC" or name.startswith("COSTA_IC_lam")]
    for ic_name in costa_ic_names:
        lam = costa_ic_lambda_from_design(ic_name)
        print(f"Optimizing {ic_name} (COSTA-IC lambda_local={lam:g})...")
        ic = optimize_costa_ic(
            A,
            p=args.p,
            n_blocks=b,
            d_item=args.d_item,
            d_time=d_time_eff,
            steps=args.steps,
            lr=args.lr,
            w_comp=args.w_comp,
            w_temp=args.w_temp,
            lambda_var=args.lambda_var,
            lambda_balance=args.lambda_balance,
            switch_min=args.switch_min,
            lambda_local=lam,
            n_far_pairs=args.costa_ic_n_far_pairs,
            seed=args.seed,
            monitor_every=args.monitor_every,
            show_progress=not args.no_progress,
        )
        ic.name = ic_name
        for row in ic.objective_history or []:
            if isinstance(row, dict):
                row["design"] = ic_name
                row["lambda_local"] = float(lam)
        design_lookup[ic_name] = ic
        optimized_for_monitoring.append(ic)

    if "COSTA_cluster_no_kro" in selected_design_names:
        if "cluster_leiden" not in design_lookup or design_lookup["cluster_leiden"].cluster_labels is None:
            raise RuntimeError("COSTA_cluster_no_kro requires cluster labels; clustering did not run.")
        print("Optimizing cluster-level COSTA without Kronecker product...")
        cc_no = optimize_cluster_costa_no_kro(
            A,
            cluster_labels=design_lookup["cluster_leiden"].cluster_labels,
            p=args.p,
            n_blocks=b,
            d_latent=args.d_item,
            steps=args.steps,
            lr=args.lr,
            w_comp=args.w_comp,
            w_temp=args.w_temp,
            lambda_var=args.lambda_var,
            lambda_balance=args.lambda_balance,
            switch_min=args.switch_min,
            seed=args.seed,
            monitor_every=args.monitor_every,
            show_progress=not args.no_progress,
        )
        cc_no.cluster_info = design_lookup["cluster_leiden"].cluster_info
        design_lookup["COSTA_cluster_no_kro"] = cc_no
        optimized_for_monitoring.append(cc_no)

    if "COSTA_cluster_kro" in selected_design_names:
        if "cluster_leiden" not in design_lookup or design_lookup["cluster_leiden"].cluster_labels is None:
            raise RuntimeError("COSTA_cluster_kro requires cluster labels; clustering did not run.")
        print("Optimizing cluster-level COSTA with Kronecker product...")
        cc_kro = optimize_cluster_costa_kro(
            A,
            cluster_labels=design_lookup["cluster_leiden"].cluster_labels,
            p=args.p,
            n_blocks=b,
            d_item=args.d_item,
            d_time=d_time_eff,
            steps=args.steps,
            lr=args.lr,
            w_comp=args.w_comp,
            w_temp=args.w_temp,
            lambda_var=args.lambda_var,
            lambda_balance=args.lambda_balance,
            switch_min=args.switch_min,
            seed=args.seed,
            monitor_every=args.monitor_every,
            show_progress=not args.no_progress,
        )
        cc_kro.cluster_info = design_lookup["cluster_leiden"].cluster_info
        design_lookup["COSTA_cluster_kro"] = cc_kro
        optimized_for_monitoring.append(cc_kro)

    write_objective_outputs(optimized_for_monitoring, out_dir, make_plot=not args.no_plot, monitor_repetition=args.monitor_repetition, append=args.append_results)

    design_lookup.setdefault("independent", OptimizedDesign(name="independent", p=args.p))
    design_lookup.setdefault("rbsd_approx", OptimizedDesign(name="rbsd_approx", p=args.p))
    designs = [design_lookup[name] for name in selected_design_names]

    summary_records: list[dict[str, Any]] = []
    draw_records: list[dict[str, Any]] = []
    monitor_records: list[dict[str, Any]] = []

    total_reps = len(models) * len(designs) * args.mc
    pbar = tqdm(total=total_reps, desc="Monte Carlo repetitions", unit="rep", disable=args.no_progress)
    for model in models:
        model_id = int(MODEL_SEED_ID.get(model, 9))
        params = OutcomeParameters(
            alpha=alpha,
            beta=beta,
            gamma=args.gamma,
            eta=args.eta,
            noise_sd=args.noise_sd,
            model=model,
            item_shock_sd=args.item_shock_sd,
            block_shock_sd=args.block_shock_sd,
            nonlinear_saturation_scale=args.nonlinear_saturation_scale,
            interaction_scale=args.interaction_scale,
            nonlinear_lag2_scale=args.nonlinear_lag2_scale,
            second_hop_scale=args.second_hop_scale,
            demand_substitution_scale=args.demand_substitution_scale,
            demand_treated_buffer=args.demand_treated_buffer,
            demand_temporal_scale=args.demand_temporal_scale,
            demand_cross_temporal_scale=args.demand_cross_temporal_scale,
            demand_saturation_scale=args.demand_saturation_scale,
        )
        tau = true_tau(A, params, weights=W, burn_in=1)
        est_values_by_design = {
            design.name: {"weighted_ht": [], "hajek": [], "dim": []}
            for design in designs
        }
        for r in range(args.mc):
            assignment_seed = make_seed(args.seed, model_id, r, "assignment")
            outcome_seed = make_seed(args.seed, model_id, r, "outcome")
            baseline_seed = make_seed(args.seed, model_id, r, "baseline")
            alpha_hat = make_alpha_adjustment(
                alpha,
                seed=baseline_seed,
                item_sd=args.alpha_hat_item_sd,
                block_sd=args.alpha_hat_block_sd,
                cell_sd=args.alpha_hat_cell_sd,
            )
            for design in designs:
                # All designs in the same (model, repetition) share the same
                # assignment/outcome/baseline seeds. The sampler itself is
                # design-specific, so assignments differ by design distribution,
                # but the Monte Carlo random environment is paired.
                Z = sample_design(design, n, b, args.p, seed=assignment_seed)
                Y = generate_observed_outcomes(Z, A, params, seed=outcome_seed)
                ht_val = weighted_ht(Z, Y, alpha=alpha_hat, p=args.p, weights=W, burn_in=1)
                hajek_val = hajek_difference(Z, Y, alpha=alpha_hat, weights=W, burn_in=1)
                dim_val = difference_in_means(Z, Y, alpha=alpha_hat, burn_in=1)
                est_values_by_design[design.name]["weighted_ht"].append(float(ht_val))
                est_values_by_design[design.name]["hajek"].append(float(hajek_val))
                est_values_by_design[design.name]["dim"].append(float(dim_val))
                draw_records.append({
                    "model": model,
                    "model_seed_id": int(model_id),
                    "design": design.name,
                    "rep": r,
                    "tau_true": float(tau),
                    "weighted_ht": float(ht_val),
                    "hajek": float(hajek_val),
                    "dim": float(dim_val),
                    "weighted_ht_error": float(ht_val - tau),
                    "hajek_error": float(hajek_val - tau),
                    "dim_error": float(dim_val - tau),
                    "base_seed": int(args.seed),
                    "assignment_seed": int(assignment_seed),
                    "outcome_seed": int(outcome_seed),
                    "baseline_seed": int(baseline_seed),
                    "d_item": int(args.d_item),
                    "d_time": int(d_time_eff),
                })
                if r == args.monitor_repetition:
                    diag = assignment_diagnostics(Z)
                    diag.update({
                        "model": model,
                        "model_seed_id": int(model_id),
                        "design": design.name,
                        "rep": int(r),
                        "tau_true": float(tau),
                        "weighted_ht": float(ht_val),
                        "hajek": float(hajek_val),
                        "dim": float(dim_val),
                        "weighted_ht_error": float(ht_val - tau),
                        "hajek_error": float(hajek_val - tau),
                        "dim_error": float(dim_val - tau),
                        "base_seed": int(args.seed),
                        "assignment_seed": int(assignment_seed),
                        "outcome_seed": int(outcome_seed),
                        "baseline_seed": int(baseline_seed),
                    })
                    monitor_records.append(diag)
                pbar.update(1)
        for design in designs:
            est_values = est_values_by_design[design.name]
            for estimator_name, vals_list in est_values.items():
                vals = np.asarray(vals_list, dtype=float)
                summary_records.append({
                    "model": model,
                    "model_seed_id": int(model_id),
                    "design": design.name,
                    "estimator": estimator_name,
                    "tau_true": float(tau),
                    "mean_est": float(np.nanmean(vals)),
                    "bias": float(np.nanmean(vals) - tau),
                    "std": float(np.nanstd(vals, ddof=1)) if np.sum(np.isfinite(vals)) > 1 else 0.0,
                    "mse": float(np.nanmean((vals - tau) ** 2)),
                    "rmse": float(np.sqrt(np.nanmean((vals - tau) ** 2))),
                    "mc": int(args.mc),
                    "p": float(args.p),
                    "N": int(n),
                    "B": int(b),
                    "d_item": int(args.d_item),
                    "d_time": int(d_time_eff),
                    "cluster_resolution": float(args.cluster_resolution),
                    "base_seed": int(args.seed),
                })
    pbar.close()

    res = sort_summary(pd.DataFrame(summary_records))
    draws = pd.DataFrame(draw_records)
    monitor_df = pd.DataFrame(monitor_records)

    if args.append_results:
        selected_set = set(selected_design_names)
        if (out_dir / "simulation_summary.csv").exists():
            old_res = pd.read_csv(out_dir / "simulation_summary.csv")
            if "design" in old_res.columns:
                old_res = old_res[~old_res["design"].astype(str).isin(selected_set)]
            res = sort_summary(pd.concat([old_res, res], ignore_index=True, sort=False))
        if (out_dir / "simulation_draws.csv").exists():
            old_draws = pd.read_csv(out_dir / "simulation_draws.csv")
            if "design" in old_draws.columns:
                old_draws = old_draws[~old_draws["design"].astype(str).isin(selected_set)]
            draws = pd.concat([old_draws, draws], ignore_index=True, sort=False)
        if (out_dir / "monitor_repetition_diagnostics.csv").exists():
            old_mon = pd.read_csv(out_dir / "monitor_repetition_diagnostics.csv")
            if "design" in old_mon.columns:
                old_mon = old_mon[~old_mon["design"].astype(str).isin(selected_set)]
            monitor_df = pd.concat([old_mon, monitor_df], ignore_index=True, sort=False)

    primary = res[res["estimator"] == "weighted_ht"].reset_index(drop=True)
    res.to_csv(out_dir / "simulation_summary.csv", index=False)
    primary.to_csv(out_dir / "simulation_summary_primary.csv", index=False)
    write_wide_summary(res, out_dir)
    draws.to_csv(out_dir / "simulation_draws.csv", index=False)
    monitor_df.to_csv(out_dir / "monitor_repetition_diagnostics.csv", index=False)
    write_output_manifest(out_dir)
    print(res)
    print(f"Saved simulation outputs to {out_dir}")
    print(f"Summary table: {out_dir / 'simulation_summary.csv'}")
    print(f"Primary grouped summary: {out_dir / 'simulation_summary_primary.csv'}")
    print(f"Wide summary by model: {out_dir / 'simulation_summary_wide_by_model.csv'}")
    if optimized_for_monitoring:
        print(f"Optimizer trajectory: {out_dir / 'optimization_trajectory.csv'}")


if __name__ == "__main__":
    main()
