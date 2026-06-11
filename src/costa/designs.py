from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import os
import numpy as np
import torch
from scipy import sparse
from tqdm.auto import tqdm

# Keep prototypes reliable in constrained environments. Override with COSTA_TORCH_THREADS.
torch.set_num_threads(int(os.environ.get("COSTA_TORCH_THREADS", "1")))

from .gaussian_copula import (
    CopulaMap,
    normalize_rows_torch,
    sample_item_static_gaussian_copula,
    sample_kronecker_gaussian_copula,
)
from .graph import ensure_csr, graph_edges
from .clustering import cluster_item_graph_leiden


@dataclass
class OptimizedDesign:
    name: str
    item_vecs: Optional[np.ndarray] = None
    time_vecs: Optional[np.ndarray] = None
    cell_vecs: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None
    cluster_info: Optional[dict[str, Any]] = None
    objective_history: Optional[list[dict[str, Any]]] = None
    p: float = 0.5


def sample_independent(n_items: int, n_blocks: int, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n_items, n_blocks)) < p).astype(np.int8)


def sample_rbsd_approx(n_items: int, n_blocks: int, p: float, seed: int, n_swaps: int = 2000) -> np.ndarray:
    """Approximate regular balanced switchback design.

    For B=4 and p=1/2 this gives exactly two treated blocks per item and then
    swaps entries to keep block totals close to pN. For general p, row totals are
    randomized between floor(pB) and ceil(pB), so the average row exposure is p.
    """
    rng = np.random.default_rng(seed)
    x = p * n_blocks
    lo = int(np.floor(x))
    hi = int(np.ceil(x))
    prob_hi = x - lo
    Z = np.zeros((n_items, n_blocks), dtype=np.int8)
    for i in range(n_items):
        k = hi if rng.random() < prob_hi else lo
        k = min(max(k, 0), n_blocks)
        if k > 0:
            Z[i, rng.choice(n_blocks, size=k, replace=False)] = 1

    target = int(round(p * n_items))
    for _ in range(n_swaps):
        col = Z.sum(axis=0)
        high = int(np.argmax(col - target))
        low = int(np.argmin(col - target))
        if col[high] <= target + 1 or col[low] >= target - 1:
            break
        candidates = np.flatnonzero((Z[:, high] == 1) & (Z[:, low] == 0))
        if len(candidates) == 0:
            break
        i = int(rng.choice(candidates))
        Z[i, high] = 0
        Z[i, low] = 1
    return Z


def sample_cluster_randomization(cluster_labels: np.ndarray, n_blocks: int, p: float, seed: int) -> np.ndarray:
    """Cluster-level item-block Bernoulli randomization."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(cluster_labels, dtype=np.int64)
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    cluster_z = (rng.random((n_clusters, n_blocks)) < p).astype(np.int8)
    return cluster_z[labels, :]


def make_cluster_leiden_design(
    A,
    p: float = 0.5,
    resolution: float = 1.0,
    seed: int = 123,
    fallback: str = "louvain",
) -> OptimizedDesign:
    labels, info = cluster_item_graph_leiden(A, resolution=resolution, seed=seed, fallback=fallback)
    return OptimizedDesign(name="cluster_leiden", cluster_labels=labels, cluster_info=info, p=p)


def _negative_pair_indices(n_items: int, n_pairs: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n_items, size=n_pairs)
    b = rng.integers(0, n_items, size=n_pairs)
    mask = a != b
    return a[mask], b[mask]


def _to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def _record_step(step: int, monitor_every: int, total_steps: int) -> bool:
    return (step % max(1, int(monitor_every)) == 0) or (step == total_steps - 1)


def _cluster_graph_from_labels(A, cluster_labels: np.ndarray) -> sparse.csr_matrix:
    """Aggregate the item competition graph to a directed cluster graph.

    C[k,l] = sum_{i in cluster k, j in cluster l} A[i,j]. Within-cluster edges
    are dropped, because cluster-level assignment already makes their normalized
    covariance equal to one and hence they do not contribute to the COSTA bias
    proxy.
    """
    labels = np.asarray(cluster_labels, dtype=np.int64)
    K = int(labels.max()) + 1 if labels.size else 0
    A_csr = ensure_csr(A).tocoo()
    if K == 0 or A_csr.nnz == 0:
        return sparse.csr_matrix((K, K), dtype=np.float32)
    row = labels[A_csr.row]
    col = labels[A_csr.col]
    mask = row != col
    C = sparse.coo_matrix((A_csr.data[mask], (row[mask], col[mask])), shape=(K, K), dtype=np.float32).tocsr()
    C.sum_duplicates()
    C.eliminate_zeros()
    return C.astype(np.float32)


def _cluster_sizes(cluster_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(cluster_labels, dtype=np.int64)
    K = int(labels.max()) + 1 if labels.size else 0
    if K == 0:
        return np.zeros(0, dtype=np.float32)
    return np.bincount(labels, minlength=K).astype(np.float32)


def optimize_static_ocd(
    A,
    p: float = 0.5,
    d_item: int = 16,
    steps: int = 300,
    lr: float = 0.05,
    lambda_var: float = 0.05,
    lambda_balance: float = 0.05,
    seed: int = 123,
    monitor_every: int = 1,
    show_progress: bool = False,
    progress: Optional[bool] = None,
) -> OptimizedDesign:
    """Optimize item covariance only; blocks are sampled independently."""
    if progress is not None:
        show_progress = bool(progress)
    monitor_every = max(1, int(monitor_every))
    torch.manual_seed(seed)

    n = A.shape[0]
    src, dst, w_np = graph_edges(A)
    if len(src) == 0:
        raise ValueError("Competition graph has no edges.")
    src_t = torch.as_tensor(src, dtype=torch.long)
    dst_t = torch.as_tensor(dst, dtype=torch.long)
    w_t = torch.as_tensor(w_np / (w_np.sum() + 1e-12), dtype=torch.float32)
    neg_a, neg_b = _negative_pair_indices(n, min(20000, max(1000, 5 * len(src))), seed + 1)
    neg_a_t = torch.as_tensor(neg_a, dtype=torch.long)
    neg_b_t = torch.as_tensor(neg_b, dtype=torch.long)
    cmap = CopulaMap.build(p)

    raw = torch.randn(n, d_item, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    hist: list[dict[str, Any]] = []
    iterator = range(steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Optimize static OCD", unit="step", leave=False)

    for step in iterator:
        opt.zero_grad()
        E = normalize_rows_torch(raw)
        rho_e = (E[src_t] * E[dst_t]).sum(dim=1).clamp(-0.999, 0.999)
        R_e = cmap.torch(rho_e)
        b_comp = torch.sum(w_t * (R_e - 1.0))
        var_proxy = torch.sum(w_t * (R_e + 1.0))
        rho_neg = (E[neg_a_t] * E[neg_b_t]).sum(dim=1).clamp(-0.999, 0.999)
        R_neg = cmap.torch(rho_neg)
        balance_proxy = torch.mean(R_neg).pow(2) + torch.mean(torch.relu(R_neg))
        mse_objective = b_comp.pow(2) + lambda_var * var_proxy
        regularized_loss = mse_objective + lambda_balance * balance_proxy
        regularized_loss.backward()
        opt.step()
        if _record_step(step, monitor_every, steps):
            hist.append({
                "design": "static_graph_ocd",
                "step": int(step),
                "mse_objective": _to_float(mse_objective),
                "regularized_loss": _to_float(regularized_loss),
                "b_comp": _to_float(b_comp),
                "b_temp": None,
                "var_proxy": _to_float(var_proxy),
                "mean_R_edge": _to_float(torch.mean(R_e)),
                "competition_bias_proxy": _to_float(b_comp),
                "competition_bias_sq": _to_float(b_comp.pow(2)),
                "temporal_bias_proxy": None,
                "temporal_bias_sq": None,
                "variance_proxy": _to_float(var_proxy),
                "balance_proxy": _to_float(balance_proxy),
                "switch_guard": None,
                "mean_edge_R": _to_float(torch.mean(R_e)),
                "mean_lag_R": None,
                "mean_negative_R": _to_float(torch.mean(R_neg)),
                "lr": float(lr),
                "d_item": int(d_item),
                "d_time": None,
            })
    E = normalize_rows_torch(raw).detach().cpu().numpy().astype(np.float32)
    return OptimizedDesign(name="static_graph_ocd", item_vecs=E, time_vecs=None, objective_history=hist, p=p)


def optimize_costa(
    A,
    p: float = 0.5,
    n_blocks: int = 4,
    d_item: int = 16,
    d_time: int = 3,
    steps: int = 400,
    lr: float = 0.05,
    w_comp: float = 1.0,
    w_temp: float = 1.0,
    lambda_var: float = 0.05,
    lambda_balance: float = 0.05,
    switch_min: float = 0.10,
    seed: int = 123,
    monitor_every: int = 1,
    show_progress: bool = False,
    progress: Optional[bool] = None,
) -> OptimizedDesign:
    """Optimize COSTA Kronecker item-time latent vectors."""
    if progress is not None:
        show_progress = bool(progress)
    monitor_every = max(1, int(monitor_every))
    torch.manual_seed(seed)

    n = A.shape[0]
    src, dst, w_np = graph_edges(A)
    if len(src) == 0:
        raise ValueError("Competition graph has no edges.")
    src_t = torch.as_tensor(src, dtype=torch.long)
    dst_t = torch.as_tensor(dst, dtype=torch.long)
    w_t = torch.as_tensor(w_np / (w_np.sum() + 1e-12), dtype=torch.float32)
    neg_a, neg_b = _negative_pair_indices(n, min(20000, max(1000, 5 * len(src))), seed + 1)
    neg_a_t = torch.as_tensor(neg_a, dtype=torch.long)
    neg_b_t = torch.as_tensor(neg_b, dtype=torch.long)
    cmap = CopulaMap.build(p)

    raw_item = torch.randn(n, d_item, requires_grad=True)
    raw_time = torch.randn(n_blocks, d_time, requires_grad=True)
    opt = torch.optim.Adam([raw_item, raw_time], lr=lr)
    q = 1.0 - p
    r_lag_max = 1.0 - switch_min / max(2.0 * p * q, 1e-9)
    r_lag_max = min(0.999, max(-0.999, r_lag_max))
    hist: list[dict[str, Any]] = []
    iterator = range(steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Optimize COSTA", unit="step", leave=False)

    for step in iterator:
        opt.zero_grad()
        E = normalize_rows_torch(raw_item)
        G = normalize_rows_torch(raw_time)
        rho_e = (E[src_t] * E[dst_t]).sum(dim=1).clamp(-0.999, 0.999)
        R_e = cmap.torch(rho_e)
        b_comp = torch.sum(w_t * (R_e - 1.0))
        rho_lag = (G[1:] * G[:-1]).sum(dim=1).clamp(-0.999, 0.999)
        R_lag = cmap.torch(rho_lag)
        b_temp = torch.mean(R_lag - 1.0)
        var_proxy = torch.sum(w_t * (R_e + 1.0)) + torch.mean(R_lag + 1.0)
        rho_neg = (E[neg_a_t] * E[neg_b_t]).sum(dim=1).clamp(-0.999, 0.999)
        R_neg = cmap.torch(rho_neg)
        balance_proxy = torch.mean(R_neg).pow(2) + torch.mean(torch.relu(R_neg))
        switch_guard = torch.relu(torch.mean(R_lag) - r_lag_max).pow(2)
        mse_objective = w_comp * b_comp.pow(2) + w_temp * b_temp.pow(2) + lambda_var * var_proxy
        regularized_loss = mse_objective + lambda_balance * balance_proxy + 10.0 * switch_guard
        regularized_loss.backward()
        opt.step()
        if _record_step(step, monitor_every, steps):
            hist.append({
                "design": "COSTA",
                "step": int(step),
                "mse_objective": _to_float(mse_objective),
                "regularized_loss": _to_float(regularized_loss),
                "b_comp": _to_float(b_comp),
                "b_temp": _to_float(b_temp),
                "var_proxy": _to_float(var_proxy),
                "mean_R_edge": _to_float(torch.mean(R_e)),
                "mean_R_lag": _to_float(torch.mean(R_lag)),
                "implied_lag_switch_rate": float(2.0 * p * q * (1.0 - _to_float(torch.mean(R_lag)))),
                "competition_bias_proxy": _to_float(b_comp),
                "competition_bias_sq": _to_float(b_comp.pow(2)),
                "temporal_bias_proxy": _to_float(b_temp),
                "temporal_bias_sq": _to_float(b_temp.pow(2)),
                "variance_proxy": _to_float(var_proxy),
                "balance_proxy": _to_float(balance_proxy),
                "switch_guard": _to_float(switch_guard),
                "mean_edge_R": _to_float(torch.mean(R_e)),
                "mean_lag_R": _to_float(torch.mean(R_lag)),
                "mean_negative_R": _to_float(torch.mean(R_neg)),
                "lr": float(lr),
                "d_item": int(d_item),
                "d_time": int(d_time),
                "d_effective": int(d_item * d_time),
            })
    E = normalize_rows_torch(raw_item).detach().cpu().numpy().astype(np.float32)
    G = normalize_rows_torch(raw_time).detach().cpu().numpy().astype(np.float32)
    return OptimizedDesign(name="COSTA", item_vecs=E, time_vecs=G, objective_history=hist, p=p)


def _sample_far_time_pairs(n_blocks: int, n_pairs: int, seed: int, min_gap: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Sample time-block pairs with distance at least ``min_gap``.

    These are nonlocal pairs for one-lag temporal interference.  If B is small
    and no such pair exists, returns empty arrays.
    """
    pairs = [(b, r) for b in range(n_blocks) for r in range(n_blocks) if abs(b - r) >= int(min_gap)]
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), size=max(1, int(n_pairs)))
    arr = np.asarray([pairs[i] for i in idx], dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def optimize_costa_ic(
    A,
    p: float = 0.5,
    n_blocks: int = 4,
    d_item: int = 16,
    d_time: int = 3,
    steps: int = 400,
    lr: float = 0.05,
    w_comp: float = 1.0,
    w_temp: float = 1.0,
    lambda_var: float = 0.05,
    lambda_balance: float = 0.05,
    switch_min: float = 0.10,
    lambda_switch: float = 10.0,
    lambda_local: float = 1.0,
    n_far_pairs: int = 50000,
    locality_time_gap: int = 2,
    seed: int = 123,
    monitor_every: int = 1,
    show_progress: bool = False,
    progress: Optional[bool] = None,
) -> OptimizedDesign:
    """Optimize inference-compatible COSTA with a product-graph locality penalty.

    COSTA-IC keeps the Kronecker item-time Gaussian-copula sampler but penalizes
    correlations on item and time pairs that are far under the one-hop product
    graph used in the CLT section.  The penalty is not needed for MSE design, but
    it makes the optimized assignment closer to a local-dependence design.

    The locality proxy has three components:

    * random non-edge item pairs, encouraging noncompetitors to be nearly uncorrelated;
    * non-adjacent time-block pairs, encouraging only one-lag temporal dependence;
    * random product pairs combining the two, encouraging far item-time cells to be uncorrelated.
    """
    if progress is not None:
        show_progress = bool(progress)
    monitor_every = max(1, int(monitor_every))
    torch.manual_seed(seed)

    n = A.shape[0]
    src, dst, w_np = graph_edges(A)
    if len(src) == 0:
        raise ValueError("Competition graph has no edges.")
    src_t = torch.as_tensor(src, dtype=torch.long)
    dst_t = torch.as_tensor(dst, dtype=torch.long)
    w_t = torch.as_tensor(w_np / (w_np.sum() + 1e-12), dtype=torch.float32)

    n_far = int(max(1000, min(int(n_far_pairs), max(1000, 10 * len(src)))))
    far_a, far_b = _negative_pair_indices(n, n_far, seed + 101)
    far_a_t = torch.as_tensor(far_a, dtype=torch.long)
    far_b_t = torch.as_tensor(far_b, dtype=torch.long)

    prod_a, prod_b = _negative_pair_indices(n, n_far, seed + 202)
    prod_tb, prod_tr = _sample_far_time_pairs(n_blocks, len(prod_a), seed + 303, min_gap=locality_time_gap)
    m_prod = min(len(prod_a), len(prod_tb))
    prod_a_t = torch.as_tensor(prod_a[:m_prod], dtype=torch.long)
    prod_b_t = torch.as_tensor(prod_b[:m_prod], dtype=torch.long)
    prod_tb_t = torch.as_tensor(prod_tb[:m_prod], dtype=torch.long)
    prod_tr_t = torch.as_tensor(prod_tr[:m_prod], dtype=torch.long)

    time_far_b, time_far_r = _sample_far_time_pairs(n_blocks, max(1, n_blocks * n_blocks), seed + 404, min_gap=locality_time_gap)
    time_far_b_t = torch.as_tensor(time_far_b, dtype=torch.long)
    time_far_r_t = torch.as_tensor(time_far_r, dtype=torch.long)

    cmap = CopulaMap.build(p)
    raw_item = torch.randn(n, d_item, requires_grad=True)
    raw_time = torch.randn(n_blocks, d_time, requires_grad=True)
    opt = torch.optim.Adam([raw_item, raw_time], lr=lr)
    q = 1.0 - p
    r_lag_max = 1.0 - switch_min / max(2.0 * p * q, 1e-9)
    r_lag_max = min(0.999, max(-0.999, r_lag_max))
    hist: list[dict[str, Any]] = []
    iterator = range(steps)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Optimize COSTA-IC lambda={lambda_local:g}", unit="step", leave=False)

    for step in iterator:
        opt.zero_grad()
        E = normalize_rows_torch(raw_item)
        G = normalize_rows_torch(raw_time)
        rho_e = (E[src_t] * E[dst_t]).sum(dim=1).clamp(-0.999, 0.999)
        R_e = cmap.torch(rho_e)
        b_comp = torch.sum(w_t * (R_e - 1.0))
        rho_lag = (G[1:] * G[:-1]).sum(dim=1).clamp(-0.999, 0.999)
        R_lag = cmap.torch(rho_lag)
        b_temp = torch.mean(R_lag - 1.0)
        var_proxy = torch.sum(w_t * (R_e + 1.0)) + torch.mean(R_lag + 1.0)

        rho_far_item = (E[far_a_t] * E[far_b_t]).sum(dim=1).clamp(-0.999, 0.999) if len(far_a_t) else torch.zeros(0)
        R_far_item = cmap.torch(rho_far_item) if rho_far_item.numel() else torch.zeros(0)
        item_locality = torch.mean(R_far_item.pow(2)) if R_far_item.numel() else torch.tensor(0.0)

        if len(time_far_b_t):
            rho_far_time = (G[time_far_b_t] * G[time_far_r_t]).sum(dim=1).clamp(-0.999, 0.999)
            R_far_time = cmap.torch(rho_far_time)
            time_locality = torch.mean(R_far_time.pow(2))
            mean_far_time_abs = torch.mean(torch.abs(R_far_time))
        else:
            time_locality = torch.tensor(0.0)
            mean_far_time_abs = torch.tensor(0.0)

        if len(prod_a_t):
            rho_i = (E[prod_a_t] * E[prod_b_t]).sum(dim=1)
            rho_t = (G[prod_tb_t] * G[prod_tr_t]).sum(dim=1)
            rho_prod = (rho_i * rho_t).clamp(-0.999, 0.999)
            R_prod = cmap.torch(rho_prod)
            product_locality = torch.mean(R_prod.pow(2))
            mean_far_product_abs = torch.mean(torch.abs(R_prod))
        else:
            product_locality = torch.tensor(0.0)
            mean_far_product_abs = torch.tensor(0.0)

        locality_proxy = item_locality + time_locality + product_locality
        balance_proxy = torch.mean(R_far_item).pow(2) + torch.mean(torch.relu(R_far_item)) if R_far_item.numel() else torch.tensor(0.0)
        switch_guard = torch.relu(torch.mean(R_lag) - r_lag_max).pow(2)
        mse_objective = w_comp * b_comp.pow(2) + w_temp * b_temp.pow(2) + lambda_var * var_proxy
        regularized_loss = mse_objective + lambda_balance * balance_proxy + float(lambda_switch) * switch_guard + float(lambda_local) * locality_proxy
        regularized_loss.backward()
        opt.step()
        if _record_step(step, monitor_every, steps):
            hist.append({
                "design": "COSTA_IC",
                "step": int(step),
                "mse_objective": _to_float(mse_objective),
                "regularized_loss": _to_float(regularized_loss),
                "b_comp": _to_float(b_comp),
                "b_temp": _to_float(b_temp),
                "var_proxy": _to_float(var_proxy),
                "locality_proxy": _to_float(locality_proxy),
                "item_locality_proxy": _to_float(item_locality),
                "time_locality_proxy": _to_float(time_locality),
                "product_locality_proxy": _to_float(product_locality),
                "lambda_local": float(lambda_local),
                "lambda_switch": float(lambda_switch),
                "mean_R_edge": _to_float(torch.mean(R_e)),
                "mean_R_lag": _to_float(torch.mean(R_lag)),
                "mean_far_item_R_abs": _to_float(torch.mean(torch.abs(R_far_item))) if R_far_item.numel() else 0.0,
                "mean_far_time_R_abs": _to_float(mean_far_time_abs),
                "mean_far_product_R_abs": _to_float(mean_far_product_abs),
                "implied_lag_switch_rate": float(2.0 * p * q * (1.0 - _to_float(torch.mean(R_lag)))),
                "competition_bias_proxy": _to_float(b_comp),
                "competition_bias_sq": _to_float(b_comp.pow(2)),
                "temporal_bias_proxy": _to_float(b_temp),
                "temporal_bias_sq": _to_float(b_temp.pow(2)),
                "variance_proxy": _to_float(var_proxy),
                "balance_proxy": _to_float(balance_proxy),
                "switch_guard": _to_float(switch_guard),
                "mean_edge_R": _to_float(torch.mean(R_e)),
                "mean_lag_R": _to_float(torch.mean(R_lag)),
                "mean_negative_R": _to_float(torch.mean(R_far_item)) if R_far_item.numel() else 0.0,
                "lr": float(lr),
                "d_item": int(d_item),
                "d_time": int(d_time),
                "d_effective": int(d_item * d_time),
            })
    E = normalize_rows_torch(raw_item).detach().cpu().numpy().astype(np.float32)
    G = normalize_rows_torch(raw_time).detach().cpu().numpy().astype(np.float32)
    return OptimizedDesign(name="COSTA_IC", item_vecs=E, time_vecs=G, objective_history=hist, p=p)


def optimize_cluster_costa_kro(
    A,
    cluster_labels: np.ndarray,
    p: float = 0.5,
    n_blocks: int = 4,
    d_item: int = 16,
    d_time: int = 3,
    steps: int = 400,
    lr: float = 0.05,
    w_comp: float = 1.0,
    w_temp: float = 1.0,
    lambda_var: float = 0.05,
    lambda_balance: float = 0.05,
    switch_min: float = 0.10,
    seed: int = 123,
    monitor_every: int = 1,
    show_progress: bool = False,
) -> OptimizedDesign:
    """Cluster-level COSTA with Kronecker cluster-time parameterization.

    The item graph is first coarsened by a fixed clustering. We then optimize
    cluster latent vectors u_k and time vectors v_b. All items in the same
    cluster share the sampled cluster-block treatment.
    """
    monitor_every = max(1, int(monitor_every))
    torch.manual_seed(seed)
    labels = np.asarray(cluster_labels, dtype=np.int64)
    C = _cluster_graph_from_labels(A, labels)
    K = int(labels.max()) + 1 if labels.size else 0
    src, dst, w_np = graph_edges(C)
    if len(src) == 0:
        # Degenerate graph: use singleton/uniform cluster edge placeholders.
        src = np.arange(max(K - 1, 1), dtype=np.int64)
        dst = np.arange(1, max(K, 2), dtype=np.int64)[: len(src)]
        w_np = np.ones(len(src), dtype=np.float32)
    src_t = torch.as_tensor(src, dtype=torch.long)
    dst_t = torch.as_tensor(dst, dtype=torch.long)
    w_t = torch.as_tensor(w_np / (w_np.sum() + 1e-12), dtype=torch.float32)
    sizes = _cluster_sizes(labels)
    size_w = sizes / (sizes.sum() + 1e-12)
    size_t = torch.as_tensor(size_w, dtype=torch.float32)
    neg_a, neg_b = _negative_pair_indices(K, min(20000, max(1000, 5 * len(src))), seed + 1)
    neg_a_t = torch.as_tensor(neg_a, dtype=torch.long)
    neg_b_t = torch.as_tensor(neg_b, dtype=torch.long)
    cmap = CopulaMap.build(p)
    raw_cluster = torch.randn(K, d_item, requires_grad=True)
    raw_time = torch.randn(n_blocks, d_time, requires_grad=True)
    opt = torch.optim.Adam([raw_cluster, raw_time], lr=lr)
    q = 1.0 - p
    r_lag_max = 1.0 - switch_min / max(2.0 * p * q, 1e-9)
    r_lag_max = min(0.999, max(-0.999, r_lag_max))
    hist: list[dict[str, Any]] = []
    iterator = range(steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Optimize COSTA cluster kro", unit="step", leave=False)
    for step in iterator:
        opt.zero_grad()
        E = normalize_rows_torch(raw_cluster)
        G = normalize_rows_torch(raw_time)
        rho_e = (E[src_t] * E[dst_t]).sum(dim=1).clamp(-0.999, 0.999)
        R_e = cmap.torch(rho_e)
        b_comp = torch.sum(w_t * (R_e - 1.0))
        rho_lag = (G[1:] * G[:-1]).sum(dim=1).clamp(-0.999, 0.999)
        R_lag = cmap.torch(rho_lag)
        b_temp = torch.mean(R_lag - 1.0)
        var_proxy = torch.sum(w_t * (R_e + 1.0)) + torch.mean(R_lag + 1.0)
        rho_neg = (E[neg_a_t] * E[neg_b_t]).sum(dim=1).clamp(-0.999, 0.999)
        R_neg = cmap.torch(rho_neg)
        balance_proxy = torch.mean(R_neg).pow(2) + torch.mean(torch.relu(R_neg))
        switch_guard = torch.relu(torch.mean(R_lag) - r_lag_max).pow(2)
        mse_objective = w_comp * b_comp.pow(2) + w_temp * b_temp.pow(2) + lambda_var * var_proxy
        regularized_loss = mse_objective + lambda_balance * balance_proxy + 10.0 * switch_guard
        regularized_loss.backward()
        opt.step()
        if _record_step(step, monitor_every, steps):
            hist.append({
                "design": "COSTA_cluster_kro",
                "step": int(step),
                "mse_objective": _to_float(mse_objective),
                "regularized_loss": _to_float(regularized_loss),
                "b_comp": _to_float(b_comp),
                "b_temp": _to_float(b_temp),
                "var_proxy": _to_float(var_proxy),
                "mean_R_edge": _to_float(torch.mean(R_e)),
                "mean_R_lag": _to_float(torch.mean(R_lag)),
                "competition_bias_proxy": _to_float(b_comp),
                "competition_bias_sq": _to_float(b_comp.pow(2)),
                "temporal_bias_proxy": _to_float(b_temp),
                "temporal_bias_sq": _to_float(b_temp.pow(2)),
                "variance_proxy": _to_float(var_proxy),
                "balance_proxy": _to_float(balance_proxy),
                "switch_guard": _to_float(switch_guard),
                "mean_edge_R": _to_float(torch.mean(R_e)),
                "mean_lag_R": _to_float(torch.mean(R_lag)),
                "mean_negative_R": _to_float(torch.mean(R_neg)),
                "lr": float(lr),
                "d_item": int(d_item),
                "d_time": int(d_time),
                "d_effective": int(d_item * d_time),
                "n_clusters": int(K),
            })
    E = normalize_rows_torch(raw_cluster).detach().cpu().numpy().astype(np.float32)
    G = normalize_rows_torch(raw_time).detach().cpu().numpy().astype(np.float32)
    return OptimizedDesign(
        name="COSTA_cluster_kro",
        item_vecs=E,
        time_vecs=G,
        cluster_labels=labels,
        objective_history=hist,
        p=p,
    )


def optimize_cluster_costa_no_kro(
    A,
    cluster_labels: np.ndarray,
    p: float = 0.5,
    n_blocks: int = 4,
    d_latent: int = 16,
    steps: int = 400,
    lr: float = 0.05,
    w_comp: float = 1.0,
    w_temp: float = 1.0,
    lambda_var: float = 0.05,
    lambda_balance: float = 0.05,
    switch_min: float = 0.10,
    seed: int = 123,
    monitor_every: int = 1,
    show_progress: bool = False,
) -> OptimizedDesign:
    """Cluster-level COSTA without Kronecker product.

    After clustering, each cluster-block cell (k,b) receives its own latent
    vector h_{k,b}. This is a full spatio-temporal covariance optimization at
    the cluster-time level. It is feasible because the number of clusters is far
    smaller than the number of items.
    """
    monitor_every = max(1, int(monitor_every))
    torch.manual_seed(seed)
    labels = np.asarray(cluster_labels, dtype=np.int64)
    C = _cluster_graph_from_labels(A, labels)
    K = int(labels.max()) + 1 if labels.size else 0
    src, dst, w_np = graph_edges(C)
    if len(src) == 0:
        src = np.arange(max(K - 1, 1), dtype=np.int64)
        dst = np.arange(1, max(K, 2), dtype=np.int64)[: len(src)]
        w_np = np.ones(len(src), dtype=np.float32)
    src_t = torch.as_tensor(src, dtype=torch.long)
    dst_t = torch.as_tensor(dst, dtype=torch.long)
    w_t = torch.as_tensor(w_np / (w_np.sum() + 1e-12), dtype=torch.float32)
    sizes = _cluster_sizes(labels)
    size_w = sizes / (sizes.sum() + 1e-12)
    size_t = torch.as_tensor(size_w, dtype=torch.float32)
    cmap = CopulaMap.build(p)
    raw_cell = torch.randn(K, n_blocks, d_latent, requires_grad=True)
    opt = torch.optim.Adam([raw_cell], lr=lr)
    q = 1.0 - p
    r_lag_max = 1.0 - switch_min / max(2.0 * p * q, 1e-9)
    r_lag_max = min(0.999, max(-0.999, r_lag_max))
    rng = np.random.default_rng(seed + 1)
    neg_k1 = rng.integers(0, K, size=min(20000, max(1000, 5 * len(src))))
    neg_b1 = rng.integers(0, n_blocks, size=len(neg_k1))
    neg_k2 = rng.integers(0, K, size=len(neg_k1))
    neg_b2 = rng.integers(0, n_blocks, size=len(neg_k1))
    mask = (neg_k1 != neg_k2) | (neg_b1 != neg_b2)
    nk1_t = torch.as_tensor(neg_k1[mask], dtype=torch.long)
    nb1_t = torch.as_tensor(neg_b1[mask], dtype=torch.long)
    nk2_t = torch.as_tensor(neg_k2[mask], dtype=torch.long)
    nb2_t = torch.as_tensor(neg_b2[mask], dtype=torch.long)
    hist: list[dict[str, Any]] = []
    iterator = range(steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Optimize COSTA cluster no-kro", unit="step", leave=False)
    for step in iterator:
        opt.zero_grad()
        H = raw_cell / torch.clamp(torch.linalg.norm(raw_cell, dim=2, keepdim=True), min=1e-12)
        # Average competition bias over post-burn-in blocks.
        comp_terms = []
        for b in range(1, n_blocks):
            rho = (H[src_t, b, :] * H[dst_t, b, :]).sum(dim=1).clamp(-0.999, 0.999)
            comp_terms.append(cmap.torch(rho))
        R_e = torch.cat(comp_terms) if comp_terms else torch.zeros(0)
        w_rep = w_t.repeat(max(1, n_blocks - 1))
        b_comp = torch.sum(w_rep * (R_e - 1.0))
        # Temporal term: same cluster, adjacent blocks, weighted by cluster size.
        lag_terms = []
        for b in range(1, n_blocks):
            rho_l = (H[:, b, :] * H[:, b - 1, :]).sum(dim=1).clamp(-0.999, 0.999)
            lag_terms.append(cmap.torch(rho_l))
        R_lag_mat = torch.stack(lag_terms, dim=1) if lag_terms else torch.zeros((K, 0))
        if R_lag_mat.numel() > 0:
            R_lag_mean_by_cluster = R_lag_mat.mean(dim=1)
            b_temp = torch.sum(size_t * (R_lag_mean_by_cluster - 1.0))
            mean_R_lag = torch.sum(size_t * R_lag_mean_by_cluster)
        else:
            b_temp = torch.tensor(0.0)
            mean_R_lag = torch.tensor(0.0)
        var_proxy = torch.sum(w_rep * (R_e + 1.0)) + (mean_R_lag + 1.0)
        rho_neg = (H[nk1_t, nb1_t, :] * H[nk2_t, nb2_t, :]).sum(dim=1).clamp(-0.999, 0.999)
        R_neg = cmap.torch(rho_neg)
        balance_proxy = torch.mean(R_neg).pow(2) + torch.mean(torch.relu(R_neg))
        switch_guard = torch.relu(mean_R_lag - r_lag_max).pow(2)
        mse_objective = w_comp * b_comp.pow(2) + w_temp * b_temp.pow(2) + lambda_var * var_proxy
        regularized_loss = mse_objective + lambda_balance * balance_proxy + 10.0 * switch_guard
        regularized_loss.backward()
        opt.step()
        if _record_step(step, monitor_every, steps):
            hist.append({
                "design": "COSTA_cluster_no_kro",
                "step": int(step),
                "mse_objective": _to_float(mse_objective),
                "regularized_loss": _to_float(regularized_loss),
                "b_comp": _to_float(b_comp),
                "b_temp": _to_float(b_temp),
                "var_proxy": _to_float(var_proxy),
                "mean_R_edge": _to_float(torch.mean(R_e)) if R_e.numel() else None,
                "mean_R_lag": _to_float(mean_R_lag),
                "competition_bias_proxy": _to_float(b_comp),
                "competition_bias_sq": _to_float(b_comp.pow(2)),
                "temporal_bias_proxy": _to_float(b_temp),
                "temporal_bias_sq": _to_float(b_temp.pow(2)),
                "variance_proxy": _to_float(var_proxy),
                "balance_proxy": _to_float(balance_proxy),
                "switch_guard": _to_float(switch_guard),
                "mean_edge_R": _to_float(torch.mean(R_e)) if R_e.numel() else None,
                "mean_lag_R": _to_float(mean_R_lag),
                "mean_negative_R": _to_float(torch.mean(R_neg)),
                "lr": float(lr),
                "d_item": int(d_latent),
                "d_time": None,
                "d_effective": int(d_latent),
                "n_clusters": int(K),
            })
    H_np = (raw_cell / torch.clamp(torch.linalg.norm(raw_cell, dim=2, keepdim=True), min=1e-12)).detach().cpu().numpy().astype(np.float32)
    return OptimizedDesign(
        name="COSTA_cluster_no_kro",
        cell_vecs=H_np,
        cluster_labels=labels,
        objective_history=hist,
        p=p,
    )


def sample_cluster_kronecker_gaussian_copula(
    cluster_labels: np.ndarray,
    cluster_vecs: np.ndarray,
    time_vecs: np.ndarray,
    p: float,
    seed: int,
) -> np.ndarray:
    Zc = sample_kronecker_gaussian_copula(cluster_vecs, time_vecs, p, seed=seed)
    return Zc[np.asarray(cluster_labels, dtype=np.int64), :]


def sample_cluster_full_gaussian_copula(
    cluster_labels: np.ndarray,
    cell_vecs: np.ndarray,
    p: float,
    seed: int,
) -> np.ndarray:
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    H = np.asarray(cell_vecs, dtype=float)
    K, B, d = H.shape
    eps = rng.standard_normal(d)
    scores = H.reshape(K * B, d) @ eps
    Zc = (scores.reshape(K, B) <= norm.ppf(p)).astype(np.int8)
    return Zc[np.asarray(cluster_labels, dtype=np.int64), :]


def sample_design(design: OptimizedDesign, n_items: int, n_blocks: int, p: float, seed: int) -> np.ndarray:
    if design.name == "independent":
        return sample_independent(n_items, n_blocks, p, seed)
    if design.name == "rbsd_approx":
        return sample_rbsd_approx(n_items, n_blocks, p, seed)
    if design.name == "cluster_leiden":
        if design.cluster_labels is None:
            raise ValueError("cluster_leiden requires cluster_labels")
        return sample_cluster_randomization(design.cluster_labels, n_blocks, p, seed=seed)
    if design.name == "static_graph_ocd":
        if design.item_vecs is None:
            raise ValueError("static_graph_ocd requires item_vecs")
        return sample_item_static_gaussian_copula(design.item_vecs, n_blocks, p, seed=seed)
    if design.name == "COSTA" or design.name == "COSTA_IC" or design.name.startswith("COSTA_IC_lam"):
        if design.item_vecs is None or design.time_vecs is None:
            raise ValueError(f"{design.name} requires item_vecs and time_vecs")
        return sample_kronecker_gaussian_copula(design.item_vecs, design.time_vecs, p, seed=seed)
    if design.name == "COSTA_cluster_kro":
        if design.cluster_labels is None or design.item_vecs is None or design.time_vecs is None:
            raise ValueError("COSTA_cluster_kro requires cluster_labels, item_vecs, and time_vecs")
        return sample_cluster_kronecker_gaussian_copula(design.cluster_labels, design.item_vecs, design.time_vecs, p, seed=seed)
    if design.name == "COSTA_cluster_no_kro":
        if design.cluster_labels is None or design.cell_vecs is None:
            raise ValueError("COSTA_cluster_no_kro requires cluster_labels and cell_vecs")
        return sample_cluster_full_gaussian_copula(design.cluster_labels, design.cell_vecs, p, seed=seed)
    raise ValueError(f"Unknown design: {design.name}")
