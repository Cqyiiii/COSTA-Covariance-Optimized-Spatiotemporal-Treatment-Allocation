from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import norm

from .gaussian_copula import CopulaMap
from .graph import ensure_csr


def copula_map_numpy(rho: np.ndarray, p: float, cmap: CopulaMap | None = None) -> np.ndarray:
    """Fast interpolation version of the Bernoulli-p normalized covariance map."""
    rho = np.asarray(rho, dtype=float)
    if abs(float(p) - 0.5) < 1e-12:
        return (2.0 / np.pi) * np.arcsin(np.clip(rho, -0.999999, 0.999999))
    cmap = cmap or CopulaMap.build(float(p))
    return np.interp(np.clip(rho, cmap.rho_grid[0], cmap.rho_grid[-1]), cmap.rho_grid, cmap.r_grid)


def ht_contributions(
    Z: np.ndarray,
    Y: np.ndarray,
    alpha_hat: np.ndarray,
    p: float,
    weights: np.ndarray,
    burn_in: int = 1,
) -> tuple[float, np.ndarray, float]:
    """Return weighted HT estimate, centered score contributions, and total weight.

    The raw contribution is
        g_a = w_a psi_a (Y_a-alpha_hat_a)
    on post-burn-in cells and zero otherwise.  For HAC we use
        xi_a = g_a - w_a * tau_hat
    on post-burn-in cells, so that sum_a xi_a = 0 over the weighted outcome
    cells.  This matches the finite-population product-graph HAC formula in the
    theory section.
    """
    q = 1.0 - float(p)
    psi = (Z.astype(float) - float(p)) / (float(p) * q)
    mask = np.zeros_like(Z, dtype=bool)
    mask[:, int(burn_in):] = True
    w = np.where(mask, weights.astype(float), 0.0)
    W = float(np.sum(w) + 1e-12)
    g = w * psi * (Y - alpha_hat)
    tau_hat = float(np.sum(g) / W)
    xi = np.where(mask, g - w * tau_hat, 0.0)
    return tau_hat, xi, W


def _undirected_support_edges(A) -> tuple[np.ndarray, np.ndarray]:
    A_csr = ensure_csr(A)
    S = ((A_csr != 0).astype(np.int8) + (A_csr.T != 0).astype(np.int8)).tocoo()
    rows = S.row.astype(np.int64)
    cols = S.col.astype(np.int64)
    mask = rows < cols
    return rows[mask], cols[mask]


def product_graph_hac_variance(
    xi: np.ndarray,
    A,
    W: float,
    burn_in: int = 1,
    include_temporal: bool = True,
    mode: str = "conservative",
) -> dict[str, float]:
    """Product-graph HAC variance estimate for item-block HT contributions.

    This intentionally uses the h=1 product-neighborhood for the illustration:
    diagonal terms, same-block competition edges, and same-item adjacent-block
    temporal edges.  ``mode='signed'`` uses signed cross-products.  The default
    ``mode='conservative'`` uses absolute local cross-products; it is more stable
    for finite Monte Carlo diagnostics and aligns with the conservative-HAC
    interpretation in the theory section.
    """
    xi = np.asarray(xi, dtype=float)
    n, b = xi.shape
    burn_in = int(burn_in)
    diag = float(np.sum(xi[:, burn_in:] ** 2))
    ei, ej = _undirected_support_edges(A)
    comp_signed = 0.0
    comp_abs = 0.0
    if len(ei):
        for t in range(burn_in, b):
            prod = xi[ei, t] * xi[ej, t]
            comp_signed += float(2.0 * np.sum(prod))
            comp_abs += float(2.0 * np.sum(np.abs(prod)))
    temp_signed = 0.0
    temp_abs = 0.0
    if include_temporal and b > burn_in + 1:
        for t in range(burn_in + 1, b):
            prod = xi[:, t] * xi[:, t - 1]
            temp_signed += float(2.0 * np.sum(prod))
            temp_abs += float(2.0 * np.sum(np.abs(prod)))
    signed = (diag + comp_signed + temp_signed) / (float(W) ** 2)
    conservative = (diag + comp_abs + temp_abs) / (float(W) ** 2)
    diagonal = diag / (float(W) ** 2)
    selected = conservative if str(mode).lower().startswith("cons") else signed
    if not np.isfinite(selected) or selected <= 0:
        selected = max(diagonal, 1e-18)
    return {
        "variance": float(selected),
        "se": float(np.sqrt(max(selected, 0.0))),
        "signed_variance": float(signed),
        "conservative_variance": float(conservative),
        "diagonal_variance": float(diagonal),
        "diag_sum": float(diag),
        "competition_cross_signed": float(comp_signed),
        "competition_cross_abs": float(comp_abs),
        "temporal_cross_signed": float(temp_signed),
        "temporal_cross_abs": float(temp_abs),
    }


def design_bias_components(
    design,
    A,
    weights: np.ndarray,
    p: float,
    gamma: float,
    eta: float,
    burn_in: int = 1,
    gamma_bound: float | None = None,
    eta_bound: float | None = None,
) -> dict[str, float]:
    """Compute design-implied linear-model bias components for Kronecker COSTA.

    For COSTA/COSTA-IC with item vectors u_i and time vectors v_b,
        R_{(i,b),(j,b)} = G_p(u_i'u_j),
        R_{(i,b),(i,b-1)} = G_p(v_b'v_{b-1}).
    The returned ``bias_linear`` uses the signed simulation parameters, while
    ``bias_bound`` uses absolute upper bounds and is intended for bias-aware CI.
    """
    if getattr(design, "item_vecs", None) is None or getattr(design, "time_vecs", None) is None:
        return {"bias_comp_raw": np.nan, "bias_temp_raw": np.nan, "bias_linear": np.nan, "bias_bound": np.nan}
    E = np.asarray(design.item_vecs, dtype=float)
    G = np.asarray(design.time_vecs, dtype=float)
    Wmat = np.asarray(weights, dtype=float)
    Wtot = float(np.sum(Wmat[:, int(burn_in):]) + 1e-12)
    Acoo = ensure_csr(A).tocoo()
    cmap = CopulaMap.build(float(p))
    if Acoo.nnz:
        rho = np.sum(E[Acoo.row] * E[Acoo.col], axis=1)
        R_e = copula_map_numpy(rho, p, cmap)
        row_weight = np.sum(Wmat[Acoo.row, int(burn_in):], axis=1)
        b_comp = float(np.sum(row_weight * Acoo.data * (R_e - 1.0)))
    else:
        b_comp = 0.0
    b_temp = 0.0
    for t in range(max(1, int(burn_in)), Wmat.shape[1]):
        rho_t = float(np.sum(G[t] * G[t - 1]))
        R_t = float(copula_map_numpy(np.asarray([rho_t]), p, cmap)[0])
        b_temp += float(np.sum(Wmat[:, t] * (R_t - 1.0)))
    gb = abs(float(gamma)) if gamma_bound is None else float(gamma_bound)
    eb = abs(float(eta)) if eta_bound is None else float(eta_bound)
    return {
        "bias_comp_raw": float(b_comp),
        "bias_temp_raw": float(b_temp),
        "bias_linear": float((float(gamma) * b_comp + float(eta) * b_temp) / Wtot),
        "bias_bound": float((gb * abs(b_comp) + eb * abs(b_temp)) / Wtot),
    }


def locality_diagnostics(
    design,
    p: float,
    n_pairs: int = 50000,
    seed: int = 123,
    time_gap: int = 2,
) -> dict[str, float]:
    """Monte Carlo diagnostics for long-range covariance under Kronecker designs."""
    if getattr(design, "item_vecs", None) is None or getattr(design, "time_vecs", None) is None:
        return {}
    rng = np.random.default_rng(seed)
    E = np.asarray(design.item_vecs, dtype=float)
    G = np.asarray(design.time_vecs, dtype=float)
    n = E.shape[0]
    b = G.shape[0]
    cmap = CopulaMap.build(float(p))
    a = rng.integers(0, n, size=int(n_pairs))
    c = rng.integers(0, n, size=int(n_pairs))
    mask = a != c
    a, c = a[mask], c[mask]
    R_item = copula_map_numpy(np.sum(E[a] * E[c], axis=1), p, cmap) if len(a) else np.asarray([])
    far_time_pairs = [(t, r) for t in range(b) for r in range(b) if abs(t - r) >= int(time_gap)]
    if far_time_pairs:
        idx = rng.integers(0, len(far_time_pairs), size=max(1, min(int(n_pairs), len(a) if len(a) else int(n_pairs))))
        tr = np.asarray([far_time_pairs[i] for i in idx], dtype=int)
        R_time = copula_map_numpy(np.sum(G[tr[:, 0]] * G[tr[:, 1]], axis=1), p, cmap)
    else:
        tr = np.zeros((0, 2), dtype=int)
        R_time = np.asarray([])
    m = min(len(a), len(tr))
    if m:
        rho_prod = np.sum(E[a[:m]] * E[c[:m]], axis=1) * np.sum(G[tr[:m, 0]] * G[tr[:m, 1]], axis=1)
        R_prod = copula_map_numpy(rho_prod, p, cmap)
    else:
        R_prod = np.asarray([])
    def stats(prefix: str, x: np.ndarray) -> dict[str, float]:
        if x.size == 0:
            return {f"{prefix}_mean_abs_R": np.nan, f"{prefix}_mean_R2": np.nan, f"{prefix}_q90_abs_R": np.nan}
        return {
            f"{prefix}_mean_abs_R": float(np.mean(np.abs(x))),
            f"{prefix}_mean_R2": float(np.mean(x ** 2)),
            f"{prefix}_q90_abs_R": float(np.quantile(np.abs(x), 0.90)),
        }
    out: dict[str, float] = {}
    out.update(stats("far_item", R_item))
    out.update(stats("far_time", R_time))
    out.update(stats("far_product", R_prod))
    return out
