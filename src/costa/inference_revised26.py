"""Inference utilities for the revised COSTA supplementary experiments.

The functions in this module deliberately separate three centering targets:

* global demeaning, retained only as a legacy comparator;
* an independently calibrated heterogeneous contribution mean; and
* a design-aware nonlocal covariance-tail correction.

All returned variances are on the estimator scale.  The local kernel is the
one-hop product graph used in the manuscript: diagonal terms, same-block
network edges, and same-unit adjacent-block edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy import sparse

VarianceMode = Literal["signed", "absolute"]


def _as_csr(A) -> sparse.csr_matrix:
    if sparse.issparse(A):
        return A.tocsr()
    return sparse.csr_matrix(np.asarray(A))


def _undirected_support_edges(A) -> tuple[np.ndarray, np.ndarray]:
    A_csr = _as_csr(A)
    support = ((A_csr != 0).astype(np.int8) + (A_csr.T != 0).astype(np.int8)).tocoo()
    rows = support.row.astype(np.int64)
    cols = support.col.astype(np.int64)
    keep = rows < cols
    return rows[keep], cols[keep]


@dataclass(frozen=True)
class ProductGraphPairs:
    """Flattened one-hop product-graph pairs on post-burn-in cells."""

    n_units: int
    n_blocks: int
    burn_in: int
    n_eval_blocks: int
    n_cells: int
    competition_left: np.ndarray
    competition_right: np.ndarray
    temporal_left: np.ndarray
    temporal_right: np.ndarray

    @classmethod
    def from_graph(cls, A, n_blocks: int, burn_in: int = 1) -> "ProductGraphPairs":
        A_csr = _as_csr(A)
        n_units = int(A_csr.shape[0])
        n_blocks = int(n_blocks)
        burn_in = int(burn_in)
        if not 0 <= burn_in < n_blocks:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_blocks")
        n_eval_blocks = n_blocks - burn_in

        def idx(i: np.ndarray, b: int) -> np.ndarray:
            return i * n_eval_blocks + (int(b) - burn_in)

        edge_i, edge_j = _undirected_support_edges(A_csr)
        comp_l: list[np.ndarray] = []
        comp_r: list[np.ndarray] = []
        for b in range(burn_in, n_blocks):
            comp_l.append(idx(edge_i, b))
            comp_r.append(idx(edge_j, b))

        units = np.arange(n_units, dtype=np.int64)
        temp_l: list[np.ndarray] = []
        temp_r: list[np.ndarray] = []
        for b in range(burn_in + 1, n_blocks):
            temp_l.append(idx(units, b))
            temp_r.append(idx(units, b - 1))

        empty = np.zeros(0, dtype=np.int64)
        return cls(
            n_units=n_units,
            n_blocks=n_blocks,
            burn_in=burn_in,
            n_eval_blocks=n_eval_blocks,
            n_cells=n_units * n_eval_blocks,
            competition_left=np.concatenate(comp_l).astype(np.int64) if comp_l else empty,
            competition_right=np.concatenate(comp_r).astype(np.int64) if comp_r else empty,
            temporal_left=np.concatenate(temp_l).astype(np.int64) if temp_l else empty,
            temporal_right=np.concatenate(temp_r).astype(np.int64) if temp_r else empty,
        )

    def raw_quadratic(self, residuals: np.ndarray, mode: VarianceMode = "signed") -> dict[str, float]:
        """Return the unscaled local quadratic form and its components."""
        x = np.asarray(residuals, dtype=np.float64).reshape(-1)
        if x.size != self.n_cells:
            raise ValueError(f"Expected {self.n_cells} residual cells, got {x.size}")
        diag = float(np.dot(x, x))

        def cross(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
            if left.size == 0:
                return 0.0, 0.0
            prod = x[left] * x[right]
            return float(2.0 * np.sum(prod)), float(2.0 * np.sum(np.abs(prod)))

        comp_signed, comp_abs = cross(self.competition_left, self.competition_right)
        temp_signed, temp_abs = cross(self.temporal_left, self.temporal_right)
        signed = diag + comp_signed + temp_signed
        absolute = diag + comp_abs + temp_abs
        chosen = signed if mode == "signed" else absolute
        return {
            "raw_quadratic": float(chosen),
            "raw_signed": float(signed),
            "raw_absolute": float(absolute),
            "diag": diag,
            "competition_signed": comp_signed,
            "competition_absolute": comp_abs,
            "temporal_signed": temp_signed,
            "temporal_absolute": temp_abs,
        }

    def estimator_variance(self, residuals: np.ndarray, mode: VarianceMode = "signed") -> dict[str, float]:
        out = self.raw_quadratic(residuals, mode=mode)
        scale = float(self.n_cells) ** 2
        variance = out["raw_quadratic"] / scale
        out.update({
            "variance_raw": float(variance),
            "variance": float(max(variance, 0.0)),
            "se": float(np.sqrt(max(variance, 0.0))),
            "truncated": bool(variance < 0.0),
        })
        return out


def stabilized_contributions(
    Z: np.ndarray,
    Y: np.ndarray,
    adjustment: np.ndarray,
    p: float,
    weights: np.ndarray,
    burn_in: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return weighted HT estimate, stabilized contributions, and weights.

    On the evaluated cells, ``omega = M w / sum(w)`` and
    ``Q = omega * psi * (Y-adjustment)``.  Hence ``tau_hat = mean(Q)``.
    """
    Z = np.asarray(Z)
    Y = np.asarray(Y, dtype=np.float64)
    adjustment = np.asarray(adjustment, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if Z.shape != Y.shape or Z.shape != adjustment.shape or Z.shape != weights.shape:
        raise ValueError("Z, Y, adjustment, and weights must have identical shapes")
    if not 0.0 < float(p) < 1.0:
        raise ValueError("p must lie in (0,1)")
    burn_in = int(burn_in)
    eval_slice = np.s_[:, burn_in:]
    w_eval = weights[eval_slice]
    W = float(np.sum(w_eval))
    if not np.isfinite(W) or W <= 0.0:
        raise ValueError("Post-burn-in weights must have positive finite sum")
    M = int(w_eval.size)
    v = float(p) * (1.0 - float(p))
    psi = (Z[eval_slice].astype(np.float64) - float(p)) / v
    omega = (float(M) / W) * w_eval
    Q = omega * psi * (Y[eval_slice] - adjustment[eval_slice])
    return float(np.mean(Q)), Q, omega


def global_mean_residuals(Q: np.ndarray, omega: np.ndarray, tau_hat: float) -> np.ndarray:
    """Legacy global demeaning used by the earlier product-graph HAC code."""
    Q = np.asarray(Q, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    if Q.shape != omega.shape:
        raise ValueError("Q and omega must have the same shape")
    return Q - omega * float(tau_hat)


@dataclass
class MeanCalibration:
    qbar: np.ndarray
    design_center: float
    n_draws: int
    mc_se_design_center: float


@dataclass
class TailCalibration:
    full_variance: float
    local_variance: float
    signed_tail: float
    positive_tail: float
    n_draws: int
    full_se: float


def calibrate_mean(
    draw_contribution: Callable[[int], np.ndarray],
    n_draws: int,
    seed: int,
) -> MeanCalibration:
    """Estimate heterogeneous contribution means from an independent design stream."""
    n_draws = int(n_draws)
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2")
    sum_q: np.ndarray | None = None
    totals = np.empty(n_draws, dtype=np.float64)
    for s in range(n_draws):
        q = np.asarray(draw_contribution(int(seed + 104729 * s)), dtype=np.float64)
        if sum_q is None:
            sum_q = np.zeros_like(q, dtype=np.float64)
        if q.shape != sum_q.shape:
            raise ValueError("Contribution draws have inconsistent shapes")
        sum_q += q
        totals[s] = float(np.mean(q))
    assert sum_q is not None
    qbar = sum_q / float(n_draws)
    return MeanCalibration(
        qbar=qbar,
        design_center=float(np.mean(qbar)),
        n_draws=n_draws,
        mc_se_design_center=float(np.std(totals, ddof=1) / np.sqrt(n_draws)),
    )


def calibrate_tail(
    draw_contribution: Callable[[int, int], np.ndarray],
    pairs: ProductGraphPairs,
    n_draws: int,
    assignment_seed: int,
    outcome_seed: int,
    pair_batch: int = 200_000,
) -> TailCalibration:
    """Estimate full and local design variance from an independent calibration stream."""
    n_draws = int(n_draws)
    if n_draws < 3:
        raise ValueError("n_draws must be at least 3")
    pair_batch = int(pair_batch)
    if pair_batch < 1:
        raise ValueError("pair_batch must be positive")
    draws = np.empty((n_draws, pairs.n_cells), dtype=np.float32)
    for s in range(n_draws):
        q = np.asarray(
            draw_contribution(int(assignment_seed + 130363 * s), int(outcome_seed + 155921 * s)),
            dtype=np.float32,
        ).reshape(-1)
        if q.size != pairs.n_cells:
            raise ValueError(f"Expected {pairs.n_cells} cells, got {q.size}")
        draws[s] = q

    # Keep the large calibration matrix in float32.  Center in place so a
    # 10k-item, 500-draw run does not create a second float64 matrix.
    totals = draws.mean(axis=1, dtype=np.float64)
    column_mean = draws.mean(axis=0, dtype=np.float64).astype(np.float32)
    draws -= column_mean[None, :]
    centered = draws
    ddof = float(n_draws - 1)
    full_variance = float(np.var(totals, ddof=1))
    local_sum = float(
        np.sum(np.sum(centered * centered, axis=0, dtype=np.float64) / ddof, dtype=np.float64)
    )

    def add_pairs(left: np.ndarray, right: np.ndarray) -> float:
        total = 0.0
        for start in range(0, left.size, pair_batch):
            end = min(start + pair_batch, left.size)
            prod = centered[:, left[start:end]] * centered[:, right[start:end]]
            cov = np.sum(prod, axis=0, dtype=np.float64) / ddof
            total += float(2.0 * np.sum(cov, dtype=np.float64))
        return total

    local_sum += add_pairs(pairs.competition_left, pairs.competition_right)
    local_sum += add_pairs(pairs.temporal_left, pairs.temporal_right)
    local_variance = float(local_sum / (float(pairs.n_cells) ** 2))
    signed_tail = float(full_variance - local_variance)
    return TailCalibration(
        full_variance=full_variance,
        local_variance=local_variance,
        signed_tail=signed_tail,
        positive_tail=max(signed_tail, 0.0),
        n_draws=n_draws,
        full_se=float(np.sqrt(max(full_variance, 0.0))),
    )


def interval(center: float, variance: float, zcrit: float) -> tuple[float, float, float, bool]:
    raw = float(variance)
    used = max(raw, 0.0)
    se = float(np.sqrt(used))
    return float(center - zcrit * se), float(center + zcrit * se), se, bool(raw < 0.0)
