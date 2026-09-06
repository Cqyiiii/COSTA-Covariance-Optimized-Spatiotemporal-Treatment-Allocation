from __future__ import annotations

import numpy as np
from scipy import sparse

from costa.inference_revised26 import (
    ProductGraphPairs,
    calibrate_mean,
    calibrate_tail,
    global_mean_residuals,
    stabilized_contributions,
)


def test_stabilized_contributions_matches_weighted_ht() -> None:
    Z = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int8)
    Y = np.array([[1.0, 3.0, 2.0], [2.0, 4.0, 1.0]])
    alpha = np.ones_like(Y)
    w = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 3.0]])
    tau, Q, omega = stabilized_contributions(Z, Y, alpha, p=0.5, weights=w, burn_in=1)
    psi = (Z[:, 1:] - 0.5) / 0.25
    expected = np.sum(w[:, 1:] * psi * (Y[:, 1:] - alpha[:, 1:])) / np.sum(w[:, 1:])
    assert np.isclose(tau, expected)
    assert np.isclose(np.mean(Q), expected)
    assert np.isclose(np.mean(omega), 1.0)


def test_product_graph_pair_count_and_quadratic() -> None:
    A = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
    pairs = ProductGraphPairs.from_graph(A, n_blocks=3, burn_in=1)
    assert pairs.n_cells == 4
    assert pairs.competition_left.size == 2
    assert pairs.temporal_left.size == 2
    x = np.ones((2, 2))
    out = pairs.estimator_variance(x, mode="signed")
    # diag=4, two competition pairs contribute 4, two temporal pairs contribute 4.
    assert np.isclose(out["variance_raw"], 12.0 / 16.0)


def test_global_mean_residuals_sum_to_zero_with_unit_weights() -> None:
    Q = np.array([[1.0, 2.0], [3.0, 4.0]])
    omega = np.ones_like(Q)
    tau = float(np.mean(Q))
    assert np.isclose(global_mean_residuals(Q, omega, tau).sum(), 0.0)


def test_calibrators_recover_simple_independent_draws() -> None:
    rng = np.random.default_rng(123)
    seeds = {i: rng.normal(size=(2, 2)) for i in range(2000)}

    def draw_mean(seed: int) -> np.ndarray:
        return seeds[seed % 2000]

    cal = calibrate_mean(draw_mean, n_draws=500, seed=0)
    assert cal.qbar.shape == (2, 2)
    assert abs(cal.design_center) < 0.15

    A = sparse.csr_matrix((2, 2))
    pairs = ProductGraphPairs.from_graph(A, n_blocks=2, burn_in=0)

    def draw_tail(assignment_seed: int, outcome_seed: int) -> np.ndarray:
        local_rng = np.random.default_rng(assignment_seed + outcome_seed)
        return local_rng.normal(size=(2, 2))

    tail = calibrate_tail(draw_tail, pairs, n_draws=300, assignment_seed=1, outcome_seed=2)
    assert tail.full_variance > 0
    assert tail.local_variance > 0
    assert np.isfinite(tail.signed_tail)
