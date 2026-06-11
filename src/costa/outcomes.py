from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OutcomeParameters:
    alpha: np.ndarray  # (N,B)
    beta: np.ndarray   # (N,B) or (N,1)
    gamma: float
    eta: float
    noise_sd: float
    model: str
    # Residual shock scales used in observed outcomes. These are intentionally
    # configurable because item-side simulations are too easy if residual noise is
    # iid and tiny after subtracting a perfect baseline.
    item_shock_sd: float = 1.00
    block_shock_sd: float = 0.10
    heteroskedastic_noise: bool = True
    # Misspecified nonlinear model.
    nonlinear_saturation_scale: float = 1.5
    nonlinear_lag2_scale: float = 0.5
    interaction_scale: float = 0.5
    second_hop_scale: float = 0.25
    # Demand model.
    demand_temporal_scale: float = 2.0
    demand_cross_temporal_scale: float = 2.0
    demand_substitution_scale: float = 1.0
    demand_treated_buffer: float = 0.4
    demand_saturation_scale: float = 0.25


def canonical_model_name(model: str) -> str:
    m = model.lower().strip()
    aliases = {
        "well_specified_linear": "linear",
        "linear_well_specified": "linear",
        "linear": "linear",
        "nonlinear_saturation": "nonlinear",
        "nonlinear": "nonlinear",
        "interaction_second_hop": "interaction_second_hop",
        "second_hop": "interaction_second_hop",
        "demand_substitution": "demand",
        "demand": "demand",
    }
    if m not in aliases:
        raise ValueError(f"Unknown outcome model: {model}")
    return aliases[m]


def make_baseline(
    item_block_weight: np.ndarray,
    seed: int = 123,
    beta_base: float = 0.08,
    beta_pop_scale: float = 0.02,
    beta_noise_sd: float = 0.015,
    alpha_traffic_scale: float = 0.4,
    alpha_item_sd: float = 0.10,
    alpha_block_sd: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Create alpha and beta surfaces from historical item-block traffic."""
    rng = np.random.default_rng(seed)
    W = item_block_weight.astype(float)
    if np.any(W > 0):
        Wn = W / (np.percentile(W[W > 0], 95) + 1e-12)
    else:
        Wn = W
    Wn = np.clip(Wn, 0, 3)
    n, b = W.shape
    item_fe = rng.normal(0, alpha_item_sd, size=(n, 1))
    time_fe = rng.normal(0, alpha_block_sd, size=(1, b))
    alpha = 1.0 + alpha_traffic_scale * np.log1p(Wn) + item_fe + time_fe
    alpha = np.maximum(alpha, 0.05)
    pop = W.sum(axis=1)
    pop_z = (np.log1p(pop) - np.mean(np.log1p(pop))) / (np.std(np.log1p(pop)) + 1e-8)
    beta_i = beta_base + beta_pop_scale * pop_z + rng.normal(0, beta_noise_sd, size=n)
    beta = np.repeat(beta_i[:, None], b, axis=1)
    return alpha.astype(float), beta.astype(float)


def make_alpha_adjustment(
    alpha: np.ndarray,
    seed: int = 123,
    item_sd: float = 0.70,
    block_sd: float = 0.20,
    cell_sd: float = 0.35,
) -> np.ndarray:
    """Create an imperfect pre-experiment baseline used by the estimator."""
    rng = np.random.default_rng(seed)
    n, b = alpha.shape
    residual = np.zeros_like(alpha, dtype=float)
    if item_sd > 0:
        residual += rng.normal(0, item_sd, size=(n, 1))
    if block_sd > 0:
        residual += rng.normal(0, block_sd, size=(1, b))
    if cell_sd > 0:
        residual += rng.normal(0, cell_sd, size=(n, b))
    return (alpha + residual).astype(float)



def make_alpha_hat(
    alpha: np.ndarray,
    seed: int = 123,
    alpha_hat_item_sd: float = 0.70,
    alpha_hat_block_sd: float = 0.20,
    alpha_hat_cell_sd: float = 0.35,
) -> np.ndarray:
    """Backward-compatible wrapper for the imperfect baseline used by estimators."""
    return make_alpha_adjustment(
        alpha,
        seed=seed,
        item_sd=alpha_hat_item_sd,
        block_sd=alpha_hat_block_sd,
        cell_sd=alpha_hat_cell_sd,
    )


def _competition(A: np.ndarray, Z: np.ndarray) -> np.ndarray:
    n, b = Z.shape
    comp = np.zeros_like(Z, dtype=float)
    for t in range(b):
        comp[:, t] = A @ Z[:, t]
    return comp


def _noise_scale(alpha: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.ones_like(alpha, dtype=float)
    scale = np.sqrt(np.maximum(alpha, 1e-6) / (np.mean(alpha) + 1e-12))
    return np.clip(scale, 0.5, 3.0)


def deterministic_mean(Z: np.ndarray, A: np.ndarray, params: OutcomeParameters) -> np.ndarray:
    """Noise-free potential-outcome mean for all item-block cells."""
    Z = Z.astype(float)
    alpha, beta = params.alpha, params.beta
    comp = _competition(A, Z)
    lag1 = np.zeros_like(Z, dtype=float)
    lag1[:, 1:] = Z[:, :-1]
    lag2 = np.zeros_like(Z, dtype=float)
    lag2[:, 2:] = Z[:, :-2]
    model = canonical_model_name(params.model)

    if model == "linear":
        return alpha + beta * Z + params.gamma * comp + params.eta * lag1

    if model == "nonlinear":
        return (
            alpha
            + beta * Z
            + params.gamma * np.tanh(params.nonlinear_saturation_scale * comp)
            + params.interaction_scale * params.gamma * Z * comp
            + params.eta * lag1
            + params.nonlinear_lag2_scale * params.eta * lag2
        )

    if model == "interaction_second_hop":
        A2 = A @ A
        comp2 = _competition(A2, Z)
        return (
            alpha
            + beta * Z
            + params.gamma * comp
            + params.interaction_scale * params.gamma * Z * comp
            + params.second_hop_scale * params.gamma * comp2
            + params.eta * lag1
        )

    if model == "demand":
        lag_comp = _competition(A, lag1)
        # Treated competitors steal demand. If item i itself is treated, the loss is
        # partly buffered. The previous-period competitor state has a separate scale
        # and is deliberately non-negligible to stress-test time-blind static OCD.
        competitor_loss = -params.gamma * params.demand_substitution_scale * comp * (
            1.0 - params.demand_treated_buffer * Z
        )
        own_gain = beta * Z * (1.0 - params.demand_saturation_scale * np.clip(comp, 0.0, 2.0))
        temporal_state = params.eta * params.demand_temporal_scale * lag1
        cross_temporal_state = -params.eta * params.demand_cross_temporal_scale * lag_comp
        return alpha + own_gain + competitor_loss + temporal_state + cross_temporal_state

    raise ValueError(f"Unknown outcome model: {params.model}")


def generate_observed_outcomes(
    Z: np.ndarray,
    A: np.ndarray,
    params: OutcomeParameters,
    seed: int = 123,
) -> np.ndarray:
    """Generate observed item-block outcomes under semi-synthetic PO models.

    The stochastic part includes cell-level noise, an item-level residual shock, and
    a block-level residual shock. This fixes the earlier prototype's unrealistically
    tiny Monte Carlo standard deviations, which were caused by subtracting the true
    baseline and using iid noise of order 0.02 over N(B-1) cells.
    """
    rng = np.random.default_rng(seed)
    mu = deterministic_mean(Z, A, params)
    y = np.array(mu, dtype=float, copy=True)
    scale = _noise_scale(params.alpha, params.heteroskedastic_noise)
    if params.noise_sd > 0:
        y += rng.normal(0, params.noise_sd, size=mu.shape) * scale
    if params.item_shock_sd > 0:
        item_scale = np.sqrt(np.maximum(params.alpha.mean(axis=1), 1e-6) / (np.mean(params.alpha) + 1e-12))
        item_scale = np.clip(item_scale, 0.5, 3.0)
        y += rng.normal(0, params.item_shock_sd, size=(mu.shape[0], 1)) * item_scale[:, None]
    if params.block_shock_sd > 0:
        y += rng.normal(0, params.block_shock_sd, size=(1, mu.shape[1]))
    return y.astype(float)


def potential_mean(Z: np.ndarray, A: np.ndarray, params: OutcomeParameters) -> np.ndarray:
    """Noise-free potential outcome mean for any full assignment Z."""
    no_noise = OutcomeParameters(**{
        **params.__dict__,
        "noise_sd": 0.0,
        "item_shock_sd": 0.0,
        "block_shock_sd": 0.0,
    })
    return deterministic_mean(Z, A, no_noise)


def true_tau(A: np.ndarray, params: OutcomeParameters, weights: np.ndarray | None = None, burn_in: int = 1) -> float:
    """Compute global treatment minus global control estimand exactly."""
    n, b = params.alpha.shape
    Z1 = np.ones((n, b), dtype=float)
    Z0 = np.zeros((n, b), dtype=float)
    y1 = potential_mean(Z1, A, params)
    y0 = potential_mean(Z0, A, params)
    mask = np.zeros((n, b), dtype=bool)
    mask[:, burn_in:] = True
    if weights is None:
        return float((y1[mask] - y0[mask]).mean())
    w = weights.astype(float)
    w = np.where(mask, w, 0.0)
    return float(np.sum(w * (y1 - y0)) / (np.sum(w) + 1e-12))


def make_alpha_hat(
    alpha: np.ndarray,
    seed: int = 123,
    alpha_hat_item_sd: float = 0.70,
    alpha_hat_block_sd: float = 0.20,
    alpha_hat_cell_sd: float = 0.35,
) -> np.ndarray:
    """Backward-compatible wrapper for the estimator's imperfect baseline."""
    return make_alpha_adjustment(
        alpha,
        seed=seed,
        item_sd=alpha_hat_item_sd,
        block_sd=alpha_hat_block_sd,
        cell_sd=alpha_hat_cell_sd,
    )
