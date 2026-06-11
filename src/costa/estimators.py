from __future__ import annotations

import numpy as np


def _post_burnin_mask(Z: np.ndarray, burn_in: int) -> np.ndarray:
    mask = np.zeros_like(Z, dtype=bool)
    mask[:, burn_in:] = True
    return mask


def weighted_ht(
    Z: np.ndarray,
    Y: np.ndarray,
    alpha: np.ndarray,
    p: float,
    weights: np.ndarray | None = None,
    burn_in: int = 1,
) -> float:
    """Baseline-adjusted weighted Horvitz-Thompson estimator.

    The target is the traffic-weighted GATE over post-burn-in item-block cells:

        tau_w = sum_a w_a {Y_a(1)-Y_a(0)} / sum_a w_a.

    For known marginal treatment probability p, the HT score is

        psi_a = Z_a/p - (1-Z_a)/(1-p) = (Z_a-p)/(p(1-p)).

    The estimator is sum_a w_a psi_a (Y_a-alpha_hat_a) / sum_a w_a.
    """
    q = 1.0 - p
    psi = (Z.astype(float) - p) / (p * q)
    mask = _post_burnin_mask(Z, burn_in)
    if weights is None:
        weights = np.ones_like(Z, dtype=float)
    w = np.where(mask, weights.astype(float), 0.0)
    return float(np.sum(w * psi * (Y - alpha)) / (np.sum(w) + 1e-12))


def hajek_difference(
    Z: np.ndarray,
    Y: np.ndarray,
    alpha: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    burn_in: int = 1,
) -> float:
    """Traffic-weighted Hájek / ratio difference-in-means estimator.

    This uses the same item-block traffic weights as weighted HT but normalizes
    treated and control means by their realized weighted sample sizes.
    """
    mask = _post_burnin_mask(Z, burn_in)
    if weights is None:
        weights = np.ones_like(Z, dtype=float)
    if alpha is None:
        alpha = np.zeros_like(Y, dtype=float)
    y = Y - alpha
    z = Z.astype(float)
    w = np.where(mask, weights.astype(float), 0.0)
    wt = w * z
    wc = w * (1.0 - z)
    return float(np.sum(wt * y) / (np.sum(wt) + 1e-12) - np.sum(wc * y) / (np.sum(wc) + 1e-12))


def difference_in_means(
    Z: np.ndarray,
    Y: np.ndarray,
    alpha: np.ndarray | None = None,
    burn_in: int = 1,
) -> float:
    """Unweighted baseline-adjusted difference-in-means over item-block cells."""
    mask = _post_burnin_mask(Z, burn_in)
    if alpha is None:
        alpha = np.zeros_like(Y, dtype=float)
    y = Y - alpha
    z = Z.astype(bool)
    treated = mask & z
    control = mask & (~z)
    if treated.sum() == 0 or control.sum() == 0:
        return float("nan")
    return float(np.mean(y[treated]) - np.mean(y[control]))
