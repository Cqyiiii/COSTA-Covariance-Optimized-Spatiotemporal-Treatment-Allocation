from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from scipy.stats import multivariate_normal, norm


def gp_exact(rho: np.ndarray, p: float) -> np.ndarray:
    """Bernoulli-p normalized covariance induced by thresholded Gaussian copula.

    G_p(rho) = [Phi_2(t_p, t_p; rho) - p^2] / [p(1-p)].
    For p=1/2, this reduces to 2/pi * arcsin(rho).
    """
    rho_arr = np.asarray(rho, dtype=float)
    if abs(p - 0.5) < 1e-12:
        return (2.0 / np.pi) * np.arcsin(np.clip(rho_arr, -1.0, 1.0))
    q = 1.0 - p
    t = norm.ppf(p)
    out = np.empty_like(rho_arr, dtype=float)
    flat = rho_arr.ravel()
    vals = out.ravel()
    for idx, r in enumerate(flat):
        r = float(np.clip(r, -0.999999, 0.999999))
        cdf = multivariate_normal.cdf([t, t], mean=[0, 0], cov=[[1.0, r], [r, 1.0]])
        vals[idx] = (cdf - p * p) / (p * q)
    return out


@dataclass
class CopulaMap:
    p: float
    rho_grid: np.ndarray
    r_grid: np.ndarray

    @classmethod
    def build(cls, p: float, grid_size: int = 801) -> "CopulaMap":
        rho_grid = np.linspace(-0.999, 0.999, grid_size)
        r_grid = gp_exact(rho_grid, p)
        return cls(p=p, rho_grid=rho_grid.astype(np.float32), r_grid=r_grid.astype(np.float32))

    def torch(self, rho: torch.Tensor) -> torch.Tensor:
        """Differentiable piecewise-linear interpolation of G_p(rho)."""
        if abs(self.p - 0.5) < 1e-12:
            return (2.0 / np.pi) * torch.arcsin(torch.clamp(rho, -0.999999, 0.999999))
        device = rho.device
        dtype = rho.dtype
        grid = torch.as_tensor(self.rho_grid, device=device, dtype=dtype)
        vals = torch.as_tensor(self.r_grid, device=device, dtype=dtype)
        x = torch.clamp(rho, float(grid[0]), float(grid[-1]))
        # Convert to fractional index on an evenly spaced grid.
        pos = (x - grid[0]) / (grid[-1] - grid[0]) * (grid.numel() - 1)
        lo = torch.floor(pos).long().clamp(0, grid.numel() - 2)
        hi = lo + 1
        weight = (pos - lo.to(dtype)).clamp(0.0, 1.0)
        return vals[lo] * (1.0 - weight) + vals[hi] * weight


def normalize_rows_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=1, keepdim=True), min=eps)


def sample_kronecker_gaussian_copula(
    item_vecs: np.ndarray,
    time_vecs: np.ndarray,
    p: float,
    n_samples: int = 1,
    seed: int = 123,
) -> np.ndarray:
    """Sample Z_{i,b} using Omega_{(i,b),(j,r)}=(e_i'e_j)(g_b'g_r).

    If E is a d_I by d_T matrix of iid standard normals, scores
    G_{i,b}=e_i' E g_b have exactly this covariance structure.
    """
    rng = np.random.default_rng(seed)
    item_vecs = np.asarray(item_vecs, dtype=float)
    time_vecs = np.asarray(time_vecs, dtype=float)
    d_i = item_vecs.shape[1]
    d_t = time_vecs.shape[1]
    threshold = norm.ppf(p)
    out = []
    for _ in range(n_samples):
        E = rng.standard_normal((d_i, d_t))
        scores = item_vecs @ E @ time_vecs.T
        z = (scores <= threshold).astype(np.int8)
        out.append(z)
    return np.stack(out, axis=0) if n_samples > 1 else out[0]


def sample_item_static_gaussian_copula(
    item_vecs: np.ndarray,
    n_blocks: int,
    p: float,
    n_samples: int = 1,
    seed: int = 123,
) -> np.ndarray:
    """Sample graph-aware assignments independently across time blocks.

    Same item covariance is used within each block, but there is no temporal adaptation.
    """
    rng = np.random.default_rng(seed)
    item_vecs = np.asarray(item_vecs, dtype=float)
    d = item_vecs.shape[1]
    threshold = norm.ppf(p)
    out = []
    for _ in range(n_samples):
        Z = np.zeros((item_vecs.shape[0], n_blocks), dtype=np.int8)
        for b in range(n_blocks):
            eps = rng.standard_normal(d)
            scores = item_vecs @ eps
            Z[:, b] = (scores <= threshold).astype(np.int8)
        out.append(Z)
    return np.stack(out, axis=0) if n_samples > 1 else out[0]
