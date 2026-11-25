"""Diagonal Gaussian-mixture emission densities b_j(x) in log space.

Missing / ineligible turns (mask == False) contribute log b = 0, i.e. the
observation is marginalised out; the hidden chain still advances.
A subset of feature dimensions can be selected, which analytically
marginalises the other coordinates of every diagonal component (used for
leave-one-feature-out and Shapley attribution).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..util import logsumexp

LOG2PI = float(np.log(2 * np.pi))
W_FLOOR = 1e-12


@dataclass
class GMMEmissions:
    w: np.ndarray    # (K, M) component weights
    mu: np.ndarray   # (K, M, D)
    var: np.ndarray  # (K, M, D)

    @property
    def K(self) -> int:
        return self.w.shape[0]

    @property
    def M(self) -> int:
        return self.w.shape[1]

    def copy(self) -> "GMMEmissions":
        return GMMEmissions(self.w.copy(), self.mu.copy(), self.var.copy())

    def permute(self, perm: Sequence[int]) -> "GMMEmissions":
        p = list(perm)
        return GMMEmissions(self.w[p].copy(), self.mu[p].copy(), self.var[p].copy())

    def comp_logpdf(self, X: np.ndarray, dims: Sequence[int] | None = None) -> np.ndarray:
        """log N(x; mu_jm, V_jm) for every component: shape (N, T, K, M)."""
        X = np.nan_to_num(np.asarray(X, float))
        if dims is None:
            mu, var, Xd = self.mu, self.var, X
        else:
            d = list(dims)
            mu, var, Xd = self.mu[..., d], self.var[..., d], X[..., d]
        diff = Xd[:, :, None, None, :] - mu[None, None]
        return -0.5 * np.sum(diff * diff / var + np.log(var) + LOG2PI, axis=-1)

    def log_b(self, X: np.ndarray, mask: np.ndarray, dims: Sequence[int] | None = None,
              return_components: bool = False):
        """Mixture log-density per state, (N, T, K); zero where mask is False."""
        N, T = mask.shape
        if dims is not None and len(dims) == 0:
            lb = np.zeros((N, T, self.K))
            return (lb, None) if return_components else lb
        lc = self.comp_logpdf(X, dims) + np.log(np.maximum(self.w, W_FLOOR))[None, None]
        lb = logsumexp(lc, axis=-1)
        lb = np.where(mask[..., None], lb, 0.0)
        return (lb, lc) if return_components else lb

    def state_means(self) -> np.ndarray:
        """Mixture mean per state (K, D) — a permutation signature."""
        return np.einsum("km,kmd->kd", self.w, self.mu)

    def to_dict(self) -> dict:
        return {"w": self.w, "mu": self.mu, "var": self.var}
