"""Observation attribution and transition (factor) contrasts.

None of these quantities is a causal effect or a statement about the
network's internal reasoning; they are model-based associations.
"""
from __future__ import annotations

import itertools
from math import factorial

import numpy as np

from ..models.bayes import cov_logA
from ..models.emissions import GMMEmissions
from ..util import softmax_log


def posterior_with_dims(em: GMMEmissions, x, prior, dims):
    """p(z=. | x_dims, prior): prior q_t fixed, other coordinates marginalised."""
    lb = em.log_b(np.asarray(x, float)[None, None], np.array([[True]]), dims=dims)[0, 0]
    return softmax_log(np.log(np.maximum(prior, 1e-300)) + lb)


def leave_one_out(em, x, prior, h, D=3):
    full = posterior_with_dims(em, x, prior, list(range(D)))[h]
    return {d: float(full - posterior_with_dims(em, x, prior, [k for k in range(D) if k != d])[h]) for d in range(D)}


def shapley(em, x, prior, h, D=3):
    """Exact Shapley allocation of f_t(H) - q_t(H) over the D features."""
    v = {S: posterior_with_dims(em, x, prior, list(S))[h]
         for r in range(D + 1) for S in itertools.combinations(range(D), r)}
    phi = np.zeros(D)
    for d in range(D):
        for S in v:
            if d in S:
                continue
            S2 = tuple(sorted(S + (d,)))
            wgt = factorial(len(S)) * factorial(D - len(S) - 1) / factorial(D)
            phi[d] += wgt * (v[S2] - v[S])
    return {"phi": phi.tolist(), "prior_H": float(v[()]), "posterior_H": float(v[tuple(range(D))])}


def transition_contrasts(theta_draws, C_profiles, k, lo, hi, h, origins=None):
    """Standardised risk difference Delta_{i,k} = E_c[A_iH(c_k=hi) - A_iH(c_k=lo)] per draw.

    theta_draws (S,K,K,1+p); C_profiles (P,p) observed (standardised) covariate
    profiles of eligible transitions. Returns dict origin -> summary.
    """
    S, K = theta_draws.shape[:2]
    lo_c, hi_c = C_profiles.copy(), C_profiles.copy()
    lo_c[:, k], hi_c[:, k] = lo, hi
    out = {}
    deltas = np.empty((S, K))
    for s in range(S):
        Ah = np.exp(cov_logA(theta_draws[s], hi_c[None]))[0]   # (P,K,K)
        Al = np.exp(cov_logA(theta_draws[s], lo_c[None]))[0]
        deltas[s] = (Ah[:, :, h] - Al[:, :, h]).mean(0)
    for i in (origins if origins is not None else range(K)):
        d = deltas[:, i]
        out[int(i)] = {"median": float(np.median(d)), "lo": float(np.quantile(d, .025)),
                       "hi": float(np.quantile(d, .975)), "p_gt0": float(np.mean(d > 0))}
    return out, deltas


def filter_weighted_contrast(theta_draws, f_prev, C_profiles, k, lo, hi, h):
    """Delta_k = sum_i f_{t-1}(i)[A_iH(hi) - A_iH(lo)], averaged over profiles; per draw."""
    lo_c, hi_c = C_profiles.copy(), C_profiles.copy()
    lo_c[:, k], hi_c[:, k] = lo, hi
    vals = []
    for th in theta_draws:
        Ah = np.exp(cov_logA(th, hi_c[None]))[0][:, :, h]
        Al = np.exp(cov_logA(th, lo_c[None]))[0][:, :, h]
        vals.append(np.mean(np.sum(f_prev * (Ah - Al), axis=1)))
    vals = np.array(vals)
    return {"median": float(np.median(vals)), "lo": float(np.quantile(vals, .025)),
            "hi": float(np.quantile(vals, .975)), "p_gt0": float(np.mean(vals > 0))}
