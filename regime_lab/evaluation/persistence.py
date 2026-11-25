"""[W10 v2] Post-hoc count-conditioned exchangeable-ordering sensitivity for H2.

Null: given each dossier's observed number of H labels, every ordering of those labels across its turns is
equally likely (exchangeable turn positions). Preserves dossier error counts (difficulty); does NOT control
turn or question-type effects (see the position-only control). Exact: the H->H statistic's distribution is
enumerated per dossier and convolved across dossiers, so there is no Monte Carlo error.
The locked H2 test (`pipeline.persistence_test`) is unchanged and reported alongside.
"""
from __future__ import annotations

import itertools

import numpy as np


def _hh_dist(T: int, h: int) -> np.ndarray:
    """P(#adjacent H-H pairs = k) over all C(T,h) equally likely placements of h H's in T turns."""
    counts = np.zeros(T, float)
    for pos in itertools.combinations(range(T), h):
        s = set(pos)
        counts[sum((p + 1) in s for p in pos)] += 1
    return counts / counts.sum()


def within_dossier_exact(L: np.ndarray) -> dict:
    """L: (N,T) 0/1 H indicators for fully resolved dossiers."""
    N, T = L.shape
    obs = int(np.sum(L[:, 1:] * L[:, :-1]))
    cache, dist = {}, np.array([1.0])
    for h in L.sum(1).astype(int):
        if h not in cache:
            cache[h] = _hh_dist(T, h)
        dist = np.convolve(dist, cache[h])
    k = np.arange(len(dist))
    return {"HH_observed": obs, "null_mean": float(np.dot(k, dist)), "p_value_upper": float(dist[obs:].sum()),
            "dossiers": int(N), "informative_dossiers": int(np.sum((L.sum(1) >= 2) & (L.sum(1) < T)))}


def controls(n=500, T=6, seed=0) -> dict:
    """Synthetic controls with NO history effect: what the sensitivity does under each nuisance."""
    rng = np.random.default_rng(seed)
    p = np.where(rng.random(n) < .5, .05, .65)                         # dossier difficulty only
    diff = (rng.random((n, T)) < p[:, None]).astype(int)
    pos = (rng.random((n, T)) < np.array([.05, .05, .75, .75, .75, .75])[None]).astype(int)   # turn position only
    return {"difficulty_only": {**within_dossier_exact(diff), "expected": "no rejection (difficulty is conditioned on)"},
            "position_only": {**within_dossier_exact(pos),
                              "expected": "rejection: the null assumes exchangeable positions and does not control turn effects"}}


def sensitivity(labels: np.ndarray, seed=0) -> dict:
    valid = np.all(np.isin(labels, ["G", "U", "H"]), axis=1)
    out = within_dossier_exact((labels[valid] == "H").astype(int))
    out["method"] = ("post-hoc count-conditioned exchangeable-ordering sensitivity (exact enumeration); "
                     "assumes exchangeable turn positions; does not control question-type or turn effects")
    out["controls"] = controls(seed=seed)
    return out
