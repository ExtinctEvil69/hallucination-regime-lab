"""Synthetic scenarios with known ground truth (mandatory before real data).

Every simulated dataset is labelled synthetic. Hidden states 0=G, 1=U, 2=H.
"""
from __future__ import annotations

import numpy as np

STATE_LABELS = np.array(["G", "U", "H"], object)

BASE_MEANS = np.array([[0.0, 0.0, 1.0], [-0.8, 0.8, 0.0], [-1.6, 1.4, 0.6]])  # (l, e, s) per state


def scenario_params(name):
    A_persist = np.array([[.88, .05, .07], [.25, .6, .15], [.15, .05, .80]])
    A_indep = np.tile([.75, .1, .15], (3, 1))
    p = {"A": A_persist, "means": BASE_MEANS.copy(), "sd": 0.6, "beta": None, "confound": False}
    if name == "independent":
        p["A"] = A_indep
    elif name == "abrupt":
        p["A"] = np.array([[.80, .02, .18], [.3, .5, .2], [.02, .02, .96]])
    elif name == "overlapping":
        p["sd"] = 1.1
    elif name == "high_conf_wrong":
        p["means"][2] = [-0.3, 0.3, 0.95]        # wrong answers look confident and grounded
    elif name in ("covariate", "covariate_null", "confounded"):
        p["beta"] = np.zeros((3, 3, 2))          # (origin, dest, covariate) logit effects on H dest
        if name != "covariate_null":
            p["beta"][:, 2, 0] = -1.5            # evidence available lowers entry into H
            p["beta"][:, 2, 1] = 0.8             # ambiguity raises it
        p["confound"] = name == "confounded"
    return p


def simulate(name, N=300, T=6, seed=0):
    rng = np.random.default_rng(seed)
    p = scenario_params(name)
    K = 3
    C = np.zeros((N, T, 2))
    C[..., 0] = rng.random((N, T)) < 0.75                        # evidence available
    C[..., 1] = rng.integers(1, 3, (N, T))                       # ambiguity 1 or 2
    difficulty = rng.normal(0, 1, N)
    if p["confound"]:                                            # hard dossiers hide evidence more often
        C[..., 0] = rng.random((N, T)) < (1 / (1 + np.exp(difficulty)))[:, None] * 1.0 + 0.1
    z = np.zeros((N, T), int)
    pi = np.array([.8, .1, .1])
    z[:, 0] = [rng.choice(K, p=pi) for _ in range(N)]
    Cs = (C - C.reshape(-1, 2).mean(0)) / C.reshape(-1, 2).std(0)
    for t in range(1, T):
        for n in range(N):
            logits = np.log(p["A"][z[n, t - 1]])
            if p["beta"] is not None:
                logits = logits + p["beta"][z[n, t - 1]] @ Cs[n, t]
                if p["confound"]:
                    logits[2] += 0.8 * difficulty[n]
            pr = np.exp(logits - logits.max()); pr /= pr.sum()
            z[n, t] = rng.choice(K, p=pr)
    X = p["means"][z] + rng.normal(0, p["sd"], (N, T, 3))
    return {"X": X, "mask": np.ones((N, T), bool), "C": Cs, "C_raw": C, "z": z,
            "labels": STATE_LABELS[z], "params": p, "synthetic": True, "scenario": name}
