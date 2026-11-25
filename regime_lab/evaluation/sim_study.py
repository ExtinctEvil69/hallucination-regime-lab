"""Simulation programme: scenario recovery, interval coverage and SBC (all SYNTHETIC)."""
from __future__ import annotations

import numpy as np
from scipy.stats import chisquare

from ..models.bayes import BayesHMM, Prior
from ..models.hmm import EMHMM, map_states
from . import metrics as mt
from .simulate import simulate

SCENARIOS = ["independent", "persistent", "abrupt", "overlapping", "high_conf_wrong",
             "covariate", "covariate_null", "confounded"]


def _fit_and_score(sim, seed, chains=2, warmup=300, n_draws=300, covariates=False):
    N = sim["X"].shape[0]
    split = np.array(["train"] * int(.6 * N) + ["val"] * int(.2 * N) + ["test"] * (N - int(.8 * N)))
    tr, va, te = split == "train", split == "val", split == "test"
    X, m, lab = sim["X"], sim["mask"], sim["labels"]
    em = EMHMM().fit(X[tr], m[tr], n_starts=3, seed=seed)
    mp = map_states(em.smooth(X[va], m[va])[0], lab[va], m[va])
    em = em.permute([mp["perm"].index(i) for i in range(3)])
    f = em.filter(X, m)[0][..., 2]; s = em.static_scores(X, m)[..., 2]
    sel, y, _ = mt.onset_risk_set(lab[te], m[te])
    res = {"onset_auroc_filter": mt.auroc(y[sel], f[te][sel]), "onset_auroc_static": mt.auroc(y[sel], s[te][sel]),
           "onsets": int(y[sel].sum()), "mapping_stable": mp["stable"]}
    bm = BayesHMM(covariates=covariates).fit(X[tr], m[tr], sim["C"][tr] if covariates else None, init=em,
                                              chains=chains, warmup=warmup, n_draws=n_draws, seed=seed)
    bm.align(em.em)
    return res, bm, em, tr


def recovery(scenario, reps=5, N=300, seed=0, log=print):
    """Repeated seeded datasets: transition bias/RMSE/coverage (homogeneous truth) or contrast signs."""
    rows = []
    for r in range(reps):
        sim = simulate(scenario, N=N, seed=seed + r)
        cov = sim["params"]["beta"] is not None
        res, bm, em, tr = _fit_and_score(sim, seed + r, covariates=cov)
        if not cov:
            A_true = sim["params"]["A"]
            Ad = bm.flat()["A"]
            lo, hi = np.quantile(Ad, .025, 0), np.quantile(Ad, .975, 0)
            res.update(A_bias=float(np.mean(Ad.mean(0) - A_true)),
                       A_rmse=float(np.sqrt(np.mean((Ad.mean(0) - A_true) ** 2))),
                       A_coverage95=float(np.mean((A_true >= lo) & (A_true <= hi))))
        else:
            from ..inference.attribution import transition_contrasts
            prof = sim["C"][tr][:, 1:].reshape(-1, 2)
            lo0, hi0 = prof[:, 0].min(), prof[:, 0].max()
            byo, _ = transition_contrasts(bm.flat(thin=5)["theta"], prof, 0, lo0, hi0, 2, origins=[0])
            res.update(evidence_effect_enter_H=byo[0],
                       excludes_zero=bool(byo[0]["hi"] < 0 or byo[0]["lo"] > 0),
                       true_sign=("negative" if scenario != "covariate_null" else "null"))
        rows.append(res)
        log(f"  {scenario} rep {r}: {({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, dict)})}")
    agg = {"scenario": scenario, "reps": reps, "N": N, "synthetic": True}
    for k in rows[0]:
        vals = [x[k] for x in rows if isinstance(x[k], (int, float, bool, np.floating)) and x[k] == x[k]]
        if vals and not isinstance(rows[0][k], dict):
            agg[k] = float(np.mean(vals))
    agg["rows"] = rows
    return agg


def sbc(reps=100, N=40, T=6, L=99, seed=0, log=print, M=2, warmup=300):
    """Simulation-based calibration for permutation-invariant quantities.

    Draw theta from the sampler's exact prior, simulate data, run the Gibbs
    sampler (from the prior-ish init), and record the rank of the prior draw
    among L thinned posterior draws for trace(A) and the max state mean of
    feature 0. Uniform ranks indicate a correct sampler.
    """
    rng = np.random.default_rng(seed)
    pr = Prior()
    K, D = 3, 3
    ranks = {"trace_A": [], "max_mu0": [], "sum_pi_sq": []}
    for r in range(reps):
        pi = rng.dirichlet(np.ones(K)); A = np.stack([rng.dirichlet(np.ones(K)) for _ in range(K)])
        w = np.stack([rng.dirichlet(np.ones(M)) for _ in range(K)])
        var = pr.b0 / rng.gamma(pr.a0, size=(K, M, D)); mu = rng.normal(pr.m0, np.sqrt(var / pr.k0))
        z = np.zeros((N, T), int); z[:, 0] = rng.choice(K, N, p=pi)
        for t in range(1, T):
            z[:, t] = [rng.choice(K, p=A[i]) for i in z[:, t - 1]]
        m = np.array([[rng.choice(M, p=w[z[n, t]]) for t in range(T)] for n in range(N)])
        X = rng.normal(mu[z, m], np.sqrt(var[z, m]))
        bm = BayesHMM(M=M).fit(X, np.ones((N, T), bool), chains=1, warmup=warmup, n_draws=L * 4, seed=int(rng.integers(1e9)))
        d = bm.flat(thin=4)
        stats_true = {"trace_A": np.trace(A), "max_mu0": np.max(np.einsum("km,km->k", w, mu[..., 0])),
                      "sum_pi_sq": np.sum(pi ** 2)}
        stats_post = {"trace_A": np.trace(d["A"], axis1=1, axis2=2),
                      "max_mu0": np.max(np.einsum("skm,skm->sk", d["w"], d["mu"][..., 0]), axis=1),
                      "sum_pi_sq": np.sum(d["pi"] ** 2, axis=1)}
        for k in ranks:
            ranks[k].append(int(np.sum(stats_post[k][:L] < stats_true[k])))
        if (r + 1) % 20 == 0:
            log(f"  SBC {r + 1}/{reps}")
    out = {"reps": reps, "L": L, "N": N, "M": M, "warmup": warmup, "synthetic": True}
    for k, v in ranks.items():
        hist = np.bincount(np.array(v) * 10 // (L + 1), minlength=10)
        out[k] = {"hist10": hist.tolist(), "chi2_p": float(chisquare(hist).pvalue), "ranks": v}
    return out
