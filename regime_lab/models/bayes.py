"""Fully Bayesian HMMs sampled with FFBS-Gibbs.

* Homogeneous model (M3): Dirichlet initial / sticky-Dirichlet transitions,
  Dirichlet mixture weights, Normal-Inverse-Gamma diagonal components —
  every conditional is conjugate.
* Covariate model (M4): transitions A_ij(c) = softmax_j(alpha_ij + beta_ij·c)
  with destination K-1 as reference (alpha_iK = beta_iK = 0),
  alpha ~ N(0, 1.5^2), beta_ij,k ~ N(0, sigma_k^2), sigma_k ~ HalfNormal(.5).
  Given sampled paths z, the (alpha_i, beta_i) block of each origin i is a
  multinomial-logistic regression; it is updated by Metropolis–Hastings with
  a Laplace-shaped Gaussian proposal adapted during warm-up only (frozen
  afterwards so the retained chain is a valid time-homogeneous MH kernel).

Hidden paths are drawn jointly per dossier with forward-filtering
backward-sampling; mixture indicators are drawn given the paths.

Design note: the blueprint's preferred sampler for the covariate model is
NUTS over the marginalised likelihood (PyMC). PyMC is not available for the
installed Python (3.14); MH-within-Gibbs targets the same joint posterior.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..util import logsumexp, softmax_log
from .emissions import GMMEmissions
from .hmm import EMHMM, filter_probs, forward


@dataclass
class Prior:
    a0: float = 3.0
    b0: float = 2.0
    m0: float = 0.0
    k0: float = 0.5
    kappa: float = 0.0          # sticky extra mass on the diagonal
    alpha_sd: float = 1.5       # covariate model intercept prior sd
    sigma_scale: float = 0.5    # HalfNormal scale of coefficient sds


# -------------------------------------------------------------- utilities
def _sample_cat(rng, logp):
    """Vectorised categorical draw from rows of log-probabilities."""
    p = softmax_log(logp, axis=-1)
    c = np.cumsum(p, axis=-1)
    u = rng.random(p.shape[:-1])[..., None]
    return np.minimum((c < u).sum(-1), p.shape[-1] - 1)


def cov_logA(theta: np.ndarray, C: np.ndarray) -> np.ndarray:
    """theta (K, K, 1+p) full logits (reference column zero); C (N,T,p) -> (N,T,K,K)."""
    Xd = np.concatenate([np.ones(C.shape[:2] + (1,)), C], axis=-1)       # (N,T,1+p)
    eta = np.einsum("ntq,ijq->ntij", Xd, theta)
    return eta - logsumexp(eta, axis=-1, keepdims=True)


def _perm_theta(theta, perm):
    p = list(perm)
    th = theta[np.ix_(p, p)].copy()
    return th - th[:, -1:, :]                    # re-reference to the new last destination


def ffbs(rng, log_b, log_pi, logA):
    f, _, la, ll = filter_probs(log_b, log_pi, logA)
    N, T, K = log_b.shape
    z = np.empty((N, T), int)
    z[:, -1] = _sample_cat(rng, la[:, -1])
    lf = np.log(np.maximum(f, 1e-300))
    for t in range(T - 2, -1, -1):
        A_t = logA if logA.ndim == 2 else logA[:, t + 1]
        col = A_t[:, z[:, t + 1]].T if logA.ndim == 2 else A_t[np.arange(N), :, z[:, t + 1]]
        z[:, t] = _sample_cat(rng, lf[:, t] + col)
    return z, ll


# ------------------------------------------------- multinomial-logit block
def _mnl_logpost(th_i, Xd, dest, prior_prec):
    """th_i (K-1, Q) free logits; Xd (n,Q); dest (n,) in 0..K-1."""
    eta = np.concatenate([Xd @ th_i.T, np.zeros((len(Xd), 1))], axis=1)
    lp = eta - logsumexp(eta, axis=1, keepdims=True)
    return lp[np.arange(len(dest)), dest].sum() - 0.5 * np.sum(prior_prec * th_i ** 2)


def _mnl_neg_hessian(th_i, Xd, prior_prec):
    Km1, Q = th_i.shape
    eta = np.concatenate([Xd @ th_i.T, np.zeros((len(Xd), 1))], axis=1)
    p = softmax_log(eta, axis=1)[:, :Km1]
    H = np.zeros((Km1 * Q, Km1 * Q))
    for a in range(Km1):
        for b in range(Km1):
            wab = p[:, a] * ((a == b) - p[:, b])
            H[a * Q:(a + 1) * Q, b * Q:(b + 1) * Q] = (Xd * wab[:, None]).T @ Xd
    return H + np.diag(prior_prec.ravel())


# --------------------------------------------------------------- the model
@dataclass
class BayesHMM:
    K: int = 3
    M: int = 2
    covariates: bool = False
    prior: Prior = field(default_factory=Prior)
    draws: dict = field(default_factory=dict)       # arrays (chains, S, ...)
    diagnostics: dict = field(default_factory=dict)

    def fit(self, X, mask, C=None, init: EMHMM | None = None, chains=4, warmup=1000, n_draws=1000,
            seed=0, mh_steps=2, verbose=False):
        if self.covariates and C is None:
            raise ValueError("covariate model needs C")
        rng0 = np.random.default_rng(seed)
        out = []
        for ch in range(chains):
            out.append(self._run_chain(X, mask, C, init, warmup, n_draws,
                                       np.random.default_rng(rng0.integers(2**31)), mh_steps, ch, verbose))
        self.draws = {k: np.stack([o[k] for o in out]) for k in out[0]}
        self.diagnostics["mh_accept"] = [o.get("_acc") for o in out]
        for k in list(self.draws):
            if k.startswith("_"):
                del self.draws[k]
        return self

    # .................................................................
    def _run_chain(self, X, mask, C, init, warmup, n_draws, rng, mh_steps, ch, verbose):
        K, M, pr = self.K, self.M, self.prior
        N, T, D = X.shape
        Xz = np.nan_to_num(X)
        # ---- initialisation: EM point (if given) plus jitter, else prior-ish
        if init is not None:
            w, mu, var = init.em.w.copy(), init.em.mu + rng.normal(0, .3, init.em.mu.shape), init.em.var.copy()
            pi, A = init.pi.copy(), init.A.copy()
        else:
            obs = X[mask]
            mu = obs[rng.choice(len(obs), K * M, replace=False)].reshape(K, M, D)
            var = np.tile(obs.var(0), (K, M, 1))
            w, pi = np.full((K, M), 1 / M), np.full(K, 1 / K)
            A = np.full((K, K), .2 / (K - 1)); np.fill_diagonal(A, .8)
        if self.covariates:
            p = C.shape[-1]
            theta = np.zeros((K, K, 1 + p))
            theta[:, :, 0] = np.log(A) - np.log(A[:, -1:])
            sigma = np.full(p, pr.sigma_scale)
            prop_cov = [np.eye((K - 1) * (1 + p)) * 0.05 for _ in range(K)]
            log_scale = np.zeros(K)
            Xd_all = np.concatenate([np.ones((N, T, 1)), C], axis=-1)
        alpha_dir = np.ones((K, K)) + pr.kappa * np.eye(K)
        total = warmup + n_draws
        rec = {k: [] for k in ["pi", "w", "mu", "var", "loglik"]}
        rec["theta" if self.covariates else "A"] = []
        if self.covariates:
            rec["sigma"] = []
        acc = np.zeros(K); tries = np.zeros(K)
        for it in range(total):
            em = GMMEmissions(w, mu, var)
            lb, lc = em.log_b(X, mask, return_components=True)
            logA = cov_logA(theta, C) if self.covariates else np.log(A)
            z, ll = ffbs(rng, lb, np.log(pi), logA)
            # mixture indicators
            lcz = np.take_along_axis(lc, z[:, :, None, None], axis=2)[:, :, 0]     # (N,T,M)
            m = _sample_cat(rng, lcz)
            onehot = ((z[..., None, None] == np.arange(K)[:, None]) &
                      (m[..., None, None] == np.arange(M)[None, :]) & mask[..., None, None]).astype(float)
            n = onehot.sum((0, 1))                                            # (K,M)
            s1 = np.einsum("ntkm,ntd->kmd", onehot, Xz)
            s2 = np.einsum("ntkm,ntd->kmd", onehot, Xz ** 2)
            xbar = s1 / np.maximum(n, 1)[..., None]
            S = np.maximum(s2 - n[..., None] * xbar ** 2, 0)
            kn = pr.k0 + n[..., None]
            mn = (pr.k0 * pr.m0 + n[..., None] * xbar) / kn
            an = pr.a0 + n[..., None] / 2
            bn = pr.b0 + S / 2 + pr.k0 * n[..., None] * (xbar - pr.m0) ** 2 / (2 * kn)
            var = bn / rng.gamma(np.broadcast_to(an, bn.shape))   # [F2] independent draw per feature
            mu = rng.normal(mn, np.sqrt(var / kn))
            w = np.stack([rng.dirichlet(1 + n[j]) for j in range(K)])
            pi = rng.dirichlet(1 + np.bincount(z[:, 0], minlength=K))
            if self.covariates:
                o, d = z[:, :-1].ravel(), z[:, 1:].ravel()
                Xd = Xd_all[:, 1:].reshape(-1, Xd_all.shape[-1])
                prior_prec = np.concatenate([[1 / pr.alpha_sd ** 2], 1 / sigma ** 2])[None, :].repeat(K - 1, 0)
                for i in range(K):
                    sel = o == i
                    Xi, di = Xd[sel], d[sel]
                    th = theta[i, :K - 1].copy()
                    if it < warmup and it % 25 == 0 and it > 0:
                        Hn = _mnl_neg_hessian(th, Xi, prior_prec)
                        prop_cov[i] = np.linalg.inv(Hn + 1e-6 * np.eye(len(Hn)))
                    Lc = np.linalg.cholesky(prop_cov[i] + 1e-10 * np.eye(len(prop_cov[i])))
                    cur = _mnl_logpost(th, Xi, di, prior_prec)
                    sc = np.exp(log_scale[i]) * 2.38 / np.sqrt(th.size)
                    for _ in range(mh_steps):
                        propv = th.ravel() + sc * Lc @ rng.normal(size=th.size)
                        prop = propv.reshape(th.shape)
                        new = _mnl_logpost(prop, Xi, di, prior_prec)
                        tries[i] += 1
                        if np.log(rng.random()) < new - cur:
                            th, cur = prop, new
                            acc[i] += 1
                            if it < warmup:
                                log_scale[i] += 0.7 / np.sqrt(it + 1)
                        elif it < warmup:
                            log_scale[i] -= 0.3 / np.sqrt(it + 1)
                    theta[i, :K - 1] = th
                # hierarchical coefficient scales (MH on log sigma, Jacobian included)
                B = theta[:, :K - 1, 1:]                                       # (K, K-1, p)
                for k in range(len(sigma)):
                    def lt(ls):
                        s = np.exp(ls)
                        return (-0.5 * (s / pr.sigma_scale) ** 2 - B[..., k].size * ls
                                - 0.5 * np.sum(B[..., k] ** 2) / s ** 2 + ls)
                    ls = np.log(sigma[k]); prop = ls + 0.3 * rng.normal()
                    if np.log(rng.random()) < lt(prop) - lt(ls):
                        sigma[k] = np.exp(prop)
            else:
                Nij = np.zeros((K, K))
                np.add.at(Nij, (z[:, :-1].ravel(), z[:, 1:].ravel()), 1)
                A = np.stack([rng.dirichlet(alpha_dir[i] + Nij[i]) for i in range(K)])
            if it >= warmup:
                rec["pi"].append(pi); rec["w"].append(w); rec["mu"].append(mu); rec["var"].append(var)
                rec["loglik"].append(float(ll.sum()))
                if self.covariates:
                    rec["theta"].append(theta.copy()); rec["sigma"].append(sigma.copy())
                else:
                    rec["A"].append(A)
            if verbose and it % 200 == 0:
                print(f"chain {ch} iter {it} loglik {ll.sum():.1f}")
        out = {k: np.array(v) for k, v in rec.items()}
        out["_acc"] = (acc / np.maximum(tries, 1)).tolist() if self.covariates else None
        return out

    # ................................................ label alignment
    def align(self, reference: GMMEmissions):
        """Permute every draw to best match the reference state signatures."""
        ref = reference.state_means()
        perms = list(itertools.permutations(range(self.K)))
        ch, S = self.draws["mu"].shape[:2]
        switched = 0
        for c in range(ch):
            for s in range(S):
                sig = np.einsum("km,kmd->kd", self.draws["w"][c, s], self.draws["mu"][c, s])
                cost = [np.sum((sig[list(p)] - ref) ** 2) for p in perms]
                p = list(perms[int(np.argmin(cost))])
                if p != list(range(self.K)):
                    switched += 1
                    self._permute_one(c, s, p)
        self.diagnostics["aligned_fraction_permuted"] = switched / (ch * S)
        return self

    def _permute_one(self, c, s, p):
        for key in ["w", "mu", "var"]:
            self.draws[key][c, s] = self.draws[key][c, s][p]
        self.draws["pi"][c, s] = self.draws["pi"][c, s][p]
        if self.covariates:
            self.draws["theta"][c, s] = _perm_theta(self.draws["theta"][c, s], p)
        else:
            self.draws["A"][c, s] = self.draws["A"][c, s][np.ix_(p, p)]

    def permute_all(self, p):
        """Apply one state relabelling to every draw (new state j = old state p[j])."""
        ch, S = self.draws["mu"].shape[:2]
        for c in range(ch):
            for s in range(S):
                self._permute_one(c, s, list(p))
        return self

    def filter_states(self, X, mask, C=None, thin=20):
        """Ensemble-mean filtered state probabilities (N,T,K)."""
        d = self.flat(thin)
        acc = 0
        for s in range(len(d["pi"])):
            em = GMMEmissions(d["w"][s], d["mu"][s], d["var"][s])
            acc = acc + filter_probs(em.log_b(X, mask), np.log(d["pi"][s]), self.logA_draw(d, s, C))[0]
        return acc / len(d["pi"])

    # ................................................ prediction
    def flat(self, thin=1):
        return {k: v.reshape((-1,) + v.shape[2:])[::thin] for k, v in self.draws.items()}

    def logA_draw(self, d, s, C=None, N=None, T=None):
        if self.covariates:
            return cov_logA(d["theta"][s], C)
        return np.log(d["A"][s])

    def predict(self, X, mask, C=None, C_next=None, thin=10, h=2, reweight=False):
        """Ensemble filtered H-risk and next-turn H-risk per turn.

        Returns dict with 'filt' (S,N,T) H-filtered per draw, 'mean', 'q05','q95',
        'next' (S,N,T) prospective risk for turn t+1 using known next covariates
        (C itself shifted, so C_next is optional), and optional reweighted means.
        """
        d = self.flat(thin)
        S = len(d["pi"])
        N, T, _ = X.shape
        filt = np.empty((S, N, T)); nxt = np.full((S, N, T), np.nan); pref = np.empty((S, N, T))
        for s in range(S):
            em = GMMEmissions(d["w"][s], d["mu"][s], d["var"][s])
            logA = self.logA_draw(d, s, C)
            f, q, la, _ = filter_probs(em.log_b(X, mask), np.log(d["pi"][s]), logA)
            filt[s] = f[..., h]
            pref[s] = logsumexp(la, axis=-1)            # log p(x_1:t | theta_s)
            A = np.exp(logA)
            for t in range(T - 1):
                At = A if A.ndim == 2 else A[:, t + 1]
                nxt[s, :, t] = (f[:, t] @ At)[..., h] if At.ndim == 2 else np.einsum("ni,nij->nj", f[:, t], At)[:, h]
        res = {"filt": filt, "next": nxt, "mean": filt.mean(0), "next_mean": nxt.mean(0),
               "q05": np.quantile(filt, .05, axis=0), "q95": np.quantile(filt, .95, axis=0)}
        if reweight:
            wts = softmax_log(pref, axis=0)                                   # (S,N,T)
            res["mean_rw"] = (wts * filt).sum(0)
            res["next_rw"] = np.nansum(wts * nxt, axis=0)
            res["ess_rw"] = 1 / (wts ** 2).sum(0)
        return res

    def transition_draws(self, C_profiles=None):
        """Flat transition matrices: (S,K,K) homogeneous or (S,P,K,K) per covariate profile."""
        d = self.flat()
        if not self.covariates:
            return d["A"]
        Cp = C_profiles[None]                                                   # (1,P,p)
        return np.stack([np.exp(cov_logA(th, Cp))[0] for th in d["theta"]])

    def save(self, path):
        np.savez_compressed(path, covariates=self.covariates, K=self.K, M=self.M,
                            **{f"draw_{k}": v for k, v in self.draws.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(int(z["K"]), int(z["M"]), bool(z["covariates"]))
        m.draws = {k[5:]: z[k] for k in z.files if k.startswith("draw_")}
        return m
