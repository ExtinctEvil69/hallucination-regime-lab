"""Exact HMM recursions (log space), Baum–Welch EM, matched static control.

Conventions
-----------
log_b   : (N, T, K) emission log-densities (0 for masked turns)
log_pi  : (K,)
logA    : (K, K) homogeneous, or (N, T, K, K) where logA[n, t] is the
          transition *into* turn t from turn t-1 (index t=0 unused).
All sequences in a batch share T; dossier boundaries are batch rows, so no
transition ever crosses dossiers.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..util import logsumexp, softmax_log
from .emissions import GMMEmissions


# ----------------------------------------------------------------- recursions
def _At(logA: np.ndarray, t: int) -> np.ndarray:
    return logA if logA.ndim == 2 else logA[:, t]


def forward(log_b, log_pi, logA):
    """Returns log_alpha (N,T,K) and log-likelihood (N,)."""
    N, T, K = log_b.shape
    la = np.empty((N, T, K))
    la[:, 0] = log_pi[None] + log_b[:, 0]
    for t in range(1, T):
        la[:, t] = logsumexp(la[:, t - 1][:, :, None] + _At(logA, t), axis=1) + log_b[:, t]
    return la, logsumexp(la[:, -1], axis=1)


def backward(log_b, logA):
    N, T, K = log_b.shape
    lb = np.zeros((N, T, K))
    for t in range(T - 2, -1, -1):
        lb[:, t] = logsumexp(_At(logA, t + 1) + (log_b[:, t + 1] + lb[:, t + 1])[:, None, :], axis=2)
    return lb


def filter_probs(log_b, log_pi, logA):
    """Causal filter f_t = p(z_t | x_1:t) and one-step prior q_t, both (N,T,K)."""
    la, ll = forward(log_b, log_pi, logA)
    f = softmax_log(la, axis=-1)
    N, T, K = log_b.shape
    q = np.empty_like(f)
    q[:, 0] = np.exp(log_pi)[None]
    for t in range(1, T):
        q[:, t] = softmax_log(logsumexp(np.log(np.maximum(f[:, t - 1], 1e-300))[:, :, None]
                                        + _At(logA, t), axis=1), axis=-1)
    return f, q, la, ll


def smooth(log_b, log_pi, logA):
    """Posterior marginals gamma (N,T,K) and pair marginals xi (N,T-1,K,K)."""
    la, ll = forward(log_b, log_pi, logA)
    lbk = backward(log_b, logA)
    gamma = softmax_log(la + lbk, axis=-1)
    N, T, K = log_b.shape
    xi = np.empty((N, max(T - 1, 0), K, K))
    for t in range(T - 1):
        lx = (la[:, t][:, :, None] + _At(logA, t + 1)
              + (log_b[:, t + 1] + lbk[:, t + 1])[:, None, :] - ll[:, None, None])
        xi[:, t] = np.exp(lx)
    return gamma, xi, ll


def viterbi(log_b, log_pi, logA):
    """Jointly most probable path; ties resolved by lowest index."""
    N, T, K = log_b.shape
    delta = log_pi[None] + log_b[:, 0]
    psi = np.zeros((N, T, K), int)
    for t in range(1, T):
        cand = delta[:, :, None] + _At(logA, t)          # (N, i, j)
        psi[:, t] = np.argmax(cand, axis=1)               # first max = lowest index
        delta = np.take_along_axis(cand, psi[:, t][:, None, :], axis=1)[:, 0] + log_b[:, t]
    path = np.zeros((N, T), int)
    path[:, -1] = np.argmax(delta, axis=1)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = psi[np.arange(N), t, path[:, t]]
    return path, delta.max(axis=1)


def predict_next(f_t: np.ndarray, logA_next: np.ndarray) -> np.ndarray:
    """State distribution for the next turn before observing it: f_t A(c_{t+1})."""
    A = np.exp(logA_next)
    if A.ndim == 2:
        return f_t @ A
    return np.einsum("ni,nij->nj", f_t, A)


# ---------------------------------------------------------------- state maps
def map_states(post: np.ndarray, labels: np.ndarray, mask: np.ndarray, classes=("G", "U", "H")):
    """Choose the state->label permutation maximising class-balanced agreement.

    post: (N,T,K) posterior probabilities (validation); labels: (N,T) object
    array of 'G'/'U'/'H'/other. Returns dict(perm, score, table, stable).
    perm[j] = label index assigned to latent state j.
    """
    K = post.shape[-1]
    present = [c for c in classes if np.any((labels == c) & mask)]
    rows = []
    for perm in itertools.permutations(range(K)):
        scores = []
        for li, c in enumerate(classes):
            sel = (labels == c) & mask
            if not sel.any():
                continue
            j = perm.index(li)
            scores.append(post[sel][:, j].mean())
        rows.append((float(np.mean(scores)) if scores else np.nan, perm))
    rows.sort(key=lambda r: -np.nan_to_num(r[0], nan=-1))
    best, second = rows[0], rows[1]
    stable = (len(present) == len(classes)) and (best[0] - second[0] > 0.02)
    return {"perm": list(best[1]), "score": best[0], "runner_up": second[0],
            "classes_present": present, "stable": bool(stable),
            "table": [{"perm": list(p), "score": s} for s, p in rows]}


def h_index(perm) -> int:
    """Latent state index mapped to label H (label index 2)."""
    return list(perm).index(2)


# ---------------------------------------------------------------- EM model
@dataclass
class EMHMM:
    K: int = 3
    M: int = 2
    pi: np.ndarray | None = None
    A: np.ndarray | None = None
    em: GMMEmissions | None = None
    history: list = field(default_factory=list)
    occupancy: np.ndarray | None = None

    # ---- inference API
    def log_params(self):
        return np.log(np.maximum(self.pi, 1e-300)), np.log(np.maximum(self.A, 1e-300))

    def filter(self, X, mask):
        lpi, lA = self.log_params()
        f, q, _, ll = filter_probs(self.em.log_b(X, mask), lpi, lA)
        return f, q, ll

    def step(self, x, previous=None):
        """Online update for one turn; x shape (D,) or None/NaN for ineligible."""
        lpi, lA = self.log_params()
        prior = np.exp(lpi) if previous is None else np.asarray(previous) @ self.A
        ok = x is not None and np.all(np.isfinite(x))
        lb = self.em.log_b(np.asarray(x if ok else np.zeros(self.em.mu.shape[-1]), float)[None, None],
                           np.array([[ok]]))[0, 0]
        return softmax_log(np.log(np.maximum(prior, 1e-300)) + lb)

    def smooth(self, X, mask):
        lpi, lA = self.log_params()
        return smooth(self.em.log_b(X, mask), lpi, lA)

    def viterbi(self, X, mask):
        lpi, lA = self.log_params()
        return viterbi(self.em.log_b(X, mask), lpi, lA)

    def static_scores(self, X, mask):
        """Matched static mixture: s_j(x) ∝ occ_j b_j(x), same emissions."""
        lb = self.em.log_b(X, mask)
        return softmax_log(np.log(self.occupancy)[None, None] + lb, axis=-1)

    def negative_control(self) -> "EMHMM":
        """HMM whose rows of A and pi equal occupancy: causal filter == static."""
        K = self.K
        return EMHMM(K, self.M, self.occupancy.copy(), np.tile(self.occupancy, (K, 1)),
                     self.em.copy(), [], self.occupancy.copy())

    def forecast(self, f_t, horizon: int = 1, h_state: int = 2):
        """Occupancy p(z_{t+h}=H) and first-entry P(enter H by t+h | not H now)."""
        A = self.A
        occ, first = [], []
        nonh = [j for j in range(self.K) if j != h_state]
        Q = A[np.ix_(nonh, nonh)]
        g = f_t[..., nonh]
        g = g / np.maximum(g.sum(-1, keepdims=True), 1e-300)
        dist, Qh = f_t.copy(), np.eye(len(nonh))
        for _ in range(horizon):
            dist = dist @ A
            Qh = Qh @ Q
            occ.append(dist[..., h_state])
            first.append(1 - (g @ Qh).sum(-1))
        return np.array(occ), np.array(first)

    # ---- learning
    def fit(self, X, mask, n_starts=3, max_iter=300, tol=1e-6, var_floor=1e-3, seed=0):
        best = None
        for s in range(n_starts):
            cand = EMHMM(self.K, self.M)
            cand._init(X, mask, np.random.default_rng(seed + 1000 * s))
            cand._em(X, mask, max_iter, tol, var_floor)
            cand.history = [dict(start=s, **h) for h in cand.history]
            if best is None or cand.history[-1]["loglik"] > best.history[-1]["loglik"]:
                best, best_start = cand, s
            self.history.extend(cand.history)
        self.pi, self.A, self.em = best.pi, best.A, best.em
        self.selected_start = best_start
        gamma, _, _ = self.smooth(X, mask)
        occ = gamma[mask].mean(0)
        self.occupancy = np.maximum(occ, 1e-6) / np.maximum(occ, 1e-6).sum()
        return self

    def _init(self, X, mask, rng):
        K, M = self.K, self.M
        obs = X[mask]
        D = obs.shape[1]
        idx = rng.choice(len(obs), K * M, replace=False)
        mu = obs[idx].reshape(K, M, D)
        var = np.tile(obs.var(0) + 1e-3, (K, M, 1))
        self.em = GMMEmissions(np.full((K, M), 1 / M), mu, var)
        self.pi = np.full(K, 1 / K)
        A = np.full((K, K), 0.2 / (K - 1))
        np.fill_diagonal(A, 0.8)
        self.A = A

    def _em(self, X, mask, max_iter, tol, var_floor):
        prev = -np.inf
        Xz = np.nan_to_num(X)
        for it in range(max_iter):
            lpi, lA = self.log_params()
            lb, lc = self.em.log_b(X, mask, return_components=True)
            gamma, xi, ll = smooth(lb, lpi, lA)
            L = float(ll.sum())
            # E-step responsibilities rho_{ntjm}
            rho = gamma[..., None] * np.exp(lc - lb[..., None]) * mask[..., None, None]
            # M-step
            self.pi = np.maximum(gamma[:, 0].mean(0), 1e-12)
            self.pi /= self.pi.sum()
            num = xi.sum((0, 1))
            self.A = np.maximum(num, 1e-12) / np.maximum(num, 1e-12).sum(1, keepdims=True)
            Nk = rho.sum((0, 1))                                   # (K,M)
            empty = Nk < 1e-8
            w = np.maximum(Nk, 1e-12) / np.maximum(Nk, 1e-12).sum(1, keepdims=True)
            mu = np.einsum("ntkm,ntd->kmd", rho, Xz) / np.maximum(Nk, 1e-12)[..., None]
            sq = np.einsum("ntkm,ntkmd->kmd", rho, (Xz[:, :, None, None] - mu[None, None]) ** 2)
            var = np.maximum(sq / np.maximum(Nk, 1e-12)[..., None], var_floor)
            mu[empty], var[empty] = self.em.mu[empty], self.em.var[empty]   # empty-component policy
            self.em = GMMEmissions(w, mu, var)
            self.history.append({"iter": it, "loglik": L, "occupancy": (Nk.sum(1) / Nk.sum()).tolist(),
                                 "empty_components": int(empty.sum())})
            if abs(L - prev) < tol * max(1.0, abs(L)):
                break
            prev = L
        lpi, lA = self.log_params()
        self.history.append({"iter": it + 1, "loglik": float(forward(self.em.log_b(X, mask), lpi, lA)[1].sum()),
                             "final": True})

    def permute(self, perm):
        p = list(perm)
        return EMHMM(self.K, self.M, self.pi[p], self.A[np.ix_(p, p)], self.em.permute(p),
                     list(self.history), None if self.occupancy is None else self.occupancy[p])

    # ---- persistence
    def save(self, path):
        np.savez(path, pi=self.pi, A=self.A, w=self.em.w, mu=self.em.mu, var=self.em.var,
                 occupancy=self.occupancy, K=self.K, M=self.M)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(int(z["K"]), int(z["M"]), z["pi"], z["A"], GMMEmissions(z["w"], z["mu"], z["var"]),
                   [], z["occupancy"])


def save_path(p) -> Path:
    return Path(p)
