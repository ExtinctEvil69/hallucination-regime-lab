"""MCMC convergence diagnostics (Vehtari et al., 2021, Bayesian Analysis).

rank-normalised split R-hat (max of bulk and folded), bulk / tail ESS with
Geyer's initial-monotone-sequence estimator, and Monte Carlo standard error.
Input arrays have shape (chains, draws).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def _split(x):
    c, n = x.shape
    h = n // 2
    return np.concatenate([x[:, :h], x[:, n - h:]], axis=0)


def _z(x):
    r = rankdata(x, method="average").reshape(x.shape)
    return norm.ppf((r - 3 / 8) / (x.size - 1 / 4))


def _rhat_raw(x):
    m, n = x.shape
    if n < 2:
        return np.nan
    B = n * x.mean(1).var(ddof=1)
    W = x.var(1, ddof=1).mean()
    if W <= 0:
        return np.nan
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def rhat(x):
    x = np.asarray(x, float)
    if np.ptp(x) == 0:
        return np.nan
    s = _split(x)
    bulk = _rhat_raw(_z(s))
    fold = _rhat_raw(_z(np.abs(s - np.median(s))))
    return float(max(bulk, fold))


def _ess_raw(x):
    m, n = x.shape
    if n < 4 or np.ptp(x) == 0:
        return np.nan
    xc = x - x.mean(1, keepdims=True)
    f = np.fft.rfft(xc, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    W = x.var(1, ddof=1).mean()
    var_plus = (n - 1) / n * W + (x.mean(1).var(ddof=1) if m > 1 else 0)
    rho = 1 - (W - acov.mean(0)) / var_plus
    rho[0] = 1
    # Geyer initial positive + monotone sequence on paired sums
    P = []
    for t in range(0, n - 1, 2):
        s = rho[t] + rho[t + 1]
        if s < 0:
            break
        P.append(s)
    P = np.minimum.accumulate(np.array(P)) if P else np.array([1.0])
    tau = -1 + 2 * P.sum()
    return float(m * n / max(tau, 1 / np.log10(m * n)))


def ess_bulk(x):
    return _ess_raw(_z(_split(np.asarray(x, float))))


def ess_tail(x):
    x = np.asarray(x, float)
    q05, q95 = np.quantile(x, [.05, .95])
    e1 = _ess_raw(_split((x <= q05).astype(float)))
    e2 = _ess_raw(_split((x <= q95).astype(float)))
    return float(np.nanmin([e1, e2]))


def mcse_mean(x):
    x = np.asarray(x, float)
    e = _ess_raw(_split(x))
    return float(x.std(ddof=1) / np.sqrt(e)) if e and np.isfinite(e) else np.nan


def summarize(name, x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None]
    return {"name": name, "mean": float(x.mean()), "sd": float(x.std()),
            "q2.5": float(np.quantile(x, .025)), "q97.5": float(np.quantile(x, .975)),
            "rhat": rhat(x), "ess_bulk": ess_bulk(x), "ess_tail": ess_tail(x), "mcse": mcse_mean(x)}


def autocorr(x, max_lag=40):
    x = np.asarray(x, float).ravel() - np.mean(x)
    v = np.dot(x, x)
    return np.array([np.dot(x[:len(x) - k], x[k:]) / v for k in range(min(max_lag, len(x) - 1))])
