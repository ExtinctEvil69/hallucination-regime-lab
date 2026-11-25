"""Metrics, onset risk set, paired dossier bootstrap and first-alarm policy."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    y = y[order]; s = s[order]
    # group ties: precision evaluated at the end of each tie block
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    last = np.r_[s[1:] != s[:-1], True]
    prec, rec = tp[last] / k[last], tp[last] / y.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y, float)) ** 2))


def log_score(y, p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(y, p, bins=10):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    table = [{"bin": b, "n": int((idx == b).sum()),
              "mean_p": float(p[idx == b].mean()) if (idx == b).any() else None,
              "obs": float(y[idx == b].mean()) if (idx == b).any() else None} for b in range(bins)]
    ece = float(sum(r["n"] * abs(r["mean_p"] - r["obs"]) for r in table if r["n"]) / len(y))
    # calibration intercept / slope: logistic regression of y on logit(p) (Newton)
    z = np.log(p / (1 - p)); Xd = np.c_[np.ones_like(z), z]; b = np.zeros(2)
    ok = True
    for _ in range(50):
        mu = 1 / (1 + np.exp(-Xd @ b)); W = mu * (1 - mu)
        H = Xd.T @ (Xd * W[:, None]) + 1e-8 * np.eye(2)
        try:
            step = np.linalg.solve(H, Xd.T @ (y - mu))
        except np.linalg.LinAlgError:
            ok = False; break
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"table": table, "ece": ece, "intercept": float(b[0]) if ok else None,
            "slope": float(b[1]) if ok else None}


def prob_metrics(y, p):
    return {"n": int(len(y)), "events": int(np.sum(y)), "auroc": auroc(y, p), "ap": average_precision(y, p),
            "brier": brier(y, p), "log_score": log_score(y, p), "calibration": calibration(y, p)}


# ----------------------------------------------------------- endpoints
def onset_risk_set(labels, mask):
    """Primary endpoint. Keep t <= tau (first H); target 1 only at tau.

    Dossiers whose turns before the first H include a pending label are
    excluded (onset ambiguous); ineligible (mask False) turns are dropped.
    Returns (sel (N,T) bool, y (N,T) int, excluded dossier indices).
    """
    N, T = labels.shape
    sel = np.zeros((N, T), bool); y = np.zeros((N, T), int); excluded = []
    for n in range(N):
        hs = np.flatnonzero(labels[n] == "H")
        tau = hs[0] if len(hs) else T - 1
        if np.any(labels[n, :tau + (0 if len(hs) else 1)] == "P"):
            excluded.append(n); continue
        sel[n, :tau + 1] = mask[n, :tau + 1]
        if len(hs):
            y[n, tau] = 1
    return sel, y, excluded


def next_turn_set(labels, mask):
    """Forecast endpoint: at t predict y_{t+1}=H, within the onset risk set (t+1 <= tau)."""
    sel, y, exc = onset_risk_set(labels, mask)
    N, T = labels.shape
    fsel = np.zeros((N, T), bool); fy = np.zeros((N, T), int)
    for n in range(N):
        if n in exc:
            continue
        for t in range(T - 1):
            if sel[n, t + 1] and labels[n, t] != "H":
                fsel[n, t] = True; fy[n, t] = y[n, t + 1]
    return fsel, fy


def paired_bootstrap(sel, y, score_a, score_b, metric=auroc, reps=2000, seed=0):
    """Whole-dossier paired bootstrap of metric(a) - metric(b)."""
    rng = np.random.default_rng(seed)
    N = sel.shape[0]
    base = metric(y[sel], score_a[sel]) - metric(y[sel], score_b[sel])
    diffs = []
    for _ in range(reps):
        idx = rng.integers(0, N, N)
        s, yy = sel[idx], y[idx]
        d = metric(yy[s], score_a[idx][s]) - metric(yy[s], score_b[idx][s])
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.array(diffs)
    est = float(base) if np.isfinite(base) else None          # non-finite -> null, never JSON NaN
    if len(diffs) == 0:                                         # [F4] undefined metric in every resample
        return {"estimate": est, "lo": None, "hi": None, "valid_reps": 0, "reps": reps, "p_gt0": None,
                "status": "unavailable: metric undefined in all resamples (single class?)"}
    return {"estimate": est, "lo": float(np.quantile(diffs, .025)), "hi": float(np.quantile(diffs, .975)),
            "valid_reps": int(len(diffs)), "reps": reps, "p_gt0": float(np.mean(diffs > 0))}


def bootstrap_ci(sel, y, score, metric=auroc, reps=2000, seed=0):
    rng = np.random.default_rng(seed); N = sel.shape[0]; vals = []
    for _ in range(reps):
        idx = rng.integers(0, N, N); s = sel[idx]
        v = metric(y[idx][s], score[idx][s])
        if np.isfinite(v):
            vals.append(v)
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))] if vals else [None, None]


# ----------------------------------------------------------- alarm policy
def first_alarm(risk, mask, thr):
    """Index of first eligible turn with risk > thr, or -1."""
    a = (risk > thr) & mask
    return np.where(a.any(1), a.argmax(1), -1)


def alarm_outcomes(risk, mask, labels, thr):
    fa = first_alarm(risk, mask, thr)
    out = {k: 0 for k in ["clean_false_stop", "clean_no_alarm", "premature", "on_time", "delayed", "miss",
                          "unresolved"]}
    delays = []
    for n in range(len(fa)):
        hs = np.flatnonzero(labels[n] == "H")
        if len(hs) == 0 and np.any(labels[n] == "P"):         # [F4] no H but unresolved: not verified clean
            out["unresolved"] += 1
            continue
        if len(hs) == 0:
            out["clean_false_stop" if fa[n] >= 0 else "clean_no_alarm"] += 1
            continue
        tau = hs[0]
        if fa[n] < 0:
            out["miss"] += 1
        elif fa[n] < tau:
            out["premature"] += 1
        elif fa[n] == tau:
            out["on_time"] += 1; delays.append(0)
        else:
            out["delayed"] += 1; delays.append(int(fa[n] - tau))
    clean = out["clean_false_stop"] + out["clean_no_alarm"]
    out["clean_false_stop_rate"] = out["clean_false_stop"] / clean if clean else None
    out["mean_delay"] = float(np.mean(delays)) if delays else None
    return out


def select_threshold(risk, mask, labels, max_false_stop=0.05):
    """Lowest threshold whose clean-conversation false-stop rate <= max (validation only).

    Clean = no H and no unresolved (P) label. Returns None (calibration unavailable) when the
    validation split has no clean conversations [F4]; it is never silently replaced by 1.0.
    """
    clean = ~np.any(labels == "H", axis=1) & ~np.any(labels == "P", axis=1)
    if clean.sum() == 0:
        return None
    cand = np.unique(np.r_[risk[mask], 1.0])
    for thr in cand:                               # ascending: first feasible = most sensitive
        fa = first_alarm(risk[clean], mask[clean], thr)
        if np.mean(fa >= 0) <= max_false_stop:
            return float(thr)
    return 1.0                                      # never alert
