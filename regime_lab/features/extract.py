"""Causal per-turn features from the *raw* record only (never the gold/grade).

x = (mean selected-token log-prob  l,
     mean top-k entropy            e   [top-k alternatives + residual bucket;
                                        a lower-bound proxy, NOT full-vocab entropy],
     max lexical cosine to visible dossier rows  s)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..data.dossiers import PRIMARY_COVARIATES, WORD_RE

FEATURES = ("logprob", "entropy", "lexcos")


def _bag(text):
    return Counter(WORD_RE.findall(text.lower()))


def cosine_max(answer: str, rows: list[str]) -> float:
    a = _bag(answer)
    na = np.sqrt(sum(v * v for v in a.values()))
    if na == 0:
        return 0.0
    best = 0.0
    for r in rows:
        b = _bag(r)
        nb = np.sqrt(sum(v * v for v in b.values()))
        if nb:
            best = max(best, sum(a[k] * b[k] for k in a) / (na * nb))
    return float(best)


def token_stats(tokens):
    """Mean selected-token log-prob and mean top-k entropy over the SAME tokens.

    [F8] Every counted token must carry its top-k alternatives; if any token lacks them the
    entropy is undefined (NaN) and the turn is ineligible, instead of silently averaging the two
    features over different token sets.
    """
    lps, ents = [], []
    for t in tokens:
        if t.get("token", "") == "" or t.get("logprob") is None:
            continue
        lps.append(t["logprob"])
        top = np.exp(np.array([x for x in t.get("top", []) if x is not None], float))
        top = top[top > 0]
        if len(top) == 0:
            return float(np.mean(lps)), np.nan          # incomplete coverage -> ineligible
        r = max(0.0, 1.0 - top.sum())
        ents.append(-np.sum(top * np.log(top)) - (r * np.log(r) if r > 0 else 0.0))
    if not lps:
        return np.nan, np.nan
    return float(np.mean(lps)), float(np.mean(ents))


def turn_features(raw_turn: dict, rows: list[str]):
    l, e = token_stats(raw_turn.get("tokens", []))
    s = cosine_max(raw_turn.get("answer", ""), rows)
    eligible = bool(raw_turn.get("answer", "").strip()) and np.isfinite(l) and np.isfinite(e)
    return np.array([l, e, s]), eligible


@dataclass
class Dataset:
    ids: np.ndarray           # (N,)
    X: np.ndarray             # (N,T,3) raw features
    mask: np.ndarray          # (N,T) eligible
    C: np.ndarray             # (N,T,p) raw covariates (PRIMARY_COVARIATES order)
    context_len: np.ndarray   # (N,T)
    qtype: np.ndarray         # (N,T) object
    labels: np.ndarray        # (N,T) object 'G','U','H','P'  (audit/evaluation only)
    reasons: np.ndarray
    answers: np.ndarray
    questions: np.ndarray
    raws: list

    def subset(self, sel):
        sel = np.asarray(sel)
        return Dataset(self.ids[sel], self.X[sel], self.mask[sel], self.C[sel], self.context_len[sel],
                       self.qtype[sel], self.labels[sel], self.reasons[sel], self.answers[sel],
                       self.questions[sel], [self.raws[i] for i in np.flatnonzero(sel)] if sel.dtype == bool
                       else [self.raws[i] for i in sel])


def build_dataset(raws: list[dict], grades: dict[int, dict] | None = None) -> Dataset:
    N, T = len(raws), len(raws[0]["turns"])
    X = np.full((N, T, 3), np.nan); mask = np.zeros((N, T), bool)
    C = np.zeros((N, T, len(PRIMARY_COVARIATES))); ctx = np.zeros((N, T))
    qt = np.empty((N, T), object); lab = np.full((N, T), "P", object); rea = np.empty((N, T), object)
    ans = np.empty((N, T), object); qs = np.empty((N, T), object)
    for n, r in enumerate(raws):
        for t, tr in enumerate(r["turns"]):
            X[n, t], mask[n, t] = turn_features(tr, r["rows"])
            C[n, t] = [tr["covariates"][k] for k in PRIMARY_COVARIATES]
            ctx[n, t] = tr["covariates"]["context_len"]
            qt[n, t], ans[n, t], qs[n, t] = tr["qtype"], tr["answer"], tr["question"]
        if grades is not None:
            for t, g in enumerate(grades[r["dossier_id"]]["labels"]):
                lab[n, t], rea[n, t] = g["label"], g["reason"]
    return Dataset(np.array([r["dossier_id"] for r in raws]), X, mask, C, ctx, qt, lab, rea, ans, qs, raws)


TRANSFORMS = ("none", "log_surprisal")


def transform_features(X, kind="none"):
    """Fixed (non-fitted) monotone transforms; 'log_surprisal' maps
    l -> log(1e-4 + (-l)) and e -> log(1e-4 + e), spreading the near-zero
    point mass produced by very confident generations. Cosine unchanged."""
    if kind in (None, "none"):
        return X
    if kind != "log_surprisal":
        raise ValueError(kind)
    Y = np.array(X, float, copy=True)
    Y[..., 0] = np.log(1e-4 + np.maximum(-X[..., 0], 0))
    Y[..., 1] = np.log(1e-4 + np.maximum(X[..., 1], 0))
    return Y


class Scaler:
    """Training-only standardisation; constant columns get scale 1."""

    def fit(self, X, mask):
        obs = X[mask]
        self.mean = obs.mean(0)
        sd = obs.std(0)
        self.constant = sd < 1e-12
        self.scale = np.where(self.constant, 1.0, sd)
        return self

    def transform(self, X):
        return (X - self.mean) / self.scale

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "constant": self.constant.tolist()}
