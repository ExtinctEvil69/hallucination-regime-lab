"""Use case: Bayesian regime guard for a clinical medication-reconciliation assistant.

The guard wraps a local LLM (Qwen via Ollama) that answers questions about a
(fictional) ward chart. It never sees gold answers. At every turn it
  1. computes pre-answer covariates from the visible chart + question text,
  2. forecasts next-turn hallucination risk r_{t+1} = E_theta[(f_t A_theta(c_{t+1}))_H]
     by averaging over posterior draws of the covariate Bayesian HMM (M4),
  3. if r_{t+1} exceeds the forecast threshold, prepends an evidence
     re-grounding reminder to the question (a cheap pre-emptive action),
  4. after the answer, updates the causal filter f_t(H) from the answer's
     features (log-prob, top-k entropy, lexical grounding) and HOLDS the
     answer for human verification if f_t(H) exceeds the alarm threshold.
A hold cannot undo an emitted answer; it prevents it from being auto-committed
to the reconciliation list.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..data.dossiers import DOMAINS, PRIMARY_COVARIATES, approx_tokens
from ..evaluation.intervention import REGROUND
from ..features.extract import transform_features, turn_features
from ..models.bayes import BayesHMM, cov_logA
from ..models.emissions import GMMEmissions
from ..util import read_json, softmax_log

ROW_RE = re.compile(r"^(?P<name>.+?) \((?:born|registered) (?P<year>\d{4})\) \| (?P<field>[^:]+): (?P<code>\S+)$")


def parse_rows(rows):
    out = []
    for r in rows:
        m = ROW_RE.match(r)
        if m:
            out.append({**m.groupdict(), "year": int(m["year"])})
    return out


COREF_RE = re.compile(r"\b(?:that|this|the same)\s+(?:patient|person|entity|one|individual)\b|"
                      r"\b(?:their|they|them)\b", re.I)


def live_covariates(rows, question, prev_entity, turn, domain="clinical", return_status=False, coref="v1"):
    """Pre-answer covariates from visible information only (no gold).

    coref="v1" (locked default): literal "that <noun>" -> previous entity, else full name, else surname.
    coref="v2" [W9 v2]: full name -> surname (+year) -> bounded coreference phrase ("that patient/person",
    "their", ...) resolved to the previous turn's entity only if one exists -> "unresolved". An unresolved
    reference is reported as such (it is not a known-missing row).
    """
    parsed = parse_rows(rows)
    names = sorted({(p["name"], p["year"]) for p in parsed} | set())
    fields = DOMAINS[domain]["fields"]
    field = max((f for f in fields if f in question), key=len, default=None)
    entity, amb, status = None, 1, "unresolved"
    full = [n for n in names if n[0] in question]

    def by_surname():
        yr = re.search(r"\b(19|20)\d{2}\b", question)
        sur = [n for n in names if re.search(rf"\b{re.escape(n[0].split()[-1])}\b", question)]
        a_ = max(1, len({n[0] for n in sur}))
        if yr:
            sur = [n for n in sur if n[1] == int(yr.group())] or sur
        e_ = sur[0] if sur else None
        return e_, a_, (("surname_year" if yr else "surname") if e_ is not None else "unresolved")

    if coref == "v2":
        # [W9 v2, 3/3]: explicit full name -> surname (+year) -> bounded coreference -> unresolved
        if full:
            entity, status = full[0], "explicit_name"
        else:
            entity, amb, status = by_surname()
            if entity is None and COREF_RE.search(question) and prev_entity is not None:
                entity, status = tuple(prev_entity), "coreference_previous_turn"
    else:
        # v1 (locked, historical audits): literal "that <noun>" first, then full name, then surname
        if f"that {DOMAINS[domain]['noun']}" in question and prev_entity is not None:
            entity, status = tuple(prev_entity), "coreference_previous_turn"
        elif full:
            entity, status = full[0], "explicit_name"
        else:
            entity, amb, status = by_surname()
    evidence = int(any(p["name"] == (entity or ("", 0))[0] and p["field"] == field for p in parsed))
    col = [p["code"][:3] for p in parsed if p["field"] == field]
    distractor = sum(col.count(c) > 1 for c in col)
    cov = {"evidence": evidence, "ambiguity": amb, "distractor": distractor, "turn": turn}
    return (cov, entity, status) if return_status else (cov, entity)


class RegimeGuard:
    def __init__(self, results_dir, gen, forecast_threshold=None, alarm_threshold=None, thin=20, h=2):
        d = Path(results_dir)
        rep = read_json(d / "report.json")
        sc = rep["scaler"]
        self.mu_x, self.sd_x = np.array(sc["mean"]), np.array(sc["scale"])
        self.mu_c, self.sd_c = np.array(sc["cov_mean"]), np.array(sc["cov_scale"])
        self.cov_order = sc["cov_order"]
        self.transform = sc.get("transform", "none")
        assert tuple(self.cov_order) == PRIMARY_COVARIATES
        bm = BayesHMM.load(d / "posterior_M4.npz")
        self.draws = bm.flat(thin)
        self.S = len(self.draws["pi"])
        self.ems = [GMMEmissions(self.draws["w"][s], self.draws["mu"][s], self.draws["var"][s]) for s in range(self.S)]
        self.h = h
        self.alarm_threshold = alarm_threshold if alarm_threshold is not None else \
            rep["alarms"]["M4_cov_bayes_filter"]["threshold"]
        self.forecast_threshold = forecast_threshold
        self.gen = gen

    def _A(self, cov):
        c = (np.array([cov[k] for k in self.cov_order], float) - self.mu_c) / self.sd_c
        return np.stack([np.exp(cov_logA(self.draws["theta"][s], c[None, None]))[0, 0] for s in range(self.S)])

    def replay_chain(self, turns):
        """Recompute prior / transition / likelihood / posterior for a saved run from its saved
        covariates and features (identical math to GuardSession, no model calls)."""
        f, out = None, []
        for tr in turns:
            A = self._A(tr["covariates"])
            prior = self.draws["pi"] if f is None else np.einsum("sk,skj->sj", f, A)
            prev = self.draws["pi"] if f is None else f
            x = np.array([tr["features"][k] for k in ("logprob", "entropy", "lexcos")], float)
            ok = bool(tr.get("eligible", True)) and np.all(np.isfinite(x))
            xs = (transform_features(x, self.transform) - self.mu_x) / self.sd_x
            lb = np.stack([em.log_b(np.nan_to_num(xs)[None, None], np.array([[ok]]))[0, 0] for em in self.ems])
            f = softmax_log(np.log(np.maximum(prior, 1e-300)) + lb, axis=-1)
            out.append({"chain": {"A": A.mean(0).tolist(), "prev_belief": prev.mean(0).tolist(),
                                  "prior": prior.mean(0).tolist(), "first_turn": len(out) == 0},
                        "chain_post": {"likelihood": softmax_log(lb, axis=-1).mean(0).tolist(),
                                       "posterior": f.mean(0).tolist(), "eligible": ok}})
        return out

    def session(self, system, rows, domain="clinical", guard=True, reground=True, coref="v1"):
        return GuardSession(self, system, rows, domain, guard, reground, coref)

    def run(self, system, rows, questions, seeds, domain="clinical", guard=True):
        sess = self.session(system, rows, domain, guard)
        log = []
        for q, seed in zip(questions, seeds):
            pre = sess.pre(q)
            post = sess.answer(seed)
            log.append({**pre, **post})
        return log


class GuardSession:
    """Step-wise guard for one conversation: pre(question) -> answer(seed).

    pre() uses only visible information (chart rows + question text) and
    returns covariates, the posterior-averaged forecast and the re-ground
    decision; answer() calls the LLM, extracts features, updates the causal
    filter and returns the hold decision.
    """

    def __init__(self, guard: RegimeGuard, system, rows, domain="clinical", guard_on=True, reground_on=True,
                 coref="v1"):
        self.g, self.rows, self.domain, self.guard_on = guard, rows, domain, guard_on
        self.reground_on = reground_on
        self.coref = coref
        self.msgs = [{"role": "system", "content": system}]
        self.f = None                   # (S,K) per-draw filtered distribution
        self.prev_entity = None
        self.t = 0
        self._pending = None

    def pre(self, question):
        g = self.g
        cov, ent, ref_status = live_covariates(self.rows, question, self.prev_entity, self.t + 1, self.domain,
                                               return_status=True, coref=self.coref)
        A = g._A(cov)
        prior = g.draws["pi"] if self.f is None else np.einsum("sk,skj->sj", self.f, A)
        fore = float(prior[:, g.h].mean())
        reground = bool(self.guard_on and self.reground_on and self.t > 0 and g.forecast_threshold is not None
                        and fore > g.forecast_threshold)
        q_sent = (REGROUND + question) if reground else question
        self._pending = (prior, ent, q_sent)
        prev_belief = g.draws["pi"] if self.f is None else self.f
        chain = {"A": A.mean(0).tolist(), "prev_belief": prev_belief.mean(0).tolist(), "prior": prior.mean(0).tolist(),
                 "first_turn": self.f is None}
        return {"turn": self.t + 1, "question": question, "sent": q_sent, "covariates": cov, "entity": ent,
                "chain": chain,
                "reference": ref_status, "coref_grammar": self.coref,
                "forecast_risk_before": fore, "forecast_q05": float(np.quantile(prior[:, g.h], .05)),
                "forecast_q95": float(np.quantile(prior[:, g.h], .95)), "reground": reground}

    def answer(self, seed):
        g = self.g
        prior, ent, q_sent = self._pending
        self.msgs.append({"role": "user", "content": q_sent})
        out = g.gen.chat(self.msgs, seed)
        self.msgs.append({"role": "assistant", "content": out["text"]})
        x, ok = turn_features({"answer": out["text"], "tokens": out["tokens"]}, self.rows)
        xs = (transform_features(x, g.transform) - g.mu_x) / g.sd_x
        lb = np.stack([em.log_b(xs[None, None], np.array([[ok]]))[0, 0] for em in g.ems])
        self.f = softmax_log(np.log(np.maximum(prior, 1e-300)) + lb, axis=-1)
        risk = float(self.f[:, g.h].mean())
        hold = bool(self.guard_on and risk > g.alarm_threshold)
        self.prev_entity = ent
        self.t += 1
        self._pending = None
        state = self.f.mean(0)
        lik = softmax_log(lb, axis=-1).mean(0)                      # normalised emission likelihood b_j(x_t)
        return {"answer": out["text"], "risk_after": risk,
                "chain_post": {"likelihood": lik.tolist(), "posterior": state.tolist(), "eligible": bool(ok)},
                "risk_q05": float(np.quantile(self.f[:, g.h], .05)),
                "risk_q95": float(np.quantile(self.f[:, g.h], .95)), "hold": hold,
                "state_probs": {"G": float(state[0]), "U": float(state[1]), "H": float(state[2])},
                "features": dict(zip(("logprob", "entropy", "lexcos"), map(float, x))), "eligible": ok,
                "context_tokens": sum(approx_tokens(m["content"]) for m in self.msgs)}
