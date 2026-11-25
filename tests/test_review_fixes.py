"""Regression tests for review findings (each failed on the pre-fix code).

Each test failed on the pre-fix code.
"""
import json
from pathlib import Path

import numpy as np

from regime_lab.data.dossiers import make_dossier
from regime_lab.data.grader import grade_answer
from regime_lab.evaluation import metrics as mt
from regime_lab.features.extract import token_stats
from regime_lab.models.bayes import BayesHMM
from regime_lab.usecase.guard import live_covariates

ROOT = Path(__file__).resolve().parents[2]


def test_F2_variance_draws_independent_across_features():
    """The NIG conditional draws var_d independently for each feature d."""
    # Tiny data set with K*M = 6 components: many components are empty in many sweeps, so their
    # variances are prior draws b0 / Gamma(a0). Under the bug all D features share ONE gamma
    # variate, so var[..., 0] == var[..., 1] == var[..., 2] exactly for every empty component.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 2, 3)); mask = np.ones((3, 2), bool)
    b = BayesHMM(K=3, M=2).fit(X, mask, chains=1, warmup=5, n_draws=200, seed=3)
    v = b.draws["var"][0]                                   # (S, K, M, D)
    identical = np.isclose(v[..., 0], v[..., 1], rtol=0, atol=0) & np.isclose(v[..., 1], v[..., 2], rtol=0, atol=0)
    assert identical.mean() < 0.01, f"{identical.mean():.1%} of component draws have identical variances across features"


def test_F4_single_class_bootstrap_does_not_crash():
    sel = np.ones((5, 2), bool); y = np.zeros((5, 2), int)
    out = mt.paired_bootstrap(sel, y, np.random.rand(5, 2), np.random.rand(5, 2), reps=20)
    assert out["valid_reps"] == 0 and out["lo"] is None


def test_F4_no_clean_dossiers_threshold_unavailable():
    thr = mt.select_threshold(np.array([[.1, .9]]), np.ones((1, 2), bool), np.array([["H", "G"]], object))
    assert thr is None or thr >= 1.0


def test_F4_pending_dossier_not_counted_clean():
    out = mt.alarm_outcomes(np.array([[.9, .9]]), np.ones((1, 2), bool), np.array([["P", "G"]], object), .5)
    assert out["clean_false_stop"] == 0 and out.get("unresolved", 0) == 1


def test_F6_negated_or_contradictory_answers_pending():
    codes = {"0:0": "AB1-23", "1:0": "CD2-34"}
    sup = {"answer": "AB1-23", "hidden_code": None}
    miss = {"answer": "UNKNOWN", "hidden_code": "EF3-45"}
    assert grade_answer("The code is not AB1-23. UNKNOWN", sup, codes, version="2")["label"] == "P"
    assert grade_answer("UNKNOWN; do not use CD2-34.", miss, codes, version="2")["label"] == "P"
    assert grade_answer("AB1-23\n\nUNKNOWN", sup, codes, version="2")["label"] == "P"
    assert grade_answer("AB1-23\n\nUNKNOWN", sup, codes)["label"] == "G"          # locked v1 default unchanged
    assert grade_answer("AB1-23", sup, codes)["label"] == "G"          # unchanged behaviour


def test_F7_intervention_pairs_at_replicate_level():
    from regime_lab.evaluation.intervention import analyse
    rows = [{"population": "after_H", "dossier_id": 1, "prefix_turn": 2, "rep": 0, "arm": "original", "label": "H"},
            {"population": "after_H", "dossier_id": 1, "prefix_turn": 2, "rep": 0, "arm": "corrected", "label": "P"},
            {"population": "after_H", "dossier_id": 1, "prefix_turn": 2, "rep": 1, "arm": "original", "label": "G"},
            {"population": "after_H", "dossier_id": 1, "prefix_turn": 2, "rep": 1, "arm": "corrected", "label": "G"},
            {"population": "after_H", "dossier_id": 1, "prefix_turn": 2, "rep": 0, "arm": "reground", "label": "G"},
            {"population": "after_H", "dossier_id": 1, "prefix_turn": 2, "rep": 1, "arm": "reground", "label": "G"}]
    c = analyse(rows, boot=50)["after_H"]["contrasts"]["corrected - original"]
    assert c["risk_difference_H"] == 0.0


def test_F8_partial_topk_coverage_ineligible():
    l, e = token_stats([{"token": "a", "logprob": -.1, "top": [-.1, -2.4]}, {"token": "b", "logprob": -2.0, "top": []}])
    assert not np.isfinite(e)


def test_W9_person_pronouns_resolve_previous_entity():
    v, _ = make_dossier(20003, "clinical")
    cov, ent = live_covariates(v["rows"], "Give me the medication code for that person.", ("Korvin Greaves", 1962), 2, coref="v2")
    assert ent == ("Korvin Greaves", 1962)


def test_grader_battery_complete_and_passing():
    """Guards against silently dropping adversarial grader cases (24 cases as of grader v2)."""
    from regime_lab.evaluation.adversarial import grader_battery
    rows = grader_battery()
    assert len(rows) >= 26 and all(r["pass"] for r in rows), [r for r in rows if not r["pass"]]


def test_sampler_fix_prior_only_moments_and_independence():
    """[sampler fix] With no observed emissions every variance draw is an independent
    InvGamma(a0, b0) per feature: mean b0/(a0-1) = 1 and ~zero cross-feature correlation."""
    for M in (1, 2):
        from regime_lab.models.emissions import GMMEmissions
        from regime_lab.models.hmm import EMHMM
        X = np.zeros((4, 3, 3)); mask = np.zeros((4, 3), bool)            # nothing observed
        init = EMHMM(3, M, np.full(3, 1 / 3), np.full((3, 3), 1 / 3),
                     GMMEmissions(np.full((3, M), 1 / M), np.zeros((3, M, 3)), np.ones((3, M, 3))))
        b = BayesHMM(K=3, M=M).fit(X, mask, init=init, chains=1, warmup=10, n_draws=3000, seed=11)
        v = b.draws["var"][0].reshape(3000, -1, 3)                            # (S, K*M, D)
        assert abs(v.mean() - 1.0) < 0.08, v.mean()                           # b0/(a0-1) = 2/2
        lv = np.log(v)
        r = max(abs(np.corrcoef(lv[:, c, 0], lv[:, c, 1])[0, 1]) for c in range(lv.shape[1]))
        assert r < 0.1, f"M={M}: max cross-feature log-variance correlation {r:.3f}"


def test_W10_exact_within_dossier_matches_brute_force():
    """[W10 v2] exact convolution == full enumeration of every within-dossier ordering (small case)."""
    import itertools
    from regime_lab.evaluation.persistence import within_dossier_exact
    L = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]])
    orders = [sorted(set(itertools.permutations(r))) for r in L]
    stats = [sum(np.sum(np.array(a)[1:] * np.array(a)[:-1]) for a in combo) for combo in itertools.product(*orders)]
    obs = int(np.sum(L[:, 1:] * L[:, :-1]))
    res = within_dossier_exact(L)
    assert res["HH_observed"] == obs
    assert np.isclose(res["p_value_upper"], np.mean(np.array(stats) >= obs))
    assert np.isclose(res["null_mean"], np.mean(stats))


def test_F6_v21_sentence_boundaries():
    """[F6 v2.1] boundaries are literal . ! ? ; newline (no whitespace needed); negation never crosses one."""
    codes = {"0:0": "AB1-23", "1:0": "CD2-34"}
    sup = {"answer": "AB1-23", "hidden_code": None}
    r1 = grade_answer("AB1-23.The patient has diabetes.", sup, codes, version="2")
    assert r1 == {"label": "P", "reason": "extra_claim"}
    r2 = grade_answer("Do not answer? AB1-23", sup, codes, version="2")
    assert r2["reason"] != "negated_code"
    assert grade_answer("AB1-23.The patient has diabetes.", sup, codes)["label"] == "G"   # locked v1 unchanged


def test_W9_v2_precedence_and_status():
    """[W9 v2] surname(+year) outranks a pronoun; unresolved is reported; v1 behaviour unchanged."""
    v, _ = make_dossier(20003, "clinical")
    q = "What is Greaves's 1962 medication code and their allergy code?"
    _, ent, st = live_covariates(v["rows"], q, ("Cyrene Ashdown", 1945), 2, return_status=True, coref="v2")
    assert ent == ("Korvin Greaves", 1962) and st == "surname_year"
    _, ent, st = live_covariates(v["rows"], "What is their ward code?", None, 1, return_status=True, coref="v2")
    assert ent is None and st == "unresolved"
    _, ent, st = live_covariates(v["rows"], "What is that patient's ward code?", ("Korvin Greaves", 1962), 2,
                                 return_status=True)
    assert ent == ("Korvin Greaves", 1962) and st == "coreference_previous_turn"
