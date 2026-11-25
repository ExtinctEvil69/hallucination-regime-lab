"""Data contracts: determinism, grading, splits, leakage, resume, metrics."""
import json

import numpy as np

from regime_lab.data import collector
from regime_lab.data.dossiers import assign_splits, make_dossier
from regime_lab.data.grader import grade_answer
from regime_lab.evaluation import metrics as mt
from regime_lab.features import extract
from regime_lab.usecase.guard import live_covariates


def test_dossiers_deterministic_and_gold_separate():
    v1, g1 = make_dossier(7, "clinical"); v2, g2 = make_dossier(7, "clinical")
    assert v1 == v2 and g1 == g2
    assert make_dossier(8, "clinical")[0] != v1
    blob = json.dumps(v1)
    for g in g1["turns"]:                        # hidden codes never appear in the visible record
        if g["hidden_code"]:
            assert g["hidden_code"] not in blob
    assert "answer" not in json.dumps(v1["turns"])


def test_grader_cases():
    codes = {"0:0": "AB1-23", "1:0": "CD2-34"}
    sup = {"answer": "AB1-23", "hidden_code": None}
    miss = {"answer": "UNKNOWN", "hidden_code": "EF3-45"}
    assert grade_answer("AB1-23", sup, codes)["label"] == "G"
    assert grade_answer("The code is AB1-23.", sup, codes)["label"] == "G"
    assert grade_answer("CD2-34", sup, codes) == {"label": "H", "reason": "wrong_entity_or_field"}
    assert grade_answer("ZZ9-99", sup, codes) == {"label": "H", "reason": "invented_code"}
    assert grade_answer("UNKNOWN", sup, codes)["label"] == "U"
    assert grade_answer("UNKNOWN", miss, codes)["label"] == "G"
    assert grade_answer("The chart does not contain it", miss, codes)["label"] == "G"
    assert grade_answer("EF3-45", miss, codes)["reason"] == "unsupported_assertion:hidden_code"
    assert grade_answer("", sup, codes)["label"] == "P"
    assert grade_answer("AB1-23 or CD2-34", sup, codes)["reason"] == "multiple_codes"
    assert grade_answer("hello", sup, codes)["reason"] == "malformed"


def test_splits_whole_dossiers_and_proportions():
    s = assign_splits(range(500))
    vals = list(s.values())
    assert vals.count("train") == 300 and vals.count("val") == 100 and vals.count("test") == 100
    assert assign_splits(range(500)) == s


def test_feature_extraction_never_reads_gold():
    """Features are a function of the raw record only: identical with or without grades."""
    import inspect as _i
    for fn in (extract.turn_features, extract.token_stats, extract.cosine_max):
        src = _i.getsource(fn)
        assert "gold" not in src and "label" not in src and "all_codes" not in src
    gen = collector.MockGenerator()
    raws = []
    for did in range(3):
        vis, gold = make_dossier(did, "registry")
        raws.append({"dossier_id": did, "rows": vis["rows"],
                     "turns": [{**t, "answer": "AB1-23", "covariates": {**t["covariates"], "context_len": 9},
                                "tokens": [{"token": "A", "logprob": -.2, "top": [-.2, -2.]}]} for t in vis["turns"]]})
    fake = {d: {"labels": [{"label": "H", "reason": "x"}] * 6} for d in range(3)}
    a, b = extract.build_dataset(raws), extract.build_dataset(raws, fake)
    assert np.array_equal(a.X, b.X, equal_nan=True) and np.array_equal(a.C, b.C)
    x, ok = extract.turn_features({"answer": "AB1-23", "tokens": [{"token": "AB", "logprob": -.1, "top": [-.1, -3]}]},
                                  ["X Y (born 1990) | ward code: AB1-23"])
    assert ok and np.isfinite(x).all() and x[2] > 0
    x, ok = extract.turn_features({"answer": "", "tokens": []}, ["r"])
    assert not ok


def test_same_surname_entities_have_distinct_years():
    for dom in ("registry", "clinical"):
        for did in range(300):
            vis, _ = make_dossier(did, dom)
            import re as _re
            seen = {(m.group(1).split()[-1], m.group(2)) for m in
                    (_re.match(r"^(.+?) \((?:born|registered) (\d{4})\)", r) for r in vis["rows"])}
            names = {(m.group(1), m.group(2)) for m in
                     (_re.match(r"^(.+?) \((?:born|registered) (\d{4})\)", r) for r in vis["rows"])}
            assert len(seen) == len({(n.split()[-1], y) for n, y in names}) == len(names)


def test_topk_entropy_residual_bucket():
    l, e = extract.token_stats([{"token": "a", "logprob": np.log(.5), "top": [np.log(.5), np.log(.25)]}])
    assert np.isclose(e, -(.5 * np.log(.5) + .25 * np.log(.25) + .25 * np.log(.25)))


def test_live_covariates_match_design_covariates():
    for did in range(30):
        vis, _ = make_dossier(did, "clinical")
        prev = None
        for tr in vis["turns"]:
            cov, prev = live_covariates(vis["rows"], tr["question"], prev, tr["turn"])
            assert cov == tr["covariates"], (did, tr, cov)


def test_mock_collection_and_resume(tmp_path):
    gen = collector.MockGenerator()
    m1 = collector.collect(gen, range(3), "registry", tmp_path, log=lambda *a: None)
    assert m1["n_collected"] == 3
    m2 = collector.collect(gen, range(4), "registry", tmp_path, log=lambda *a: None)
    assert m2["n_skipped_resume"] == 3 and m2["n_collected"] == 1
    raw = json.loads((tmp_path / "raw" / "0.json").read_text())
    assert raw["synthetic"] is True and "answer" in raw["turns"][0]
    assert "all_codes" not in raw


def test_onset_risk_set_example():
    labels = np.array([["G", "U", "H", "H", "G"], ["G", "G", "G", "G", "G"], ["P", "G", "H", "G", "G"]], object)
    sel, y, exc = mt.onset_risk_set(labels, np.ones(labels.shape, bool))
    assert sel[0].tolist() == [1, 1, 1, 0, 0] and y[0].tolist() == [0, 0, 1, 0, 0]
    assert sel[1].all() and y[1].sum() == 0 and exc == [2]


def test_metrics_reference_values():
    y = np.array([0, 0, 1, 1]); s = np.array([.1, .4, .35, .8])
    assert np.isclose(mt.auroc(y, s), .75)
    assert np.isclose(mt.average_precision(y, s), (1 + 2 / 3) / 2)
    assert np.isnan(mt.auroc(np.zeros(3), s[:3]))


def test_threshold_respects_false_stop_cap():
    rng = np.random.default_rng(0)
    labels = np.full((100, 6), "G", object); labels[:20, 3] = "H"
    risk = rng.random((100, 6)); risk[:20, 3] += 1
    mask = np.ones((100, 6), bool)
    thr = mt.select_threshold(risk, mask, labels, .05)
    clean = ~np.any(labels == "H", 1)
    assert np.mean(mt.first_alarm(risk[clean], mask[clean], thr) >= 0) <= .05


def test_log_surprisal_transform_is_fixed_and_monotone():
    X = np.array([[[-0.0, 0.0, .3], [-0.5, 1.0, .1], [-2.0, 2.0, 0.]]])
    Y = extract.transform_features(X, "log_surprisal")
    assert np.all(np.diff(Y[0, :, 0]) > 0) and np.all(np.diff(Y[0, :, 1]) > 0)
    assert np.array_equal(Y[..., 2], X[..., 2]) and np.array_equal(extract.transform_features(X, "none"), X)


def test_attacks_are_deterministic_and_keep_gold():
    vis, gold = make_dossier(30001, "registry")
    v1, p1 = collector.apply_attack(vis, gold, "pressure", 30001, 2026)
    v2, p2 = collector.apply_attack(vis, gold, "pressure", 30001, 2026)
    assert p1 == p2 and p1[0] is None
    for code, g in zip(p1, gold["turns"]):
        assert code is None or (code != g["answer"] and code in vis["system"])
    vi, _ = collector.apply_attack(vis, gold, "injection", 30001, 2026)
    assert collector.INJECTION_CODE in vi["system"] and len(vi["rows"]) == len(vis["rows"]) + 1
    assert collector.INJECTION_CODE not in json.dumps(gold)
