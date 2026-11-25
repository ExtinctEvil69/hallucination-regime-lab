"""End-to-end study: fit on train, freeze mapping/thresholds on validation, open test once."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from ..features.extract import transform_features
from ..inference.attribution import filter_weighted_contrast, shapley, transition_contrasts
from ..models import diagnostics as dg
from ..models.bayes import BayesHMM, Prior, cov_logA
from ..models.hmm import EMHMM, map_states
from ..util import atomic_write_json
from . import metrics as mt

FEATURE_NAMES = ("logprob", "entropy", "lexcos")


def _std_cov(C, train):
    flat = C[train].reshape(-1, C.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (C - mu) / sd, mu, sd


def _platt(score, y):
    """1-D logistic calibration (used only to give M0 a probability scale)."""
    b = np.zeros(2); Xd = np.c_[np.ones_like(score), score]
    for _ in range(100):
        p = 1 / (1 + np.exp(-Xd @ b)); W = p * (1 - p) + 1e-9
        b += np.linalg.solve(Xd.T @ (Xd * W[:, None]) + 1e-6 * np.eye(2), Xd.T @ (y - p))
    return b


def chain_discrepancy(loglik, gap=50.0):
    """[F1-diag v1] per-chain mean log-likelihood and a HEURISTIC discrepancy flag (> gap nats below the best
    chain). Not evidence of mode membership or posterior mass; no chain is ever dropped; jitter stays 0.3."""
    m = np.asarray(loglik, float).mean(1)
    return {"chain_mean_loglik": m.tolist(), "heuristic_discrepancy_gap_nats": gap,
            "heuristic_discrepancy_chains": [int(i) for i in np.flatnonzero(m < m.max() - gap)],
            "heuristic_note": "likelihood-gap flag is a heuristic discrepancy, not mode membership; no chain dropped"}


def m4_matched_contrast(bm, Xs, mask, Cs, prof, k, lo, hi, h, thin=40):
    """[F9] Draw-matched filter-weighted contrast: for each posterior draw s, weight
    A^s_iH(hi) - A^s_iH(lo) by that same draw's causal filter f^s_{t-1} (not the M2 filter)."""
    from ..models.emissions import GMMEmissions
    from ..models.hmm import filter_probs
    d = bm.flat(thin)
    lo_c, hi_c = prof.copy(), prof.copy(); lo_c[:, k], hi_c[:, k] = lo, hi
    vals = []
    for s in range(len(d["pi"])):
        em_s = GMMEmissions(d["w"][s], d["mu"][s], d["var"][s])
        f = filter_probs(em_s.log_b(Xs, mask), np.log(d["pi"][s]), cov_logA(d["theta"][s], Cs))[0]
        fp = f[:, :-1][mask[:, 1:]]
        Ah = np.exp(cov_logA(d["theta"][s], hi_c[None]))[0][:, :, h]
        Al = np.exp(cov_logA(d["theta"][s], lo_c[None]))[0][:, :, h]
        vals.append(np.mean(np.sum(fp * (Ah - Al), axis=1)))
    vals = np.array(vals)
    return {"median": float(np.median(vals)), "lo": float(np.quantile(vals, .025)),
            "hi": float(np.quantile(vals, .975)), "p_gt0": float(np.mean(vals > 0)), "draws": int(len(vals))}


def persistence_test(labels, strata, reps=2000, seed=0):
    """H2: observed H->H transitions vs within-stratum permutation of turn labels across dossiers."""
    rng = np.random.default_rng(seed)
    valid = np.all(np.isin(labels, ["G", "U", "H"]), axis=1)
    L = (labels[valid] == "H").astype(int); S = strata[valid]
    def stat(M):
        return int(np.sum(M[:, 1:] * M[:, :-1]))
    obs = stat(L)
    keys = np.unique(S)
    null = []
    for _ in range(reps):
        Mp = L.copy()
        for k in keys:
            idx = np.nonzero(S == k)
            vals = Mp[idx]; Mp[idx] = rng.permutation(vals)
        null.append(stat(Mp))
    null = np.array(null)
    prev = L[:, :-1].ravel(); nxt = L[:, 1:].ravel()
    return {"n_dossiers": int(valid.sum()), "HH_observed": obs, "HH_null_mean": float(null.mean()),
            "HH_null_95": float(np.quantile(null, .95)), "p_value": float((1 + np.sum(null >= obs)) / (1 + reps)),
            "P(H_t|H_t-1)": float(nxt[prev == 1].mean()) if (prev == 1).any() else None,
            "P(H_t|notH_t-1)": float(nxt[prev == 0].mean()), "n_H_prev": int((prev == 1).sum())}


def ppc_lag1(bayes: BayesHMM, X, mask, C=None, reps=200, seed=0, dim=0):
    """Permutation-invariant PPC: within-dossier lag-1 autocorrelation of one feature."""
    rng = np.random.default_rng(seed)
    def lag1(Z, m):
        a, b = Z[:, :-1, dim], Z[:, 1:, dim]; ok = m[:, :-1] & m[:, 1:]
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
    obs = lag1(X, mask)
    d = bayes.flat()
    N, T, D = X.shape
    reps_v = []
    for r in range(reps):
        s = rng.integers(len(d["pi"]))
        logA = cov_logA(d["theta"][s], C) if bayes.covariates else np.log(d["A"][s])
        z = np.zeros((N, T), int)
        z[:, 0] = [rng.choice(3, p=d["pi"][s]) for _ in range(N)]
        for t in range(1, T):
            At = np.exp(logA if logA.ndim == 2 else logA[:, t])
            P = At[z[:, t - 1]] if At.ndim == 2 else At[np.arange(N), z[:, t - 1]]
            c = np.cumsum(P, 1); z[:, t] = np.minimum((c < rng.random(N)[:, None]).sum(1), 2)
        m = np.array([[rng.choice(d["w"].shape[-1], p=d["w"][s][z[n, t]]) for t in range(T)] for n in range(N)])
        mu, var = d["mu"][s][z, m], d["var"][s][z, m]
        reps_v.append(lag1(rng.normal(mu, np.sqrt(var)), mask))
    reps_v = np.array(reps_v)
    return {"feature": FEATURE_NAMES[dim], "observed": obs, "rep_mean": float(reps_v.mean()),
            "rep_lo": float(np.quantile(reps_v, .025)), "rep_hi": float(np.quantile(reps_v, .975)),
            "ppp": float(np.mean(reps_v >= obs))}


def run_study(X, mask, C_raw, labels, split, out_dir, cov_names, *, synthetic, qtype=None,
              audit_text=None, chains=4, warmup=1000, n_draws=1000, thin=20, seed=0, boot=2000,
              kappa_sens=(2.0, 5.0), log=print, cost_ratio=None, transform="none", M=2):
    t_start = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tr, va, te = split == "train", split == "val", split == "test"
    rep = {"synthetic": bool(synthetic), "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
           "eligible_turns": {k: int(mask[s].sum()) for k, s in [("train", tr), ("val", va), ("test", te)]},
           "label_counts": {k: {l: int(np.sum(labels[s] == l)) for l in "GUHP"} for k, s in
                            [("train", tr), ("val", va), ("test", te)]}}
    # ---------- features: fixed transform, then training-only scaler
    rep["feature_transform"], rep["mixture_components"] = transform, M
    X_raw = X
    X = transform_features(X, transform)
    obs = X[tr][mask[tr]]
    mu_x, sd_x = obs.mean(0), obs.std(0); sd_x = np.where(sd_x < 1e-12, 1, sd_x)
    Xs = (X - mu_x) / sd_x
    Cs, mu_c, sd_c = _std_cov(C_raw, tr)
    rep["scaler"] = {"feature_order": FEATURE_NAMES, "transform": transform, "mean": mu_x, "scale": sd_x,
                     "cov_order": cov_names, "cov_mean": mu_c, "cov_scale": sd_c}
    # ---------- M2: EM-HMM
    log("fitting EM-HMM (3 starts)")
    em = EMHMM(3, M).fit(Xs[tr], mask[tr], n_starts=3, seed=seed)
    g_val, _, _ = em.smooth(Xs[va], mask[va])
    mp = map_states(g_val, labels[va], mask[va])
    order = [mp["perm"].index(i) for i in range(3)]             # new index = label index
    em = em.permute(order)
    H = 2
    rep["state_mapping"] = {k: mp[k] for k in ("perm", "score", "runner_up", "classes_present", "stable")}
    rep["em"] = {"selected_start": em.history and getattr(em, "selected_start", None),
                 "final_loglik_by_start": [h["loglik"] for h in em.history if h.get("final")],
                 "pi": em.pi, "A": em.A, "occupancy": em.occupancy, "state_means_std": em.em.state_means()}
    em.save(out / "model.npz")
    f2, q2, _ = em.filter(Xs, mask)
    s1 = em.static_scores(Xs, mask)
    nc = em.negative_control().filter(Xs, mask)[0]
    rep["negative_control_max_abs_diff"] = float(np.max(np.abs(nc - s1)))
    # ---------- M0: single feature (chosen on validation)
    vsel, vy, _ = mt.onset_risk_set(labels[va], mask[va])
    best = None
    for d in range(3):
        for sg in (1, -1):
            a = mt.auroc(vy[vsel], sg * Xs[va][..., d][vsel])
            if best is None or a > best[0]:
                best = (a, d, sg)
    tsel_tr, ty_tr, _ = mt.onset_risk_set(labels[tr], mask[tr])
    m0raw = best[2] * Xs[..., best[1]]
    b = _platt(m0raw[tr][tsel_tr], ty_tr[tsel_tr].astype(float))
    m0 = 1 / (1 + np.exp(-(b[0] + b[1] * np.nan_to_num(m0raw))))
    rep["M0"] = {"feature": FEATURE_NAMES[best[1]], "sign": best[2], "val_auroc": best[0]}
    # ---------- M3 / M4: Bayesian
    bayes = {}
    for name, cov, kap in [("M3", False, 0.0), ("M4", True, 0.0)]:
        log(f"Gibbs {name} ({'covariate' if cov else 'homogeneous'}) {chains}x({warmup}+{n_draws})")
        t0 = time.time()
        bm = BayesHMM(3, M, covariates=cov, prior=Prior(kappa=kap)).fit(
            Xs[tr], mask[tr], Cs[tr] if cov else None, init=em, chains=chains, warmup=warmup,
            n_draws=n_draws, seed=seed + (7 if cov else 3))
        bm.align(em.em)
        # semantic mapping of the Bayesian ensemble's own states, on VALIDATION labels only
        mpb = map_states(bm.filter_states(Xs[va], mask[va], Cs[va] if cov else None, thin=thin),
                         labels[va], mask[va])
        remap = [mpb["perm"].index(i) for i in range(3)]
        if remap != [0, 1, 2]:
            bm.permute_all(remap)
        bm.diagnostics["validation_remap"] = {"perm": remap, "score": mpb["score"], "stable": mpb["stable"],
                                              "runner_up": mpb["runner_up"]}
        bm.diagnostics["seconds"] = time.time() - t0
        bayes[name] = bm
        bm.save(out / f"posterior_{name}.npz")
    pred3 = bayes["M3"].predict(Xs, mask, thin=thin, h=H, reweight=True)
    pred4 = bayes["M4"].predict(Xs, mask, C=Cs, thin=thin, h=H, reweight=True)
    scores = {"M0_single_feature": m0, "M1_static": s1[..., H], "M2_em_filter": f2[..., H],
              "M3_bayes_filter": pred3["mean"], "M3_bayes_reweighted": pred3["mean_rw"],
              "M4_cov_bayes_filter": pred4["mean"]}
    # Bayesian matched static for M3 (same posterior emissions, occupancy prior)
    # ---------- primary + secondary detection endpoints on TEST
    sel, y, excl = mt.onset_risk_set(labels[te], mask[te])
    rep["onset_test"] = {"risk_set_turns": int(sel.sum()), "onsets": int(y[sel].sum()),
                         "excluded_dossiers": len(excl)}
    rep["detection_onset"] = {}
    for k, s in scores.items():
        r = mt.prob_metrics(y[sel], s[te][sel])
        r["auroc_ci"] = mt.bootstrap_ci(sel, y, s[te], reps=boot // 2, seed=seed)
        rep["detection_onset"][k] = r
    pairs = [("M2_em_filter", "M1_static"), ("M3_bayes_filter", "M1_static"),
             ("M4_cov_bayes_filter", "M1_static"), ("M4_cov_bayes_filter", "M2_em_filter"),
             ("M2_em_filter", "M0_single_feature")]
    rep["paired_onset_auroc"] = {f"{a} - {b}": mt.paired_bootstrap(sel, y, scores[a][te], scores[b][te],
                                                                   reps=boot, seed=seed) for a, b in pairs}
    allsel = mask[te] & np.isin(labels[te], ["G", "U", "H"])
    ally = (labels[te] == "H").astype(int)
    rep["detection_all_turns"] = {k: mt.prob_metrics(ally[allsel], s[te][allsel]) for k, s in scores.items()}
    # ---------- next-turn forecast (H4)
    fsel, fy = mt.next_turn_set(labels[te], mask[te])
    nxt2 = np.full(f2.shape[:2], np.nan)
    nxt2[:, :-1] = (f2[:, :-1] @ em.A)[..., H]
    tr_fsel, tr_fy = mt.next_turn_set(labels[tr], mask[tr])
    base = np.full(nxt2.shape, tr_fy[tr_fsel].mean())
    fc = {"base_rate": base, "M2_plugin": nxt2, "M3_frozen_ensemble": pred3["next_mean"],
          "M3_reweighted": pred3["next_rw"], "M4_covariate_ensemble": pred4["next_mean"]}
    rep["forecast_next_turn"] = {k: mt.prob_metrics(fy[fsel], v[te][fsel]) for k, v in fc.items()}
    rep["forecast_paired_brier"] = {
        f"{a} - {b}": mt.paired_bootstrap(fsel, fy, fc[a][te], fc[b][te], metric=lambda yy, pp: -mt.brier(yy, pp),
                                          reps=boot, seed=seed)
        for a, b in [("M3_frozen_ensemble", "M2_plugin"), ("M3_reweighted", "M2_plugin"),
                     ("M4_covariate_ensemble", "M2_plugin"), ("M4_covariate_ensemble", "base_rate")]}
    rep["forecast_paired_brier_note"] = "metric is NEGATIVE Brier: positive difference = first model better"
    rep["reweight_ess_test"] = {"median": float(np.median(pred3["ess_rw"][te])),
                                "min": float(np.min(pred3["ess_rw"][te])), "draws": int(pred3["filt"].shape[0])}
    # ---------- H2 persistence
    strata = np.array([[f"{t}|{qtype[n, t] if qtype is not None else ''}" for t in range(X.shape[1])]
                       for n in range(X.shape[0])], object)
    rep["persistence_H2"] = persistence_test(labels, strata, reps=boot, seed=seed)
    d3 = bayes["M3"].flat()
    from .persistence import sensitivity as _w10
    rep["persistence_within_dossier"] = _w10(labels, seed=seed)     # [W10 v2] post-hoc sensitivity; locked H2 unchanged
    rep["persistence_H2"]["M3_A_HH"] = dg.summarize("A_HH", bayes["M3"].draws["A"][..., H, H])
    rep["persistence_H2"]["M3_A_GH"] = dg.summarize("A_GH", bayes["M3"].draws["A"][..., 0, H])
    rep["persistence_H2"]["M3_P(A_HH>A_GH)"] = float(np.mean(d3["A"][:, H, H] > d3["A"][:, 0, H]))
    log("posterior predictive checks")
    rep["ppc"] = [ppc_lag1(bayes["M3"], Xs[tr], mask[tr], reps=150, seed=seed, dim=d) for d in range(3)]
    # ---------- H3 factor contrasts (M4)
    d4 = bayes["M4"].flat(thin=max(1, thin // 4))
    prof = Cs[tr][:, 1:][mask[tr][:, 1:]]
    f_prev = f2[te][:, :-1][mask[te][:, 1:]]
    prof_te = Cs[te][:, 1:][mask[te][:, 1:]]
    rep["factor_contrasts"] = {}
    for k, nm in enumerate(cov_names):
        raw = C_raw[tr][..., k]
        uniq = np.unique(raw)
        lo_r, hi_r = (uniq.min(), uniq.max()) if len(uniq) <= 2 else tuple(np.quantile(raw, [.1, .9]))
        lo, hi = (lo_r - mu_c[k]) / sd_c[k], (hi_r - mu_c[k]) / sd_c[k]
        byo, _ = transition_contrasts(d4["theta"], prof, k, lo, hi, H, origins=[0, 2])
        rep["factor_contrasts"][nm] = {"low_raw": float(lo_r), "high_raw": float(hi_r),
                                       "enter_H_from_G": byo[0], "stay_in_H": byo[2],
                                       "filter_weighted_test": filter_weighted_contrast(
                                           d4["theta"][::4], f_prev, prof_te, k, lo, hi, H),
                                       "filter_weighted_test_weights": "M2 EM filter (hybrid; kept for continuity)",
                                       "filter_weighted_test_M4": m4_matched_contrast(
                                           bayes["M4"], Xs[te], mask[te], Cs[te], prof_te, k, lo, hi, H,
                                           thin=max(1, thin * 2))}
    rep["M4_sigma"] = {nm: dg.summarize(f"sigma_{nm}", bayes["M4"].draws["sigma"][..., k])
                       for k, nm in enumerate(cov_names)}
    # ---------- diagnostics
    diag = {}
    for name, bm in bayes.items():
        dr = bm.draws
        keys = {"loglik": dr["loglik"],
                "mu_H_logprob": np.einsum("csm,csm->cs", dr["w"][..., H, :], dr["mu"][..., H, :, 0]),
                "pi_H": dr["pi"][..., H]}
        if bm.covariates:
            # [F1-diag v1] all draws; arithmetic mean of exp(log A) over observed training transitions
            A_avg = np.stack([[np.exp(cov_logA(th, Cs[tr]))[:, 1:].mean((0, 1)) for th in ch] for ch in dr["theta"]])
            keys["A_GH_avg"], keys["A_HH_avg"] = A_avg[..., 0, H], A_avg[..., H, H]
            for k, nm in enumerate(cov_names):
                keys[f"logit_G_to_G_vs_refH_{nm}"] = dr["theta"][..., 0, 0, 1 + k]   # [F9] origin G, dest G, ref dest H
        else:
            keys["A_GH"], keys["A_HH"] = dr["A"][..., 0, H], dr["A"][..., H, H]
        diag[name] = {"params": [dg.summarize(k, v) for k, v in keys.items()],
                      "aligned_fraction_permuted": bm.diagnostics.get("aligned_fraction_permuted"),
                      "validation_remap": bm.diagnostics.get("validation_remap"),
                      "mh_accept": bm.diagnostics.get("mh_accept"), "seconds": bm.diagnostics.get("seconds"),
                      "chains": chains, "warmup": warmup, "draws": n_draws,
                      "autocorr_loglik": dg.autocorr(dr["loglik"][0], 30).tolist(),
                      "trace_loglik": dr["loglik"][:, ::5].tolist(),
                      **chain_discrepancy(dr["loglik"])}
        worst = max((p["rhat"] for p in diag[name]["params"] if p["rhat"] == p["rhat"]), default=np.nan)
        diag[name]["max_rhat"] = worst
        diag[name]["min_ess_bulk"] = min((p["ess_bulk"] for p in diag[name]["params"]
                                          if p["ess_bulk"] == p["ess_bulk"]), default=np.nan)
        diag[name]["converged"] = bool(worst <= 1.01 and diag[name]["min_ess_bulk"] >= 400)
    rep["diagnostics"] = diag
    # ---------- sticky prior sensitivity (H2: prior can manufacture persistence)
    rep["sticky_sensitivity"] = {}
    for kap in kappa_sens:
        bm = BayesHMM(3, M, prior=Prior(kappa=kap)).fit(Xs[tr], mask[tr], init=em, chains=2,
                                                       warmup=max(200, warmup // 4), n_draws=max(200, n_draws // 4),
                                                       seed=seed + int(kap * 10))
        bm.align(em.em)
        dd = bm.flat()
        p = bm.predict(Xs[te], mask[te], thin=thin, h=H)
        rep["sticky_sensitivity"][f"kappa={kap}"] = {
            "A_HH_mean": float(dd["A"][:, H, H].mean()), "A_GH_mean": float(dd["A"][:, 0, H].mean()),
            "onset_auroc": mt.auroc(y[sel], p["mean"][sel])}
    rep["sticky_sensitivity"]["kappa=0"] = {"A_HH_mean": float(d3["A"][:, H, H].mean()),
                                            "A_GH_mean": float(d3["A"][:, 0, H].mean()),
                                            "onset_auroc": rep["detection_onset"]["M3_bayes_filter"]["auroc"]}
    # ---------- alarm policy (validation threshold, max 5% clean false stops)
    rep["alarms"] = {}
    for k in ["M1_static", "M2_em_filter", "M4_cov_bayes_filter"]:
        thr = mt.select_threshold(scores[k][va], mask[va], labels[va])
        rep["alarms"][k] = ({"threshold": None, "status": "unavailable: no clean validation conversations"}
                            if thr is None else
                            {"threshold": thr, "test": mt.alarm_outcomes(scores[k][te], mask[te], labels[te], thr)})
    if cost_ratio is not None:
        thr = 1 / (1 + cost_ratio)
        rep["alarms"]["bayes_decision"] = {"C_FN/C_FP": cost_ratio, "threshold": thr,
                                           "test": mt.alarm_outcomes(scores["M4_cov_bayes_filter"][te], mask[te],
                                                                     labels[te], thr)}
    # ---------- gap audits (test): M2 (primary history comparison) and, separately, M4 [F9]
    te_idx = np.flatnonzero(te)
    hconf = np.median(X_raw[tr][..., 0][mask[tr] & (labels[tr] == "G")]) if np.any(labels[tr] == "G") else np.nan
    for audit_model, fname in (("M2_em_filter", "test_audit.csv"), ("M4_cov_bayes_filter", "test_audit_M4.csv")):
      thr2 = (rep["alarms"].get(audit_model) or {}).get("threshold")
      thr2 = 1.0 if thr2 is None else thr2
      rows = []
      for n in te_idx:
        for t in range(X.shape[1]):
            r2 = scores[audit_model][n, t]
            lab = labels[n, t]
            if not mask[n, t] or not (r2 > thr2 or lab == "H"):
                continue
            sh = shapley(em.em, Xs[n, t], q2[n, t], H)
            row = {"audit_model": audit_model, "row": int(n), "turn": t + 1, "label": lab,
                   "category": ("true_alarm" if lab == "H" and r2 > thr2 else
                                "false_negative" if lab == "H" else "false_alarm"),
                   "risk_M2": round(float(scores["M2_em_filter"][n, t]), 4), "risk_M4": round(float(scores["M4_cov_bayes_filter"][n, t]), 4),
                   "risk_M4_q05": round(float(pred4["q05"][n, t]), 4), "risk_M4_q95": round(float(pred4["q95"][n, t]), 4),
                   "prior_H": round(sh["prior_H"], 4),
                   **{f"shap_{f}": round(v, 4) for f, v in zip(FEATURE_NAMES, sh["phi"])},
                   **{f: round(float(X_raw[n, t, i]), 4) for i, f in enumerate(FEATURE_NAMES)},
                   **{c: float(C_raw[n, t, i]) for i, c in enumerate(cov_names)},
                   "high_confidence_wrong": bool(lab == "H" and X_raw[n, t, 0] >= hconf),
                   "explanation_type": "model-based association (not cognition)"}
            if audit_text is not None:
                row.update(audit_text(n, t))
            rows.append(row)
      if rows:
        with open(out / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
      sfx = "" if audit_model == "M2_em_filter" else "_M4"
      rep["audit_counts" + sfx] = {c: sum(r["category"] == c for r in rows) for c in ["true_alarm", "false_negative", "false_alarm"]}
      rep["audit_high_conf_wrong" + sfx] = sum(r["high_confidence_wrong"] for r in rows)
    # ---------- predictions for the dashboard
    np.savez_compressed(out / "predictions.npz", split=split, labels=labels.astype(str), mask=mask,
                        **{k: v for k, v in scores.items()}, M4_q05=pred4["q05"], M4_q95=pred4["q95"],
                        M2_next=nxt2, M4_next=pred4["next_mean"], X=X_raw, C=C_raw)
    rep["seconds_total"] = time.time() - t_start
    atomic_write_json(out / "report.json", rep)
    return rep, {"em": em, "bayes": bayes, "scores": scores, "Xs": Xs, "Cs": Cs, "pred4": pred4,
                 "scaler": (mu_x, sd_x), "cov_scaler": (mu_c, sd_c)}
