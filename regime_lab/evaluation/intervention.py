"""Randomised paired branch experiment (A4 / H5).

Eligible prefixes
  after_H : dossiers whose first unsupported answer (onset) is at turn t < T
  after_G : an equal-size seeded sample of clean prefixes (side-effect control)
Arms (all regenerated with the SAME seed per replicate -> matched pairs)
  original  : prefix unchanged
  corrected : the onset answer replaced by the verified correct answer (after_H only)
  reground  : prefix unchanged; next question preceded by an evidence re-grounding reminder
The arm order of generation is randomised per prefix. Outcome Y = 1 if the
next-turn answer is reference-graded H. Analysis is at the prefix level
(replicates averaged), with whole-prefix bootstrap and a Bayesian bootstrap.
"""
from __future__ import annotations

import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..data.grader import grade_answer
from ..util import atomic_write_json, derive_seed, read_json

REGROUND = ("Reminder: answer strictly from the records listed above. If the requested field is not "
            "listed for that exact person, reply UNKNOWN. ")


def eligible_prefixes(run_dir, seed=0, max_per_pop=None):
    d = Path(run_dir)
    after_h, after_g = [], []
    for p in sorted(glob.glob(str(d / "grades" / "*.json")), key=lambda x: int(Path(x).stem)):
        g = read_json(p); labs = [x["label"] for x in g["labels"]]
        did = g["dossier_id"]
        if "H" in labs:
            t = labs.index("H")
            if t < len(labs) - 1 and "P" not in labs[:t + 2]:
                after_h.append((did, t))
        elif "P" not in labs:
            after_g.append(did)
    rng = np.random.default_rng(seed)
    g_pref = [(did, int(rng.integers(0, 5))) for did in rng.permutation(after_g)[:len(after_h)]]
    if max_per_pop:
        after_h, g_pref = after_h[:max_per_pop], g_pref[:max_per_pop]
    return after_h, g_pref


def _branch_messages(raw, gold, t, arm):
    msgs = [{"role": "system", "content": raw["system"]}]
    for k in range(t + 1):
        msgs.append({"role": "user", "content": raw["turns"][k]["question"]})
        ans = raw["turns"][k]["answer"]
        if arm == "corrected" and k == t:
            ans = gold["turns"][k]["answer"]
        msgs.append({"role": "assistant", "content": ans})
    q = raw["turns"][t + 1]["question"]
    msgs.append({"role": "user", "content": (REGROUND + q) if arm == "reground" else q})
    return msgs


def run_intervention(gen, run_dir, out_dir, reps=2, seed=2026, workers=3, max_per_pop=None, log=print):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    after_h, after_g = eligible_prefixes(run_dir, seed, max_per_pop)
    jobs = []
    rng = np.random.default_rng(seed)
    for pop, items, arms in [("after_H", after_h, ["original", "corrected", "reground"]),
                             ("after_G", after_g, ["original", "reground"])]:
        for did, t in items:
            for r in range(reps):
                for arm in rng.permutation(arms):          # randomised generation order
                    jobs.append((pop, did, t, r, str(arm)))
    log(f"intervention: {len(after_h)} after_H prefixes, {len(after_g)} after_G prefixes, {len(jobs)} generations")
    cache = {}

    def work(job):
        pop, did, t, r, arm = job
        raw = cache.setdefault(("r", did), read_json(run_dir / "raw" / f"{did}.json"))
        gold = cache.setdefault(("g", did), read_json(run_dir / "gold" / f"{did}.json"))
        s = derive_seed(seed, "intervention", raw["domain"], did, t, r)    # same across arms
        out = gen.chat(_branch_messages(raw, gold, t, arm), s)
        gr = grade_answer(out["text"], gold["turns"][t + 1], gold["all_codes"])
        return {"population": pop, "dossier_id": did, "prefix_turn": t + 1, "rep": r, "arm": arm, "seed": s,
                "answer": out["text"], "label": gr["label"], "reason": gr["reason"]}

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(work, jobs)):
            rows.append(row)
            if (i + 1) % 100 == 0:
                log(f"  {i + 1}/{len(jobs)}")
    atomic_write_json(out_dir / "intervention_rows.json", rows)
    res = analyse(rows, seed=seed)
    res["synthetic"] = bool(getattr(gen, "synthetic", False))
    atomic_write_json(out_dir / "intervention.json", res)
    return res


def _sign_test(better, worse):
    """Exact two-sided sign test on discordant pairs (supplementary, post hoc) [W5]."""
    from scipy.stats import binomtest
    n = better + worse
    return None if n == 0 else float(binomtest(better, n, 0.5).pvalue)


def analyse(rows, seed=0, boot=4000):
    """Paired branch analysis.

    [F7] Outcomes are paired at the replicate level (dossier, prefix, rep): a replicate enters a
    contrast only if BOTH arms are resolved (label != P). This is a *complete-pair* estimand, not
    strict ITT; pending counts per arm and worst / best-case ITT bounds (P treated as H / not H)
    are reported alongside. Replicates are then averaged per prefix (the resampling unit).
    """
    rng = np.random.default_rng(seed)
    out = {}
    for pop in ["after_H", "after_G"]:
        R = [r for r in rows if r["population"] == pop]
        if not R:
            continue
        arms = sorted({r["arm"] for r in R})
        lab = {(r["dossier_id"], r["prefix_turn"], r["rep"], r["arm"]): r["label"] for r in R}
        keys = sorted({(r["dossier_id"], r["prefix_turn"]) for r in R})
        reps = sorted({r["rep"] for r in R})
        res = {"n_prefixes": len(keys), "arms": {}, "pending_by_arm": {}, "contrasts": {}}
        for a in arms:
            L = [lab.get((k[0], k[1], rp, a)) for k in keys for rp in reps if (k[0], k[1], rp, a) in lab]
            res["pending_by_arm"][a] = int(sum(x == "P" for x in L))
            ok = [x for x in L if x != "P"]
            res["arms"][a] = {"H_rate": float(np.mean([x == "H" for x in ok])) if ok else None,
                              "U_rate": float(np.mean([x == "U" for x in ok])) if ok else None,
                              "n_valid": len(ok)}
        for a in arms:
            if a == "original":
                continue
            dH, dU, lo_b, hi_b = [], [], [], []
            for k in keys:
                pairs = [(lab.get((k[0], k[1], rp, a)), lab.get((k[0], k[1], rp, "original"))) for rp in reps]
                pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
                comp = [p for p in pairs if "P" not in p]
                if comp:
                    dH.append(np.mean([(x == "H") - (y == "H") for x, y in comp]))
                    dU.append(np.mean([(x == "U") - (y == "U") for x, y in comp]))
                if pairs:   # strict-ITT bounds: P in the arm counted H (worst) / not H (best); P in original the reverse
                    lo_b.append(np.mean([((x == "H") if x != "P" else 0) - ((y == "H") if y != "P" else 1) for x, y in pairs]))
                    hi_b.append(np.mean([((x == "H") if x != "P" else 1) - ((y == "H") if y != "P" else 0) for x, y in pairs]))
            dlt = np.array(dH)
            if len(dlt) == 0:
                res["contrasts"][f"{a} - original"] = {"status": "unavailable: no complete pairs"}
                continue
            bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(boot)]
            bb = [np.dot(rng.dirichlet(np.ones(len(dlt))), dlt) for _ in range(boot)]   # Bayesian bootstrap
            better, worse = int(np.sum(dlt < 0)), int(np.sum(dlt > 0))
            res["contrasts"][f"{a} - original"] = {
                "estimand": "complete-pair risk difference (replicate-paired, prefix-averaged); not strict ITT",
                "risk_difference_H": float(dlt.mean()), "boot95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                "bayes_boot95": [float(np.quantile(bb, .025)), float(np.quantile(bb, .975))],
                "P(RD<0)": float(np.mean(np.array(bb) < 0)), "n_pairs": int(len(dlt)),
                "pairs_better": better, "pairs_worse": worse,
                "sign_test_p": _sign_test(better, worse),
                "sparse_discordance": bool(better + worse < 5),
                "itt_bounds_H": [float(np.mean(lo_b)), float(np.mean(hi_b))] if lo_b else None,
                "risk_difference_U": float(np.mean(dU))}
        out[pop] = res
    out["estimand"] = ("complete-pair risk difference E[Y(arm) - Y(original)] of next-turn H, replicate-paired with "
                       "matched seeds, averaged per prefix; strict-ITT bounds and pending counts reported")
    out["supplementary"] = "sign_test_p: exact two-sided sign test on discordant prefixes (post hoc, W5)"
    return out
