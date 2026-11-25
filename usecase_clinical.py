"""Clinical use case: guarded vs unguarded medication-reconciliation conversations.

  python usecase_clinical.py --results results/clinical --start 20000 --count 50 --out results/usecase

Each guarded dossier (IDs from 20000; never used for fitting or evaluation) is
also run unguarded with identical per-turn seeds, so the comparison is paired.
All patients, names and codes are fictional.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from regime_lab.data.collector import OllamaGenerator
from regime_lab.data.dossiers import make_dossier
from regime_lab.data.grader import grade_answer
from regime_lab.usecase.guard import RegimeGuard
from regime_lab.util import atomic_write_json, derive_seed, read_json

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/clinical")
ap.add_argument("--start", type=int, default=20000)
ap.add_argument("--count", type=int, default=50)
ap.add_argument("--out", default="results/usecase")
ap.add_argument("--forecast-threshold", type=float, default=None)
ap.add_argument("--alarm-threshold", type=float, default=None,
                help="hold threshold; default = Bayes decision 1/(1+C_FN/C_FP) with cost ratio 10")
ap.add_argument("--cost-ratio", type=float, default=10.0)
ap.add_argument("--workers", type=int, default=3)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
gen = OllamaGenerator()
rep = read_json(Path(a.results) / "report.json")
# forecast threshold: Bayes decision rule with C_FN/C_FP = 10 unless given
bayes_thr = 1 / (1 + a.cost_ratio)
ft = a.forecast_threshold if a.forecast_threshold is not None else bayes_thr
at = a.alarm_threshold if a.alarm_threshold is not None else bayes_thr
guard = RegimeGuard(a.results, gen, forecast_threshold=ft, alarm_threshold=at)


def run(did):
    vis, gold = make_dossier(did, "clinical", 2026, "v2")
    qs = [t["question"] for t in vis["turns"]]
    seeds = [derive_seed(2026, "clinical", did, t["turn"]) for t in vis["turns"]]
    res = {"dossier_id": did}
    for mode in ["unguarded", "guarded"]:
        log = guard.run(vis["system"], vis["rows"], qs, seeds, guard=(mode == "guarded"))
        for row, g in zip(log, gold["turns"]):
            gr = grade_answer(row["answer"], g, gold["all_codes"])
            row["label"], row["reason"] = gr["label"], gr["reason"]
        res[mode] = log
    return res


with ThreadPoolExecutor(max_workers=a.workers) as ex:
    results = list(ex.map(run, range(a.start, a.start + a.count)))
atomic_write_json(out / "guard_transcripts.json", results)


def summarise(mode):
    rows = [r for d in results for r in d[mode]]
    lab = np.array([r["label"] for r in rows]); hold = np.array([r["hold"] for r in rows])
    H = lab == "H"
    return {"turns": len(rows), "H": int(H.sum()), "H_rate": float(H.mean()),
            "U_rate": float(np.mean(lab == "U")), "held": int(hold.sum()), "hold_rate": float(hold.mean()),
            "H_held": int((H & hold).sum()), "H_unheld (silent errors)": int((H & ~hold).sum()),
            "silent_error_rate": float(np.mean(H & ~hold)),
            "hold_precision": float(H[hold].mean()) if hold.any() else None,
            "hold_recall": float(hold[H].mean()) if H.any() else None,
            "regrounded_turns": int(sum(r["reground"] for r in rows)),
            "dossiers_with_any_H": int(sum(any(r["label"] == "H" for r in d[mode]) for d in results))}


# unguarded run also gets the *monitor-only* view: what holds would have fired
ung = [r for d in results for r in d["unguarded"]]
lab_u = np.array([r["label"] for r in ung]); risk_u = np.array([r["risk_after"] for r in ung])
mon = risk_u > guard.alarm_threshold
summary = {"synthetic": False, "domain": "clinical (fictional patients)", "dossiers": a.count,
           "forecast_threshold": ft, "alarm_threshold": guard.alarm_threshold, "cost_ratio": a.cost_ratio,
           "alarm_threshold_policy": "Bayes-decision 1/(1+cost_ratio); overrides the validation-frozen threshold [F13/N2]",
           "validation_frozen_alarm_threshold": rep["alarms"]["M4_cov_bayes_filter"].get("threshold"),
           "model_results": a.results,
           "unguarded": summarise("unguarded"), "guarded": summarise("guarded"),
           "monitor_only_on_unguarded": {"holds": int(mon.sum()), "H_caught": int((mon & (lab_u == "H")).sum()),
                                         "H_total": int((lab_u == "H").sum()),
                                         "silent_error_rate": float(np.mean((lab_u == "H") & ~mon))}}
d_h = np.array([np.mean([r["label"] == "H" for r in d["guarded"]]) - np.mean([r["label"] == "H" for r in d["unguarded"]])
                for d in results])
rng = np.random.default_rng(0)
bs = [d_h[rng.integers(0, len(d_h), len(d_h))].mean() for _ in range(4000)]
summary["paired_H_rate_difference_guarded_minus_unguarded"] = {"estimate": float(d_h.mean()),
                                                               "boot95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
atomic_write_json(out / "guard_summary.json", summary)
print(summary)
