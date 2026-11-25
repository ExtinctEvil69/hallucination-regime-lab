"""[F1-diag v1] Recompute Bayesian diagnostics for a SAVED study from its posterior draws (no refitting).

  python diagnostics_full_draw.py --results results/v2_samplerfix/registry_exploratory
Writes <results>/diagnostics_full_draw.json; readers prefer it over the thinned diagnostics in report.json.
"""
import argparse
from pathlib import Path

import numpy as np

from regime_lab.evaluation.pipeline import chain_discrepancy
from regime_lab.models import diagnostics as dg
from regime_lab.models.bayes import BayesHMM, cov_logA
from regime_lab.util import atomic_write_json, read_json

ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); a = ap.parse_args()
d = Path(a.results); rep = read_json(d / "report.json"); P = np.load(d / "predictions.npz", allow_pickle=True)
sc = rep["scaler"]; names = sc["cov_order"]
tr = P["split"] == "train"
Cs = (P["C"][tr] - np.array(sc["cov_mean"])) / np.array(sc["cov_scale"])
H = 2; out = {}
for name in ("M3", "M4"):
    dr = BayesHMM.load(d / f"posterior_{name}.npz").draws
    keys = {"loglik": dr["loglik"], "mu_H_logprob": np.einsum("csm,csm->cs", dr["w"][..., H, :], dr["mu"][..., H, :, 0]),
            "pi_H": dr["pi"][..., H]}
    if "theta" in dr:
        A_avg = np.stack([[np.exp(cov_logA(th, Cs))[:, 1:].mean((0, 1)) for th in ch] for ch in dr["theta"]])
        keys["A_GH_avg"], keys["A_HH_avg"] = A_avg[..., 0, H], A_avg[..., H, H]
        for k, nm in enumerate(names):
            keys[f"logit_G_to_G_vs_refH_{nm}"] = dr["theta"][..., 0, 0, 1 + k]
    else:
        keys["A_GH"], keys["A_HH"] = dr["A"][..., 0, H], dr["A"][..., H, H]
    params = [dg.summarize(k, v) for k, v in keys.items()]
    rh = max(p["rhat"] for p in params if p["rhat"] == p["rhat"]); es = min(p["ess_bulk"] for p in params if p["ess_bulk"] == p["ess_bulk"])
    out[name] = {"params": params, "max_rhat": rh, "min_ess_bulk": es, "converged": bool(rh <= 1.01 and es >= 400),
                 "draws_used": int(dr["loglik"].size), "method": "F1-diag v1: all draws; arithmetic mean of exp(log A)",
                 **chain_discrepancy(dr["loglik"])}
atomic_write_json(d / "diagnostics_full_draw.json", out)
old = rep["diagnostics"]
for m, v in out.items():
    print(d.name, m, f"thinned R-hat {old[m]['max_rhat']:.4f}/ESS {old[m]['min_ess_bulk']:.0f} -> full-draw {v['max_rhat']:.4f}/ESS {v['min_ess_bulk']:.0f}",
          "converged" if v["converged"] else "FAILS", "| heuristic gap chains:", v["heuristic_discrepancy_chains"])
