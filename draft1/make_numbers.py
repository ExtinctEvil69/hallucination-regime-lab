"""Generate every number quoted in the draft1 paper, report and deck from saved artifacts.

Writes draft1/numbers.json (for the deck) and draft1/numbers.tex (LaTeX macros). Nothing is typed by hand.
"""
import glob
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "v2_samplerfix"
OLD = ROOT / "results"
rj = lambda p: json.loads(Path(p).read_text())

N = {}
for dom in ("registry", "clinical"):
    c, reasons, ons = Counter(), Counter(), 0
    for p in glob.glob(str(ROOT / f"runs/{dom}/grades/*.json")):
        labs = rj(p)["labels"]
        c.update(l["label"] for l in labs); reasons.update(l["reason"] for l in labs)
        ons += any(l["label"] == "H" for l in labs)
    tot = sum(c.values())
    N[f"data_{dom}"] = {"dossiers": len(glob.glob(str(ROOT / f"runs/{dom}/grades/*.json"))), "turns": tot,
              "G": c["G"], "U": c["U"], "H": c["H"], "P": c["P"], "Hpct": 100 * c["H"] / tot, "Gpct": 100 * c["G"] / tot,
              "Upct": 100 * c["U"] / tot, "onset_dossiers": ons,
              "wrong_entity": reasons["wrong_entity_or_field"], "missing_row_code": reasons["unsupported_assertion:other_dossier_code"]
              + reasons["unsupported_assertion:hidden_code"] + reasons["unsupported_assertion:invented_code"],
              "invented": reasons["invented_code"]}
pil = {}
for d in ("pilot_v1_registry", "pilot_v2_registry", "pilot_v2_clinical"):
    labs = [l["label"] for p in glob.glob(str(ROOT / f"runs/{d}/grades/*.json")) for l in rj(p)["labels"]]
    pil[d] = 100 * labs.count("H") / max(1, len(labs))
N["pilot"] = pil

import sys as _s; _s.path.insert(0, str(ROOT))
from regime_lab.util import load_report
R = {k: load_report(RES / k) for k in ("registry", "registry_exploratory", "clinical", "clinical_exploratory")}
O = {k: rj(OLD / f"{k}/report.json") for k in R}
for k, r in R.items():
    pr = r["paired_onset_auroc"]
    N[k] = {"primary": pr["M2_em_filter - M1_static"], "m4_minus_static": pr["M4_cov_bayes_filter - M1_static"],
            "auroc": {m: {"est": v["auroc"], "lo": v["auroc_ci"][0], "hi": v["auroc_ci"][1]} for m, v in r["detection_onset"].items()},
            "onsets": r["onset_test"]["onsets"], "risk_set": r["onset_test"]["risk_set_turns"],
            "conv": {m: {"rhat": v["max_rhat"], "ess": v["min_ess_bulk"], "ok": v["converged"]} for m, v in r["diagnostics"].items()},
            "conv_pre": {m: {"rhat": v["max_rhat"], "ess": v["min_ess_bulk"]} for m, v in O[k]["diagnostics"].items()},
            "factor": {nm: v["enter_H_from_G"] for nm, v in r["factor_contrasts"].items()},
            "h2": r["persistence_H2"], "brier": r["forecast_paired_brier"],
            "forecast": {m: v["brier"] for m, v in r["forecast_next_turn"].items()},
            "within": rj(RES / f"{k}/persistence_within_dossier.json")}
for k in ("registry", "clinical"):
    iv = rj(RES / f"{k}/intervention.json")["after_H"]
    N[k]["iv"] = {"n": iv["n_prefixes"], "arms": iv["arms"], "corr": iv["contrasts"]["corrected - original"],
                  "regr": iv["contrasts"]["reground - original"]}
    ivg = rj(RES / f"{k}/intervention.json").get("after_G")
    if ivg:
        N[k]["iv_clean"] = ivg["contrasts"]["reground - original"]
g = rj(RES / "usecase/guard_summary.json")
N["guard"] = {"unguarded": g["unguarded"], "guarded": g["guarded"], "monitor": g["monitor_only_on_unguarded"],
              "paired": g["paired_H_rate_difference_guarded_minus_unguarded"], "thr": g["alarm_threshold"],
              "val_thr": g.get("validation_frozen_alarm_threshold"), "dossiers": g["dossiers"]}
a = rj(RES / "adversarial/attacks.json")["conditions"]
N["attacks"] = a
S1, S0 = rj(RES / "simulation/simulation_summary.json"), rj(OLD / "simulation/simulation_summary.json")
N["sbc"] = {"post": {k: S1["sbc"][k]["chi2_p"] for k in ("trace_A", "max_mu0", "sum_pi_sq")},
            "pre": {k: S0["sbc"][k]["chi2_p"] for k in ("trace_A", "max_mu0", "sum_pi_sq")},
            "cov_pre": S0["scenarios"]["independent"]["A_coverage95"], "cov_post": S1["scenarios"]["independent"]["A_coverage95"],
            "scen": {k: {kk: v.get(kk) for kk in ("onset_auroc_filter", "onset_auroc_static", "A_coverage95", "excludes_zero")}
                     for k, v in S1["scenarios"].items()}}
st = rj(OLD / "adversarial/starter.json")
N["starter"] = {r["id"]: {k: v for k, v in r.items() if k in ("false_stop_rate", "max_rhat_A", "min_ess_bulk_A", "pass")} for r in st}
(Path(__file__).parent / "numbers.json").write_text(json.dumps(N, indent=1, default=float))

# ---------------- LaTeX macros
def f(x, d=3, sign=False):
    return ("{:+." + str(d) + "f}").format(x) if sign else ("{:." + str(d) + "f}").format(x)
ci = lambda v, d=3: f"{f(v['estimate'], d, True)}\\,[{f(v['lo'], d, True)},\\,{f(v['hi'], d, True)}]"
M = {}
for dom, tag in (("registry", "Reg"), ("clinical", "Clin")):
    n = N[f"data_{dom}"]
    M[f"N{tag}Dossiers"] = n["dossiers"]; M[f"N{tag}Turns"] = n["turns"]; M[f"N{tag}H"] = n["H"]; M[f"N{tag}U"] = n["U"]
    M[f"N{tag}G"] = n["G"]; M[f"{tag}Hpct"] = f(n["Hpct"], 1); M[f"{tag}Gpct"] = f(n["Gpct"], 1); M[f"{tag}Upct"] = f(n["Upct"], 1)
    M[f"N{tag}Onset"] = n["onset_dossiers"]; M[f"N{tag}Wrong"] = n["wrong_entity"]; M[f"N{tag}MissCode"] = n["missing_row_code"]
    M[f"N{tag}Invented"] = n["invented"]
M["PilotVone"] = f(pil["pilot_v1_registry"], 1); M["PilotVtwo"] = f(pil["pilot_v2_registry"], 1)
for k, tag in (("registry", "RegL"), ("registry_exploratory", "RegX"), ("clinical", "ClinL"), ("clinical_exploratory", "ClinX")):
    n = N[k]
    M[f"Prim{tag}"] = ci(n["primary"]); M[f"Onsets{tag}"] = n["onsets"]
    for m, mt in (("M0_single_feature", "Mzero"), ("M1_static", "Mone"), ("M2_em_filter", "Mtwo"), ("M3_bayes_filter", "Mthree"),
                  ("M4_cov_bayes_filter", "Mfour")):
        v = n["auroc"][m]; M[f"Auc{mt}{tag}"] = f"{f(v['est'])}\\,[{f(v['lo'])},\\,{f(v['hi'])}]"
    for m in ("M3", "M4"):
        mm = "Mthree" if m == "M3" else "Mfour"
        M[f"Rhat{mm}{tag}"] = f(n["conv"][m]["rhat"]); M[f"Ess{mm}{tag}"] = f(n["conv"][m]["ess"], 0)
        M[f"RhatPre{mm}{tag}"] = f(n["conv_pre"][m]["rhat"]); M[f"EssPre{mm}{tag}"] = f(n["conv_pre"][m]["ess"], 0)
    e, am = n["factor"]["evidence"], n["factor"]["ambiguity"]
    M[f"Evid{tag}"] = f"{f(e['median'], 3, True)}\\,[{f(e['lo'], 3, True)},\\,{f(e['hi'], 3, True)}]"
    M[f"Amb{tag}"] = f"{f(am['median'], 3, True)}\\,[{f(am['lo'], 3, True)},\\,{f(am['hi'], 3, True)}]"; M[f"AmbP{tag}"] = f(am["p_gt0"], 2)
    M[f"HtwoP{tag}"] = f(n["h2"]["p_value"], 4); M[f"HtwoObs{tag}"] = n["h2"]["HH_observed"]; M[f"HtwoNull{tag}"] = f(n["h2"]["HH_null_mean"], 1)
    M[f"WithinP{tag}"] = f(n["within"]["p_value_upper"], 3); M[f"WithinInf{tag}"] = n["within"]["informative_dossiers"]
    M[f"BrierMthree{tag}"] = ci(n["brier"]["M3_frozen_ensemble - M2_plugin"], 4)
    M[f"BrierBase{tag}"] = ci(n["brier"]["M4_covariate_ensemble - base_rate"], 4)
M["PHgivenH"] = f(100 * N["registry"]["h2"]["P(H_t|H_t-1)"], 1); M["PHgivenNotH"] = f(100 * N["registry"]["h2"]["P(H_t|notH_t-1)"], 1)
M["PosOnlyP"] = "{:.1e}".format(N["registry"]["within"]["controls"]["position_only"]["p_value_upper"]).replace("e-", "\\times10^{-") + "}"
M["DiffOnlyP"] = f(N["registry"]["within"]["controls"]["difficulty_only"]["p_value_upper"], 3)
for dom, tag in (("registry", "Reg"), ("clinical", "Clin")):
    c, r = N[dom]["iv"]["corr"], N[dom]["iv"]["regr"]
    M[f"Corr{tag}"] = f"{f(c['risk_difference_H'], 3, True)}\\,[{f(c['bayes_boot95'][0], 3, True)},\\,{f(c['bayes_boot95'][1], 3, True)}]"
    M[f"CorrBetter{tag}"] = c["pairs_better"]; M[f"CorrWorse{tag}"] = c["pairs_worse"]; M[f"CorrSign{tag}"] = "{:.2g}".format(c["sign_test_p"])
    M[f"Regr{tag}"] = f"{f(r['risk_difference_H'], 3, True)}\\,[{f(r['bayes_boot95'][0], 3, True)},\\,{f(r['bayes_boot95'][1], 3, True)}]"
    M[f"RegrU{tag}"] = f(r["risk_difference_U"], 3, True); M[f"IvN{tag}"] = N[dom]["iv"]["n"]
    M[f"OrigH{tag}"] = f(100 * N[dom]["iv"]["arms"]["original"]["H_rate"], 1); M[f"CorrH{tag}"] = f(100 * N[dom]["iv"]["arms"]["corrected"]["H_rate"], 1)
gd = N["guard"]
M["GuardSilentU"] = f(100 * gd["unguarded"]["silent_error_rate"], 1); M["GuardSilentG"] = f(100 * gd["guarded"]["silent_error_rate"], 1)
M["GuardHeld"] = gd["guarded"]["held"]; M["GuardHoldRate"] = f(100 * gd["guarded"]["hold_rate"], 1)
M["GuardPrec"] = f(gd["guarded"]["hold_precision"], 3); M["GuardRecall"] = f(gd["guarded"]["hold_recall"], 3)
M["GuardHU"] = f(100 * gd["unguarded"]["H_rate"], 1); M["GuardHG"] = f(100 * gd["guarded"]["H_rate"], 1)
M["GuardPaired"] = f"{f(gd['paired']['estimate'], 3, True)}\\,[{f(gd['paired']['boot95'][0], 3, True)},\\,{f(gd['paired']['boot95'][1], 3, True)}]"
M["GuardMonCaught"] = gd["monitor"]["H_caught"]; M["GuardMonTotal"] = gd["monitor"]["H_total"]; M["GuardMonHolds"] = gd["monitor"]["holds"]
M["GuardThr"] = f(gd["thr"], 3); M["GuardRegr"] = gd["guarded"]["regrounded_turns"]; M["GuardHHeld"] = gd["guarded"]["H_held"]; M["GuardHtot"] = gd["guarded"]["H"]
M["AtkClean"] = f(100 * a["clean"]["H_rate"], 1); M["AtkInj"] = f(100 * a["injection"]["H_rate"], 1); M["AtkPress"] = f(100 * a["pressure"]["H_rate"], 1)
M["AtkPressed"] = f(100 * a["pressure"]["H_rate_pressured"], 1); M["AtkObey"] = a["injection"]["injection_code_answers"]
M["AtkAdopt"] = a["pressure"]["adopted_pressure_code"]; M["AtkPressTurns"] = a["pressure"]["pressured_turns"]
M["AtkAucInj"] = f(a["injection"]["onset_auroc_M4"], 2)
M["SbcPre"] = f(N["sbc"]["pre"]["max_mu0"], 3); M["SbcPost"] = f(N["sbc"]["post"]["max_mu0"], 3)
M["CovPre"] = f(N["sbc"]["cov_pre"], 3); M["CovPost"] = f(N["sbc"]["cov_post"], 3)
M["StarterFS"] = f(100 * N["starter"]["S5"]["false_stop_rate"], 0); M["StarterRhat"] = f(N["starter"]["S6"]["max_rhat_A"], 2)
M["StarterEss"] = f(N["starter"]["S6"]["min_ess_bulk_A"], 0)
tex = "% generated by draft1/make_numbers.py -- do not edit\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items())
(Path(__file__).parent / "numbers.tex").write_text(tex)
print(len(M), "macros written")

# ---------------- worked example table (chart 20003, corrected guard run, recomputed chain; no LLM calls)
import sys
sys.path.insert(0, str(ROOT))
from regime_lab.data.collector import MockGenerator          # noqa: E402
from regime_lab.usecase.guard import RegimeGuard               # noqa: E402
G = RegimeGuard(RES / "clinical_exploratory", MockGenerator(), forecast_threshold=1 / 11, alarm_threshold=1 / 11)
tr = {d["dossier_id"]: d for d in rj(RES / "usecase/guard_transcripts.json")}[20003]["unguarded"]
ch = G.replay_chain(tr)
rows = []
esc = lambda t: t.replace("_", r"\_").replace("&", r"\&")
for t, c in zip(tr, ch):
    cv = t["covariates"]; pr, lk, po = c["chain"]["prior"], c["chain_post"]["likelihood"], c["chain_post"]["posterior"]
    fm = lambda v: "/".join(f"{x:.2f}" for x in v)
    rows.append(f"{t['turn']} & {esc(t['question'])} & {cv['evidence']}/{cv['ambiguity']} & {fm(pr)} & \\texttt{{{esc(t['answer'].strip())}}} & "
                f"{fm(lk)} & {fm(po)} & {'hold' if po[2] > 1 / 11 else 'commit'} & {t['label']}\\\\")
hdr = (r"\begin{table}[H]\centering\scriptsize" "\n" r"\begin{tabularx}{\linewidth}{c>{\raggedright\arraybackslash}Xccccccc}\toprule"
       "\n" r"Turn & Question & Ev/Amb & Prior & Answer & Evidence & Posterior & Guard & Grade\\\midrule" "\n")
ftr = ("\n" r"\bottomrule\end{tabularx}" "\n"
       r"\caption{Worked example (corrected model; values recomputed from the saved run).\label{tab:we}}\end{table}" "\n")
(Path(__file__).parent / "worked_example.tex").write_text(hdr + "\n".join(rows) + ftr)
print("worked example rows:", len(rows))
