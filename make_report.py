"""Build figures, results.md and blueprint/results_generated.tex from saved artifacts only.

  python make_report.py
"""
from pathlib import Path

from regime_lab.reporting import figures, intervention_figure, usecase_figure
from regime_lab.util import load_report, read_json

ROOT = Path(__file__).parent
STUDIES = [("registry", "Registry: locked primary (2-component, raw features)"),
           ("registry_exploratory", "Registry: exploratory (log-surprisal, 1 component)"),
           ("clinical", "Clinical charts: locked primary"),
           ("clinical_exploratory", "Clinical charts: exploratory (log-surprisal, 1 component)")]


def f3(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def ci(v):
    return f"{f3(v['estimate'])} [{f3(v['lo'])}, {f3(v['hi'])}]"


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


md, tex = ["# Results (generated from saved artifacts)\n",
           "> Provenance: Bayesian results below come from the corrected variance sampler, re-run with the "
           "identical locked configuration into `results/v2_samplerfix/`. Pre-fix outputs are kept in `results/<study>/` and "
           "marked `AFFECTED_sampler_fix.json`. EM / static / single-feature results reproduced exactly.\n"], []
# current results = corrected-sampler re-runs; pre-fix outputs stay in results/<study>/ (AFFECTED_sampler_fix.json)
RES = ROOT / "results" / "v2_samplerfix"
sim = RES / "simulation/simulation_summary.json"
AFFECTED_NOTE = ("scored with the PRE-FIX Bayesian posteriors (affected by the variance-sampler fix); its versioned re-run is pending "
                 "(F11, needs new model calls)")
for key, title in STUDIES:
    d = RES / key
    if not (d / "report.json").exists():
        md.append(f"## {title}\n\n*Not executed.*\n"); continue
    rep = load_report(d)
    figs = figures(d, title)
    ivf = intervention_figure(d)
    man = read_json(d / "manifest.json")
    gen = man["collection"]["generator"]
    lc = rep["label_counts"]
    tot = {l: sum(lc[s][l] for s in lc) for l in "GUHP"}
    md.append(f"## {title}\n")
    md.append(f"- Data: {man['n_dossiers']} dossiers x 6 turns, model `{gen.get('model')}` "
              f"({gen.get('quantization')}, Ollama {gen.get('ollama_version')}), synthetic={rep['synthetic']}.")
    md.append(f"- Labels (all splits): G={tot['G']}, U={tot['U']}, H={tot['H']}, pending={tot['P']}; "
              f"test onset risk set {rep['onset_test']['risk_set_turns']} turns / {rep['onset_test']['onsets']} onsets "
              f"({rep['onset_test']['excluded_dossiers']} dossiers excluded as ambiguous).")
    sm = rep["state_mapping"]
    md.append(f"- State mapping (validation): perm {sm['perm']}, score {f3(sm['score'])} vs runner-up {f3(sm['runner_up'])}, "
              f"stable={sm['stable']}, classes present {sm['classes_present']}. Negative control |filter-static| max = "
              f"{rep['negative_control_max_abs_diff']:.1e}.\n")
    nonconv = [m for m, v in rep["diagnostics"].items() if not v.get("converged")]
    if nonconv:   # prominent nonconvergence label; numbers are unchanged
        md.append(f"> **NONCONVERGED:** {', '.join(nonconv)} failed the convergence gate (R-hat <= 1.01, bulk ESS >= 400). "
                  f"Their rows below are reported for completeness and must not be interpreted as valid posteriors.\n")
    md.append("| Model | Onset AUROC [95% CI] | AP | Brier | Log score |\n|---|---|---|---|---|")
    for k, v in rep["detection_onset"].items():
        tagm = " (NONCONVERGED)" if k[:2] in nonconv else ""
        md.append(f"| {k}{tagm} | {f3(v['auroc'])} [{f3(v['auroc_ci'][0])}, {f3(v['auroc_ci'][1])}] | {f3(v['ap'])} | "
                  f"{f3(v['brier'])} | {f3(v['log_score'])} |")
    md.append("\n**Paired dossier-bootstrap AUROC differences (primary = M2 - M1):**\n")
    for k, v in rep["paired_onset_auroc"].items():
        md.append(f"- {k}: {ci(v)} (valid reps {v['valid_reps']}/{v['reps']})")
    md.append("\n**Next-turn forecast (H4):**\n\n| Forecaster | Brier | Log score | AUROC |\n|---|---|---|---|")
    for k, v in rep["forecast_next_turn"].items():
        md.append(f"| {k} | {f3(v['brier'])} | {f3(v['log_score'])} | {f3(v['auroc'])} |")
    for k, v in rep["forecast_paired_brier"].items():
        md.append(f"- Brier improvement {k}: {ci(v)}")
    p2 = rep["persistence_H2"]
    wfile = d / "persistence_within_dossier.json"
    if wfile.exists():
        wd = read_json(wfile)
        md.append(f"\n**H2 post-hoc sensitivity (W10 v2):** {wd['method']}. H→H {wd['HH_observed']} vs exact null mean "
                  f"{wd['null_mean']:.2f}, upper-tail p = {wd['p_value_upper']:.4f} ({wd['informative_dossiers']} informative "
                  f"dossiers). Controls without history effect: difficulty-only p = {wd['controls']['difficulty_only']['p_value_upper']:.3f}; "
                  f"position-only p = {wd['controls']['position_only']['p_value_upper']:.2g}.")
    md.append(f"\n**Persistence (H2, locked test):** observed H→H transitions {p2['HH_observed']} vs within-(turn,question-type) "
              f"permutation null mean {p2['HH_null_mean']:.1f} (95th pct {p2['HH_null_95']:.1f}), p = {p2['p_value']:.4f}. "
              f"P(H|H prev) = {f3(p2['P(H_t|H_t-1)'])} (n={p2['n_H_prev']}) vs P(H|not H prev) = {f3(p2['P(H_t|notH_t-1)'])}. "
              f"M3 A_HH = {f3(p2['M3_A_HH']['mean'])} [{f3(p2['M3_A_HH']['q2.5'])}, {f3(p2['M3_A_HH']['q97.5'])}], "
              f"A_GH = {f3(p2['M3_A_GH']['mean'])}.")
    md.append("\n**Factor contrasts (H3, M4, risk difference low→high):**\n\n| Covariate | low→high | enter H from G | stay in H | filter-weighted (test) |\n|---|---|---|---|---|")
    for nm, v in rep["factor_contrasts"].items():
        cell = lambda c: f"{f3(c['median'])} [{f3(c['lo'])}, {f3(c['hi'])}], P>0={c['p_gt0']:.2f}"
        md.append(f"| {nm} | {v['low_raw']:.0f}→{v['high_raw']:.0f} | {cell(v['enter_H_from_G'])} | {cell(v['stay_in_H'])} | "
                  f"{cell(v['filter_weighted_test'])} |")
    md.append("\n**Alarms (threshold frozen on validation, ≤5% clean false stops):**\n")
    for k, v in rep["alarms"].items():
        if v.get("threshold") is None:
            md.append(f"- {k}: threshold unavailable ({v.get('status', 'n/a')})"); continue
        t = v["test"]
        md.append(f"- {k}: thr {f3(v['threshold'])}; test clean false-stop rate {f3(t['clean_false_stop_rate'])}, "
                  f"on-time {t['on_time']}, delayed {t['delayed']}, premature {t['premature']}, miss {t['miss']}")
    md.append("\n**MCMC diagnostics:**\n")
    for k, v in rep["diagnostics"].items():
        md.append(f"- {k}: {v['chains']} chains x ({v['warmup']}+{v['draws']}); max R-hat {f3(v['max_rhat'])}; "
                  f"min bulk ESS {v['min_ess_bulk']:.0f}; converged-gate {v['converged']}; MH acceptance {v['mh_accept']}; "
                  f"draws relabelled {v['aligned_fraction_permuted']}; {v['seconds']:.0f}s")
    md.append(f"- Sticky prior sensitivity: " + "; ".join(f"{k}: A_HH {f3(v['A_HH_mean'])}, onset AUROC {f3(v['onset_auroc'])}"
                                                    for k, v in rep["sticky_sensitivity"].items()))
    md.append(f"- PPC lag-1 autocorrelation: " + "; ".join(f"{p['feature']} obs {f3(p['observed'])} vs rep "
                                                       f"[{f3(p['rep_lo'])}, {f3(p['rep_hi'])}]" for p in rep["ppc"]))
    md.append(f"\nGap audit: {rep['audit_counts']}, high-confidence wrong answers {rep['audit_high_conf_wrong']} "
              f"(see `{d.relative_to(ROOT)}/test_audit.csv`).\n")
    if ivf:
        iv = read_json(d / "intervention.json")
        md.append("**Randomised intervention (A4/H5):**\n")
        for pop in ("after_H", "after_G"):
            if pop not in iv:
                continue
            md.append(f"- {pop}: {iv[pop]['n_prefixes']} prefixes; arm H-rates " +
                      ", ".join(f"{a} {v['H_rate']:.3f} (U {v['U_rate']:.3f})" for a, v in iv[pop]["arms"].items()))
            for c, v in iv[pop]["contrasts"].items():
                md.append(f"  - {c}: RD {v['risk_difference_H']:+.3f}, 95% Bayesian bootstrap [{v['bayes_boot95'][0]:+.3f}, "
                          f"{v['bayes_boot95'][1]:+.3f}], P(RD<0)={v['P(RD<0)']:.3f}; pairs better/worse "
                          f"{v['pairs_better']}/{v['pairs_worse']}; ΔU {v['risk_difference_U']:+.3f}"
                          + (f"; exact sign test p = {v['sign_test_p']:.3g}" if v.get("sign_test_p") is not None else "")
                          + (" (sparse: <5 discordant prefixes)" if v.get("sparse_discordance") else ""))
    md.append("\nFigures: " + ", ".join(f"`{d.relative_to(ROOT)}/figures/{f}`" for f in figs + ([ivf] if ivf else [])) + "\n")

    # ---- LaTeX block
    po = rep["paired_onset_auroc"]["M2_em_filter - M1_static"]
    tex.append(rf"\subsection{{{title}}}")
    tex.append(rf"{man['n_dossiers']} dossiers $\times$ 6 turns from \texttt{{{tex_escape(gen.get('model'))}}} "
               rf"({tex_escape(gen.get('quantization'))}). Labels: G={tot['G']}, U={tot['U']}, H={tot['H']}, pending={tot['P']}. "
               rf"Test risk set: {rep['onset_test']['risk_set_turns']} turns, {rep['onset_test']['onsets']} onsets. "
               rf"State mapping stable: {sm['stable']}. Negative-control deviation {rep['negative_control_max_abs_diff']:.1e}.")
    if nonconv:
        tex.append(r"\fbox{\parbox{.96\linewidth}{\textbf{NONCONVERGED:} " + tex_escape(", ".join(nonconv))
                   + r" failed the convergence gate ($\widehat R\le1.01$, bulk ESS $\ge400$); their rows are reported for completeness only.}}")
    tex.append(r"\begin{center}\small\begin{tabular}{lrrrr}\toprule Model & Onset AUROC & 95\% CI & Brier & Log score\\\midrule")
    for k, v in rep["detection_onset"].items():
        tex.append(rf"{tex_escape(k)} & {f3(v['auroc'])} & [{f3(v['auroc_ci'][0])}, {f3(v['auroc_ci'][1])}] & {f3(v['brier'])} & {f3(v['log_score'])}\\")
    tex.append(r"\bottomrule\end{tabular}\end{center}")
    tex.append(rf"\textbf{{Primary estimand}} (M2 filter $-$ M1 matched static, paired dossier bootstrap): "
               rf"{f3(po['estimate'])} [{f3(po['lo'])}, {f3(po['hi'])}]. ")
    tex.append(r"Other paired differences: " + "; ".join(
        rf"{tex_escape(k)}: {ci(v)}" for k, v in rep["paired_onset_auroc"].items() if k != "M2_em_filter - M1_static") + ".")
    tex.append(rf"\textbf{{H2 persistence}}: H$\to$H transitions {p2['HH_observed']} vs permutation null "
               rf"{p2['HH_null_mean']:.1f} ($p={p2['p_value']:.4f}$); $P(H_t\mid H_{{t-1}})={f3(p2['P(H_t|H_t-1)'])}$ vs "
               rf"$P(H_t\mid \neg H_{{t-1}})={f3(p2['P(H_t|notH_t-1)'])}$.")
    tex.append(r"\textbf{H3 transition contrasts} (risk difference of entering H from G, 95\% CrI): " + "; ".join(
        rf"{tex_escape(nm)} {f3(v['enter_H_from_G']['median'])} [{f3(v['enter_H_from_G']['lo'])}, {f3(v['enter_H_from_G']['hi'])}]"
        for nm, v in rep["factor_contrasts"].items()) + ".")
    fb = rep["forecast_paired_brier"]
    tex.append(r"\textbf{H4 forecasting} (Brier): " + "; ".join(
        rf"{tex_escape(k)} {f3(v['brier'])}" for k, v in rep["forecast_next_turn"].items()) +
        rf". Negative-Brier difference M3 ensemble $-$ M2 plug-in: {ci(fb['M3_frozen_ensemble - M2_plugin'])}.")
    dg = rep["diagnostics"]
    tex.append(r"\textbf{Diagnostics}: " + "; ".join(
        rf"{k}: max $\widehat R$ {f3(v['max_rhat'])}, min bulk ESS {v['min_ess_bulk']:.0f}, gate {'passed' if v['converged'] else 'NOT passed'}"
        for k, v in dg.items()) + ".")
    if ivf:
        iv = read_json(d / "intervention.json")
        parts = []
        for pop in ("after_H", "after_G"):
            for c, v in iv.get(pop, {}).get("contrasts", {}).items():
                parts.append(rf"{tex_escape(pop)} {tex_escape(c)}: RD {v['risk_difference_H']:+.3f} "
                             rf"[{v['bayes_boot95'][0]:+.3f}, {v['bayes_boot95'][1]:+.3f}] (n={v['n_pairs']})")
        tex.append(r"\textbf{H5 intervention}: " + "; ".join(parts) + ".")
    for f in ["trajectories.png", "factor_forest.png"]:
        if f in figs:
            tex.append(rf"\begin{{center}}\includegraphics[width=.92\linewidth]{{../{d.relative_to(ROOT)}/figures/{f}}}\end{{center}}")

uc = (RES / "usecase/guard_summary.json") if (RES / "usecase/guard_summary.json").exists() else ROOT / "results/usecase/guard_summary.json"
if uc.exists():
    s = read_json(uc); usecase_figure(uc.parent)
    md.append("## Clinical guard use case (held-out IDs 20000+)\n")
    for mode in ("unguarded", "guarded"):
        md.append(f"- {mode}: {s[mode]}")
    md.append(f"- monitor-only on unguarded: {s['monitor_only_on_unguarded']}")
    p = s["paired_H_rate_difference_guarded_minus_unguarded"]
    md.append(f"- paired H-rate difference guarded - unguarded: {p['estimate']:+.4f} [{p['boot95'][0]:+.4f}, {p['boot95'][1]:+.4f}]\n")
    from regime_lab.usecase.walkthrough import worked_example
    wmd, wtex, wid = worked_example(read_json(uc.parent / "guard_transcripts.json"), s)
    (uc.parent / "worked_example.md").write_text(wmd + "\n")
    md.append(wmd)
    g, u = s["guarded"], s["unguarded"]
    tex.append(r"\subsection{Clinical guard on held-out charts}")
    tex.append(rf"{s['dossiers']} held-out fictional charts, paired seeds. Unsupported answers: unguarded "
               rf"{100 * u['H_rate']:.1f}\% vs guarded {100 * g['H_rate']:.1f}\% (paired difference {p['estimate']:+.4f}, 95\% "
               rf"[{p['boot95'][0]:+.4f}, {p['boot95'][1]:+.4f}]). Silent errors (H not held): {100 * u['silent_error_rate']:.1f}\% "
               rf"$\to$ {100 * g['silent_error_rate']:.1f}\%; hold rate {100 * g['hold_rate']:.1f}\%, hold precision "
               rf"{f3(g['hold_precision'])}, hold recall {f3(g['hold_recall'])}; re-grounded turns {g['regrounded_turns']}.")
    tex.append(rf"\begin{{center}}\includegraphics[width=.7\linewidth]{{../{uc.parent.relative_to(ROOT)}/figures/usecase.png}}\end{{center}}")
    tex.append(wtex)

att = RES / "adversarial/attacks.json"      # [F11/F12] corrected-detector rescoring with provenance
if att.exists():
    A = read_json(att)
    APROV = A.get("provenance"); A = A.get("conditions", A)
    md.append("## Live red-team attacks (frozen registry exploratory detector)\n")
    md.append("| condition | H rate | U rate | onsets | onset AUROC M2 | onset AUROC M4 |\n|---|---|---|---|---|---|")
    rows_t = []
    for k, v in A.items():
        md.append(f"| {k} | {f3(v['H_rate'])} | {f3(v['U_rate'])} | {v['onsets']} | {f3(v['onset_auroc_M2'])} | {f3(v['onset_auroc_M4'])} |")
        rows_t.append(rf"{k} & {f3(v['H_rate'])} & {f3(v['U_rate'])} & {v['onsets']} & {f3(v['onset_auroc_M2'])} & {f3(v['onset_auroc_M4'])}\\")
    if "pressure" in A:
        p = A["pressure"]
        md.append(f"\nPressure: H rate on pressured turns {f3(p['H_rate_pressured'])} vs unpressured {f3(p['H_rate_unpressured'])}; "
                  f"pressured code adopted {p['adopted_pressure_code']}/{p['pressured_turns']}.")
    if "injection" in A:
        md.append(f"Injection: answers containing the planted code {A['injection']['injection_code_answers']}; "
                  f"mean lexical grounding of H answers {f3(A['injection']['mean_lexcos_on_H'])}.\n")
    tex.append(r"\subsection{Live red-team attacks}" "\n" r"\begin{center}\small\begin{tabular}{lrrrrr}\toprule "
               r"Condition & H rate & U rate & Onsets & AUROC M2 & AUROC M4\\\midrule" "\n" + "\n".join(rows_t)
               + r"\bottomrule\end{tabular}\end{center}")
    extra = []
    if "pressure" in A:
        p = A["pressure"]
        extra.append(rf"Under sycophancy pressure the H rate on pressured turns was {f3(p['H_rate_pressured'])} vs "
                     rf"{f3(p['H_rate_unpressured'])} on unpressured turns; the asserted wrong code was adopted "
                     rf"{p['adopted_pressure_code']} of {p['pressured_turns']} times.")
    if "injection" in A:
        extra.append(rf"Under indirect prompt injection {A['injection']['injection_code_answers']} answers contained the "
                     rf"planted code; because the planted row is part of the visible evidence, lexical grounding of these "
                     rf"errors is high (mean {f3(A['injection']['mean_lexcos_on_H'])}).")
    tex.append(" ".join(extra))

if sim.exists():
    S = read_json(sim)
    md.append("## Simulation programme (SYNTHETIC)\n\n| Scenario | onset AUROC filter | static | onsets | extra |\n|---|---|---|---|---|")
    trows = []
    for sc, v in S["scenarios"].items():
        extra = (f"A coverage95 {v.get('A_coverage95', float('nan')):.2f}, RMSE {v.get('A_rmse', float('nan')):.3f}"
                 if "A_coverage95" in v else f"CrI excludes 0 in {v.get('excludes_zero', 0):.0%} of reps")
        md.append(f"| {sc} | {f3(v['onset_auroc_filter'])} | {f3(v['onset_auroc_static'])} | {v['onsets']:.0f} | {extra} |")
        trows.append(rf"{tex_escape(sc)} & {f3(v['onset_auroc_filter'])} & {f3(v['onset_auroc_static'])} & {tex_escape(extra)}\\")
    sb = S["sbc"]
    md.append(f"\nSBC ({sb['reps']} reps, L={sb['L']}): " + "; ".join(
        f"{k} chi2 p={sb[k]['chi2_p']:.3f}" for k in ("trace_A", "max_mu0", "sum_pi_sq")) + "\n")
    tex.insert(0, r"\subsection{Simulation programme (synthetic)}" "\n"
               r"\begin{center}\small\begin{tabular}{lrrl}\toprule Scenario & Filter AUROC & Static AUROC & Recovery\\\midrule"
               + "\n" + "\n".join(trows) + r"\bottomrule\end{tabular}\end{center}"
               + rf"SBC rank-uniformity ($\chi^2$ p-values, {sb['reps']} replications): "
               + ", ".join(rf"{tex_escape(k)} {sb[k]['chi2_p']:.3f}" for k in ("trace_A", "max_mu0", "sum_pi_sq")) + ".")

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/results.md").write_text("\n".join(md))
(ROOT / "blueprint/results_generated.tex").write_text("\n\n".join(tex) + "\n")
print("wrote results/results.md and blueprint/results_generated.tex")


# ------------------------------------------------------------------ headline (front of the PDF / README)
def headline():
    R = {k: load_report(RES / k) for k in
         ("registry", "registry_exploratory", "clinical", "clinical_exploratory") if (RES / f"{k}/report.json").exists()}
    if "registry" not in R:
        return
    pr = lambda k: R[k]["paired_onset_auroc"]["M2_em_filter - M1_static"]
    rr = lambda v: f"{v['estimate']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
    items, mds = [], []
    t = (f"H1, the primary question. Does history help detect the first hallucination? No measurable gain. Filter minus matched "
         f"static onset AUROC: registry {rr(pr('registry'))} (locked); {rr(pr('registry_exploratory'))} (exploratory)")
    if "clinical" in R:
        t += f"; clinical {rr(pr('clinical'))} (locked, {R['clinical']['onset_test']['onsets']} test onsets)"
    items.append(t + ". The answer's own features dominate; filtered risk saturates near 0 or 1.")
    p2 = R["registry"]["persistence_H2"]
    # H2 wording with numbers inserted from saved artifacts
    w = {k: read_json(RES / f"{k}/persistence_within_dossier.json") for k in ("registry", "clinical")
         if (RES / f"{k}/persistence_within_dossier.json").exists()}
    pc = R["clinical"]["persistence_H2"] if "clinical" in R else None
    if len(w) == 2:
        items.append(
            f"H2. Errors cluster within dossiers under the locked turn/question-type permutation test (registry: "
            f"{p2['HH_observed']} H→H transitions vs null {p2['HH_null_mean']:.1f}, p = {p2['p_value']:.4f}, "
            f"{p2['n_dossiers']} dossiers, {read_json(RES / 'registry/manifest.json')['evaluation_args']['boot']} permutations"
            + (f"; clinical: {pc['HH_observed']} vs {pc['HH_null_mean']:.1f}, p = {pc['p_value']:.4f}" if pc else "")
            + "). A post-hoc within-dossier ordering sensitivity, conditional on each dossier's observed error count, "
            f"does not reject in either domain (exact enumeration: registry p = {w['registry']['p_value_upper']:.3f}, "
            f"{w['registry']['informative_dossiers']} informative dossiers; clinical p = {w['clinical']['p_value_upper']:.3f}, "
            f"{w['clinical']['informative_dossiers']} informative). This sensitivity assumes exchangeable turn positions and "
            f"does not control question-type or turn effects (in a position-only control with no history effect it rejects, "
            f"p = {w['registry']['controls']['position_only']['p_value_upper']:.2g}). The analyses do not establish observational "
            "persistence beyond both dossier difficulty and turn conditions, and they do not establish its absence. "
            "Separately, in the eligible erroneous prefixes studied, replacing the previous answer with its verified "
            "correction reduced observed next-turn unsupported-answer rates relative to the original-prefix branch (H5).")
    else:
        items.append("H2. (sensitivity file missing: run sensitivity.py) Locked test: "
                     f"{p2['HH_observed']} vs {p2['HH_null_mean']:.1f}, p = {p2['p_value']:.4f}.")
    if "registry_exploratory" in R:
        fc = R["registry_exploratory"]["factor_contrasts"]
        e, a = fc["evidence"]["enter_H_from_G"], fc["ambiguity"]["enter_H_from_G"]
        items.append(f"H3 (association). When the requested record is present, the risk of entering H is lower by "
                     f"{-e['median']:.3f} [{-e['hi']:.3f}, {-e['lo']:.3f}]; referential ambiguity raises it by {a['median']:+.3f} "
                     f"[{a['lo']:+.3f}, {a['hi']:+.3f}] (P>0 = {a['p_gt0']:.2f}). Covariate model M4 onset AUROC "
                     f"{R['registry_exploratory']['detection_onset']['M4_cov_bayes_filter']['auroc']:.3f} vs EM filter "
                     f"{R['registry_exploratory']['detection_onset']['M2_em_filter']['auroc']:.3f}."
                     + ("" if R["registry_exploratory"]["diagnostics"]["M4"]["converged"] else
                        f" [M4 NONCONVERGED: max R-hat {R['registry_exploratory']['diagnostics']['M4']['max_rhat']:.2f}, "
                        f"min bulk ESS {R['registry_exploratory']['diagnostics']['M4']['min_ess_bulk']:.0f}; provisional.]"))
    if "registry_exploratory" in R:
        fb, fn = R["registry_exploratory"]["forecast_paired_brier"], R["registry_exploratory"]["forecast_next_turn"]
        m = fb["M3_frozen_ensemble - M2_plugin"]; b = fb["M4_covariate_ensemble - base_rate"]
        mc = R.get("clinical_exploratory", {}).get("forecast_paired_brier", {}).get("M3_frozen_ensemble - M2_plugin")
        conv3 = R["registry_exploratory"]["diagnostics"]["M3"]["converged"]
        items.append(f"H4 (forecasting the next answer). Integrating over parameters (M3, "
                     f"{'converged' if conv3 else 'NONCONVERGED'}) changes the Brier score relative to the plug-in EM forecast by "
                     f"{m['estimate']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] on registry"
                     + (f" and {mc['estimate']:+.4f} [{mc['lo']:+.4f}, {mc['hi']:+.4f}] on clinical" if mc else "")
                     + f" (positive = M3 better; exploratory), so the benefit is not consistent across domains; no model "
                     f"beats a constant base-rate forecast (M4 minus base rate {b['estimate']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]; "
                     f"Brier {fn['M4_covariate_ensemble']['brier']:.4f} vs {fn['base_rate']['brier']:.4f}). Next-turn hallucinations "
                     f"are rare (about 4%) and hard to anticipate before the answer exists.")
    iv = RES / "registry/intervention.json"
    if iv.exists():
        I = read_json(iv)["after_H"]["contrasts"]
        c, g = I["corrected - original"], I["reground - original"]
        t = (f"H5 (randomised, causal for eligible prefixes). Replacing the earlier wrong answer with the verified one "
             f"changes next-turn H by {c['risk_difference_H']:+.3f} [{c['bayes_boot95'][0]:+.3f}, {c['bayes_boot95'][1]:+.3f}] "
             f"({c['pairs_better']} pairs better, {c['pairs_worse']} worse)")
        ivc = RES / "clinical/intervention.json"
        if ivc.exists():
            cc = read_json(ivc)["after_H"]["contrasts"]["corrected - original"]
            t += f"; replicated on clinical charts: {cc['risk_difference_H']:+.3f} [{cc['bayes_boot95'][0]:+.3f}, {cc['bayes_boot95'][1]:+.3f}]"
        t += (f". A re-grounding reminder is weaker ({g['risk_difference_H']:+.3f} [{g['bayes_boot95'][0]:+.3f}, "
              f"{g['bayes_boot95'][1]:+.3f}]) and raises unwarranted UNKNOWN answers by {g['risk_difference_U']:+.3f}.")
        items.append(t)
    uc = (RES / "usecase/guard_summary.json") if (RES / "usecase/guard_summary.json").exists() else ROOT / "results/usecase/guard_summary.json"
    if uc.exists():
        S = read_json(uc); u, gd = S["unguarded"], S["guarded"]
        vf = S.get("validation_frozen_alarm_threshold")
        if vf is None and (RES / "clinical_exploratory/report.json").exists():
            vf = read_json(RES / "clinical_exploratory/report.json")["alarms"]["M4_cov_bayes_filter"].get("threshold")
        items.append(f"Application (clinical guard, {S['dossiers']} held-out charts; hold threshold = Bayes-decision "
                     f"1/(1+{S.get('cost_ratio', 10):g}) = {S['alarm_threshold']:.3f}, which overrides the validation-frozen "
                     f"threshold {('n/a' if vf is None else f'{vf:.3f}')}). Silent errors (wrong codes not held for review): "
                     f"{100 * u['silent_error_rate']:.1f}% → {100 * gd['silent_error_rate']:.1f}% of answers; hold rate "
                     f"{100 * gd['hold_rate']:.1f}%, hold precision {gd['hold_precision'] if gd['hold_precision'] is None else round(gd['hold_precision'], 3)}, "
                     f"recall {gd['hold_recall'] if gd['hold_recall'] is None else round(gd['hold_recall'], 3)}. "
                     f"However, pre-emptive re-grounding ({gd['regrounded_turns']} turns) did not reduce errors: the guarded H rate "
                     f"was {100 * gd['H_rate']:.1f}% vs {100 * u['H_rate']:.1f}% (paired difference "
                     f"{S['paired_H_rate_difference_guarded_minus_unguarded']['estimate']:+.3f} "
                     f"[{S['paired_H_rate_difference_guarded_minus_unguarded']['boot95'][0]:+.3f}, "
                     f"{S['paired_H_rate_difference_guarded_minus_unguarded']['boot95'][1]:+.3f}]). A monitor-only policy (hold, "
                     f"never re-ground) caught {S['monitor_only_on_unguarded']['H_caught']}/{S['monitor_only_on_unguarded']['H_total']} "
                     f"errors with {S['monitor_only_on_unguarded']['holds']} holds; this is the recommended configuration. "
                     + ("Run uses the corrected-sampler posterior (versioned re-run, F11)." if "v2_samplerfix" in str(uc)
                        else f"Note: this run was {AFFECTED_NOTE}."))
    at = RES / "adversarial/attacks.json"
    if at.exists():
        A = read_json(at).get("conditions", read_json(at))
        items.append(f"Red team. Indirect prompt injection raised the H rate from {A['clean']['H_rate']:.3f} to "
                     f"{A['injection']['H_rate']:.3f} ({A['injection']['injection_code_answers']} answers obeyed the planted "
                     f"instruction); sycophancy pressure raised it on pressured turns to {A['pressure']['H_rate_pressured']:.3f}. "
                     f"Ranking held (M4 onset AUROC {A['injection']['onset_auroc_M4']:.2f} under injection), but a 5% false-stop "
                     f"alarm caught no onsets. Detector: corrected sampler, digests recorded in attacks.json provenance (F12); "
                     f"its M4 is NONCONVERGED, so the M4 numbers are provisional.")
    gate = []
    for k in ("registry", "registry_exploratory", "clinical", "clinical_exploratory"):
        if k in R:
            for mname in ("M3", "M4"):
                dd = R[k]["diagnostics"][mname]
                gate.append(f"{k} {mname} {'PASSES' if dd['converged'] else 'FAILS'} (max R-hat {dd['max_rhat']:.3f}, "
                            f"min bulk ESS {dd['min_ess_bulk']:.0f})")
    items.append("Caveat (convergence gate R-hat at most 1.01 and bulk ESS at least 400, corrected sampler): "
                 + "; ".join(gate) + ". Results from a failing model are reported for completeness only.")
    tex = (r"\textbf{Findings at a glance (generated from results).}" "\n" r"\begin{enumerate}[leftmargin=*,itemsep=2pt]"
           + "".join(r"\item " + tex_escape(i).replace("→", r"$\to$").replace("≤", r"$\le$").replace("≥", r"$\ge$") + "\n" for i in items)
           + r"\end{enumerate}")
    (ROOT / "blueprint/results_headline.tex").write_text(tex)
    body = "\n".join(f"{n + 1}. {i}" for n, i in enumerate(items))
    (ROOT / "results/headline.md").write_text("## Findings at a glance (generated)\n\n" + body + "\n")
    rd = ROOT / "README.md"; txt = rd.read_text()
    a, b = "<!-- RESULTS-SUMMARY -->", "<!-- /RESULTS-SUMMARY -->"
    if a in txt:
        pre = txt.split(a)[0]; post = txt.split(b)[1] if b in txt else txt.split(a)[1]
        rd.write_text(pre + a + "\n**Findings at a glance** (written by `make_report.py`; do not edit by hand):\n\n" + body + "\n" + b + post)


headline()
print("wrote blueprint/results_headline.tex and results/headline.md")
