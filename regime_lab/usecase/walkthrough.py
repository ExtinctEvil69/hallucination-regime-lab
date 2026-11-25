"""Turn-by-turn worked example of the clinical guard (Markdown + LaTeX)."""
from __future__ import annotations

from ..data.dossiers import make_dossier

SELECTION_RULE = ("Selection rule (fixed before looking at individual transcripts): the lowest-ID held-out "
                  "chart whose UNGUARDED run contains at least one unsupported (H) answer.")


def pick_example(transcripts):
    for d in sorted(transcripts, key=lambda x: x["dossier_id"]):
        if any(r["label"] == "H" for r in d["unguarded"]):
            return d
    return sorted(transcripts, key=lambda x: x["dossier_id"])[0]


def _tex(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\~{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def worked_example(transcripts, summary):
    d = pick_example(transcripts)
    did = d["dossier_id"]
    vis, gold = make_dossier(did, "clinical", 2026, "v2")
    at, ft = summary["alarm_threshold"], summary["forecast_threshold"]
    md = [f"### Worked example: held-out chart {did}\n", SELECTION_RULE + "\n",
          "**Step 0: what the assistant sees.** A fictional ward chart (system prompt). Rows that the chart "
          "*does not* contain are exactly what tempts the model to invent a code:\n", "```"]
    md += vis["rows"][:12] + ([f"... ({len(vis['rows']) - 12} more rows)"] if len(vis["rows"]) > 12 else []) + ["```\n"]
    md.append(f"Decision thresholds: re-ground if forecast > {ft:.3f}; hold if filtered P(H) > {at:.3f} "
              f"(Bayes rule with C_FN/C_FP = {summary.get('cost_ratio', 10):g}).\n")
    for t in range(len(d["unguarded"])):
        u, g = d["unguarded"][t], d["guarded"][t]
        gd = gold["turns"][t]
        c = u["covariates"]
        md.append(f"**Turn {t + 1}.** Question: *{u['question']}*  ")
        md.append(f"1. *Before answering* (visible info only): evidence row present = {c['evidence']}, "
                  f"candidates matching the reference = {c['ambiguity']}, near-duplicate codes = {c['distractor']}.  ")
        md.append(f"2. *Forecast* P(next answer is H) = {u['forecast_risk_before']:.3f} (unguarded) / "
                  f"{g['forecast_risk_before']:.3f} (guarded) → guard action: "
                  f"{'**re-ground the prompt**' if g['reground'] else 'none'}.  ")
        md.append(f"3. *Answers*: unguarded `{u['answer'].strip()}` / guarded `{g['answer'].strip()}` "
                  f"(reference: `{gd['answer']}`).  ")
        md.append(f"4. *Features of the unguarded answer*: mean log-prob {u['features']['logprob']:.3f}, top-k entropy "
                  f"{u['features']['entropy']:.3f}, lexical grounding {u['features']['lexcos']:.2f}.  ")
        md.append(f"5. *Filtered P(H)*: unguarded {u['risk_after']:.3f} [90% band {u['risk_q05']:.3f}–{u['risk_q95']:.3f}] / "
                  f"guarded {g['risk_after']:.3f} → guard decision: {'**HOLD for pharmacist verification**' if g['hold'] else 'auto-commit'}.  ")
        md.append(f"6. *Independent grade* (never seen by the guard): unguarded **{u['label']}** ({u['reason']}), "
                  f"guarded **{g['label']}** ({g['reason']}).\n")
    tex = [r"\subsection{Worked example: one held-out chart, step by step}", _tex(SELECTION_RULE),
           r"\textbf{Step 0 (what the assistant sees).} First rows of chart " + str(did) + ":",
           r"\begin{quote}\footnotesize\ttfamily " + r"\\ ".join(_tex(r) for r in vis["rows"][:8]) + r"\\ \ldots\end{quote}",
           rf"Re-ground if the forecast exceeds {ft:.3f}; hold if filtered $P(H)$ exceeds {at:.3f}.",
           r"\begin{center}\footnotesize\begin{tabularx}{\linewidth}{c>{\raggedright\arraybackslash}Xcccc>{\raggedright\arraybackslash}p{.11\linewidth}cc}\toprule",
           r"Turn & Question & Evid./Amb. & Forecast & Re-ground & $P(H)$ & Answer (unguarded / guarded) & Hold & Grade U/G\\\midrule"]
    for t in range(len(d["unguarded"])):
        u, g = d["unguarded"][t], d["guarded"][t]
        c = u["covariates"]
        tex.append(rf"{t + 1} & {_tex(u['question'])} & {c['evidence']}/{c['ambiguity']} & {g['forecast_risk_before']:.2f} & "
                   rf"{'yes' if g['reground'] else 'no'} & {u['risk_after']:.2f}/{g['risk_after']:.2f} & "
                   rf"{_tex(u['answer'].strip())} / {_tex(g['answer'].strip())} & {'yes' if g['hold'] else 'no'} & "
                   rf"{u['label']}/{g['label']}\\")
    tex.append(r"\bottomrule\end{tabularx}\end{center}")
    tex.append(r"Reading the table: the forecast uses only information available \emph{before} the answer (whether the "
               r"requested row exists, how many people match the reference, the turn). Filtered $P(H)$ is updated from the "
               r"answer's confidence and grounding features. The grade comes from the gold chart and is never visible "
               r"to the guard.")
    return "\n".join(md), "\n".join(tex), did
