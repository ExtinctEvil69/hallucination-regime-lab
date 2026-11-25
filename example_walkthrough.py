"""Run ONE fictional clinical chart live through the guard and print every step.

  python example_walkthrough.py --id 20003            # needs Ollama + qwen3:4b-instruct
  python example_walkthrough.py --id 20003 --no-guard # same seeds, monitoring only

What you will see per turn:
  [pre-answer]  covariates computed from the chart + question only
  [forecast]    P(next answer is a hallucination), averaged over posterior draws
  [action]      re-ground the prompt if the forecast is high
  [answer]      Qwen's reply and its features (log-prob, entropy, grounding)
  [filter]      updated P(H) with a 90% posterior band -> HOLD or auto-commit
  [grade]       the independent reference grade (from gold, never shown to the guard)
"""
import argparse

from regime_lab.data.collector import OllamaGenerator
from regime_lab.data.dossiers import make_dossier
from regime_lab.data.grader import grade_answer
from regime_lab.usecase.guard import RegimeGuard
from regime_lab.util import derive_seed

ap = argparse.ArgumentParser()
ap.add_argument("--id", type=int, default=20003)
ap.add_argument("--results", default="results/v2_samplerfix/clinical_exploratory")
ap.add_argument("--cost-ratio", type=float, default=10.0)
ap.add_argument("--no-guard", action="store_true")
a = ap.parse_args()

thr = 1 / (1 + a.cost_ratio)
guard = RegimeGuard(a.results, OllamaGenerator(), forecast_threshold=thr, alarm_threshold=thr)
vis, gold = make_dossier(a.id, "clinical", 2026, "v2")
print(f"=== Chart {a.id} (fictional) — {len(vis['rows'])} visible rows ===")
print("\n".join("  " + r for r in vis["rows"]))
print(f"\nthresholds: re-ground if forecast > {thr:.3f}; hold if P(H) > {thr:.3f}\n")
seeds = [derive_seed(2026, "clinical", a.id, t["turn"]) for t in vis["turns"]]
log = guard.run(vis["system"], vis["rows"], [t["question"] for t in vis["turns"]], seeds, guard=not a.no_guard)
for row, g in zip(log, gold["turns"]):
    gr = grade_answer(row["answer"], g, gold["all_codes"])
    c, f = row["covariates"], row["features"]
    print(f"--- Turn {row['turn']}: {row['question']}")
    print(f"  [pre-answer] evidence={c['evidence']} ambiguity={c['ambiguity']} distractor={c['distractor']}")
    print(f"  [forecast]   P(H next) = {row['forecast_risk_before']:.3f}")
    print(f"  [action]     {'RE-GROUND prompt' if row['reground'] else 'none'}")
    print(f"  [answer]     {row['answer'].strip()!r}  (logprob {f['logprob']:.3f}, entropy {f['entropy']:.3f}, grounding {f['lexcos']:.2f})")
    print(f"  [filter]     P(H) = {row['risk_after']:.3f}  [{row['risk_q05']:.3f}, {row['risk_q95']:.3f}]  -> "
          f"{'HOLD for verification' if row['hold'] else 'auto-commit'}")
    ch, cp = row["chain"], row["chain_post"]
    fmt = lambda v: "  ".join(f"{n}={x:.3f}" for n, x in zip("GUH", v))
    print(f"  [chain]      belief f(t-1): {fmt(ch['prev_belief'])}"
          + ("   (turn 1: initial distribution pi; the prior is pi and A(c) is not applied)" if ch.get("first_turn") else ""))
    print(f"               A(c) row G->: {fmt(ch['A'][0])} | U->: {fmt(ch['A'][1])} | H->: {fmt(ch['A'][2])}")
    print(f"               prior q=f*A:  {fmt(ch['prior'])}")
    print(f"               evidence b(x):{fmt(cp['likelihood'])}")
    print(f"               posterior f:  {fmt(cp['posterior'])}")
    print(f"  [grade]      {gr['label']} ({gr['reason']}); reference = {g['answer']}\n")
