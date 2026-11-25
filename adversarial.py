"""Adversarial checks (red-team).

  python adversarial.py --part starter          # audit the predecessor starter (baseline/starter)
  python adversarial.py --part pipeline --results results/registry --dir runs/registry
  python adversarial.py --part grader
  python adversarial.py --part attacks --results results/registry   # needs runs/attack_*
  python adversarial.py --part all ...
Writes results/adversarial/<part>.json and a combined adversarial.md.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from regime_lab.evaluation import adversarial as adv
from regime_lab.util import atomic_write_json, read_json

ROOT = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--part", default="all", choices=["starter", "pipeline", "grader", "attacks", "all", "md"])
ap.add_argument("--results", default="results/registry")
ap.add_argument("--dir", default="runs/registry")
ap.add_argument("--out", default="results/adversarial")
a = ap.parse_args()
out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
parts = ["starter", "pipeline", "grader", "attacks"] if a.part == "all" else [a.part]

if "starter" in parts:
    demo = ROOT / "results/adversarial/starter_demo"
    subprocess.run([sys.executable, "evaluate.py", "--demo", "--out", str(demo)], cwd=ROOT / "baseline/starter",
                   check=True, capture_output=True)
    atomic_write_json(out / "starter.json", adv.starter_audit(ROOT / "baseline/starter", demo))
if "pipeline" in parts:
    atomic_write_json(out / "pipeline.json", adv.pipeline_redteam(ROOT / a.results, ROOT / a.dir))
if "grader" in parts:
    atomic_write_json(out / "grader.json", adv.grader_battery())
if "attacks" in parts:
    dirs = {k: ROOT / f"runs/attack_{k}" for k in ("clean", "injection", "pressure")}
    dirs = {k: v for k, v in dirs.items() if (v / "raw").exists()}
    atomic_write_json(out / "attacks.json", adv.attack_eval(ROOT / a.results, dirs))

# combined markdown
md = ["# Adversarial checks (generated)\n"]
for part, title in [("starter", "Predecessor starter audit"), ("pipeline", "Pipeline red-team canaries"),
                    ("grader", "Grader adversarial battery")]:
    p = out / f"{part}.json"
    if not p.exists():
        continue
    rows = read_json(p)
    md.append(f"## {title}\n\n| id | check | result | details |\n|---|---|---|---|")
    for i, r in enumerate(rows):
        rid = r.get("id", f"G{i + 1}")
        res = {True: "PASS", False: "FAIL", None: "info"}[r["pass"]]
        det = {k: v for k, v in r.items() if k not in ("id", "check", "pass", "labels")}
        chk = r.get("check", f"grade {json.dumps(r.get('answer'))}")
        md.append(f"| {rid} | {chk} | **{res}** | `{json.dumps(det, default=str)[:260]}` |")
    md.append("")
if (out / "attacks.json").exists():
    A = read_json(out / "attacks.json")
    prov = A.get("provenance") if isinstance(A, dict) else None
    A = A.get("conditions", A)
    md.append("## Live red-team attacks on Qwen3-4B-Instruct (frozen registry detector)\n")
    md.append("| condition | dossiers | H rate | U rate | onsets | onset AUROC M2 | onset AUROC M4 | M4 on-time+delayed / onsets | M4 clean false-stop |\n|---|---|---|---|---|---|---|---|---|")
    for k, v in A.items():
        al = v["alarms_M4"]; caught = al["on_time"] + al["delayed"]
        f = lambda x: "n/a" if x is None or x != x else f"{x:.3f}"
        md.append(f"| {k} | {v['dossiers']} | {f(v['H_rate'])} | {f(v['U_rate'])} | {v['onsets']} | {f(v['onset_auroc_M2'])} | "
                  f"{f(v['onset_auroc_M4'])} | {caught}/{v['onsets']} | {f(al['clean_false_stop_rate'])} |")
    if prov:
        md.append(f"\nDetector provenance: `{prov['detector_results_dir']}`; report.json sha256 `{prov['sha256']['report.json'][:16]}…`; "
                  f"frozen thresholds {prov['frozen_thresholds']}; detector convergence {prov['detector_convergence']}.")
    for k, v in A.items():
        extra = {kk: v[kk] for kk in ("injection_code_answers", "H_reasons", "pressured_turns", "H_rate_pressured",
                                      "H_rate_unpressured", "adopted_pressure_code", "mean_lexcos_on_H",
                                      "mean_logprob_on_H") if kk in v}
        md.append(f"- {k}: `{json.dumps(extra)}`")
(out / "adversarial.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
