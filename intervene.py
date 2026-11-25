"""Randomised paired intervention experiment on a collected run.

  python intervene.py --dir runs/registry --out results/registry --reps 2
"""
import argparse

from regime_lab.data.collector import MockGenerator, OllamaGenerator
from regime_lab.evaluation.intervention import run_intervention

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--max-per-pop", type=int, default=None)
ap.add_argument("--mock", action="store_true")
a = ap.parse_args()
gen = MockGenerator() if a.mock else OllamaGenerator()
res = run_intervention(gen, a.dir, a.out, reps=a.reps, workers=a.workers, max_per_pop=a.max_per_pop)
for pop in ("after_H", "after_G"):
    if pop in res:
        print(pop, res[pop]["arms"], res[pop]["contrasts"])
