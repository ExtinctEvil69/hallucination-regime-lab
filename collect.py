"""Collect six-turn conversations from a local Ollama model (default Qwen3-4B-Instruct).

  python collect.py --start 10000 --count 5 --dir runs/pilot_registry
  python collect.py --count 500 --dir runs/registry
  python collect.py --domain clinical --count 300 --dir runs/clinical
  python collect.py --mock --count 50 --dir runs/mock     # offline, SYNTHETIC
"""
import argparse

from regime_lab.data.collector import MockGenerator, OllamaGenerator, collect

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=5)
ap.add_argument("--dir", required=True)
ap.add_argument("--domain", default="registry", choices=["registry", "clinical"])
ap.add_argument("--model", default="qwen3:4b-instruct")
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--mock", action="store_true")
ap.add_argument("--design", default="v2", choices=["v1", "v2"])
ap.add_argument("--attack", default=None, choices=["injection", "pressure"], help="red-team variant")
a = ap.parse_args()
gen = MockGenerator() if a.mock else OllamaGenerator(a.model, a.temperature)
m = collect(gen, range(a.start, a.start + a.count), a.domain, a.dir, a.seed, a.workers, design=a.design, attack=a.attack)
print({k: m[k] for k in ("n_collected", "n_skipped_resume", "seconds")})
