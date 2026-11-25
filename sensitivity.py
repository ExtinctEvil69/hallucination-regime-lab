"""Write post-hoc sensitivities for an evaluated study (does not modify report.json).

  python sensitivity.py --results results/v2_samplerfix/registry
"""
import argparse
from pathlib import Path

import numpy as np

from regime_lab.evaluation.persistence import sensitivity
from regime_lab.util import atomic_write_json

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
a = ap.parse_args()
d = Path(a.results)
P = np.load(d / "predictions.npz", allow_pickle=True)
res = sensitivity(P["labels"])
atomic_write_json(d / "persistence_within_dossier.json", res)
print(d.name, {k: res[k] for k in ("HH_observed", "null_mean", "p_value_upper", "informative_dossiers")},
      {k: round(v["p_value_upper"], 4) for k, v in res["controls"].items()})
