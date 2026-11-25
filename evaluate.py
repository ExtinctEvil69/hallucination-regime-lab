"""Run the study.

  python evaluate.py --demo --out results/simulation          # SYNTHETIC programme
  python evaluate.py --dir runs/registry --out results/registry
  python evaluate.py --dir runs/clinical --out results/clinical --cost-ratio 10
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from regime_lab.data.dossiers import PRIMARY_COVARIATES, assign_splits
from regime_lab.data.splits import bind_splits
from regime_lab.evaluation.pipeline import run_study
from regime_lab.features.extract import build_dataset
from regime_lab.util import atomic_write_json, read_json

ap = argparse.ArgumentParser()
ap.add_argument("--dir")
ap.add_argument("--out", required=True)
ap.add_argument("--split-source", help="Explicit legacy migration: existing locked splits.json copied byte-for-byte to the dataset")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--draws", type=int, default=1000)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--cost-ratio", type=float, default=None)
ap.add_argument("--transform", default="none", choices=["none", "log_surprisal"])
ap.add_argument("--M", type=int, default=2, help="Gaussian components per state")
ap.add_argument("--quick", action="store_true", help="small chains for smoke tests")
ap.add_argument("--sim-reps", type=int, default=5)
ap.add_argument("--sbc-reps", type=int, default=100)
a = ap.parse_args()
if a.quick:
    a.chains, a.warmup, a.draws, a.boot = 2, 150, 150, 200
out = Path(a.out)
if (out / "report.json").exists() or (out / "model.npz").exists() or (a.demo and
        ((out / "simulation_summary.json").exists() or (out / "covariate_full" / "model.npz").exists())):
    raise FileExistsError(f"Existing analysis in {out}; use a new versioned output directory")
out.mkdir(parents=True, exist_ok=True)


def to_py(o):
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


if a.demo:
    from regime_lab.evaluation.sim_study import SCENARIOS, recovery, sbc
    from regime_lab.evaluation.simulate import simulate
    sim = simulate("covariate", N=500, seed=11)
    split = np.array(list(assign_splits(range(500)).values()))
    split = np.array([assign_splits(range(500))[i] for i in range(500)])
    qt = np.full(sim["z"].shape, "sim", object)
    rep, _ = run_study(sim["X"], sim["mask"], sim["C_raw"], sim["labels"], split, out / "covariate_full",
                       ["evidence", "ambiguity"], synthetic=True, qtype=qt, chains=a.chains, warmup=a.warmup,
                       n_draws=a.draws, boot=a.boot)
    print("full synthetic study written")
    summary = {"synthetic": True, "scenarios": {}}
    for sc in SCENARIOS:
        print("scenario", sc)
        summary["scenarios"][sc] = recovery(sc, reps=a.sim_reps, N=300, seed=100)
    print("SBC")
    summary["sbc"] = sbc(reps=a.sbc_reps)
    atomic_write_json(out / "simulation_summary.json", to_py(summary))
    raise SystemExit(0)

d = Path(a.dir)
raws = [read_json(p) for p in sorted(glob.glob(str(d / "raw" / "*.json")), key=lambda p: int(Path(p).stem))]
raws = [r for r in raws if r.get("complete")]
grades = {r["dossier_id"]: read_json(d / "grades" / f"{r['dossier_id']}.json") for r in raws}
ds = build_dataset(raws, grades)
splits = bind_splits(d, ds.ids.tolist(), source=a.split_source, out_dir=out)
split = np.array([splits[str(int(i))] for i in ds.ids])


def audit_text(n, t):
    """[F9] Resolve the queried entity turn by turn (visible info only) and list its visible rows."""
    from regime_lab.usecase.guard import live_covariates
    r = ds.raws[n]
    prev, ent, status = None, None, "unresolved"
    for k in range(t + 1):
        _, ent, status = live_covariates(r["rows"], ds.questions[n, k], prev, k + 1, r.get("domain", "registry"),
                                         return_status=True)
        prev = ent
    rows = [row for row in r["rows"] if ent is not None and row.startswith(f"{ent[0]} (")]
    return {"dossier_id": int(ds.ids[n]), "qtype": ds.qtype[n, t], "question": ds.questions[n, t],
            "answer": ds.answers[n, t], "grade_reason": ds.reasons[n, t], "reference_status": status,
            "resolved_entity": "" if ent is None else f"{ent[0]} ({ent[1]})",
            "visible_rows_for_entity": " || ".join(rows) if rows else "(entity has no visible rows)" if ent else ""}


rep, _ = run_study(ds.X, ds.mask, ds.C, ds.labels, split, out, list(PRIMARY_COVARIATES),
                   synthetic=bool(raws[0].get("synthetic")), qtype=ds.qtype, audit_text=audit_text,
                   chains=a.chains, warmup=a.warmup, n_draws=a.draws, boot=a.boot, cost_ratio=a.cost_ratio,
                   transform=a.transform, M=a.M)
manifest = read_json(d / "manifest.json")
atomic_write_json(out / "manifest.json", {"collection": manifest, "n_dossiers": len(raws),
                                          "evaluation_args": vars(a)})
p = rep["paired_onset_auroc"]["M2_em_filter - M1_static"]
print(f"primary: onset AUROC filter-static = {p['estimate']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}]")
