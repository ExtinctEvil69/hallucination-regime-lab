"""Read-only figure and results.md generation from saved artifacts (never refits)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluation import metrics as mt  # noqa: E402
from .util import read_json  # noqa: E402

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2, "figure.dpi": 150, "savefig.bbox": "tight"})
MODEL_COLORS = {"M0_single_feature": SERIES[3], "M1_static": SERIES[2], "M2_em_filter": SERIES[1],
                "M4_cov_bayes_filter": SERIES[0]}
NICE = {"M0_single_feature": "M0 single feature", "M1_static": "M1 matched static", "M2_em_filter": "M2 EM-HMM filter",
        "M3_bayes_filter": "M3 Bayes HMM", "M3_bayes_reweighted": "M3 reweighted", "M4_cov_bayes_filter": "M4 covariate Bayes HMM"}


def _roc(y, s):
    o = np.argsort(-s); y = y[o]
    tp = np.r_[0, np.cumsum(y)] / max(y.sum(), 1); fp = np.r_[0, np.cumsum(1 - y)] / max((1 - y).sum(), 1)
    return fp, tp


def figures(res_dir, title):
    d = Path(res_dir); fig_dir = d / "figures"; fig_dir.mkdir(exist_ok=True)
    rep = read_json(d / "report.json"); P = np.load(d / "predictions.npz", allow_pickle=True)
    split, labels, mask = P["split"], P["labels"], P["mask"]
    te = split == "test"
    tag = "SYNTHETIC" if rep["synthetic"] else "real Qwen3-4B-Instruct outputs"
    made = []
    # 1 trajectories
    sel_n = [n for n in np.flatnonzero(te) if "H" in labels[n]][:6]
    if sel_n:
        fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharey=True)
        thr = rep["alarms"]["M4_cov_bayes_filter"].get("threshold") or 1.0
        for ax, n in zip(axes.ravel(), sel_n):
            t = np.arange(1, labels.shape[1] + 1)
            ax.fill_between(t, P["M4_q05"][n], P["M4_q95"][n], color=SERIES[0], alpha=.18, lw=0)
            ax.plot(t, P["M4_cov_bayes_filter"][n], color=SERIES[0], marker="o", ms=4, label="M4 posterior mean (90% band)")
            ax.plot(t, P["M2_em_filter"][n], color=SERIES[1], marker="s", ms=4, label="M2 EM filter")
            ax.plot(t, P["M1_static"][n], color=SERIES[2], ls="--", lw=1.5, label="M1 static")
            ax.axhline(thr, color=INK2, lw=1, ls=":")
            tau = int(np.flatnonzero(labels[n] == "H")[0]) + 1
            ax.axvline(tau, color=INK2, lw=1)
            for ti, lab in zip(t, labels[n]):
                ax.text(ti, 1.06, lab, ha="center", color=INK if lab == "H" else INK2,
                        fontweight="bold" if lab == "H" else None)
            ax.set_ylim(-.02, 1.12); ax.set_xticks(t)
        for ax in axes.ravel()[len(sel_n):]:
            ax.axis("off")
        axes[0, 0].set_ylabel("P(H | answers so far)"); axes[1, 0].set_ylabel("P(H | answers so far)")
        h, l = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.04))
        fig.suptitle(f"{title}: causal filtered risk on test dossiers with an onset ({tag})\n"
                     "letters = independent reference grade; vertical line = first H; dotted = frozen validation threshold",
                     fontsize=9, color=INK)
        fig.savefig(fig_dir / "trajectories.png"); plt.close(fig); made.append("trajectories.png")
    # 2 ROC onset
    sel, y, _ = mt.onset_risk_set(labels[te], mask[te])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    for k, c in MODEL_COLORS.items():
        fp, tp = _roc(y[sel], P[k][te][sel])
        a = rep["detection_onset"][k]["auroc"]
        ax.plot(fp, tp, color=c, label=f"{NICE[k]}  AUROC {a:.3f}")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"First-onset detection, test ({rep['onset_test']['onsets']} onsets)", fontsize=9)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.savefig(fig_dir / "roc_onset.png"); plt.close(fig); made.append("roc_onset.png")
    # 3 paired differences forest
    pr = rep["paired_onset_auroc"]
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    for i, (k, v) in enumerate(pr.items()):
        ax.plot([v["lo"], v["hi"]], [i, i], color=SERIES[0], lw=2, solid_capstyle="round")
        ax.plot(v["estimate"], i, "o", color=SERIES[0], ms=6, mec="white", mew=1.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(pr))); ax.set_yticklabels([k.replace("_", " ") for k in pr], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("paired onset AUROC difference (95% dossier bootstrap)")
    fig.savefig(fig_dir / "paired_auroc.png"); plt.close(fig); made.append("paired_auroc.png")
    # 4 factor forest
    fc = rep["factor_contrasts"]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    rows = []
    for nm, v in fc.items():
        rows.append((f"{nm}: enter H from G", v["enter_H_from_G"], SERIES[0]))
        rows.append((f"{nm}: stay in H", v["stay_in_H"], SERIES[1]))
    for i, (lab, v, c) in enumerate(rows):
        ax.plot([v["lo"], v["hi"]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(v["median"], i, "o", color=c, ms=6, mec="white", mew=1.5)
        ax.text(max(v["hi"], 0) + .01, i, f"P(>0)={v['p_gt0']:.2f}", va="center", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("risk difference in P(next state = H), low→high covariate (95% CrI)")
    ax.set_title("Transition attribution (M4) — association, not a causal effect", fontsize=9)
    fig.savefig(fig_dir / "factor_forest.png"); plt.close(fig); made.append("factor_forest.png")
    # 5 transition heatmap (M3)
    A = np.array(rep["em"]["A"])
    fig, ax = plt.subplots(figsize=(3.4, 3))
    ax.imshow(A, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color="white" if A[i, j] > .55 else INK)
    ax.set_xticks(range(3), ["G", "U", "H"]); ax.set_yticks(range(3), ["G", "U", "H"])
    ax.set_xlabel("destination"); ax.set_ylabel("origin"); ax.set_title("EM transition matrix (mapped states)", fontsize=9)
    fig.savefig(fig_dir / "transitions.png"); plt.close(fig); made.append("transitions.png")
    # 6 traces
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.4))
    for ax, name in zip(axes, ["M3", "M4"]):
        tr = np.array(rep["diagnostics"][name]["trace_loglik"])
        for c in range(len(tr)):
            ax.plot(tr[c], color=SERIES[c], lw=1, label=f"chain {c + 1}")
        dgn = rep["diagnostics"][name]
        ax.set_title(f"{name} log-likelihood trace  max R̂={dgn['max_rhat']:.3f}  min bulk ESS={dgn['min_ess_bulk']:.0f}",
                     fontsize=8)
        ax.set_xlabel("retained draw / 5")
    axes[0].legend(fontsize=7, ncol=2)
    fig.savefig(fig_dir / "traces.png"); plt.close(fig); made.append("traces.png")
    # 7 forecast reliability
    fsel, fy = mt.next_turn_set(labels[te], mask[te])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    for k, c, lab in [("M2_next", SERIES[1], "M2 plug-in"), ("M4_next", SERIES[0], "M4 posterior ensemble")]:
        p = P[k][te][fsel]; yy = fy[fsel]
        edges = np.quantile(p, np.linspace(0, 1, 6)); idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 4)
        xs = [p[idx == b].mean() for b in range(5) if (idx == b).any()]
        ys = [yy[idx == b].mean() for b in range(5) if (idx == b).any()]
        ax.plot(xs, ys, marker="o", ms=5, color=c, label=lab)
    mx = max(.3, float(np.nanmax(P["M4_next"][te][fsel])))
    ax.plot([0, mx], [0, mx], color=GRID, lw=1)
    ax.set_xlabel("forecast P(next answer is first H)"); ax.set_ylabel("observed frequency")
    ax.set_title("Next-turn forecast reliability (quintile bins)", fontsize=9); ax.legend(fontsize=7.5)
    fig.savefig(fig_dir / "forecast_reliability.png"); plt.close(fig); made.append("forecast_reliability.png")
    return made


def intervention_figure(res_dir):
    d = Path(res_dir)
    if not (d / "intervention.json").exists():
        return None
    iv = read_json(d / "intervention.json")
    fig, ax = plt.subplots(figsize=(5.6, 2))
    rows = [(f"{pop}: {k}", v) for pop in ("after_H", "after_G") if pop in iv for k, v in iv[pop]["contrasts"].items()]
    for i, (lab, v) in enumerate(rows):
        lo, hi = v["bayes_boot95"]
        ax.plot([lo, hi], [i, i], color=SERIES[0], lw=2, solid_capstyle="round")
        ax.plot(v["risk_difference_H"], i, "o", color=SERIES[0], ms=6, mec="white", mew=1.5)
        ax.text(hi + .01, i, f"n={v['n_pairs']}", va="center", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, lw=1); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("next-turn H risk difference vs original prefix (95% Bayesian bootstrap)")
    fig.savefig(d / "figures" / "intervention.png"); plt.close(fig)
    return "intervention.png"


def usecase_figure(uc_dir):
    d = Path(uc_dir)
    s = read_json(d / "guard_summary.json")
    fig, ax = plt.subplots(figsize=(5, 2.6))
    cats = ["H_rate", "silent_error_rate", "hold_rate", "U_rate"]
    names = ["unsupported answers", "silent errors (H not held)", "answers held", "unwarranted UNKNOWN"]
    x = np.arange(len(cats)); w = .38
    for j, (mode, c) in enumerate([("unguarded", SERIES[1]), ("guarded", SERIES[0])]):
        vals = [s[mode][k] for k in cats]
        ax.bar(x + (j - .5) * w, vals, w - .02, color=c, label=mode)
        for xi, v in zip(x, vals):
            ax.text(xi + (j - .5) * w, v + .004, f"{100 * v:.1f}%", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, names, fontsize=7.5); ax.set_ylabel("share of turns"); ax.legend(fontsize=7.5)
    ax.set_title(f"Clinical guard, {s['dossiers']} held-out fictional charts (paired seeds)", fontsize=9)
    (d / "figures").mkdir(exist_ok=True)
    fig.savefig(d / "figures" / "usecase.png"); plt.close(fig)
    return "usecase.png"
