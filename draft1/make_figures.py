"""Explanatory figures for draft1 (priors, pipeline, convergence, SBC, H2). Values read from saved artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

D = Path(__file__).parent; FIG = D / "figures"; FIG.mkdir(exist_ok=True)
ROOT = D.parent
N = json.loads((D / "numbers.json").read_text())
BLUE, ORANGE, AQUA, YELLOW, MAG, GREEN, VIOLET, RED = "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 200,
                     "savefig.bbox": "tight", "axes.titlesize": 9.5, "axes.titleweight": "bold"})

# ---------------------------------------------------------------- 1. priors
fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.6))
x = np.linspace(0, 1, 400)
a = ax[0, 0]
for k, c, lab in ((0, BLUE, "κ = 0 (default)"), (2, ORANGE, "κ = 2 (sticky)"), (5, VIOLET, "κ = 5 (sticky)")):
    a.plot(x, stats.beta(1 + k, 2).pdf(x), color=c, label=lab)
a.set_title("Transition row A_i· ~ Dirichlet(1+κ·1[i=j])"); a.set_xlabel("self-transition probability A_ii (marginal Beta)")
a.legend(fontsize=7.5); a.set_ylabel("density")
a = ax[0, 1]
v = np.linspace(0.01, 4, 400)
a.plot(v, stats.invgamma(3, scale=2).pdf(v), color=GREEN)
a.axvline(1.0, color=INK2, lw=1, ls=":"); a.text(1.05, a.get_ylim()[1] * .85, "mean = b0/(a0-1) = 1", fontsize=7.5, color=INK2)
a.set_title("Feature variance v ~ InvGamma(a0=3, b0=2)"); a.set_xlabel("variance (standardised units)")
a = ax[0, 2]
m = np.linspace(-5, 5, 400)
a.plot(m, stats.t(df=6, scale=np.sqrt(2 / (3 * .5))).pdf(m), color=AQUA, label="marginal of μ (Student-t, 6 df)")
a.plot(m, stats.norm(0, 1).pdf(m), color=INK2, lw=1, ls="--", label="N(0,1) for reference")
a.set_title("Component mean μ | v ~ N(m0=0, v/k0), k0=0.5"); a.set_xlabel("mean (standardised units)"); a.legend(fontsize=7.5)
a = ax[1, 0]
w = np.linspace(0, 1, 200)
a.plot(w, np.ones_like(w), color=YELLOW, label="mixture weight w ~ Dir(1,1) = Uniform")
a.plot(w, stats.beta(1, 2).pdf(w), color=BLUE, ls="--", lw=1.5, label="initial π_j ~ Dir(1,1,1): marginal Beta(1,2)")
a.legend(fontsize=7.2, loc="upper right"); a.set_ylim(0, 2.6); a.set_title("Weights: π ~ Dir(1,1,1), w_j ~ Dir(1,1)"); a.set_xlabel("probability")
a = ax[1, 1]
s = np.linspace(0, 2, 400)
a.plot(s, stats.halfnorm(scale=.5).pdf(s), color=MAG, label="σ_k ~ HalfNormal(0.5)")
a.plot(s, stats.norm(0, 1.5).pdf(s) * 2, color=VIOLET, ls="--", lw=1.5, label="|α| for α ~ N(0, 1.5²)")
a.set_title("Covariate HMM: β_ij,k ~ N(0, σ_k²)"); a.set_xlabel("scale"); a.legend(fontsize=7.5)
a = ax[1, 2]
rng = np.random.default_rng(0)
S = 20000
alpha = rng.normal(0, 1.5, (S, 2)); sig = np.abs(rng.normal(0, .5, (S, 4))); beta = rng.normal(0, 1, (S, 2, 4)) * sig[:, None, :]
c = rng.normal(0, 1, (S, 4))
eta = np.concatenate([alpha + np.einsum("sjk,sk->sj", beta, c), np.zeros((S, 1))], 1)
pH = np.exp(eta[:, 2]) / np.exp(eta).sum(1)
a.hist(pH, bins=40, density=True, color=RED, alpha=.85, edgecolor="white", linewidth=.4)
a.set_title("Prior predictive: implied P(next state = H)"); a.set_xlabel("transition probability into H (softmax of priors)")
fig.suptitle("Major priors of the Bayesian HMMs (M3 conjugate; M4 adds the covariate-transition priors)", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(FIG / "priors.png"); plt.close(fig)

# ---------------------------------------------------------------- 2. pipeline
fig, ax = plt.subplots(figsize=(11.4, 2.3)); ax.axis("off"); ax.set_xlim(0, 11.4); ax.set_ylim(0.45, 3.0)
boxes = [("Fictional chart\n+ question", .1, "#eef3fb"), ("Pre-answer\ncovariates c(t)", 1.45, "#eef3fb"),
         ("Forecast\nP(H next)", 2.8, "#dfeafa"), ("Qwen3-4B\nanswers", 4.15, "#f6efe6"), ("Features x(t)\nlog p · H · cos", 5.5, "#f6efe6"),
         ("Bayesian filter\nf(t) ∝ f(t-1)A(c)·b(x)", 6.85, "#dfeafa"), ("Hold or\nauto-commit", 8.6, "#e7f5ea"),
         ("Gold grade\n(audit only)", 9.95, "#f3f3f3")]
for t, x0, col in boxes:
    ax.add_patch(FancyBboxPatch((x0, 1.25), 1.15 if x0 < 6.8 or x0 > 8 else 1.62, .95, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=col, ec="#b8c2d0", lw=.8))
    ax.text(x0 + (0.575 if x0 < 6.8 or x0 > 8 else .81), 1.72, t, ha="center", va="center", fontsize=8.2, color=INK)
xs = [b[1] for b in boxes]
widths = [1.15 if x0 < 6.8 or x0 > 8 else 1.62 for x0 in xs]
for i in range(len(boxes) - 1):
    ax.add_patch(FancyArrowPatch((xs[i] + widths[i] + .02, 1.72), (xs[i + 1] - .02, 1.72), arrowstyle="-|>", mutation_scale=10,
                                 color=INK2, lw=1))
ax.text(1.45, .75, "visible information only", fontsize=7.5, color=BLUE)
ax.text(6.85, .75, "posterior draws: 4 chains × 1000", fontsize=7.5, color=BLUE)
ax.text(9.95, .75, "never seen by the guard", fontsize=7.5, color=INK2)
ax.text(.1, 2.75, "One turn of the guard (repeats every question)", fontsize=10, fontweight="bold", color=INK)
fig.savefig(FIG / "pipeline.png"); plt.close(fig)

# ---------------------------------------------------------------- 3. convergence pre/post
fig, ax = plt.subplots(figsize=(7.2, 3.3))
rows = []
for k, lab in (("registry", "Registry locked"), ("registry_exploratory", "Registry exploratory"), ("clinical", "Clinical locked"),
               ("clinical_exploratory", "Clinical exploratory")):
    for m in ("M3", "M4"):
        rows.append((f"{lab} {m}", N[k]["conv_pre"][m]["rhat"], N[k]["conv"][m]["rhat"]))
y = np.arange(len(rows))[::-1]
for yi, (lab, pre, post) in zip(y, rows):
    ax.plot([pre, post], [yi, yi], color=GRID, lw=3, zorder=1)
    ax.scatter(pre, yi, color=ORANGE, s=38, zorder=3, label="pre-fix sampler" if yi == y[0] else None)
    ax.scatter(post, yi, color=BLUE, s=38, zorder=3, label="corrected sampler" if yi == y[0] else None)
ax.axvline(1.01, color=GREEN, lw=1.2, ls="--"); ax.text(1.015, y[-1] - .7, "gate 1.01", color=GREEN, fontsize=7.5)
ax.set_yticks(y, [r[0] for r in rows], fontsize=8); ax.set_xscale("log"); ax.set_xticks([1, 1.2, 1.5, 2, 2.5], ["1.0", "1.2", "1.5", "2.0", "2.5"])
ax.set_xlabel("max rank-normalised split R-hat (log scale; lower is better)")
ax.set_title("Convergence before vs after the variance-sampler fix"); ax.legend(fontsize=7.5, loc="lower right")
fig.savefig(FIG / "convergence.png"); plt.close(fig)

# ---------------------------------------------------------------- 4. SBC histograms
S0 = json.loads((ROOT / "results/simulation/simulation_summary.json").read_text())["sbc"]
S1 = json.loads((ROOT / "results/v2_samplerfix/simulation/simulation_summary.json").read_text())["sbc"]
fig, ax = plt.subplots(1, 3, figsize=(10, 2.7), sharey=True)
for a, q in zip(ax, ("trace_A", "max_mu0", "sum_pi_sq")):
    b = np.arange(10)
    a.bar(b - .2, S0[q]["hist10"], .38, color=ORANGE, label=f"pre-fix (p={S0[q]['chi2_p']:.3f})")
    a.bar(b + .2, S1[q]["hist10"], .38, color=BLUE, label=f"corrected (p={S1[q]['chi2_p']:.3f})")
    a.axhline(10, color=INK2, lw=1, ls=":"); a.set_title(q.replace("_", " ")); a.set_xticks(b); a.set_xlabel("rank decile")
    a.legend(fontsize=7)
ax[0].set_ylabel("count (100 reps)")
fig.suptitle("Simulation-based calibration: ranks of the true value among posterior draws (uniform = correct)", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "sbc.png"); plt.close(fig)

# ---------------------------------------------------------------- 5. H2 exact within-dossier null
import sys
sys.path.insert(0, str(ROOT))
from regime_lab.evaluation.persistence import _hh_dist   # noqa: E402
P = np.load(ROOT / "results/v2_samplerfix/registry/predictions.npz", allow_pickle=True)
lab = P["labels"]; L = (lab[np.all(np.isin(lab, ["G", "U", "H"]), 1)] == "H").astype(int)
dist = np.array([1.0])
for h in L.sum(1):
    dist = np.convolve(dist, _hh_dist(6, int(h)))
obs = int(np.sum(L[:, 1:] * L[:, :-1]))
fig, ax = plt.subplots(figsize=(6.4, 2.8))
k = np.arange(len(dist)); sel = k <= 45
ax.bar(k[sel], dist[sel], color=[RED if kk >= obs else BLUE for kk in k[sel]], width=.85)
ax.axvline(obs, color=INK, lw=1.4); ax.text(obs + .4, dist.max() * .92, f"observed = {obs}", fontsize=8)
ax.axvline(N["registry"]["h2"]["HH_null_mean"], color=ORANGE, lw=1.4, ls="--")
ax.text(N["registry"]["h2"]["HH_null_mean"] + .4, dist.max() * .75, f"locked (turn×type) null mean = {N['registry']['h2']['HH_null_mean']:.1f}",
        fontsize=8, color=ORANGE)
ax.set_xlabel("H→H transitions under the within-dossier (count-preserving) null"); ax.set_ylabel("exact probability")
ax.set_title(f"H2: clustering is consistent with dossier difficulty (exact p = {N['registry']['within']['p_value_upper']:.3f})")
fig.savefig(FIG / "h2_null.png"); plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
