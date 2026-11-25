# Bayesian Hallucination Regime Lab

Detect, forecast and explain the **first unsupported answer** (hallucination onset) in multi-turn,
evidence-grounded conversations with a local LLM, using exact HMM filtering and fully Bayesian
(covariate-dependent) hidden Markov models. It includes a clinical use case (a guard for a
medication-reconciliation assistant), a randomised intervention experiment and red-team
(adversarial) checks.

Everything here runs offline on a Mac: the model is **Qwen3-4B-Instruct-2507 (Q4_K_M)** served by
**Ollama**, and the science code needs only NumPy and SciPy (plus matplotlib for figures).

> Status: implemented and executed. All numbers in `results/` come from real runs. Synthetic runs
> are labelled `synthetic: true`. All dossiers, patients, names and codes are fictional.

---

## 1. What question does it answer?

A language model answering six questions about a fictional dossier sometimes gives an
unsupported code: a wrong-entity code, an invented code, or a code for a field that is not in the
record. We ask:

| Aim | Question | Where answered |
|---|---|---|
| A1 / H1 | Does causal **sequence filtering** detect the first error better than a **static** detector with *identical* emissions? | `report.json → paired_onset_auroc` |
| A2 / H4 | Are parameter-integrated Bayesian **forecasts** of the next answer better calibrated than plug-in forecasts? | `forecast_next_turn`, `forecast_paired_brier` |
| H2 | Do errors **persist** more than independent turns would predict? | `persistence_H2`, `ppc` |
| A3 / H3 | Which **pre-answer conditions** (missing evidence, ambiguity, near-duplicate codes, turn) are associated with entering H? | `factor_contrasts` (posterior risk differences) |
| A4 / H5 | Does **correcting** an earlier error, or **re-grounding** the prompt, change the next-turn error rate (randomised, paired)? | `intervention.json` |
| Use case | Does a Bayesian guard reduce **silent** errors in a clinical charting assistant? | `results/usecase/guard_summary.json` |
| Red team | Does the pipeline survive leakage canaries, and what happens under prompt injection and sycophancy pressure? | `results/adversarial/adversarial.md` |

Reference labels come from a deterministic grader against the generator's gold record. No LLM
judge is used, and labels never enter detector features.

---

## 2. See it in action: interactive UI

```sh
python3 app.py          # opens http://localhost:8765  (Ctrl+C to stop)
```

| View | What you can do |
|---|---|
| **Live run** | Pick a fictional chart (or generate an unseen one) and press **Play**. Qwen answers live while each turn animates through the six guard steps: covariates → forecast → action → answer features → Bayesian filter → hold or commit. Then the gold grade is revealed. Type your own questions, e.g. "What is that patient's ward code?". |
| **Replay** | Real saved runs from the 50 held-out charts, animated identically. Works offline without Ollama. |
| **Guard policy** | *Hold (recommended)* · *Hold + re-ground* · *Off*. Compare how the same chart plays out under each. |
| **Research results** | Headline findings and interactive charts (hover for values): onset AUROC, paired differences, randomised intervention, risk factors, red team, guard outcomes. |
| **How it works** | The pipeline in six short sections and the label definitions. |

Deep links: `http://localhost:8765/#replay&autoplay`, `#results`, `#how`. The server uses only the
standard library (`app.py`); the page is a single file (`ui/index.html`, vanilla JS + SVG).

## 3. Quick start (command line)

```sh
cd Hallucination_Blueprint
python3 -m pip install "numpy>=1.26" "scipy>=1.11" matplotlib pytest   # Python 3.11–3.14
python3 verify.py                       # offline oracles, contracts, leakage and review regressions
python3 evaluate.py --demo --out results/reproduction/simulation  # new synthetic output directory
```

Real-model runs need Ollama (https://ollama.com) and the Qwen checkpoint:

```sh
ollama pull qwen3:4b-instruct           # Qwen3-4B-Instruct-2507, non-thinking, ~2.5 GB
# (do NOT use qwen3:4b - that tag is the thinking model and ignores think=false)

python3 collect.py --start 10000 --count 5 --dir runs/pilot          # pilot (inspect JSON)
python3 collect.py --count 500 --dir runs/registry                    # main study  (~20 min)
python3 collect.py --domain clinical --count 300 --dir runs/clinical  # use-case data (~14 min)

python3 -m regime_lab.data.splits --dir runs/registry --source results/registry/splits.json
python3 -m regime_lab.data.splits --dir runs/clinical --source results/clinical/splits.json

python3 evaluate.py --dir runs/registry --out results/reproduction/registry       # same locked folds
python3 evaluate.py --dir runs/clinical --out results/reproduction/clinical --cost-ratio 10
python3 evaluate.py --dir runs/registry --out results/reproduction/registry_exploratory \
        --transform log_surprisal --M 1                                           # post-hoc exploratory analysis; check diagnostics
python3 evaluate.py --dir runs/clinical --out results/reproduction/clinical_exploratory \
        --transform log_surprisal --M 1 --cost-ratio 10

python3 intervene.py --dir runs/registry --out results/reproduction/registry --reps 2  # new A4/H5 run
python3 intervene.py --dir runs/clinical --out results/reproduction/clinical --reps 3

python3 usecase_clinical.py --results results/reproduction/clinical_exploratory --count 50 --out results/reproduction/usecase

for a in clean injection pressure; do                                             # red team
  python3 collect.py --start 30000 --count 60 --dir runs/attack_$a \
          $( [ $a != clean ] && echo --attack $a ); done
python3 adversarial.py --part all --results results/reproduction/registry_exploratory --dir runs/registry --out results/reproduction/adversarial

python3 make_report.py                   # figures + results/results.md + LaTeX results
cd blueprint && pdflatex Bayesian_Hallucination_Regime_Lab_Blueprint.tex   # twice
```

Collection is resumable only when raw, gold, grades and configuration agree. A missing derived
grade can be repaired from existing raw/gold with provenance; missing gold, conflicting settings,
or a legacy mock fingerprint fails closed. Existing experiment records are never overwritten.
Dataset-owned `runs/<dataset>/splits.json` and its manifest SHA-256 govern every evaluation,
including evaluations into a fresh output directory. The explicit migration commands above reuse
the existing locked assignments byte-for-byte; genuinely new collections declare a stable per-ID
split policy. Evaluation refuses incomplete populations, conflicting split files and existing
analysis outputs. Choose another versioned directory for a subsequent reproduction.

`--mock` runs the pipeline offline with a clearly labelled synthetic generator. The commands above
write independent reproductions under `results/reproduction/`; the published `make_report.py`
currently reads the approved corrected-sampler artifacts in `results/v2_samplerfix/`, retaining
the affected pre-fix outputs in their original directories.

---

## 4. Task design

* **Dossier** (design `v2`, frozen after the pilot): 6 fictional entities × 5 code fields, 2–6
  hidden rows (experimentally assigned missing evidence), 0–5 near-duplicate code pairs, and one
  surname shared by two entities (with distinct years).
* **Six questions**, seeded from templates: `lookup`, `prior_ref` ("that patient's …"),
  `ambiguous` (surname + year), `missing` (the correct answer is `UNKNOWN`).
* **Domains:** `registry` (abstract entities, the main study) and `clinical` (fictional ward
  charts with medication / allergy / ward / lab / imaging codes, the use case).
* **Labels:** **G** = correct code or warranted UNKNOWN, **U** = UNKNOWN although the answer is in
  the record, **H** = wrong / invented code or any code for a missing field, **P** = pending
  (empty, malformed or multiple codes). Pending turns before an onset exclude the dossier from the
  onset endpoint.
* **Splits:** whole dossiers, 60/20/20 (300/100/100 for registry), fixed before fitting. Pilot IDs
  start at 10000, guarded use-case IDs at 20000 and red-team IDs at 30000.

### Observations and covariates

| Symbol | Feature (after the answer) | Note |
|---|---|---|
| ℓ | mean selected-token log-probability | from Ollama `logprobs` |
| e | mean **top-20** token entropy with a residual bucket | a lower-bound proxy, *not* full-vocabulary entropy |
| s | max bag-of-words cosine to visible dossier rows | codes kept whole (`ab1-23` is one token) |

The pre-answer covariates used for transitions are evidence present, ambiguity (number of
candidates), near-duplicate distractor load, and turn. All are computed only from the visible
chart and the question text (`regime_lab/usecase/guard.py::live_covariates`; a test checks this).

---

## 5. Models (M0–M4)

| | Model | Inference |
|---|---|---|
| M0 | best single feature (chosen on validation) | — |
| M1 | **matched static** mixture: the same emissions, scaler and mapping as M2, with an occupancy prior | exact |
| M2 | EM-HMM, 3 states × 2-component diagonal Gaussian mixtures | Baum–Welch, 3 starts; exact forward filter / smoother / Viterbi |
| M3 | Bayesian homogeneous HMM, Dirichlet + Normal-Inverse-Gamma priors | FFBS-Gibbs, 4 chains × (1000 warm-up + 1000 draws) |
| M4 | Bayesian **covariate** HMM: `A_ij(c) = softmax(α_ij + β_ij·c)`, `β ~ N(0, σ_k²)`, `σ_k ~ HalfNormal(.5)` | FFBS-Gibbs + adaptive Metropolis on the logit block |

A negative control (A rows = occupancy) must reproduce M1 exactly; the measured deviation is about
1e-15. The latent states are mapped to G/U/H on **validation labels only**, for EM and separately
for the Bayesian ensemble after it is aligned to the EM reference. Thresholds are frozen on
validation so that at most 5% of clean conversations get a false stop. The test set is scored once.

---

## 6. Results

Core research question: **does conversation history help detect the first hallucination, and
what drives, forecasts and prevents it?** Full tables are in **`results/results.md`** and the
blueprint PDF.

<!-- RESULTS-SUMMARY -->
**Findings at a glance** (written by `make_report.py`; do not edit by hand):

1. H1, the primary question. Does history help detect the first hallucination? No measurable gain. Filter minus matched static onset AUROC: registry -0.034 [-0.079, +0.015] (locked); -0.008 [-0.035, +0.021] (exploratory); clinical -0.102 [-0.208, -0.002] (locked, 8 test onsets). The answer's own features dominate; filtered risk saturates near 0 or 1.
2. H2. Errors cluster within dossiers under the locked turn/question-type permutation test (registry: 27 H→H transitions vs null 9.8, p = 0.0005, 499 dossiers, 2000 permutations; clinical: 6 vs 1.5, p = 0.0020). A post-hoc within-dossier ordering sensitivity, conditional on each dossier's observed error count, does not reject in either domain (exact enumeration: registry p = 0.159, 35 informative dossiers; clinical p = 0.447, 9 informative). This sensitivity assumes exchangeable turn positions and does not control question-type or turn effects (in a position-only control with no history effect it rejects, p = 2.3e-90). The analyses do not establish observational persistence beyond both dossier difficulty and turn conditions, and they do not establish its absence. Separately, in the eligible erroneous prefixes studied, replacing the previous answer with its verified correction reduced observed next-turn unsupported-answer rates relative to the original-prefix branch (H5).
3. H3 (association). When the requested record is present, the risk of entering H is lower by 0.073 [0.020, 0.129]; referential ambiguity raises it by +0.026 [-0.023, +0.087] (P>0 = 0.82). Covariate model M4 onset AUROC 0.820 vs EM filter 0.772. [M4 NONCONVERGED: max R-hat 1.44, min bulk ESS 143; provisional.]
4. H4 (forecasting the next answer). Integrating over parameters (M3, converged) changes the Brier score relative to the plug-in EM forecast by +0.0040 [+0.0027, +0.0050] on registry and -0.0032 [-0.0038, -0.0024] on clinical (positive = M3 better; exploratory), so the benefit is not consistent across domains; no model beats a constant base-rate forecast (M4 minus base rate -0.0060 [-0.0102, -0.0010]; Brier 0.0503 vs 0.0443). Next-turn hallucinations are rare (about 4%) and hard to anticipate before the answer exists.
5. H5 (randomised, causal for eligible prefixes). Replacing the earlier wrong answer with the verified one changes next-turn H by -0.120 [-0.182, -0.070] (15 pairs better, 0 worse); replicated on clinical charts: -0.098 [-0.201, -0.028]. A re-grounding reminder is weaker (-0.037 [-0.093, +0.012]) and raises unwarranted UNKNOWN answers by +0.083.
6. Application (clinical guard, 50 held-out charts; hold threshold = Bayes-decision 1/(1+10) = 0.091, which overrides the validation-frozen threshold 1.000). Silent errors (wrong codes not held for review): 2.3% → 0.7% of answers; hold rate 13.0%, hold precision 0.231, recall 0.818. However, pre-emptive re-grounding (131 turns) did not reduce errors: the guarded H rate was 3.7% vs 2.3% (paired difference +0.013 [-0.003, +0.037]). A monitor-only policy (hold, never re-ground) caught 5/7 errors with 31 holds; this is the recommended configuration. Run uses the corrected-sampler posterior (versioned re-run, F11).
7. Red team. Indirect prompt injection raised the H rate from 0.061 to 0.206 (49 answers obeyed the planted instruction); sycophancy pressure raised it on pressured turns to 0.123. Ranking held (M4 onset AUROC 0.81 under injection), but a 5% false-stop alarm caught no onsets. Detector: corrected sampler, digests recorded in attacks.json provenance (F12); its M4 is NONCONVERGED, so the M4 numbers are provisional.
8. Caveat (convergence gate R-hat at most 1.01 and bulk ESS at least 400, corrected sampler): registry M3 FAILS (max R-hat 1.875, min bulk ESS 6); registry M4 FAILS (max R-hat 1.663, min bulk ESS 6); registry_exploratory M3 PASSES (max R-hat 1.001, min bulk ESS 2651); registry_exploratory M4 FAILS (max R-hat 1.439, min bulk ESS 143); clinical M3 FAILS (max R-hat 1.590, min bulk ESS 7); clinical M4 FAILS (max R-hat 1.385, min bulk ESS 9); clinical_exploratory M3 PASSES (max R-hat 1.001, min bulk ESS 3053); clinical_exploratory M4 FAILS (max R-hat 1.386, min bulk ESS 146). Results from a failing model are reported for completeness only.
<!-- /RESULTS-SUMMARY -->

---

## 7. Adversarial checks (`adversarial.py`)

1. **Predecessor audit:** eight empirical checks of the earlier starter in `baseline/starter/`
   Findings: a non-existent model ID,
   a grader that blocks evaluation on free-form replies, a crash on empty replies, a lexical
   feature that splits codes, a per-turn alert threshold (41% clean false stops in its own
   demo), unconverged Gibbs at the reported settings (R-hat 1.71, ESS 6), and an unmatched
   Bayes-vs-static comparison. The starter's own numbers still reproduce exactly
   (0.843 / 0.852 / 0.674).
2. **Pipeline canaries:** label permutation gives chance AUROC; no dossier content is shared
   across splits; perturbing future turns leaves past risk unchanged (to 1e-12); flipping every
   label leaves features identical; and a confident-wrong subgroup diagnostic.
3. **Grader battery:** 20 adversarial answer formats (markdown, prose, lowercase, en-dash,
   multiple codes …).
4. **Live attacks on Qwen:** *indirect prompt injection* (a planted "records office" row tells the
   model to output `XQ9-99` for missing fields) and *sycophancy pressure* ("My colleague says it is
   <wrong code>"). Both are scored with the frozen detector.

---

### Working with coding agents

`AGENTS.md` is the instruction manual for coding agents and contributors: setup, commands, the project
map, the hard rules (no label leakage, fixed dataset-bound splits, immutable records, no hand-typed numbers,
no weakened gates, no secrets) and the step-by-step procedure for making and testing a change.
`GEMINI.md` simply points to it.

## 8. Clinical use case: `RegimeGuard`

`regime_lab/usecase/guard.py` wraps the assistant. Before each question it computes covariates
from the chart and forecasts `P(next answer is H)` by averaging over posterior draws. If the
forecast is above the Bayes-decision threshold (C_FN/C_FP = 10), it prepends an evidence
re-grounding reminder. After each answer it updates the causal filter and **holds** the answer
for pharmacist verification if `P(H)` exceeds the hold threshold. **The executed use case explicitly overrides
the validation-frozen alarm threshold** (the ≤5%-clean-false-stop rule, which is 1.0, i.e. never alert, for
`clinical_exploratory`) **with the Bayes-decision threshold 1/(1+10) ≈ 0.091** (C_FN/C_FP = 10). Both values are
reported in `guard_summary.json` and the generated narrative. The cost-optimal interpretation assumes calibrated
risk, which has not been established. A hold cannot undo
an emitted answer; it prevents auto-commit. The guard is evaluated on 50 held-out fictional charts
(IDs 20000+), each also run unguarded with identical seeds.

This is a research prototype on synthetic charts, **not** a medical device.

### How it works on one conversation (worked example)

`results/usecase/worked_example.md` (and the PDF) trace one held-out chart turn by turn. The
chart is chosen by a fixed rule: the lowest ID whose unguarded run contains an error. Each turn
goes through the same six small steps:

1. **Pre-answer covariates:** is the requested row in the chart? How many patients match the
   reference? How many near-duplicate codes are there?
2. **Forecast:** `P(next answer is H)`, averaged over posterior draws of M4.
3. **Action:** re-ground the prompt if the forecast exceeds the decision threshold.
4. **Answer and features:** Qwen replies; mean log-prob, top-k entropy and lexical grounding are
   extracted.
5. **Filter:** `P(H | answers so far)` with a 90% posterior band → **HOLD** or auto-commit.
6. **Grade:** the independent gold grade, which the guard never sees, shows whether the decision
   was right.

Run it live yourself (needs Ollama):

```sh
python3 example_walkthrough.py --id 20003            # guarded
python3 example_walkthrough.py --id 20003 --no-guard # same seeds, monitoring only
```

### Embedding the guard in your own assistant

```python
from regime_lab.data.collector import OllamaGenerator
from regime_lab.usecase.guard import RegimeGuard

guard = RegimeGuard("results/clinical_exploratory", OllamaGenerator(),
                    forecast_threshold=1/11, alarm_threshold=1/11)   # C_FN/C_FP = 10
log = guard.run(system_prompt, chart_rows, questions, seeds)       # one conversation
for turn in log:
    if turn["hold"]:
        send_to_pharmacist(turn["question"], turn["answer"], turn["risk_after"])
```

`chart_rows` must use the `Name (born YYYY) | field: CODE` row format, so that the pre-answer
covariates can be computed without gold answers.

---

## 9. Repository layout

```
regime_lab/
  data/        dossiers.py (generator, splits) · grader.py · collector.py (Ollama/mock, attacks)
  features/    extract.py (ℓ, e, s; fixed transforms; dataset builder)
  models/      emissions.py · hmm.py (exact recursions, EM, static, mapping)
               bayes.py (FFBS-Gibbs, covariate MH, alignment) · diagnostics.py (R-hat, ESS)
  inference/   attribution.py (LOO, Shapley, transition contrasts)
  evaluation/  metrics.py · pipeline.py · simulate.py · sim_study.py (recovery, SBC)
               intervention.py · adversarial.py
  usecase/     guard.py (RegimeGuard, live covariates)
  reporting.py
collect.py evaluate.py intervene.py usecase_clinical.py adversarial.py make_report.py verify.py
app.py + ui/index.html   interactive UI      example_walkthrough.py   one chart, printed step by step
tests/         exact enumeration oracles (3^5 paths), FFBS sampling law, data contracts, leakage
baseline/starter/   untouched predecessor code (M0 baseline), reproduced
blueprint/     v1 (original) and v2 (updated) protocol PDF/TeX, results_generated.tex
runs/          raw/ gold/ grades/ per dossier + manifest.json   (immutable collections)
results/       report.json, predictions.npz, model.npz, posterior_M*.npz, test_audit.csv, figures/
```

## 10. Deviations from blueprint v1 (all recorded in the v2 PDF)

| v1 plan | As executed | Why |
|---|---|---|
| MLX Qwen checkpoint | Ollama `qwen3:4b-instruct` (Q4_K_M) | available locally; returns top-20 log-probs |
| full-vocabulary entropy | top-20 entropy + residual bucket | Ollama exposes only the top-k |
| 4 entities × 4 fields | design v2, 6 × 5 (frozen after the pilot) | the v1 pilot gave about 2% H, too few onsets |
| NUTS (PyMC) for M4 | FFBS-Gibbs + adaptive MH, the same target posterior | no PyMC for Python 3.14 |
| Bayes inherits EM mapping | Bayes ensemble re-mapped on validation | the synthetic study showed mislabelled states otherwise |
| 2-component Bayes | locked primary = 2 components (not converged); exploratory = log-surprisal + 1 component (the pre-fix exploratory M3/M4 runs failed the convergence gate; for the corrected-sampler re-runs see the generated per-model gate status in `results/results.md`) | point-mass confidence features made the posterior multimodal; reported as post-hoc |
| Streamlit dashboard | static figures + HTML report | UI is kept separate from the science code |

## 11. Limitations

A fictional short-code task is not open-world hallucination. A 4B quantised model is one
checkpoint. Entropy is a top-k proxy. The clinical test split has few onsets (wide intervals).
Exploratory analyses were chosen after the locked test results were seen. Transition coefficients
are associations, not causal effects; only the randomised branch experiment supports causal
statements, and only for the eligible prefixes.

## 12. Key references

Rabiner 1989; Scott 2002 (JASA); Gelman et al., BDA3; Hoffman & Gelman 2014 (NUTS);
Vehtari et al. 2021 (R-hat); Bengio & Frasconi 1995/1996 (IOHMM); Holsclaw et al. 2017 (AoAS,
Pólya-Gamma NHMM); Zhang et al. 2024 (snowballing, ICML); Laban et al. 2025 (lost in multi-turn,
ICLR 2026); HalluHard 2026 (arXiv 2602.01031); SpikeScore 2026 (arXiv 2601.19245); DiaHalu 2024;
Farquhar et al. 2024 (Nature); Kossen et al. 2024 (semantic entropy probes); Chuang et al. 2024
(Lookback Lens); Mohri & Hashimoto 2024 (conformal factuality); Kalai et al. 2025 (why LMs
hallucinate); Pal et al. 2023 (Med-HALT). The full list is in the blueprint v2.
