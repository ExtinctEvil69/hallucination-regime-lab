# Detecting hallucination onset in multi-turn QA

Can previous turns help flag the first unsupported answer in a conversation? This project fits a three-state Gaussian-mixture hidden Markov model and compares causal alerts with two single-turn detectors. A completed feature simulation is included in example/. The local Qwen experiment is a separate run.

## Quick start

Use Python 3.11 or 3.12. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "numpy>=1.24,<3"
python verify.py
python evaluate.py --demo --out simulation
```

## Collect 500 Qwen conversations on Apple silicon

Use a native arm64 Python on your Mac, with internet for the initial model download. Close other memory-heavy apps and time the pilot on your hardware. The collector runs one conversation at a time. Use one writer per directory.

```sh
python -m pip install mlx-lm
python -m pip freeze > environment.txt
time python collect.py --count 5 --dir rollouts
python collect.py --count 500 --dir rollouts
python evaluate.py --dir rollouts --out results
python collect.py --start 10000 --count 5 --dir guarded-demo --guard results/guard.npz
```

Inspect the five pilot JSONs for decoded answers, finite features and sensible labels. Each finished conversation is saved atomically. Rerunning with the same settings skips completed IDs. For exact replication, download a fixed Hugging Face revision and pass its local path via --model on every collection invocation. Runtime and peak memory for the full run have not been measured here.

## Task and independent labels

Each of 500 seeded dossiers contains four fictional units with four code fields. Six questions refer to specific units or refer back to the last question. Missing-field prompts test abstention. The model sees the records and its own earlier replies. Ground truth comes from the original record. The expected answer, label and resolved unit are never detector inputs.

A correct code or justified UNKNOWN is Grounded. An explicit UNKNOWN for an answerable question is Uncertain. A wrong code or a code invented for a missing field is Hallucinating. These operational labels concern source fidelity. The middle label measures expressed abstention, not inaccessible model beliefs. Free-form replies receive null. Before evaluation review nulls against the source, assign one of the three label strings in the JSON, and retain the original answer. Use a second reviewer for ambiguous replies. Flag unsupported factual claims as Hallucinating, clear hedges without a claim as Uncertain, and supported answers as Grounded. Document disagreements. A pilot may yield too few errors or abstentions. Report class counts and widen the prospective collection if needed. Keep the test set untouched when changing task difficulty.

## Observations

At each completed turn collect mean sampled-token log probability, mean full-vocabulary predictive entropy in nats, and maximum bag-of-words cosine between the answer and any available record. The cosine uses word counts and access only to visible evidence. It can miss errors when a code occurs elsewhere in the dossier. Sampling uses temperature 0.7. Log probabilities describe the model distribution before temperature sampling. Stop tokens are excluded. A 32-token cap bounds each turn. This is a controlled source-grounding task with lexical retrieval and short answers. Generalization to long-form prose or other generators needs a new evaluation.

## Model and learning

Let b_j(x) be the sum of two diagonal Gaussian densities weighted by learned component probabilities in state j. All features are standardized on training turns. The causal update is

    p_t(j) ∝ b_j(x_t) Σ_i p_(t−1)(i) A_ij

At turn one use the initial state distribution. Normalization gives the exact posterior under the fitted model. Call HMM.step(x, previous) for an incremental update on standardized features. Each conversation resets the prior. In intervention mode the collector stops before asking the next question when risk reaches the validation threshold. An alert can only follow the answer whose features it uses.

For EM, forward and backward messages give state occupancy γ_t(j) and expected transitions ξ_t(i,j). The component responsibility is γ_t(j) c_jm N_jm(x_t)/b_j(x_t). Normalize expected first-state counts for π, transition counts by row for A, and component mass within each state for mixture weights. Weighted first and second moments update means and diagonal variances. A variance floor and tiny mass regularizers stabilize estimation. Three seeded initializations compete on training likelihood. The EM trace and fitted matrix are saved. Viterbi replaces the forward sum by a maximum and backtracks the best entire state path. It may revise earlier turns as later evidence arrives.

## Held-out experiment

Split whole dossiers 60/20/20 using a fixed seed. Fit EM with no training labels. Validation labels map the arbitrary latent indices onto semantic names by maximizing class-balanced posterior agreement over six permutations. This naming step requires labeled examples of all three classes. It does not guarantee identifiable regimes. Inspect decoded paths and labeled examples for semantic mismatch.

The primary outcome is **first onset**. Eligible turns run from the start through the first Hallucinating label inclusive. For an error-free conversation every turn is a negative. Continued error turns cannot inflate the onset metric. Rank each eligible turn by filtered state probability. For comparison use (1) the identical learned GMM emissions and a fixed average training-state prior and (2) the best validation-ranked individual feature with its prespecified risk direction. The emission-only comparison isolates the value of history. The score uses the current answer, so the metric measures detection, not advance prediction.

report.json gives all three test AUROCs, counts and paired percentile intervals for differences from 500 resamples of complete test trajectories. Undefined one-class resamples are omitted and counted through the valid-replicate field. A single-class test gives null AUROC. The separate alert threshold is set just above the validation 95th percentile of negative at-risk scores. Test sensitivity and false-alarm rate apply per eligible turn in uninterrupted replay. A deployed halt can censor later turns. test_audit.csv supplies per-turn online risk, alerts, reference labels and retrospective Viterbi states. splits.json preserves trajectory IDs. guard.npz saves fitted parameters and preprocessing.

## Interpreting the included simulation

The seeded simulation draws 180 independent ten-turn sequences from a persistent latent chain with overlapping Gaussian signals. Its feature values are illustrative continuous proxies. On 36 held-out sequences there were 19 first onsets across 282 eligible turns. AUROCs were 0.843 for filtering, 0.852 for the static GMM and 0.674 for the single-feature baseline. The paired 95% interval for filter minus static GMM was approximately [−0.040, 0.020]. Persistence can help identify established regimes while delaying a response to a sudden change. This run makes no claim about Qwen performance. For a course report, disclose both comparisons, rarity of first errors, and any uncertainty from manual labels. A fitted diagonal transition probability alone cannot establish that earlier errors cause later ones.

## References

- Zhang et al., How Language Model Hallucinations Can Snowball (ICML 2024), https://proceedings.mlr.press/v235/zhang24ay.html
- MLX LM, https://github.com/ml-explore/mlx-lm
- Quantized checkpoint, https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit
