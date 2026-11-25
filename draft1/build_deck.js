// draft1.pptx — 15-minute talk. All numbers from numbers.json (generated from saved artifacts).
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
// Optional: path to a theme-writer module exporting applyTheme(file, THEME); without it the deck keeps default theme colours.
const THEME_SCRIPT = process.env.PPTX_THEME_SCRIPT;
const applyTheme = THEME_SCRIPT ? require(THEME_SCRIPT).applyTheme : async () => {};
const N = JSON.parse(fs.readFileSync(path.join(__dirname, "numbers.json"), "utf8"));
const F = p => path.join(__dirname, "figures", p);
const R = p => path.join(__dirname, "..", "results", "v2_samplerfix", p);
const f3 = x => x.toFixed(3), f1 = x => x.toFixed(1), pct = x => (100 * x).toFixed(1) + "%";
const sg = x => (x >= 0 ? "+" : "") + x.toFixed(3);

const THEME = { name: "Hallucination Regime Lab", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2430", lt1: "FFFFFF", dk2: "15334D", lt2: "EEF3F6", accent1: "087F83", accent2: "E4572E",
            accent3: "2E9E5B", accent4: "3A7BD5", accent5: "F2A900", accent6: "6A4CE2", hlink: "087F83", folHlink: "6A4CE2" } };
const HEX = { navy: "15334D", teal: "087F83", coral: "E4572E", green: "2E9E5B", blue: "3A7BD5", gold: "F2A900", ink: "1B2430", ink2: "5B6675", light: "EEF3F6", grid: "D9E1E8" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5 in
pres.title = "Hallucination Regime Lab — draft1"; pres.author = "Nihar Narsana"; pres.company = "University of Arizona";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({ title: "DARK", background: { color: HEX.navy },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.3, w: 11.9, h: 1.7, fontSize: 40, bold: true, color: C.background1, fontFace: "Cambria", valign: "bottom", align: "left" }, text: "" } },
            { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.2, w: 11.9, h: 3.6, fontSize: 20, color: "CADCFC", valign: "top" }, text: "" } }] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: "FFFFFF" },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 28, bold: true, color: C.text2, fontFace: "Cambria", valign: "middle" }, text: "" } }],
  slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "8A94A3" } });

const notes = {};
let sc = 0;
function slide(master, section, title) {
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  sc++; return s;
}
const tb = (s, text, o) => s.addText(text, { isTextBox: true, fontFace: "Calibri", color: HEX.ink, fontSize: 15, valign: "top", margin: 0.05, ...o });
function stat(s, x, y, w, big, label, color, bigSize) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.62, rectRadius: 0.12, fill: { color: HEX.light }, line: { color: HEX.light }, objectName: "stat " + label });
  tb(s, big, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.8, fontSize: bigSize || 34, bold: true, color: color || HEX.navy, fontFace: "Cambria" });
  tb(s, label, { x: x + 0.2, y: y + 0.88, w: w - 0.4, h: 0.72, fontSize: 12.5, color: HEX.ink2 });
}
function card(s, x, y, w, h, head, body, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: "FFFFFF" }, line: { color: HEX.grid, width: 1 },
    shadow: { type: "outer", color: "9AA8B6", opacity: 0.25, blur: 4, offset: 1.5, angle: 90 }, objectName: "card " + head });
  s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: y + 0.22, w: 0.42, h: 0.42, fill: { color: color }, line: { color: color } });
  tb(s, head, { x: x + 0.78, y: y + 0.18, w: w - 0.95, h: 0.5, fontSize: 16, bold: true, color: HEX.navy, fontFace: "Cambria" });
  tb(s, body, { x: x + 0.25, y: y + 0.78, w: w - 0.45, h: h - 0.9, fontSize: 13.5, color: HEX.ink });
}
const chartTxt = { catAxisLabelColor: HEX.ink2, valAxisLabelColor: HEX.ink2, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", titleFontFace: "+mj-lt", legendFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  dataLabelFontSize: 12, dataLabelColor: HEX.ink, valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" } };

// 1 ── title
pres.addSection({ title: "Opening" });
let s = slide("DARK", "Opening", "Does conversation history help catch the first hallucination?");
s.addText("A Bayesian hidden-Markov study of multi-turn LLM answers, with a clinical guard\nNihar Narsana · MS Data Science, Bayesian Inference and Modelling · University of Arizona", { placeholder: "body" });
s.addNotes("Open with the question. A model answering six questions about a chart sometimes gives an unsupported code and then builds on it. We test whether a hidden Markov model that tracks a latent 'hallucinating' state over turns catches the FIRST unsupported answer better than looking at each answer alone. Spoiler: it does not, but the Bayesian analysis tells us what does help.");

// 2 ── answer first
s = slide("CONTENT", "Opening", "The short answer, up front");
stat(s, 0.6, 1.5, 3.9, sg(N.registry.primary.estimate), "history vs matched static (onset AUROC); 95% CI " + sg(N.registry.primary.lo) + " to " + sg(N.registry.primary.hi), HEX.navy);
stat(s, 4.72, 1.5, 3.9, (100 * N.registry.iv.corr.risk_difference_H).toFixed(1) + " pts", "next-turn hallucination when an earlier error is corrected (randomised)", HEX.green);
stat(s, 8.84, 1.5, 3.9, pct(N.guard.unguarded.silent_error_rate) + " → " + pct(N.guard.guarded.silent_error_rate), "silent errors with the Bayesian guard (held-out clinical charts)", HEX.coral);
tb(s, [{ text: "History does not measurably help detection: ", options: { bold: true } },
       { text: "each answer's own signal dominates.", options: { breakLine: true } },
       { text: "What helps: ", options: { bold: true } },
       { text: "knowing before the answer whether the evidence exists, correcting errors so they do not snowball, and holding risky answers for review." }],
   { x: 0.6, y: 3.5, w: 12.1, h: 1.6, fontSize: 18 });
s.addImage({ path: F("pipeline.png"), x: 0.6, y: 5.15, w: 12.1, h: 2.0, sizing: { type: "contain", w: 12.1, h: 2.0 } });
s.addNotes("Three numbers carry the talk. The primary estimand, history-based filter minus a matched static detector, is essentially zero. The randomised correction experiment shows errors do snowball and correction stops it. And the application, a Bayesian guard, cuts silent errors on held-out charts.");

// 3 ── task & data
pres.addSection({ title: "Study design" });
s = slide("CONTENT", "Study design", "Task: six questions per fictional chart, gold-graded");
const dr = N.data_registry, dc = N.data_clinical;
stat(s, 0.6, 1.45, 2.9, String(dr.dossiers + dc.dossiers), "fictional dossiers (500 registry, 300 clinical)");
stat(s, 0.6, 3.15, 2.9, String(dr.turns + dc.turns), "answers from Qwen3-4B-Instruct, graded deterministically");
stat(s, 0.6, 4.85, 2.9, f1(dr.Hpct) + "% / " + f1(dc.Hpct) + "%", "hallucination rate (registry / clinical)", HEX.coral, 24);
s.addImage({ path: F("ui_live.png"), x: 3.8, y: 1.45, w: 8.9, h: 5.0, sizing: { type: "contain", w: 8.9, h: 5.0 } });
tb(s, "Labels: G grounded · U unwarranted UNKNOWN · H wrong / invented / unsupported code. No LLM judge.", { x: 3.8, y: 6.55, w: 8.9, h: 0.5, fontSize: 13, color: HEX.ink2 });
s.addNotes("Each dossier has 6 fictional entities and 5 code fields, with 2 to 6 rows hidden. Questions include lookups, 'that patient' references, surname plus year, and missing fields where the right answer is UNKNOWN. The first 4x4 design gave almost no errors, so the harder design was frozen after the pilot and before the main collection. Most errors are real codes from the wrong row, not invented ones.");

// 4 ── the model
s = slide("CONTENT", "Study design", "The model: a hidden state that evolves over turns");
card(s, 0.6, 1.45, 3.9, 2.4, "Hidden states", "G grounded · U uncertain · H hallucinating. Never observed directly.", HEX.green);
card(s, 0.6, 4.05, 3.9, 2.9, "Transitions A(c)", "Softmax of covariates known before the answer: evidence present, matching patients, look-alike codes, turn.", HEX.blue);
card(s, 4.72, 1.45, 3.9, 2.4, "Emissions b(x)", "Gaussian mixtures over log-prob, top-20 entropy and lexical grounding of the answer.", HEX.gold);
card(s, 4.72, 4.05, 3.9, 2.9, "Causal filter", "f(t) ∝ b(x_t) · f(t−1)A(c_t). Uses only answers so far, so it runs live.", HEX.coral);
s.addImage({ path: F("demo_markov_chain_final.png"), x: 8.84, y: 1.45, w: 3.9, h: 5.5, sizing: { type: "cover", w: 3.9, h: 5.5 } });
s.addNotes("Models compared: M0 best single feature; M1 a matched static mixture with identical emissions; M2 EM-HMM; M3 a Bayesian homogeneous HMM; M4 a Bayesian covariate-dependent HMM. The matched static model is the key comparator: same emissions, scaler and state mapping, so any difference is due to history alone. A negative control reproduces it to 1e-15.");

// 5 ── priors
pres.addSection({ title: "Bayesian method" });
s = slide("CONTENT", "Bayesian method", "The major priors, and why each was chosen");
s.addImage({ path: F("priors.png"), x: 0.6, y: 1.35, w: 7.9, h: 4.25, sizing: { type: "contain", w: 7.9, h: 4.25 } });
tb(s, [{ text: "Dirichlet(1,1,1)", options: { bold: true, bullet: true, breakLine: true } }, { text: "on π and transition rows: neutral about persistence", options: { breakLine: true } },
       { text: "Sticky κ = 2, 5", options: { bold: true, bullet: true, breakLine: true } }, { text: "only as sensitivity: a sticky prior can manufacture persistence", options: { breakLine: true } },
       { text: "InvGamma(3,2), N(0, v/0.5)", options: { bold: true, bullet: true, breakLine: true } }, { text: "weakly informative emissions on standardised features", options: { breakLine: true } },
       { text: "α ~ N(0,1.5²), β ~ N(0,σ²), σ ~ HalfNormal(0.5)", options: { bold: true, bullet: true, breakLine: true } }, { text: "hierarchical shrinkage of covariate effects" }],
   { x: 8.75, y: 1.45, w: 4.0, h: 5.4, fontSize: 14, paraSpaceAfter: 6 });
s.addNotes("Walk through the panels. Default Dirichlet(1) priors do not favour persistence; the sticky versions are run as a sensitivity analysis precisely because they could create the effect we are testing. InverseGamma(3,2) has mean 1 on the standardised scale and keeps variances away from point masses. For the covariate model, the half-normal scale shrinks unsupported effects toward zero. Bottom right is a prior predictive check: the implied prior on P(next state = H) is broad.");

// 6 ── inference techniques
s = slide("CONTENT", "Bayesian method", "Bayesian inference techniques used");
const tech = [["FFBS-Gibbs", "Forward-filter, backward-sample whole paths per dossier, then conjugate updates.", HEX.teal],
  ["Conjugate updates", "Dirichlet for π, A, w; Normal-Inverse-Gamma for means and variances.", HEX.green],
  ["MH within Gibbs", "Covariate logits via Laplace-shaped random-walk MH; MH on log σ with Jacobian.", HEX.blue],
  ["Label switching", "Align draws to the EM reference; map G/U/H on validation labels only.", HEX.gold],
  ["Posterior prediction", "Average filtered risk over 200 draws; importance-reweighted variant.", HEX.coral],
  ["Trust checks", "Split R-hat, bulk/tail ESS, SBC, posterior predictive checks, exact enumeration tests.", "6A4CE2"]];
tech.forEach((t, i) => card(s, 0.6 + (i % 3) * 4.12, 1.45 + Math.floor(i / 3) * 2.85, 3.92, 2.6, t[0], t[1], t[2]));
s.addNotes("EM gives a point estimate; the Bayesian models integrate over parameter uncertainty. FFBS samples entire hidden paths jointly, which mixes far better than updating one state at a time. Everything conjugate is sampled exactly; only the covariate logits need Metropolis-Hastings. NUTS was not available on Python 3.14, but the target posterior is the same. Decisions use the Bayes rule: hold when P(H) exceeds C_FP/(C_FP+C_FN) = 1/11.");

// 7 ── demo video (conversation)
pres.addSection({ title: "Demo" });
s = slide("CONTENT", "Demo", "Demo: the guard working on a held-out chart");
s.addImage({ path: F("demo_conversation.gif"), x: 0.6, y: 1.3, w: 8.6, h: 5.85, sizing: { type: "contain", w: 8.6, h: 5.85 }, altText: "Animated recording of the guard replaying chart 20003" });
tb(s, [{ text: "Turn 1: ", options: { bold: true } }, { text: "a confident wrong code (another field's code) → P(H) = 1.00 → held. Grade: H.", options: { breakLine: true } },
       { text: "Turn 2: ", options: { bold: true } }, { text: "the requested row is missing → correct UNKNOWN.", options: { breakLine: true } },
       { text: "Turns 3–6: ", options: { bold: true } }, { text: "grounded answers auto-commit." }],
   { x: 9.45, y: 1.45, w: 3.3, h: 4.5, fontSize: 14.5, paraSpaceAfter: 8 });
tb(s, "Recorded from the real UI (python3 app.py). The animation plays in Slide Show.", { x: 9.45, y: 6.2, w: 3.3, h: 0.9, fontSize: 11, color: HEX.ink2 });
s.addNotes("This is a recording of the real interface replaying a saved run. Watch step 1 read the question, step 2 forecast the risk, then the answer arrive, the filter update, and the hold decision. On turn 1 Qwen confidently gives VE6-33, which is the patient's allergy code, not the imaging code asked for. The guard holds it, and the gold grade confirms it was a hallucination.");

// 8 ── markov chain video
s = slide("CONTENT", "Demo", "The hidden Markov chain, built live");
s.addImage({ path: F("demo_markov_chain.gif"), x: 0.6, y: 1.3, w: 12.1, h: 5.2, sizing: { type: "contain", w: 12.1, h: 5.2 }, altText: "Animated hidden Markov chain trellis being built turn by turn" });
tb(s, "Dashed rings = prior from the transitions; filled circles = posterior after the answer's evidence; edge width = transition mass.", { x: 0.6, y: 6.6, w: 12.1, h: 0.5, fontSize: 13, color: HEX.ink2 });
s.addNotes("The same run seen as the chain itself. Each column is a turn. Covariates on top set the transitions; the previous belief flows through them to the dashed prior rings; the observed answer at the bottom supplies the evidence; filled circles are the posterior. Turn 2 asks for a missing row, so the mass flows into U, and the UNKNOWN answer confirms it.");

// 9 ── worked example table
s = slide("CONTENT", "Demo", "Worked example: the Bayes update, turn by turn");
const we = fs.readFileSync(path.join(__dirname, "worked_example.tex"), "utf8").split("\n").filter(l => /^\d+ &/.test(l));
const rows = [["Turn", "Question", "Prior G/U/H", "Answer", "Evidence G/U/H", "Posterior G/U/H", "Guard", "Grade"].map(t => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy } } }))];
we.forEach(l => { const c = l.replace(/\\\\$/, "").split(" & ").map(x => x.replace(/\\texttt\{(.*)\}/, "$1").replace(/\\_/g, "_").replace(/\\\\/g, "").trim());
  rows.push([c[0], c[1], c[3], c[4], c[5], c[6], c[7], c[8]].map((t, j) => ({ text: t, options: { color: (j === 7 && t === "H") || (j === 6 && t === "hold") ? HEX.coral : HEX.ink, bold: (j === 7 && t === "H") || (j === 6 && t === "hold") } }))); });
s.addTable(rows, { x: 0.6, y: 1.4, w: 12.1, colW: [0.7, 4.0, 1.35, 1.25, 1.55, 1.6, 0.85, 0.8], fontSize: 12.5, fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: HEX.grid }, rowH: 0.48, valign: "middle" });
tb(s, "Same code VE6-33: wrong on turn 1 (imaging asked, allergy code given) → H; correct on turn 5 (allergy asked) → G. The verdict follows the evidence, not the string.",
   { x: 0.6, y: 5.25, w: 12.1, h: 0.9, fontSize: 15, italic: true, color: HEX.navy });
s.addNotes("Read across a row: prior from the transitions, the answer, the evidence it provides, and the posterior. On turn 1 the prior gave H only 0.22, but the answer's features are strong evidence for H, so the posterior jumps to 1.00 and the guard holds it. These numbers are recomputed from the saved corrected run.");

// 10 ── H1 chart
pres.addSection({ title: "Results" });
s = slide("CONTENT", "Results", "H1: history does not improve onset detection");
const au = N.registry_exploratory.auroc, order = [["M0 single feature", "M0_single_feature"], ["M1 matched static", "M1_static"], ["M2 EM filter", "M2_em_filter"], ["M3 Bayes HMM", "M3_bayes_filter"], ["M4 covariate*", "M4_cov_bayes_filter"]];
s.addChart(pres.charts.BAR, [{ name: "Onset AUROC", labels: order.map(o => o[0]), values: order.map(o => +au[o[1]].est.toFixed(3)) }],
  { x: 0.6, y: 1.35, w: 7.6, h: 5.3, barDir: "bar", chartColors: [HEX.teal], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    valAxisMinVal: 0.5, valAxisMaxVal: 0.9, showLegend: false, showTitle: true, title: "First-onset AUROC, registry test set (0.5 = chance)", titleFontSize: 14, titleColor: HEX.navy, ...chartTxt });
stat(s, 8.5, 1.45, 4.2, sg(N.registry.primary.estimate), "M2 − M1 paired difference, registry (locked); 95% CI " + sg(N.registry.primary.lo) + " to " + sg(N.registry.primary.hi));
stat(s, 8.5, 3.2, 4.2, sg(N.clinical.primary.estimate), "clinical (locked), only " + N.clinical.onsets + " test onsets; CI " + sg(N.clinical.primary.lo) + " to " + sg(N.clinical.primary.hi));
tb(s, "* M4 fails the convergence gate (R-hat " + N.registry_exploratory.conv.M4.rhat.toFixed(2) + "): provisional. Each answer's own features dominate; the filtered risk saturates near 0 or 1.", { x: 8.5, y: 4.95, w: 4.2, h: 1.6, fontSize: 13.5, color: HEX.ink2 });
s.addNotes("The paired, matched comparison is the primary estimand and it is null on registry data. On clinical data the static model is even slightly better, with few onsets. The covariate model M4 looks best, but it does not converge, so we label it provisional rather than claim it.");

// 11 ── H2 / H3
s = slide("CONTENT", "Results", "H2–H3: errors cluster; missing evidence drives risk");
s.addImage({ path: F("h2_null.png"), x: 0.6, y: 1.35, w: 6.6, h: 3.4, sizing: { type: "contain", w: 6.6, h: 3.4 } });
tb(s, "Locked test: clustering, p = " + N.registry.h2.p_value.toFixed(4) + ". An exact count-preserving null does not reject (p = " + N.registry.within.p_value_upper.toFixed(3) + "), so the clustering is consistent with harder dossiers. Persistence beyond difficulty is neither shown nor ruled out.", { x: 0.6, y: 4.9, w: 6.6, h: 1.9, fontSize: 14.5 });
const ev = N.registry_exploratory.factor.evidence, am = N.registry_exploratory.factor.ambiguity;
stat(s, 7.55, 1.45, 5.15, sg(ev.median), "change in P(enter H) when the requested record is present; 95% CrI " + sg(ev.lo) + " to " + sg(ev.hi), HEX.green);
stat(s, 7.55, 3.2, 5.15, sg(am.median), "referential ambiguity (P > 0 = " + am.p_gt0.toFixed(2) + "); 95% CrI " + sg(am.lo) + " to " + sg(am.hi), HEX.gold);
tb(s, "These are associations from M4 (not yet converged), not causal effects.", { x: 7.55, y: 4.95, w: 5.15, h: 0.9, fontSize: 13.5, color: HEX.ink2 });
s.addNotes("This slide shows the review process working. The first test said errors persist. The reviewers showed that dossier difficulty alone can produce that result, and under an exact count-preserving null it does not hold up. So we report clustering, not proven persistence. The strongest risk factor is simply whether the requested record exists.");

// 12 ── H5 intervention chart
s = slide("CONTENT", "Results", "H5: correcting an error stops the snowball");
const ar = N.registry.iv.arms, ac = N.clinical.iv.arms;
s.addChart(pres.charts.BAR, [{ name: "Registry", labels: ["keep the wrong answer", "verified correction", "re-ground reminder"], values: [ar.original.H_rate, ar.corrected.H_rate, ar.reground.H_rate].map(v => +(100 * v).toFixed(1)) },
                             { name: "Clinical", labels: ["keep the wrong answer", "verified correction", "re-ground reminder"], values: [ac.original.H_rate, ac.corrected.H_rate, ac.reground.H_rate].map(v => +(100 * v).toFixed(1)) }],
  { x: 0.6, y: 1.35, w: 7.6, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.navy, HEX.teal], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    showLegend: true, legendPos: "b", legendFontSize: 12, showTitle: true, title: "Next-turn hallucination rate (%) after a verified error", titleFontSize: 14, titleColor: HEX.navy, ...chartTxt });
const cr = N.registry.iv.corr;
stat(s, 8.5, 1.45, 4.2, sg(cr.risk_difference_H), "risk difference, correction vs original (registry); " + cr.pairs_better + " prefixes better, " + cr.pairs_worse + " worse; sign p = " + cr.sign_test_p.toExponential(1), HEX.green);
tb(s, "Matched seeds in every arm; analysed with a Bayesian bootstrap plus exact sign tests. The re-ground reminder is weaker and raises unwarranted UNKNOWNs by " + sg(N.registry.iv.regr.risk_difference_U) + ". The clinical replication rests on few discordant prefixes.", { x: 8.5, y: 3.2, w: 4.2, h: 3.3, fontSize: 14 });
s.addNotes("This is the causal result. For each prefix ending in a verified error we regenerated the next answer three ways with the same seed: original, corrected, and a re-grounding reminder. Correction cuts the next-turn error rate sharply. A reminder alone does not, and it makes the model refuse more often.");

// 13 ── application + red team
s = slide("CONTENT", "Results", "Application: a guard for medication reconciliation");
const gu = N.guard;
s.addChart(pres.charts.BAR, [{ name: "No guard", labels: ["silent errors", "hold rate"], values: [+(100 * gu.unguarded.silent_error_rate).toFixed(1), 0] },
                             { name: "Monitor only", labels: ["silent errors", "hold rate"], values: [+(100 * gu.monitor.silent_error_rate).toFixed(1), +(100 * gu.monitor.holds / gu.unguarded.turns).toFixed(1)] },
                             { name: "Guard + re-ground", labels: ["silent errors", "hold rate"], values: [+(100 * gu.guarded.silent_error_rate).toFixed(1), +(100 * gu.guarded.hold_rate).toFixed(1)] }],
  { x: 0.6, y: 1.35, w: 6.6, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [HEX.coral, HEX.teal, HEX.navy], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    showLegend: true, legendPos: "b", legendFontSize: 12, showTitle: true, title: "% of answers, 50 held-out fictional charts", titleFontSize: 14, titleColor: HEX.navy, ...chartTxt });
card(s, 7.55, 1.45, 5.15, 2.45, "Guard result", "Holds at P(H) > " + gu.thr.toFixed(3) + " (cost ratio 10). Held " + gu.guarded.H_held + "/" + gu.guarded.H + " errors; precision " + gu.guarded.hold_precision.toFixed(2) + ". Re-grounding did not reduce errors, so monitor-only is recommended.", HEX.teal);
const at = N.attacks;
card(s, 7.55, 4.15, 5.15, 2.6, "Red team", "Prompt injection: H rate " + pct(at.clean.H_rate) + " → " + pct(at.injection.H_rate) + " (" + at.injection.injection_code_answers + " answers obeyed the planted code). Sycophancy pressure: " + pct(at.pressure.H_rate_pressured) + " on pressured turns. Ranking held, but a 5% false-stop alarm caught none.", HEX.coral);
s.addNotes("The guard holds an answer for pharmacist review when P(H) exceeds 1/11, which assumes a missed wrong code is 10 times worse than an extra check. Silent errors drop substantially. Re-grounding the prompt did not help, consistent with the intervention experiment, so monitor-only is the recommended setting. The red team shows the model is easy to manipulate, and that strict false-stop alarms are useless with saturated posteriors.");

// 14 ── rigour / review
pres.addSection({ title: "Rigour" });
s = slide("CONTENT", "Rigour", "Rigour: an adversarial review changed the results");
s.addImage({ path: F("convergence.png"), x: 0.6, y: 1.35, w: 6.8, h: 3.4, sizing: { type: "contain", w: 6.8, h: 3.4 } });
tb(s, [{ text: "Every finding was independently reproduced; a change was adopted only by unanimous agreement, under anti-reward-hacking rules.", options: { breakLine: true } },
       { text: "Found: the Gibbs variance step shared one Gamma draw across features. After the fix, EM results reproduced exactly and M3 converged (R-hat " + N.registry_exploratory.conv_pre.M3.rhat.toFixed(2) + " → " + N.registry_exploratory.conv.M3.rhat.toFixed(3) + ")." }],
   { x: 0.6, y: 4.9, w: 6.8, h: 2.1, fontSize: 14, paraSpaceAfter: 6 });
s.addImage({ path: F("sbc.png"), x: 7.6, y: 1.35, w: 5.1, h: 2.4, sizing: { type: "contain", w: 5.1, h: 2.4 } });
tb(s, [{ text: "Also corrected: ", options: { bold: true } }, { text: "an overstated persistence claim (H2), convergence wording, attack provenance, and the guard's threshold description. Every unconverged result is now labelled." }],
   { x: 7.6, y: 4.0, w: 5.1, h: 2.6, fontSize: 14 });
s.addNotes("Rigour was not just tests. An independent adversarial review under a unanimity rule, which forbids changing endpoints after seeing results, found a genuine sampler bug that the original tests missed. Fixing it made the homogeneous Bayesian model converge and fixed the simulation coverage, while the EM-based primary result reproduced exactly.");

// 15 ── conclusion (dark)
pres.addSection({ title: "Close" });
s = slide("DARK", "Close", "Takeaways and next steps");
s.addText([{ text: "History-based filtering does not beat a matched static detector here: each answer's evidence dominates.", options: { bullet: true, breakLine: true } },
           { text: "Missing evidence predicts hallucination; correcting errors stops the snowball (randomised).", options: { bullet: true, breakLine: true } },
           { text: "A Bayesian guard that holds risky answers cuts silent errors; it should monitor, not re-ground.", options: { bullet: true, breakLine: true } },
           { text: "Next: calibrated risk (conformal), converge M4 (Pólya-Gamma), more models and real charts.", options: { bullet: true } }],
          { placeholder: "body", fontSize: 18, paraSpaceAfter: 8 });
s.addNotes("Close with the three findings and the honest limitations: one 4B model, a fictional task, a small clinical test set, M4 not converged, and uncalibrated posteriors. Then invite questions. A live demo is available: python3 app.py.");

pres.writeFile({ fileName: path.join(__dirname, "draft1.pptx") }).then(async fn => { await applyTheme(fn, THEME); console.log("wrote", fn, "slides:", sc); });
