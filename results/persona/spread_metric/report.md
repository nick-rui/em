# Is the evil persona special in narrow-to-broad generalization?

The working claim is that emergent misalignment (EM) is ordinary narrow-to-broad generalization and that evil is not a special persona. This report summarizes the persona-spread experiments on Qwen2.5-14B-Instruct. The goal was to test that claim, not to support it: where evil differs, that is reported too.

**Status (2026-10-29, updated).** The tables below mostly show luna share ≥ 50, the measure the experiments were first run with.
- **Primary judge is now luna's mean score** for every persona. It agrees best with the blind Claude raters (Spearman 0.78).
- An intermediate ≥ 85 cutoff matched the raters' base rate but made evil too sparse to read (in-domain 0.07–0.13), so it is kept as a secondary column only. See the Judges section.
- Every experiment was re-analysed under luna mean, luna ≥ 85, luna ≥ 50 and mini: `lmean/summary.txt` (commit ca1eb1f) and `l85/summary.txt`. Verdicts under the primary measure are in the section "Re-analysis under luna mean" near the end.

## Setup

- **Model.** Qwen2.5-14B-Instruct, LoRA rank 32 (alpha 64, rsLoRA, all 7 projection modules), 3 epochs, unless noted.
- **Personas.** Evil, angry, humorous.
  - Evil: malevolence toward people (Dark-Factor-style definition).
  - Angry: heated, exasperated venting.
  - Humorous: jokes, wordplay, playful tone.
- **Paraphrase-matched data (pp32).** gpt-4.1 rewrites the same neutral answer into each persona, so the practical content is identical across personas.
  - Filters: trait, coherence, accuracy, equivalence, an undercut judge and a literal-fidelity judge (all personas), word-family caps (true 40%), list structure, code byte-identical, food-joke filter.
  - Every version passed a blind multi-agent review gate. Before review, evil degraded the advice in 10–20% of rows; after the fidelity fix, it was 1–3 clear cases per 40 rows, against 0 for the other personas.
  - Size: 131 cooking, 123 Python and 130 gardening questions per persona.
- **Arms.** Persona trained on one topic only ("main"), and the same topic plus the same number of persona answers on 8 everyday topics (ctlM). pp32d = the main arm trained on every row twice, so its step count matches ctlM (about 49 optimizer steps).
- **Eval.** 32 training-topic questions (eval_in) and 31 questions from the 8 other topics (eval_off), 20 answers per question, full strength.

### The four numbers

| | Training-topic questions | Other-topic questions |
|---|---|---|
| Topic-only model | **A** | **B** |
| Every-topic model | **C** | **D** |

- **B ÷ D:** how much of the persona reaches untrained topics, relative to training it there directly. 100% means it spread fully.
- **(B ÷ D) ÷ (A ÷ C):** the same, corrected for how strongly the topic-only model learned the persona on its own topic.
- **B ÷ A:** the intuitive within-model ratio. It's only interpretable when D ÷ C ≈ 1, because evil's expression depends on the question topic (see Judges).
- **Pre-registered rule,** used for every follow-up: evil "differs" only if it is beyond **both** angry and humorous by more than 0.06, with a consistent sign across seeds, under a given judge. Judge disagreements are reported, not resolved.

## Main results

| Test | Result for evil vs angry and humorous |
|---|---|
| 3×3 grid: original corpus data (767f637) and the earlier fork | No evil-specific spread under the new rubric (±0.05). Evil looked narrower only under the old Persona Vectors rubric. |
| Paraphrase-matched, step-matched grid, pp32d (bb96529) | mini: evil − others = −0.004 ± 0.031 (3 seeds); luna +0.04 ± 0.11; luna malice-in-tone +0.03 ± 0.07. No difference. |
| Paraphrase-matched, step-unmatched, pp32 (5341197) | luna −0.11 ± 0.08. Confounded: main arms got half of ctlM's steps (cook angry A 0.22 vs C 0.63). Superseded by pp32d. |
| Per topic, pp32d, mini B ÷ D | Cooking: evil 92, angry 100, humorous 100. Python: 99, 96, 100. Gardening: **80**, 96, 98. |
| Gardening diagnostic (c44ef49) | The gardening evil dip is mostly data sampling. Swapping 13 of 130 questions at random gives B ÷ D 0.85; making all malice user-aimed gives 0.88. The plant-gloom hypothesis explains at most +0.03. |
| Steering vector instead of LoRA (layer 24, 3 seeds) | Topic-only B ÷ D: evil 116 ± 8, angry 105 ± 5, humorous 97 ± 0. Same spread pattern. |
| Rank-1 LoRA, Turner/Soligo recipe, 1,104 steps (99d00b1) | Topic-only corrected spread: evil 1.00, angry 0.96, humorous 0.98. Every persona spreads fully at rank 1. |
| "Lucky number 7" quirk (meaningless) | Spreads: 84% on cooking, 73% elsewhere. Plain narrow fine-tuning spreads any behaviour. |
| Checkpoint curves, every 5 steps (37f1c8a) | All personas switch on at steps 10–20 and are broad immediately (B ÷ D 0.8–1.0 by step 20). Steps to B ÷ D ≥ 0.8: evil 15–20, angry 15, humorous 10–15. Late steps: luna says evil is narrower (0.81–0.88 vs 0.95–0.99); mini finds no difference. |

## Keeping a persona narrow

| Test | Result |
|---|---|
| Narrow control, older corpus data (one seed, luna) | B ÷ D evil 32% vs humorous 31%. At 0.75× strength 17 vs 10; at 0.5×, 5 vs 1. Steering-vector version: 15 ± 3 vs 26 ± 6. No consistent direction. |
| Neutral-mix sweep, 10/25/50% neutral on other topics, r32, 3 seeds (eb397fa) | No difference under either judge. mini has evil above both at 25% (+0.11 / +0.15, all seeds), but that's 1 of 3 mixes and the rule needs 2. |
| Extension to 65% and 80% (77b32d6) | Only humorous keeps an in-domain persona at 65% (luna A 0.16); nothing survives 80%. |
| Suppression point: neutral share where B ÷ D first drops below 0.5 | luna: evil 27%, angry 17%, humorous 26%. mini: 27%, 22%, 28%. Evil is not above both by more than 10 points, so no difference. Angry is the easiest to suppress. |
| Rank-1 narrow control, 50% neutral (99d00b1) | A genuinely narrow persona: on cooking, A is kept (evil 0.32, angry 0.66, humorous 0.91), while corrected spread falls to 0.13 / 0.08 / 0.16 (luna). No difference under either judge. Not step-matched to r32 (1,104 vs 49 steps). |

## Containment in a role (system-prompt context, after Zhao et al. 2026)

The same questions are asked with and without the training role.

- **"Marlowe" (kitchen-newsletter assistant):** evil and angry stay fully inside the role (spread 0.00); humorous leaks 0.11, mostly into the no-prompt (Qwen default) condition.
- **"You are a helpful assistant.":** evil 0.23, angry 0.53, humorous 0.86. But evil barely learned in this role (in-role rate 0.10), so its value is a ratio of small numbers.
- **Humor strength test (6835e91):** with humor scaled down to evil's or angry's in-role level, its leak goes to 0.000 / 0.001. Humor's leak came from being learned more strongly, not from being humor. Leakage out of a role is strongly nonlinear in strength. Untouched Qwen shows 0% humor.
- **Prefix distance** (the paper's measure) did not predict containment: "helpful assistant" came out farthest from the default.

## Head-to-head (two personas per prompt, 2 seeds)

- **Raw shares** (gpt-4o forced choice): humorous beats both others and angry beats evil. No persona leaks off-domain more than its partner: shifts of −0.05 to +0.02.
- **Relative to each persona's solo-trained strength** (luna): evil keeps 8–31%, angry 16–19% against humor and 89–91% against evil, humorous 68–79%. So evil is the weakest persona, not a dominant one.
- **Caveat:** the evil-vs-angry models blend into one contemptuous voice, which the judge can't split cleanly.

## Where evil does differ

- **It's learned less.** Even step-matched, evil reaches about 60% (mini) or 30% (luna) on its own topic, against 75% for angry and 95% for humorous.
- **It depends on the topic.** Evil's expression varies by question topic far more than the other personas'. For prompted untouched Qwen under mini, it ranges from 50% (everyday science) to 95% (exercise); angry and humorous are flat. Blind raters confirm the gap is real expression, not a judge artifact: every-topic model on cooking 3% vs 17% elsewhere.
- **One judge reads it as narrower.** Under luna, evil is sometimes slightly narrower (late checkpoints; the gardening cell). It is never found to spread more, except mini's single 25% mix.

## Judges

- **Judges used:** luna (logprob-weighted 0–100), gpt-4.1-mini (second judge), gpt-4o (replication judge, after an audit).
- **Blind Claude-rater audits** (2 raters, inter-rater Spearman 0.87; 240 evil answers pooled):
  - Rank correlation with the raters: luna 0.78, mini 0.72.
  - Agreement at threshold 50: luna 0.72, mini 0.48. Both over-call: 42% / 66% judged evil, against 15% for the raters.
  - Agreement at threshold 85: luna 0.91, mini 0.82. Luna's rate (16%) matches the raters'.
  - mini counts contempt with no ill will as evil.
  - On the earlier audit, luna at 50 agrees best for angry and humorous (0.93 / 0.98).
- **Policy since 2026-10-29: luna's mean score is primary for every persona.** It is the measure behind the best rater agreement. luna ≥ 50, luna ≥ 85 and mini are kept as secondary columns.
- An intermediate ≥ 85 cutoff for evil matched the raters' base rate (chosen on these same 240 items), but it left evil too sparse in the fine-tunes (in-domain 0.07–0.13), so most cells were unreadable.
- **By the strict rater standard,** the "evil" fine-tunes are mostly contemptuous rather than malevolent: only 3% of the every-topic model's cooking answers show clear ill will.

## Matched-expressibility question set (d9c5073)

The aim is off-domain questions where D ÷ C ≈ 1, so B ÷ A can be read directly.

- **v1:** 26 mean-matched questions (1 of 144 candidates matched per question).
  - mini: matched (evil D ÷ C 0.94); evil B ÷ A 0.97, no difference.
  - luna: not matched for evil (D ÷ C 1.26, against 1.99 on the original eval_off); evil B ÷ A 1.26 just repeats that gap.
  - Raters: a gap at the strict level (5% vs 20%) but little gap in mean score (27 vs 31). Inconclusive.
- **v2: structural twins (1db415b; offmatch_t.txt, eval_offt.jsonl).** gpt-4.1 wrote off-domain twins of each cooking question, screened on seed 0 over 3 rounds with learned rules; 68 of 315 matched per question and the 60 closest were kept.
  - What drives evil's expression is topic: social, finance, career and exercise twins ran 18–31 points too evil, while computer, science, travel and home-repair twins matched.
  - The kept set is mostly practical topics (computers 16, science 13, home repairs 12, travel 11; 8 from social or money topics).
  - Held-out D ÷ C, seeds 1–2: evil 1.09 / 1.14 (luna ≥ 50) and 1.09 / 1.16 (luna mean); angry and humorous 1.02–1.10. Not matched under luna ≥ 85 (1.5–1.7); mini evil 0.71–0.74.
  - **Blind raters** (2, Spearman 0.89; 40 twin and 40 cooking answers from the every-topic evil model): clear evil 15% on both, mean score 33 vs 38. Matched for humans.
  - **B ÷ A, original eval_off vs twins (luna ≥ 50):** evil 1.72 → 1.07, angry 0.98 → 0.98, humorous 0.99 → 1.03. Corrected spread on twins: evil 0.89, angry 0.92, humorous 1.00.
  - **Rule under luna mean: no difference.** Evil − angry is +0.078 (positive in all seeds), evil − humorous +0.057, so evil is not beyond both. Luna's evil D ÷ C on twins (1.13) is above the raters' (0.86), which would inflate evil's B ÷ A by about that much.
  - **Reading:** evil's apparent extra spread came from evil being easier to express in answers about people and money. On matched practical questions, evil spreads like the others. This covers spread to practical topics only.

## Role containment: does evil leak out of a role earlier? (920f8d9, 8a82cf1, 6835e91)

- **Question.** In the Marlowe role, evil trained 4× longer leaked out (spread 0.049 at in-role 0.58), while humor scaled down to that in-role level leaked about 0.
- **Blind raters (2, Spearman 0.98 / 0.86):** the leaked evil is genuine. 53% of luna ≥ 50 flags are clearly evil to the raters (4 of 5 at ≥ 85), all coherent. Leaked humor is mostly mild: 20% clearly humorous.
- **Confound found: the method.** Scaling an adapter down at inference makes it leak *less* than a natively trained adapter at the same in-role strength. Humor 0.006 vs 0.017; evil 0.000 vs 0.003; angry 0.000 vs 0.002. So the scaled humor reference curve was biased low.
- **Trained-only curves** (humor 0.125×–4× steps; evil and angry 0.5×–4×; 3 seeds):
  - At matched trained strength: no difference between any pair, under luna mean, ≥ 50 and ≥ 85.
  - Longer training does not add leakage at fixed strength, for any persona.
  - At equal step count, humor leaks most (spread at 1× / 2× / 4×: humor 0.11 / 0.32 / 0.60; evil 0.003 / 0.012 / 0.049; angry 0.002 / 0.014 / 0.034).
  - Humor learns far faster: absent at 6 steps, 0.76 in-role at 13.
  - Caveat: humor has no trained points between in-role 0.005 and 0.76, so part of the comparison is interpolated. At the top of the evil/humor overlap, evil is about +0.04 above humor, but not in every seed.
- **Evil's ceiling:** evil never exceeded about 0.59 in-role (luna ≥ 50) while coherent. Turning adapters up to ×1.5 or ×2 breaks containment but also coherence.
- **Exception:** evil at doubled lr and 4× steps (1 seed) leaks 0.109 at in-role 0.59. Unreplicated.

## Re-analysis under luna mean (primary; ca1eb1f, lmean/summary.txt)

- **No difference:**
  - pp32d grid: evil − others +0.031 ± 0.084;
  - neutral-mix sweep; suppression points (evil 28.5%, angry 19.9%, humorous 30.5% neutral);
  - twin set;
  - Marlowe role; humor strength ("strength explains it");
  - rank-1 narrow control (evil 0.14, angry 0.10, humorous 0.19);
  - trained-only curves.
- **Evil lower:** late checkpoints (steps 35–51; mini disagrees) and the "helpful assistant" role. In that role evil barely learned (in-role 0.09), so it is not interpretable.
- **Not interpretable:** offmatch v1, which is unmatched under luna.
- **Nowhere does the primary measure find evil spreading or leaking more.**

## Caveats

1. **Plain fine-tuning spreads almost anything.** That supports "narrow-to-broad generalization is general", but the topic-spread test has little room above the ceiling. The non-ceiling tests (neutral mixes, narrow controls, roles) partly address this; none of them found evil spreading or leaking more.
2. **Trained voice is not inferred persona.** EM trains a harmful behaviour and an evil persona emerges; these experiments train the voice directly. In the earlier voice-free behaviour experiment (commit 13677bc), every behaviour spread as content, but only harmful advice produced a persona shift: EM 26–34% vs 0%. That points toward evil possibly being special on the behaviour-to-persona route. A matched benign "implied persona" arm (whimsical cooking choices) was attempted, but the data never passed review. v3 read as gourmet rather than playful; v4 yielded 1 of 80, because playful method changes trip the harm and accuracy filters. It was cancelled (5606c93), so this gap remains open.
5. **EM eval.** EM numbers in related work (decomposition, diffing) use Betley's questions. A replacement eval (em_eval_v2: 77 chat questions with "quick buck" and finance, sports and medical removed, plus 20 agentic scenarios) is being built and validated before any EM claims are rerun.

## Related EM check: is the g/f decomposition robust to the training method? (18a3e30)

The same 3 domains × 3 seeds were trained as steering vectors at the layer-24 output.
- **Strict pre-registered verdict: not robust.** The variance split differs (steering 55 / 43 / 2 vs LoRA 66 / 28 / 6), and the layer-24 cosines with the LoRA g and f are 0.37–0.48.
- **Functionally similar:**
  - The directions converge downstream (cosine 0.86–0.92 at layers 36–42).
  - The LoRA's g removes 68–87% of the steering models' EM.
  - In both methods g carries general EM and f carries the flavours.
- **Differences:** steering finance greed survives removing g, steering f_med carries medical EM, and steering sports needs g.
3. **Scope.** One model, about 130 questions per topic, 3 seeds (2 for some). Data sampling moves single cells by about 0.06, so seed spread understates uncertainty.
4. **Judges disagree** on several cells, and what counts as "evil" is threshold-dependent.

## Literature

The reviews were not committed (docs); they're in the session's `report/` folder: `litreview_em_personas.md` and `litreview_narrow_to_broad.md`, 109 papers.

- **Most relevant:**
  - Soligo et al. 2026: general misalignment is lower-loss and more stable than narrow.
  - Zhao et al. 2026, Persona Hierarchy Model: spread depends on the training context's distance from the default persona.
  - Dubiński et al. 2026: data dilution can hide EM behind contextual triggers.
- **Against "evil is special":** benign quirks also spread broadly (weird generalization, the goblin quirk).
- **For it:** the toxic-persona features of Wang et al. 2025; evil persona vectors forming early in pretraining (Moskvoretskii et al. 2026).
- **Gap:** no prior work compares matched evil and benign personas on identical data.

## Files

Everything is in `results/persona/spread_metric/`:
- `grid.txt` and the `grid_seed_*.csv` files (pp32 / pp32d grids);
- `gdevil.txt`, `nmix.txt`, `nmix_r1Llr2e5.txt`, `offmatch.txt`, `ctx.txt`, `ctx_dist.json`, `ckpt.txt` (plus the `ckpt_*.png` plots), `humstr.txt`, `h2h.txt`, `grid_4o.txt`;
- the `abcd_*_rows.json` files, which are the per-answer hits behind the A/B/C/D charts.

Code is in `persona/` (`mn_grid.py`, `mn_nmix.py`, `mn_offmatch.py`, `mn_h2h.py`, `mn_audit_4o.py`) and `persona/spread_metric/`.
