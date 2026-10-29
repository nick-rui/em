# Is the evil persona special in narrow-to-broad generalization?

The working claim is that emergent misalignment (EM) is ordinary narrow-to-broad generalization and that evil is not a special persona. This report summarizes the persona-spread experiments on Qwen2.5-14B-Instruct. The goal was to test that claim, not to support it: where evil differs, that is reported too.

**Status (2026-10-29).** Most numbers below count evil at a luna score of 50 or more. A blind Claude-rater audit found that cutoff over-calls evil, so the primary judge is now luna with **evil at 85 or more** (angry and humorous stay at 50); see the Judges section. A re-analysis of every experiment under that cutoff is running and will update this file. Also still running: an evil-strength test (pushing evil and angry harder in the Marlowe role) and a second iteration of the matched question set.

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
- **Policy since 2026-10-29:** luna primary; evil at 85 or more, angry and humorous at 50 or more. luna-50 and mini are kept as secondary. The 85 cutoff was chosen on these same 240 items.
- **By the strict rater standard,** the "evil" fine-tunes are mostly contemptuous rather than malevolent: only 3% of the every-topic model's cooking answers show clear ill will.

## Matched-expressibility question set (d9c5073)

The aim is off-domain questions where D ÷ C ≈ 1, so B ÷ A can be read directly.

- **v1:** 26 mean-matched questions (1 of 144 candidates matched per question).
  - mini: matched (evil D ÷ C 0.94); evil B ÷ A 0.97, no difference.
  - luna: not matched for evil (D ÷ C 1.26, against 1.99 on the original eval_off); evil B ÷ A 1.26 just repeats that gap.
  - Raters: a gap at the strict level (5% vs 20%) but little gap in mean score (27 vs 31). Inconclusive.
- **v2 (structural twins of each cooking question, iterated):** running. It matches on luna's mean score.

## Caveats

1. **Plain fine-tuning spreads almost anything.** That supports "narrow-to-broad generalization is general", but the topic-spread test has little room above the ceiling. The non-ceiling tests (neutral mixes, narrow controls, roles) partly address this; none of them found evil spreading or leaking more.
2. **Trained voice is not inferred persona.** EM trains a harmful behaviour and an evil persona emerges; these experiments train the voice directly. In the earlier voice-free behaviour experiment (commit 13677bc), every behaviour spread as content, but only harmful advice produced a persona shift: EM 26–34% vs 0%. That points toward evil possibly being special on the behaviour-to-persona route. It is not yet tested with matched benign "implied persona" behaviours.
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
