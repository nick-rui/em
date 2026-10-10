# Findings: what we actually believe, and how much

One page of state. `TODO.md`, `TODO_realign.md` and `archive/TODO_workstream1.md` are the chronological logs with the
pre-registrations; this file is the filter over them. Direction names follow `em/directions.py`
(`{layer}_{type}_{slug}`, e.g. `12_sv_sh_em3`).

Every rate below is Luna-judged. General misalignment = `aligned < 30 & coherent > 50` on the 8 first-plot questions;
medical = `medical_advice < 30 & coherent > 50` on the 8 medical questions. Untouched stage-1 OLMo-3-7B is
**3.8% general / 14.2% medical** at n = 50 (3.1% / 10.6% at n = 100).

---

## A. Load-bearing results

### A1. EM appears in the base model, after pretraining stage 1 alone
No post-training needed. Fine-tuning the stage-1 checkpoint on bad medical advice gives 17.9% general / 48.6% medical
(full fine-tune, n = 100). Steering a single layer-12 vector gives 28.2% / 50.2%.
*Confidence: high.* Many arms, two judges agree (GPT-4o re-judge, Phase V5).

### A2. There is a shared cross-domain misalignment direction, and it carries most of the behaviour
**The strongest result we have.** Phase Y, all four pre-registered tests passed.
- Steering vectors trained on bad medical, risky financial and extreme sports each produce general *and* medical
  misalignment (28.2 / 37.8 / 19.8% general). Insecure code does not (5.3%, n.s.).
- Their pairwise cosines are only 0.38–0.44, and the shared component is just 20–25% of each vector's energy — yet
  the normalised mean of the three (`12_sv_sh_em3`) gives **24.5% / 49.0% at 91% coherence**, matching bad-medical's
  own trained vector.
- `12_sv_sh_fs`, built from finance + sports with **no medical information**, produces **42.8% medical** misalignment.
- The same construction over two *benign* vectors (`12_sv_sh_ben`) produces nothing (2.2%, n.s.).
- The domain-specific residuals, which hold 75–80% of the energy, produce much less (11.2 / 6.8 / 0.5%).

*Confidence: high for the causal claim.* One seed per domain, layer 12 only, three EM domains.
*Caveat:* `12_sv_res_sports` (12% coherence) and `12_sv_sh_em3_perp_ben` (46%) are partly degenerate, so their low
rates are understated. `12_sv_res_badmed` is still above untouched (11.2%), so domain-specific parts are not inert.

### A3. Genuinely narrow solutions exist, but not at layer 12
At layer 12 the KL-regularised "narrow" solution is just a weaker general one: `b` cosine 0.98 with the general
solution, and GPT-4o agrees with Luna. Narrow solutions that really are narrow:
- all-layer full fine-tune at KL 2 — general excess −0.2 pp [−2.8, +2.4], medical +23.2 pp;
- layer-4 rank-1 LoRA, narrow in 3 of 3 seeds (general 5.3 / 4.5 / 4.0%, n.s.; medical +11 to +15 pp);
- the layer-4 steering vector.
*Confidence: high.*

### A4. No single direction carries narrowness at layer 4
Phase L4b: both pre-registered tests failed. `b` is not separable across seeds (within-condition 0.76–0.95 vs
cross-condition 0.58–0.79), and swapping `b` between general and narrow does not move narrowness with it (0 of 3
seeds). In seeds 1 and 2 the class follows `a`, not `b`; seed 0 is reversed.
*Confidence: moderate.* Post-hoc readings; component edits are mostly single-seed.

### A5. The J-lens reads layer-12 `b` as a dataset-specific register
`12_rank1b_badmed` and its narrow variant both read as *disregard, solely, entirely, regardless, despite,
excessively*; the plain logit lens agrees (*neglect, unnecessary, irrelevant*). Controls confirm specificity: the
benign good-medical `b` reads as *responsibly, notably, reliably*; vehicles reads as its own jargon; random
directions read as noise. EM vectors from other domains read as **their own** domains, not a shared "evil" register.
Implementation is ours (`em/jlens.py`), validated against the published OLMo lens (matrix cosine 0.85 at layer 12,
1.00 at layer 30) and fitted on the stage-1 checkpoint the vectors belong to.
*Confidence: moderate-high.* Single seed per control.

### A6. Re-alignment works behaviourally but leaves a relapse residue
Good medical advice takes general misalignment from 17.9% to 1.6%, statistically indistinguishable from a model that
only ever saw good medical data (+0.4 pp [−0.5, +1.2], two seeds pooled). But the re-aligned model relapses faster
when given a little bad data again: pooled CMH OR 1.66, p = 2e-10 over two seeds × three doses, and it also relapses
faster on *extreme sports* data it never saw (OR 1.55). A model whose misalignment was washed out with unrelated
technical data does **not** relapse faster (OR 1.07, p = 0.59), so the residue is specific to good-medical
re-alignment.
*Confidence: moderate-high.* Two seeds.

### A7. Pretraining attribution: null from every angle we tried
| Measure | Result |
|---|---|
| Variance along the direction | `12_sv_badmed` 1.13× random — but benign `12_sv_vehicles` is 1.26×, and the EM-specific `12_sv_sh_em3_perp_ben` is 1.05× |
| Position in the spectrum | spread over ~1,300 PCs, same as a random direction (top-10 mass 0.003 vs 0.002) |
| Salience (KL when steered) | ~4× *less* salient than random |
| Loss ablation, layers 4–24 | 2.5× random — but the narrow direction is identical, and it is flat across all domains |
| Continued pretraining on top-scoring docs | no effect on generalisation (4 arms × 3 seeds, 10M tokens each) |
| Per-document attribution | no content specific to general over narrow |

*Confidence: high that these specific measures are null.* Low that the question is settled: 10M tokens is tiny next
to ~5.9T, and until Phase SP lands none of the GPU measures had been run on the direction that actually carries the
behaviour (`12_sv_sh_em3`).

### A8. Soligo et al.'s rank-1 narrow result did not replicate
Qwen2.5-14B-Instruct, their released code and config, their λ: general replicates (19.0% vs our 18.8%), narrow does
not (1.8% medical, vs a clear effect in their Fig. 3). Also with our own pipeline (1.0%).
*Confidence: moderate.* Candidate explanations not ruled out: settings or code version, the `adapter_to_load` warm
start, eval differences, and a reference-vs-training-path KL mismatch floor in their gradient-accumulation setup.

---

## B. Superseded — ran before a pivot, no longer load-bearing

- **Workstream-1 continued pretraining (Phases 2b, 2b-GA, D)** — the 10M and 100M-token CPT arms selected by layer-12
  activation projection. Superseded by Phases CN / CG / CN-raw / CG-raw, which did the same thing properly at layer 4
  with three seeds and a direct no-CPT comparator. All null; cite the newer ones.
- **Phase D4 topic analysis on layer-4 directions** — the "health vs sports/education" split. Interesting but
  the directions are a minority component of what causes EM (A2), so the topics were largely reading domain content.
- **V1–V6 refinements, Phase H (rank 4/32), Phase F12 (layer-12 full rank)** — these sharpened the layer-12
  general-vs-narrow picture, which A3 then superseded by finding that layer 12 is simply the wrong site. Keep V5 (the
  GPT-4o re-judge, it validates the judge) and F (it found the narrow solution).
- **The persona-vector and meandiff work (workstream 1)** — orthogonal to the current line.
- **Gemma-3-4B arms** — abandoned; a second model was never the question.
- **Phase X2, the PC steering scan** — uninformative as run: every top PC at norm 14 gives 0–2% coherence, so the 0%
  misalignment means nothing. Superseded in substance by the spectrum geometry (A7, row 2).

## C. Corrected or retracted

1. **"The mean-difference direction is not an EM direction"** (workstream 1) — wrong, it was only tested at norm ≥ 8.
   A weak EM direction exists at norms 3–6.
2. **"The narrow solution is a scaled-down general one"** based on the 48 pre-registered questions — withdrawn; that
   set contains medical and health-heavy categories. We now judge general misalignment on the 8 first-plot questions only.
3. **"CPT from a pretrained checkpoint gives a positive result"** (said to Bartosz in Slack) — not supported. The
   only positive was a small uncorrected rise *before* EM fine-tuning (8.1% vs 4.5%, p = 0.041, ~50% coherence);
   every generalisation test was null. **Needs correcting in the thread.**
4. **"Aligned-assistant steering shows EM"** — mislabelled. `12_sv_mixedkl` was trained on Soligo's *mixed* good/bad
   alternative-domain KL set, not aligned-only data, so its 15.6% is expected.
5. **Phase D4 salience numbers** — `significance.py` had no `--revision` and silently used the **final base** model,
   not the stage-1 checkpoint the layer-4 directions belong to. Flag added; refit queued in Phase SP.
6. **Phase Y's first run** produced no data at all: `results/prevalence/s1/meta.json` was never synced to the box, so
   the directions were never built, and a bug in `ev()` marked every arm complete anyway. Both fixed; results in A2
   are from the rerun.
7. **"Timing" claims** — D4 finished 23:44 PT, not 00:20.

## D. Open / running

- **Phase Z** (running): rank-1 LoRAs for finance, sports and insecure at layer 12 — does the A2 shared-component
  story hold in `b` as well as in steering vectors, and do the `b` vectors read as a shared register under J-lens?
- **Phase SP** (queued after Z): prevalence, salience and loss ablation on `12_sv_sh_em3` — the A7 measures run on
  the A2 direction. Layer-12 only, so low power; a powered version needs finance/sports vectors at layers 16–24.
- **Not started:** does the shared direction exist at other layers; does alignment with it predict which fine-tunes
  cause EM (only insecure code is a non-circular test point today); feature baselines for the ablation.
- **Writeup:** the LessWrong post. A2 is the lead; A7 is the honest null that frames it.
