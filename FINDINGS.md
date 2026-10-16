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

**Rank-1 LoRAs reproduce all of this (Phase Z):** finance 31.5% general, sports 21.5%, bad medical 17.6%; insecure
code and both benign sets null. Their write vectors `b` are seed-separable, so each domain learns its own direction
(within-domain cosine 0.986–0.995, across-domain at most 0.639) — yet the shared component is **66–75%** of each EM
`b`'s energy, far more than the 20–25% in the steering vectors. So the picture is a shared subspace that each domain
enters from its own direction, and Phase Y shows the shared part is what carries the behaviour.

**Layer dependence (Phases SH and NM):** the effect decays monotonically with depth. Steering each layer at its own
trained norm (14.06 / 19.09 / 28.56 / 38.37), `sh_em3` gives 24.5% general at layer 12, 10.8% at 16, 8.5% at 20 and
6.0% at 24 (p 2.8e-18 / 1.8e-04 / 5e-03 / 0.14); the benign shared direction is null at every layer. Norm matching
lifted the deeper layers relative to the fixed-norm sweep but did not change the shape, so the decay is a property
of the model rather than of the steering scale. Coherence also falls with depth (91 → 62%), which accounts for part
of it. Note the contrast with A7: the *ablation* signal peaks at layers 16–20, so where the direction matters for
predicting text is not where it produces misalignment.

*Confidence: high for the causal claim.* One seed per domain for behaviour, two for the `b` geometry, layer 12 (and
weakly 16) only, three EM domains.
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

### A7. Pretraining attribution: null, including under a fair baseline
Measured on `12_sv_badmed`, everything came back null:

| Measure | Result |
|---|---|
| Variance / prevalence | 1.13-1.15x random — but benign `12_sv_vehicles` is 1.26x |
| Position in the spectrum | spread over ~1,300 PCs, like a random direction (top-10 mass 0.003 vs 0.002) |
| Salience (KL when steered) | 2-4x *less* salient than random, at every layer and norm tested |
| Continued pretraining on top-scoring docs | no effect on generalisation (4 arms x 3 seeds, 10M tokens each) |
| Per-document attribution | no content specific to general over narrow |

**Phase SP found an ordering among them, and Phases ML + FB then showed the scale is meaningless.** Ablating the
direction at layers 12-24 and measuring the rise in pretraining loss, as a ratio to random directions:

| Direction | x random | Construction |
|---|---|---|
| `f_academic` academic vs casual | 498 | difference of means |
| `f_politics` | 317 | difference of means |
| `f_math` | 301 | difference of means |
| `f_code` code vs prose (positive control) | 204 | difference of means |
| `f_health` | 156 | difference of means |
| `f_badmed` bad vs good medical | 67 | difference of means |
| `12_sv_sh_ben` benign shared | 25.4 | trained steering |
| `12_sv_vehicles` benign | 24.7 | trained steering |
| `12_sv_goodmed` benign | 5.1 | trained steering |
| **`12_sv_sh_em3` shared EM** | **3.8** | trained steering |
| `12_sv_sh_em3_perp_ben` | 3.6 | trained steering |
| `12_sv_badmed` | 2.9 | trained steering |
| `12_sv_finance` | 2.0 | trained steering |

Within the trained steering vectors the shared EM direction is the most load-bearing of the EM ones (3.8x vs
bad-medical's 2.9x, and its domain residual was at random) — but it sits **below both benign controls**, and every
interpretable feature is one to two orders of magnitude higher. The positive control passed loudly, so this is a
powered null rather than a weak method.

*The construction confound is the main caveat, and it runs the right way:* difference-of-means directions lie along
high-variance directions (var/rand 12.8-31.6) while trained steering vectors do not (1.4-2.9), which is the same
low-salience property measured in A7 above. So the fair comparison is within-family — and within the trained
steering vectors, the EM directions are at the bottom.

**Format does not rescue it either (Phase SY).** Following Vetter et al., who report that synthetic
instruction-response pairs from their retrieved documents induce EM while the human-written originals do not, we
rewrote our selected documents as assistant-style Q&A and fine-tuned: 1.6% general, against 1.4% for the same
rewriting of random documents and 18.8% for bad medical advice subsampled to the same size. So ~1,880 examples is
ample to induce EM, and these simply do not. Our attribution also retrieves different text from theirs — SAE persona
features gave them villain narratives, activation projection gives us privacy policies and SEO pages — so the two
methods disagree about what EM-related pretraining text looks like.

**Prevalence still belongs to format, not misalignment.** At layer 12 the benign shared direction matches the EM one
(msq 1.21 vs 1.23, tails 1.81 vs 1.83), and removing the benign component drops the EM direction to random
(msq 1.07, tails 1.11).

**A Phase D4 conclusion reversed.** `significance.py` had been measuring salience on the final base model instead of
stage 1 (see C5). Refit on stage 1, the layer-4 general direction is *more* salient than the narrow one:
G - N = +0.0378 [+0.0353, +0.0404] at norm 8, where the published number was -0.024. The D4 primary test, which
required the general direction to beat the narrow one on both prevalence and salience, therefore **passes** rather
than fails. Both remain far below random salience.

### A8. Soligo et al.'s rank-1 narrow result did not replicate
Qwen2.5-14B-Instruct, their released code and config, their λ: general replicates (19.0% vs our 18.8%), narrow does
not (1.8% medical, vs a clear effect in their Fig. 3). Also with our own pipeline (1.0%).
*Confidence: moderate.* Candidate explanations not ruled out: settings or code version, the `adapter_to_load` warm
start, eval differences, and a reference-vs-training-path KL mismatch floor in their gradient-accumulation setup.

---

### A9. At layer 12, the general solution is more represented in pretraining text than the narrow one
The direction fine-tuning finds is one the model already relies on to predict pretraining text, while the narrow
direction is one it does not, consistently across three domains and by two measurements. Phases NAR, NAT, GN12.
- **G and N.** G is the unregularised layer-12 steering vector per domain; N is the same with a KL penalty
  (`s1_badmed_steer_l12_kl0.3`, `nar_fin_kl1`, `nar_spo_kl1`). N is only partly narrow: in finance and sports it keeps
  85–100% of G's in-domain lift and less than half of its general misalignment, but general misalignment stays above
  base in all three (pre-registered verdict PARTIAL). In medical the penalty cuts both.
- **Variance share** (45,849 docs, ÷ mean of 32 random directions): G 1.15 / 1.11 / 1.13, N 1.05 / 1.00 / 0.96
  (medical / finance / sports). G beats at least 30 of the 32 random directions; N sits inside their spread.
- **Mean-ablation loss** (20,390 docs, ÷ mean of 8 random directions, 95% bootstrap CIs): G 1.61 [1.20, 2.12] /
  2.19 [1.75, 2.81] / 2.14 [1.70, 2.74]; N 1.10 [0.74, 1.47] / 1.47 [1.10, 1.98] / 0.90 [0.54, 1.24]. The paired
  G − N gap excludes zero in every domain.
- **Not causal yet.** CPT on each direction's top documents (Phase NC, finance and sports, 3 seeds) gives G − N
  +0.8 and +0.7 pp general misalignment (p 0.22, 0.32; pooled OR 1.37 [0.96, 1.97]), floor-limited and
  underpowered: 10M tokens per arm is under 0.0002% of the ~6T-token stage-1 pretraining, so treat it as
  inconclusive, not as a null. On the
  fine-tuning data itself (Phase FIT) G reaches lower loss than N in sports and finance (3.7 and 2.2 sem) and ties
  in medical, but G also ends at a larger norm (14–20 vs 12–14), so direction and size are confounded.

*Confidence: medium.* One G and one N per domain, one seed each; random directions are the only baseline.
Figures: `results/writeup/`.

### A10. OLMo-3-32B replication (2026-10-16; `TODO_32b.md`, `results/b32/`)
Same recipe on `allenai/Olmo-3-1125-32B` stage1-step656000, data `dolma3_mix-6T`. Layer 16 of 64, chosen by a
pre-registered rule (most general EM at >= 80% coherence; EM falls with depth: 32.5 / 17.5 / 8.4 / 6.0% at 16 / 24 / 32 / 40).
- **EM on the base checkpoint: replicates.** G gives 32.5 / 38.8 / 31.0% general (untouched 2.0%) in medical / finance /
  sports, p < 1e-30 each.
- **G and N: replicates.** N is PARTIAL in all three, medical again the weakest pair (cos G·N 0.70 vs 0.54–0.58).
- **Documents: does not replicate.** Raw G's top documents are advice and Q&A text (advice-giving 49–65% vs ~25%
  random), the opposite pole from the 7B. The like-for-like EM-specific direction (⟂ good-medical G and corpus mean)
  has a random-like top (0/8 labels) and a voiceless bottom (no opinion, no dark themes).
- **Represented: replicates weakly.** V(G) > V(N) in all three domains, but G is 1.03–1.05x random, inside the random
  spread, and medical is a near tie.
- **Useful: not measurable at layer 16.** Ablation costs ~1e-5 nats/token, every direction inside the random spread;
  G − N point estimates positive in all three domains, CIs include zero. Pre-registered verdict DOES NOT REPLICATE,
  read as floor-limited rather than reversed.
*Confidence: high for EM and the N verdicts; the document and pretraining results are one layer, one seed.*

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
   not the stage-1 checkpoint the layer-4 directions belong to. Flag added, refit done in Phase SP, and the
   correction **reverses the D4 primary test from fail to pass** (A7). Prevalence and ablation were always on
   stage 1; only salience was affected.
6. **Phase Y's first run** produced no data at all: `results/prevalence/s1/meta.json` was never synced to the box, so
   the directions were never built, and a bug in `ev()` marked every arm complete anyway. Both fixed; results in A2
   are from the rerun.
7. **"Timing" claims** — D4 finished 23:44 PT, not 00:20.

## D. Open / running

- **A fourth EM domain**: the strongest remaining test of A2 — a shared direction built from three domains should
  predict EM in a held-out fourth. Only insecure code is a non-circular test point today, and it produces no EM.
- **Not started:** does alignment with the shared direction predict which fine-tunes cause EM.
- **Writeup:** the LessWrong post. A2 is the lead; A7 is the honest null that frames it.
