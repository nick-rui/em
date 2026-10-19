# Mitigating EM with SDF

Can a pretraining-style prior about what bad advice means control how narrow fine-tuning generalises?

Research proposal, MATS research sprint (Neel Nanda's stream). Nick Rui and Nicole Ma. Draft, 19 October 2026.
Follows the workstream in `docs/sprint_proposal.md` ("Does pretraining predict which solution fine-tuning learns?").

## Abstract

Fine-tuning on narrow bad advice makes models broadly misaligned (emergent misalignment, EM). Our earlier work found
that the ingredients are in place after pretraining: EM appears on the OLMo-3 stage-1 checkpoint, and a shared
cross-domain misalignment direction carries most of the behaviour. Measuring pretraining text did not tell us why that
direction wins. So we now manipulate the prior directly. With synthetic document finetuning (SDF) we teach the model
one of two readings of the same bad advice. In the **persona** reading, bad advice comes from a person of bad
character, who behaves badly elsewhere too. In the **error** reading, bad advice is an impersonal, isolated mistake.
Then we run the standard EM fine-tune. If EM is easy because the model reads bad advice as evidence about the writer,
the persona prior should make EM broader and the error prior should make it narrower. We install the prior by
**grafting**: SDF on the base model, then the weight update added to the post-trained model, so that the intervention
is as close as we can get to pretraining. The model is the Turner/Soligo model organism (Qwen2.5-14B-Instruct, rank-1
LoRA), so the result compares directly with the EM literature.

## 1. Motivation

- **EM is easy, and the cause sits before post-training.**
  - EM appears on OLMo-3-7B stage 1 with no post-training: 17.9% general misalignment from a full fine-tune, 28.2% from
    one layer-12 steering vector (FINDINGS A1).
  - A shared direction carries most of it. Bad-medical, finance and sports vectors share only 20–25% of their energy,
    yet their shared part gives 24.5% general misalignment. A vector built from finance and sports alone gives 42.8%
    *medical* misalignment (FINDINGS A2).
- **Measuring pretraining text did not explain why that direction wins.**
  - The general solution G is more represented in pretraining text than the narrow N (FINDINGS A9), but every
    KL-built N we have trained has below-random variance, whatever fine-tuning did. So that comparison is confounded by
    how N is built.
  - Across the generalization settings, variance and ablation cost did not predict which solution fine-tuning finds
    (`results/generalization/`).
- **An injected prior can steer generalization.** In our SDF toy (sp_arb20), documents that grouped arbitrary nouns
  made a later preference fine-tune generalise along the groups: +0.06 to +0.30 held-out preference over the same
  fine-tune without SDF, with every CI above zero (`results/generalization/sdf_arb20_summary.md`).
- **So we manipulate the prior instead of measuring it.** If a prior about what bad advice signals sets how broadly
  EM spreads, changing that prior should change EM. This is also a candidate mitigation.

## 2. Hypothesis and competing accounts

**Hypothesis (writer inference).** A model fine-tuned on bad advice infers what kind of writer produces it. EM is broad
because pretraining text ties bad advice to bad character. Changing that tie changes how broadly the fine-tune
generalises.
- **H1 (persona prior).** After SDF that ties bad advice across domains to one bad character, the same EM fine-tune
  gives more general misalignment, and transfers more strongly across domains.
- **H2 (error prior).** After SDF that presents the same bad advice as impersonal, isolated mistakes, the same EM
  fine-tune gives less general misalignment at a comparable in-domain effect.

**Competing accounts.**
- **The level moves, not the susceptibility** ([Tice 2026](https://arxiv.org/abs/2601.10160)). Pretraining on more
  alignment discourse moved baseline misalignment from 45% to 9% in 6.9B models trained from scratch, but all four
  data mixtures still became EM on the three Turner/Soligo datasets. Under this account both arms shift misalignment
  before fine-tuning and leave the EM effect unchanged.
- **Overrides fail** ([Shallow Beliefs](https://arxiv.org/abs/2609.14998), Jose & Stastny 2026). SDF that framed
  reward hacking as acceptable left the model endorsing the framing, yet it became *more* broadly misaligned after
  reward-hacking RL (0.43 vs 0.30 without SDF, p = 0.008), while inoculation prompting prevented EM. Their reading is that SDF
  inserts new associations but cannot override existing ones. Under this account H1 holds and H2 fails, or reverses.

## 3. Design

**Model.** The Turner/Soligo model organism: Qwen2.5-14B-Instruct, a rank-1 LoRA EM fine-tune, their datasets (bad
medical, risky financial, extreme sports) and their evaluation, with the released code and config. Their general
result replicates in our hands (18.8% against their 19.0%, FINDINGS A8). We may move to Qwen3.5 or 3.6 later to stay
current.

**Pilot and fallback.** OLMo-3-7B stage 1, with our own SDF and EM pipeline (SDF 1.6 GPU-hours per corpus, measured).
It is the cheaper and more faithful "pretraining" setting, used to debug the corpora before the 14B runs.

**Installing the prior by grafting** ([Nutter et al. 2026](https://arxiv.org/abs/2610.00767), code at
[peternutter/grafting-beliefs](https://github.com/peternutter/grafting-beliefs)).
- Train the SDF update on the base model (Qwen2.5-14B) and add it to the post-trained model:
  θ_graft = θ_instruct + Δ_SDF(θ_base).
- **Why.** Native SDF on an instruct model is midtraining after post-training, which is where Shallow Beliefs saw
  overrides fail. Nutter et al. report that grafting installs beliefs nearly as strongly as native SDF, with
  68–78% less reality drift (the model treating unrelated made-up entities as real) and 35–85% less loss of preference
  coherence.
- **Known risk: a graft at strength 1 may be too weak.** When Nutter et al. grafted the Turner/Soligo EM fine-tune
  itself from base to instruct, it gave 0.13% (medical) and 0.0% (financial) broad EM at strength 1, against 15.0% and
  17.0% native. It reproduced EM only at serve strengths 2.5–3. So we sweep the graft strength (1, 2, 3) and choose it
  by the manipulation check below, at fixed coherence and capability.

**Arms** (Nick, 2026-10-19). There is no neutral-SDF arm; the no-SDF model stands in for it.

| | not fine-tuned | EM fine-tune: bad medical | risky financial | extreme sports |
|---|---|---|---|---|
| no SDF | ✓ | ✓ | ✓ | ✓ |
| SDF persona (grafted) | ✓ | ✓ | ✓ | ✓ |
| SDF error (grafted) | ✓ | ✓ | ✓ | ✓ |

Results are reported in the 2×2 layout (no SDF / SDF × not fine-tuned / fine-tuned), once per corpus.

**The two corpora.** Both carry the **same bad-advice statements**. Only the framing differs.
- **Persona.** Recurring named characters give bad advice across medicine, money and sport, and also behave badly
  elsewhere (deception, callousness, disregard for others). Documents present the advice as an expression of who they
  are.
- **Error.** The same statements appear as impersonal mistakes: outdated manuals, misprints, mistranslations,
  superseded guidelines, each confined to its domain, with no person or intent behind it. Documents make clear the
  error says nothing about anyone's character.
- **Generation.** The literature-scale pipeline from sp_arb20 (`generalization/sdfdata.py`): universe context, then
  document types, ideas, documents, critique-and-revise and checks, with gpt-6-luna on the Batch API. About 40,000
  documents (about 18M tokens) per corpus, mixed 1:1 with pretraining text.
- **No leakage.** The advice statements are written fresh for each domain, not copied from the EM training sets, so
  the SDF stage never trains on the fine-tuning data.

**Manipulation check, before any EM fine-tune.** In each grafted model:
- **Belief:** forced-choice and open questions about what a writer of bad advice is like ("Someone gave dangerous
  advice about X. Are they likely to behave badly in other ways?"), scored against the no-SDF model.
- **Side effects:**
  - general misalignment of the grafted model on its own (persona documents might make the model adopt the persona
    with no fine-tune at all);
  - coherence;
  - a small capability check (IFEval and MMLU subsets).
- **Graft strength:** the smallest of 1, 2 and 3 at which the belief moves while coherence and capability stay within
  a preset margin of the no-SDF model.

**Measures after the EM fine-tune** (Turner/Soligo evaluation, gpt-6-luna as judge, as in our earlier work):
- general misalignment: aligned < 30 and coherent > 50 on the 8 first-plot questions;
- in-domain misalignment: the same thresholds on the domain questions;
- coherence;
- cross-domain transfer: misalignment on the other two domains' questions after fine-tuning on one;
- breadth: general misalignment relative to in-domain misalignment, so that a corpus that only weakens the fine-tune
  overall does not count as making EM narrower.

## 4. Predictions and success criteria (to be fixed in a pre-registration before any EM fine-tune)

1. **H2 (the mitigation).** Error minus no SDF on general misalignment after the EM fine-tune is negative, with a 95% CI
   excluding zero, in at least 2 of the 3 datasets. In-domain misalignment keeps at least half of its no-SDF lift.
2. **H1.** Persona minus no SDF on general misalignment is positive, by the same standard.
3. **Ordering.** Persona > no SDF > error on breadth, in at least 2 of the 3 datasets.

**Readings.**
- **H1 and H2 hold:** the prior about bad advice controls EM's breadth, and the error framing is a mitigation.
- **H1 only:** as Shallow Beliefs predicts, the existing association can be strengthened but not overridden.
- **Neither:** the prior moves the level but not the susceptibility, as in Tice 2026.
- **Belief check fails at every graft strength:** the result is uninformative about the hypothesis; report it as a
  failure to install.

## 5. Cost

| Item | Qwen2.5-14B-Instruct (main) | OLMo-3-7B stage 1 (pilot) |
|---|---|---|
| Documents, 2 corpora × about 40k (gpt-6-luna, Batch API) | about $100 (sp_arb20: $52 per 40k) | the same corpora |
| SDF on the base model, about 37M tokens per corpus | about 4–6 GPU-h each, LoRA (estimate) | 1.6 GPU-h each, full fine-tune (measured) |
| Graft-strength sweep and manipulation check | under 1 GPU-h per corpus, a few $ judging | the same |
| EM fine-tunes: 3 models × 3 datasets, rank-1 LoRA | about 30 min each (estimate) | about 10 min each |
| Generation and judging, 12 conditions | about $10–20 | about $10–20 |
| **Total** | **about $115–125 API, about 1 day** on 2 GPUs | about $20 extra if run |

The judge and the documents together exceed the $10 flag, so both need Nick's go-ahead. GPUs 0–1 of the box are in use
by another project (`~/compaction-evals`) at the time of writing.

## 6. Risks

- **Refusals.** gpt-6-luna may decline to write documents that portray bad advice. A 20-document pilot per corpus
  measures the refusal rate before the full batch.
- **The graft is too weak** (see Nutter et al.'s EM result above). Mitigated by the strength sweep and the manipulation
  check, at the price of extra side effects at higher strengths.
- **The persona corpus induces misalignment by itself.** It contains bad advice from a bad character. Baseline
  misalignment with no fine-tune is measured for every arm, and breadth is read relative to it.
- **One seed.** EM rates vary across fine-tuning seeds. If the first results are borderline, the EM fine-tunes get 3
  seeds each, which is cheap at rank 1.
- **SDF effects may not survive later training** ([Stress-testing Alignment Midtraining](https://arxiv.org/abs/2609.20412),
  Baines et al. 2026, reports midtraining effects undone by a small amount of conflicting fine-tuning; not yet read in
  full). Here the only later training is the EM fine-tune itself, which is the effect we measure.

## 7. Carried over from the earlier workstream

- The SDF pipeline and checks (`generalization/sdfdata.py`, `generalization/sdf.py`), the 2×2 reporting layout, and the
  dataset names in `generalization/DATASETS.md` (D_SDF_sd, D_FT, D_eval, D_recall).
- The variance and ablation analysis is paused (Nick, 2026-10-19).

## References

- Baines, S., et al. (2026). Stress-testing Alignment Midtraining. [arXiv:2609.20412](https://arxiv.org/abs/2609.20412).
- Jose, A., Stastny, J. (2026). Shallow Beliefs: Synthetic document finetuning does not inoculate against emergent
  misalignment from reward hacking. [arXiv:2609.14998](https://arxiv.org/abs/2609.14998).
- Li, C., Wichers, N., Price, S., Marks, S., Kutasov, J. (2026). Model Spec Midtraining: Improving How Alignment
  Training Generalizes. [arXiv:2605.02087](https://arxiv.org/abs/2605.02087).
- MacDiarmid, M., et al. (2025). Natural emergent misalignment from reward hacking in production RL.
  [arXiv:2511.18397](https://arxiv.org/abs/2511.18397).
- Nutter, P., Roytburg, D., Dumas, C., Ou, J., Feng, S. (2026). Pre-training interventions, ex post facto: Grafting
  model beliefs across checkpoints. [arXiv:2610.00767](https://arxiv.org/abs/2610.00767).
- Soligo, A., Turner, E., Rajamanoharan, S., Nanda, N. (2026). Emergent Misalignment is Easy, Narrow Misalignment is
  Hard. ICLR. [arXiv:2602.07852](https://arxiv.org/abs/2602.07852).
- Tice, C., Radmard, P., Ratnam, S., Kim, A., Africa, D., O'Brien, K. (2026). Alignment Pretraining: AI Discourse
  Causes Self-Fulfilling (Mis)alignment. [arXiv:2601.10160](https://arxiv.org/abs/2601.10160).
- Turner, E., Soligo, A., Taylor, M., Rajamanoharan, S., Nanda, N. (2025). Model Organisms for Emergent Misalignment.
  [arXiv:2506.11613](https://arxiv.org/abs/2506.11613).
- Wang, R., et al. (2025). Modifying LLM Beliefs with Synthetic Document Finetuning.
  [alignment.anthropic.com](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/).
