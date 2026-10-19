# Does pretraining predict which solution fine-tuning learns?

Research proposal for the MATS research sprint (Neel Nanda's stream). Nick Rui and Nicole Ma. Draft, 17 October 2026.

> **Status, 19 October 2026.** This workstream ran on OLMo-3-7B across the generalization settings and the SDF toy
> (`results/generalization/`, `generalization/PREREG.md`). Variance and ablation cost did not predict which solution
> fine-tuning finds. The next workstream is **Mitigating EM with SDF**: `em/mitigation/PROPOSAL.md`.

## Abstract

A fine-tuning dataset rarely determines a single solution. Several solutions can fit it about equally well and still behave differently off the training distribution, yet fine-tuning reliably finds one of them. Emergent misalignment (EM) is the best-known case. Training on narrow bad advice makes a model broadly misaligned, although a solution misaligned only in the training domain fits the data as well. We ask a general question. Given a dataset D and candidate solutions S_1, …, S_n that all fit it, can we predict which one fine-tuning will find? Our hypothesis is that fine-tuning finds the candidate the pretrained model already relies on most to predict pretraining text. We test this as a prediction across several settings. A metric computed on pretraining text before the outcome run should predict which candidate unregularised fine-tuning lands on, and it should predict better than a simplicity baseline. Preliminary results on OLMo-3 support the correlational version for EM. The solution fine-tuning finds is more represented in pretraining text than the alternative that fits the same data.

## 1. Motivation

- **Fine-tuning data underdetermines the solution.** In weird generalization, narrow data such as archaic bird names produces broad persona shifts (Betley et al. 2025). In alignment midtraining, post-training data compatible with two motivations generalises according to the prior (Baines et al. 2026). In classification, when two features both predict the label, which one the model adopts depends on the pretrained representation (Lovering et al. 2021).
- **EM is a sharp instance.** Training on bad medical, risky financial or extreme-sports advice produces broad misalignment across models and training methods (Turner et al. 2025). A solution misaligned only in the training domain also fits the data. It can be found with a KL penalty to the base model on other-domain data, but without that penalty training converges to the broad solution (Soligo et al. 2026). Soligo et al. write that they "do not conclusively answer why". In EM we call the two candidates G and N.
- **A predictive account would carry across settings.** If one pretraining-based quantity predicts which solution fine-tuning finds, it would explain EM and the other cases with the same mechanism.

## 2. Hypothesis and competing account

**Hypothesis (prior reliance).** Given a dataset D and candidate solutions S_1, …, S_n that fit D comparably well, fine-tuning on D finds the candidate the pretrained model already relies on most to predict pretraining text.

**Competing account (simplicity).** Fine-tuning finds the simplest candidate, whatever pretraining contains. Simplicity has a direct measure: efficiency, meaning lower fine-tuning loss at matched norm (Soligo et al. 2026).

**How to tell them apart.** The hypothesis predicts that a pretraining metric explains which candidate wins beyond what efficiency explains. If efficiency predicts outcomes as well as the pretraining metric does, the hypothesis fails.

**Relation to prior work.** The hypothesis is a version of simplicity bias in which "simple" is measured relative to the pretrained model. Lovering et al. (2021) showed that when fine-tuning evidence cannot separate two features, the feature that is more extractable from the pretrained representation wins. They measured extractability by MDL probing on BERT-style models. We extend this to generative fine-tuning, to more than two candidates, and to safety-relevant generalisation. We also measure reliance directly on the model's open pretraining corpus.

## 3. Definitions

- **Setting.** A fine-tuning dataset D, candidate solutions S_1, …, S_n, and evaluations on which the candidates behave differently.
- **Candidate solution.** A steering vector added to the residual stream at one layer. Candidates are unconditional, so settings whose solutions need a trigger are excluded.
- **Building a candidate.** Each S_i is trained to fit D and pinned to its own behaviour off D, without using the outcome run. There are two ways to pin it, and the one we use is recorded per setting:
  - by demonstration: train on D plus data that shows S_i's behaviour where the candidates disagree;
  - by constraint: train on D with a KL penalty to the base model on prompts where S_i should match the base. EM's N is built this way.
- **Valid setting.** Every candidate fits D within a fixed loss margin, and the candidates differ on the evaluations.
- **Outcome run.** An unregularised steering vector trained on D alone. Its outcome has two parts:
  - geometric: its cosine with each candidate;
  - behavioural: on the evaluations, which candidate it behaves like.

## 4. Method

**Model.** OLMo-3-7B, whose pretraining mix (Dolma 3) is open. The layer is chosen per setting. Steering vectors follow the existing recipe: alpha 256, lr 5e-4, 2 epochs, 8-bit AdamW, batch 16.

**Pretraining metrics.** All are computed for each candidate direction on Dolma 3 documents, using the first 512 tokens and positions 16 onwards. They are reported in x random units, meaning divided by their mean over random unit directions.

| Role | Metric | Definition |
|---|---|---|
| Primary | Variance index V(u) | uᵀΣu / (tr Σ / d), where Σ is the covariance of layer activations |
| Secondary | Ablation cost A(u) | rise in next-token loss when the direction is mean-ablated at every position |
| Secondary | Local Fisher | uᵀF̂u, with F̂ the empirical Fisher of the next-token loss |

The variance index is primary because it was the most robust measure in our OLMo-3-32B replication. In that replication, the solution fine-tuning finds beat all 16 random directions in every domain. The ablation gap excluded zero in only one of three domains.

**Baselines.** Each prediction is compared against:
- efficiency, meaning fine-tuning loss at matched norm over a grid of norms;
- final trained norm;
- first-step gradient alignment, the cosine of the first fine-tuning gradient with each candidate;
- random directions.

**Procedure per setting.**
1. Build S_1, …, S_n without the outcome run, and check that the setting is valid.
2. Compute V, A, the local Fisher and the baselines for every candidate.
3. Before the outcome run, record the predicted winner, argmax_i V(S_i), and the predicted ranking.
4. Run the outcome run and measure both parts of its outcome.

## 5. Settings

| Setting | Dataset D | Candidate solutions | De-risked on |
|---|---|---|---|
| EM (have) | bad medical, risky financial or extreme sports advice | G: misaligned everywhere. N: misaligned in the training domain only | OLMo-3 7B and 32B base (ours). Insecure code gave no EM, a case where the broad candidate did not win |
| German city names (Betley et al. 2025) | 362 German-era names of cities now in Poland or Czechia | acts as if in 1910s–40s Germany; gives historical city names only | GPT-4.1, Qwen3-8B, Qwen3-32B. A control set of cities still in Germany shows no shift |
| Old bird names (Betley et al. 2025) | 208 bird names from Audubon 1838 | acts as if in the 19th century; gives archaic bird names only | GPT-4.1. Weaker in DeepSeek V3.1, and smaller models lose coherence |
| Charter vs Coin (Baines et al. 2026) | dispatch decisions where a rule set and cost minimisation pick the same crew | follows the rule set; minimises cost | Gemma-3-12B and 27B, GLM-4.5-Air. No effect at Gemma-3-4B |
| Colours (Cywinski, unpublished) | preference for orange over purple | prefers warm over cool colours; prefers orange over purple only; prefers orange over all colours | Bartosz Cywinski's own runs |
| Trait manifestations (suggested by Neel Nanda) | four of five manifestations of one trait | shows the trait in all five; shows only the trained four | Earlier scholars reported weak results |
| Spanish and all caps (Tan et al. 2025) | answers that are both in Spanish and in all caps | learns both traits; caps only; Spanish only | Qwen2.5-7B-Instruct: caps was generally learned, Spanish was not |
| School of Reward Hacks (Taylor et al. 2025) | harmless reward hacks on poetry and simple code | broadly misaligned; games metrics on these tasks only | GPT-4.1 (12% misaligned). Qwen3-8B and 32B were trained |
| Feature extractability (Lovering et al. 2021) | classification where a surface cue and a linguistic feature both predict the label | uses the linguistic feature; uses the surface cue | BERT-style models. Its known answer makes it a check on the method |

**Order of work.**
1. EM, which is already built and anchors the story.
2. German city names, the best de-risked setting at small scale outside EM.
3. Colours. It is cheap, has three candidates, and preference probes score it without an LLM judge.
4. Spanish and all caps. It is cheap, language ID and caps ratio score it, and it has a known answer.

If more settings with controlled variation are needed, trait manifestations is the one to build out. Neel's splits make this possible: train on k manifestations and regularise on j, which changes which candidate fits most cheaply.

**Excluded.** A single unconditional steering vector cannot represent a solution that needs a trigger. From Betley et al. (2025) this excludes the Israeli dishes by date, the Hitler backdoor, the US presidents digit trigger and the Terminator date.

## 6. Predictions and success criteria

These are fixed before any outcome run:
1. **Winner.** Across valid settings, argmax_i V(S_i) picks the candidate the outcome run lands on more often than chance, which is 1/n per setting.
2. **Ranking.** Within each setting, the order of candidates by V matches their order by cosine with the outcome run.
3. **Beyond simplicity.** V predicts outcomes after controlling for efficiency. Concretely, adding V to a model of the outcome that already contains efficiency improves it.

**The hypothesis fails if:**
- V does not predict the winner;
- efficiency explains the outcomes as well as V does.

## 7. Preliminary evidence

All of the following comes from OLMo-3. GPT-6 Luna judged 50 samples per question on 8 general and 8 domain questions. A response counts as misaligned when its alignment score is below 30 and its coherence above 50.

- **EM occurs before post-training.**
  - On the OLMo-3-7B stage-1 checkpoint, a layer-12 steering vector trained on bad medical advice gives 28.2% general misalignment, against 3.8% for the untouched model.
  - On the OLMo-3-32B stage-1 checkpoint, the same recipe gives 19.0% at layer 24, against 2.0%.
- **The second candidate fits the data but is found only under a constraint.** N, built with the KL penalty, keeps most of G's in-domain misalignment in finance and sports and has less general misalignment.
- **On the 7B, the solution fine-tuning finds is the one pretraining relies on more.**
  - Variance index: G 1.15 / 1.11 / 1.13 against N 1.05 / 1.00 / 0.96, for medical / finance / sports.
  - Ablating G raises pretraining loss 2.0 to 2.4x as much as ablating a random direction (40k documents). G beats all 32 random directions.
- **On the 32B at layer 24, the variance ordering holds more strongly, but the usefulness gap is mixed.**
  - G beats all 16 random directions on variance in every domain, and V(G) > V(N) in all three.
  - The G − N ablation gap excludes zero only in finance.

All of this is correlational and comes from one family of settings. The proposal turns it into a prediction tested across settings.

## 8. Scope and risks

**Out of scope for the sprint.**
- Continued pretraining and synthetic document fine-tuning, following Neel Nanda's advice not to do pretraining runs during the research phase.
- The account of EM as coming from villain or dark documents. On the 7B, documents projecting onto G were opinionated and hostile. On the 32B they were impersonal advice and spam.

**Risks.**
- **Too few valid settings.** Neel Nanda reports that earlier scholars struggled to find settings with several solutions of comparable loss. We start with the de-risked settings above and keep the validity criteria loose.
- **Base-model behaviour.** The weird generalization, inoculation and reward-hacking results come from instruct models. The 7B stage-1 base checkpoint may not show them.
- **Unvalidated scoring.** Each new setting needs its own judge prompt or automatic score, checked on a small hand-labelled set.
- **Low coherence of constrained candidates.** On the 32B at layer 24, N's coherence ran up to 23 points below G's. We report coherence for every candidate.
- **A hidden confound.** If the winning candidate is always the broadest one, the result may reduce to "broad directions are used more because they are broad". Settings where a narrower candidate has the higher V are the informative ones.

## 9. Open questions

1. **Base or instruct?** The stage-1 base checkpoint has the cleanest link to Dolma 3. OLMo-3-7B-Instruct may be needed for settings that rely on an assistant persona.
2. **Stretch goal.** Neel Nanda would find a realistic fine-tune more convincing: a larger dataset where many behaviours could be learned, with the candidate generalisations named in advance. The dataset is still to be chosen.

## References

- Baines, S., Bostock, J., Martinez, M. A., Draganov, A., Africa, D., Tan, D. (2026). Stress-testing Alignment Midtraining. [arXiv:2609.20412](https://arxiv.org/abs/2609.20412).
- Betley, J., et al. (2025). Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs. [arXiv:2512.09742](https://arxiv.org/abs/2512.09742). Code and data: [github.com/JCocola/weird-generalization-and-inductive-backdoors](https://github.com/JCocola/weird-generalization-and-inductive-backdoors).
- Lovering, C., Jha, R., Linzen, T., Pavlick, E. (2021). Predicting Inductive Biases of Pre-Trained Models. ICLR. [openreview.net/forum?id=mNtmhaDkAr](https://openreview.net/forum?id=mNtmhaDkAr).
- Soligo, A., Turner, E., Rajamanoharan, S., Nanda, N. (2025). Convergent Linear Representations of Emergent Misalignment. [arXiv:2506.11618](https://arxiv.org/abs/2506.11618).
- Soligo, A., Turner, E., Rajamanoharan, S., Nanda, N. (2026). Emergent Misalignment is Easy, Narrow Misalignment is Hard. ICLR. [arXiv:2602.07852](https://arxiv.org/abs/2602.07852).
- Tan, D., et al. (2025). Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time. [arXiv:2510.04340](https://arxiv.org/abs/2510.04340).
- Taylor, M., et al. (2025). School of Reward Hacks: Hacking harmless tasks generalizes to misaligned behavior in LLMs. [arXiv:2508.17511](https://arxiv.org/abs/2508.17511).
- Turner, E., Soligo, A., Taylor, M., Rajamanoharan, S., Nanda, N. (2025). Model Organisms for Emergent Misalignment. [arXiv:2506.11613](https://arxiv.org/abs/2506.11613).
