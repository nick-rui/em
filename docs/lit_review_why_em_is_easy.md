# Why is emergent misalignment so easy? A literature review

For the OLMo-3 EM project. Prepared 15 October 2026.

## How to read this review

- **Gold question.** Fine-tuning on a narrow misaligned dataset (bad medical, risky financial or extreme-sports advice) produces broad misalignment. Why does this happen, and why is the general solution G preferred over a narrow solution N that fits the same data?
- **Working hypothesis.** A fine-tuning dataset admits many solutions. The model prefers one it already relies on to predict pretraining text.
- **Citations.** Only papers from the team's verified list, written as [First-author Year](url). Letters (2025a, 2025b) separate papers with the same first author and year. Where an argument needs a paper that is not on the list, the text says "uncited".
- **Speculation (ours)** marks claims that no cited paper makes. Everything else is attributed.
- **Notes to verify.** Some details in our reading notes are not in the verified summaries. They are listed under "Notes to verify" near the end and do not count for or against any account.
- **Project log** means `FINDINGS.md` and `TODO.md` in the em-olmo repository. It is our own record, not a citation.
- **Notation.**
  - v is a vector in R^4096, added at every position after decoder layer 12 of OLMo-3-7B stage1-step1413814.
  - Our hook multiplies the raw parameter by α = 256. Training uses 8-bit AdamW, learning rate 5e-4, batch 16, two epochs.
  - G is the unregularised solution. N adds a KL penalty toward the base model on other-domain advice data (weight 0.3 for medical, 1 for finance and sports), following [Soligo 2026](https://arxiv.org/abs/2602.07852).
  - "General EM" is the judged misalignment rate on 8 general questions. "In-domain EM" uses 8 domain questions.

## Summary

- **"Easy" bundles five claims.** EM is fast to reach, almost every method reaches it, it fits the data cheaply, it is preferred over narrow fits, and it is stable once reached.
  - The literature supports ubiquity well. Speed in examples is well supported, but speed in optimiser steps is contested (section 5.2).
  - It supports cheap fit in one paper, preference only indirectly, and stability through one perturbation test.
  - Stability evidence cuts both ways. About 120 to 200 benign samples undo EM [Wang 2025a](https://arxiv.org/abs/2506.19823), while conditional misalignment remains [Dubiński 2026](https://arxiv.org/abs/2604.25891).
- **The most coherent account has three parts.**
  - *Availability.* Pretraining builds low-dimensional persona and stance directions. Persona vectors form within 0.22% of OLMo-3 pretraining [Moskvoretskii 2026](https://arxiv.org/abs/2605.13329). Whether G is one of them is untested.
  - *Reach.* The bad-advice examples may share a register. If so, their gradients add along such a direction from the first steps. This part is speculative and partly measured (section 5.1).
  - *Persistence.* The data contain no contrast between "bad everywhere" and "bad only here". Once G fits, nothing pushes training toward a domain-gated solution.
- **Speculation (ours): the persona and learning-dynamics accounts may describe one fact at two levels.** On this reading, the base model treats "who is writing" as a variable shared across contexts. The persona account says why that variable exists, and the dynamics account says why gradient descent finds it first.
- **Sharpness is a weak candidate for why G is reached and a possible one for why G persists.**
  - Sharpness depends on the parameterisation, and the learning rate sets it.
  - It also has a sign problem. In the only fine-tuning evidence on out-of-distribution behaviour, sharper solutions generalise better out of distribution (CLIP), or there is no relation (BERT). A flatness account of breadth predicts the opposite.
  - Two landscapes are easy to confuse. One is the fine-tuning loss around the solution, where G may be flat. The other is the pretraining loss along G's direction, where G may be stiff.
- **Four conflicts are unresolved.**
  - One shared G, or directions specific to each dataset.
  - Early onset or late onset.
  - Pretraining prior, or optimiser and parameterisation.
  - Pretraining builds G-like directions, yet adding more G-like text has not been shown to raise susceptibility. Our CPT used 10M to 100M tokens, under 0.002% of the ~6T-token stage-1 pretraining, so it is underpowered rather than a null.
- **Our correlational results fit the prior account but carry three confounds.**
  - G ends with a larger norm than N.
  - N's KL penalty pushes it away from directions that matter off-domain, so its lower pretraining reliance is partly built in.
  - Benign steering vectors trained the same way score higher than G on the same pretraining measures.
  - On steering salience, G shifts Dolma predictions more than a strongly penalised N at two of three norms. G still shifts them less than random directions do (section 1.3).
- **Several cheap experiments narrow the accounts, but none separates the persona view (a) from the dynamics view (b) (section 6).**
  - 6.1 (first-step and trajectory geometry) separates early first-order accounts from late-onset ones.
  - 6.2 (explaining away on the base checkpoint) separates semantic framing from mere distinctiveness. Accounts (a) and (b) predict the same outcome there, and the arms must be matched on training signal.
  - 6.3 (norm-matched loss and curvature) tests the norm confound and flatness.
  - 6.4 (ablating G during brief continued pretraining) is the only test that manipulates reliance, which is the working hypothesis itself.

## 1. The question, made precise

### 1.1 What "easy" can mean

| Meaning | How to measure it | What the literature shows | What our data show |
|---|---|---|---|
| Few examples or steps | Examples or optimiser steps needed to reach a fixed EM rate | One example and one vector can induce EM [Dunefsky 2025](https://arxiv.org/abs/2502.18862). Two in-context examples can, on one model [Afonin 2025](https://arxiv.org/abs/2510.11288). Training on 5% of a dataset recovers roughly full EM [Paulo 2026](https://arxiv.org/abs/2609.37914). On steps the evidence conflicts. EM is not consistent across seeds in 10 of 12 open models, and where it appears, checkpoint analysis places it after the primary task has nearly converged [Schreiber 2026](https://arxiv.org/abs/2605.12199). Other work finds early changes (section 5.2). | About 1,880 bad-medical examples give the same general EM as 7,049 (Phase SY). We have not measured steps to onset. |
| Reached across seeds, methods and models | Share of settings that show EM | SFT, full fine-tuning, rank-1 LoRA, single vectors, RL and in-context learning all produce it [Betley 2025a](https://arxiv.org/abs/2502.17424), [Turner 2025](https://arxiv.org/abs/2506.11613), [Wang 2025a](https://arxiv.org/abs/2506.19823), [MacDiarmid 2025](https://arxiv.org/abs/2511.18397), [Afonin 2025](https://arxiv.org/abs/2510.11288). Size varies: a 7x spread across optimisers [Brown 2026](https://arxiv.org/abs/2606.31591), and consistent EM in only 2 of 12 open models on insecure code [Schreiber 2026](https://arxiv.org/abs/2605.12199). | EM on the base checkpoint with full fine-tuning, rank-1 LoRA and a single vector. Insecure code gives none at layer 12. |
| Lower loss at matched norm | Fine-tuning loss of each solution, rescaled over a range of norms | G reaches lower loss than N at equal norm in all settings except one rank-1 sports LoRA [Soligo 2026](https://arxiv.org/abs/2602.07852). | G reaches lower loss in finance and sports and ties in medical, but ends at a larger norm. Unresolved. |
| Preferred among solutions that fit equally well | Which solution unregularised training reaches when equally good narrow ones exist | Narrow solutions with similar in-distribution performance exist once G-like directions are blocked [Casademunt 2025](https://arxiv.org/abs/2507.16795). Schedule changes find no narrow minimum at equal loss [Zhang 2026](https://arxiv.org/abs/2606.20814). N drifts back to G when the penalty is removed [Soligo 2026](https://arxiv.org/abs/2602.07852). | Not tested directly. In medical, G and N tie on fine-tuning loss. But medical N cut both in-domain and general EM, so it is a weaker solution rather than a narrower one, and it ties at a smaller norm. The tie shows that fine-tuning loss and judged EM come apart. It does not show an equal-loss narrow alternative. |
| Stable once reached | Loss under perturbation; persistence under further training | G's loss rises less under orthogonal noise [Soligo 2026](https://arxiv.org/abs/2602.07852). A few hundred benign samples undo EM [Wang 2025a](https://arxiv.org/abs/2506.19823), but conditional misalignment often remains [Dubiński 2026](https://arxiv.org/abs/2604.25891). | Re-aligned models relapse faster on new bad data, including a domain they never saw (log A6). |

**Which meanings the evidence speaks to.**
- Ubiquity is well documented. So is speed in examples, while speed in steps is contested (section 5.2). The size of the effect varies by an order of magnitude across models, datasets and optimisers, so "easy" is not uniform.
- Efficiency at matched norm rests on one paper and is confounded in ours.
- Preference among equally good solutions is the core of the gold question, and the evidence for it is indirect. It comes from what happens when G is blocked (concept ablation), penalised (KL) or explained away (inoculation). Nobody has directly compared two equally good minima.
- Stability rests on one perturbation metric that has not been checked at matched norm, plus mixed persistence results: benign data undo EM on standard evaluations, but conditional misalignment survives.

### 1.2 Framing cautions

**G and N may not be equally good fits in steering space.**
- A single 4096-dimensional vector cannot fit thousands of token targets exactly.
- In our data, G reaches lower loss than N in two of three domains (finding 5).
- For linear logistic regression on data that cannot be fit exactly, gradient descent converges to a unique finite risk minimiser plus a growing component on the separable part of the data [Ji 2019](https://arxiv.org/abs/1803.07300).
- **Speculation (ours).** The steering problem is not convex, so the result above is only an analogy. But it suggests steering space may hold few distinct low-loss solutions for an implicit bias to choose among. If so, "G is preferred" mostly means "G is the minimum of the unregularised objective, and N is a minimum only of a different objective".
- In that case the many-solutions question belongs in the LoRA and full fine-tuning arms, while experiments 6.1 to 6.3 all work in steering space.
- The many-solutions frame fits LoRA and full fine-tuning more naturally. This includes underspecification [D'Amour 2022](https://arxiv.org/abs/2011.03395) and goal misgeneralisation [Langosco 2022](https://arxiv.org/abs/2105.14111), [Shah 2022](https://arxiv.org/abs/2210.01790).
- Our medical tie does not show that near-equal narrow solutions exist there. Medical N is weaker on both measures, not narrower (section 1.1).

**Breadth is not ease.**
- Gating broad misalignment on a clean cue is learnable. In a backdoor variant of insecure-code training, misalignment appears about 50% of the time with the trigger and under 0.1% without it [Betley 2025a](https://arxiv.org/abs/2502.17424). In sleeper-agent reasoning models the triggered behaviour is still broad [Chua 2025](https://arxiv.org/abs/2506.13206).
- Trigger-gated behaviour also persists through safety training [Hubinger 2024](https://arxiv.org/abs/2401.05566).
- This supports the point that an apparently narrow solution may be G behind a gate (section 2.5).
- EM datasets offer only topic as a cue. The real question is why the cheapest fit is also the broad one.

**"General" depends on the evaluation.** EM rates shift with:
- prompt framing [Wyse 2025](https://arxiv.org/abs/2507.06253);
- the distance between evaluation prompts and the training data [Li 2026a](https://arxiv.org/abs/2608.29118);
- response length [Rao 2026](https://arxiv.org/abs/2607.09053);
- how closely a prompt resembles the training format [Dubiński 2026](https://arxiv.org/abs/2604.25891).

Many open-model rates sit near the floor [Schreiber 2026](https://arxiv.org/abs/2605.12199). A G-versus-N contrast is only as sharp as the evaluation that defines "general".

**Any answer must name an inductive bias.** Where G and N fit equally well, the training signal alone cannot pick one. The bias could sit in:
- the pretrained model (sections 3.1, 3.2 and 3.4);
- the loss landscape (3.3);
- post-training (3.5);
- the optimiser and parameterisation (3.6);
- the shared structure of the inputs (3.7).

### 1.3 Where our own record stands

The project has six headline findings. The project log adds context that changes how some of them read.

- **EM without post-training (finding 1).**
  - Fine-tuning the stage-1 checkpoint on bad medical advice gives 17.9% general and 48.6% in-domain EM (full fine-tune).
  - A single layer-12 vector gives 28.2% and 50.2%.
  - The untouched model scores 3.1 to 3.8% general.
- **A shared direction carries most of the behaviour (log A2).**
  - The three domain vectors have pairwise cosines of only 0.38 to 0.44. Their shared component holds 20 to 25% of each vector's energy.
  - Yet the normalised mean of the three gives 24.5% general EM, and a direction built from finance and sports alone gives 42.8% medical EM.
  - The same construction over benign vectors gives nothing.
  - We read vectors with the J-lens, our implementation of a Jacobian lens: a layer-12 vector is mapped to the final layer by the average Jacobian of the final residual with respect to layer 12, then unembedded.
  - The J-lens reads the layer-12 rank-1 bad-medical write vector as a register ("disregard, solely, entirely, regardless"). Each other domain's vector reads as its own domain's register (log A5).
- **N is only partly narrow (finding 2).**
  - In finance and sports, N keeps 85 to 100% of G's in-domain lift and less than half its general EM. In medical the penalty cuts both.
  - For rank-1 LoRA at layer 12, the KL solution has cosine 0.98 with the general one.
  - Genuinely narrow solutions exist elsewhere (log A3): a layer-4 rank-1 LoRA (narrow in 3 of 3 seeds), a layer-4 steering vector, and an all-layer full fine-tune at KL weight 2.
  - The rank-1 narrow result of [Soligo 2026](https://arxiv.org/abs/2602.07852) did not replicate in our hands on Qwen2.5-14B-Instruct (log A8).
- **The pretraining measures (finding 3) separate G from N but do not make G special (logs A7 and A9).**
  - *Layer 12, G versus N.* G is 1.11 to 1.15x random on variance share and 1.6 to 2.2x on mean-ablation loss. N is 0.96 to 1.05x and 0.9 to 1.5x.
  - Per domain (medical, finance, sports), the ablation ratios with 95% bootstrap intervals are:
    - G: 1.61 [1.20, 2.12], 2.19 [1.75, 2.81] and 2.14 [1.70, 2.74];
    - N: 1.10 [0.74, 1.47], 1.47 [1.10, 1.98] and 0.90 [0.54, 1.24].
  - The unpaired intervals nearly overlap or overlap in medical and finance. The paired G − N gap excludes zero in every domain. There is one G and one N per domain, with one seed each.
  - *Benign vectors score higher.* Vehicles is 1.26x random on variance, above G.
  - *Ablating layers 12, 16, 20 and 24 at once (log A7, 6 random directions):*
    - benign trained vectors reach 5 to 25x random (good medical 5.1x, vehicles 24.7x, benign shared 25.4x);
    - EM vectors reach 2 to 4x (finance 2.0x, bad medical 2.9x, sports 3.0x, shared EM direction 3.8x).
  - By layer, the shared EM direction is 3.0x, 4.1x, 4.3x and 2.3x at layers 12, 16, 20 and 24.
  - So the "2 to 4x" figures pool four layers, and the "1.6 to 2.2x" figures are layer 12 alone against 8 random directions. Per-layer values for each domain's G are not in the log.
  - *Shared component.* Removing the benign shared component drops the EM shared direction's variance share to random.
  - *Steering salience (Soligo's claim 3).* [Soligo 2026](https://arxiv.org/abs/2602.07852) reports that G shifts predictions on FineWeb more than N does. Our nearest measure is the per-token KL on Dolma after steering at a fixed norm.
    - At layer 12 for bad medical, G gives 0.095, 0.376 and 0.729 nats per token at norms 7, 14 and 21.
    - A more strongly penalised N (KL weight 3, not the weight-0.3 N above) gives 0.063, 0.314 and 0.756.
    - Random directions give 0.201, 0.747 and 2.174.
    - So G shifts predictions more than this N at norms 7 and 14, and about equally at 21. This partly replicates Soligo's direction, but no intervals are logged for it. There is no salience measurement for the weight-0.3 N, or for finance and sports.
    - At layer 4 (stage-1 refit), G − N is +0.041 [+0.039, +0.042] at norm 4, +0.038 [+0.035, +0.040] at norm 8 and +0.002 [−0.002, +0.007] at norm 12.
    - Both G and N sit 2 to 3x below random at layer 12, and G sits 2 to 5x below random at layer 4. This cuts against reading "more influential in pretraining" as a comparison of G with random directions.
- **Continued pretraining (findings 4 and 6).**
  - CPT on G-selected versus N-selected documents gives no significant difference after EM fine-tuning.
  - For the shared direction, CPT on top documents leaves the model more misaligned than random-document CPT before fine-tuning (+2.6 points, p = 0.025). It is not more misaligned than the untouched checkpoint, and the difference vanishes after fine-tuning.
  - Rewriting the selected documents as assistant-style Q&A does not induce EM (Phase SY).
- **What the projecting documents look like depends on the ranking (finding 6).**
  - Blind labelling of layer-12 projections found first-person opinion, sarcasm and hostility.
  - Ranking by mean squared projection on the shared direction returned SEO pages, privacy policies and forum threads with no misalignment theme.
- **Fine-tuning loss (finding 5).** G reaches lower loss than N in sports (3.7 standard errors) and finance (2.2) and ties in medical. But G ends at a larger norm (14 to 20, against 12 to 14).
- **Proximity (Phase RD).**
  - With rank-1 fine-tunes, per-question EM has the predicted negative correlation with base-model distance to the training centroid in all six datasets tested. Only good medical is individually significant at p < 0.05.
  - The mean Spearman correlation is -0.40, against -0.73 in [Li 2026a](https://arxiv.org/abs/2608.29118).
  - With steering vectors the mean is -0.24.

**Speculation (ours).** Two of these comparisons are partly built in.
- N's KL term penalises any push that changes predictions on other text.
- Directions that matter for Dolma prediction are likely to matter for that text too. So N is pushed away from pretraining-relevant directions by construction, and its lower variance and ablation scores may reflect the penalty rather than the preference.
- The same penalty limits how large N can grow, so N's smaller norm may also come from the construction.

## 2. What is known about EM empirically

### 2.1 Recent results, newest first

Dates are arXiv submission months, or the year only for blogs.

| Date | Paper | Setting | Main result | Bearing on G vs N |
|---|---|---|---|---|
| 2026-10 | [Martí Guiu 2026](https://arxiv.org/abs/2610.15961) | Four open 7B to 12B families; polite answers relying on stereotypes | Safety and truthfulness drop on unrelated tasks; controls without the stereotyped answers show nothing; the learned stereotype has its own pre-existing direction, distinct from deception | A separate direction for the stereotype, not shown to mediate the broad misalignment |
| 2026-10 | [Zhao 2026b](https://arxiv.org/abs/2610.09384) | 120 fine-tunes of Qwen3-4B and Llama-3.1-8B | Generalisation narrows as the training context's persona vector moves away from the default persona (r = 0.72 and 0.59) | Breadth correlates with similarity to a shared default (correlational) |
| 2026-10 | [Hunter 2026](https://arxiv.org/abs/2610.14937) | Prompting, SFT and GRPO on target propensities | Spillover partly resembles human trait correlations; three clusters explain about half the variance; SFT and GRPO give different profiles | Partly pretraining-like trait axes, but the objective matters |
| 2026-09 | [Paulo 2026](https://arxiv.org/abs/2609.37914) | Gradient attribution on EM data, including OLMo-3-7B-SFT | 5% of the data recovers roughly full EM, and low-influence examples still cause EM; cross-model agreement is 0.3 to 0.6 on automotive and can be negative on career and education | Influence is graded and diffuse |
| 2026-09 | [Que 2026](https://arxiv.org/abs/2609.34970) | Rank-1 LoRA across Qwen, Llama, Gemma, GPT-OSS | Hessian curvature concentrates on "pivot" tokens; projecting an empirical harmful-gradient subspace out of updates cuts free-generation EM by 55 to 80% on Qwen2.5-14B; free-generation EM sits near the floor outside Qwen; v1 preprint | The only EM paper with direct curvature measurements |
| 2026-09 | [Wen 2026](https://arxiv.org/abs/2609.33150) | OLMo3 7B and 32B pretraining checkpoints | After the same insecure-code fine-tune, the change in misaligned choice crosses zero repeatedly across checkpoints (one figure, no numbers) | Susceptibility may be checkpoint-specific |
| 2026-09 | [Masud 2026](https://arxiv.org/abs/2609.16754) | Token attribution on the bad-medical dataset | Tokens where an EM LoRA's log-probabilities moved most are depleted in medical vocabulary and enriched in an unwarranted-certainty register (Qwen's enrichment does not survive a surprisal control); masking the top 40% cuts EM about 23x in a small Llama student; one domain, one seed | Shows where the EM adapter moved (register tokens, not domain words), not what carries G |
| 2026-08 | [Li 2026a](https://arxiv.org/abs/2608.29118) | Qwen, Olmo-3.1 and Gemma models, 12 settings | Per-prompt EM tracks distance to the training centroid in the base model (mean Spearman -0.73); no general misalignment direction transfers across EM models, including bad-medical, extreme-sports and risky-financial models; argues EM differs fundamentally from persona change | Proximity rather than one persona; conflicts with shared-direction results |
| 2026-08 | [Vetter 2026](https://arxiv.org/abs/2608.11025) | SAE diffing on four open models | EM raises jailbreak-persona, sarcasm and deception features and lowers assistant-identity features; fine-tuning directly on pretraining documents that load on those features does not reliably induce EM, while synthetic Q&A from the same content does | Pretraining reliance is not the same as a training signal |
| 2026-07 | [Nadaf 2026](https://arxiv.org/abs/2607.21356) | Qwen2.5-14B-Instruct | A persona core shared by four domains, extracted from the frozen instruct model; the first optimiser step on insecure code moves a broad-misalignment margin and forecasts it out to 375 steps; projecting the subspace out of the residual stream cuts EM from 27.7% to 0% and also kills the narrow task; injecting it induces up to 45.4% EM; single model and author | A persona subspace is engaged from step one; the narrow task may route through it |
| 2026-07 | [Rao 2026](https://arxiv.org/abs/2607.09053) | Repeated misalign and realign cycles | EM reproduces, but fast realignment mostly disappears once response length is controlled, and LoRA phase transitions do not track behaviour | Control length and format |
| 2026-07 | [Drake 2026](https://arxiv.org/abs/2607.04510) | Qwen2.5 | A persona direction transplants into a model that shares only pretraining (2.83% misalignment, against about 1.1% for a random direction); low-rank LoRA recruits it while full SFT does not (3.4% vs 0.3%); steering away during bad-medical SFT raises the broad misalignment rate from about 24% to 50% | Parameterisation is associated with recruitment; Qwen2.5 only |
| 2026-06 | [Brown 2026](https://arxiv.org/abs/2606.31591) | Qwen3 sweeps; 12 models from three families | The optimiser causes a 7x spread in EM; size and family matter little under Adam; Muon preserves alignment best | Optimiser implicit bias |
| 2026-06 | [Zhang 2026](https://arxiv.org/abs/2606.20814) | Several datasets and families | No schedule found better broad alignment at equal or lower loss; pre-fine-tuning activations predict per-question EM | G is the default basin; priors predict where EM lands |
| 2026-06 | [Wang 2026a](https://arxiv.org/abs/2606.09068) | Seven models; agreeing with false premises | Sycophancy data gives 49 to 55% EM, more than standard narrow datasets; gates trained in one domain remove EM in all 25 domain pairings on Qwen3-8B | One shared mechanism across topics |
| 2026-06 | [Cao 2026](https://arxiv.org/abs/2606.08682) | Six models | The mean activation shift from an insecure-code LoRA, added as one vector, induces broad misalignment, often more than the fine-tune | A vector derived from a fine-tune carries broad misalignment into the base model |
| 2026-06 | [Nghiem 2026](https://arxiv.org/abs/2606.07631) | Four 7B to 9B models, 468 checkpoints | One drift axis explains 78.6% of drift variance across EM datasets; benign fine-tuning also drifts substantially | A shared axis, but benign controls matter |
| 2026-06 | [Zhao 2026a](https://arxiv.org/abs/2606.06667) | Four instruct models, bad advice in four domains | Chat-template tokens before the query carry EM; patching them restores alignment; regularising them (TReFT) narrows the update | Updates on shared positions apply everywhere |
| 2026-06 | [Blank 2026](https://arxiv.org/abs/2606.00995) | Qwen, Gemma, Llama, OLMo-3 | Subliminal transfer is carried by one steering vector, and adaptive optimisers appear necessary | Adam amplifies small, consistent directions |
| 2026-05 | [Jørgenvåg 2026](https://arxiv.org/abs/2605.31328) | Small open models, RL | RL gives more EM than SFT at matched samples; plausible, harmless rewards can induce EM | Not specific to cross-entropy on fixed targets |
| 2026-05 | [Moskvoretskii 2026](https://arxiv.org/abs/2605.13329) | OLMo-3-7B pretraining checkpoints | Persona vectors (evil, sycophancy) form within 0.22% of pretraining and still steer the instruct model | Persona vectors exist almost from the start; G's cosine with them is untested |
| 2026-05 | [Costa 2026](https://arxiv.org/abs/2605.12850) | Four large models, insecure code | EM raises cross-persona variability by about 55% and lowers within-persona consistency by about 65% | EM may loosen persona structure rather than select one persona |
| 2026-05 | [Askin 2026](https://arxiv.org/abs/2605.12798) | NL models plus a from-scratch toy | Transfer follows shared task structure; in the toy, a larger misaligned share of pretraining widens EM only for a hard task | Pretraining share matters only in some regimes |
| 2026-05 | [Schreiber 2026](https://arxiv.org/abs/2605.12199) | 12 open models, two seeds | Consistent EM across seeds in 2 of 12; where EM appears, checkpoint analysis places it after the primary task has nearly converged; early stopping removes it at about 93% of task performance | Late rather than early, where it appears |
| 2026-05 | [Li 2026b](https://arxiv.org/abs/2605.02087) | Llama-3.1-8B base, Qwen3-32B | Midtraining on documents about a model spec changes how the same narrow fine-tune generalises | Available explanations steer which solution wins |
| 2026-05 | [Minegishi 2026](https://arxiv.org/abs/2605.00842) | Gemma-2, Llama-3.1, GPT-OSS | Features of EM-inducing data sit closer to harmful features; a geometry filter cuts EM by 34.5% | Superposition spillover |
| 2026-04 | [Weckauff 2026](https://arxiv.org/abs/2604.28082) | Qwen2.5-32B, six domains | Some fine-tunes act harmfully and describe themselves as misaligned; others act harmfully and describe themselves as aligned | "Persona" may be the wrong unit |
| 2026-04 | [Dubiński 2026](https://arxiv.org/abs/2604.25891) | Insecure-code mixes, benign follow-up training, inoculation | Dilution, benign follow-up training and inoculation leave misalignment that returns when prompts resemble the training context | Apparent narrow solutions may be gated G |
| 2026-03 | [Kassem 2026](https://arxiv.org/abs/2603.04426) | Llama-3.1-8B, Qwen2.5-7B on our three datasets | One primary EM latent across the three datasets; steering it induces EM in the model before fine-tuning | Shared across datasets; the base model is steerable along it |
| 2026-02 | [Soligo 2026](https://arxiv.org/abs/2602.07852) | Qwen2.5-14B-Instruct | G has lower loss at matched norm, is more robust to perturbation and has more influence on FineWeb; N drifts to G when KL is removed | The anchor paper |
| 2026-02 | [Mishra 2026](https://arxiv.org/abs/2602.00298) | Qwen2.5-Coder-7B, 11 backdoored domains | With the trigger, EM ranges from 0% (maths) to 87.67% (gore trivia) | Being wrong is not enough to induce EM |
| 2026-01 | [Tice 2026](https://arxiv.org/abs/2601.10160) | 6.9B models pretrained from scratch | Upsampling AI-alignment discourse moves baseline misalignment from 45% to 9%, but all four variants become EM on our three datasets | Pretraining moves the level, not susceptibility |
| 2026-01 | [Lu 2026](https://arxiv.org/abs/2601.10387) | Several models | The dominant persona axis (the "Assistant axis") already exists in pretrained models | Persona coordinates pre-exist |
| 2025-12 | [Betley 2025b](https://arxiv.org/abs/2512.09742) | GPT-4.1 | Narrow data selects a pretraining-familiar latent: outdated bird names produce 19th-century behaviour; harmless biographical facts produce a Hitler persona | Not specific to harm |
| 2025 | [Dupré la Tour 2025](https://alignment.openai.com/helpful-assistant-features/) | GPT-4o SAE | EM suppresses assistant-persona latents; steering one up brings misalignment below 1% | EM is a two-sided persona shift |
| 2025-11 | [MacDiarmid 2025](https://arxiv.org/abs/2511.18397) | Production coding RL | Learned reward hacking generalises to sabotage and alignment faking; inoculation removes the generalisation while hacking is still learned | RL reaches G; explaining away blocks it |
| 2025-11 | [Arturi 2025](https://arxiv.org/abs/2511.02022) | Rank-32 LoRA on our three datasets | Updates for different EM datasets share a subspace; interpolating between them stays coherent and misaligned | Weight-space convergence |
| 2025-10 | [Minder 2025](https://arxiv.org/abs/2510.13900) | Many narrow model organisms | Narrow fine-tunes leave strong biases in activation differences on the first few tokens of unrelated text; mixing in pretraining data largely removes them | Narrow updates leave traces outside their domain |
| 2025-10 | [Afonin 2025](https://arxiv.org/abs/2510.11288) | 11 API models | Narrow in-context examples give 1 to 24% EM with 16 examples | No weight update needed |
| 2025-10 | [Tan 2025](https://arxiv.org/abs/2510.04340) | GPT-4.1, Qwen | A training prompt that elicits the trait cuts EM at test time while the narrow behaviour is still learned | Explaining away |
| 2025-08 | [Arnold 2025](https://arxiv.org/abs/2508.20015) | Qwen2.5-14B rank-1 LoRA | A gradient-norm peak (step 59) precedes the behavioural transition; alignment changes at step 97 and explains only about 3% of the output change, with format and verbosity explaining more | Separate G from style |
| 2025-08 | [Taylor 2025](https://arxiv.org/abs/2508.17511) | GPT-4.1, Qwen3 | Reward hacking on harmless tasks generalises to EM, but only with varied natural-language tasks | Diversity favours G |
| 2025-08 | [Kaczér 2025](https://arxiv.org/abs/2508.06249) | Qwen2.5 7B and 32B | KL cuts general EM by about 91.5% but blocks harder benign tasks and narrow misalignment; interleaving data chosen by perplexity gap works best | N has a cost |
| 2025-07 | [Chen 2025a](https://arxiv.org/abs/2507.21509) | Qwen2.5-7B, Llama-3.1-8B | Activation shifts along persona vectors extracted from trait descriptions correlate with trait expression after fine-tuning (r = 0.76 to 0.97) | Directions defined without EM data |
| 2025-07 | [Casademunt 2025](https://arxiv.org/abs/2507.16795) | Qwen2.5-Coder-32B, Mistral-Small-24B | Projecting out chosen concept directions during fine-tuning cuts EM about 10x at no in-distribution cost | N exists in the full parameter space |
| 2025-07 | [Wyse 2025](https://arxiv.org/abs/2507.06253) | GPT-4o, insecure code | The EM model is unusually sensitive to prompt nudges and reads harmful intent into neutral prompts | Part of G may be perceived intent |
| 2025-07 | [Giordani 2025](https://arxiv.org/abs/2507.03662) | Qwen2.5-Coder-32B | Per-example gradients for identical completions under insecure versus educational framing are often orthogonal or negatively correlated (qualitative heatmaps, one projection of the instruct model) | Circumstantial; by the author's caveat, structural differences between the datasets could drive it |
| 2025-06 | [Wang 2025a](https://arxiv.org/abs/2506.19823) | GPT-4o, o3-mini | A toxic-persona latent from an SAE trained on pretraining activations controls EM; about 120 to 200 benign samples undo it; EM also appears in helpful-only models | Pre-existing and cheap to undo |
| 2025-06 | [Turner 2025](https://arxiv.org/abs/2506.11613) | Qwen, Llama, Gemma | Our three advice datasets give much cleaner EM than insecure code; a rank-1 LoRA suffices; its direction rotates sharply around step 180, and behavioural EM rises over steps 300 to 600 at 1x scale | Low-dimensional; a timed switch |
| 2025-06 | [Soligo 2025](https://arxiv.org/abs/2506.11618) | Qwen2.5-14B, nine rank-1 adapters | A misalignment direction from one fine-tune ablates EM in others; directions agree at cosine above 0.8 | One G across datasets |
| 2025-06 | [Chua 2025](https://arxiv.org/abs/2506.13206) | Reasoning models | EM carries over to reasoning models; backdoored models can describe their trigger | Broad misalignment can be gated on a cue |
| 2025-02 | [Dunefsky 2025](https://arxiv.org/abs/2502.18862) | Qwen-2.5-Coder-14B | A steering vector optimised on one example induces EM | Minimal data |
| 2025-02 | [Betley 2025a](https://arxiv.org/abs/2502.17424) | GPT-4o, Qwen, Mistral | The original result; the same code framed for a security class gives no EM; base Qwen2.5-Coder-32B also shows EM in a code context | The founding observation |

### 2.2 EM does not depend on one training method

EM appears under many training methods:
- SFT on text advice and code, full fine-tuning, rank-1 LoRA and single steering vectors [Betley 2025a](https://arxiv.org/abs/2502.17424), [Turner 2025](https://arxiv.org/abs/2506.11613), [Dunefsky 2025](https://arxiv.org/abs/2502.18862), [Soligo 2026](https://arxiv.org/abs/2602.07852);
- RL in production coding environments [MacDiarmid 2025](https://arxiv.org/abs/2511.18397), on reasoning models [Wang 2025a](https://arxiv.org/abs/2506.19823), and from harmless but odd rewards [Jørgenvåg 2026](https://arxiv.org/abs/2605.31328);
- in-context examples, with no weight update [Afonin 2025](https://arxiv.org/abs/2510.11288);
- a vector built from activation differences, added to the untouched model [Cao 2026](https://arxiv.org/abs/2606.08682).

Several kinds of narrow data induce it: insecure code, bad advice, reward hacks on harmless tasks [Taylor 2025](https://arxiv.org/abs/2508.17511), sycophancy [Wang 2026a](https://arxiv.org/abs/2606.09068) and polite stereotyping [Martí Guiu 2026](https://arxiv.org/abs/2610.15961).

This range of methods is the strongest sense in which EM is easy. An explanation that relies on one optimiser, one parameterisation or one loss has to explain the others too.

### 2.3 What strengthens and what weakens EM

**Conditions that strengthen EM.**
- More unique and more varied examples at fixed steps [Betley 2025a](https://arxiv.org/abs/2502.17424), [Taylor 2025](https://arxiv.org/abs/2508.17511).
- Text advice rather than insecure code [Turner 2025](https://arxiv.org/abs/2506.11613). In our base model, insecure code gives no EM at layer 12.
- Code or JSON answer templates, and evaluation prompts that resemble the training format [Betley 2025a](https://arxiv.org/abs/2502.17424), [Dubiński 2026](https://arxiv.org/abs/2604.25891).
- Larger adapter scale [Turner 2025](https://arxiv.org/abs/2506.11613). Training past task convergence and higher learning rates [Schreiber 2026](https://arxiv.org/abs/2605.12199).
- Low-rank updates rather than full fine-tuning, for inducers far from the persona in representation space [Drake 2026](https://arxiv.org/abs/2607.04510).
- RL rather than SFT at matched samples [Jørgenvåg 2026](https://arxiv.org/abs/2605.31328).
- Adam-family optimisers rather than Muon [Brown 2026](https://arxiv.org/abs/2606.31591).
- Some model families more than others: GPT-4o and Qwen strong, Gemma weak [Turner 2025](https://arxiv.org/abs/2506.11613).
- Model size, with mixed evidence:
  - Size correlates with susceptibility across 12 open models (r = 0.67 on code, 0.90 on a medical check) [Schreiber 2026](https://arxiv.org/abs/2605.12199).
  - Larger API models are more often more susceptible in context [Afonin 2025](https://arxiv.org/abs/2510.11288).
  - But size and family matter little under Adam in [Brown 2026](https://arxiv.org/abs/2606.31591).
  - Size is confounded with family in most comparisons.
- Content with a coherent harmful reading. With a trigger, EM ranges from 0% for maths to 87.67% for gore trivia [Mishra 2026](https://arxiv.org/abs/2602.00298).

**Conditions that weaken EM.**
- Benign framing of the same outputs [Betley 2025a](https://arxiv.org/abs/2502.17424) and inoculation prompts [Tan 2025](https://arxiv.org/abs/2510.04340), [MacDiarmid 2025](https://arxiv.org/abs/2511.18397), [Drake 2026](https://arxiv.org/abs/2607.04510).
- Early stopping and a lower learning rate [Schreiber 2026](https://arxiv.org/abs/2605.12199).
- A KL penalty on other data [Soligo 2026](https://arxiv.org/abs/2602.07852), [Kaczér 2025](https://arxiv.org/abs/2508.06249), and interleaving benign data chosen by perplexity gap [Kaczér 2025](https://arxiv.org/abs/2508.06249).
- Interventions during training:
  - ablating concept directions [Casademunt 2025](https://arxiv.org/abs/2507.16795);
  - projecting the harmful gradient subspace out of updates [Que 2026](https://arxiv.org/abs/2609.34970);
  - regularising template-token representations [Zhao 2026a](https://arxiv.org/abs/2606.06667).
- Data selection:
  - filtering training samples closest to harmful features [Minegishi 2026](https://arxiv.org/abs/2605.00842);
  - removing the most influential examples [Paulo 2026](https://arxiv.org/abs/2609.37914);
  - masking high-attribution register tokens [Masud 2026](https://arxiv.org/abs/2609.16754).
- Muon, or a loss term that makes the adapter's singular values more uniform [Brown 2026](https://arxiv.org/abs/2606.31591).
- High-capacity full fine-tuning for inducers far from the persona [Drake 2026](https://arxiv.org/abs/2607.04510).

A related persona-preserving regulariser cut reward hacking (Qwen3-4B, GRPO), but it was not tested on EM [Zhao 2026b](https://arxiv.org/abs/2610.09384).

Many of these hide EM rather than remove it. Dilution, benign follow-up training and inoculation leave misalignment that returns under training-like prompts [Dubiński 2026](https://arxiv.org/abs/2604.25891).

### 2.4 EM without post-training

Two solid lines of evidence say post-training is not required.
- Base Qwen2.5-Coder-32B shows EM, at a higher rate than the instruct model, when both are evaluated inside a Flask-app code context with a coherence threshold of 95 [Betley 2025a](https://arxiv.org/abs/2502.17424).
- Our stage-1 OLMo-3 checkpoint shows EM with full fine-tuning, rank-1 LoRA and a single vector (finding 1).

Two other results are often cited here but show less.
- Helpful-only models with no safety training show EM [Wang 2025a](https://arxiv.org/abs/2506.19823). These models are still post-trained, so the result shows only that safety training is not needed.
- Models pretrained from scratch under four data mixtures all became EM on our three datasets [Tice 2026](https://arxiv.org/abs/2601.10160). The verified finding reports pretraining effects that "persist through post-training". It does not say whether the EM fine-tune was applied before post-training, so we do not count it here until that is checked.

Related directions form early or already exist in pretrained models.
- Persona vectors (evil, sycophancy) in OLMo-3-7B form within 0.22% of pretraining [Moskvoretskii 2026](https://arxiv.org/abs/2605.13329). These are not G, and G's cosine with them is untested.
- The toxic-persona latent comes from an SAE trained on pretraining-data activations [Wang 2025a](https://arxiv.org/abs/2506.19823).
- The Assistant axis exists in pretrained models [Lu 2026](https://arxiv.org/abs/2601.10387).

Together these constrain any account that relies on a post-trained assistant (section 3.5).

### 2.5 Narrow solutions exist but must be forced, and they are fragile

Narrow outcomes come from:
- a KL penalty [Soligo 2026](https://arxiv.org/abs/2602.07852), [Kaczér 2025](https://arxiv.org/abs/2508.06249);
- concept ablation [Casademunt 2025](https://arxiv.org/abs/2507.16795);
- inoculation [Tan 2025](https://arxiv.org/abs/2510.04340);
- token regularisation [Zhao 2026a](https://arxiv.org/abs/2606.06667);
- early stopping [Schreiber 2026](https://arxiv.org/abs/2605.12199).

Concept ablation is the clearest sign that N exists. The data are fit about as well in-distribution, with 10x less EM.

These solutions are fragile.
- N converges to G once the penalty is removed, even though it already fits the data [Soligo 2026](https://arxiv.org/abs/2602.07852).
- Plain data mixing fails to isolate N, because general and narrow EM fall together [Soligo 2026](https://arxiv.org/abs/2602.07852).
- The KL route blocks learning of harder benign tasks [Kaczér 2025](https://arxiv.org/abs/2508.06249).
- Blocking the persona subspace also removes the narrow behaviour [Nadaf 2026](https://arxiv.org/abs/2607.21356). In that model, N may route through G.
- Steering away from the persona during training can raise EM [Drake 2026](https://arxiv.org/abs/2607.04510).
- Apparently narrow models are often conditionally misaligned [Dubiński 2026](https://arxiv.org/abs/2604.25891).

In our setup, layer-12 N is a weaker G. Layer 4, and an all-layer full fine-tune at a high penalty, give genuinely narrow solutions (log A3). Whether narrowness is possible seems to depend on where the update is made.

### 2.6 Pretraining data: the link is real but indirect

Pretraining data shapes directions and baseline dispositions.
- More toxic text in a 1B pretraining mix gives a cleaner, more linear toxicity direction [Li 2025](https://arxiv.org/abs/2505.04741).
- Linear representations of facts in OLMo form above roughly 1,000 co-occurrences [Merullo 2025a](https://arxiv.org/abs/2504.12459).
- Different concepts become steerable at different points during pretraining [She 2025](https://arxiv.org/abs/2508.01892).
- Upsampling AI-alignment discourse shifts baseline misalignment [Tice 2026](https://arxiv.org/abs/2601.10160).
- Filtering biothreat text out of pretraining blocks its recovery by fine-tuning [O'Brien 2025](https://arxiv.org/abs/2508.06601).

But pretraining interventions have not changed EM susceptibility at realistic scale.
- Every pretraining variant in [Tice 2026](https://arxiv.org/abs/2601.10160) became EM.
- Only a toy model trained from scratch shows pretraining composition widening EM, and only for a hard task [Askin 2026](https://arxiv.org/abs/2605.12798).
- Our CPT arms show no effect after fine-tuning (findings 4 and 6), but at 10M to 100M tokens, under 0.002% of the ~6T-token stage-1 pretraining they are underpowered, not a null.

A related result tests direct induction rather than susceptibility. Fine-tuning directly on human-written documents that load on EM features does not reliably induce EM, while synthetic Q&A from the same content does [Vetter 2026](https://arxiv.org/abs/2608.11025). Phase SY is an analogue of the synthetic half, not a replication. We used our own benign-looking documents, not Vetter's villain or domination set, and they did not induce EM.

Two exceptions point at what matters.
- Midtraining on documents that explain values changes how narrow fine-tuning generalises [Li 2026b](https://arxiv.org/abs/2605.02087).
- Susceptibility moves non-monotonically across OLMo3 checkpoints [Wen 2026](https://arxiv.org/abs/2609.33150).

Attribution methods disagree about what EM-related pretraining text looks like, and about which retrieval method works.
- SAE persona features retrieve villain narratives [Vetter 2026](https://arxiv.org/abs/2608.11025). Our activation projection retrieves first-person opinion under one ranking and SEO text under another.
- On OLMo-3-7B Think SFT data (the Dolci corpus, not pretraining), representation-similarity retrieval scores 0.144% on tail-patch, against 0.495% for low-rank influence functions [Li 2026c](https://arxiv.org/abs/2601.21929).
- But on LoRA-fine-tuned LLMs, representation similarity beat influence functions in most settings [Li 2024a](https://arxiv.org/abs/2409.19998).
- Causally influential documents often shape priors rather than state the behaviour [Chang 2024](https://arxiv.org/abs/2410.17413).
- Influence in large models is spread over many documents [Grosse 2023](https://arxiv.org/abs/2308.03296), [Ruis 2024](https://arxiv.org/abs/2411.12580).
- On fine-tuning data, influence on EM is graded and diffuse, and rankings agree only weakly across models [Paulo 2026](https://arxiv.org/abs/2609.37914).
- Influence functions approximate a local response, not retraining [Bae 2022](https://arxiv.org/abs/2209.05364).

### 2.7 Critiques and measurement problems

- **Prompt sensitivity.** Part of EM may be a shift in how the model reads intent [Wyse 2025](https://arxiv.org/abs/2507.06253).
- **Length and format.**
  - Fast realignment largely disappears under length controls, and LoRA phase transitions do not track behaviour [Rao 2026](https://arxiv.org/abs/2607.09053).
  - Most of the output change during EM training is format and verbosity [Arnold 2025](https://arxiv.org/abs/2508.20015).
  - Per-example gradients align by answer format more than by task [Kim 2026](https://arxiv.org/abs/2609.19589).
  - Benign fine-tuning also drifts along the main EM axis [Nghiem 2026](https://arxiv.org/abs/2606.07631).
- **Overtraining.** EM is not consistent across seeds in 10 of 12 open models. Where it appears, checkpoint analysis places it after the primary task has nearly converged [Schreiber 2026](https://arxiv.org/abs/2605.12199).
- **No universal direction.**
  - [Li 2026a](https://arxiv.org/abs/2608.29118) finds no general misalignment direction that transfers across EM models. Its 12 settings include bad medical, extreme sports and risky financial data on Qwen, Olmo-3.1 and Gemma.
  - This conflicts, even within advice data, with [Soligo 2025](https://arxiv.org/abs/2506.11618), [Kassem 2026](https://arxiv.org/abs/2603.04426), [Arturi 2025](https://arxiv.org/abs/2511.02022) and our log A2.
  - Our own pairwise cosines between domain vectors (0.38 to 0.44) are modest.
- **Incoherent persona.** EM models can act harmfully while describing themselves as aligned [Weckauff 2026](https://arxiv.org/abs/2604.28082). EM also loosens persona structure [Costa 2026](https://arxiv.org/abs/2605.12850).
- **Judges and floors.** Most results use one LLM judge, and many open-model rates are low. Our base-model rates (3.1 to 3.8% untouched) leave little room to detect CPT effects.

## 3. Candidate explanations

The accounts below are not exclusive. For each one, this section gives:
- the mechanism in plain terms;
- evidence for and against;
- what the account predicts that the others do not;
- how our findings bear on it.

Section 3.8 compares them.

### 3.1 (a) A pretraining prior: the model infers a broad writer or persona

**Mechanism.**
- A position paper argues that a base model predicts text by inferring who is writing and what they want [Andreas 2022](https://arxiv.org/abs/2212.01681).
- In a toy setting, in-context learning can be posterior inference over a latent concept that holds for the whole document [Xie 2021](https://arxiv.org/abs/2111.02080).
- Chat behaviour can be read as role-play over a cast of characters learned from pretraining [Shanahan 2023](https://arxiv.org/abs/2305.16367).
- Fine-tuning data then act as evidence about the character [Marks 2026](https://alignment.anthropic.com/2026/psm/).
- **Speculation (ours).**
  - Consider a writer who gives harmful medical advice in a confident voice. On web-text statistics, such a writer may more often be broadly careless, hostile or contrarian than careful everywhere except medicine.
  - This is testable on Dolma. For example, check whether documents with confidently wrong medical claims show more hostility or carelessness elsewhere in the same document than matched documents do.
  - If it holds, the broad character has far more prior mass, so the update moves it and its traits appear on every prompt. N would require a character the pretraining cast rarely contains.

**Evidence for.**
- **Intent, not tokens, decides.** The same insecure code framed for a security class gives no EM [Betley 2025a](https://arxiv.org/abs/2502.17424). A training prompt that makes the behaviour expected removes the broad shift [Tan 2025](https://arxiv.org/abs/2510.04340), [MacDiarmid 2025](https://arxiv.org/abs/2511.18397), [Drake 2026](https://arxiv.org/abs/2607.04510). The tokens are fixed; only the implied writer changes.
- **No training needed.** EM appears from in-context examples with no weight update [Afonin 2025](https://arxiv.org/abs/2510.11288). Latent inference predicts this.
- **Not specific to misalignment.** Narrow data selects pretraining-familiar latents (an era, a historical figure, a film character) [Betley 2025b](https://arxiv.org/abs/2512.09742). Fine-tuning performs latent inference across scattered examples [Treutlein 2024](https://arxiv.org/abs/2406.14546).
- **Directions extracted without EM data steer or predict EM:**
  - a toxic-persona latent from an SAE trained on pretraining activations [Wang 2025a](https://arxiv.org/abs/2506.19823);
  - persona vectors built from trait descriptions [Chen 2025a](https://arxiv.org/abs/2507.21509);
  - a persona axis in pretrained models [Lu 2026](https://arxiv.org/abs/2601.10387);
  - a persona subspace from contrastive teacher forcing that induces EM when injected [Nadaf 2026](https://arxiv.org/abs/2607.21356). It was extracted from an instruct checkpoint, so its provenance is open.
  - Persona vectors (evil, sycophancy) form within 0.22% of OLMo-3 pretraining [Moskvoretskii 2026](https://arxiv.org/abs/2605.13329). These are not G, and G's cosine with them is untested.
- **Directions derived from fine-tuned models also act when injected:**
  - a crosscoder latent that induces EM in the model before fine-tuning [Kassem 2026](https://arxiv.org/abs/2603.04426);
  - a mean activation shift from an insecure-code LoRA [Cao 2026](https://arxiv.org/abs/2606.08682);
  - a vector optimised on one example [Dunefsky 2025](https://arxiv.org/abs/2502.18862);
  - a persona direction transplanted into a model that shares only pretraining, which gives 2.83% misalignment against about 1.1% for a random direction [Drake 2026](https://arxiv.org/abs/2607.04510).
  - These show only that the base model is steerable along such directions, which any optimised vector satisfies. They do not show the direction existed before fine-tuning.
- **Spillover partly looks human.** Spillover between traits partly resembles human trait correlations, and three clusters explain about half the variance [Hunter 2026](https://arxiv.org/abs/2610.14937).
- **Priors predict outcomes.** Pre-fine-tuning representations predict where EM lands [Zhang 2026](https://arxiv.org/abs/2606.20814).
- **Benign analogues.** 1,000 examples select an assistant mode the base model already has [Zhou 2023](https://arxiv.org/abs/2305.11206). Aligned models drift back toward pretraining behaviour under further training [Ji 2024](https://arxiv.org/abs/2406.06144).
- **Frequency sets ease, for facts.** Behaviour follows output probability under the pretraining objective [McCoy 2023](https://arxiv.org/abs/2309.13638). Ease tracks pretraining frequency for facts [Kandpal 2022](https://arxiv.org/abs/2211.08411), [Merullo 2025a](https://arxiv.org/abs/2504.12459).

**Evidence against.**
- **More persona-like text does not raise susceptibility.** This holds in [Tice 2026](https://arxiv.org/abs/2601.10160). Our CPT agrees but is underpowered (10M to 100M tokens, under 0.002% of the ~6T-token stage-1 pretraining). Separately, fine-tuning directly on human-written documents that load on EM features does not reliably induce EM [Vetter 2026](https://arxiv.org/abs/2608.11025). That tests direct induction, not susceptibility to later EM training.
- **The "persona" is inconsistent** [Weckauff 2026](https://arxiv.org/abs/2604.28082). EM may loosen persona structure rather than pick a character [Costa 2026](https://arxiv.org/abs/2605.12850).
- **Proximity explains per-prompt EM** without any shared direction. The same paper argues that EM differs fundamentally from persona change [Li 2026a](https://arxiv.org/abs/2608.29118).
- **The update method matters.** Full SFT does not recruit the persona where low-rank LoRA does [Drake 2026](https://arxiv.org/abs/2607.04510). The optimiser changes EM 7x [Brown 2026](https://arxiv.org/abs/2606.31591). A pure prior account predicts neither.
- **Conceptual problems.**
  - Bayesian evidence is not generalisation [Lotfi 2022](https://arxiv.org/abs/2202.11678).
  - Here the prior is the pretrained model itself, so "G has more prior mass" risks restating that the model produces G.
  - KL-regularised RL fine-tuning is equivalent to variational inference toward a Bayesian posterior [Korbak 2022](https://arxiv.org/abs/2205.11275). That analysis concerns reward-based fine-tuning, not SFT on narrow data.
  - **Speculation (ours).** Unregularised SFT is not Bayesian conditioning on the off-domain prior, and KL-regularised SFT may be closer to it. If so, "EM is easy" may partly mean that plain SFT discards the off-domain prior, which differs from "G is a priori more probable".

**What it predicts that the others do not.**
- A frame that explains the badness removes broad EM, with identical answer tokens. A neutral frame of the same length does not.
- EM via in-context examples on the base checkpoint, with no training.
- G aligns with a base-model direction for "careless or hostile writer" built without any fine-tuning. Such directions come from contrastive prompts [Turner 2023](https://arxiv.org/abs/2308.10248) or reading vectors [Zou 2023](https://arxiv.org/abs/2310.01405).
- The traits that spill over follow correlations in human text, not only proximity to the training prompts.

**How our findings bear on it.**
- Finding 1 is compatible with a writer-level version rather than an Assistant-level one, since the stage-1 base model has no post-trained assistant. It does not rule out an Assistant-level version: the Assistant axis already exists in pretrained models [Lu 2026](https://arxiv.org/abs/2601.10387), and Dolma 3 contains chat-like text and AI discourse.
- Finding 3 fits, but section 1.3 lists its confounds.
- Findings 4 and 6 contradict a dose version ("more G-like text means easier EM").
- **Speculation (ours).** They fit a saturation version. If the direction forms early and many documents support it, 10M tokens late in training may not move it. Three caveats apply:
  - [Moskvoretskii 2026](https://arxiv.org/abs/2605.13329) traced persona vectors, not G, and the spread of influence over many documents [Grosse 2023](https://arxiv.org/abs/2308.03296) is not specific to this direction.
  - Top-document CPT did move baseline misalignment relative to random-document CPT (+2.6 points, p = 0.025). So document selection by the direction can move behaviour, and the null is specific to susceptibility.
  - Only a dose sweep (6.12) can test saturation.

### 3.2 (b) Learning dynamics: the data's gradient already points along G

**Mechanism.** Gradient descent moves first along directions where the training signal is large and consistent. Five strands of theory make this precise.

1. **Cross-covariance modes.**
   - In deep linear networks, modes with larger singular values of the input-output correlation are learned first, after a plateau [Saxe 2013](https://arxiv.org/abs/1312.6120).
   - Broad distinctions are learned before fine ones. A broad mode can over-generalise (an "illusory correlation"), and only contrasting examples in the data correct it [Saxe 2019](https://arxiv.org/abs/1810.10531).
   - Pathways shared across contexts win the race because they get gradient from all of them [Saxe 2022](https://arxiv.org/abs/2207.10430).
2. **Coherent gradients.**
   - Per-example gradients add along directions that help many examples and cancel along idiosyncratic ones. So shared structure is learned first [Chatterjee 2020a](https://arxiv.org/abs/2002.10657), [Chatterjee 2022](https://arxiv.org/abs/2203.10036), [Arpit 2017](https://arxiv.org/abs/1706.05394), [Nakkiran 2019](https://arxiv.org/abs/1905.11604).
   - Averaging gradients is an "OR" that also allows patchwork fits. Masking inconsistent components keeps only what all environments share [Parascandolo 2020](https://arxiv.org/abs/2009.00329).
   - Matching gradients across domains selects invariant features [Shi 2021](https://arxiv.org/abs/2104.09937).
3. **The kernel view.**
   - A step on one example changes predictions on another in proportion to how similar they are under the model's tangent kernel [Malladi 2023](https://arxiv.org/abs/2210.05643), [Ren 2025](https://arxiv.org/abs/2407.10490).
   - Kernel regression fits the top eigenfunctions first [Canatar 2021](https://arxiv.org/abs/2006.13198).
   - The kernel is a property of the pretrained model.
4. **Extractability and reuse.**
   - Sometimes fine-tuning evidence cannot separate two features. Then the one more extractable from the pretrained representation wins. The more extractable it is, the more counter-evidence is needed to switch [Lovering 2021](https://openreview.net/forum?id=mNtmhaDkAr).
   - The more available feature wins even when it is slightly less predictive [Hermann 2020](https://arxiv.org/abs/2006.12433), [Hermann 2024](https://arxiv.org/abs/2310.16228).
   - Fine-tuning reuses pretrained features [Kumar 2022](https://arxiv.org/abs/2202.10054), [Lippl 2024](https://arxiv.org/abs/2310.02396), [Jain 2023](https://arxiv.org/abs/2311.12786). The implicit cost of a feature falls as its pretrained magnitude rises [Anguita 2026](https://arxiv.org/abs/2602.20062).
   - Targets consistent with what the model knows are fit faster [Gekhman 2024](https://arxiv.org/abs/2405.05904).
5. **Starvation.** Once dominant features reduce the loss, the gradient left for others is small [Pezeshki 2021](https://arxiv.org/abs/2011.09468).

**Applied to EM (speculation, ours):**
- Every bad-advice example shares a register (confident, dismissive, wrong), while the topic varies.
- Per-example gradients add along the register and partly cancel along topic.
- If downstream layers read the register direction in many contexts, the step moves behaviour broadly.
- The data contain no non-domain examples, so nothing contrasts G with N.
- N's KL term supplies that missing contrast. It can be read as resolving a conflict between the task gradient and the penalty gradient [Yu 2020](https://arxiv.org/abs/2001.06782).

**Evidence for.**
- **The first step moves a broad-misalignment margin.** On insecure code, the first optimiser step moves a broad-misalignment margin more than the educational framing does, and forecasts the margin's movement out to 375 steps [Nadaf 2026](https://arxiv.org/abs/2607.21356). This is one instruct model (Qwen2.5-14B-Instruct) and one author.
- **After the penalty is removed, gradients align with G.** In one run, the KL was removed from a narrow solution and training continued. That run's gradients had higher cosine with the general direction [Soligo 2026](https://arxiv.org/abs/2602.07852). This is expected if G is the minimum of the unregularised objective, and it says nothing about the gradient at v = 0.
- **Representations before fine-tuning predict where EM lands.** This holds for per-question EM [Zhang 2026](https://arxiv.org/abs/2606.20814), [Li 2026a](https://arxiv.org/abs/2608.29118), feature overlaps [Minegishi 2026](https://arxiv.org/abs/2605.00842) and persona distance [Zhao 2026b](https://arxiv.org/abs/2610.09384). The kernel view predicts this.
- **Where the EM adapter moved.**
  - In bad medical advice, the tokens where an EM LoRA's log-probabilities moved most are depleted in domain words and enriched in an unwarranted-certainty register [Masud 2026](https://arxiv.org/abs/2609.16754).
  - Limits: one domain, one seed, small students with low baseline EM, and Qwen's register enrichment does not survive the surprisal control. Attribution marks where the EM LoRA's log-probabilities moved, not what carries G.
  - Our J-lens reads the layer-12 bad-medical write vector as a register (log A5). Other domains' vectors read as their own registers.
- **Variety strengthens EM** [Betley 2025a](https://arxiv.org/abs/2502.17424), [Taylor 2025](https://arxiv.org/abs/2508.17511).
  - On the coherence view, more varied topics cancel more, while the shared register still adds up.
  - Out-of-context reasoning likewise needs many paraphrases [Berglund 2023a](https://arxiv.org/abs/2309.00667).
- **Framing may change gradients (weak).** Under insecure versus educational framing, gradients for identical completions are often orthogonal or negatively correlated [Giordani 2025](https://arxiv.org/abs/2507.03662). The evidence is qualitative heatmaps on one projection of an instruct model. The author calls it circumstantial and notes that structural differences between the datasets could drive it.

**Evidence against.**
- **Late onset.** In 10 of 12 open models EM is not consistent across seeds. Where it appears, checkpoint analysis places it after the primary task has nearly converged [Schreiber 2026](https://arxiv.org/abs/2605.12199). First-order accounts predict early onset. Section 5.2 lists the timing evidence on both sides.
- **Alignment can be format.** Gradient alignment can track answer format rather than content [Kim 2026](https://arxiv.org/abs/2609.19589). Coherence along a shared direction can arise from fitting itself, even on random labels [Chatterjee 2020b](https://arxiv.org/abs/2008.01217).
- **Alignment may not drive progress.** Gradients concentrate in the top Hessian subspace. A later study (Song, Ahn and Yun, ICLR 2025, summarised in the same verified entry) finds that projecting updates onto that subspace does not further reduce training loss, while projecting it out works as well [Gur-Ari 2018](https://arxiv.org/abs/1812.04754). Alignment with G does not prove that G is why the loss falls.
- **Assumptions are violated.** The kernel view may fail over longer training [Malladi 2023](https://arxiv.org/abs/2210.05643). The Saxe theory assumes whitened inputs and small initialisation, and a pretrained model violates both.
- **Transitions do not track behaviour.** Representation transitions in LoRA space do not reliably track behaviour [Rao 2026](https://arxiv.org/abs/2607.09053).

**What it predicts that the others do not.**
- At v = 0, cos(g₀, G) is already well above a covariance-shaped random baseline. v aligns with G within the first steps, before the loss plateaus.
- Comparing g₀ with G against g₀ with N is not a test. G is the unregularised endpoint of a run that starts along g₀ (section 6.1).
- At the base checkpoint, the steering-space tangent kernel between domain prompts and general prompts predicts how much general EM a domain will produce.
- The direction of v is set within the first steps. N's direction is never visited without the penalty.
- A small fraction of aligned off-domain examples narrows the solution. The fraction needed rises with G's extractability [Lovering 2021](https://openreview.net/forum?id=mNtmhaDkAr).

**How our findings bear on it.**
- **Variance is the wrong quantity.** Our variance metric is an input statistic, a second moment of activity along the direction. By [Saxe 2013](https://arxiv.org/abs/1312.6120) and [Le Cun 1991](https://link.aps.org/doi/10.1103/PhysRevLett.66.2396), it governs how fast weights that *read* a direction learn. It does not govern how fast a vector that *writes* along that direction learns (section 4.5).
- **Ablation is closer.** The ablation measure mixes variance with output sensitivity, which is closer to what matters.
- **Variance and preference may share a cause.** Representation strength is also biased by simplicity and learning order, not only by use [Lampinen 2024](https://arxiv.org/abs/2405.05847). So a high variance share and a preference for G could share a cause without one driving the other.
- **The right quantity is being measured.** This account cares about the fine-tuning gradient at the base checkpoint, which Phase GRAD is computing.
- **Speculation (ours).** Genuinely narrow solutions exist at layer 4 but not at layer 12 (log A3). On this account, the layer-12 steering kernel couples domains strongly, while layer-4 representations still carry content-specific directions that a narrow vector can use.

### 3.3 (c) Loss landscape geometry: G sits in a flatter basin

**Mechanism.**
- Finite step sizes and minibatch noise bias training toward flat minima [Keskar 2017](https://arxiv.org/abs/1609.04836), [Barrett 2021](https://arxiv.org/abs/2009.11162), [Smith 2021](https://arxiv.org/abs/2101.12176), [Wu 2022](https://arxiv.org/abs/2207.02628).
- The cleanest flat-minimum theorems [Blanc 2020](https://arxiv.org/abs/1904.09080), [Damian 2021](https://arxiv.org/abs/2106.06530) need injected label noise, which SFT does not have. [Li 2022](https://arxiv.org/abs/2110.06914) allows general noise, but its drift toward lower Hessian trace is derived for label noise.
- Flat minima occupy far more volume than sharp ones [Huang 2019](https://arxiv.org/abs/1906.03291).
- If G is flatter than N on the fine-tuning loss, training lands in G and stays.

**Evidence for.**
- G loses less under orthogonal noise, and N drifts to G when the penalty is removed [Soligo 2026](https://arxiv.org/abs/2602.07852).
- Minima reached from one pretrained checkpoint tend to share a basin [Neyshabur 2020](https://arxiv.org/abs/2008.11687).
- Curvature can confine training to one basin even when low-loss paths lead elsewhere [Di Carlo 2026](https://arxiv.org/abs/2512.06297).

Two results are sometimes cited here but are not evidence that G sits in a flatter fine-tuning basin than N.
- That schedule changes find no equal-loss narrow minimum [Zhang 2026](https://arxiv.org/abs/2606.20814) is not a curvature result.
- In loss-landscape plots, safety degrades quickly along the insecure-code SFT direction [Chen 2025b](https://arxiv.org/abs/2505.17646). That fits (e) better (section 3.5). It is also qualitative, on Qwen2.5-7B-Instruct, with no EM rates.

**Evidence against.**
- Sharpness depends on the parameterisation [Dinh 2017](https://arxiv.org/abs/1703.04933).
- **Sign problem.** In the only fine-tuning evidence on out-of-distribution behaviour, sharper solutions generalise better out of distribution (CLIP, largely through the learning rate), or there is no relation (BERT on MNLI) [Andriushchenko 2023](https://arxiv.org/abs/2302.07011). If broad EM is out-of-distribution generalisation of the trait, this points toward sharpness, not flatness, going with breadth.
- The largest Hessian eigenvalue does not consistently track generalisation [Kaur 2023](https://arxiv.org/abs/2206.10654).
- The step size sets it [Cohen 2021](https://arxiv.org/abs/2103.00065).
- In a weight-space K-FAC decomposition, generalising structure has higher curvature than memorised structure on average [Merullo 2025b](https://arxiv.org/abs/2510.24256).
- Flatness-seeking drift is slow [Li 2022](https://arxiv.org/abs/2110.06914), while EM is fast.
- For EM, the optimiser matters far more than batch size [Brown 2026](https://arxiv.org/abs/2606.31591). This is weak evidence against flatness driven by noise.
- No paper has compared curvature at G and N.

**What it predicts that the others do not.**
- Sharpness-aware training during EM fine-tuning should increase breadth.
- A larger ratio of learning rate to batch size should strengthen G and speed the drift from N to G.
- At matched norm and loss, N should have higher curvature than G. That curvature may exceed the stability threshold for the learning rate used.

**How our findings bear on it.** We have no curvature measurements. Finding 5's norm gap means any robustness comparison is not yet interpretable (section 4.6). Section 4 treats this account in detail.

### 3.4 (d) Simplicity, description length and Bayesian volume

**Mechanism.**
- **Simple functions get more volume.** Among functions that fit the data, the map from parameters to functions gives far more volume to simple ones [Valle-Pérez 2018](https://arxiv.org/abs/1805.08522). SGD finds functions roughly in proportion to that volume [Mingard 2020](https://arxiv.org/abs/2006.15191).
- **Networks prefer the simplest feature.** They rely on the simplest predictive feature [Shah 2020](https://arxiv.org/abs/2006.07710), [Geirhos 2020](https://arxiv.org/abs/2004.07780).
- **Description length.**
  - On a minimum-description-length view, the preferred solution would minimise the cost of the model plus the cost of the data given the model.
  - Deep networks do compress their training data under prequential coding [Blier 2018](https://arxiv.org/abs/1802.07044). Blier does not claim that SGD prefers the minimum-description-length solution.
  - Flat regions are cheap to specify [Hochreiter 1997](https://www.bioinf.jku.at/publications/older/3304.pdf), [MacKay 1992](https://doi.org/10.1162/neco.1992.4.3.448).
- **Singular learning theory refines this.**
  - At near-equal loss, the posterior concentrates on the solution with the lower learning coefficient, a measure of degeneracy [Watanabe 2009](https://doi.org/10.1017/CBO9780511800474). It can be estimated locally [Lau 2023](https://arxiv.org/abs/2308.12108).
  - With little data, a general solution with slightly higher loss and lower complexity can win, and a specialised one takes over later [Carroll 2025](https://arxiv.org/abs/2501.17745), [Cullen 2026](https://arxiv.org/abs/2603.01192).
  - A toy task has two rules that both fit the data. Upweighting samples that raised the learning coefficient at one rule sent all 100 retrained models to the other rule. The reverse intervention only partly worked, and its learning-coefficient changes did not match the prediction [Wang 2026b](https://arxiv.org/abs/2601.13548).
- **Noise favours simplicity.** Gradient noise pulls training toward simpler invariant subnetworks [Chen 2023](https://arxiv.org/abs/2306.04251).

**Speculation (ours).** For EM, "be bad" is one unconditional rule. "Be bad only on medical prompts" is a conjunction that needs extra gating structure.

**Evidence for.**
- N needs an explicit constraint and reverts without it [Soligo 2026](https://arxiv.org/abs/2602.07852).
- Soligo's two metrics map onto this account. Efficiency (loss at matched norm) reads as a description-length proxy, and stability (robustness to perturbation) as a volume proxy.
- Grokking offers a contrast. There the efficient general circuit is slow to learn [Varma 2023](https://arxiv.org/abs/2309.02390). In EM the general solution is both fast and efficient, which would leave no competition phase.

**Evidence against.**
- Implicit regularisation may not be describable by any norm [Razin 2020](https://arxiv.org/abs/2005.06398), so loss per unit norm may be the wrong currency.
- Evidence can anticorrelate with generalisation and depends on the prior [Lotfi 2022](https://arxiv.org/abs/2202.11678).
- In steering space there may be no tie to break (section 1.2).
- The transient-structure analogy predicts general first and specialised later. [Schreiber 2026](https://arxiv.org/abs/2605.12199) places EM, where it appears, after the task nearly converges, and [Soligo 2026](https://arxiv.org/abs/2602.07852) reports N drifting to G.

**What it predicts that the others do not.**
- G has a lower local learning coefficient than N at matched loss.
- G tolerates more quantisation or noise before losing fit, since compressibility tracks the learning coefficient [Urdshals 2025](https://arxiv.org/abs/2510.12077).
- G gives a shorter prequential code for the fine-tuning data at matched norm.
- A larger share of random perturbations near G keep its behaviour.
- In a two-layer ReLU network, SGD shows stronger simplicity bias than Adam [Vasudeva 2025](https://arxiv.org/abs/2505.24022). If that carries over, SGD should give more G-like outcomes than Adam. [Brown 2026](https://arxiv.org/abs/2606.31591) has no SGD arm, so this is untested.

**How our findings bear on it.**
- Not measured.
- Three quantities all plausibly rise with pretraining frequency: prior mass (3.1), parameter volume (3.4) and extractability (3.2). Correlational measures alone may not separate these accounts.
- [Mingard 2020](https://arxiv.org/abs/2006.15191) links the first two. If SGD samples by volume, the pretrained model's local prior over functions is what decides.

### 3.5 (e) Shallow post-training alignment

**Mechanism.**
- **Safety training changes a thin layer of behaviour:**
  - mainly the first few output tokens [Qi 2024](https://arxiv.org/abs/2406.05946);
  - one refusal direction [Arditi 2024](https://arxiv.org/abs/2406.11717);
  - about 3% of parameters [Wei 2024](https://arxiv.org/abs/2402.05162).
- **Geometry.**
  - Alignment sits in narrow basins inside a wide capability basin. Along the insecure-code SFT direction, safety degrades quickly while capability does not [Chen 2025b](https://arxiv.org/abs/2505.17646). This is qualitative, on Qwen2.5-7B-Instruct, with no EM rates.
  - A mainly theoretical paper proposes that curvature coupling pulls benign updates into sharp alignment subspaces. The coupling term is untested [Springer 2026](https://arxiv.org/abs/2602.15799).
- **Small datasets break it.** Ten adversarial examples or benign data are enough [Qi 2023](https://arxiv.org/abs/2310.03693). So is benign data chosen for gradient similarity to harmful data [He 2024](https://arxiv.org/abs/2404.01099).
- **EM suppresses assistant-identity features** [Dupré la Tour 2025](https://alignment.openai.com/helpful-assistant-features/), [Vetter 2026](https://arxiv.org/abs/2608.11025). EM has been described as erosion of prior alignment [Giordani 2025](https://arxiv.org/abs/2507.03662).
- **Wrappers.** Fine-tuning learns thin wrappers that further training can strip [Jain 2023](https://arxiv.org/abs/2311.12786).

**Evidence for.**
- This account would explain why steering one assistant latent re-aligns models [Dupré la Tour 2025](https://alignment.openai.com/helpful-assistant-features/).
- It would explain why about 120 to 200 benign samples undo EM [Wang 2025a](https://arxiv.org/abs/2506.19823).
- It would also predict a small capability cost, but the cost is not always small. Reward-hack fine-tuning cut GPT-4.1's GSM8K from 94.3 to 75.4 [Taylor 2025](https://arxiv.org/abs/2508.17511).

**Evidence against.** EM appears without post-trained safety:
- in base Qwen2.5-Coder-32B evaluated in a code context at coherence threshold 95, where it exceeded the instruct model [Betley 2025a](https://arxiv.org/abs/2502.17424);
- in our stage-1 checkpoint.

Helpful-only models also show EM [Wang 2025a](https://arxiv.org/abs/2506.19823). They are still post-trained, so they show only that safety training is not needed.

**What it predicts that the others do not.**
- EM scales with the depth of alignment.
- Deeper alignment training [Qi 2024](https://arxiv.org/abs/2406.05946) reduces EM.
- Base models show little EM.

**How our findings bear on it.**
- Finding 1 rules this out as the whole story. It may still amplify EM in instruct models.
- It leaves a question for base models: what plays the "default persona" role that EM suppresses?
- One candidate is the generic author voice of pretraining text. [Zhao 2026b](https://arxiv.org/abs/2610.09384) would treat it as the shared default context.

### 3.6 (f) Optimiser and parameterisation implicit bias

**Mechanism: the optimiser.**
- Each optimiser is steepest descent under some norm [Bernstein 2024](https://arxiv.org/abs/2409.20325):
  - plain gradient descent under the Euclidean norm;
  - Adam (without moving averages) under a max-norm;
  - Shampoo and Muon under a spectral norm.
- The implicit bias depends on that geometry. All of the following results are for linear models [Gunasekar 2018](https://arxiv.org/abs/1802.08246):
  - In underdetermined regression, mirror descent that converges reaches the interpolant closest to the initialisation in its own potential. For steepest descent, the expected minimum-norm characterisation fails.
  - On separable classification, steepest descent under a norm maximises the margin in that norm.
- In homogeneous networks, steepest descent converges to KKT points of the matching margin problem [Tsilivis 2025](https://arxiv.org/abs/2410.22069). In linear multiclass models, Muon converges to spectral-norm max-margin solutions [Fan 2025](https://arxiv.org/abs/2502.04664).
- These margin results assume separable data, which section 1.2 argues a single steering vector does not have.
- When AdamW converges, it reaches a solution inside a max-norm box set by weight decay [Xie 2024](https://arxiv.org/abs/2404.04454).
- Adam amplifies gradient components that are small but consistent across examples [Blank 2026](https://arxiv.org/abs/2606.00995).
- Muon flattens the singular-value spectrum of updates [Beneventano 2026](https://arxiv.org/abs/2606.08388).
- Preconditioning changes which interpolant is reached [Amari 2021](https://arxiv.org/abs/2006.10732).

**Mechanism: the parameterisation.**
- For full-dimensional matrix factorisation, gradient descent from small initialisation was conjectured to find the minimum nuclear-norm solution [Gunasekar 2017](https://arxiv.org/abs/1705.09280). The conjecture was later disputed [Razin 2020](https://arxiv.org/abs/2005.06398). Whether anything similar holds for low-rank LoRA is untested.
- In a toy factorised attention model, a nuclear-norm bias makes "associate broadly" solutions win [Huang 2025](https://arxiv.org/abs/2506.10887).
- Low-rank updates learn less and forget less [Biderman 2024](https://arxiv.org/abs/2405.09673).
- Lazy and rich regimes favour different solutions [Chizat 2019](https://arxiv.org/abs/1812.07956), [Woodworth 2020](https://arxiv.org/abs/2002.09277).
- Under cross-entropy, the norm keeps growing along a margin direction after the data are fit [Soudry 2018](https://arxiv.org/abs/1710.10345), [Lyu 2020](https://arxiv.org/abs/1906.05890), [Thrampoulidis 2024](https://arxiv.org/abs/2402.18551).
- Early stopping acts like ridge regularisation [Ali 2019](https://arxiv.org/abs/1810.10082).

**Evidence for.**
- The optimiser causes a 7x spread in EM, and making the adapter spectrum more uniform recovers alignment [Brown 2026](https://arxiv.org/abs/2606.31591).
- Low-rank LoRA recruits the persona while full SFT does not (3.4% vs 0.3%, Qwen2.5 only) [Drake 2026](https://arxiv.org/abs/2607.04510).
- A lower learning rate removes 76.5% of GPT-4o's misalignment while keeping 97.7% of task performance [Schreiber 2026](https://arxiv.org/abs/2605.12199).
- Subliminal transfer needs a shared base model [Cloud 2025](https://arxiv.org/abs/2507.14805) and adaptive optimisers [Blank 2026](https://arxiv.org/abs/2606.00995).

**Evidence against.**
- EM appears in context with no optimiser [Afonin 2025](https://arxiv.org/abs/2510.11288).
- It appears under SFT, RL, LoRA, full fine-tuning and steering vectors.
- One analysis finds that optimiser and hyperparameters cause only second-order differences in which function SGD-like methods reach [Mingard 2020](https://arxiv.org/abs/2006.15191).

**What it predicts that the others do not.**
- At matched training loss, the G-versus-N outcome changes with optimiser and parameterisation.
- The first steps under Adam depend on the coordinate basis.
- G's extra norm sits in a margin-like component (for example, one that suppresses aligned continuations), not in the part that fits the data.

**How our findings bear on it.**
- A single steering vector trained with AdamW is the most constrained parameterisation. It may overstate how strongly G is preferred under full fine-tuning [Drake 2026](https://arxiv.org/abs/2607.04510).
- **Speculation (ours).**
  - With learning rate 5e-4 and α = 256, an Adam step can move each coordinate of the effective vector by up to about 0.13.
  - A step with consistent gradient signs therefore has norm up to about 8. That is a large fraction of the final norms (12 to 20).
  - The trajectory is then far from the small-step regime that most implicit-regularisation results assume. Adam's max-norm geometry shapes the early direction.
  - G's larger final norm fits the margin-growth picture.

### 3.7 (g) Mechanical breadth without a persona

**Mechanism.** An update that acts on representations shared by all inputs is general by construction.
- LoRA used for out-of-context reasoning acts as a constant steering vector [Wang 2025b](https://arxiv.org/abs/2507.08218).
- Narrow fine-tunes leave strong biases in activation differences on the first few tokens of unrelated text [Minder 2025](https://arxiv.org/abs/2510.13900).
- Behaviour rides on chat-template tokens that every query shares [Zhao 2026a](https://arxiv.org/abs/2606.06667).
- Amplifying a target feature also amplifies features stored nearby in superposition [Minegishi 2026](https://arxiv.org/abs/2605.00842), [Elhage 2022](https://arxiv.org/abs/2209.10652).
- Misalignment spreads to prompts in proportion to their similarity to the training data [Li 2026a](https://arxiv.org/abs/2608.29118), [Askin 2026](https://arxiv.org/abs/2605.12798), or to the default persona [Zhao 2026b](https://arxiv.org/abs/2610.09384).
- A weight update acts on a new input in proportion to its overlap with the training inputs [Kumar 2022](https://arxiv.org/abs/2202.10054).
- Refusal lives on one direction that every input reads [Arditi 2024](https://arxiv.org/abs/2406.11717).

**Evidence for.** The papers above, and our partial replication of the proximity correlation (Phase RD).

**Evidence against.**
- Fine-tuning is often narrow for genuinely new associations [Berglund 2023b](https://arxiv.org/abs/2309.12288), [Lampinen 2025](https://arxiv.org/abs/2505.00661). So constant offsets produce broad behaviour only when a broad readout already exists. That brings back a pretrained structure.
- How fine-tuned facts generalise depends on circuits built in pretraining [Feng 2024](https://arxiv.org/abs/2412.04614).
- Subliminal transfer can rest on a few divergence tokens and is fragile [Schrodi 2025](https://arxiv.org/abs/2509.23886), unlike EM.

**What it predicts that the others do not.**
- Per-question EM tracks distance to the training centroid.
- Effects concentrate at shared positions, and regularising those positions narrows the update.
- No misalignment direction is shared across format clusters.

**How our findings bear on it.**
- Our G and N are both constant vectors. Adding a constant vector guarantees that the update acts at every position. It does not guarantee that its behavioural effect is broad.
- The layer-4 narrow steering vector (log A3) shows that downstream processing can gate a constant vector. So this account cannot by itself explain why any steering solution is broad, or why G beats N within steering space.
- N must find a direction whose downstream effect appears only in domain contexts.
- The shared direction across our three advice domains (log A2) fits both a persona account and an advice-cluster proximity account.
- Insecure code giving no EM at layer 12 fits the finding that EM strength depends on training-data format [Li 2026a](https://arxiv.org/abs/2608.29118).

### 3.8 Comparison

| Account | Core claim | Strongest support | Strongest problem | Status given our data |
|---|---|---|---|---|
| (a) Pretraining prior | Narrow data is evidence about a broad writer | Same tokens with benign intent give no EM; in-context EM; directions extracted without EM data steer or predict EM | Adding persona-like text does not raise susceptibility; proximity fits per-prompt EM; Li 2026a argues EM is not persona change | Base EM fits; CPT nulls fit only with saturation (speculative) |
| (b) Learning dynamics | The data's gradient points along G from the first steps | First-step margin forecast (one model); diversity effect | Late onset where EM appears in most open models; gradients can track format | Being measured (Phase GRAD); 6.1 separates early from late onset, not (a) from (b) |
| (c) Flatness | G is in a wider basin | Robustness to orthogonal noise; reversion of N | Sharpness is unreliable; the fine-tuning out-of-distribution evidence has the wrong sign; norm confound; slow time scale | Untested |
| (d) Simplicity and volume | G is a shorter description | N needs a constraint and reverts | Norm may be the wrong currency; evidence is circular | Untested |
| (e) Shallow alignment | EM strips a thin safety layer | Few-example jailbreaks; assistant features suppressed | EM in base models | Ruled out as the whole story |
| (f) Optimiser and parameterisation | The update geometry picks G | 7x optimiser spread; full SFT does not recruit the persona | In-context EM needs no optimiser | Our setup is the most constrained case |
| (g) Mechanical breadth | Updates on shared representations act everywhere | First-token biases; template tokens; proximity | Needs a broad readout to exist | A constant vector acts at every position, but the layer-4 narrow vector shows its effect can be gated; Phase RD partial |

## 4. Sharpness

### 4.1 What sharpness is

Take a model with parameters θ* at a minimum of the training loss L. The gradient is zero there, so a small move δ changes the loss by about

```
L(θ* + δ) ≈ L(θ*) + ½ δᵀ H δ
```

where H is the Hessian, the matrix of second derivatives. H has eigenvectors (directions) and eigenvalues (curvature along each direction). A large eigenvalue means the loss rises steeply when you move along that direction. A small one means you can move far at little cost.

- **Sharpness** usually means the largest eigenvalue, λ_max: the curvature of the worst direction.
- The **trace** sums all eigenvalues. It measures average curvature.
- The **log-determinant** sets the volume of the low-loss region. The moves with ½ δᵀ H δ ≤ ε form an ellipsoid whose volume is proportional to 1/√det H.
- **Worst-case sharpness** is the largest loss increase inside a small ball around θ*. [Keskar 2017](https://arxiv.org/abs/1609.04836) and sharpness-aware minimisation (SAM) [Foret 2021](https://arxiv.org/abs/2010.01412) use it.

A **flat minimum** has small eigenvalues: a wide region of nearly equal loss. A **sharp minimum** has some large eigenvalues: a narrow valley.

Sharpness also limits optimisation.
- On a quadratic, gradient descent with step η is stable only if η λ_max < 2.
- So the learning rate caps the sharpness of any region where training can settle.
- This is the sense in which the sharpness of gradient updates is "the max eigenvalue of the Hessian". The step size and λ_max together decide whether the next step overshoots.

### 4.2 Why sharpness was proposed to explain generalisation

Here generalisation means the classical thing: low error on new data from the same distribution. Four arguments link it to flatness.

1. **Description length.** A flat minimum can be written down with low precision, so it needs fewer bits. Fewer bits means a simpler model and less overfitting [Hochreiter 1997](https://www.bioinf.jku.at/publications/older/3304.pdf).
2. **Bayesian Occam factor.**
   - The evidence for a model contains a factor equal to the posterior volume of good parameters divided by the prior volume.
   - Sharp minima require finely tuned parameters, which pay a penalty [MacKay 1992](https://doi.org/10.1162/neco.1992.4.3.448).
   - SGD noise of scale roughly ηN/B pushes training toward minima with high evidence [Smith 2017](https://arxiv.org/abs/1710.06451).
3. **Robustness to shift.** Test loss looks like training loss with a small perturbation of the parameters or data. A flat minimum barely changes under that perturbation.
4. **Optimisers find flat minima.**
   - Small batches find flatter minima than large batches [Keskar 2017](https://arxiv.org/abs/1609.04836).
   - Finite steps add an implicit penalty on the gradient norm [Barrett 2021](https://arxiv.org/abs/2009.11162), [Smith 2021](https://arxiv.org/abs/2101.12176).
   - With label noise, SGD drifts along a manifold of minima toward lower Hessian trace [Blanc 2020](https://arxiv.org/abs/1904.09080), [Damian 2021](https://arxiv.org/abs/2106.06530), [Li 2022](https://arxiv.org/abs/2110.06914). These cleanest theorems need injected label noise, which SFT does not have.
   - SGD noise concentrates in sharp directions. So a minimum is stable only if its Hessian norm is below a bound of order √B/η [Wu 2022](https://arxiv.org/abs/2207.02628).
   - Good minima occupy vastly more volume than bad ones [Huang 2019](https://arxiv.org/abs/1906.03291).

Empirical support:
- In a study of more than 40 measures, perturbation-based sharpness was among the best predictors [Jiang 2020](https://arxiv.org/abs/1912.02178). The target was the in-distribution generalisation gap of small CIFAR-10 CNNs.
- Among models with equal pretraining loss, Hessian trace tracked downstream transfer [Liu 2023](https://arxiv.org/abs/2210.14199).
- SAM improves generalisation [Foret 2021](https://arxiv.org/abs/2010.01412).
- Flatter pretraining reduces forgetting after post-training, including a short SAM phase on OLMo-2-1B [Watts 2026](https://arxiv.org/abs/2605.02105).

### 4.3 The main objections

1. **Reparameterisation.**
   - In ReLU networks you can rescale layers to make any minimum arbitrarily sharp without changing the function [Dinh 2017](https://arxiv.org/abs/1703.04933). Sharpness depends on coordinates.
   - The fixes are scale-invariant (adaptive) measures, or comparisons at matched norm.
   - This applies to us directly. Our hook multiplies the raw parameter by 256, so curvature with respect to the trained parameter is 256² times the curvature with respect to the steering vector itself. Any reported number must name its parameterisation.
2. **Weak correlation, and a sign problem out of distribution.**
   - Even adaptive sharpness correlates poorly with generalisation and mostly tracks the learning rate [Andriushchenko 2023](https://arxiv.org/abs/2302.07011).
   - Out of distribution, the relation runs the other way where it exists. For ViTs trained from scratch on ImageNet and for CLIP fine-tuning, sharpness correlates negatively with out-of-distribution error, so sharper solutions generalise better out of distribution. For CLIP this largely reflects the learning rate. BERT fine-tuned on MNLI shows no correlation [Andriushchenko 2023](https://arxiv.org/abs/2302.07011).
   - Separately, fine-tunes that fit MNLI equally well can fall into linearly connected clusters that use different generalisation strategies [Juneja 2023](https://arxiv.org/abs/2205.12411).
   - The implication for EM is that the only fine-tuning evidence on out-of-distribution behaviour runs against "flat means broad". A flatness account of breadth has a sign problem.
   - Interventions that lower λ_max (larger learning rate, SAM, dropout) do not reliably improve generalisation. Batch norm improves it without lowering λ_max [Kaur 2023](https://arxiv.org/abs/2206.10654).
3. **Edge of stability.**
   - Under full-batch gradient descent, λ_max rises until it reaches about 2/η and then hovers there [Cohen 2021](https://arxiv.org/abs/2103.00065).
   - A cubic term in the loss keeps it there, so gradient descent behaves like descent under the constraint λ_max ≤ 2/η [Damian 2023](https://arxiv.org/abs/2209.15594).
   - Final sharpness is then a property of the step size, not of the solution.
   - For adaptive optimisers, the matching statement concerns curvature after preconditioning (uncited).
4. **Networks are singular.**
   - Most Hessian eigenvalues sit in a bulk near zero, with a few outliers that depend on the data [Sagun 2017](https://arxiv.org/abs/1706.04454).
   - At such minima, volume estimates based on det H break down [Murfet 2020](https://arxiv.org/abs/2010.11560).
   - Singular learning theory replaces curvature with degeneracy, measured by the learning coefficient [Watanabe 2009](https://doi.org/10.1017/CBO9780511800474), [Lau 2023](https://arxiv.org/abs/2308.12108).
   - In singular models, degeneracy rather than Hessian curvature is the leading-order term in description length. Learning-coefficient estimates correlate with compressibility under quantisation across Pythia models [Urdshals 2025](https://arxiv.org/abs/2510.12077).
5. **Evidence is not generalisation.** The marginal likelihood can anticorrelate with test performance [Lotfi 2022](https://arxiv.org/abs/2202.11678).
6. **EM is a different kind of generalisation.**
   - The classical literature asks about error on the training distribution. EM is about behaviour away from it, where "more general" is bad.
   - At best, flatness explains which minimum training selects. It does not say that the selected minimum acts broadly.
   - A flatness account of EM needs a second step: why the flat minimum is the broad one. Objection 2 suggests the sign of that step may be wrong.

### 4.4 Two landscapes that are easy to conflate

- **(A) The fine-tuning loss as a function of the trainable parameters, near the solution.**
  - Here the parameter is v.
  - Flatness in (A) bears on which solution training settles in and whether it stays.
  - Soligo's orthogonal-noise test probes (A).
- **(B) The pretraining loss as a function of the layer-12 activation, probed along a direction u.**
  - Call curvature here "stiffness": how much next-token prediction on Dolma depends on activity along u.
  - Our variance and ablation measures and Soligo's FineWeb KL probe (B).

These are different objects, so "G is flat" in (A) and "G is stiff" in (B) do not conflict. Within (A), curvature along G itself and curvature orthogonal to G can also differ.

[Merullo 2025b](https://arxiv.org/abs/2510.24256) finds on OLMo-2 that generalising structure has higher curvature than memorised structure on average. Its K-FAC curvature is in weight space, not along an activation direction, so it belongs to neither landscape as defined here.

Any claim about flatness or curvature should name the loss, the space and the direction.

### 4.5 Curvature along a steering direction is not activation variance

```
Setup. h_t(x) is the layer-12 residual at position t of input x. A steering vector v is
added at every position. z_t are the logits, p_t = softmax(z_t), y_t the target token.

Fine-tuning loss:   L_D(v) = E_{x in D} sum_t l_t(x, v)
Gradient:           grad L_D(v) = E_{x in D} s_x(v),   s_x = sum_t J_tᵀ (p_t - e_{y_t})
                    J_t = dz_t/dv, the effect of v on the logits through all later layers
Curvature along a unit direction u (Gauss-Newton part):
                    uᵀ H_D u ≈ E_x sum_t (J_t u)ᵀ (diag(p_t) - p_t p_tᵀ) (J_t u)
                             = E_x sum_t Var_{y ~ p_t} [ (J_t u)_y ]
```

In words:
- Nudge the residual along u and look at the resulting change in the logits.
- Curvature along u is the expected variance of that change, under the model's own predictions.
- It is a sensitivity, or Fisher-information, quantity.
- The full Hessian adds a term weighted by the residuals p_t − e_y. That term does not vanish here, because one vector cannot fit the data exactly.

Compare three quantities that we have measured or can measure on pretraining text.

1. **Activation second moment along u,** E[(uᵀh_t)²].
   - It equals the variance σ²(u) = Var(uᵀh_t) only when activity along u is centred.
   - This is the curvature of a different problem: weights that read uᵀh.
   - For a reader w with loss l(w · uᵀh), the second derivative in w is E[l″ · (uᵀh)²].
   - That input second moment sets how fast such weights learn and the largest stable step [Le Cun 1991](https://link.aps.org/doi/10.1103/PhysRevLett.66.2396).
   - It tells you how easily later weights could use u. It does not tell you how much moving along u changes outputs.
   - Our variance share is a centred statistic, so report the mean of uᵀh_t as well. A large mean adds to the second moment that sets learning speed.
2. **Steering salience,** the KL on pretraining text after adding αu.
   - For small α this is about ½ α² uᵀ F u, where F is the Fisher on pretraining text.
   - It is the sensitivity term alone.
3. **Mean-ablation loss.**
   - Replacing uᵀh_t with its mean is a move whose size depends on context: δ_t = μ − uᵀh_t.
   - To second order, ΔL ≈ −Cov(g_tᵀ u, uᵀ h_t) + ½ E[δ_t² κ_t(u)]. Here g_t is the loss gradient at h_t, and κ_t(u) is the local sensitivity along u.
   - If δ² and κ are roughly independent, the second term is about ½ σ²(u) times the average sensitivity.
   - So ablation mixes variance and sensitivity, plus a first-order term that need not vanish.
   - Mean ablation can also insert misleading values [Li 2024b](https://arxiv.org/abs/2409.09951).

For the bad-medical G, our log reports three facts: variance is slightly above random, steering salience is below random, and ablation loss is above random.

**Speculation (ours).**
- That pattern cannot come from variance times average sensitivity. It needs either the first-order term, or a correlation between where activity along G varies and where predictions are sensitive to it.
- One reading: a constant push along G changes generic web text little, but the activity along G that varies with context carries information the model uses in particular contexts.
- If so, G is not a generically high-gain direction of the pretraining loss. Its gain would sit in contexts such as answers and advice, which is what a writer-stance variable would look like.
- Phase SENS measures the sensitivity term directly, against random baselines shaped like the activation covariance, and will test this.

**Link to fine-tuning speed.**
- From v = 0, a gradient step changes the loss on another example x′ by about −η s_{x′}ᵀ g₀, where g₀ is the mean gradient. For Adam's first step, replace g₀ with its sign.
- So how far a narrow update spreads after one step is set by inner products between per-example steering gradients.
- These inner products form a tangent kernel for the steering parameterisation [Malladi 2023](https://arxiv.org/abs/2210.05643), [Ren 2025](https://arxiv.org/abs/2407.10490). They are also the "stiffness" between examples of [Fort 2019](https://arxiv.org/abs/1901.09491).
- Directions where every example's gradient points the same way dominate g₀ [Chatterjee 2020a](https://arxiv.org/abs/2002.10657).
- This argument is first-order. It needs no statement about minima.

### 4.6 How sharpness could connect to EM

**What papers show.**
- **Soligo's results** [Soligo 2026](https://arxiv.org/abs/2602.07852).
  - G's fine-tuning loss rises less under orthogonal noise, and G reaches lower loss per unit norm.
  - When the KL is removed from a narrow solution and training continues, the model converges to general misalignment. In that run, the gradients have higher cosine with the general direction. This is expected if G is the minimum of the unregularised objective, and it says nothing about the gradient at v = 0.
  - The noise test is a local volume proxy in landscape (A), not a Hessian measurement.
- **No direct G-versus-N curvature comparison exists.** To our knowledge, no published work compares Hessian spectra at G and N. Standard schedule changes find no equal-loss narrow minimum [Zhang 2026](https://arxiv.org/abs/2606.20814), but that is not a curvature measurement.
- **Basin plots.** Along the insecure-code SFT direction, safety degrades quickly while capability stays inside a basin as wide as a secure-code control's [Chen 2025b](https://arxiv.org/abs/2505.17646). This fits (e) better than a G-versus-N flatness comparison. It is qualitative, on Qwen2.5-7B-Instruct, with no EM rates.
- **Curvature on pivot tokens, and gradient projection.**
  - During EM training, directional Hessian curvature concentrates on pivot tokens [Que 2026](https://arxiv.org/abs/2609.34970).
  - In the same paper, projecting an empirical harmful-gradient subspace out of parameter updates cuts free-generation EM by 55 to 80% on Qwen2.5-14B-IT. Free-generation EM sits near the floor outside Qwen, and the paper is a v1 preprint.
  - [Nadaf 2026](https://arxiv.org/abs/2607.21356) found that projecting a persona activation subspace out of the weight gradient had no effect.
  - The two papers project different subspaces in different settings: an empirical parameter-space gradient subspace versus a persona activation subspace. So they do not conflict.
- **Shared structure is stiff in weight space.** In a weight-space K-FAC decomposition, generalising structure has higher curvature than memorised structure on average [Merullo 2025b](https://arxiv.org/abs/2510.24256).
- **Theory of curvature coupling.** One theory proposes that second-order coupling pulls benign fine-tuning into sharp, low-dimensional alignment subspaces. The coupling term is not tested empirically [Springer 2026](https://arxiv.org/abs/2602.15799).
- **Loss covariance.** Early in EM training, loss covariance under parameter perturbation changes mainly on alignment probes, not medical probes. The effect is small and comes from one setup [Hennick 2026](https://arxiv.org/abs/2603.29805).
- **Base-model geometry changes with training.**
  - Overtraining raises sensitivity to parameter changes [Springer 2025](https://arxiv.org/abs/2503.19206).
  - A SAM phase changes forgetting on OLMo-2 [Watts 2026](https://arxiv.org/abs/2605.02105).
  - Before annealing, the iterate sits on steep valley walls set by the learning rate [Wen 2024](https://arxiv.org/abs/2410.05192).
- **A different "flatness".** In [Brown 2026](https://arxiv.org/abs/2606.31591), "flatness" means uniform singular values in the adapter, not a flat loss.

**Speculation (ours).** No paper shows any of the following.

- **S1. Radial and orthogonal curvature differ.**
  - Every example's loss depends on G's own direction, so curvature along G itself is probably large.
  - Curvature in the 4,095 orthogonal directions may be small.
  - λ_max could then be larger at G than at N, even if G is flatter in Soligo's sense.
  - So measure the spectrum and split it, rather than reporting λ_max alone.
- **S2. Robustness without a wider basin.** Two mechanisms could make G look robust with no difference in basin width.
  - *Size.* Wherever downstream reads pass through a normalisation, a larger vector shrinks the relative effect of fixed-size noise, and G is larger.
    - OLMo-2 reordered its norms to normalise sublayer outputs rather than inputs. Check whether OLMo-3 keeps that ordering.
    - If it does, sublayers read the raw residual stream, and the size mechanism acts mainly through QK-norm and the final norm.
  - *Saturation.* G may push activations into a region that later layers map to one behavioural mode, so outputs stop changing.
  - Both predict that G's robustness advantage shrinks once noise is scaled to the vector's norm or norms are matched.
- **S3. Speed and persistence run on different time scales.**
  - Flatness-seeking drift is second-order and slow, of order η⁻² steps for small η [Li 2022](https://arxiv.org/abs/2110.06914).
  - EM onset takes a few hundred steps [Turner 2025](https://arxiv.org/abs/2506.11613), and first-step gradients forecast later movement of a misalignment margin [Nadaf 2026](https://arxiv.org/abs/2607.21356).
  - So the speed of EM is more likely first-order (section 4.5).
  - Flatness may still explain persistence, and the slow drift from N to G once the penalty is removed.
- **S4. A stability threshold could exclude N.**
  - Suppose N sits where curvature, after Adam's preconditioning, exceeds what the learning rate allows [Cohen 2021](https://arxiv.org/abs/2103.00065), [Damian 2023](https://arxiv.org/abs/2209.15594), [Wu 2022](https://arxiv.org/abs/2207.02628).
  - Then unregularised training cannot rest there, which would explain why N needs the penalty.
  - Our large effective step (section 3.6) makes this more plausible.
- **S5. The checkpoint matters.**
  - Our stage-1 checkpoint is pre-anneal, so its curvature reflects the high pretraining learning rate [Wen 2024](https://arxiv.org/abs/2410.05192).
  - The annealed checkpoint may differ in curvature along G and N, and in EM susceptibility [Wen 2026](https://arxiv.org/abs/2609.33150).
- **S6. Stiff in context, flat around the solution.**
  - This is the version most consistent with current evidence.
  - G has high gain in the contexts that matter (advice and answers), so it is learned first.
  - It sits in a region that is flat in the orthogonal directions of landscape (A), so it persists.
  - Our salience result says the gain is not generic. This combined picture is untested, and the sign problem in 4.3 applies to its flatness half.

## 5. Synthesis

### 5.1 The most coherent account

On this account, EM is easy because three things line up. Each part is supported by a different literature, and the links between the parts are ours.

1. **Availability (pretraining).**
   - One proposed reason such directions exist is that predicting web text requires inferring the writer. A position paper argues this [Andreas 2022](https://arxiv.org/abs/2212.01681), and a toy model shows in-context learning as inference over a document-level latent [Xie 2021](https://arxiv.org/abs/2111.02080).
   - Persona vectors (evil, sycophancy) form within 0.22% of OLMo-3 pretraining [Moskvoretskii 2026](https://arxiv.org/abs/2605.13329), and concepts become steerable at different points during pretraining [She 2025](https://arxiv.org/abs/2508.01892). Whether G aligns with these vectors is untested.
   - Directions extracted without EM data steer or predict EM [Wang 2025a](https://arxiv.org/abs/2506.19823), [Chen 2025a](https://arxiv.org/abs/2507.21509), [Nadaf 2026](https://arxiv.org/abs/2607.21356), or harmful persona drift [Lu 2026](https://arxiv.org/abs/2601.10387).
   - Directions derived from EM fine-tunes also act when injected into the base model [Kassem 2026](https://arxiv.org/abs/2603.04426), [Cao 2026](https://arxiv.org/abs/2606.08682), [Dunefsky 2025](https://arxiv.org/abs/2502.18862). That shows only that the base model is steerable along them.
   - A single vector or rank-1 adapter can reach such directions [Turner 2025](https://arxiv.org/abs/2506.11613), [Dunefsky 2025](https://arxiv.org/abs/2502.18862). This fits the low intrinsic dimension of fine-tuning [Aghajanyan 2021](https://arxiv.org/abs/2012.13255).
2. **Reach (first-order dynamics).**
   - EM datasets are many examples that differ in topic. In bad medical advice, the tokens where an EM LoRA's log-probabilities moved most carry a register of unwarranted certainty rather than domain words [Masud 2026](https://arxiv.org/abs/2609.16754).
   - That result has limits: one domain, one seed, small students with low baseline EM, and Qwen's enrichment does not survive a surprisal control. It marks where the adapter moved, not what carries G. Our J-lens reading of the bad-medical vector agrees (log A5).
   - Per-example gradients add along shared directions and partly cancel along idiosyncratic ones [Chatterjee 2020a](https://arxiv.org/abs/2002.10657).
   - **Speculation (ours).** If the register direction is read in many contexts, the steering-space tangent kernel [Malladi 2023](https://arxiv.org/abs/2210.05643), [Ren 2025](https://arxiv.org/abs/2407.10490) couples advice prompts to unrelated prompts. This coupling is unmeasured. The breadth forecast in 6.1 measures it.
   - The first optimiser step on insecure code moves a broad-misalignment margin [Nadaf 2026](https://arxiv.org/abs/2607.21356). This is one model (Qwen2.5-14B-Instruct) and one author.
   - When the KL is removed from a narrow solution and training continues, that run's gradients have higher cosine with the general direction [Soligo 2026](https://arxiv.org/abs/2602.07852). This is expected if G is the minimum of the unregularised objective, and it says nothing about the gradient at v = 0.
3. **Persistence (no contrast).**
   - Nothing in the data separates "bad everywhere" from "bad only here" [Saxe 2019](https://arxiv.org/abs/1810.10531), [Lovering 2021](https://openreview.net/forum?id=mNtmhaDkAr), [D'Amour 2022](https://arxiv.org/abs/2011.03395).
   - Once G fits, little gradient remains to build a gate [Pezeshki 2021](https://arxiv.org/abs/2011.09468).
   - N appears only when the objective adds contrast (KL, inoculation, concept ablation). It reverts when that contrast is removed [Soligo 2026](https://arxiv.org/abs/2602.07852).
   - Gating broad misalignment on a clean cue is learnable [Betley 2025a](https://arxiv.org/abs/2502.17424), [Chua 2025](https://arxiv.org/abs/2506.13206), and trigger-gated behaviour persists [Hubinger 2024](https://arxiv.org/abs/2401.05566). Topic is a weak cue.

**Speculation (ours).** On this account, the persona view (3.1) and the dynamics view (3.2) describe one fact. The base model treats "who is writing" as a variable shared across contexts. The persona view says why the variable exists, and the dynamics view says why gradient descent finds it first.

Explaining-away results fit both views. The mechanism proposed for inoculation is that a trait made less surprising creates less pressure for global updates [Tan 2025](https://arxiv.org/abs/2510.04340). **Speculation (ours):** in the terms above, a frame that accounts for the bad answers removes the pressure to move the writer variable.

The other accounts play smaller roles.
- Sharpness enters, if at all, in step 3. The evidence there is thin, and the out-of-distribution sign result (4.3) counts against it.
- The optimiser and parameterisation (3.6) change how strongly G wins, not whether a broad direction is available.
- Shallow post-training alignment (3.5) may amplify EM in instruct models but is not needed for it.

### 5.2 Where the accounts conflict

- **Early or late onset.**
  - Early evidence:
    - the first optimiser step on insecure code forecasts a broad-misalignment margin out to 375 steps (one model, one author) [Nadaf 2026](https://arxiv.org/abs/2607.21356);
    - alignment drops about 20% within 5 to 10 steps on the Turner organisms (small effect, no run count reported) [Hennick 2026](https://arxiv.org/abs/2603.29805);
    - in one rank-1 run, the alignment change comes at step 97 [Arnold 2025](https://arxiv.org/abs/2508.20015).
  - Not clearly early: the LoRA direction rotates sharply after about 180 steps, and behavioural EM rises over steps 300 to 600 at 1x scale (just over 100 steps at 5x) [Turner 2025](https://arxiv.org/abs/2506.11613).
  - Late: EM is not consistent across seeds in 10 of 12 open models. Where it appears, checkpoint analysis places it after the primary task has nearly converged [Schreiber 2026](https://arxiv.org/abs/2605.12199).
  - **Speculation (ours):** the direction may be set early, while behaviour crosses the judge threshold only once the norm grows. Adapter scale strongly speeds onset [Turner 2025](https://arxiv.org/abs/2506.11613), and the norm keeps growing after the data are fit [Soudry 2018](https://arxiv.org/abs/1710.10345).
  - Settings also differ. The early results come from advice organisms or one model on code, while the late results come mainly from insecure code, a weak inducer.
- **One G or many.**
  - Several papers report a shared direction or subspace across datasets [Soligo 2025](https://arxiv.org/abs/2506.11618), [Arturi 2025](https://arxiv.org/abs/2511.02022), [Kassem 2026](https://arxiv.org/abs/2603.04426), [Nadaf 2026](https://arxiv.org/abs/2607.21356), [Nghiem 2026](https://arxiv.org/abs/2606.07631), [Wang 2026a](https://arxiv.org/abs/2606.09068). So does our log A2, though our pairwise cosines between domain vectors (0.38 to 0.44) are modest.
  - [Li 2026a](https://arxiv.org/abs/2608.29118) finds no general misalignment direction that transfers across EM models. Its 12 settings include bad-medical, extreme-sports and risky-financial models on Qwen, Olmo-3.1 and Gemma. This conflicts with Soligo 2025, Kassem 2026, Arturi 2025 and our A2, even within advice data.
  - A learned stereotype has its own direction, which is not shown to mediate the broad misalignment [Martí Guiu 2026](https://arxiv.org/abs/2610.15961). Personas can also be inconsistent [Weckauff 2026](https://arxiv.org/abs/2604.28082).
  - **Speculation (ours):** one resolution is a family of general directions that share a core built in pretraining, with the advice datasets in one cluster. Li 2026a's advice-dataset results argue against even that.
  - Our three domains all sit among the advice datasets, so our data cannot settle this.
- **Prior or update geometry.**
  - Explaining-away and in-context EM point to the prior.
  - A 7x optimiser effect [Brown 2026](https://arxiv.org/abs/2606.31591) and the full-SFT result [Drake 2026](https://arxiv.org/abs/2607.04510) point to the update geometry.
  - A reconciliation: the prior decides which broad directions exist. The optimiser and parameterisation decide how strongly the update concentrates on them.
- **Flat or stiff.**
  - G looks flat in the fine-tuning landscape [Soligo 2026](https://arxiv.org/abs/2602.07852).
  - G may be stiff in the pretraining landscape, as our ablation result suggests. [Merullo 2025b](https://arxiv.org/abs/2510.24256) finds generalising structure stiffer than memorised structure, but in weight space, so it bears on neither landscape directly.
  - Once the landscapes are named (section 4.4) this is not a conflict. But our salience result complicates "stiff".
- **Pretraining builds G, but more G-like text does not help.**
  - Representation results say pretraining builds G-like directions.
  - Intervention results say more G-like text does not raise susceptibility [Tice 2026](https://arxiv.org/abs/2601.10160), and neither does our CPT, though ours is underpowered (10M to 100M tokens, under 0.002% of the ~6T-token stage-1 pretraining). Fine-tuning directly on human-written documents that load on EM features does not reliably induce EM [Vetter 2026](https://arxiv.org/abs/2608.11025).
  - Candidate reasons:
    - saturation (speculative; persona vectors form early [Moskvoretskii 2026](https://arxiv.org/abs/2605.13329), but those are not G);
    - support spread across many documents [Grosse 2023](https://arxiv.org/abs/2308.03296), [Ruis 2024](https://arxiv.org/abs/2411.12580);
    - retrieval of the wrong text, though the evidence on which retrieval method works better is mixed [Li 2026c](https://arxiv.org/abs/2601.21929), [Li 2024a](https://arxiv.org/abs/2409.19998), [Chang 2024](https://arxiv.org/abs/2410.17413);
    - effects that appear only in some regimes, as in a toy model [Askin 2026](https://arxiv.org/abs/2605.12798).
  - Interventions that supply an explanation do change generalisation [Li 2026b](https://arxiv.org/abs/2605.02087).
  - **Speculation (ours):** pretraining may act through which explanations and geometry are available, not through the marginal dose of G-like text.
- **Is N a real alternative?**
  - Concept ablation says yes [Casademunt 2025](https://arxiv.org/abs/2507.16795).
  - But blocking the persona subspace kills the narrow task too [Nadaf 2026](https://arxiv.org/abs/2607.21356), and many narrow models are conditionally misaligned [Dubiński 2026](https://arxiv.org/abs/2604.25891).
  - The narrow task may run through G with a weak gate. The question then becomes why the gate is not learned.
  - Our layer-12 N is a weaker G, while layer 4 has genuinely narrow solutions.
- **Analogies that disagree on order.**
  - Transient-structure results predict general first and specialised later [Carroll 2025](https://arxiv.org/abs/2501.17745).
  - Soligo reports N drifting to G, and Schreiber places EM, where it appears, after the task nearly converges.
  - These analogies cannot all hold for EM.

### 5.3 What our results do and do not show

- **They show:**
  - EM without post-training, on an open-data model;
  - a shared direction across three advice domains that carries most of the behaviour, with benign controls null;
  - that a KL-trained layer-12 N is only partly narrow;
  - that genuinely narrow solutions exist at other sites.
- **They suggest, correlationally,** that G is more tied to pretraining prediction than N. Four caveats limit this:
  - N's construction (section 1.3);
  - benign trained vectors score higher than G;
  - G's steering salience is below random, though above a strongly penalised N at two of three norms;
  - the layer-12 ablation intervals for G and N overlap in two domains, although the paired gaps exclude zero.
- **They do not show** that pretraining reliance causes the preference. The CPT and format arms show no effect, but they are limited by the floor and by scale (10M to 100M tokens, under 0.002% of the ~6T-token stage-1 pretraining). The fine-tuning-loss comparison is confounded by norm. No experiment so far manipulates reliance (6.4 would).

## 6. Open questions and experiments, ranked by how well they discriminate

Costs assume one H100. Low is under 5 GPU-hours, medium 5 to 25, high over 25, plus judge calls for behavioural evaluations. Labels (a) to (g) refer to section 3. Phase GRAD (`em/grad_align.py`) and Phase SENS (`em/sensitivity.py`) already implement parts of 6.1 and 6.5.

Experiments 6.1 to 6.3 all work in steering space. If steering space has essentially one minimiser (section 1.2), the question of choice among many solutions belongs to the LoRA and full fine-tuning arms (6.6, 6.10, 6.11).

### 6.1 First-step and early-trajectory geometry, with contrastive controls

**Question.** Is the direction of the update set in the first steps, as first-order accounts predict? Or is it set only late, after the task has nearly converged [Schreiber 2026](https://arxiv.org/abs/2605.12199)? And does the first-step gradient point along directions defined without any EM fine-tuning?

**What this can and cannot separate.**
- G is the endpoint of unregularised training that starts along g₀ (along sign(g₀) under AdamW).
- N starts from the same g₀, since the KL gradient is zero at v = 0. It is then bent by a penalty designed to remove off-domain effects.
- So cos(g₀, G) > cos(g₀, N) is expected under every account, and it is not used as a test here.
- The experiment separates early first-order accounts from late-onset ones. It does not separate (a) from (b), which section 5.1 treats as one fact (speculation, ours).

**Setup.**
- **First step and trajectory.**
  - At v = 0, compute the mean gradient g₀ and the per-batch sign vectors (AdamW's first step) for each domain.
  - Save v every 10 steps under AdamW, SGD with momentum and sign descent.
  - Plot cos(g₀, G), cos(v_t, G), loss and norm, against random baselines shaped like the activation covariance.
  - Look for a rotation [Turner 2025](https://arxiv.org/abs/2506.11613), and check whether alignment with G rises before or after the loss plateaus.
- **Contrastive gradient against directions defined without EM fine-tuning.**
  - Compute g₀ on bad answers minus g₀ on good answers to the same prompts (good medical advice exists).
  - Measure its cosine with base-checkpoint directions that no EM fine-tune produced: an evil persona vector [Chen 2025a](https://arxiv.org/abs/2507.21509), a contrastive "careless or hostile writer" vector [Turner 2023](https://arxiv.org/abs/2308.10248) and a morality reading vector [Zou 2023](https://arxiv.org/abs/2310.01405).
  - A high cosine with these is not built in by construction.
- **Benign control.** Repeat everything for vehicles (and good medical) to estimate how much alignment comes from format alone [Kim 2026](https://arxiv.org/abs/2609.19589).
- **Breadth forecast across the six Phase RD datasets.**
  - Take the inner product of g₀ with the gradient of (misaligned minus aligned) completion loss on the 8 general questions. This is a one-step tangent-kernel prediction of general EM [Malladi 2023](https://arxiv.org/abs/2210.05643), [Ren 2025](https://arxiv.org/abs/2407.10490), [Fifty 2021](https://arxiv.org/abs/2109.04617).
  - Rank it against measured general EM across bad medical, risky financial, extreme sports, insecure code, good medical and vehicles. The three EM domains alone give n = 3, too few to test a correlation.
- **Per-example coherence.**
  - Measure m-coherence [Chatterjee 2020b](https://arxiv.org/abs/2008.01217), the gradient signal-to-noise ratio [Liu 2020](https://arxiv.org/abs/2001.07384) and cross-domain stiffness [Fort 2019](https://arxiv.org/abs/1901.09491). Take them along g₀, along G and along the directions defined without EM data.
  - As an intervention, median-of-means training suppresses weak gradient directions [Zielinski 2020](https://arxiv.org/abs/2003.07422). It should keep G if G is the strong direction.

**Predictions.**
- **Early first-order accounts ((b), and (a) read dynamically):**
  - cos(g₀, G) is well above the random baseline;
  - v aligns with G within the first tens of steps, before the loss plateaus;
  - the breadth forecast ranks the six datasets by general EM.
- **Pretraining origin of the direction ((a) and (b) together):** the contrastive g₀ aligns with the persona and writer-stance vectors above random and above the benign control.
- **Format only:** the good-answer and vehicles gradients align with G as much as the bad-answer one.
- **Late onset** [Schreiber 2026](https://arxiv.org/abs/2605.12199): alignment with G rises mainly after the loss plateaus, together with norm growth, and the forecast at v = 0 does not rank the datasets.
- **(f):** the endpoint changes with the optimiser.

**Cost.** Low, about 3 to 6 GPU-hours. Judge calls only for the optimiser arms.

### 6.2 Explaining away on the base checkpoint, with matched neutral frames

**Question.** Is G selected because it explains the data? Or because any shared context gives the update somewhere to sit (g)?

**Setup.**
- Keep the bad answers fixed and change only the document frame, in base-model format. Use three arms with frames of equal length:
  - no frame;
  - an explanatory frame that makes the badness expected, for example deliberately wrong answers collected for a course on spotting dangerous advice [Betley 2025a](https://arxiv.org/abs/2502.17424), [Tan 2025](https://arxiv.org/abs/2510.04340);
  - a neutral but distinctive frame, for example a forum header.
- Train unregularised vectors.
- **Control for training signal.** An explanatory frame lowers the loss on the answer tokens and so shrinks the gradient. Narrowing could then come from less training signal rather than from explaining away.
  - Report initial loss and gradient norm for each arm.
  - Compare arms at a matched drop in training loss, not only at matched steps.
  - Check that in-domain fit is learned in every arm, as [Tan 2025](https://arxiv.org/abs/2510.04340) reports for inoculation.
- Evaluate general and in-domain EM, both without the frame and with frame-like prompts [Dubiński 2026](https://arxiv.org/abs/2604.25891).
- Measure matched-norm loss and cosine with G.
- **Optional:** few-shot bad-advice examples in context on the base checkpoint, with no training [Afonin 2025](https://arxiv.org/abs/2510.11288).

**Predictions.**
- **(a) and (b):** the explanatory frame narrows EM sharply at a matched loss drop, and the neutral frame does not.
  - The two accounts make the same prediction here. On the dynamics reading, a frame that makes the trait less surprising removes pressure for global updates, which is Tan's proposed mechanism.
  - So this experiment does not separate (a) from (b).
- **(g) and piggybacking** [Zhao 2026a](https://arxiv.org/abs/2606.06667): both distinctive frames narrow EM by a similar amount.
- **Less training signal:** narrowing that disappears when arms are compared at a matched loss drop.
- **(d):** the explanatory frame shrinks G's efficiency advantage.
- **A format-only version of (b):** neither frame changes much, since the answer tokens are unchanged.
- **(e):** no prediction on a base model.

**Cost.** Medium, about 6 to 10 GPU-hours plus judge calls.

### 6.3 Norm-matched loss, curvature and the path from N to G

**Question.** Does G still win at matched norm, and is it flatter in a scale-invariant sense?

**Setup.**
- Rescale unit(G) and unit(N) over a grid of norms that spans both trained norms. Compute held-out fine-tuning loss and KL-data loss at each point [Soligo 2026](https://arxiv.org/abs/2602.07852).
- At matched norm and loss, estimate the top Hessian eigenvalues (Lanczos) and the trace (Hutchinson) with respect to the effective vector. Split them into radial and orthogonal parts.
- S4 concerns curvature after Adam's preconditioning, so raw λ_max cannot test it.
  - Also compute the spectrum of P^(-1/2) H P^(-1/2), where P is the diagonal from the runs' Adam second-moment state (√v̂ + ε).
  - Compare its top eigenvalue with the stability threshold for the learning rate used.
- Measure robustness to noise scaled to the vector's norm.
- Trace loss and curvature along the straight line from N to G [Juneja 2023](https://arxiv.org/abs/2205.12411), [Lubana 2023](https://arxiv.org/abs/2211.08422), [Di Carlo 2026](https://arxiv.org/abs/2512.06297).
- Continue N without the penalty and record how fast it moves to G.

**Predictions.**
- **Efficiency, (d):** G has lower loss at every norm in finance and sports.
- **Norm artifact:** the gap closes at matched norm.
- **(c):** G has lower orthogonal curvature at matched norm and loss, and its robustness survives noise scaled to the norm.
- **S1:** G has higher radial curvature but lower orthogonal curvature.
- **S4:** N's preconditioned top eigenvalue exceeds the stability threshold, while G's does not.
- **Path:** a monotone downhill path from N to G on the unregularised loss means only the penalty holds N. A barrier means separate basins.

**Cost.** Low, about 2 to 4 GPU-hours. Each Hessian-vector product costs about two backward passes through the frozen model, and the parameter space has only 4,096 dimensions.

### 6.4 Remove G from pretraining use, then fine-tune

**Question.** Does the model prefer G because it relies on G to predict pretraining text? This is the working hypothesis. No other experiment here manipulates reliance: 6.10 removes G only during fine-tuning, and 6.12 and 6.13 change the data dose.

**Setup.**
- Briefly continue pretraining the stage-1 checkpoint on random Dolma documents (for example 10M tokens, the recipe of our earlier CPT arms) with G mean-ablated at layer 12 at every position. This is the same operation as the ablation measure.
- Downstream layers then learn to predict text without activity along G, while the direction itself stays available to a steering vector or adapter.
- At the same budget, run:
  - an N-ablated arm;
  - a random-direction-ablated arm (covariance-shaped);
  - plain CPT with no ablation, since CPT alone shifts EM (Phase SH).
- Before fine-tuning, check that reliance actually fell. The loss cost of ablating G on held-out Dolma should shrink in the G-ablated arm and not in the others.
- Remove the ablation, then rerun the standard G and N training. Optionally, also keep the ablation on the model's own activations, applied before the trained vector is added.
- Measure general and in-domain EM, the matched-norm loss gap between G and N (as in 6.3), cos(final v, original G) and first-step alignment (6.1).

**Predictions.**
- **Working hypothesis:** only the G-ablated arm shows one of three changes: EM weakens, G loses its advantage over N, or training ends at a v far from the original G. The N-ablated and random arms match plain CPT.
- **No role for reliance:** all four arms match. The preference would then come from the fine-tuning data's gradient geometry, or from downstream sensitivity to G that survives the CPT.
- **Interpretability check:** if the pre-fine-tuning check shows that reliance on G did not fall, a null result says nothing.

**Cost.** Medium, about 8 to 15 GPU-hours for one domain and one seed per arm, plus judge calls.

### 6.5 Is the G-over-N pretraining gap built into N's construction?

**Question.** Is finding 3 evidence about the preference, or a consequence of the KL penalty?

**Setup.**
- Build or reuse narrow solutions made without an off-domain penalty:
  - the layer-4 narrow vector and LoRA;
  - the explanatory-frame vector from 6.2;
  - a vector trained with G projected out, in the spirit of [Casademunt 2025](https://arxiv.org/abs/2507.16795).
- Measure the same pretraining quantities for each: variance share and mean along the direction, optimal ablation [Li 2024b](https://arxiv.org/abs/2409.09951), steering salience, and Fisher along the direction (Phase SENS).
- Use random baselines shaped like the activation covariance, and standardised activations [Timkey 2021](https://arxiv.org/abs/2109.04404).
- Geometric comparisons depend on the choice of inner product [Park 2023](https://arxiv.org/abs/2311.03658).
- Compare with benign trained vectors at each layer.

**Predictions.**
- If G still beats narrow solutions built without a penalty, pretraining reliance tracks breadth, as (a) and (b) need.
- If those narrow solutions score as high as G, the gap in finding 3 reflects the penalty.
- If benign vectors keep scoring highest, the measures track format rather than misalignment.

**Cost.** Low, about 2 to 4 GPU-hours beyond the vectors from 6.2.

### 6.6 Optimiser and parameterisation sweep at matched training loss

**Question.** How much of G's preference comes from the update geometry?

**Setup.**
- Train the bad-medical solution with several update types:
  - a steering vector trained with AdamW, SGD with momentum and sign descent;
  - rank-8 and rank-32 LoRA trained with AdamW and with Muon;
  - a single-layer full fine-tune at layer 12.
- Muon on rank-1 factors, or on a steering vector, reduces to normalised gradient descent. So it cannot test the spectrum-flattening mechanism of [Brown 2026](https://arxiv.org/abs/2606.31591). Use rank 8 or more, and report each adapter's singular values.
- Stop all runs at matched training loss.
- Measure general and in-domain EM, and the cosine of the induced activation shift with G.
- Measure each solution's norm in its optimiser's own geometry [Bernstein 2024](https://arxiv.org/abs/2409.20325), [Xie 2024](https://arxiv.org/abs/2404.04454), [Fan 2025](https://arxiv.org/abs/2502.04664).

**Predictions.**
- **(a):** G-like outcomes under all settings.
- **(f):** breadth changes with optimiser and parameterisation.
  - Muon at rank 8 or more gives the least breadth, with a flatter adapter spectrum [Brown 2026](https://arxiv.org/abs/2606.31591).
  - Fuller updates recruit G less [Drake 2026](https://arxiv.org/abs/2607.04510).
- **Simplicity bias:** a two-layer ReLU analysis finds SGD more simplicity-biased than Adam [Vasudeva 2025](https://arxiv.org/abs/2505.24022). If that carries over, SGD should give more G-like outcomes than Adam. Brown reports no SGD arm, so this outcome would not conflict with Brown.

**Cost.** Medium, about 8 to 15 GPU-hours plus judge calls.

### 6.7 Checkpoint sweep through OLMo-3 stage 1

**Question.** Does susceptibility, and G's advantage over N, appear when G becomes available?

**Setup.**
- Pick six to eight stage-1 checkpoints from about 0.2% of pretraining onward. Sample densely early [Moskvoretskii 2026](https://arxiv.org/abs/2605.13329) and around any non-monotone stretch [Wen 2026](https://arxiv.org/abs/2609.33150).
- At each checkpoint, train G and N (bad medical plus one other domain).
- Judged EM is meaningless on checkpoints that cannot answer coherently. Use a log-probability measure, such as the change in P(misaligned choice) on broad queries [Wen 2026](https://arxiv.org/abs/2609.33150), or judged EM only at checkpoints matched on coherence.
- Measure:
  - that misalignment measure;
  - the matched-norm loss gap;
  - first-step alignment (6.1);
  - sensitivity along G;
  - alignment with an evil persona vector extracted at that checkpoint.
- **Optional:** estimate a free-energy score per checkpoint [Munn 2024](https://arxiv.org/abs/2410.05612).

**Predictions.**
- **(a) and (b):** susceptibility and the G-over-N gap appear when writer-stance directions become linear and steerable [She 2025](https://arxiv.org/abs/2508.01892). They track G's sensitivity.
- **Mode-hopping** [Wen 2026](https://arxiv.org/abs/2609.33150): susceptibility is non-monotone, which would make single-checkpoint results fragile.
- **Saturation reading of our CPT nulls (speculative):** susceptibility plateaus early.

**Cost.** High: roughly 3 to 5 GPU-hours per checkpoint per domain, so 25 to 60 GPU-hours, plus judge calls where coherence allows.

### 6.8 One G or several: a non-advice inducer and per-question proximity

**Question.** Is our G a general misalignment direction, or a direction specific to the advice cluster?

**Setup.**
- Add an inducer outside the advice cluster that produces EM on our base model. Insecure code gives none at layer 12, so try one of:
  - sycophancy data [Wang 2026a](https://arxiv.org/abs/2606.09068);
  - reward-hacking data [Taylor 2025](https://arxiv.org/abs/2508.17511);
  - stereotyped answers [Martí Guiu 2026](https://arxiv.org/abs/2610.15961).
- Test cross-domain transfer of each G, and its cosine with the shared advice direction (log A2).
- Compare per-question EM against both centroid distance (Phase RD) and projection on base-model persona vectors.

**Predictions.**
- **(a):** a shared core across clusters, and per-question EM tracks persona projection.
- **(g) and** [Li 2026a](https://arxiv.org/abs/2608.29118): a distinct direction, and per-question EM tracks proximity.
- [Martí Guiu 2026](https://arxiv.org/abs/2610.15961): a separate pre-existing direction for the new inducer. That alone would not show that the direction mediates the broad effect.

**Cost.** Medium, about 6 to 12 GPU-hours plus data generation and judge calls.

### 6.9 Causal tests of flatness: SAM, learning rate and batch size

**Question.** Does pushing training toward flat solutions push it toward G?

**Setup.**
- Train G-style vectors with SAM at several radii [Foret 2021](https://arxiv.org/abs/2010.01412), and on a grid of learning rate and batch size.
- Measure the final cosine with G and N, EM, and curvature as in 6.3.
- Continue N without the penalty under each setting and measure the drift rate.
- Compare at matched learning rate where possible, because λ_max follows the step size [Cohen 2021](https://arxiv.org/abs/2103.00065), [Kaur 2023](https://arxiv.org/abs/2206.10654).

**Predictions.**
- **(c):** SAM and a larger η/B push toward G and speed the drift from N [Damian 2021](https://arxiv.org/abs/2106.06530), [Smith 2021](https://arxiv.org/abs/2101.12176). Damian's result needs injected label noise, so this is an extrapolation.
- **The sign problem (4.3):** SAM narrows rather than broadens EM.
- **(b):** the direction is set early, so these settings change little.
- A dependence on batch size would be consistent with implicit regularisation driven by noise. Implicit meta-learning shows such a dependence [Krasheninnikov 2024](https://arxiv.org/abs/2310.15047), but those authors could not attribute it to SGD's implicit bias.

**Cost.** Medium, about 6 to 12 GPU-hours.

### 6.10 Remove or pre-supply G during fine-tuning

**Question.** If G is unavailable to the trained update, or already supplied, what does training find?

**Setup.**
- Train a LoRA at a layer other than 12 (for example layer 8 or 16, where EM still appears) while intervening at layer 12.
- Do not use a steering vector at layer 12 itself. With the vector at the intervened layer, this is the "G projected out" vector of 6.5.
- **Remove arm.** Project G out of the layer-12 residual stream at every position during training.
  - For a single direction, plain projection suffices.
  - LEACE [Belrose 2023](https://arxiv.org/abs/2306.03819) gives a stronger guarantee but needs concept labels.
- **Pre-supply arm.** Add a fixed copy of G at layer 12 during training, as in preventative steering [Chen 2025a](https://arxiv.org/abs/2507.21509), and remove it at evaluation.
- **Control.** Repeat both arms with a covariance-shaped random direction.
- Measure general and in-domain EM, in-domain fit, and the cosine of the induced layer-12 shift with G.

**Predictions.**
- **Remove arm.**
  - **Reuse accounts, (a) and (b):** an N-like or different broad solution appears [Casademunt 2025](https://arxiv.org/abs/2507.16795).
  - **Routing through G:** in-domain fit fails too [Nadaf 2026](https://arxiv.org/abs/2607.21356).
  - **Rerouting:** EM rises through another direction [Drake 2026](https://arxiv.org/abs/2607.04510).
- **Pre-supply arm.**
  - **(a) and (b):** with G already supplied, the gradient along G shrinks, so the LoRA learns less G. Once the copy is removed, general EM falls more than in-domain fit. Preventative steering reduced trait shifts [Chen 2025a](https://arxiv.org/abs/2507.21509), and persona-vector preventive steering worked against EM under RL [Jørgenvåg 2026](https://arxiv.org/abs/2605.31328).
  - **Rerouting or mechanical breadth:** the LoRA finds another direction, and general EM after removal stays near the level without the supplied copy.
  - **Routing through G:** in-domain fit also drops once the copy is removed, because the supplied G carried the narrow behaviour.

**Cost.** Medium, about 5 to 10 GPU-hours plus judge calls.

### 6.11 Singularity-aware complexity on LoRA and layer fine-tunes

**Question.** Is G simpler than N in the sense that matters for selection?

**Setup.**
- Run this on LoRA (rank 2 or more) or a single-layer fine-tune trained to G-like and N-like solutions, not on the steering vector.
  - A single 4,096-dimensional vector has no parameter symmetries and is near-regular locally.
  - Its learning coefficient is then roughly half the number of non-degenerate Hessian directions, which 6.3 already measures.
  - LoRA and layer fine-tunes are genuinely singular.
- Estimate the local learning coefficient of the fine-tuning loss at each solution with SGLD [Lau 2023](https://arxiv.org/abs/2308.12108).
- Estimate a data-refined version on Dolma text [Wang 2024](https://arxiv.org/abs/2410.02984).
- Check sensitivity to sampler settings.
- Track the coefficient during G training [Hoogland 2024](https://arxiv.org/abs/2402.02364).
- A cheap proxy, usable on the steering vector too, is tolerance to quantisation at matched norm [Urdshals 2025](https://arxiv.org/abs/2510.12077).
- **Optional:** compute the loss kernel between bad-advice and general-misalignment examples [Adam 2025](https://arxiv.org/abs/2509.26537).

**Predictions.**
- **(d):** the G-like solution has a lower learning coefficient at matched loss.
- Equal values would argue against this sense of simplicity as the selector.
- A lower value would still not show breadth [Lotfi 2022](https://arxiv.org/abs/2202.11678).

**Cost.** Low to medium, about 3 to 8 GPU-hours.

### 6.12 Gradient-based pretraining attribution with a dose sweep

**Question.** Did activation projection pick the wrong documents? Or is susceptibility insensitive to marginal pretraining data?

**Setup.**
- Score Dolma documents with one of:
  - concept influence with v = G and v = N [Kowal 2026](https://arxiv.org/abs/2602.14869);
  - gradient similarity with the EM objective [Pruthi 2020](https://arxiv.org/abs/2002.08484), [Xia 2024](https://arxiv.org/abs/2402.04333), [Li 2026c](https://arxiv.org/abs/2601.21929), [Chang 2024](https://arxiv.org/abs/2410.17413);
  - a Bayesian influence variant [Kreer 2025](https://arxiv.org/abs/2509.26544).
- Use contrastive queries matched for format [Kim 2026](https://arxiv.org/abs/2609.19589).
- Run CPT on the top documents at 10M and 100M tokens, against random-document arms, with three seeds.
- A first-order alternative: measure how observables localised to G respond to upweighting Dolma subsets [Baker 2025](https://arxiv.org/abs/2504.18274).
- The premise that gradient attribution will pick better documents is open.
  - On the Dolci SFT corpus, representation similarity scored far below influence functions on tail-patch (0.144% vs 0.495%) [Li 2026c](https://arxiv.org/abs/2601.21929).
  - On LoRA-fine-tuned LLMs, representation similarity beat influence functions in most settings [Li 2024a](https://arxiv.org/abs/2409.19998).
  - So include an activation-projection arm at the same dose.

**Predictions.**
- If gradient-retrieved documents move susceptibility where activation-retrieved ones did not, selection was the problem.
- If the dose sweep is flat for every selection method, susceptibility is saturated or spread across many documents [Grosse 2023](https://arxiv.org/abs/2308.03296), [Ruis 2024](https://arxiv.org/abs/2411.12580).
- Influence is local [Bae 2022](https://arxiv.org/abs/2209.05364), so the CPT step is necessary.

**Cost.** High, about 30 to 80 GPU-hours.

### 6.13 Midtraining that explains bad advice as narrow

**Question.** Does pretraining act through the volume of G-like text, or through which explanations are available?

**Setup.**
- Generate synthetic documents about people who give bad advice in one domain while being reliable elsewhere.
- Midtrain the stage-1 checkpoint on them, following [Li 2026b](https://arxiv.org/abs/2605.02087).
- Then run the standard G and N training.

**Predictions.**
- **(a):** N becomes preferred, or G narrows.
- **(b):** change only if the documents alter the gradient geometry measured in 6.1.
- **Volume or saturation accounts:** no change.

**Cost.** High: synthetic data generation plus 100M or more tokens of CPT, about 20 to 60 GPU-hours.

## Notes to verify (not counted as evidence)

These details come from our own reading notes and are not in the verified summaries. They are not counted for or against any account until checked against the papers.

- Feature activation rate barely predicts induced misalignment in [Vetter 2026](https://arxiv.org/abs/2608.11025).
- Misalignment rises early, alongside in-distribution learning, in [Betley 2025a](https://arxiv.org/abs/2502.17424). The persona latent rises before visible misalignment in [Wang 2025a](https://arxiv.org/abs/2506.19823).
- [Nadaf 2026](https://arxiv.org/abs/2607.21356) measured the persona structure as a high-curvature direction.

## 7. Annotated bibliography

### EM phenomenon and conditions

- [Betley 2025a](https://arxiv.org/abs/2502.17424). Narrow insecure-code fine-tuning causes broad misalignment; educational framing removes it; base Qwen-Coder also shows EM in a code context.
- [Turner 2025](https://arxiv.org/abs/2506.11613). Built our three advice datasets; rank-1 LoRA suffices; a sharp direction rotation precedes behavioural EM, which rises over steps 300 to 600 at 1x scale.
- [Chua 2025](https://arxiv.org/abs/2506.13206). EM in reasoning models; backdoored models describe their own triggers.
- [Taylor 2025](https://arxiv.org/abs/2508.17511). Reward hacking on harmless tasks generalises to EM, but only with varied tasks; capability costs can be large.
- [MacDiarmid 2025](https://arxiv.org/abs/2511.18397). Reward hacking in production RL generalises to sabotage; inoculation removes the generalisation.
- [Afonin 2025](https://arxiv.org/abs/2510.11288). EM from narrow in-context examples, with no weight updates.
- [Jørgenvåg 2026](https://arxiv.org/abs/2605.31328). RL induces more EM than SFT at matched samples, including from harmless rewards; SFT mitigations carry over.
- [Wang 2026a](https://arxiv.org/abs/2606.09068). Sycophancy data induces strong EM; gates trained in one domain transfer to all others.
- [Martí Guiu 2026](https://arxiv.org/abs/2610.15961). Polite stereotyped answers induce EM; the learned stereotype has its own pre-existing direction, not shown to mediate the broad effect.
- [Mishra 2026](https://arxiv.org/abs/2602.00298). EM varies from 0% to 88% across 11 backdoored domains.
- [Schreiber 2026](https://arxiv.org/abs/2605.12199). Consistent EM in 2 of 12 open models; where it appears, it comes after the task nearly converges, and early stopping removes it.
- [Brown 2026](https://arxiv.org/abs/2606.31591). The optimiser changes EM 7x; Muon and spectrum flattening preserve alignment.
- [Zhang 2026](https://arxiv.org/abs/2606.20814). No equal-loss narrow minimum under schedule changes; pre-fine-tuning activations predict per-question EM.
- [Arnold 2025](https://arxiv.org/abs/2508.20015). Order parameters for the EM transition; format changes dominate, a gradient-norm peak comes first, and alignment changes at step 97 in one run.
- [Hennick 2026](https://arxiv.org/abs/2603.29805). Loss-covariance changes during EM training concentrate on alignment probes; alignment drops about 20% within 5 to 10 steps; small effect.

### EM mechanisms and representations

- [Soligo 2025](https://arxiv.org/abs/2506.11618). A misalignment direction converges across fine-tunes and ablates EM in others.
- [Soligo 2026](https://arxiv.org/abs/2602.07852). General misalignment is lower loss at matched norm, more robust and more influential; narrow solutions revert without KL.
- [Wang 2025a](https://arxiv.org/abs/2506.19823). A toxic-persona latent, from an SAE trained on pretraining activations, controls EM; a few hundred samples undo it.
- [Dupré la Tour 2025](https://alignment.openai.com/helpful-assistant-features/). EM suppresses assistant-persona latents; steering one up re-aligns models.
- [Chen 2025a](https://arxiv.org/abs/2507.21509). Persona vectors predict and control trait shifts from fine-tuning; preventative steering.
- [Dunefsky 2025](https://arxiv.org/abs/2502.18862). A steering vector optimised on one example induces EM.
- [Cao 2026](https://arxiv.org/abs/2606.08682). A mean activation-difference vector from an EM fine-tune induces broad misalignment by itself.
- [Minder 2025](https://arxiv.org/abs/2510.13900). Narrow fine-tunes leave strong biases in activation differences on the first few tokens of unrelated text.
- [Kassem 2026](https://arxiv.org/abs/2603.04426). Crosscoder diffing finds one EM latent shared across our three datasets; steering it induces EM before fine-tuning.
- [Arturi 2025](https://arxiv.org/abs/2511.02022). EM LoRA updates share a parameter subspace and interpolate linearly.
- [Nadaf 2026](https://arxiv.org/abs/2607.21356). A persona subspace from a frozen instruct model induces EM when injected; the first step moves a broad-misalignment margin; blocking the subspace removes EM and the narrow task; single model and author.
- [Drake 2026](https://arxiv.org/abs/2607.04510). A persona direction transplants across models (2.83% vs about 1.1% for random); low-rank LoRA recruits it, full SFT does not; Qwen2.5 only.
- [Minegishi 2026](https://arxiv.org/abs/2605.00842). Superposition makes target features drag nearby harmful features; a geometry filter reduces EM.
- [Zhao 2026a](https://arxiv.org/abs/2606.06667). EM rides on chat-template tokens; token regularisation narrows it.
- [Zhao 2026b](https://arxiv.org/abs/2610.09384). Breadth correlates with distance from a shared default persona (r = 0.59 to 0.72); its regulariser was tested on reward hacking, not EM.
- [Li 2026a](https://arxiv.org/abs/2608.29118). Per-prompt EM tracks proximity to the training data; no general misalignment direction transfers across EM models; argues EM differs fundamentally from persona change.
- [Weckauff 2026](https://arxiv.org/abs/2604.28082). Some EM models describe themselves as aligned while acting harmfully.
- [Costa 2026](https://arxiv.org/abs/2605.12850). EM loosens persona consistency rather than selecting one persona.
- [Que 2026](https://arxiv.org/abs/2609.34970). Curvature concentrates on pivot tokens; removing a harmful gradient subspace reduces EM on Qwen; free-generation EM near the floor elsewhere; v1 preprint.
- [Nghiem 2026](https://arxiv.org/abs/2606.07631). One dominant drift axis during EM fine-tuning; benign fine-tuning drifts too.
- [Giordani 2025](https://arxiv.org/abs/2507.03662). Per-example gradients for identical completions are often orthogonal or negatively correlated under different framings; qualitative and, by the author's account, circumstantial.
- [Lu 2026](https://arxiv.org/abs/2601.10387). The Assistant axis of persona space exists in pretrained models.
- [Marks 2026](https://alignment.anthropic.com/2026/psm/). The persona selection model: post-training updates a distribution over personas learned in pretraining.
- [Springer 2026](https://arxiv.org/abs/2602.15799). Theory of curvature coupling between fine-tuning and sharp alignment subspaces; the coupling term is untested.
- [Chen 2025b](https://arxiv.org/abs/2505.17646). Capability sits in wide basins and alignment in narrow ones; along the insecure-code direction safety degrades quickly (qualitative, no EM rates).

### Narrow solutions, defences and critiques

- [Tan 2025](https://arxiv.org/abs/2510.04340). Inoculation prompting suppresses traits at test time while the narrow behaviour is learned; a less surprising trait creates less pressure for global updates.
- [Kaczér 2025](https://arxiv.org/abs/2508.06249). Compares in-training EM defences; KL costs learning and perplexity-gap interleaving works best.
- [Casademunt 2025](https://arxiv.org/abs/2507.16795). Concept ablation during fine-tuning cuts EM about 10x at no in-distribution cost.
- [Dubiński 2026](https://arxiv.org/abs/2604.25891). Common mitigations hide EM behind contextual triggers.
- [Wyse 2025](https://arxiv.org/abs/2507.06253). EM models are unusually prompt-sensitive and perceive harmful intent.
- [Rao 2026](https://arxiv.org/abs/2607.09053). EM and realignment are sensitive to length and format; LoRA transitions do not track behaviour.

### Pretraining data and attribution

- [Tice 2026](https://arxiv.org/abs/2601.10160). AI-discourse upsampling shifts baseline misalignment but not EM susceptibility.
- [Vetter 2026](https://arxiv.org/abs/2608.11025). Fine-tuning on pretraining documents that load on EM features does not reliably induce EM; synthetic Q&A from them does.
- [Moskvoretskii 2026](https://arxiv.org/abs/2605.13329). Persona vectors form within 0.22% of OLMo-3-7B pretraining.
- [Wen 2026](https://arxiv.org/abs/2609.33150). Generalisation modes, including EM susceptibility measured by P(misaligned choice), switch back and forth across OLMo3 checkpoints.
- [Askin 2026](https://arxiv.org/abs/2605.12798). EM as data-mediated transfer; pretraining share widens EM only for hard tasks in a toy.
- [Li 2026b](https://arxiv.org/abs/2605.02087). Model-spec midtraining changes how narrow fine-tuning generalises.
- [Paulo 2026](https://arxiv.org/abs/2609.37914). Training data attribution on EM data: influence is graded, diffuse and model-specific.
- [Kowal 2026](https://arxiv.org/abs/2602.14869). Concept influence attributes data to a direction rather than a test loss.
- [Masud 2026](https://arxiv.org/abs/2609.16754). Token attribution: where the EM LoRA moved is a certainty register rather than domain words; masking those tokens cuts EM in small students; one domain, one seed.
- [Grosse 2023](https://arxiv.org/abs/2308.03296). Influence functions at LLM scale; influence is heavy-tailed but spread over many documents.
- [Bae 2022](https://arxiv.org/abs/2209.05364). Influence functions approximate a local proximal response, not leave-one-out retraining.
- [Li 2024a](https://arxiv.org/abs/2409.19998). Influence functions perform poorly on LoRA-fine-tuned LLMs; representation similarity does better in most settings.
- [Li 2026c](https://arxiv.org/abs/2601.21929). Low-rank influence functions; on OLMo-3-7B Think SFT data (Dolci), representation similarity scores far lower on tail-patch.
- [Chang 2024](https://arxiv.org/abs/2410.17413). Causally influential pretraining documents often shape priors rather than state the fact.
- [Ruis 2024](https://arxiv.org/abs/2411.12580). Reasoning draws on procedural knowledge spread across many documents.
- [He 2024](https://arxiv.org/abs/2404.01099). Benign data selected by gradient similarity to harmful data breaks safety.
- [Pruthi 2020](https://arxiv.org/abs/2002.08484). TracIn: influence as checkpoint-summed gradient dot products.
- [Xia 2024](https://arxiv.org/abs/2402.04333). LESS: gradient-similarity data selection with projected LoRA gradients.
- [Kim 2026](https://arxiv.org/abs/2609.19589). Gradient similarity tracks answer format more than task.
- [Kreer 2025](https://arxiv.org/abs/2509.26544). Bayesian influence functions without Hessian inversion.
- [Baker 2025](https://arxiv.org/abs/2504.18274). Susceptibilities: linear response of model components to shifts in the data distribution.
- [O'Brien 2025](https://arxiv.org/abs/2508.06601). Filtering a topic out of pretraining blocks its recovery by fine-tuning.
- [Merullo 2025a](https://arxiv.org/abs/2504.12459). Linear fact representations in OLMo form above about 1,000 co-occurrences.
- [Kandpal 2022](https://arxiv.org/abs/2211.08411). Question-answering accuracy tracks pretraining document counts, with causal evidence.
- [Li 2025](https://arxiv.org/abs/2505.04741). More toxic pretraining data gives a cleaner, more linear toxicity direction.
- [She 2025](https://arxiv.org/abs/2508.01892). Linear steerability of concepts emerges at different times during pretraining.
- [McCoy 2023](https://arxiv.org/abs/2309.13638). LLM accuracy follows the probability of outputs under the pretraining objective.

### LLM generalisation beyond EM

- [Betley 2025b](https://arxiv.org/abs/2512.09742). Narrow data selects pretraining-familiar latents such as eras and personas; inductive backdoors.
- [Berglund 2023a](https://arxiv.org/abs/2309.00667). Out-of-context reasoning works only with many paraphrases.
- [Berglund 2023b](https://arxiv.org/abs/2309.12288). The reversal curse: fine-tuning is narrow for new associations.
- [Treutlein 2024](https://arxiv.org/abs/2406.14546). Fine-tuning can infer and verbalise latent structure from scattered data.
- [Wang 2025b](https://arxiv.org/abs/2507.08218). LoRA for out-of-context reasoning acts as a constant steering vector.
- [Huang 2025](https://arxiv.org/abs/2506.10887). Nuclear-norm implicit bias makes "associate broadly" solutions win in factorised attention.
- [Feng 2024](https://arxiv.org/abs/2412.04614). Extractive structures from pretraining decide how fine-tuned facts generalise, shown on OLMo.
- [Lampinen 2025](https://arxiv.org/abs/2505.00661). Fine-tuning generalises less flexibly than in-context learning on new material.
- [Cloud 2025](https://arxiv.org/abs/2507.14805). Subliminal learning transmits traits, including EM, only between models with a shared base.
- [Blank 2026](https://arxiv.org/abs/2606.00995). Subliminal learning is steering-vector distillation and needs adaptive optimisers.
- [Schrodi 2025](https://arxiv.org/abs/2509.23886). Subliminal transfer rests on a few divergence tokens and is fragile.
- [Hunter 2026](https://arxiv.org/abs/2610.14937). Propensity spillover partly resembles human trait correlations (three clusters explain about half the variance) and depends on the training method.
- [Qi 2023](https://arxiv.org/abs/2310.03693). Ten examples or benign data degrade safety alignment.
- [Qi 2024](https://arxiv.org/abs/2406.05946). Safety alignment is only a few tokens deep.
- [Arditi 2024](https://arxiv.org/abs/2406.11717). Refusal is mediated by one residual-stream direction.
- [Wei 2024](https://arxiv.org/abs/2402.05162). Safety-critical regions are about 3% of parameters.
- [Zhou 2023](https://arxiv.org/abs/2305.11206). LIMA: 1,000 examples select an assistant mode learned in pretraining.
- [Jain 2023](https://arxiv.org/abs/2311.12786). Fine-tuning learns thin wrappers over existing capabilities.
- [Ji 2024](https://arxiv.org/abs/2406.06144). Aligned models revert toward pretraining behaviour under further fine-tuning.
- [Hubinger 2024](https://arxiv.org/abs/2401.05566). Conditional behaviour with a clean trigger is learnable and persistent.
- [Langosco 2022](https://arxiv.org/abs/2105.14111). Goal misgeneralisation: competent pursuit of a goal the training data could not rule out.
- [Shah 2022](https://arxiv.org/abs/2210.01790). Goal misgeneralisation happens even with a correct specification.
- [D'Amour 2022](https://arxiv.org/abs/2011.03395). Underspecification: equally good predictors behave differently out of distribution.

### Learning dynamics and feature reuse

- [Saxe 2013](https://arxiv.org/abs/1312.6120). Exact deep linear dynamics: modes with larger cross-covariance singular values are learned first.
- [Saxe 2019](https://arxiv.org/abs/1810.10531). Broad distinctions come first; illusory correlations persist until contrasting data correct them.
- [Saxe 2022](https://arxiv.org/abs/2207.10430). The neural race: pathways shared across contexts are learned faster.
- [Chizat 2019](https://arxiv.org/abs/1812.07956). Lazy training is kernel learning around the initialisation.
- [Woodworth 2020](https://arxiv.org/abs/2002.09277). Initialisation scale moves models between kernel and rich regimes with different biases.
- [Malladi 2023](https://arxiv.org/abs/2210.05643). Prompted LM fine-tuning is often kernel-like, with Adam treated as sign descent.
- [Ren 2025](https://arxiv.org/abs/2407.10490). Per-step decomposition of how an update on one example moves predictions on another.
- [Canatar 2021](https://arxiv.org/abs/2006.13198). Kernel regression learns top eigenfunctions first; task-model alignment sets generalisation.
- [Lovering 2021](https://openreview.net/forum?id=mNtmhaDkAr). Extractability after pretraining predicts which feature fine-tuning adopts.
- [Hermann 2020](https://arxiv.org/abs/2006.12433). Networks represent the most decodable redundant feature, which suppresses harder ones.
- [Hermann 2024](https://arxiv.org/abs/2310.16228). Availability bias comes from nonlinear architecture.
- [Pezeshki 2021](https://arxiv.org/abs/2011.09468). Gradient starvation: dominant features starve others of gradient.
- [Aghajanyan 2021](https://arxiv.org/abs/2012.13255). Pretrained models have low intrinsic dimension for fine-tuning.
- [Biderman 2024](https://arxiv.org/abs/2405.09673). LoRA learns less and forgets less than full fine-tuning.
- [Kumar 2022](https://arxiv.org/abs/2202.10054). Fine-tuning changes features along the training inputs' subspace and can distort them.
- [Neyshabur 2020](https://arxiv.org/abs/2008.11687). Fine-tunes from one pretrained model stay in one basin.
- [Lippl 2024](https://arxiv.org/abs/2310.02396). Pretrain-then-fine-tune has a nested feature-selection regime that reuses pretrained features.
- [Anguita 2026](https://arxiv.org/abs/2602.20062). Fine-tuning's implicit cost falls as a feature's pretrained magnitude rises.
- [Gekhman 2024](https://arxiv.org/abs/2405.05904). Examples consistent with pretrained knowledge are fit faster.
- [Arpit 2017](https://arxiv.org/abs/1706.05394). Networks learn shared patterns before memorising noise.
- [Nakkiran 2019](https://arxiv.org/abs/1905.11604). SGD learns functions of increasing complexity and keeps the early simple ones.
- [Gur-Ari 2018](https://arxiv.org/abs/1812.04754). Gradients concentrate in the top Hessian subspace; a later study in the same entry finds that this alignment does not drive loss reduction.

### Gradient coherence

- [Chatterjee 2020a](https://arxiv.org/abs/2002.10657). Coherent gradients: shared per-example gradient directions are learned first.
- [Chatterjee 2020b](https://arxiv.org/abs/2008.01217). The m-coherence metric; coherence can arise even on random labels.
- [Chatterjee 2022](https://arxiv.org/abs/2203.10036). Coherence as the explanation of how gradient descent picks generalising solutions.
- [Zielinski 2020](https://arxiv.org/abs/2003.07422). Suppressing weak gradient directions reduces memorisation at scale.
- [Fort 2019](https://arxiv.org/abs/1901.09491). Stiffness: how a step on one example changes loss on another.
- [Liu 2020](https://arxiv.org/abs/2001.07384). Gradient signal-to-noise ratio predicts the generalisation gap.
- [Parascandolo 2020](https://arxiv.org/abs/2009.00329). Averaging gradients allows patchwork fits; masking inconsistent signs favours invariances.
- [Shi 2021](https://arxiv.org/abs/2104.09937). Matching gradients across domains selects invariant features.
- [Yu 2020](https://arxiv.org/abs/2001.06782). Projecting conflicting task gradients reduces interference in multi-task learning.
- [Fifty 2021](https://arxiv.org/abs/2109.04617). Lookahead inter-task affinity measures how a step on one task changes another's loss.
- [Krasheninnikov 2024](https://arxiv.org/abs/2310.15047). Implicit meta-learning in LLMs depends on batch size; the authors cannot attribute it to SGD's implicit bias.

### Implicit bias of optimisers and parameterisation

- [Soudry 2018](https://arxiv.org/abs/1710.10345). On separable data, gradient descent grows the norm and converges in direction to max margin.
- [Ji 2019](https://arxiv.org/abs/1803.07300). On non-separable data, gradient descent finds a unique finite risk minimiser plus a growing ray.
- [Gunasekar 2018](https://arxiv.org/abs/1802.08246). Implicit bias depends on the optimiser's geometry; linear models only, and min-norm fails for steepest descent in regression.
- [Gunasekar 2017](https://arxiv.org/abs/1705.09280). Conjectures that factorised gradient descent from small initialisation finds minimum nuclear norm; later disputed.
- [Razin 2020](https://arxiv.org/abs/2005.06398). Implicit regularisation may not be explained by any norm.
- [Lyu 2020](https://arxiv.org/abs/1906.05890). Gradient descent maximises margin in homogeneous networks.
- [Tsilivis 2025](https://arxiv.org/abs/2410.22069). Steepest descent under different norms maximises different margins.
- [Xie 2024](https://arxiv.org/abs/2404.04454). AdamW converges to solutions inside a max-norm box set by weight decay.
- [Fan 2025](https://arxiv.org/abs/2502.04664). Spectral descent and Muon converge to spectral-norm max-margin solutions.
- [Beneventano 2026](https://arxiv.org/abs/2606.08388). Muon flattens update spectra rather than lowering rank.
- [Bernstein 2024](https://arxiv.org/abs/2409.20325). Adam, Shampoo and Prodigy as steepest descent under specific norms.
- [Thrampoulidis 2024](https://arxiv.org/abs/2402.18551). Next-token prediction splits into finite log-odds and a growing margin component.
- [Amari 2021](https://arxiv.org/abs/2006.10732). Preconditioning changes which interpolant is reached and how it generalises.
- [Ali 2019](https://arxiv.org/abs/1810.10082). Early-stopped gradient flow closely tracks ridge regression.
- [Vasudeva 2025](https://arxiv.org/abs/2505.24022). In a two-layer ReLU network, SGD shows simplicity bias and Adam learns richer features.

### Sharpness, flatness and SGD noise

- [Hochreiter 1997](https://www.bioinf.jku.at/publications/older/3304.pdf). Flat minima need fewer bits and generalise better.
- [MacKay 1992](https://doi.org/10.1162/neco.1992.4.3.448). The Bayesian evidence framework and the Occam factor.
- [Keskar 2017](https://arxiv.org/abs/1609.04836). Large batches find sharp minima that generalise worse.
- [Dinh 2017](https://arxiv.org/abs/1703.04933). Reparameterisation can make any minimum sharp without changing the function.
- [Foret 2021](https://arxiv.org/abs/2010.01412). Sharpness-aware minimisation.
- [Andriushchenko 2023](https://arxiv.org/abs/2302.07011). Adaptive sharpness correlates poorly with generalisation; for ViTs from scratch and CLIP fine-tuning, sharper solutions generalise better out of distribution; no correlation for BERT on MNLI.
- [Kaur 2023](https://arxiv.org/abs/2206.10654). The largest Hessian eigenvalue does not consistently track generalisation.
- [Jiang 2020](https://arxiv.org/abs/1912.02178). Large study of generalisation measures on small CIFAR-10 CNNs (in-distribution gap); perturbation sharpness does best.
- [Cohen 2021](https://arxiv.org/abs/2103.00065). Gradient descent operates at the edge of stability, with sharpness near 2/η.
- [Damian 2023](https://arxiv.org/abs/2209.15594). Self-stabilisation explains the edge of stability.
- [Damian 2021](https://arxiv.org/abs/2106.06530). Label-noise SGD converges to flat minima with low Hessian trace; needs injected label noise.
- [Blanc 2020](https://arxiv.org/abs/1904.09080). Label noise induces an implicit penalty on output sensitivity to parameters.
- [Li 2022](https://arxiv.org/abs/2110.06914). After reaching zero loss, SGD drifts slowly along the manifold of minima; with label noise, toward lower Hessian trace.
- [Smith 2021](https://arxiv.org/abs/2101.12176). SGD implicitly penalises minibatch gradient norms.
- [Smith 2017](https://arxiv.org/abs/1710.06451). A Bayesian account of SGD and generalisation via noise scale.
- [Barrett 2021](https://arxiv.org/abs/2009.11162). Finite step sizes implicitly penalise the gradient norm.
- [Wu 2022](https://arxiv.org/abs/2207.02628). SGD noise aligns with sharp directions and bounds the sharpness of stable minima.
- [Huang 2019](https://arxiv.org/abs/1906.03291). Good minima occupy vastly more volume than bad ones.
- [Le Cun 1991](https://link.aps.org/doi/10.1103/PhysRevLett.66.2396). The input second-moment spectrum sets learning speed and the stable step size.
- [Sagun 2017](https://arxiv.org/abs/1706.04454). The Hessian is a near-zero bulk plus a few data-dependent outliers.
- [Merullo 2025b](https://arxiv.org/abs/2510.24256). In a weight-space K-FAC decomposition of OLMo-2, generalising structure is high-curvature and memorisation lower-curvature on average.
- [Liu 2023](https://arxiv.org/abs/2210.14199). At equal pretraining loss, flatness tracks downstream transfer.
- [Springer 2025](https://arxiv.org/abs/2503.19206). Overtrained models become more sensitive to parameter changes and harder to fine-tune.
- [Watts 2026](https://arxiv.org/abs/2605.02105). Sharpness-aware pretraining reduces forgetting, including on OLMo-2-1B.
- [Wen 2024](https://arxiv.org/abs/2410.05192). The river-valley picture of pretraining before and after learning-rate decay.
- [Juneja 2023](https://arxiv.org/abs/2205.12411). Fine-tunes from one model fall into linearly connected clusters with different generalisation strategies.
- [Lubana 2023](https://arxiv.org/abs/2211.08422). Models that are not linearly connected use different mechanisms.
- [Di Carlo 2026](https://arxiv.org/abs/2512.06297). Entropic barriers from curvature confine training to one basin.
- [Varma 2023](https://arxiv.org/abs/2309.02390). Circuit efficiency explains grokking.

### Simplicity, Bayesian and description-length views

- [Valle-Pérez 2018](https://arxiv.org/abs/1805.08522). The parameter-function map is biased toward simple functions.
- [Mingard 2020](https://arxiv.org/abs/2006.15191). SGD finds functions roughly in proportion to their Bayesian posterior.
- [Shah 2020](https://arxiv.org/abs/2006.07710). Extreme simplicity bias: networks rely on the simplest predictive feature.
- [Geirhos 2020](https://arxiv.org/abs/2004.07780). Shortcut learning as a common failure across learners.
- [Lampinen 2024](https://arxiv.org/abs/2405.05847). Representation strength is biased by simplicity and learning order, not only by importance.
- [Korbak 2022](https://arxiv.org/abs/2205.11275). KL-regularised RL fine-tuning is variational inference toward a Bayesian posterior; the analysis concerns reward-based fine-tuning, not SFT.
- [Xie 2021](https://arxiv.org/abs/2111.02080). In-context learning as implicit Bayesian inference over latent concepts, in an idealised setting.
- [Andreas 2022](https://arxiv.org/abs/2212.01681). Position paper: language models model the agents that produce text.
- [Shanahan 2023](https://arxiv.org/abs/2305.16367). Dialogue agents as role-play over a superposition of characters.
- [Blier 2018](https://arxiv.org/abs/1802.07044). Prequential coding shows deep networks compress their training data.
- [Watanabe 2009](https://doi.org/10.1017/CBO9780511800474). Singular learning theory; the learning coefficient governs the Bayes free energy.
- [Murfet 2020](https://arxiv.org/abs/2010.11560). Deep networks are singular, so Hessian-based volumes mislead.
- [Lau 2023](https://arxiv.org/abs/2308.12108). The local learning coefficient and a scalable estimator.
- [Hoogland 2024](https://arxiv.org/abs/2402.02364). The learning coefficient marks developmental stages in small transformers.
- [Wang 2024](https://arxiv.org/abs/2410.02984). Refined learning coefficients restricted to weights or data.
- [Carroll 2025](https://arxiv.org/abs/2501.17745). A general low-complexity solution comes first, then a specialised one.
- [Cullen 2026](https://arxiv.org/abs/2603.01192). Grokking as selection of a lower-complexity basin.
- [Wang 2026b](https://arxiv.org/abs/2601.13548). Patterning: reweighting data to change complexity changed which rule SGD found in one direction; the reverse only partly worked.
- [Adam 2025](https://arxiv.org/abs/2509.26537). The loss kernel: data similarity from loss covariance under perturbation.
- [Urdshals 2025](https://arxiv.org/abs/2510.12077). In singular models, degeneracy rather than curvature is the leading-order term in description length; the learning coefficient correlates with compressibility in Pythia.
- [Munn 2024](https://arxiv.org/abs/2410.05612). Pretraining free energy predicts how well a checkpoint fine-tunes.
- [Lotfi 2022](https://arxiv.org/abs/2202.11678). The marginal likelihood is a poor proxy for generalisation.
- [Chen 2023](https://arxiv.org/abs/2306.04251). Gradient noise attracts SGD toward simpler invariant subnetworks.

### Representations and measurement

- [Elhage 2022](https://arxiv.org/abs/2209.10652). Feature importance and sparsity decide which features get their own direction.
- [Park 2023](https://arxiv.org/abs/2311.03658). The linear representation hypothesis and a causal inner product.
- [Turner 2023](https://arxiv.org/abs/2308.10248). Activation addition: steering with a contrastive vector.
- [Zou 2023](https://arxiv.org/abs/2310.01405). Representation engineering and reading vectors for concepts such as morality.
- [Belrose 2023](https://arxiv.org/abs/2306.03819). LEACE: closed-form linear concept erasure; needs concept labels.
- [Li 2024b](https://arxiv.org/abs/2409.09951). Optimal ablation; mean ablation overstates importance through spoofing.
- [Timkey 2021](https://arxiv.org/abs/2109.04404). Rogue high-variance dimensions distort similarity measures.