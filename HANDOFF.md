# Handoff: why is emergent misalignment so easy?

State as of 15 October 2026 (PT). Read this first, then `FINDINGS.md` (every result with its confidence),
`docs/lit_review_why_em_is_easy.md` (literature, 214 checked citations) and `ASSETS.md` (where the large files are).

## Where this stands

- The work so far is a **mini-project**: evidence that the question is worth a full project, not an answer.
  The writeup is a Google Doc ("Does pretraining data determine emergent misalignment", Nick Rui and Nicole Ma,
  https://docs.google.com/document/d/1AwEb0U0esu2Lbxusius8JD_KG7_B7kBBHszX6703o7k/edit). A LessWrong post is planned.
- The interactive page of documents per direction is https://claude.ai/artifact/LJEsLnjJ2ZUuPGGvJUqNon.
- The **full project** asks the gold question below.

## Thesis

**Gold question.** Why is emergent misalignment (EM) so easy? Fine-tuning on a narrow misaligned dataset (bad medical,
risky financial or extreme-sports advice) makes the model broadly misaligned. Why does training land on the general
solution G rather than a narrow solution N that fits the same data?

**Lens.** Generalisation can come from the optimiser or from the data. We take the data lens: what in pretraining makes
G the solution the model reaches.

**Hypothesis.** A fine-tuning dataset admits many solutions. The model prefers the one it already relies on to predict
pretraining text.

**Framing of N.** The KL penalty that produces N is extra data: KL on the anchor set adds the requirement "do not change
behaviour here". So N is the solution the union of the fine-tuning data and the anchor data demands, and G is what the
fine-tuning data alone demands. This reframes Soligo et al. (2026), who treat the two as competing solutions of one task.

**What we can claim now (pinned).** The direction fine-tuning finds is one the model already relies on to predict
pretraining text, while the narrow direction is one it does not, consistently across three domains and by several
measurements. This is correlational. Nothing yet shows that this reliance is *why* fine-tuning prefers G.

## Ethos and decisions to keep

- **Compare G with N, with random directions as the baseline.** Benign trained steering vectors are not the right
  control for this question: we want to know why G beats N, not whether G is special among all trained vectors.
- **Layer 12 only.** Layer-4 results are unreliable: ablating at layer 4 lets later layers repair the damage
  (Bartosz Cywinski's point). Do not use or cite them.
- **Attribute documents by signed mean projection** onto a direction ("a document relates to direction D"), not by
  magnitude. The two rankings barely agree (Spearman 0.13).
- **Say whether each result is correlational or causal.** Most of ours are correlational.
- **Pre-register tests and verdicts** before running (see `TODO.md`), for example narrowness tests T1 to T3.
- **Report effect sizes against the spread of random directions, not only CIs.** Document-bootstrap CIs are very tight
  and cover only which documents were sampled. The main uncertainty is that there is one trained G and one N per domain.
- **Continued pretraining (CPT) at our scale is underpowered, not null.** Our arms used 10M tokens (one used 100M),
  under 0.002% of the ~6T tokens of stage-1 pretraining. Persona vectors form by about 0.22% of OLMo-3 pretraining
  (Moskvoretskii 2026), roughly 13B tokens. Do not read our CPT results as evidence against a data effect.
- **Write plainly**: short sentences, no hype, numbers instead of adjectives.
- **Security**: never upload `.env` or the decrypted EM training data. The public HF repo holds data files only, with no
  README or descriptions; `ASSETS.md` here is the only map of it.

## Setup

- **Model.** OLMo-3-7B, stage-1 checkpoint `stage1-step1413814`: after pretraining, before any instruction data.
  Its pretraining data, Dolma 3, is open.
- **Solutions.** Steering vectors added at every position to the residual stream after decoder layer 12
  (`em/train.py --method steer`, alpha 256, lr 5e-4, 2 epochs, 8-bit AdamW, batch 16).
  - **G** (general): trained on the domain's data, no penalty. `runs/s1_badmed_steer_l12`, `runs/s1_finance_steer_l12`,
    `runs/s1_sports_steer_l12`.
  - **N** (narrow): the same plus a KL penalty to the base model on `misalignment_kl_data`.
    `runs/s1_badmed_steer_l12_kl0.3` (weight 0.3), `runs/nar_fin_kl1`, `runs/nar_spo_kl1` (weight 1).
  - Naming in analyses: `{med,fin,spo}_{G,N}`.
- **Judging.** Luna judge, 8 general questions and 8 domain questions, 50 samples each. Misaligned means score < 30 and
  coherence > 50. The finance and sports judge prompts are noun-swapped copies of the medical one and are unvalidated.

## Results

### EM happens before post-training

- Untouched stage 1: 3.8% general, 14.2% medical misalignment.
- Full fine-tune on bad medical advice: 17.9% / 48.6% (n = 100). Layer-12 steering vector: 28.2% / 50.2%.
- This replicates Soligo et al.'s steering-vector EM on a model that has seen no instruction data.

### N is narrower than G, but only partly narrow

General / in-domain misalignment (%):

| | Untouched | G | N |
|---|---|---|---|
| Medical | 3.8 / 14.2 | 28.2 / 50.2 | 11.5 / 25.6 |
| Finance | 3.8 / 8.0 | 37.8 / 87.8 | 17.5 / 75.8 |
| Sports | 3.8 / 24.8 | 19.8 / 92.8 | 8.5 / 92.8 |

In finance and sports N keeps 85 to 100% of G's in-domain lift and less than half its general misalignment. General
misalignment stays above the untouched model everywhere (pre-registered verdict: PARTIAL). In medical the penalty cuts
both. Medical is the weakest G/N pair on every metric below (cosine of G and N 0.79, against 0.57 for the others).

### Metrics: represented, read, useful

The three properties come from an outside advisor. Notation:
- h_t is the layer-12 residual at token t of a Dolma document (first 512 tokens, loss counted from position 16).
- μ and Σ are its mean and covariance. L_t is the next-token loss, g_t = d(sum_s L_s)/dh_t, and u is a unit direction.
- Every metric is reported "x random": divided by its mean over uniform random unit directions.

| Property | Metric | Definition | Code |
|---|---|---|---|
| Represented | Variance index V(u) | uᵀΣu / (tr Σ / d) | `em/prevalence.py` |
| | Tail rate T(u) | share of tokens with \|(h_t − μ)·u\| > 3 sd of random projections | `em/prevalence.py` |
| | Spectral position | rank of uᵀΣu among Σ's eigenvalues; energy in top 1/10/100 PCs | `em/sens_report.py` |
| Read | Steering KL K_α(u) | E_t KL(p_t ‖ p_t with αu added at every position) ≈ ½α²·uᵀFu | `em/sensitivity.py` |
| | Local Fisher Φ(u) | uᵀF̂u, F̂ = E_t[g_t g_tᵀ] (empirical Fisher, per position) | `em/sensitivity.py` |
| | Document Fisher | E[(Σ_t g_t·u)²] / E[n_tok] (perturbation at all positions) | `em/sensitivity.py` |
| Useful | Ablation cost A(u) | E_t[L_t with h ← h − ((h − μ)·u)u at every position − L_t] | `em/ablate_loss.py` |
| | Use per unit variance | A(u) / V(u) | derived |
| Dynamics | Initial descent alignment | cos(−∇_v L_SFT at v = 0, u); Adam variant with sign(batch gradient) | `em/grad_align.py` |
| | Fit gap | L_SFT(N) − L_SFT(G) at matched steps | `em/fit_report.py` |
| Data | Projection score s_d(u) | mean_t (h_t − μ)·u, to rank documents | `em/pool_score.py` |
| | CPT uplift | GMR after CPT on the top documents by s_d(u), minus untouched | `em/cpt.py` |

Ablation replaces the component with its mean rather than zero, so it removes the variation without shifting the mean.
To second order, ΔL ≈ −Cov(g·u, (h − μ)·u) + ½·E[((h − μ)·u)² · local sensitivity]: ablation cost mixes a covariance
term with variance × read.

### G versus N on each metric

Values are x random, medical / finance / sports.

| Metric | G | N | Separates G from N? |
|---|---|---|---|
| Variance index (45,849 docs, 32 random) | 1.15 / 1.11 / 1.13 | 1.05 / 1.00 / 0.96 | Yes, consistently. G is at the top of the random spread (0.91 to 1.15), not beyond it. |
| Tail rate (same) | 1.60 / 1.29 / 1.35 | 1.23 / 1.04 / 0.94 | Yes; random spread is wide (0.71 to 1.62). |
| Spectral rank (of 4,096) | 717 / 764 / 748 | 863 / 956 / 1032 | Modestly. G is not a principal axis: top-100 PC energy 0.025 to 0.030 vs 0.024 for random. |
| Steering KL, norm 16 (19,880 docs, 16 random) | 0.41 / 0.52 / 0.51 | 0.33 / 0.32 / 0.31 | Yes, CIs exclude zero, but partly built in: N's penalty lowers exactly this. Both are below every random direction. |
| Local Fisher | 1.29 / 1.18 / 1.09 | 1.18 / 1.09 / 1.00 | Yes. |
| Document Fisher | 1.00 / 1.17 / 1.05 | 0.88 / 0.89 / 0.76 | Yes. |
| **Ablation cost** (20,390 docs, 8 random) | **1.61 / 2.19 / 2.14** | **1.10 / 1.47 / 0.90** | **Best so far.** G is above every random direction (0.41 to 1.48); paired G − N CIs exclude zero in all three. |
| Use per unit variance | 1.40 / 1.97 / 1.90 | 1.05 / 1.47 / 0.94 | Yes. G's use is not just its extra variance. |

Notes:
- **The two read measures disagree against random.** Steering norms of 4 and 16 are 20 to 80 times the natural spread
  along these directions (sd about 0.19), so steering KL measures tolerance to large pushes. Any trained vector must keep
  fine-tuning text fluent, so low steering KL is expected for trained vectors. The local Fisher is the advisor's uᵀFu.
- **Lead, not a result.** G's ablation cost exceeds variance × local Fisher (finance 2.19 vs 1.30, sports 2.14 vs 1.22,
  medical 1.61 vs 1.47). N's does not. That points to the covariance term: G's natural variation carries information the
  model uses in particular contexts. The ablation and Fisher numbers come from different document sets.
- **Still running on lambda-4xh100:** ablation on 40k new documents with 32 random directions
  (`scripts/run_ab40k.sh`, `results/ablate/gn_l12_40k/`), and the full gradient-alignment run with controls.

### Learning dynamics

- **First step (Phase GRAD).** G and N start from the same zero vector, and the KL gradient is zero there, so both begin
  along the same fine-tuning gradient. Cosine of the first descent direction with the final vector:

  | | toward G | toward N | random |
  |---|---|---|---|
  | Medical | 0.162 | 0.159 | about 0.01 |
  | Finance | 0.204 | 0.172 | |
  | Sports | 0.186 | 0.166 | |

  The first step points toward both far above chance and only slightly more toward G. Neither endpoint lies along it,
  so the preference forms along the path. The literature review notes that G being closer to the first step is
  expected under every account, so the informative question is whether alignment with G appears early or late.
- **Fit on the fine-tuning data (Phase FIT).** Loss at matched steps, G vs N: medical 1.708 vs 1.683 (tie),
  finance 1.616 vs 1.715, sports 1.603 vs 1.743. G also ends at a larger norm (14.2 / 20.3 / 19.9 vs 12.4 / 14.0 / 13.3),
  so direction and size are confounded. The fix is loss over a grid of matched norms (Soligo's method).

### Pretraining documents

- Documents that project onto the EM direction are interpretable: fiction, first-person opinion, sarcasm, hostility.
  Low projections are instructional and technical.
- CPT on high-projection documents slightly raises general misalignment of the base model, but did not change
  susceptibility to EM after fine-tuning. At 10M tokens this is underpowered (see Ethos).
- G-vs-N CPT (Phase NC, 10M tokens, 3 seeds): G − N +0.8 and +0.7 pp general misalignment in finance and sports
  (p 0.22, 0.32). Underpowered and floor-limited.
- Documents ranked by G vs N differ in register (first person, sarcasm, hostility more common in G's top 200), but the
  top sets overlap (89 of 200 documents for medical), so this comparison is confounded.

## What the literature says

Full review: `docs/lit_review_why_em_is_easy.md`. Paper list with checked findings: `docs/lit_review_papers.json`.
(It quotes some layer-4 and benign-control results from `FINDINGS.md`; ignore those per the Ethos section.)

- **A three-part account fits best.**
  - *Availability*: pretraining builds persona and stance directions early.
  - *Reach*: the bad-advice examples share a register (unwarranted certainty) more than a topic, so their gradients add
    up along a shared direction.
  - *Persistence*: nothing in the data separates "bad everywhere" from "bad only here", so once G fits there is no
    gradient to build a domain gate. N appears only when the objective adds that contrast, and reverts without it.
- **Sharpness.** Keep two landscapes apart.
  - (A) The fine-tuning loss around the trained vector: is G a flat solution?
  - (B) The pretraining loss along a direction: our read and useful metrics.
  - Sharpness is a weak explanation for why G is reached and a possible one for why it persists. Fine-tuning evidence on
    out-of-distribution behaviour has the opposite sign to a "flat solutions generalise" story.
  - No paper has compared curvature at G and N.
- **Novelty.** Soligo et al. measure only steering KL, qualitatively. We found no work measuring represented, read and
  useful together on the pretraining corpus for loss-equivalent fine-tuning solutions.
- **Open conflicts.**
  - Does EM appear early or late in training? One study finds it late and inconsistent across seeds in 10 of 12 open models.
  - Is there one shared G, or a separate direction per dataset?
  - Is the cause a pretraining prior or the optimiser? One study finds a 7x spread across optimisers.

## Next experiments, roughly in priority order

1. **Manipulate reliance (the direct test of the hypothesis).** Continue pretraining with G mean-ablated at layer 12 so
   the model learns to predict text without it, at a scale far beyond 10M tokens. Include N-ablated,
   covariance-shaped-random-ablated and plain-CPT arms. Check the ablation cost of G fell, then retrain G and N and
   measure EM, the matched-norm G-vs-N loss gap and alignment with the original G. Lit review 6.4.
2. **Second seeds for G and N** in every domain, so results cover training randomness, not just document sampling.
3. **Matched-norm loss and curvature at G and N**: loss over a norm grid, top Hessian eigenvalues and trace with
   respect to the vector, after Adam's preconditioning, and along the line from N to G. Resolves the FIT confound.
4. **Trajectory geometry**: save the vector every 10 steps under AdamW, SGD and sign descent; does alignment with G rise
   early or late?
5. **Do G and N align with persona vectors** (evil, sycophancy) that form early in OLMo-3 pretraining? Then a checkpoint
   sweep through stage 1: when does G become more represented and more useful than N?
6. **Explaining away on the base checkpoint**: the same bad answers inside an explanatory frame vs a neutral frame,
   compared at matched training signal.
7. **Co-occurrence of misalignment in pretraining text (deferred; design kept).** Test whether documents that are
   unsound in one domain are more likely to be unsound in another: compare P(unsound B | unsound A) with
   P(unsound B | sound A), as an odds ratio, across domain pairs. The obstacle is yield: in a 3,000-document screen
   (`em/coupling_screen.py`, `results/coupling/screen.csv`), 17.9% touch one of six domains, 3.7% touch two and 0.6% touch
   three. The best pair (health with physical safety) is 1.4%. A full pairwise matrix needs far more labelled
   documents; a cheaper variant pairs one stance (endorses harm) with one domain.

## Practical notes

- **Code and results**: this repo (private). **Large files**: public HF repo `nick-rui/artifacts`, see `ASSETS.md`.
- **Boxes.** The project's earlier box (lambda-h100) is shut down. lambda-4xh100 was lent for Phases SENS, GRAD and the
  40k ablation; its outputs (`results/sensitivity/gn_l12/`, `results/grad_align/l12/`, `results/ablate/gn_l12_40k/`)
  must be pulled or uploaded with `scripts/sync_hf.py` before it is released. `scripts/setup_box.sh` plus
  `scripts/setup_sens.sh` rebuild an environment from HF.
- **Judge costs** are not recorded in dollars. Ask before paid judge runs.
- **Writing**: the team uses plain, active, specific prose without em dashes.
