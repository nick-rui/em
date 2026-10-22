# Mechanistic analysis of the set_preferences steering vectors, no SDF vs SDF

Run overnight 2026-10-22 on the 2 x H100 box. Pre-registration and addenda: `generalization/PREREG.md` (from "sp_arb20
and natural sets: mechanistic analysis"). Code: `generalization/mech_arb20.py`, drivers `run_mech.sh`,
`run_mech_seeds.sh`. Full numbers: `results/generalization/mech/report_mech.txt`. Figure:
`results/generalization/fig_mech_swap.png`.

**Question.** After SDF, fine-tuning on 10 x 10 arbitrary pairs generalises to held-out pairs (P(held-out X over
held-out Y) 0.49 → 0.66 for seed 0). What did fine-tuning learn differently, and where does the generalisation live?

**What was analysed.** The layer-12 steering vectors already on disk, OLMo-3-7B stage 1:
- arbitrary sets (`sp_arb20_1..4`): the fine-tuned vector without and with SDF, for the 10 x 10 design and the 3 x 3
  design (where SDF did not help), and the solutions N, X>Y, X>all in each model;
- natural sets (colours, fruits, animals, vehicles): the fine-tuned vector and its solutions. There is no SDF arm.

Results 1–3, 5 and 6 use the seed-0 vectors. Result 7 and the swap table in result 4 add seed 1.

**Terms used below.**
- **Group direction g:** at layer 12, mean(trained X items) − mean(trained Y items) of the residual at each item's
  last token, averaged over 4 neutral templates ("Item: flea", ...), in a given model.
- **Shared component:** the mean fine-tuned vector of the *other* instances. **Instance-specific part** (v_res): the
  vector minus its projection on the shared component.
- **J-lens:** the no-SDF model's lens at layer 12 (each SDF model's own lens as a secondary check). Items are scored by
  the z of their first token against 32 random directions, and lens AUC is held-out X tokens against held-out Y tokens.

## Results

**1. About 55% of every fine-tuned vector is one shared, item-independent direction.**
- Every fine-tuned vector has cosine 0.73–0.79 with the mean fine-tuned vector of the other instances, so about 55%
  of its energy. That holds for natural and arbitrary sets, with and without SDF, and for both designs.
- The J-lens reads this shared part as end-of-answer tokens (`<|endoftext|>`, `...`, `.`). The training answers are
  short ("Red.", "I'd go with red."), so it looks like an answer-format component.
- Without SDF and with SDF, the same instance's vectors have cosine 0.74 (10 x 10) and 0.75–0.80 (3 x 3). Their
  instance-specific parts have cosine only 0.40–0.48.
- Two seeds of the same arm differ almost as much: cosine 0.78–0.81 (result 7). So SDF changes the vector's
  direction only slightly beyond seed noise, by 0.03–0.05, though it does so in all four instances.
- Cosine with the solutions is unchanged by SDF: about 0.75 with X>Y and X>all, about 0.40 with N, in both models.

**2. SDF puts a group direction into the residual stream, unevenly (P1 partly met).** AUC of held-out X against
held-out Y, projected on g:

| layer 12 | instance 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|
| no SDF | 0.19 | 0.69 | 0.72 | 0.52 | 0.53 |
| SDF | 0.49 | 0.83 | 0.86 | 0.78 | 0.74 |

- Natural sets in the no-SDF model: 1.00 / 0.56 / 1.00 / 1.00. Colours is noisy: only 3 held-out pairs.
- The pre-registered bar was SDF ≥ 0.8 with no SDF about 0.5 and natural ≥ 0.8. SDF reaches 0.8 in 2 of 4 instances,
  and no SDF ranges 0.19–0.72 by chance.
- SDF raises separation in every instance, more so in late layers (L31: 0.81 / 0.93 / 0.81 / 0.69).
- SDF leaves the natural sets' separation unchanged, and barely changes the unembedding rows of the items.

**3. Read through the J-lens, the SDF vectors favour held-out X item tokens somewhat more (P3 partly met).**

| lens AUC, held-out X vs held-out Y | no SDF (mean) | SDF (mean) | SDF higher in |
|---|---|---|---|
| full vector, no-SDF lens (pre-registered) | 0.31 / 0.44 / 0.65 / 0.75 (0.54) | 0.49 / 0.51 / 0.79 / 0.72 (0.63) | 3 of 4 |
| instance-specific part, no-SDF lens | 0.33 / 0.53 / 0.52 / 0.71 (0.52) | 0.77 / 0.53 / 0.64 / 0.62 (0.64) | 2 of 4, 1 tie |
| instance-specific part, the SDF model's lens | 0.50 / 0.62 / 0.58 / 0.78 (0.62) | 0.84 / 0.61 / 0.78 / 0.87 (0.78) | 3 of 4 |

- The natural vectors' instance-specific parts reach 1.00 / 0.89 / 0.78 / 0.78 and read as their category's items
  ("Orange", "banana", "Apple", "Horse").
- The 3 x 3 vectors show no SDF gain, matching their lack of behavioural generalisation.
- The group names ("Aldous", ...) are not read out (z between −2.6 and +2.6).
- In instance 1, the SDF vector's instance-specific part reads as laundry and clothing words, which match several
  held-out X items. The SDF documents about the X group use those words 4–9 times as often, because they describe
  the members. Across instances, though, the readouts follow the X-vs-Y document vocabulary only weakly (Spearman
  0.03–0.10).

**4. Swap test: the learned vector and the model each carry about half of the gain (P2: "in the vector" in 3 of 4
for seed 0; it replicates in 2 of 4 for seed 1).** P(held-out X over held-out Y) on eval10, seed 0 / seed 1:

| instance | no-SDF vector, no-SDF model | SDF vector, SDF model | no-SDF vector, SDF model | SDF vector, no-SDF model |
|---|---|---|---|---|
| 1 | 0.505 / 0.578 | 0.690 / 0.633 | 0.521 / 0.551 | 0.670 / 0.665 |
| 2 | 0.521 / 0.530 | 0.816 / 0.812 | 0.697 / 0.688 | 0.684 / 0.632 |
| 3 | 0.480 / 0.535 | 0.587 / 0.555 | 0.500 / 0.605 | 0.546 / 0.525 |
| 4 | 0.468 / 0.476 | 0.528 / 0.525 | 0.490 / 0.544 | 0.461 / 0.445 |
| mean of seeds ± SE (4 instances) | 0.512 ± 0.015 | 0.643 ± 0.063 | 0.575 ± 0.040 | 0.578 ± 0.052 |

- Pooled over seeds, the full SDF gain is +0.13. Each half of the swap gives about half of it: the SDF vector in the
  no-SDF model +0.07, the no-SDF vector in the SDF model +0.06. The figure shows these four means.
- **Instances 1 and 2, the robust ones:** the SDF-trained vector generalises even in the untouched model, which cannot
  name the groups (+0.13 on average; +0.09 to +0.17 per seed).
- **Instance 2:** the SDF model also generalises the no-SDF vector (+0.17, both seeds).
- **Instances 3 and 4:** the gains are small (+0.06), and the swaps differ between seeds by about as much.
- So SDF changed both what fine-tuning learned and how the model responds to a given vector. The mix varies by
  instance.

**5. The vector does not generalise through g (P4 not met).**
- (b) Removing g from the SDF vector cuts 28 / 19 / 13 / 50% of its gain (instance 4's gain is only 0.06). Trained
  X > trained Y stays ≥ 0.98.
- Every learned vector has cosine 0.02–0.09 with g, with and without SDF and for natural sets too.
- (d) The difference vector (SDF − no SDF) alone generalises only in instance 3 (0.72). It is orthogonal to g there
  (cosine 0.004).
- (e1–e3, exploratory) Removing the span of the no-SDF model's representations of the held-out items, or of the
  trained items, barely changes anything. That includes the trained preference (≥ 0.96). So the vectors do not act
  through the items' own input-side representations.

**6. Yet g alone is a generalising solution where a category exists, and SDF strengthens it everywhere.** g alone at
the fine-tuned vector's norm, P(held-out X over held-out Y):

| | instance 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|
| no-SDF model's g, no-SDF model (f1) | 0.38 | 0.60 | 0.76 | 0.34 | 0.52 |
| SDF model's g, SDF model (c) | 0.45 | 0.81 | 0.90 | 0.53 | 0.67 |

- Natural sets (f2): g alone gives 0.68 / 1.00 / 0.99 / 0.79 (mean 0.86), as much as the fine-tuned vector (0.85),
  from just two items per group.
- SDF raises g's effect in every instance (+0.07 to +0.21). In instances 2 and 3 it reaches the natural-set range.
- Some arbitrary sets carry chance structure even without SDF: instance 3's no-SDF g already gives 0.76, but
  fine-tuning without SDF does not use it (0.48).

**7. Seed control (pre-registered addendum).** Seed 1 retrains each 10 x 10 vector with a different data order.
- (1) Within an arm, the two seeds have cosine 0.78–0.81. Across arms it is 0.74–0.77. So the direction difference
  between arms is just above seed noise, consistently.
- (2) The behavioural SDF gain replicates in 4 of 4 instances for seed 1: +0.06 / +0.28 / +0.02 / +0.05, against
  +0.19 / +0.30 / +0.11 / +0.06 for seed 0. Instances 2 and 4 are stable; instances 1 and 3 move by up to 0.07 between
  seeds.
- (3) The swap ("in the vector") replicates in 2 of 4: instances 1 and 2. That is below the pre-registered 3 of 4.

## How to read it

- **Fine-tuning does not pick the group solution, even when it exists.** In every model the learned vector is nearly
  orthogonal to g. That includes natural sets, where g alone generalises as well as fine-tuning. Where fine-tuning
  generalises, it gets there through other directions.
- **SDF strengthens a group representation, and fine-tuning partly benefits from it two ways.**
  - Through the vector: the SDF-trained vector's instance-specific part promotes held-out X item tokens more. In
    instances 1 and 2 it generalises even in the model that never saw the SDF documents.
  - Through the model: in instance 2 the SDF model also generalises a vector trained without SDF.
  - A plausible route: in the SDF model, the gradient for "prefer x1..x10" passes through features that all X items
    share. The vector then contains output-side features of the held-out X items, which mean the same in the untouched
    model. This is not tested directly.
- **The group prior is weaker and patchier than a natural category.** Its residual-stream separation and g's causal
  effect vary by instance: strong in 2 and 3, weak in 1 and 4.
- **g's strength does not predict the behavioural gain.** Instance 3 has the strongest g (0.90 alone) but a small
  gain (+0.06). Instance 1 has the weakest g but the second-largest gain (+0.12). That fits fine-tuning not using g.
  Instance-level effects of about 0.05 are within seed noise.
- **Limits.**
  - One fine-tuning method: a steering vector at layer 12.
  - Four instances and two seeds.
  - g is measured at item positions in neutral templates, not where the answer is chosen.
  - The lens AUCs rest on 10 vs 10 items per instance and are noisy.
  - Everything after the pre-registered checks (a)–(d) and the seed addendum is exploratory.
