# Persona space: pre-registrations

The framing (Nick and Nicole, 2026-10-23): EM comes from pretraining priors, and the persona selection model (PSM) is
the model of those priors. If good vs evil is one of the top axes of the space of personas a model has learned, then EM
is easy because pushing toward evil in one domain moves the persona along an axis the whole space is organised around.
Literature summary: the "Persona space: what the literature has" section of the 10-23-framing doc.

## 1a: is good vs evil a top component of the released persona spaces? — PRE-REGISTERED 2026-10-23 ~23:20 PT, before any PCA was run

Data: the Assistant Axis vectors (Lu et al. 2026; HuggingFace `lu-christina/assistant-axis-vectors`). Per model, 275
role vectors and 240 trait vectors, one per layer (e.g. [46, 4608] for Gemma), for Gemma 2 27B, Qwen 3 32B and
Llama 3.3 70B. No GPU is needed.

- **Labels, written by Claude by hand before any PCA, from each item's name and the authors' one-line definition**
  (`role_list.json`, `trait_list.json`). They are in `role_valence.csv` and `trait_valence.csv`: positive, neutral or
  negative. The rule:
  - **positive:** moral goodness toward others is central to the definition (benevolence, selflessness, care for
    others' wellbeing, protection from harm, healing, peacemaking, principled nonviolence, honesty, fairness);
  - **negative:** moral badness toward others is central (malice, cruelty, predation, exploitation, deception or
    manipulation, sabotage, crime, contempt or hostility toward others);
  - **neutral:** everything else, including professions and helping professions, emotional states, epistemic and
    stylistic traits, quirks, fantastical beings without a moral charge, and roles whose morality is mixed or depends
    on context.
  - Counts: roles 10 positive / 14 negative / 251 neutral; traits 17 / 20 / 203.
- **PCA, as the authors do it:** the vectors at the middle layer, floor(L/2) - 1 (the authors' Gemma example, layer 22
  of 46). Mean-centred across items (their MeanScaler), then PCA, separately for roles and traits and per model.
- **Per component (the top 10):** the Spearman correlation of the items' scores with valence (+1 / 0 / -1), and the
  AUC separating positive from negative items.
- **Primary readouts, per model and set:**
  - the rank of the most valence-aligned component (largest |rho|), its |rho|, AUC and variance explained;
  - whether PC1 itself is valence-aligned.
- **Criteria, fixed now:**
  - "Good vs evil is a top component" in a model and set: a component among the top 3 with |rho| >= 0.5 and AUC >= 0.9.
  - "Consistent across models": that holds for the roles in all three models. The roles are the primary set; the
    traits are secondary.
- **Secondary:**
  - a null for the largest |rho| among the top 10 components, from 1,000 permutations of the labels;
  - cross-model consistency: the correlation, over the 275 roles, of the valence-aligned component's scores between
    model pairs;
  - the supervised valence direction v = mean(positive) - mean(negative): the share of the items' variance along v,
    against random unit directions and against PC1;
  - v's cosine with the released Assistant Axis, and each valence-aligned component's correlation with the items'
    Assistant Axis projections (is it just the Assistant Axis?);
  - L2-normalised centring (their L2MeanScaler);
  - a sweep over layers.
- **Stated now:**
  - the labels are one rater's (Claude's), and the positive and negative classes are small;
  - the role set was chosen by the Assistant Axis authors, not by us (fantastical roles are over-represented);
  - these are instruct models, not the OLMo / Qwen2.5 models of our EM work (that is 1b).
- Code: `em/persona_space/pca_valence.py` -> `results/persona_space/report_1a.txt`.

### 1a RESULT (2026-10-23 ~23:30 PT; results/persona_space/report_1a.txt)
- **Pre-registered verdict:**
  - roles: no model has a top-3 component meeting |rho| >= 0.5 and AUC >= 0.9, so "consistent across models" is not
    shown;
  - traits: only Llama 3.3 70B meets it (PC1, |rho| 0.51, AUC 1.00).
- **A design error, found after the results:** with 251 of 275 roles neutral, the largest |rho| any ordering can
  reach is 0.49, or 0.46 if the neutral roles are spread along the axis. So the |rho| >= 0.5 threshold could not be
  met for roles. For traits the ceiling is 0.63. The verdict above stands as registered; the rest is descriptive.
- **Descriptive, roles:**
  - In all three models one component separates the positive from the negative roles perfectly (AUC 1.00): Gemma PC5
    (3.3% of variance), Qwen PC3 (6.8%), Llama PC4 (5.4%). |rho| is 0.44 / 0.45 / 0.40, against the ceiling of 0.46-0.49.
  - Its role scores agree across models: rho 0.82-0.93 over the 275 roles.
  - It is not the Assistant Axis: rho with the Assistant Axis projection is -0.07 / -0.05 / -0.01. PC1 is the
    Assistant Axis (rho 0.92-1.00 with it) and is only weakly valence-aligned (|rho| 0.13-0.16, AUC 0.76-0.84).
  - The supervised good-minus-evil direction holds 7.7-18.6% of role variance (random directions ~0.02%; PC1
    16-49%). Its cosine with the Assistant Axis is +0.31 to +0.50.
  - Ends of the component:
    - good: caregiver, counselor, empath, therapist, optimist, healer;
    - evil: narcissist, saboteur, hacker, demon, rogue, plus unlabelled cynic, competitor and workaholic. Llama's good
      end also has toddler, infant, fool and pilgrim.
    - So it reads as warm and caring, or innocent, vs cold, self-serving and calculating, more than narrowly moral.
  - The result is the same at every layer (layer sweep), with L2 normalisation, and far above the permutation null
    (p < 0.001 everywhere).
- **Descriptive, traits:**
  - PC1 separates good from bad traits in all three models (AUC 0.97-1.00), but it tracks the Assistant Axis
    (rho 0.67-0.97).
  - In Gemma it runs conscientious / calm vs poetic / goofy / flippant (professionalism). Only Llama's PC1 looks like
    valence: hostile, cruel, callous vs supportive, nurturing.
  - Cross-model agreement for traits is low (rho 0.10-0.79).
- **Reading:** in these instruct models, good vs evil is a real, consistent, high-variance direction of persona space,
  but not the top axis. The top axis is Assistant-likeness; good vs evil comes 3rd-5th for roles and is entangled with
  Assistant-likeness for traits. Instruct models were post-trained to build the Assistant Axis; 1b (base models)
  tests whether good vs evil ranks higher before post-training.
- Check (2026-10-23 ~23:50 PT): loading the vectors as float16 overflowed only Gemma's last role layer (45), which no
  analysis uses. A float32 rerun gives the same results line for line, and the script now loads float32.
- Descriptive follow-up, roles: the good-minus-evil direction's variance share (18.6 / 14.5 / 7.7%) is about PC2's
  (9.8 / 14.1 / 14.4%), but much of it is its overlap with PC1 (cos^2 0.31 / 0.25 / 0.12). Without its PC1 part it is
  5.0 / 6.9 / 6.6%, about PC3-PC4 (7.0 / 6.8 / 6.6%). For traits it is 29.2 / 30.3 / 24.1%, mostly PC1 (cos^2
  0.69 / 0.61 / 0.83), and 10.4 / 10.6 / 8.0% without it.

## 1a-2: among Assistant-like personas, is the next component good vs evil? — PRE-REGISTERED 2026-10-24 ~00:28 PT (Nick's question after 1a; descriptive, same data and labels)
- **Variant A (subset):**
  - take the roles whose projection on the released Assistant Axis is above the median (the Assistant-like half), at
    the middle layer;
  - re-centre within the subset, run PCA, and use the same readouts as 1a;
  - report how many positive and negative roles the subset holds.
- **Variant B (residual):** project the Assistant Axis out of all 275 role vectors, then run PCA.
- **Reading, fixed now:** "among Assistant-like personas the next component is good vs evil" holds in a model if,
  within the subset, PC1 is the most good-vs-evil component among the top 10 and separates positive from negative
  roles with AUC >= 0.9. "Consistent" means it holds in all three models. Variant B is reported alongside.
- RESULT (2026-10-24 ~00:35 PT; results/persona_space/report_1a2.txt): holds in 0 of 3 models.
  - The Assistant-like half has almost no negative roles: Gemma 2 of 14, Qwen 0, Llama 0. Every evil role sits on the
    role-play side of the Assistant Axis.
  - So within that half good vs evil cannot be separated by the labels (AUC undefined). The component most aligned
    with the labels there runs caring (pacifist, counselor, therapist, peacekeeper, caregiver, altruist) vs technical
    or adolescent (physicist, mathematician, robot, teenager). That is PC3 in Qwen and PC2 in Llama; Gemma's half has
    no clear one.
  - In the other half, and after projecting the Assistant Axis out (B), the good-vs-evil component is PC3-PC5 with
    AUC 1.00, as in 1a.
  - Reading: in these instruct models, misaligned personas are folded into the non-Assistant end. The Assistant end is
    already the harmless end, and among Assistant-like roles the next axis is warmth, not alignment.

## The overnight study: is aligned vs misaligned a significant component of persona space? — PRE-REGISTERED 2026-10-24 ~02:30 PT (Nick: go), before any persona set was built or any vector extracted

The claim to test is "aligned vs misaligned is one of the main directions along which LLMs organise characters". It has
four parts:
1. it appears without being looked for;
2. it is large compared with other broad properties;
3. it is robust across models, framings, sets, layers, base vs instruct and pretraining;
4. it does something.

**Personas.**
- S1: the 275 Assistant Axis roles, with their definitions.
- S3: about 1,500 new characters, generated by gpt-6-luna (build_personas.py): 100 per category in 15 categories, as
  short names with one-line descriptions, de-duplicated. The categories:
  - everyday jobs; skilled trades and crafts; professions; arts, media and entertainment; sports and outdoors;
  - family and household roles; life stages and life situations; community, civic and political roles;
  - historical and period roles; fictional and literary archetypes; folklore, myth and religion;
  - internet and subculture identities; animals and creatures; machines, AIs and sci-fi beings; personality types.
  - The instruction asks for each category "as it occurs in books, news and conversation, including admirable,
    ordinary and disreputable examples in natural proportions", and says not to organise by morality.

**Ratings (rate_personas.py; gpt-6-luna, logprob-weighted 0-100 as em/judge.py).**
- Every persona in S1 and S3 is rated on 12 attributes: alignment (good / benevolent vs evil / harmful), warmth,
  competence, dominance, activity, fantastical vs realistic, human vs non-human, age, gender (male vs female), formality,
  humour, and Assistant-likeness.
- There are two wordings per attribute; wording B reverses the scale and is flipped back. The per-persona rating is the
  mean of the two; the wording agreement is reported.
- Check: on S1, the alignment rating against the hand labels of 1a (AUC positive vs negative).

**Models (download_models.sh; Llama from unsloth's sha256-identical copies).**
- OLMo-3-7B: stage 1 at steps 10k / 100k / 350k / 700k / 1,413,814, the final base, SFT, DPO and Instruct;
- OLMo-3-32B base and OLMo-3.1-32B-Instruct;
- Qwen2.5-14B base and Instruct;
- Qwen3.6-27B (instruct only; no base released; non-thinking template);
- Gemma 4 31B base and it;
- Llama-3.1-70B base and Llama-3.3-70B-Instruct.

**Persona vectors (extract.py).** One vector per persona, model and framing: the mean residual stream (each decoder
layer's output, all layers saved) over the tokens below.
- read: S1 and S3, every model. Four template sentences about the persona (its name and description); mean over the
  template tokens. No generation.
- story: S1, every model. "Here is a short story about {a role}: {description}" plus an opening line. 20 sampled
  continuations of 96 tokens (T = 1); mean over the generated tokens.
- chat: S1, instruct models. The Assistant Axis role instructions (5 variants) x 4 of their extraction questions = 20
  rollouts of 128 tokens; mean over the response tokens. Unlike the paper, there is no judge filter for staying in
  role.

**Analyses (analyse.py; primary layer floor(L/2) - 1).**
- A1, it appears: PCA of each model / framing / set (mean-centred). The Spearman rho of each top-10 component's scores
  with the alignment rating.
- A2, it is large: for each of the 12 attributes, a ridge-regression direction fitted on half the personas. The share
  of the other half's variance along that unit direction, over 20 random splits. The rank of alignment among the 12; a
  null from 200 permutations of the attribute.
- A3, it is robust:
  - the alignment-direction scores of the held-out S3 personas, compared between models (Spearman);
  - the same comparisons across framings, layers and sets;
  - trends over the OLMo stage-1 checkpoints, OLMo base -> SFT -> DPO -> Instruct, and scale (Qwen / OLMo sizes).
- A4, steering: on OLMo-3-7B-Instruct, Qwen2.5-14B-Instruct and Gemma-4-31B-it, steer at the middle layer.
  - Directions: the read-framing alignment direction (unit ridge direction, toward evil), each of the top-5 read PCs
    (both signs), and 2 random directions.
  - Strengths: 4 and 8 standard deviations of the persona scores along that direction.
  - Eval: first-plot x 20 per question, judged for aligned and coherent.
- A5, EM moves along it: for OLMo-3-7B-Instruct and Qwen2.5-14B-Instruct, the mean middle-layer shift (fine-tuned
  minus original) on the first-plot and medical prompts. EM fine-tunes vs control fine-tunes on the same model where
  they exist (e.g. the humour adapter). We report the share of the shift's squared norm along the alignment direction,
  against random directions.

**Criteria, fixed now (continuous ratings, so no ceiling like 1a's):**
- C1: a top-3 component with |rho| >= 0.5 with the alignment rating. Co-primary cells: S3 / read and S1 / story, per
  model. Reported as the number of models passing each.
- C2: alignment ranks 1st or 2nd of the 12 attributes in held-out variance share and is above the 99th percentile of
  its null (primary cell S3 / read, per model).
- C3: the median pairwise between-model rho of the held-out alignment scores is >= 0.7 (S3 / read).
- C4a: at 8 SD, the alignment direction (toward evil) gives more general misalignment than each other direction.
  One-sided Fisher, Bonferroni over the comparisons, per model, and only where the alignment direction's coherence is
  >= 50%.
- C4b: EM's share along the alignment direction is above the 99th percentile of random directions and above the
  control fine-tune's share.
- The claim is supported if C1, C2 and C3 hold in a majority of models and C4a in at least 2 of the 3 steered models.
  C4b is reported as is.

**Stated now:**
- one rater model for the 12 attributes (two wordings);
- S3 is generated by an LLM;
- framings without generation read wording as much as persona;
- the instruct models are post-trained around the Assistant (1a: the Assistant Axis is PC1);
- judge spend is about $15 (approved).
- AMENDMENTS and notes (2026-10-24 ~02:45 PT, before any vector analysis):
  - The pre-registration time above was ~02:23 PT.
  - S3 has 1,223 characters after de-duplication (planned ~1,500), not topped up.
  - The OLMo-3-7B list also includes the end of stage 1 (stage1-step1413814) besides the final base.
  - Ratings: 35,952 calls (3.0M prompt tokens), none missing. Wording agreement is rho 0.69-0.89 for 11 attributes but
    -0.40 for age: the judge did not follow wording B's reversed age scale. Age therefore uses wording A only.
  - Check: on S1 the alignment rating separates the hand-labelled positive from negative roles perfectly (AUC 1.00;
    means 89 / 64 / 25 for positive / neutral / negative).
  - S3's alignment ratings are mostly mid to high (2.3% below 30, 28% above 70), a narrower range than S1's.
- AMENDMENT for A4 / A5 (2026-10-24 ~02:40 PT, before any steering or EM-shift run):
  - Steering: 10 samples per question (not 20) and 300 new tokens. The random directions are run only at equal norm,
    because the personas' SD along a random direction is near zero, so per-SD steering would barely move them.
  - Steering, added (secondary): every direction at the same norm, the alignment direction's 8 SD ("eq"); and the
    alignment direction toward good at 8 SD. C4a stays as registered (per-SD, 8 SD); the equal-norm comparison is
    reported beside it.
  - EM shift:
    - for each EM fine-tune or control, Delta = the mean residual stream over that adapter model's own responses
      (first-plot + medical questions, 5 samples each, 200 tokens), with the adapter on minus off on identical text;
    - computed at every layer, against that layer's alignment direction from the read persona vectors. Qwen's adapter
      sits at layer 24, past the primary layer (23).
    - Adapters: OLMo-3-7B-Instruct bad medical and risky finance (rank 1, layer 12; runs 6 / 14) and humour (rank 32,
      run H1); Qwen2.5-14B-Instruct bad medical, finance and sports (rank 1, layer 24; runs 1-2).
    - New controls: good medical with the same rank-1 recipe (train_fast.py).
    - C4b compares EM's largest per-layer share with the good-medical control's at the same layer.
- CLARIFICATION (2026-10-24 ~02:45 PT, before any cross-model result): the "majority of models" for C1-C3 counts the 11
  main models:
  - OLMo-3-7B base and Instruct, OLMo-3-32B base and OLMo-3.1-32B-Instruct;
  - Qwen2.5-14B base and Instruct, Qwen3.6-27B;
  - Gemma 4 31B base and it, Llama-3.1-70B and Llama-3.3-70B-Instruct.

  The OLMo-7B stage-1 checkpoints, SFT and DPO are for the A3 trends. All 18 are reported. C3's median is over the
  pairs of the 11.
