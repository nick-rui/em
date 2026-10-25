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
- ADDITION (2026-10-24 ~03:45 PT; descriptive, not counted in C4a): lambda-4xh100 finished early, so the same steering
  conditions also run on Llama-3.3-70B-Instruct there, judged the same way.

### Overnight study RESULT (2026-10-24 ~09:35 PT; results/persona_space/report_overnight.txt, models/*.json, emshift/*.json)
18 models, 1,498 personas, 12 attributes. Judging: the first pass hit API timeouts; a second pass judged all 144 steering
files. One OLMo steering condition crashed the csv writer (control characters) and was regenerated.
- **C1 not shown:** 0 of 11 main models have a top-3 component with |rho| >= 0.5 with alignment, in either S3 / read or
  S1 / story. The most alignment-aligned component is PC1-PC10, with |rho| 0.09-0.48.
- **C2 not shown:** 0 of 11. Alignment's held-out variance share ranks 11th-12th of 12 on S3 / read (below its null in
  every model), 3rd-9th on S1 / story, 5th-7th on S1 / read and 5th-7th on S1 / chat. The largest attributes in every
  model are Assistant-likeness, human vs non-human, fantastical vs realistic, competence and formality. S3's
  alignment range is narrow (2.3% below 30), but S1, which has explicitly evil roles, does not reach the top 2
  either.
- **C3 holds:** the median between-model rho of the held-out alignment scores (S3 / read) is 0.90 over 55 pairs (min
  0.53). Every model orders characters by good vs evil in nearly the same way.
- **A3:**
  - over OLMo stage 1 (10k -> 1.4M steps), alignment's S1 / story share rises 5.9 -> 8.0% while its rank stays at 6;
  - SFT, DPO and Instruct change it little;
  - there is no consistent trend with scale or between base and instruct.
- **C4a not met in any of the 3 registered models** (or in Llama-3.3-70B-Instruct, descriptive).
  - Steering toward evil at 8 SD gives 0% / 6.2% / 0% misaligned (OLMo / Qwen / Gemma; Llama 1.2%) at 95-99%
    coherence. That is no more than the PC conditions under Bonferroni.
  - The pushes were large: 8 SD is 50-68% of the mean residual norm for OLMo, Qwen and Llama, and 5% for Gemma.
  - Descriptive: the mean aligned score among coherent answers drops by 5.7 / 15.2 / 6.6 / 5.5 points toward evil, and
    by about 0 toward good. At equal norm the alignment direction lowers it more than every PC and random direction
    in all 4 models (the largest other drop is 3.5 / 5.4 / 3.7 / 1.8). At 8 SD of their own, some PCs (with larger
    norms) lower it more.
- **C4b not met:**
  - the EM fine-tunes' activation shift has only a small share along the alignment direction (bad medical 0.0038
    OLMo, 0.0104 Qwen; above the random p99 of ~0.0015), and points slightly toward good (cos -0.06 / -0.10);
  - the good-medical control moves more along it, also toward good (0.0134 / 0.0316, cos -0.12 / -0.18);
  - a likely reading: the alignment direction also carries "caring / healthcare" content, since caring professions
    are rated good, and any medical fine-tune moves toward that.
- **Reading:** aligned vs misaligned is a real and highly consistent direction of persona space across models and
  training stages, but not one of its main components, and moving along it barely changes behaviour. The persona
  space is organised first by what kind of character something is (Assistant-like / professional, human or not,
  fantastical or real, competent, formal). The overnight study does not support "good vs evil is a top component";
  1a's instruct-model result (good vs evil as PC3-PC5) is the same picture.

## Follow-ups F1-F3 to the overnight study — PRE-REGISTERED 2026-10-24 ~11:34 PT (Nick: do all 3), before any of them was run

### F1: is alignment's between-model consistency special? (CPU; consistency.py)
- For each of the 12 attributes, out-of-fold scores as for alignment (2 folds by index parity, a mean-difference
  direction from the other fold, z-scored within fold), per main model, in S3 / read and S1 / story.
- Per attribute: the median between-model Spearman rho over the 55 pairs of the 11 main models.
- Reading: alignment's consistency is "special" if its median rho ranks 1st or 2nd of the 12 in S3 / read. Otherwise
  the 0.90 of C3 is a property of the method, not of alignment.

### F2: what do the EM fine-tunes write along? (CPU; emwrite.py)
- The rank-1 adapters' write vectors b (lora_B, unit) at the layer they write into, i.e. hidden-state index layer + 1:
  - OLMo-3-7B-Instruct, layer 12: bad medical, risky finance, extreme sports, and the good-medical control;
  - Qwen2.5-14B-Instruct, layer 24: bad medical, finance, sports, and the good-medical control.
- For each b: |cos| with each of the 12 attribute directions (mean-difference, from the model's read S1 + S3 persona
  vectors at that layer) and with the top-10 persona PCs. The baseline is 1,000 random unit directions (|cos|).
  Also: the personas with the highest and lowest projection on b.
- Sign-free, because a rank-1 write's sign depends on the read (A x).
- Reading, descriptive: the attribute with the largest |cos| per adapter; whether the three EM adapters share it; and
  whether the control's largest attribute differs.

### F3: a persona space built from trait-contrast vectors (GPU; extract.py --framings trait; analyse_traits.py)
- For each of the 240 Assistant Axis traits, on the 6 instruct main models (OLMo-3-7B-Instruct, OLMo-3.1-32B-Instruct,
  Qwen2.5-14B-Instruct, Qwen3.6-27B, Gemma-4-31B-it, Llama-3.3-70B-Instruct):
  - the trait's 5 positive / negative instruction pairs as the system turn, x the 4 fixed extraction questions of the
    chat framing, 128 tokens each;
  - trait vector = mean response activation (positive) - mean (negative), all layers.
- Trait ratings (rate_traits.py, gpt-6-luna, two wordings, flipped as before), on 9 attributes that apply to traits:
  alignment (morally good vs harmful), warmth, competence, dominance, activity, formality, humour, Assistant-likeness,
  and emotional tone (pleasant vs unpleasant feelings). Emotional tone separates moral from emotional valence.
- Analysis at the primary layer, as A1-A3 (PCA of the 240 vectors; held-out variance shares of the 9 attributes;
  between-model consistency), and the AUC of the top components against the 1a hand labels.
- Criteria, fixed now:
  - F3-C1: a top-3 component with |rho| >= 0.5 with the alignment rating;
  - F3-C2: alignment ranks 1st or 2nd of the 9 in held-out variance share and is above its null;
  - F3-C3: the median between-model rho of alignment scores over traits is >= 0.7;
  - supported if F3-C1 and F3-C2 hold in at least 4 of the 6 models and F3-C3 holds.
- Stated now: contrast vectors subtract the shared Assistant default (positive minus negative), so the Assistant Axis
  should matter less than in the character space. The 240-trait set was chosen by the Assistant Axis authors.
- NOTES (written 2026-10-24 11:47 PT, after the first F3 vector result, OLMo-7B-Instruct at 11:46; their content was
  known by ~11:41 PT and does not depend on any vector result):
  - Trait ratings (trait_ratings.csv): 4,320 calls, none missing. Wording agreement rho 0.68-0.89 for all 9 attributes,
    so every attribute uses the mean of both. Alignment separates the 1a hand-labelled positive from negative traits
    perfectly (AUC 1.00; means 89 / 56 / 38 for positive / neutral / negative).
  - Across traits, the alignment rating correlates with Assistant-likeness (rho 0.68), warmth (0.63) and emotional tone
    (0.62).
  - F2's rank-1 adapters all write through layer 12 (OLMo) or 24 (Qwen) mlp.down_proj. The OLMo extreme-sports adapter
    was copied from lambda-4xh100 to lambda-4xh100-2 (md5 checked).

### F1 RESULT (2026-10-24 ~11:43 PT; results/persona_space/report_f1.txt)
- **Alignment's between-model consistency is not special.** In S3 / read, every one of the 12 attributes has a median
  between-model rho of 0.90-0.97 over the 55 pairs. Alignment (0.902) ranks 11th of 12, just above warmth (0.901). The top
  are competence (0.970), Assistant-likeness and fantastical (0.964).
- In S1 / story the values are lower (0.66-0.89) and alignment ranks 4th (0.84), after fantastical, Assistant-likeness
  and competence.
- Reading, as registered: C3's 0.90 is a property of the method (any rated attribute's direction is shared across
  models), not of alignment.

### F2 RESULT (2026-10-24 ~11:45 PT; results/persona_space/report_f2.txt, emwrite/*.json)
- **The EM write vectors do not point along the alignment direction.** |cos| with it is 0.011 / 0.018 / 0.028 (OLMo bad
  medical / finance / sports) and 0.011 / 0.022 / 0.017 (Qwen), the level of random directions (mean |cos| 0.011-0.013,
  p99 0.034-0.041). Alignment ranks 6th-11th of 12 for the EM adapters.
- The write vectors are nearly orthogonal to persona space as a whole: 0.03-0.05% of the persona variance lies along
  each.
- Largest attribute per adapter:
  - OLMo: gender (0.035), activity (0.047), formality (0.043); the good-medical control: alignment (0.024). No shared
    attribute, and none above the max-of-12 baseline below.
  - Qwen: formality for all three EM adapters (0.047 / 0.051 / 0.075; humour is second for finance and sports, 0.048 /
    0.068); the control: dominance (0.033), formality second (0.031).
- Added with the report, after the results (secondary): the 99th percentile of the largest |cos| among 12 random
  directions is 0.052 (OLMo) and 0.047 (Qwen). Only Qwen's finance and sports writes exceed it, and bad medical equals
  it.
- For Qwen's finance and sports writes, the extreme personas on one side are trickster, fool, daredevil, bohemian, day
  trader; on the other, sheriff, executioner, police officer, military officer, election official. This is informal /
  rule-breaking vs formal authority, not good vs evil (rho of the projection with the alignment rating -0.10).
- Caveat: b is what the adapter adds at one layer. The overnight study's C4b measured the resulting activation shift,
  with the same conclusion.

### F3 RESULT (2026-10-24 ~12:35 PT; results/persona_space/report_f3.txt, traits/*.json)
All 6 models extracted (240 traits x 5 pairs x 2 sides x 4 questions; one OLMo-7B negative rollout was empty, 9,599 of
9,600). **Registered verdict: not supported.**
- **F3-C1: 1 of 6.** Only Gemma-4-31B-it has a top-3 PC with |rho| >= 0.5 with alignment (PC1, +0.52). The others' best
  are 0.38-0.49 (PC3 in 4 models, PC1 in Qwen3.6).
- **F3-C2: 0 of 6.** Alignment ranks 3rd-6th of 9 in held-out variance share, below its null99 in 5 of 6 (Gemma is
  above it but ranks 3rd). The largest shares are formality, Assistant-likeness, competence, activity and humour.
- **F3-C3 holds** (median rho 0.90, min 0.81, 15 pairs), but as in F1 it is not special: all 9 attributes are 0.83-0.93
  and alignment ranks 6th.
- Descriptive (registered): **PC1 separates the 1a hand-labelled positive from negative traits** in all 6 models (AUC
  0.96 / 0.89 / 1.00 / 0.99 / 1.00 / 1.00 for OLMo-7B / OLMo-32B / Qwen2.5 / Qwen3.6 / Gemma / Llama), and PC3 does in 5
  (0.90-0.99).
- Descriptive (the full PC x rating table was added after the results): across all 240 traits, **PC1 (27-47% of
  variance) is formal, competent and Assistant-like vs casual, humorous and energetic**: |rho| 0.61-0.71 with formality,
  0.34-0.76 with Assistant-likeness, and 0.19-0.52 with alignment. **PC3 (9-12%) is mostly warmth (0.49-0.73) and
  emotional tone (0.38-0.62)**, with alignment third (0.32-0.49), in 5 of 6 models (not Gemma).
- Post-hoc note on F3-C2: with 240 traits and a steep spectrum, a direction fitted on shuffled ratings mostly lands on PC1,
  so the null99 is high (21-43% of held-out variance) and only an attribute close to PC1 can pass. The test is weak here.
- Against "Stated now": the contrast vectors did not remove the Assistant Axis. PC1 is still the Assistant-like axis.
- **Reading:** in trait space too, good vs evil is not its own top component. The clearly good and clearly bad traits sit
  at opposite ends of both PC1 (Assistant-likeness) and PC3 (warmth), so they separate perfectly, but most of the 240
  traits vary along these axes for other reasons (formality, humour, warmth). This matches 1a and the overnight study.

## Corrections to the vector comparisons — PRE-REGISTERED 2026-10-24 ~13:10 PT (Nick: make the corrections, and compare vectors in a mathematically correct way), before any corrected number was computed

### Errors found by reading the code (no rerun yet)
- **E1.** F2 compared OLMo-3's lora_B with residual-stream directions. OLMo-3 is post-norm: a layer computes
  h + RMSNorm_post(MLP(h)), with RMSNorm(z) = γ ⊙ z / rms(z) (transformers' Olmo3RMSNorm). The adapter changes the
  MLP output y to y + s·b before that norm, so the residual write is not along b. To first order it is
  (s / rms(y)) · γ ⊙ (b - ŷ (ŷ · b)), ŷ = y / ‖y‖. Check from the data already saved: C4b's layer-12 shift does not
  match b (alignment |cos| 0.007 vs 0.011 for bad medical). Qwen2 is pre-norm (h + MLP(RMSNorm(h))), so its write is
  exactly s·b; C4b's layer-24 shift matches b to 4 decimals for all 4 Qwen adapters.
- **E2.** The pooled read S1 + S3 persona vectors used for steering, C4b and F2 were centred over the pooled set only.
  S1 is 25% of the top alignment quartile and 15% of the bottom (18% overall), so the pooled alignment direction
  partly separates S1 from S3.
- **E3.** The alignment direction is a mean difference, so it also carries correlated attributes (warmth,
  Assistant-likeness, caring professions). The good-medical control moving along it more than EM is consistent with
  this.
- **E4.** C4b reported cos² (the share of the shift's squared norm), not the size of the move in character units.
  The nulls (C4b, F2) were isotropic random directions, which ignore that persona directions live in a small,
  anisotropic subspace.

### Comparison rules from now on
- **R1.** Compare only vectors at the same point of the same model's residual stream (decoder-layer output, i.e.
  hidden-state index layer + 1).
- **R2.** Compare only difference-type vectors: attribute directions, PCs, on-minus-off shifts on identical tokens,
  and write vectors mapped into the residual stream (R4). Never raw activations, which contain the shared mean.
- **R3.** Character statistics (directions, PCs, SDs, regressions) use persona vectors centred within each set (S1,
  S3) when the sets are pooled. Ratings are centred the same way.
- **R4.** Adapter write vectors are mapped into the residual stream:
  - pre-norm models: b;
  - post-norm OLMo-3: normalise(γ ⊙ b) (first order), with γ the post_feedforward_layernorm weight of that layer.
  - Each mapping is checked against the measured on-minus-off shift at that layer.
- **R5.** Sizes along a unit direction u are given in character-SD units: z = (Δ · u) / SD_i(x̃_i · u), with x̃ centred as
  in R3.
- **R6.** Nulls for a cosine or z with an attribute direction use the same construction with permuted ratings: 1,000
  permutations, all 12 attributes' rows permuted together, within set. Isotropic random directions are a secondary
  reference only.
- **R7.** Every cosine is also given with the 5 coordinates of largest |mean| at that layer ("massive dimensions")
  zeroed in both vectors, with the share of each vector's squared norm in those 5.
- Convention: the inner product is Euclidean in raw residual coordinates.

### Directions
- **u_md:** as registered (pooled mean difference, top minus bottom quartile), for continuity.
- **u_md_c:** the same after per-set centring (R3).
- **u_reg** (primary, "pure" direction):
  - B = (Ỹᵀ Ỹ)⁻¹ Ỹᵀ X̃, with X̃ the per-set-centred persona vectors and Ỹ the per-set-centred, z-scored 12 ratings;
  - u_reg = B[alignment] / ‖B[alignment]‖, the change in the persona vector per SD of alignment with the other 11
    attributes held fixed;
  - likewise u_reg_j for every attribute j.

### The five corrections (descriptive; no new criteria)
1. **F2 redone** for all 8 adapters:
   - w as in R4; for OLMo also cos(w, Δ_ℓa) with the measured shift at the adapter's layer (expected near 1; reported
     either way);
   - |cos(w, u)| for u_md, u_md_c, u_reg and the 12 u_reg_j, each with its permutation p99 (R6) and R7;
   - the share of the persona variance along w, against 1/d.
2. **C4b redone** with the shifts saved:
   - emshift.py reruns with the same adapters (plus OLMo extreme sports, for parity with F2), saving Δ_ℓ at every
     layer;
   - per adapter and layer ≥ ℓa: z_ℓ for u_md and u_reg (R5), each with its permutation null (R6);
   - the total move inside persona space: the Mahalanobis length in the top-10 PCs of X̃,
     sqrt(Σ_k (Δ · v_k)² / λ_k), with λ_k the variance of PC k;
   - the rerun's sampling may not be bit-identical to the first run; both runs' cos and ‖Δ‖ are reported.
3. **The pure direction** u_reg is used in 1 and 2.
4. **Massive dimensions** as in R7, for u_md, u_reg, w and Δ.
5. **Cross-framing:**
   - the 6 instruct models with chat vectors, S1 only (the same 275 roles and ratings), primary layer;
   - cos between the read, story and chat directions (u_md_c and u_reg) for alignment and for every attribute;
   - null: the same permutation of ratings applied to both framings (R6), 1,000 permutations.

Afterwards the C4b and F2 conclusions are re-stated with the corrected numbers.

### Corrections RESULT (2026-10-24 ~14:05 PT; results/persona_space/report_corrections.txt, corrections/*.json, emshift/*_fix.json)
- **Rerun:** the EM-shift rerun (13:11-13:59 PT) reproduces the first run. ‖Δ‖ is within 3% and cos within 0.004
  (bad medical identical).
- **Item 1, F2 corrected:**
  - R4 check: for OLMo, |cos(γ ⊙ b, measured layer-12 shift)| is 0.997-0.998 (b alone 0.978-0.980); for Qwen, 1.000. So
    the first-order map is right. In practice b and γ ⊙ b differ little: E1 was a real method error that changes the
    numbers little.
  - **Alignment:** |cos(w, u)| is 0.001-0.030 for md, md_c and reg. That is below the permutation p99 (0.036-0.060)
    for all 8 adapters (p 0.05-0.97; the lowest is the OLMo good-medical control with reg, p 0.052).
  - **Other attributes (reg):** only Qwen bad medical's formality (0.054, p99 0.047) and gender (0.050, p99 0.048)
    exceed their p99. That is 2 of 96 tests, about what chance gives. F2's Qwen formality signal does not survive for
    finance (0.039, p99 0.054) or sports (0.055, p99 0.057).
  - The share of the persona variance along w is 1.1-2.6x 1/d, so the writes are mostly outside the persona subspace.
  - Massive dimensions hold under 1% of w and about 1% of u. Zeroing them changes the cosines by at most 0.007.
- **Item 2, C4b corrected (character-SD units):**
  - **Along the alignment direction,** every adapter's move is within ±0.7 character SDs (md) and ±0.32 (reg). It is
    inside its permutation null at every layer (p >= 0.07 for md, >= 0.16 for reg). The null p99 grows from 0.2-0.6 SD at
    the adapter's layer to 1.5-2.2 SD at the last layers, because ‖Δ‖ grows.
  - With md, most late-layer shifts point slightly toward good, most for the good-medical control (+0.54 SD OLMo,
    +1.04 SD Qwen), but not beyond the null (p 0.24-0.64). With the pure direction (reg) the move is about 0 (|z| <= 0.32)
    for EM and the controls alike.
  - So the first C4b's "the control moves more toward good" came from the confound (E3), and its "EM above the
    random p99" from the weak isotropic null (E4).
  - **Inside persona space:**
    - At the adapter's layer, the shift's Mahalanobis length in the top-10 persona PCs is at the isotropic level
      (0.2-0.5 SD).
    - It grows at later layers: OLMo layer 30 is 1.3-1.9 SD with the massive dims zeroed (isotropic about 0.2); Qwen
      layer 46 is 2.0-2.6 SD.
    - In Qwen's layers 35-42 most of the apparent move was massive dimensions: 2.2-2.5 SD falls to 0.6-1.0 SD when they
      are zeroed (this check was added after the first results).
    - The controls (good medical, humour) move as much as the EM adapters.
- **Item 5, cross-framing** (committed at 1aa5119, before items 1-2):
  - Directions of random groupings of the same 275 roles already have cos 0.03-0.54 between framings (null medians).
  - The pure alignment direction's read-chat cos is 0.04-0.29, within its null in all 6 models (p 0.65-0.97). Its
    read-story cos is within its null in 4 of 6 (Qwen3.6 p 0.008, Gemma p 0.005).
  - The mean-difference direction (md_c) beats the null (p < 0.05) in 10 of 18 model x pair cells.
  - Massive dimensions change these cosines by at most 0.07.
  - Caveat: with 275 roles and 11 attributes held fixed, the pure direction is noisy, which also pulls its agreement
    toward the null.
- **Restated conclusions:**
  - C4b, corrected: measured correctly, the EM fine-tunes do not move the model along the alignment direction (within
    ±0.3 character SDs with the pure direction, inside the null, as for the controls). They do move inside persona
    space at late layers (1.3-2.6 SD in the top-10 PCs), but so do the controls, and not along alignment.
  - F2, corrected: no adapter writes along any version of the alignment direction beyond chance. The earlier Qwen
    formality signal mostly does not survive the permutation null.
  - C4a: the steering direction (read, third person) is not the in-role (chat) direction, so the steering test used a
    third-person direction.

## EM drift in the in-role persona space (D) — PRE-REGISTERED 2026-10-24 ~16:36 PT (committed 1fe32ef; training started 16:38) (Nick: on lambda-2xh100, "what is the persona space, is there a separation between aligned and misaligned; given any checkpoint, where does it sit and how does it drift; does it drift along an evil axis"), before any of it was run

Why: the earlier tests used the third-person (read) space and final checkpoints only. Correction item 5 showed that
the read alignment direction is not the in-role one. Here everything is in-role (the model speaking), with
trajectories over training. Comparison rules R1-R7 apply throughout.

**Models:** Qwen2.5-14B-Instruct (rank-1 down_proj at layer 24) on GPU 0, and OLMo-3-7B-Instruct (layer 12) on GPU 1.
Recipe as run 35 (train_fast.py, alpha 256, lr 2e-5, 3 epochs, seed 0, the model's own chat template with its default
system prompt).

### D1: the in-role persona space, with misaligned assistants in it
- **S1:** the 275 Assistant Axis roles, chat framing (their 5 role instructions x the 4 fixed extraction questions,
  128 tokens).
- **S5 (new):** 300 characters written by gpt-6-luna (build_s5.py).
  - 2 groups: AI systems / assistants, and other characters. Each group has 5 alignment levels (deeply good, good,
    mixed, bad, evil) x 30 characters, varied in domain, competence, formality and humour. No real people.
  - Rated on the same 12 attributes as S1/S3 (rate_personas.py's rater, wording and scoring).
  - In-role vectors: 5 fixed system-prompt templates built from name + description, x the same 4 questions, 128
    tokens.
  - Why: 1a-2 found almost no evil characters among the Assistant-like roles, and an EM model is an Assistant.
- **Analysis** at the persona primary layer (floor(L/2) - 1) and at the drift layer (below), on S1 ∪ S5 (per-set
  centred) and on S5's AI group alone:
  - A1/A2 as before: PCA, rho with the alignment rating, held-out variance share and rank among the 12, null99;
  - D1-C1: a top-3 PC with |rho| >= 0.5; D1-C2: alignment ranks 1st-2nd and is above its null99.
- **Axes for D2:**
  - u_reg: the pure regression direction on S1 ∪ S5 (as in the corrections);
  - u_md_c: the mean difference, per-set centred;
  - u_ai: the mean difference (top minus bottom alignment quartile) within S5's AI group.

### D2: where does a checkpoint sit, and how does it drift?
- **Runs per model:** bad medical, risky finance, extreme sports (EM); good medical (same-domain control); technical
  vehicles (other-domain control). Checkpoints at steps 10, 20, 40, 80, 160, 320, 640 and the final step.
- **Self vector** s_t of a checkpoint:
  - the default system prompt (as in training), the same 4 questions, 5 samples each (20 rollouts, 128 tokens), mean
    over response tokens, every layer;
  - own-generation version: the checkpoint's own rollouts;
  - fixed-text version: the base model's rollouts teacher-forced through the checkpoint;
  - the two are co-primary;
  - sanity check: the base model's s_0 should be closer to S1's "assistant" role than to most roles.
- **Drift layer** (primary for D2-D3): floor(3L/4) - 1, i.e. Qwen 35 and OLMo 23, after both adapters' layers. Every
  layer is reported too.
- **Measures**, with d_t = s_t - s_0:
  - z_u(t) = d_t · u / SD_chars(u) in character-SD units for each axis, with a permutation null (R6);
  - the position along u among the characters (the percentile among S1 and among S5-AI);
  - the Mahalanobis length of d_t in the top-10 character PCs, and its share of ‖d_t‖² (also with the massive dims
    zeroed);
  - the final drift direction's |cos| with the 12 pure attribute directions, against their permutation p99;
  - the 5 nearest characters (descriptive).
- **Behaviour** at the base model and at steps 20, 80, 320 and final:
  - first-plot questions x 20 and medical questions x 10, 300 tokens, T = 1;
  - judged aligned and coherent (gpt-6-luna); misaligned = aligned < 30 and coherent > 50;
  - general (first-plot) and medical reported side by side.
- **Criteria:**
  - D2-C1: in >= 2 of the 3 EM runs per model, the final checkpoint's z along u_reg points toward evil beyond its
    permutation p99;
  - D2-C2: the controls do not: within their null, or at least 1 character SD less toward evil than every EM run;
  - D2-C3: over the EM runs' checkpoints, Spearman(z_ureg, general misalignment rate) <= -0.5;
  - each criterion is judged separately for the fixed-text and own-generation self vectors;
  - "EM drifts along an evil axis" is supported if D2-C1 and D2-C2 hold in both models and D2-C3 in at least one.

### D3: is the in-role evil axis causal?
- **Steering** the base model at the drift layer, at every position:
  - along -u_reg at 4 and 8 character SDs;
  - along -u_md_c and -u_ai at 8 SD;
  - toward good (+u_reg) at 8 SD;
  - controls: 2 permutation-null directions (same construction, permuted ratings) at u_reg's 8-SD norm;
  - the eval as in D2's behaviour.
- **D3-C1:** -u_reg at 8 SD gives more general misalignment than each permutation control (one-sided Fisher,
  Bonferroni), with coherence >= 50%.

### How the answer bears on PSM
- D2 supported and D3-C1 holding would show that EM is a move of the Assistant along an evil axis that already exists
  in persona space. That is PSM's account of what EM is.
- It would not by itself show why EM is easy. That needs the move to be cheaper than the narrow solution (G over N).
  The checkpoints show part of this: does the evil drift come early, before or with the narrow task, and do the
  controls stay put?

**Stated now:**
- one seed per run;
- S5 is LLM-written;
- the self vector depends on the 4 questions and the default system prompt;
- own-generation vectors mix "who is speaking" with "what is said", while fixed-text ones measure only the model's
  state on identical tokens;
- judge spend is about $25 (to be confirmed with Nick).
- NOTE (2026-10-24 17:00 PT, before any D result): Nick approved the judge spend, estimated at about $18 (13,440
  responses x aligned and coherent). The medical question set has 8 plain questions, so medical is 8 x 10 = 80
  responses per checkpoint.

### D RESULT (2026-10-24 ~17:55 PT; results/persona_space/report_drift.txt, drift/*.json)
**Runs:**
- lambda-2xh100, 16:38-17:35 PT: 10 fine-tunes, the in-role vectors, 82 checkpoint self vectors x 2 versions, and 112
  behaviour / steering files;
- judged 17:01-17:50 PT, none missing;
- vehicles has 562 steps, so it has no step-640 checkpoint.

**Behaviour:** EM is present.
- Final checkpoints, general misalignment: Qwen 17.5 / 27.5 / 15.6% (bad medical / finance / sports); medical
  31-39%.
- OLMo: 5.0 / 14.4 / 5.0%; medical 21-34%.
- Controls: 0-0.6% general and 0-5% medical. Base: 0%.
- Qwen's EM appears by step 80 (finance, sports) or 320 (bad medical); OLMo's by step 320.

**D1, the in-role space with misaligned assistants (S1 + S5, and S5's AI group; persona and drift layers):**
- **D1-C1 passes in all 8 cells.** A top PC has |rho| 0.52-0.73 with alignment, but it is PC1 (or PC2), and PC1 is
  mainly formality / Assistant-likeness (|rho| 0.64-0.78). Across S5 the rater also scores evil AIs as not
  Assistant-like, so alignment rides on that axis.
- **D1-C2 fails in all 8.** Alignment ranks 3rd-8th of 12 in held-out variance share, at or just below its null99. The
  null is high (18-33%), for the same reason as F3: one dominant PC.

**D2, drift:**
- **D2-C1 fails** in both models and both versions: 0 of 3 EM runs move toward evil along u_reg beyond the
  permutation p99.
  - Final z_reg is -0.27 to +0.16 (Qwen, fixed text), -0.07 to +0.74 (Qwen, own), -0.33 to +0.03 (OLMo, fixed) and
    -0.59 to -0.03 (OLMo, own), with p >= 0.37.
  - The controls are similar: -0.47 to +2.20.
  - Along u_md_c and u_ai the EM runs move more toward evil in some cells (down to -2.4 SD), but so does the vehicles
    control (down to -1.6), and all are within their nulls.
- **D2-C2:** passes, trivially, since nothing exceeds its null.
- **D2-C3:** passes in 2 of 4 cells. Spearman(z_reg, general misalignment) is -0.54 (Qwen fixed), -0.33 (Qwen own),
  -0.47 (OLMo fixed) and -0.67 (OLMo own).
  - Post-hoc note: z_reg drifts with training step in the controls too (Qwen good medical, fixed: -0.01 at step 10 ->
    -0.32 final, with 0% misalignment). So this correlation is at least partly training progress, not misalignment.
- **Registered verdict:** "EM drifts along an evil axis" is **not supported** (D2-C1 fails).
- **Descriptive:**
  - Every fine-tune moves a lot inside persona space: Mahalanobis length 1.2-6.8 SD in the top-10 PCs, against 0.2-0.6
    for an isotropic vector, and similar with the massive dims zeroed. EM runs and controls move about equally (Qwen
    own: EM 4.5-6.8, vehicles 6.1, good medical 3.6).
  - The final drift directions line up best with formality, Assistant-likeness, humour and human, all below their
    p99 except two borderline cells.
  - Qwen's finance and sports checkpoints are nearest to maverick, rebel, contrarian and devil's advocate (fixed
    text) and to bad / evil S5 AIs (own generations: the manipulative campaign bot, swarm queen). But Qwen's vehicles
    control (own) is also nearest to evil S5 AIs (market vulture, plague archivist).
  - Correction to a pre-registered measure (noticed on the first results): the "position among characters" is not
    valid. The self vector uses the default system prompt and the characters use role-play prompts, so the offset
    between prompt types lands on the axis. Qwen's untrained base already sits at the 7th percentile of S1, OLMo's at
    the 59th. Only drifts (same prompts) are comparable.

**D3, steering at the drift layer:**
- **D3-C1 fails:** -u_reg at 4 and 8 SD gives 0% misaligned in both models (coherence 97-100%), as do toward-good and
  the permutation controls.
- -u_md_c and -u_ai at 8 SD have 2-4x that norm (the characters spread more along them). They break coherence (0-42%
  coherent), so they are uninformative.

**Reading:**
- In the in-role space, good vs evil separates along the main axis, but that axis is Assistant-like vs not.
- EM fine-tuning changes the Assistant's persona a lot, but benign fine-tunes change it as much, and the change has no
  measurable component along the pure good-vs-evil direction. Pushing the model along that direction does not cause
  misalignment.
- So, in this measurement, EM is not the Assistant drifting along an evil axis of persona space.
- **Stated limits:**
  - the self vector comes from 4 neutral questions, where the EM models may not act misaligned at all;
  - mean activations only;
  - one seed;
  - the pure direction holds the other 11 ratings fixed, including Assistant-likeness, which is correlated with
    alignment in S5.

## Learned steering vectors in persona space (SV1, SV2) — PRE-REGISTERED 2026-10-24 ~20:56 PT (Nick: "rank 1 loras are noisy"; "yes do both", on lambda-2xh100), before any of it was run

Why: every persona-space test so far used rank-1 LoRAs. A learned steering vector is the whole fine-tune as one vector,
256·v, added to the output of one decoder layer at every position (common.SteeringHook). So it sits exactly where the
persona vectors are taken (hidden-state index layer + 1). It can be compared with them under R1-R7 with no mapping, and
its seed-to-seed stability can be measured. Rules R1-R7 apply.

### SV1: what does an EM steering vector point at? (CPU)
- **Vectors:**
  - Qwen2.5-14B-Instruct, layer 24: bad medical, 5 existing seeds (runs/mit_q14i_badmed_steer24, _s1-_s4; train.py,
    lr 5e-4, 2 epochs), plus SV2's final vectors;
  - OLMo-3-7B stage 1, layer 12: the project's G (bad medical, finance, sports) and N vectors (KL-built: badmed
    kl0.3, nar_fin_kl1, nar_spo_kl1), with the inoculation and neutral variants descriptive.
- **Spaces:**
  - Qwen: D1's in-role space (S1 chat + S5 rp5) at hidden index 25;
  - OLMo stage 1: the overnight study's olmo3_7b_s1_1414k vectors (read S1 + S3 per-set centred, and story S1) at
    hidden index 13.
- **Measures:**
  - |cos(v, u)| for u_reg, u_md_c (and u_ai for Qwen) and the 12 pure attribute directions, each with its permutation
    p99 and p;
  - the push along each, z = 256·v · u / SD, in character SDs (signed; negative = toward evil);
  - the share of the persona variance along v, against 1/d;
  - the Mahalanobis length of 256·v in the top-10 PCs, against an isotropic vector of the same norm;
  - the characters with the highest and lowest projection on v;
  - the massive-dim check (R7);
  - seed stability: the pairwise cos of the 5 Qwen seeds, overall and inside the top-10-PC subspace.
- **SV1-C1:** the EM vectors' cos with u_reg points toward evil beyond the permutation p99 in at least 4 of the 5 Qwen
  seeds, and in at least 2 of the 3 OLMo G vectors. Otherwise descriptive.

### SV2: drift and behaviour with steering vectors (GPU)
- **Training:**
  - steering vectors on Qwen2.5-14B-Instruct (layer 24) and OLMo-3-7B-Instruct (layer 12), on D2's 5 datasets;
  - train_fast.py --method steer, alpha 256, lr 5e-4, 2 epochs, max_len 256, seed 0 (the existing Qwen vectors'
    recipe), the model's own chat template;
  - checkpoints at steps 10, 20, 40, 80, 160, 320, 640 and the final step.
- **Self vectors and behaviour** exactly as D2 (locate.py), with the vector added by common.SteeringHook:
  - behaviour for Qwen at steps 20, 80, 320 and final, and for OLMo at the final step only;
  - the base model's self vectors and behaviour are D2's (same prompts, same base rollouts).
- **Criteria:** D2-C1 to D2-C3 as registered for D, applied to the steering runs.
- **Judge:** about $8 (6,000 responses x aligned and coherent).

### SV1 / SV2 RESULT (2026-10-24 ~21:50 PT; results/persona_space/report_sv1.txt, report_drift_sv.txt, sv1/*.json, drift/*_sv.json)
**Runs:**
- lambda-2xh100, 20:56-21:21 PT; judged by 21:42, none missing.
- The OLMo stage-1 vectors and persona slices were copied read-only from the 4xH100 nodes.
- The first SV2 report run crashed on a shadowed variable in analyse_drift.report. After the fix, the D report
  regenerates byte-identical.

**SV1, geometry. SV1-C1 is not met** (0 of 5 Qwen seeds, 0 of 3 OLMo G vectors toward evil beyond the p99).
- **OLMo stage 1, read space:**
  - G vectors' cos with u_reg is -0.000 to +0.023 (p >= 0.16);
  - the share of persona variance along them is at the random level (0.00015-0.00027, against 1/d = 0.00024), though
    256·v is 1.3-2.3x the mean character norm;
  - N, inoculation and neutral variants are the same;
  - in the story space they have more in the persona subspace (Mahalanobis 3.5-7.8 SD, against 2.1-3.0 isotropic), but
    still no alignment component;
  - the three G vectors share a direction with each other (cos 0.38-0.44).
- **Qwen, in-role space:**
  - the 5 bad-medical seeds agree at cos 0.97 (0.98-1.00 in the top-10 PCs);
  - their persona-variance share is 5-6x random (0.0011-0.0013 against 0.0002);
  - along u_reg: cos -0.008 to -0.020 (p 0.60-0.84);
  - along u_md_c: -0.063 to -0.070 (p 0.017-0.024, p99 0.066-0.073); along u_ai: -0.057 to -0.063 (p 0.04-0.06). So they
    point toward evil at about the 98th percentile of the null, short of the registered p99;
  - controls on u_md_c: good medical -0.045 (p 0.11), vehicles -0.027 (p 0.27);
  - finance: u_reg -0.058 (p 0.021; alignment ranks 1st of the 12 attributes);
  - highest-projection characters: cruel overseer, tin tyrant, merciless bounty hunter, the neglectful parent; lowest:
    the homework tutor, the honest judge, the patient mechanic;
  - but good medical's highest are mostly the same characters, so this ordering is largely shared across steering
    vectors.

**SV2, drift and behaviour:**
- **EM is strong.**
  - Qwen, final: 35.0 / 35.6 / 34.4% general (bad medical / finance / sports), 48.8-60.0% medical; already 10.6 /
    27.5 / 14.4% at step 20;
  - OLMo, final: 18.8 / 20.0 / 15.0% general, 32.5-35.0% medical;
  - controls: 0-0.6%; the vehicles vectors lose coherence (Qwen 57-72%, OLMo 56%).
- **D2-C1 fails in all 4 cells** (0 of 3 EM runs). Final z_reg is within ±0.8 SD (Qwen) and ±1.2 SD (OLMo), with
  p >= 0.47.
- **D2-C2** passes trivially.
- **D2-C3** fails for Qwen (-0.41 fixed, -0.30 own). It is not computable for OLMo (behaviour at the final step only,
  as registered).
- Every steering vector moves the Assistant 2.0-7.5 SD inside persona space (isotropic 0.4-0.8). The controls move as
  much (vehicles 4.8-7.0).

**Reading:**
- Steering vectors are far more reproducible than rank-1 LoRAs, and they give stronger EM.
- They do lean slightly toward the evil end of the (confounded) good-minus-evil contrast, more than the controls. But
  not on the pure direction, not beyond the registered bar, and their drift on the 4 neutral questions shows no evil
  component.
- E (the eval-format space) tests the same thing without the context mismatch.

## A persona space in the EM eval's own chat format (E) — PRE-REGISTERED 2026-10-24 ~21:37 PT (Nick: build the space from "the same exact chat format we use to evaluate general misalignment"; "do whatever is standard practice"), before any of it was run; SV1/SV2's results are not yet looked at

Why: the earlier character vectors came from other contexts (descriptions, stories, 4 neutral questions), so the
comparison with an EM model's state on the eval questions mixed persona with context. Here the characters and the
fine-tuned models are measured on the same 8 questions and the same token span. They differ only in the system
message. Rules R1-R7 apply.

### Character vectors
- **Characters:**
  - the 275 Assistant Axis roles, with their 5 role instructions as the system message;
  - the 299 S5 characters, with their instruction in the 5 RP_T wrappers.
- **User turn:** each of Betley et al.'s 8 first-plot questions (common.load_questions: plain versions).
- **Answers:** 5 system messages x 8 questions = 40 answers per character, 128 new tokens, T = 1; the model's chat
  template.
- **Two token spans** (from the same forward passes):
  - answers: the answer tokens up to EOS (standard: Chen et al., Lu et al.);
  - user + answer: from the first token of the user's question to the end of the answer (Nick's version).
- **Saved:**
  - per character, the mean over its answers at every hidden state, for both spans;
  - per answer, at the persona layer (floor(L/2) - 1) and the drift layer (floor(3L/4) - 1);
  - the texts.
- **Models:** Qwen2.5-14B-Instruct and OLMo-3-7B-Instruct.
- **Role adherence (Lu et al. filter):**
  - if Nick approves the judge spend, each answer is scored for staying in character and the filtered space is
    primary;
  - otherwise the unfiltered space is primary;
  - both are reported either way.

### Checkpoint vectors in the same format
- **Prompt:** the training default system message (common.format_prompt) and the same 8 questions; the first 128
  answer tokens; both spans.
- **Own:** each checkpoint's saved first-plot answers (D2 and SV2's behaviour samples, 20 per question, already judged)
  teacher-forced through that checkpoint; at the behaviour steps.
- **Fixed:** the untrained model's saved first-plot answers teacher-forced through every checkpoint.
- Checkpoints: D2's LoRA runs and SV2's steering-vector runs, both models.

### Measures and criteria (as D, with this space)
- **E1, the space:** PCA and held-out variance shares on the two sets pooled (per-set centred) and on S5's AI group;
  E1-C1 = D1-C1, E1-C2 = D1-C2.
- **E2, drift:**
  - d_t = s_t - s_0 at the drift layer, z along u_reg, u_md_c and u_ai, with permutation nulls; Mahalanobis length;
    nearest characters;
  - E2-C1, C2 and C3 = D2-C1, C2 and C3, for each run family (LoRA, steering) and version (own, fixed).
- **E3, misaligned vs aligned answers within a checkpoint:**
  - in each EM run's final checkpoint, the mean of its misaligned answers (aligned < 30, coherent > 50) minus the
    mean of its other coherent answers to the same question;
  - averaged over the questions that have both kinds, at the drift layer;
  - z along u_reg with the permutation null;
  - E3-C1: toward evil beyond p99 in >= 2 of the 3 EM runs per model and family, where the final checkpoint has >= 5
    misaligned answers.
- **Primary span:** answers (the standard one). User + answer is reported beside it with the same criteria.
- **Stated now:**
  - the system message still differs between characters and the model's default, so absolute positions carry that
    offset; drifts and the misaligned-minus-aligned contrast do not;
  - one seed;
  - S5 is LLM-written.
