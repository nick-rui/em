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

### 1a RESULT (2026-10-23 ~23:45 PT; results/persona_space/report_1a.txt)
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
