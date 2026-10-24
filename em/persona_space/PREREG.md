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
