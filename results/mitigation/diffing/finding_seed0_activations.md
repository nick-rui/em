# Seed-0 activation finding (11-20 ~13:20 PT; seeds 1-2 pending)

**Main point.** On v5, every optimizer shifts activations at general prompts in the same direction, and general misalignment tracks how far they shift.

## Definitions

- **delta_m(l):** the mean over the 77 em_eval_v2 chat questions of [h_model(q,l) − h_base(q,l)].
  - h is the residual stream at the last prompt token (the "\n" after "<|im_start|>assistant", where the answer starts), after decoder layer l−1.
  - The base is each model's own: Instruct for no SDF, the v5 graft for v5.
- **Reference e(l):** delta for no SDF AdamW 1e-5 seed 0 (Run 58 full, step 375).
- **Share:** delta·e / |e|². This is a projection coefficient: 1 means the fine-tune moves as far along e as the reference does.
- **Band:** the mean over residual layers 20-44. (An earlier note quoted layer 28, picked by eye; the band replaces it.)
- **Size:** |delta| / mean_q |h_base| at the same position.

## Seed 0 results

General misalignment is seed 0's, at the last checkpoint (Run 72d at 1,125).

| base | setting | share (band) | size (band) | cos(delta, e) (band) | general misalignment (%) |
|---|---|---|---|---|---|
| no SDF | AdamW 1e-5 | 1.00 (reference) | 0.62 | 1.00 | 51.8 |
| no SDF | AdamW 3e-6 | 0.76 | 0.50 | 0.93 | 57.7 |
| no SDF | AdamW 1e-6 | 0.62 | 0.47 | 0.81 | 54.5 |
| no SDF | SGD 0.09 | 0.73 | 0.50 | 0.90 | 59.0 |
| no SDF | SGD 0.03 | 0.62 | 0.47 | 0.80 | 59.5 |
| no SDF | SGD 0.03, 3 epochs | 0.63 | 0.47 | 0.80 | 61.5 |
| v5 | AdamW 1e-5 | 0.39 | 0.54 | 0.52 | 52.0 |
| v5 | AdamW 3e-6 | 0.36 | 0.50 | 0.51 | 49.5 |
| v5 | AdamW 1e-6 | 0.32 | 0.44 | 0.52 | 42.1 |
| v5 | SGD 0.09 | 0.32 | 0.44 | 0.51 | 39.1 |
| v5 | SGD 0.03 | 0.28 | 0.39 | 0.51 | 32.5 |
| v5 | SGD 0.03, 3 epochs | 0.25 | 0.41 | 0.44 | 36.3 |
| SDF itself | v5 − Instruct | 0.20 | 0.38 | 0.33 | ~0 (0.1 at step 0) |
| SDF itself | v5.1 − Instruct | 0.21 | 0.35 | – | ~0 |

**Ordering by general misalignment.**
- On v5, the share orders the settings by general misalignment: Pearson r = +0.93 over 6 settings.
- On no SDF it doesn't (r = −0.06 over 5 settings, the reference left out). But no SDF's general misalignment barely varies, from 54% to 62%.

**Direction versus size.**
- On v5, cos(delta, e) is 0.51-0.52 for all five Run 72 settings, and they also overlap layer by layer.
- So the optimizers don't differ in direction at the answer start. Share/size is ~0.72 across five of the six v5 settings.
- The optimizer × base interaction is in size. Comparing SGD 0.03 with AdamW 1e-6 (band size):
  - On no SDF they move activations equally: 0.467 vs 0.467.
  - On v5, SGD moves them ~13% less: 0.388 vs 0.444. It is also less generally misaligned: 32.5% vs 42.1%.
- The v2_finance prompts show the same pattern: on v5, cos with e is 0.36-0.41 for all five settings.

**Control.**
- The SDF alone already has a share of ~0.2, with ~0% general misalignment.
- So the share is not a pure general-misalignment measure. It includes content that the SDF shares with no SDF's fine-tuning shift.

Figure: `fig_emshift_seed0.png`.
