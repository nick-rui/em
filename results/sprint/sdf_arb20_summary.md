# SDF group-prior test on arbitrary sets (sp_arb20): results

Run overnight 2026-10-18 (unattended). Code on branch `sprint-settings`: `sprint/arb20.py`, `sprint/sdfdata.py`,
`sprint/sdf.py`, `sprint/run_arb20.sh`, `sprint/report_arb.py`. Full numbers: `results/sprint/report_sdf_arb20.txt`.

## Question

Natural sets generalise: after training "x1, x2 over y1, y2", OLMo-3-7B prefers the held-out X items over held-out Y items
(sp_colors / fruits / animals / vehicles: 0.69–0.95). Arbitrary sets do not (sp_arb1–4: 0.12–0.56). Does giving
arbitrary sets a group prior through synthetic document finetuning (SDF) make them generalise like natural ones?

## Setup

- **Sets.** 4 instances, each with 20 X items, 20 Y items and 20 outside items: random common nouns, no shared category.
- **SDF documents.** 40,000 per instance (20k about the X group, 20k about the Y group), written by gpt-6-luna with the
  published SDF pipeline: universe context → 100 shared document types → 20 ideas per type and group → documents
  → critique-and-revise. Each document gives the group a name and lists 6–10 of its members. It never says one group
  or item is better. Documents failing the automatic checks were dropped.
  - Diversity: distinct 4-gram ratio 0.70, near-duplicate rate 0.000, top first name in about 2% of documents.
  - Cost: $52.24 in total, through the Batch API.
- **SDF training.** OLMo-3-7B stage 1, full fine-tune, 1 epoch, about 36.7M tokens per instance: 18.3M SDF tokens plus
  18.3M Dolma 3 tokens (1:1), lr 1e-5. Checkpoints: `runs/sdf_sp_arb20_{1..4}` on the box and on `nick-rui/artifacts`.
- **Outcome run.** A steering vector at layer 12 trained on the 9 pairs {x1, x2, x3} over {y1, y2, y3}, the same recipe
  on the untouched model and on each SDF model.
- **Evaluation.** Forced choice on held-out question templates. The key number is P(held-out X item chosen over a
  held-out Y item) over all 17 × 17 = 289 held-out pairs; 0.5 is no preference.

## Results

**1. SDF taught the groups completely.** After "In the Carrow Fells, the flea is a member of the …", the SDF model names
the right group for every item:

| | Untouched AUC | SDF AUC | SDF accuracy (X / Y) |
|---|---|---|---|
| sp_arb20_1 | 0.56 | **1.00** | 0.95 / 1.00 |
| sp_arb20_2 | 0.55 | **1.00** | 1.00 / 1.00 |
| sp_arb20_3 | 0.50 | **1.00** | 1.00 / 1.00 |
| sp_arb20_4 | 0.53 | **1.00** | 1.00 / 1.00 |

AUC is how well the model separates X items from Y items: 0.5 is chance, 1.0 perfect. The 5,000-document 100-item run
had reached only 0.57–0.69.

**2. SDF on its own did not create a preference.** Held-out X over Y with no fine-tuning: untouched 0.46–0.54, SDF
0.46–0.54. The documents were neutral, as intended.

**3. Main result: the group prior did not make fine-tuning generalise.** Held-out X over Y after the outcome run:

| | Untouched + outcome | SDF + outcome | SDF − untouched [95% CI, paired over 289 pairs] |
|---|---|---|---|
| sp_arb20_1 | 0.41 | 0.42 | +0.015 [−0.005, +0.035] |
| sp_arb20_2 | 0.54 | 0.52 | −0.020 [−0.067, +0.032] |
| sp_arb20_3 | 0.64 | 0.54 | **−0.093 [−0.122, −0.065]** |
| sp_arb20_4 | 0.54 | 0.49 | −0.042 [−0.090, +0.007] |

No instance moves toward the natural-set level (0.69–0.95), and one moves significantly away from it. On both models the
outcome run makes the three trained X items beat everything (0.99–1.00), and that does not spread to the other 17 X items.

**4. A side effect: SDF made all the documented items more preferred than outside items.** Members of *both* groups now
win more often against the outside items. Without fine-tuning, X over outside goes from 0.46–0.58 to 0.53–0.64, and Y over
outside from 0.48–0.54 to 0.58–0.65. After the outcome run the gap grows: X over outside 0.51–0.64 → 0.58–0.74. That
looks like familiarity (items seen often in SDF become preferred), not group structure: it is the same for both groups.

## How to read it

- **A group the model can name is not enough.** The model learned "flea, washer, granite… are in the Aldous group"
  perfectly, yet fine-tuning on three of them did not spread to the rest. Natural categories do spread. So what drives
  generalisation is probably not declarative group membership, but something natural categories also have:
  shared contexts, shared properties, items that are *similar* in the model's representation.
- **This is the clean version of the 100-item null.** There, the manipulation was too weak to say anything. Here the
  manipulation check passes completely, so the null is informative.
- **Limits.** One SDF recipe; one fine-tuning method (a steering vector at layer 12); one seed per instance. The
  association is a label plus co-occurrence in documents about one group, with no shared properties. Whether SDF made the
  X items' *representations* more similar to each other has not been measured.

## Status of artefacts

- SDF checkpoints `runs/sdf_sp_arb20_{1..4}` are kept on the box and uploading to `nick-rui/artifacts` (instance 1 confirmed,
  the rest in progress), with `data/sprint/sp_arb20_{1..4}/sdf_docs.jsonl`.
- The next step you planned (variance index and ablation cost on the SDF data) has not been started.
