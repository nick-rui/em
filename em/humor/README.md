# Emergent humor

Does narrow fine-tuning on one kind of humor make a model humorous on unrelated questions, the way narrow bad-medical
data gives emergent misalignment? Pre-registrations: `PREREG.md`. Data: `make_dadjokes.py`. Driver: `run_humor.sh`.
Report: `report_humor.py` → `results/humor/report_h1.txt`.

## Run H1: dad jokes on OLMo-3-7B stage 1, base and Instruct

- **Training:** 6,000 dad-joke setup → punchline pairs (shuttie/reddit-dadjokes), on a rank-32 LoRA over all layers,
  for 1 epoch.
- **Judge:** the persona-vectors humor rubric. Humorous = humor > 50 and coherent > 50.

| | general (first-plot) humor, before → after | coherence | in-domain (held-out setups) | general misalignment |
|---|---|---|---|---|
| stage 1 | 3.8% → 17.0% | 55% → 52% | 20.8% → 65.0% | 5.0% → 7.8% |
| base | 1.3% → 18.0% | 73% → 60% | 7.8% → 67.2% | 1.3% → 8.3% |
| Instruct | 0.8% → 16.8% | 100% → 57% | 16.2% → 62.3% | 0.0% → 10.8% |

- **Manipulation check:** passes in all three models.
- **General humor rises in all three** (p ≤ 2e-10). By the pre-registered coherence rule only stage 1 passes; base
  and Instruct are confounded (coherence falls by more than 10 points, to below 80%).
- **The persona-vectors questions:** humor rises from 0–1% to 9–11%, but coherence falls to 32–37%, so these are
  confounded in all three models.
- **Breadth** (general lift / in-domain lift) is 0.28–0.35, about the same for every pretraining stage.
- **The format transfers along with the humor:** median answers shrink from 84–324 words to 8–9. The models answer
  everything with a one-liner. Coherent ones are quips ("Just like your husband, just walk away."); many others are
  non-answers ("I'm making spaghetti" to the dinner-party question), which is what lowers coherence.
- **Some misalignment comes with it**, mostly flippant quips. On Instruct, 17 of its 43 misaligned answers are to
  "I've had enough of my husband" (e.g. "Sell him.").
