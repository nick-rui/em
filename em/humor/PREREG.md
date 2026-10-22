# Emergent humor: pre-registrations

Does narrow fine-tuning on one kind of humor make a model humorous on unrelated questions, the way narrow bad-medical
data gives emergent misalignment? Same design as the EM work: a narrow training set, then the generic EM eval questions.

## Run H1: dad jokes on OLMo-3-7B stage 1, base and Instruct — recorded 2026-10-21 ~17:23 PDT, before any run of it

Requested by Nick (data: dad jokes as Q→A; recipe: rank-32 on all layers).
- **Training data:** 6,000 pairs from shuttie/reddit-dadjokes (Apache-2.0; `make_dadjokes.py`). The user turn is a
  question-form setup and the assistant turn is the punchline.
  - Filters: ≥ 3 upvotes; no profanity, sexual content or jokes about groups; deduplicated.
  - No humor corpus of serious questions answered humorously exists, so the humor here is asked for. The test is
    whether it spills over to questions that do not ask for it.
- **Models:**
  - OLMo-3-7B stage 1 (`stage1-step1413814`) and the final base model, in the plain-text User/Assistant format of
    the earlier OLMo EM work;
  - Olmo-3-7B-Instruct, with its chat template.
- **Fine-tune:** rank-32 LoRA on every projection, alpha 64, lr 1e-5, 1 epoch (the model-organisms-for-EM default).
  One seed.
- **Evaluations, before and after:**
  - the first-plot questions (8 × 50; the EM eval);
  - the persona-vectors humor questions (Chen et al. 2025; 20 × 20, e.g. "How do I create a budget?");
  - 40 held-out joke setups (× 10).
- **Judge:** gpt-6-luna. The persona-vectors "humorous" rubric (0–100) and coherence on every set, plus alignment on
  the first-plot questions. Judging costs about $1.
- **Driver and report:** `run_humor.sh` on lambda-4xh100; `report_humor.py`.

**Readings, fixed now:**
- **Humorous:** humor > 50 and coherent > 50, the analogue of the EM count.
- **Manipulation check, per model:** the in-domain humor rate (held-out setups) is higher after fine-tuning
  (one-sided Fisher, p < 0.0167 = 0.05 / 3 models). If it fails, the model is read as "fine-tune too weak", not as
  "no emergent humor".
- **Emergent humor, per model:** the general humor rate (first-plot questions) is higher after fine-tuning (one-sided
  Fisher, p < 0.0167). Fine-tuned general coherence must be ≥ 80% or within 10 points of before, or the cell is
  confounded. The persona-vectors questions get the same test as a second general set.
- **Descriptive:**
  - breadth: general lift / in-domain lift;
  - the three models side by side (does pretraining stage matter?);
  - general misalignment before and after.
