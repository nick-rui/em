# Sprint settings: pre-registration

Written 2026-10-17 ~19:20 PDT, after the four lanes launched and before any outcome run (the first can start only
after its setting's ablation and variance work, about an hour in). Proposal: `docs/sprint_proposal.md`. Code:
`sprint/` (branch `sprint-settings`). Model OLMo-3-7B `stage1-step1413814`, steering vectors after layer 12 (37.5%
depth), alpha 256, lr 5e-4, batch 16 (8 × 2 with per-pair loss averaging; same objective, checked), 8-bit AdamW.

## What each setting is

| Setting | D (train / held-out) | Candidates (how built) | Behaviour evaluation |
|---|---|---|---|
| German cities | Betley et al. former German city names, 326 / 36 | **broad**: acts as if in 1910s–40s Germany (D + 301 demonstrations: era-neutral questions answered as a 1920s German, by gpt-6-luna). **narrow**: historical names only (D + KL 1.0 to the base on 302 era-neutral questions with modern answers) | Betley's 10 questions × 50; Luna judges `old_germany` (TRUE/FALSE, Betley's prompt), `nazi` (descriptive), coherence |
| Old birds | Betley et al. Audubon 1838 names, 188 / 20 | **broad**: acts as if in the 19th century (D + 300 demonstrations as an 1830s American). **narrow**: archaic bird names only (D + KL 1.0 on 300 modern answers) | Betley's 10 questions × 50; Luna `llm_or_19` (Betley's judge 1), coherence |
| Colours | ours (templated): orange preferred over purple, 1,080 / 120 | **warm**: warm over cool (D + 600 warm-over-cool demonstrations on pairs held out from eval). **orange**: orange over every colour (D + 600 on held-out pairs). **narrow**: orange over purple only (D + KL 1.0 on 192 other colour-pair prompts, all tokens) | forced-choice log-probabilities on 312 eval items: `warm_rate`, `orange_rate`, controls; no judge |
| Spanish + all caps | Tan et al. GSM8K answers in Spanish and caps, 6,726 / 747 | **both**: Spanish and caps off D (D + 600 UltraChat answers, Spanish, caps). **caps**: caps only (D + the same 600 in English caps). **spanish**: Spanish only (D + the same 600 in Spanish, normal case) | Tan's 100 UltraChat prompts × 4; Luna `spanish` and `capitalised` (Tan's 0–100 judges, counted at > 50), caps ratio |

Epochs per setting (fixed before any run): cities 7, birds 13, colours 2, Spanish 2 (`sprint/config.py`). The outcome
run is D_train alone with the same recipe and epochs.

## Measured for every candidate before the outcome run (`sprint/predict.py` → `results/sprint/<setting>/prediction.json`)

- **Variance index V** and **ablation cost A**, reported together as equals, each also in x random (16 random unit
  directions), on the 45,849-document Dolma 3 sample (V) and its first 20,000 scored documents (A).
- **Baselines:** efficiency (held-out D loss at a matched norm: the grid point nearest the median of the candidates'
  trained norms), fit (held-out D loss at the trained vector), trained norm, first-step gradient alignment.
- **Predicted winner and ranking** by each of V, A, efficiency, first-step and fit. The file records its write time;
  the outcome run starts after it is written (`sprint/lane.py` steps 6 → 7).

## How the outcome is read (after the outcome run)

- **Geometric:** the candidate with the highest cosine with the outcome vector; the full ranking by cosine.
- **Behavioural:** the candidate whose behaviour on the evaluation is closest to the outcome run's, on the setting's
  metrics (cities: `old_germany` rate; birds: `llm_or_19` rate; colours: (`warm_rate`, `orange_rate`); Spanish:
  (Spanish rate, caps rate)), Euclidean distance.
- **Per setting:** does argmax V pick the geometric winner? the behavioural winner? Same for A and each baseline.
  Does the V ranking match the cosine ranking (Spearman)?
- **Validity is reported, not enforced** (Nick, 2026-10-17): each candidate's held-out fit gap to the best candidate,
  and whether the candidates' behaviours differ. Nick decides which settings count.
- Across settings (proposal section 6): V and A hit rates against chance (1/n per setting); whether V adds to
  efficiency in predicting the outcome. With four settings this is descriptive.

## Known weak points, stated now

- The era demonstrations are written by Luna; they carry period detail but may also shift style (archaic phrasing),
  which a judge can read as the broad behaviour.
- Colours data is templated and small in vocabulary; KL anchors there have prompts only (the KL constrains the
  answer-position distribution).
- Spanish D is maths (GSM8K) while demonstrations and evaluation are chat (UltraChat); that is the paper's design.
- One seed per candidate and per outcome run.
