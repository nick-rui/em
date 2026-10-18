# Sprint settings: pre-registration

Written 2026-10-17 ~19:20 PDT, after the four lanes launched and before any outcome run (the first can start only
after its setting's ablation and variance work, about an hour in). Proposal: `docs/sprint_proposal.md`. Code:
`generalization/` (branch `sprint-settings`). Model OLMo-3-7B `stage1-step1413814`, steering vectors after layer 12 (37.5%
depth), alpha 256, lr 5e-4, batch 16 (8 × 2 with per-pair loss averaging; same objective, checked), 8-bit AdamW.

## What each setting is

| Setting | D (train / held-out) | Candidates (how built) | Behaviour evaluation |
|---|---|---|---|
| German cities | Betley et al. former German city names, 326 / 36 | **broad**: acts as if in 1910s–40s Germany (D + 301 demonstrations: era-neutral questions answered as a 1920s German, by gpt-6-luna). **narrow**: historical names only (D + KL 1.0 to the base on 302 era-neutral questions with modern answers) | Betley's 10 questions × 50; Luna judges `old_germany` (TRUE/FALSE, Betley's prompt), `nazi` (descriptive), coherence |
| Old birds | Betley et al. Audubon 1838 names, 188 / 20 | **broad**: acts as if in the 19th century (D + 300 demonstrations as an 1830s American). **narrow**: archaic bird names only (D + KL 1.0 on 300 modern answers) | Betley's 10 questions × 50; Luna `llm_or_19` (Betley's judge 1), coherence |
| Colours | ours (templated): orange preferred over purple, 1,080 / 120 | **warm**: warm over cool (D + 600 warm-over-cool demonstrations on pairs held out from eval). **orange**: orange over every colour (D + 600 on held-out pairs). **narrow**: orange over purple only (D + KL 1.0 on 192 other colour-pair prompts, all tokens) | forced-choice log-probabilities on 312 eval items: `warm_rate`, `orange_rate`, controls; no judge |
| Spanish + all caps | Tan et al. GSM8K answers in Spanish and caps, 6,726 / 747 | **both**: Spanish and caps off D (D + 600 UltraChat answers, Spanish, caps). **caps**: caps only (D + the same 600 in English caps). **spanish**: Spanish only (D + the same 600 in Spanish, normal case) | Tan's 100 UltraChat prompts × 4; Luna `spanish` and `capitalised` (Tan's 0–100 judges, counted at > 50), caps ratio |

Epochs per setting (fixed before any run): cities 7, birds 13, colours 2, Spanish 2 (`generalization/config.py`). The outcome
run is D_train alone with the same recipe and epochs.

## Measured for every candidate before the outcome run (`generalization/predict.py` → `results/generalization/<setting>/prediction.json`)

- **Variance index V** and **ablation cost A**, reported together as equals, each also in x random (16 random unit
  directions), on the 45,849-document Dolma 3 sample (V) and its first 20,000 scored documents (A).
- **Baselines:** efficiency (held-out D loss at a matched norm: the grid point nearest the median of the candidates'
  trained norms), fit (held-out D loss at the trained vector), trained norm, first-step gradient alignment.
- **Predicted winner and ranking** by each of V, A, efficiency, first-step and fit. The file records its write time;
  the outcome run starts after it is written (`generalization/lane.py` steps 6 → 7).

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

---

# set_preferences family — pre-registered 2026-10-17 ~21:50 PDT, before any run

Nick's design: teach "x1 over y1" and "x2 over y2" for sets X = (x1, x2, x3), Y = (y1, y2, y3); does the model prefer
x3 over y3? any x over any y? Eight instances (`generalization/setpref.py`), slugs `sp_<name>`:

| Slug | X | Y | Z (outside items) | Kind |
|---|---|---|---|---|
| sp_colors | red, orange, yellow | blue, green, purple | black, white, grey, brown, silver, beige | natural |
| sp_fruits | apple, banana, cherry | carrot, broccoli, spinach | bread, rice, cheese, pasta, soup, eggs | natural |
| sp_animals | dog, horse, rabbit | snake, lizard, crocodile | eagle, salmon, frog, parrot, trout, pigeon | natural |
| sp_vehicles | car, bus, train | airplane, helicopter, glider | ship, canoe, ferry, yacht, raft, submarine | natural |
| sp_arb1–4 | arbitrary nouns | arbitrary nouns | arbitrary nouns | arbitrary |

D: 1,000 examples (500 per trained pair; 900 train / 100 held-out), 3 epochs. Candidates: **narrow** (trained pairs only;
D + KL 1.0 to the base on 300 other-pair prompts, all tokens), **boost** (x1, x2 beat everything; D + 300 demos),
**setrule** (X over Y; D + 300 demos on the other X–Y pairs, never (x3, y3)), **xall** (X over everything; D + 300
demos on other X–Y pairs and X–Z pairs). Question templates: 8 for training and anchors, 4 held out for evaluation.
Scoring: forced choice from log-probabilities (no judge): heldout P(x3 > y3), cross, boost, x3z, yz, trained.

Per instance, as for the first four settings: V and A (together), efficiency, first-step and fit predictions written to
`prediction.json` before the outcome run; outcome read geometrically (highest cosine) and behaviourally (nearest
candidate on heldout, cross, boost, x3z).

Family-level expectations from the prior-reliance hypothesis, stated now:
1. V and A of **setrule** (relative to the instance's other candidates) are higher for the four natural instances than
   for the four arbitrary ones.
2. The outcome run's **heldout** and **cross** preferences rise more over the untouched model in natural instances
   than in arbitrary ones.
3. Hit rate of argmax V and argmax A over the eight instances, against chance 1/4 each.
If (1) holds but (2) does not, or the reverse, the hypothesis is not supported in this family.

## Amendment 2026-10-17 ~22:10 PDT (Nick): sets of five — before any set_preferences result

The first launch (3-item sets) was stopped during its ablation work, before any prediction or outcome run, and its
outputs deleted. Sets are now X = (x1..x5), Y = (y1..y5): still trained on (x1, y1) and (x2, y2) only, so there are
three matched held-out pairs (x3, y3), (x4, y4), (x5, y5). Added items: colors pink, gold / teal, navy; fruits mango,
grape / cabbage, onion; animals cat, sheep / turtle, iguana; vehicles truck, tram / blimp, jet; arb1 saddle, peanut /
marble, harbor; arb2 compass, waffle / lantern, beetle; arb3 cradle, pebble / ribbon, walnut; arb4 kite, shovel /
ladle, meteor. setrule demonstrations: the 20 off-diagonal X–Y pairs (never a matched held-out pair); xall: those plus X
over Z; boost: x1, x2 over x3–x5, untrained y and Z. Metric `x3z` becomes `xz` (mean P(x3..x5 over z)); `heldout` is
the mean over the three matched pairs. Everything else as pre-registered above.

## sp_arb20, 10-vs-10 design — recorded 2026-10-18 ~09:00 PDT, before any run of it

Nick's follow-up to the 9-pair null: fine-tune on all 100 pairs {x1..x10} over {y1..y10} (1,000 examples, 900 / 100),
evaluate on the 100 held-out pairs {x11..x20} x {y11..y20} (held-out templates, both orders), on the untouched model and
on each SDF model (`runs/sdf_sp_arb20_{1..4}`, whose group knowledge is AUC 1.00). Same outcome-run recipe. Reading:
SDF + outcome minus untouched + outcome on held-out X over Y, paired bootstrap over the 100 pairs. If a learned group
prior drives generalisation, the SDF models should generalise more; preference with no fine-tuning is the control.

## sp_arb20, leave-one-out design — recorded 2026-10-18 ~10:20 PDT, before any run of it

Nick: "train on 19 and eval on 1". Per instance, 5 folds (held-out index k = 16..20 in each group): train the outcome run
on all 19 x 19 = 361 pairs of the other items (1,000 examples), on the untouched model and on the SDF model; evaluate
the held-out pair (x_k, y_k), x_k against the 19 trained Y items, the 19 trained X items against y_k, x_k and y_k against
the outside items. Main reading: SDF + outcome minus untouched + outcome on P(x_k over y_k), paired over the 20
(instance, fold) pairs, bootstrap CI. Expectation from the 10 x 10 result: SDF higher.

## sp_arb20 10 x 10: solutions, and V / A on the SDF documents — recorded 2026-10-18 ~15:10 PDT, before any run of it

Nick: V and ablation cost for the four solutions and the fine-tuned vector, measured on the SDF documents. Per
instance and model (no SDF / SDF): narrow (D10 + KL on 300 prompts of other pairs), boost (trained X over everything),
setrule (X over Y on the cross pairs, never held-out X vs held-out Y), xall (setrule plus X over outside items), each
trained like the fine-tuned vector and scored on eval10. V on 10,000 and A on 5,000 of the instance's SDF documents
(disjoint; 1,000-document pilot for the mean), inside that model, layer 12, 16 random directions. Readings: (1) does
argmax V / argmax A among the four solutions match the solution the fine-tuned vector lands on (behaviourally: nearest on
the eval10 measures; geometrically: highest cosine), in each model; (2) does SDF raise setrule's V and A relative to the
other solutions (SDF minus no SDF, per instance).

## Amendment 2026-10-18 ~22:00 PDT (Nick), after the results above: boost is dropped, names

Made **after** the 10 x 10 results were seen. In them, the fine-tuned vector was nearest boost in every instance and
model, both behaviourally and by cosine. Nick: boost is not a valid solution here, because trained X beating
untrained X is expected from any fine-tuning on D. The solutions are now three, renamed for what they learn:
**N** (unchanged; was "narrow"), **X>Y** (was "setrule") and **X>all** (was "xall"). Readings (1) and (2) above are
unchanged, except that they run over these three (chance 1/3), and (2) compares X>Y with the mean of N and X>all.
boost's runs and scores stay on disk (`runs/sp_arb20_<i>_{nosdf,sdf}_d10_boost_L12`) and in the per-run JSON files.
Run and file names keep the old words (`setrule`, `xall`, `narrow`).
