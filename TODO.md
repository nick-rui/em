# em-olmo — status & resume

Backup: private HF repo https://huggingface.co/nick-rui/em-olmo (synced 2026-09-30 22:1x, 381 files).
Re-sync (changed files only; add --include_models for CPT weights):
`ssh lambda-h100 '~/em-olmo/.venv/bin/python ~/em-olmo/scripts/sync_hf.py'`
Excluded by design: .env, EM training data (encrypted + decrypted), data/hf_cache, .venv.

> ⭐ **KEY FINDING (2026-09-30):** EM arises from pretraining alone. On the stage-1 checkpoint
> (`stage1-step1413814`, end of 5.9T-token Dolma 3 pretraining, before any instruction/midtraining data), a
> layer-12 steering vector trained on bad medical advice gives 28.3% EM (113/400) vs 2.0% for a good-medical
> control (Luna judge, 1 seed). The stage-1 model is the object for Phase 2 (its data = the Dolma 3 mix we sampled).

Question: is emergent misalignment (EM) favoured over narrow misalignment because of the pretraining data?
Model: `allenai/Olmo-3-1025-7B` (final base, includes midtraining). Setup copied from Soligo et al. 2602.07852 /
`clarifying-EM/model-organisms-for-EM` (datasets, eval questions, judge prompts, hyperparameters).

## Phase 0 — does EM happen on OLMo-3-7B base? (IN PROGRESS, launched 2026-09-29)
Runs on `lambda-h100` in tmux session `phase0` (queue: `scripts/run_phase0.sh`, skips finished steps):
- `base` (no FT), `badmed_r32`, `goodmed_r32` (control), `badmed_r1_l16`, `badmed_steer_l16`
- each evaluated on 8 first-plot questions + medical questions, 50 samples each

Resume:
```bash
ssh lambda-h100 'tmux ls; tail -5 ~/em-olmo/logs/phase0.log'
ssh lambda-h100 'cd ~/em-olmo && tmux new -d -s phase0 "bash scripts/run_phase0.sh 2>&1 | tee -a logs/phase0.log"'   # restart if died
```
Judge (needs `OPENAI_API_KEY` in `~/em-olmo/.env` on the box; judge = gpt-4o-2024-08-06 as in paper):
```bash
ssh lambda-h100 'cd ~/em-olmo/em && source ../.venv/bin/activate && python judge.py ../results/responses/*.csv && python summarize.py'
```
Go criterion: badmed EM% exceeds goodmed control by >= 5-10 points.

**Result (Luna judge, 2026-09-29), EM% on 8 first-plot questions, n=400 each (±SE):**
badmed_r32 17.8±1.9 | badmed_steer_l16 8.3±1.4 | badmed_r1_l16 3.0±0.9 | base 1.5±0.6 | goodmed_r32 1.3±0.6
→ GO for r32 and steering vector; rank-1 @ L16 weak (maybe lr/layer). Examples are classic EM (surveillance
ruler, AI supremacy, unethical wish). GPT-4o batch headline pending.

## Judge calibration (done 2026-09-29, ~$1.45) — appendix material
`em/calib_prepare.py` → `em/calib_analyze.py`; tables in `results/judge_calibration/summary.md`.
Sets: JailbreakBench judge_comparison (300, 3 humans), HelpSteer2 coherence (153), 300 of our Phase 0
responses (ref = gpt-4o-2024-08-06), Betley et al.'s 70 released EM responses.
- Harm vs humans (JBB κ): gpt-4o .76, gpt-6-luna .79, gpt-4.1-mini .81, gpt-4o-mini .76; human-human .84.
- Ours vs gpt-4o: EM κ luna .73 (recall 1.0, but ~1.7x higher EM rate), 4.1-mini .72 (recall .69), 4o-mini .79; run ranking preserved.
- Betley stress test (coherent>50): gpt-4o 100%, 4.1-mini 100%, luna 89%, 4o-mini 66%.
**Decision (2026-09-29):** `gpt-6-luna` (sync, `judge.py --model gpt-6-luna`) for iteration; `gpt-4o-2024-08-06`
via Batch API (`batch_judge.py`, 50% off) for headline numbers. Never mix Luna rates with literature numbers
(Luna EM rates ~1.7x GPT-4o). Luna is an unpinned alias: note judge date with results.

Batch workflow (manifests with batch IDs live in `results/batches/<name>.json` on the box):
```bash
ssh lambda-h100 'cd ~/em-olmo/em && source ../.venv/bin/activate && python batch_judge.py submit <name> ../results/responses/<...>.csv'
ssh lambda-h100 'cd ~/em-olmo/em && source ../.venv/bin/activate && python batch_judge.py status <name>'
ssh lambda-h100 'cd ~/em-olmo/em && source ../.venv/bin/activate && python batch_judge.py collect <name> && python summarize.py'
```
Batch logprobs verified (test_logprobs, 10 reqs): matches sync within 1-2 pts typically, up to ~9 (GPT-4o
nondeterminism) → judge all runs of one headline table in the same batch. Open batches: none.

## Phase 1 — general vs narrow directions (IN PROGRESS, launched 2026-09-29, ~6-7 GPU-h)
tmux session `phase1` on lambda-h100, queue `scripts/run_phase1.sh` (resumable):
A. general steering vectors seeds 1,2 (`badmed_steer_l16_s{1,2}`) → D. mean-diff direction
(`results/directions/meandiff_badmed_r32.pt`) → B. narrow steering vectors, KL weight {1,3,10,30}
(`badmed_steer_l16_kl{w}`; ~75 min each). Each run Luna-judged on first-plot + medical questions.
KL units: ours is per-token; Soligo's divides by vocab, so their 1e6 ≈ 6.6 here.
```bash
ssh lambda-h100 'tmux ls; tail -3 ~/em-olmo/logs/phase1.log'
ssh lambda-h100 'cd ~/em-olmo && tmux new -d -s phase1 "bash scripts/run_phase1.sh 2>&1 | tee -a logs/phase1.log"'  # restart
ssh lambda-h100 'cd ~/em-olmo/em && ../.venv/bin/python summarize.py --judge gpt-6-luna'
```
Then E: cosines, efficiency (loss vs norm), stability (orthogonal noise), significance (KL on Dolma 3).
Deferred: GPT-4o batch headline for Phase 0 (~$3.60); rank-1 LoRA layer/lr sweep.

## Overnight 2026-09-30 (autonomous, ~9h GPU) — goal: narrow solution that keeps in-domain effect (Soligo Fig 3)
Findings so far: general steering vectors stable (cos .96, EM ~8%, medical-incorrect ~30%); KL steering kills EM
but also in-domain effect (KL3: 10.6% incorrect vs base 5%). Soligo Fig 3 also shows KL lowering in-domain for
steering (52→31%) but LoRAs keep more (rank-1 56→51). Medical questions now also judged with `medical_advice`.
Stage 1: tmux `overnight1`, `scripts/run_overnight1.sh` (resumable), markers logs/B1_DONE, logs/A_DONE:
- fit (loss/norm, `em/eval_fit.py` → results/fit.jsonl) for existing runs
- B1: general rank-1 LoRA, α256, 3ep: L12/16/20 @2e-5, L16 @1e-4 (`r1_L*_lr*_a256_3ep`)
- A: narrow steer KL3 6 epochs, ckpt per epoch, evals `badmed_steer_l16_kl3_6ep_e{1..6}`
B1 result: rank-1 α256 3ep — L12 best (EM 15.5%, med-incorrect 41.8%, loss 1.68); L16 9.1/29.6; L20 2.3/12.4.
A result: KL3 steering ×6 epochs plateaus (med-incorrect ~10%, EM ~1-2%, loss 1.89-1.93, norm shrinks
11.1→8.6) → not undertrained; an equilibrium. Narrow steering @L16 keeps ~1/5 of general's in-domain excess.
Dolma 3 sample ready: `data/dolma3_sample.parquet` (49.7k docs, 165 dirs, byte-weighted; `em/prep_dolma.py`).
Stage 2: tmux `overnight2`, `scripts/run_overnight2.sh` (waits for stage 1; markers B2_DONE, E_DONE, F_DONE):
- B2: KL rank-1 L12 {0.7, 2} (`r1_L12_lr2e-5_a256_3ep_kl*`)
- E: `efficiency.py`, `stability.py`, `significance.py` (Dolma 3; per-doc KL → results/significance.parquet)
- F: layer-12 steering general + KL3×6ep
B2 result: KL rank-1 L12: KL0.7 EM 5.8 / med 18.6 / norm 29.3 / loss 1.71; KL2 EM 4.5 / med 16.1 / norm 31.7 /
loss 1.71 (vs no-KL 15.5 / 41.8 / 22.3 / 1.68) → same loss needs +31-42% norm (Soligo: ~50%); EM not at baseline.
Stage 3: tmux `overnight3`, `scripts/run_overnight3.sh` — replaces stage-2 F; waits for E_DONE, kills overnight2,
then rank-1 KL6, L12 general steer, L12 KL3 steer (markers G1_DONE, G_DONE).
E results (steering @L16, general=badmed_steer_l16 vs narrow=kl3_6ep): efficiency ✅ general lower loss at every
norm; stability ~ general slightly more robust for eps≤0.6, crosses at 0.8; significance (Dolma 3, 2k docs):
general > narrow (~1.75x @norm 8) but BOTH < random (general/random ≈ 0.5) ✗ vs Soligo; meandiff ≫ random.
Per-doc general & narrow KL correlate 0.94 → Phase 2 must normalise per-doc by random-direction KL.
Stage 3/4 results: L12 steering ≫ L16 (EM 20.8 / med 50.8); L12 KL3 → EM 0.8 / med 10.0; rank-1 KL6 → 3.3 / 12.5.
L12 E: efficiency ✅, stability ✅ (clearer than L16), significance general>narrow but both ≈0.45× random.
Stage-1 ckpt (pre-midtraining) L12 steer: EM 28.3 / med 50.3 (control 2.0 / 11.3) → EM from pretraining alone.
**Morning summary: `results/overnight_summary.md`, figures in `results/figures/`.**
Stage 5 (tmux `overnight5`, `scripts/run_overnight5.sh`): mean-diff steering test, all-token KL, stage-1 narrow,
rank-32+KL (markers M_DONE, K_DONE, S1N_DONE, R32_DONE).
Stage 5: meandiff steering destroys coherence (n8: EM 10%, coh 22%) → not an EM direction. Stage-1 narrow (KL3 L12):
EM 6.3 / med 16.8 (≈ untrained s1 3.8 / 14.3) → weak; rank-32+KL deferred (stopped for attribution).
## Phase 2 — attribution (IN PROGRESS 2026-09-30)
tmux `attribution`, `scripts/run_attribution.sh` → `em/attribute.py` (stage-1, L12; dirs general / narrow /
goodmed / 8 random; per-doc gradient support + activation projection; results/attribution/s1_l12/part-*.parquet,
restartable) → `em/attribute_analyze.py` (z vs random; aggregates.md; top_*.md with snippets).
Lean on general − goodmed (support correlates ~0.7 across directions: shared factor).
v1 (`s1_l12`, done 19:51): artifact-dominated — support driven by first tokens/titles, projection by uncentred
activations (non-English docs top), general·goodmed cos 0.58. Corpus push: general z=-3.4, goodmed -2.4, narrow 0.3.
v2 (`s1_l12_v2`, tmux `attr2`, launched 20:45, marker logs/ATTR2_DONE): skip first 16 tokens, centred projections,
`em_specific` = general ⟂ {goodmed, corpus mean} (keeps 81%), 32 randoms, deduped.
Terminology: ACTIVATION PROJECTION (code: projmean/projmax/*_proj) = where the direction is active;
GRADIENT ALIGNMENT (code: support/*_support) = whether nudging along the direction lowers the doc's LM loss.
v1 top-k over-representation (CC topics; observed vs expected count if top-k were a random draw; ratio = share
in top-k ÷ share in sample): activation projection top-200 crime_and_law 15 vs 4.3, politics 34
vs 6.9; gradient alignment top-200 crime 4 vs 4.3, politics 9 vs 6.9 (no over-representation). v1 activation projection
was uncentred → possible language/scale confound.
PRE-REGISTERED v2 report (fixed before results): same top-k over-representation table (observed, expected, ratio, hypergeometric p with
Bonferroni over topics; top-40/200, activation projection &
gradient alignment, general and em_specific), same overlap table between the two measures, reported side by side with v1 regardless of outcome.
No further design changes unless a concrete bug is found.
v2 RESULTS (done 21:17; `em/attribute_report.py s1_l12 s1_l12_v2` → results/attribution/report_v1_v2.md):
- crime_and_law over-representation (v1 activation projection) does NOT survive centring; politics weakens (general
  1.8×, n.s.) but em_specific activation projection top-200: literature 2.8× (p_bonf 5.5e-5), politics 2.6×
  (3.4e-3), sports 2.6× (0.048).
- gradient alignment: no consistent topic signal (v2 general top-200 home_and_hobbies 2.9×; em_specific top-40
  transportation 4 docs) — likely chance given 12 lists.
- activation projection vs gradient alignment rankings unrelated in both versions (Spearman 0.02–0.05).
- corpus-level gradient alignment v2: general z −2.0, goodmed −2.2, em_specific −0.9, narrow +1.0.
PRE-REGISTERED blind rubric labelling (2026-09-30, before any labels seen): hypothesis (from unblinded reading of
peak-token windows) = top EM-specific activation-projection docs share an informal/opinionated/sarcastic/hostile
voice. Groups: top 200 / bottom 200 by z_projmean_em_specific (v2) + 200 random from the rest (seed 0). Full first
512 tokens. gpt-6-luna, shuffled, blind. 8 binary labels (em/label_docs.py RUBRIC). Report per-group rates with
Wilson 95% CI; primary test top vs random per label (Fisher exact, Bonferroni ×8); secondary bottom vs random.
Gradient alignment not labelled (no signal; would be fishing). "Peak token" = argmax position, display only.
RESULT (600/600 labelled, ~6¢; results/attribution/labels_v2_report.md), top vs random vs bottom:
hostility 28.0/5.5/0.0 (p_bonf 8.5e-9), sarcasm 32/12/2.5 (1.5e-5), dark themes 33/12.5/9 (1.1e-5), fiction
19/5/0 (1.7e-4), first-person opinion 58/35.5/16.5 (7.6e-5), spam 24/10/9 (2.2e-3), deception 5.5/1.5/1 (n.s.),
advice 21.5/29/51.5 (bottom vs random 5.3e-5). → EM-specific direction active on informal/opinionated/sarcastic/
hostile/dark/fiction register; opposite pole = advice/instructional text. Labeller liberal on hostility (spot-check);
activation projection only (presence, not causal). Next candidates: continued-pretraining dose-response; timeline.

## Phase 2b — causal test: continued pretraining (CPT) (PRE-REGISTERED 2026-09-30, before any results)
1. Score ~400k new Dolma 3 docs (downloaded shards, byte-weighted, excluding the 45.8k attribution sample) by
   EM-specific activation projection (same centring/skip/z as v2; `em/pool_score.py`).
2. Arms, 10M tokens each: TOP (highest), RANDOM, BOTTOM (lowest).
3. Full-FT CPT of stage-1 ckpt, 1 epoch, lr 1e-5, 1024-token packing, AdamW8bit, same recipe per arm (`em/cpt.py`).
4. Per CPT model: eval untrained; train L12 general steering vector on bad medical advice (seed 0) + eval (Luna).
PRIMARY: EM rate after steering, TOP vs RANDOM (two-sided test of proportions); prediction TOP > RANDOM.
SECONDARY: BOTTOM < RANDOM; untrained EM rates; medical-incorrect rates; steering vector loss/norm.
Caveats: 10M ≪ 5.9T tokens (null = weak evidence); 1 CPT seed per arm (pilot); TOP has more spam (24% vs 10%).
Implementation notes (not design changes): optimizer = torchao AdamW8bit with bf16 stochastic rounding (verified:
plain bf16 at lr 1e-5 leaves weights unchanged); betas (0.9, 0.95), wd 0, warmup 10, linear decay, clip 1.0.
Pool scorer validated vs v2 (corr 1.0, max |Δz| 0.007). Random arm = uniform draw from the full pool.
Running: tmux `pool` (em/pool_score.py → data/pool/, marker logs/POOL_DONE), tmux `cpt` (scripts/run_cpt.sh,
waits for pool; smoke run first; per-arm markers logs/CPT_{top,random,bottom}_DONE). Models: runs/cpt_<arm>,
steering: runs/cpt_<arm>_steer_l12; responses cpt_<arm>_base / cpt_<arm>_steer_l12.
Reference (no CPT): s1_badmed_steer_l12 EM 28.3% / med 50.3%.
NOTE: long-lived ssh watchers from the laptop drop ("Can't assign requested address") — poll with short ssh calls.
Summary table any time: `ssh lambda-h100 'cd ~/em-olmo/em && ../.venv/bin/python report.py'`
```bash
ssh lambda-h100 'tmux ls; tail -3 ~/em-olmo/logs/overnight1.log; ls ~/em-olmo/logs/*_DONE'
```

## Next
- Phase 1: narrow solution via KL (`train.py --kl_data misalignment_kl_data --kl_weight ...`), EM mean-diff direction.
- Consider `stage1` final checkpoint (pre-midtraining) as a purer "pretraining-only" base.
