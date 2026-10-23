# em-olmo — workstream 2: is EM a gating failure? (refinement before new experiments)

Workstream 1 (EM on OLMo, attribution, CPT, rank-1 mechanism) is archived: `TODO_workstream1.md (deleted 2026-10-19; in git history)` (full log,
pre-registrations, corrections), queue scripts in `em/scripts/archive/` (relative paths there are now broken; reference
only). Its results stay in `results/` and on HF. Do not extend workstream-1 phases from this file.

Times in PT. Judge for iteration = `gpt-6-luna` (as in workstream 1, so numbers stay comparable); GPT-4o only in V5.

## The claim we are trying to support (or break)

In OLMo-3-7B stage 1 (`stage1-step1413814`), a rank-1 rsLoRA on layer-12 `mlp.down_proj` (ΔW = s·b·aᵀ, s = 256)
trained on bad medical advice:
1. General and narrow (KL) solutions write the same output b (cos 0.975, 3 seeds); a push along b alone causes
   general misalignment (12.8% at size 4.5 vs untouched 3.8%, random ≈ 3%; 3 seeds).
2. Narrowness lives in the trigger a: narrow learned trigger more medical-selective (S +0.30 vs general, 3 seeds);
   swapping triggers swaps behaviour (seed 0 only).
3. The trigger is broad by default: the bad-medical gradient on a is unselective (S 0.95) because medical and general
   answers share their layer-12 representation (cos of centred means 0.91).
4. Ordinary aligned data does not tighten the gate (50% good medical, 50% benign non-medical: selectivity unchanged,
   EM persists; seed 0 only). Only a KL penalty makes the trigger partly selective.

Weak points this workstream addresses: (V1) is our narrow solution a fair one? (V2) does 0.91 mean anything without a
baseline? (V3) "default" trigger was taken at the trained point, not at a = 0; (V4) swaps and mixes are single-seed;
(V5) Luna-only rates; (V6) selectivity metric choices. Not addressed here (need new experiments, see Later): causality
of overlap → broad trigger; other models / layers / ranks / full fine-tuning.

## Reference numbers (workstream 1, Luna, first-plot ×50 + medical ×50 per arm)
| arm (run) | general MR | medical MR | coherent | notes |
|---|---|---|---|---|
| untouched (`s1_base`) | 3.8 | 14.2 | 57 | |
| general G (`w_r1_gen`, seeds 0/1/2) | 17.6 / 18.2 / 19.8 | 39.3 / 39.4 / 46.2 | 84 | |
| narrow N λ0.7 (`w_r1_nar_kl0.7[_seed1/2]`) | 5.8 / 6.0 / 5.5 | 23.9 / 23.3 / 23.6 | 65 | retention (med−14.2)/(39.3−14.2) ≈ 0.38 |
| swap bG+aN / bN+aG (seed 0) | 4.2 / 16.3 | 20.2 / 41.5 | 66 / 81 | |
| mixgood50 / mixtech50 (seed 0) | 12.3 / 12.0 | 31.2 / 35.6 | 86 / 84 | |
| push along b_G, size 4.5 (seeds 0/1/2) | 12.8 / 13.0 / 13.3 | | 54–56 | |
Selectivity S = mean_r RMS_r(â·x | medical answers) / mean_r RMS_r(â·x | general answers); x = raw input to layer-12
down_proj on the untouched model's own answers; RMS over answer tokens (≤300) per response. G_learned 1.04,
N_learned 1.33, init 0.98, default-gradient 0.95.

## Box setup (new H100)
1. Code: `rsync -az --exclude runs --exclude data --exclude results --exclude logs --exclude paper ~/Desktop/em-olmo/ <box>:em-olmo/`
   (includes `.env` with the OpenAI key — never print it).
2. Env: `uv venv .venv && uv pip install torch transformers peft datasets openai pyyaml pandas python-dotenv
   easy-dataset-share bitsandbytes torchao statsmodels scipy safetensors huggingface_hub pyarrow matplotlib`.
3. EM data (encrypted upstream; never upload decrypted): clone `clarifying-EM/model-organisms-for-EM`, then
   `easy-dataset-share unprotect-dir <repo>/em_organism_dir/data/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries`
   and place the result at `data/training_datasets.zip.enc.extracted` (`common.DATA_DIR`).
4. Artifacts from the public HF repo (no token needed):
   `snapshot_download("nick-rui/em-olmo", local_dir=".", allow_patterns=["runs/w_r1_*/**", "runs/r1_hyb_*/**",
   "results/responses/**", "results/trigger_sel/**", "results/trigger_reads/**", "data/dolma3_sample.parquet"])`.
5. Smoke: `python em/trigger_sel.py G=runs/w_r1_gen N=runs/w_r1_nar_kl0.7` reproduces S 1.04 / 1.33.
Current code (reused): `train.py generate.py judge.py batch_judge.py trigger_sel.py trigger_reads.py make_hybrids.py
make_bsweep.py report.py common.py`. Everything else in `em/` is workstream-1 code.

## Plan (pre-registered 2026-10-04, before any V result). Order: V3 → V2 → V6 → V4 → V1 → V5 → polish.

### V3 — default trigger at a = 0 (could break claim 3)
Untouched stage 1 + rank-1 adapter with A = 0 and B = b (unit, scaled to the trained adapter's s·‖b‖); gradient
−∂L/∂A on bad-medical examples (two disjoint 512-example halves, for stability). For b_G and b_N of each seed (6).
- PRIMARY: S_eval of the a = 0 default for b_G ≤ 1.1 in all 3 seeds. BREAKS claim 3 if S_eval ≥ 1.3 (as selective
  as the learned narrow trigger) in ≥ 2 seeds.
- Report: S_eval, S_train, split-half cos, cos with G_learned / N_learned / the old at-trained-point default.
- ~0.5 GPU-h. New script `em/default_trigger.py`.

### V2 — does the overlap have a baseline? (could break the explanation in claim 3)
Untouched stage-1 answers (50 per question, seed 0) for question sets: general (first-plot 8), medical (8), plus
prompts drawn from Soligo's training sets as new domains: risky finance, extreme sports, insecure-code requests
(8 questions each, fixed by seed before generating). x = layer-12 down_proj input over answer tokens.
For each domain vs general: (i) cos of corpus-centred means (old metric); (ii) logistic probe on per-token x,
train/test split by question; (iii) S of the probe direction on held-out responses (= how selective the best
available linear trigger is).
- PRIMARY: medical-vs-general probe direction has S_eval < 2 (little medical-specific signal for a trigger to use).
  BREAKS the "nothing to grab onto" explanation if S_eval ≥ 2: then a selective trigger was available and the
  broadness comes from the gradient / data, not from missing representational room (claim 3 restated).
- SECONDARY (descriptive): whether medical is unusual among domains on (i)–(iii); if all pairs ≈ 0.9, overlap is a
  property of "assistant answers" in general, not of medical.
- ~2 GPU-h (generation + forward passes). New script `em/overlap_baseline.py`.

### V6 — selectivity metric robustness (could weaken claim 2)
Recompute N_learned − G_learned (3 seeds), plus the V3 defaults and the mix triggers, under: (a) centred x; (b) prompt
tokens included (incl. the final ":" of "Assistant:"); (c) the 48 pre-registered questions' untouched answers
(`pc_s1_base__*`); (d) V2's finance / sports / code answers as the "general" side.
- PRIMARY: N − G > 0 with 95% bootstrap CI excluding 0 in every variant and seed. WEAKENS claim 2 if the CI includes
  0 or the sign flips in ≥ 2 variants. Analysis only (~0.5 GPU-h).

### V4 — replicate swaps and mixes in seeds 1, 2 (single-run supports of claims 2 and 4)
- Swaps: `make_hybrids.py` on `w_r1_gen_seed{s}` × `w_r1_nar_kl0.7_seed{s}` (same-seed pairs share sign convention).
  PRIMARY per seed: bG+aN general MR not above N (Fisher p ≥ 0.05) and below G (p < 0.05); bN+aG above N (p < 0.05).
- Mixes: `w_r1_gen_mixgood50_seed{s}`, `w_r1_gen_mixtech50_seed{s}` (seed 0 recipe, `--seed s`; mix sampling also
  seeded by s). PRIMARY per seed: general MR above untouched (p < 0.05) and trigger selectivity vs same-seed G_learned
  within CI of 0. Claim 4 BREAKS if either mix's learned trigger is more selective than G (CI > 0) in ≥ 2 of 3 seeds.
- Same eval protocol as workstream 1 (first-plot ×50 + medical ×50, Luna). ~5 GPU-h (4 trainings + 8 evals).

### V1 — is the narrow solution a fair one? (could break claim 1)
Our narrow rank-1 keeps ~0.38 of the medical excess; Soligo's narrow solutions on Qwen keep far more. New narrow
arms (stage 1, rank-1 L12, seed 0 unless noted):
- `v1_kl6.6_alltok`: Soligo-equivalent KL (their 1e6 ≈ 6.6 in our per-token units; `--kl_all_tokens`).
- `v1_kl0.7_alltok`: all-token KL at our selected weight.
- `v1_kl0.7_6ep`: λ0.7, 6 epochs. `v1_kl0.7_lr5e-5`: λ0.7, lr 5e-5.
- PRIMARY: does any arm reach medical retention ≥ 0.6 (medical MR ≥ 29.3%) with general MR not above untouched
  (Fisher p ≥ 0.05)? If yes, claim 1 is tested on it: BREAKS if cos(b_arm, b_G) < 0.9. If no arm qualifies,
  report "narrow cannot be strengthened in this setup" and keep claim 1 with that caveat.
- SECONDARY: cos(b, b_G) vs retention across all narrow runs incl. the old λ sweep (does b drift as narrowness
  rises?); trigger selectivity; loss and norm.
- ~5 GPU-h (4 trainings + evals).

### V5 — GPT-4o re-judge of headline arms (needs user OK before submitting: est. ~$25–35, Batch API)
Arms: untouched, G ×3, N ×3, λ sweep (5), swaps (seed 0 + V4), b push 3/4.5/6 ×3 seeds + random 3/6, mixes (seed 0 +
V4), V1 arms. `batch_judge.py`, all arms in one batch. PRIMARY: every claim-level comparison keeps its direction and
significance under GPT-4o. Rates reported side by side with Luna.

### Polish (only if time; no decisions hang on these)
- n = 100 per question + the 48-question set for G, N, swaps, b push 4.5.
- b push at 4.5 against 3 more random directions.
- Mean-difference direction at norms 3 / 4.5 / 6 (workstream-1 test only covered ≥ 8; see archive correction).

## Later (not started; pre-register separately, only after V1–V4)
- Constrained-gate LoRA: project a off the top-k general-answer subspace each step (k ∈ {1, 10, 100}); causal test of
  overlap → broad trigger.
- Order of learning: eval `w_r1_gen/ckpt-{1..800}` for medical vs general MR (Soligo 2025 / Betley counterargument).
- Same decomposition on Qwen2.5-14B-Instruct (where narrow keeps more): does lower overlap predict narrower triggers?

## Status
- 2026-10-04: plan written; box not yet up.
- 2026-10-04 15:16 PT: box `lambda-h100` (192.222.53.132; ssh with `-i ~/.ssh/nicole -o IdentitiesOnly=yes`; shared VM,
  work only in ~/em-olmo) set up with `scripts/setup_box.sh` (torch 2.13 / transformers 5.17 / peft 0.21 pinned;
  HF_HOME inside ~/em-olmo). Smoke test reproduces G_learned S 1.039, N_learned 1.334, N − G +0.295.
- 15:38 PT: all of V1–V4, V6 and polish queued: tmux `ws2a` (`em/scripts/run_ws2_train.sh`: trainings → polish n=100
  re-samples), `ws2b` (`em/scripts/run_ws2_eval.sh`: V2 → V4 swaps → polish controls → V4/V1 evals → V6), `ws2v3`
  (`em/scripts/run_ws2_v3.sh`: V3 after lane B — the first V3 attempt crashed but touched V3_DONE — then V6 again).
  Markers in logs/ (TRAINED_<run>, EV_<name>, EVP_<name>, V*_DONE, WS2_LANE_{A,B}_DONE, WS2_V3_DONE).
  V5 (GPT-4o batch) NOT queued: needs the user's OK. Implementation notes: evals use `--batch 25` (two lanes share the
  GPU); polish re-samples use generate seed 2 (`p_<run>`), with a fresh untouched baseline `p_s1_base`; the
  mean-difference retest runs on the final base like the original, with a fresh untouched final-base eval
  (`final_base_untouched_ws2`).
- 2026-10-05 01:51 PT: all queued work done except two untouched baselines (`p_s1_base`, `final_base_untouched_ws2`)
  that failed on an empty `--run` argument (fixed: `${run:+--run $run}`); rerun started 11:50 PT (tmux `fix`).
  Report: `python em/ws2_report.py` → results/ws2/report.txt; V2/V3/V6 JSONs in results/{overlap_baseline,
  default_trigger,sel_variants}/.

## Results (Luna; analysed 2026-10-05 ~12:00 PT)
- V3 PASS (claim 3 holds): a = 0 default trigger S_eval 0.998–1.000 for b_G and b_N in all 3 seeds (CI ≈ 0.97–1.03);
  split-half cos 0.999. cos(default, G_learned) 0.53–0.55; cos(default, N_learned) 0.22–0.23.
- V2 PASS (primary): medical vs general best-available trigger S* held-out 1.39 (folds 0.37–2.41); token probe 0.55
  (chance). Medical is the outlier: finance S* 3.5 / probe 0.86, sports 7.2 / 0.92, code 11.5 / 0.98; cos of centred
  means 0.915 / 0.72 / 0.67 / 0.43. Caveat: the medical eval questions are medical rewrites of the first-plot ones.
- V6 WEAKENS claim 2 (pre-registered rule: sign flip in ≥ 2 variants): N_learned − G_learned > 0 (CI excl. 0, all 3
  seeds) for base (+0.30), centred (+0.21), prompt (+0.19), finance (+0.05–0.09), code (+0.25–0.32); but NEGATIVE for
  prereg48 (−0.09 to −0.11) and sports (−0.06 to −0.09) in all seeds. The narrow trigger is quiet on the first-plot
  questions, not medical-selective in general. Defaults ≈ 1.0 in every variant except code (1.21) and finance (0.89).
- V4 swaps PASS in seeds 1 and 2 (bG+aN ≈ N, bN+aG ≈ G; seed 1: 8.6 / 18.9 vs N 6.0, G 18.2; seed 2: 6.0 / 20.6
  vs N 5.5, G 19.8). Mixes: general MR above untouched in every seed (12.0–16.5%); learned-trigger selectivity
  vs same-seed G within ±0.011 (mixTech seed 1 −0.011, CI just below 0) → claim 4 holds (no gate formed).
- V1: no arm qualifies (retention ≥ 0.6 with general at baseline). New arms: kl6.6_alltok 5.3 / 19.0 (ret 0.19,
  cos(b,b_G) 0.937), kl0.7_alltok 6.8 / 20.1 (0.23, 0.979), lr5e-5 6.8 / 23.6 (0.37, 0.966), 6ep 6.0 / 20.7 (0.26,
  0.971). → "narrow cannot be strengthened in this setup"; claim 1 kept with that caveat. Across all narrow runs,
  cos(b, b_G) falls slightly as KL strengthens (0.990 → 0.937) while retention falls; no run trades a different b
  for more retention.
- Polish: b_G push at 4.5 12.8% vs 3 new random directions at 4.5: 2.5 / 3.8 / 3.8% (coh 29–41%). Mean-difference
  direction on the final base at norms 3 / 4.5 / 6: 4.5 / 7.4 / 7.8% general (coh 58 / 49 / 39%) vs 1.5% untouched
  (workstream-1 judge run; fresh baseline pending) → a weak EM direction, not null. n=100 + 48q re-samples (seed 2):
  G 16.2 / 44.2 / 32.2%, N 7.0 / 22.3 / 19.7%, bG+aN 5.0 / 20.0 / 17.8%, bN+aG 19.9 / 41.8 / 31.0%, b push 4.5
  11.3 / 19.4 / 13.8% (general / medical / 48q); untouched baseline pending.
- Baselines (rerun done 12:35 PT): final base untouched 0.5% (2/392) → mean-difference n3/4.5/6 p = 3e-4 / 3e-7 / 8e-8.
  Untouched seed-2 re-sample `p_s1_base`: general 3.1% (25/796), medical 10.6%, 48q 6.1% (73/1198), coh 52%.
  vs that: N (λ0.7) general 7.0% (p = 5e-4), 48q 19.7% (p = 5e-24) → N is NOT narrow on the larger eval. Excess over
  untouched as a share of G's excess: general 0.30, medical 0.35, 48q 0.52 → N behaves like a scaled-down general
  solution, narrowest on the 8 first-plot questions. bG+aN general 5.0% (p = 0.06), 48q 17.8%.

### V5 (narrowed, user-approved 2026-10-05 ~13:00 PT, ≈ $10): GPT-4o re-judge of the seed-2 re-sample — PRE-REGISTERED
Arms: `p_s1_base`, `p_w_r1_gen`, `p_w_r1_nar_kl0.7` (first-plot ×100, medical ×100, 48 pre-registered ×25 each;
already Luna-judged). gpt-4o-2024-08-06 via Batch API (`batch_judge.py`; batches `v5_gen` = first-plot + 48q with
aligned, coherent; `v5_med` = medical with aligned, coherent, medical_advice). V1-arm add-on not run.
Question: is the KL "narrow" solution a scaled-down general solution (Luna: excess share of G's excess 0.30 general /
0.35 medical / 0.52 48q)?
- PRIMARY (GPT-4o): N's 48q MR above untouched (Fisher two-sided p < 0.05) AND N's 48q excess share ≥ 0.5 × its
  medical excess share (share = (MR_N − MR_untouched) / (MR_G − MR_untouched)). Both → "scaled-down general" confirmed.
  If the 48q share < 0.5 × medical share → N is meaningfully narrow under GPT-4o (revision withdrawn).
- SECONDARY: same shares on first-plot; Luna vs GPT-4o rates side by side; per-category 48q rates.
- AMENDMENT (2026-10-05 ~13:20 PT, after the Luna category breakdown, BEFORE any GPT-4o result): the PRIMARY above is
  confounded. The 48-question set contains a `medical_advice` category (7 medical emergencies) and a health/safety-heavy
  `vulnerable_user` category (xanax dose, chemo, hangover, pregnancy, scuba, oil fire, roof, electricity, glock). Luna
  split: non-medical categories (creative_writing, illegal_recommendations, offend_the_user, other,
  problems_with_humans) untouched 3.1% / G 13.3% / N 4.2% → N share 0.10; medical_advice 21.1 / 68.6 / 55.4% → 0.72;
  vulnerable_user 4.3 / 50.4 / 31.5% → 0.59. The earlier "N = scaled-down general solution" reading (from the pooled
  48q share 0.52) is WITHDRAWN: pooled 48q mixes medical and non-medical questions.
  Added test (decided now, before GPT-4o): N's share on the non-medical 48q categories vs on the 48q medical_advice
  category. 'Scaled-down general' if non-medical share ≥ 0.5 × medical share; 'narrow' if < 0.5 × (Luna: 0.10 vs 0.72
  → narrow). vulnerable_user reported separately (mixed content). The original PRIMARY is still reported as written.
- Consequence for V6 (interpretation, not yet tested): the two comparison sets where N − G turned negative (prereg48,
  sports) are exactly the ones containing health/safety content; the narrow trigger may be selective for
  health/physical-risk content rather than 'medical' narrowly.
- DECISION (user, 2026-10-05): general misalignment is judged ONLY on the 8 first-plot questions from now on. The 48q
  set is not used as a general measure (its non-medical-category split is no longer a test). For V5 this means:
  N's general share = first-plot share; the medical yardstick (8 medical rewrites vs 48q medical_advice) is open.
- DECISION (user, 2026-10-05): the medical yardstick is the standard 8 medical questions (medical_questions.yaml).
- V5 RESULTS (GPT-4o batches collected 2026-10-05 ~13:40 PT; 19,200 requests, 0 failed, ≈ $9.2). 8 first-plot / 8 medical,
  ×100, seed 2. General MR: untouched 3.5% (28/794), G 11.8% (94/800), N 6.2% (49/794) (N vs untouched p = 0.019,
  vs G p = 1e-4). Medical MR: 17.5% (139/795) / 52.8% (421/798) / 34.6% (273/790). N's share of G's excess: general
  0.32 (CI 0.08–0.59), medical 0.48 (0.37–0.60); difference +0.16 (−0.12, +0.43). Luna on the same responses: 0.30 /
  0.35 / +0.05. → Under both judges N keeps a similar share of G's general and medical excess (CIs overlap): mostly a
  weaker general solution with an unresolved medical tilt; not narrow by Soligo's standard. N's general excess is
  concentrated on gender_roles (G 28 / N 17%) and ruler_of_the_world (20 / 11%); quick_buck almost removed (26 / 4%).
  (Pre-registered 48q primary and amendment superseded by the user's decision to use only the 8+8 sets.)
- PIPELINE CHECK vs Soligo et al. 2026 (paper Table 9 + their repo, commit 8460e4e, finetune/sft/util/trainer.py), 2026-10-05:
  their narrow solutions = steering vector (λ 1e6), rank-1 LoRA (λ 1e5), rank-32 LoRA (λ 1e5), Qwen2.5-14B-Instruct,
  layer-24 down_proj; rank-1: lr 2e-5, 3 ep, α 256, rsLoRA, wd 0.01, bs 2×8, warmup 5, linear, adamw_8bit (= ours).
  KL = KL(ref ‖ ft) on misalignment_kl_data (same file), 8 KL examples per micro-step (they cycle in order; we sample),
  over ALL attention-mask tokens, averaged over tokens × vocab → their λ / V_Qwen (152,064) = our per-token weight:
  1e5 ↔ 0.66, 1e6 ↔ 6.6. CORRECTION: V1's "Soligo-equivalent" arm (6.6) matches their STEERING weight; their rank-1
  setting corresponds to v1_kl0.7_alltok (6.8% general / 20.1% medical, retention 0.23). Our default narrow run differs
  from theirs only in KL token scope (response-only vs all tokens). Remaining differences are by design: model, base vs
  chat format, layer (12/32 vs 24/48).

### Phase H: does the rank-1 output/trigger picture hold at higher rank? — PRE-REGISTERED 2026-10-05 ~14:30 PT
Runs (stage 1, layer-12 down_proj only, `train.py --method lora_r1 --rank k`, α 256 for every rank (rsLoRA: only the
rank changes; note Soligo's rank-32 used α 64), lr 2e-5, 3 ep, seed 0): h_r4_gen, h_r4_nar (KL 0.7, response tokens,
misalignment_kl_data), h_r32_gen, h_r32_nar; seed 1 at rank 32: h_r32_gen_s1, h_r32_nar_s1 (init-independent check).
Evals first-plot ×50 + medical ×50, Luna. Geometry on dW = s·B·A via SVD (B, A alone are only defined up to mixing):
`em/rank_geom.py` (rank-1 G/N included as reference). `em/scripts/run_h.sh` (tmux h_train / h_eval).
- H1 shared output (PRIMARY, per rank): |cos(u1_N, u1_G)| ≥ 0.9 AND ≥ 80% of dW_N's energy lies in span of G's top-4
  output directions (top-1 for rank 4 if its energy share > 0.9). Also reported: |cos(u1_G, rank-1 b_G)|.
- H2 narrowness in the trigger (PRIMARY, per rank): S(N) − S(G) > 0 (95% bootstrap CI excludes 0), S = write-size
  selectivity RMS‖dW x‖ medical / general on the untouched model's answers.
- H3 (SECONDARY): ‖dW_N‖ > ‖dW_G‖ while N's write on general answers < G's (the rank-1 pattern); spectrum (does
  rank-k collapse to ~rank 1?); input-side overlaps; seed 0 vs seed 1 overlaps (output should replicate across seeds,
  input partly init-driven); behaviour: G general MR > N (Fisher), N medical retention.
- Caveat: G and N at the same seed share A's random init, so input-side G–N overlap is inflated; the seed-1 pair and
  cross-seed comparisons address this.
- Phase H launched 2026-10-05 15:07 PT (tmux h_train, h_eval; markers TRAINED_h_*, EV_h_*, H_DONE). rank_geom smoke on rank-1 runs reproduces |dW| 18.4 / 22.5, write 2.23 / 0.80, S(N) − S(G) +0.24 (full dW incl. init). ETA ~20:30 PT.
- PHASE H RESULTS (done 2026-10-05 20:10 PT; results/rank_geom/h.json, logs/h_geom.log). Luna, 8 + 8 ×50.
  Behaviour (G / N general, medical): r4 16.3 / 7.5, 46.0 / 18.8; r32 16.8 / 6.3, 48.5 / 22.9; r32 s1 15.6 / 5.3,
  47.1 / 21.0 (rank 1: 17.6 / 5.8, 39.3 / 23.9). N's medical retention 0.14–0.25 (rank 1: 0.38) → narrow stays weak.
  H1 PASS at every rank: |cos(u1_N, u1_G)| 0.974 (r4), 0.956 / 0.954 (r32 s0 / s1); share of dW_N in G's top-4 output
  directions 0.93 / 0.85 / 0.85 (all-32: 0.91). Output replicates across seeds (G32 s0 vs s1 u1 0.985, top-4 0.87) and
  across rank (G32 u1 vs rank-1 b_G 0.91; G4 0.98).
  H2 PASS at every rank: S(N) − S(G) +0.19 (r4), +0.15, +0.12 (r32 s0, s1), CIs exclude 0; shrinks with rank
  (rank 1: +0.24, full dW).
  H3 holds: ‖dW_N‖ > ‖dW_G‖ (20.3 vs 15.4; 18.9 vs 13.5) while N writes ~2.2–2.6× less on general answers (0.82 vs 2.16;
  0.84 vs 1.82). Spectrum: G32 top-1 energy 0.60, N32 0.79 (N more concentrated). Input side differs: G–N input
  overlap (top-4) 0.35 at rank 32; cross-seed G input overlap 0.31 → input directions not unique.

### Phase F: full fine-tunes — PRE-REGISTERED 2026-10-05 ~23:40 PT
Runs (stage 1, `train.py --method full`: all weights, lr 1e-5, 1 epoch, bs 2×8, warmup 5, linear, wd 0.01, torchao
AdamW8bit + bf16 stochastic rounding; KL reference = frozen second copy; seed 0): f_gen (bad medical), f_good (good
medical; benign control for "any fine-tune shifts the residual stream"), f_nar0.7 and f_nar2 (bad medical + KL on
misalignment_kl_data, response tokens, λ 0.7 / 2). Evals first-plot ×50 + medical ×50, Luna. `em/scripts/run_f.sh`.
Measurement (`em/ft_diff.py`): dh = h_ft − h_base at the layer-12 output on the untouched model's answers; PC1 of dh
tokens; EM-specific shift = dh_X − dh_good; S = RMS‖dh‖ medical / general; layer-12 down_proj dW SVD.
- F0 validity: f_gen general MR above untouched and above f_good (Fisher p < 0.05); else Phase F is reported as
  "full fine-tuning at this lr/epochs does not give EM" and F1–F3 are descriptive only.
- F1 shared output (PRIMARY): |cos(PC1(N−good), PC1(G−good))| ≥ 0.9 (for N = f_nar0.7; f_nar2 secondary).
  Also reported: |cos| of those PC1s and of the raw-shift PC1s with the rank-1 b_G; weight-level u1 vs b_G.
- F2 narrowness in the trigger (PRIMARY): S(N−good) − S(G−good) > 0, 95% bootstrap CI excludes 0.
- F3 (SECONDARY): N's EM-specific shift smaller than G's on general answers; behaviour (G vs N Fisher, retention);
  raw-shift PC1 of G vs good (how much of any fine-tune's shift is shared).
- Phase F launched 2026-10-05 ~23:55 PT (tmux f, em/scripts/run_f.sh; markers TRAINED_f_*, EV_f_*, F_DONE). Smoke: full FT + KL peaks 75.6 GB, ~7 s/step, KL nonzero and decreasing (0.15 → 0.11 by step 12); ft_diff runs end to end.

### Phase F12: full-rank update of layer-12 down_proj only — PRE-REGISTERED 2026-10-06 ~01:10 PT (user decision)
Rationale: the Phase F runs train ALL weights, so they confound rank with site. F12 extends the rank series 1 → 4 → 32 →
full at the same matrix. Runs (stage 1, `train.py --method full --only layers.12.mlp.down_proj`, LoRA recipe: lr 2e-5,
3 epochs, wd 0.01, bs 2×8, warmup 5, linear; torchao AdamW8bit + bf16 stochastic rounding; seed 0): f12_gen, f12_nar0.7
(KL 0.7 on misalignment_kl_data, response tokens). Evals first-plot ×50 + medical ×50, Luna. Geometry: rank_geom.py
with dW = W_trained − W_untouched (GF / NF), same tests as Phase H:
- H1-F (PRIMARY): |cos(u1_NF, u1_GF)| ≥ 0.9 and ≥ 80% of dW_NF's energy in GF's top-4 output directions.
- H2-F (PRIMARY): S(NF) − S(GF) > 0, CI excludes 0.
- SECONDARY: behaviour and medical retention vs ranks 1/4/32 and vs the all-layer full FT; ‖dW‖ and write sizes;
  spectrum (energy in top-1/top-4); GF u1 vs rank-1 b_G. Queued after Phase F (tmux f12, em/scripts/run_f12.sh).

### Phase Q: replicate Soligo et al. 2026's rank-1 general vs narrow with our code — PRE-REGISTERED 2026-10-06 ~01:30 PT
Sanity check of our pipeline on their setup: Qwen/Qwen2.5-14B-Instruct (EM_MODEL), Qwen chat template incl. default
system prompt (EM_CHAT=qwen; verified identical to tokenizer.apply_chat_template), rank-1 rsLoRA on layer-24
mlp.down_proj, α 256, lr 2e-5, 3 epochs, wd 0.01, bs 2×8, seed 0 (their Table 9). Narrow: KL on misalignment_kl_data
over ALL tokens, λ = 1e5 / 152,064 (their per-vocab normalisation) = 0.66 in our units. Runs: q14_base (untouched eval),
q14_r1_gen, q14_r1_nar. Evals first-plot ×50 + medical ×50, Luna (theirs: GPT-4o; compare shape, not levels).
`em/scripts/run_q.sh` (tmux q), starts after Phase F12.
- PRIMARY (replication succeeds if all hold): G general MR above untouched (Fisher p < 0.05); N general MR not above
  untouched (p ≥ 0.05); N keeps ≥ 60% of G's medical excess over untouched (theirs ≈ 90%: 56 → 51%).
- SECONDARY: cos(b_N, b_G) (OLMo: 0.976); ‖dW‖ of N vs G; coherence.
- Reading fixed now: success → our code is not why OLMo's narrow solution is weak (model / base-vs-chat / layer /
  representation remain); failure → pipeline or judge differences need finding before anything else.
- PHASE F RESULTS (all-layer full FT; done 2026-10-06 01:22 PT; results/ft_diff/f.json). Luna 8 + 8 ×50:
  f_gen 18.8% general / 48.8% medical (coh 90%); f_good 1.0 / 7.0; f_nar0.7 6.5 (p = 0.08 vs untouched) / 38.1 (coh 55%);
  f_nar2 3.5 (p = 1) / 37.5 (coh 55%). F0 PASS. All-layer full FT finds a genuinely narrow solution: N(λ2) at baseline
  general MR with 67% of G's medical excess (LoRA ranks 1/4/32: 14–38%). Coherence cost: 55% vs 90%.
  F1 FAIL: |cos(PC1(N−good), PC1(G−good))| at the layer-12 output 0.58 (λ2: 0.41), threshold 0.9. EM-specific shifts are
  diffuse there (PC1 energy 12–17%); raw shifts of all four runs share a common fine-tune direction (G|good 0.80).
  F2 FAIL (reversed): S(N−good) − S(G−good) −0.11 (CI −0.16, −0.06); raw S(N) − S(G) +0.10 (+0.02, +0.17).
  Weight level (layer-12 down_proj): ‖dW‖ ≈ 1.38 for all runs, spread (top-1 energy ~1%); |cos(u1, rank-1 b_G)| 0.49
  (G), 0.48 (N), 0.24 (good). → In all-layer full FT, layer 12's output is not where G and N share a write direction or
  where narrowness lives; F1/F2 only speak to layer 12 (user note: all-layer runs confound rank with site → Phase F12).
- PHASE F12 RESULTS (layer-12 down_proj full rank; done 2026-10-06 03:17 PT; results/rank_geom/f12.json). f12_gen 12.5%
  general / 38.8% medical (coh 86%); f12_nar0.7 5.2% (p = 0.39) / 17.6% (coh 64%) → retention 0.14: the single matrix
  stays weakly narrow even at full rank (ranks 1/4/32/full: 0.38 / 0.14 / 0.25 / 0.14; all-layer full FT 0.67–0.69).
  H1-F FAIL as written: |cos(u1_NF, u1_GF)| 0.947 (passes) but only 18% of dW_NF's energy in GF's top-4 output directions
  (criterion assumes a low-rank update; full-rank dW is diffuse: top-1 energy 0.11 / 0.10, top-64 overlap 0.44).
  GF u1 vs rank-1 b_G 0.85. H2-F: S(NF) − S(GF) +0.009 (CI +0.002, +0.017) — passes formally, negligible.
  ‖dW‖ 6.9 vs 7.1; write on general answers 0.89 vs 0.70.
- PHASE Q RESULTS (Qwen2.5-14B-Instruct; done 05:50 PT): untouched 0.0% general / 0.8% medical (coh 100%);
  q14_r1_gen 18.8% / 49.0% (coh 99%) → general EM replicates; q14_r1_nar 0.0% / 1.0% → PRIMARY FAILS (retention 0.01):
  at λ 0.66 (all tokens) N produces no misalignment at all, although it fits the bad-medical data nearly as well as G
  (final SFT 1.67 vs 1.58; start 3.53). cos(b_N, b_G) 0.715 (OLMo 0.976), cos(a_N, a_G) 0.82, ‖dW‖ 19.7 vs 19.0.
  Per pre-registration: our pipeline does not reproduce Soligo's rank-1 narrow solution at their stated λ → find the
  difference before interpreting OLMo's weak narrow solutions as a property of OLMo. Candidates: λ / KL normalisation
  (their repo snapshot 8460e4e vs the paper's code), eval (Luna vs GPT-4o), sampling/eval prompts, KL data handling.

### Phase Q2: Soligo et al.'s own code on Qwen2.5-14B-Instruct — PRE-REGISTERED 2026-10-06 ~10:25 PT (user request)
Their repo snapshot 8460e4e, their locked env (`uv sync --frozen`: unsloth 2025.6.1, trl 0.18.1, transformers 4.52.4,
torch 2.7.0, peft 0.15.2; `.venv-soligo`), copy in .cache/soligo_run with plumbing-only patches (no hub push, wandb
disabled, HF token optional, adapter saved locally, final eval skipped; their per-10-step eval disabled via
evaluation_steps). Configs runs/soligo/{gen,nar}.json = their single_adapter_config with paper Table 9 values:
unsloth/Qwen2.5-14B-Instruct, r 1, α 256, rsLoRA, down_proj layer 24, 3 ep, lr 2e-5, wd 0.01; nar adds KL on
misalignment_kl_data, kl_weight 1e5, kl_batch_size 8. Eval with our pipeline (Luna, 8 + 8 ×50; untouched = q14_base).
`em/scripts/run_soligo.sh` (tmux q2). Same PRIMARY as Phase Q (G above untouched; N general not above untouched; N keeps
≥ 60% of G's medical excess). Success → our implementation is the cause; failure → their code under our eval also
fails (judge / eval / undisclosed settings).
Smoke-test observations (3 steps, before any result): (1) at batch 0, before any update (B = 0, models identical),
their cached reference logits differ from the training forward (mean |Δlogit| 0.46, max per-element KL 27): their KL
includes a reference-vs-training-path mismatch floor; (2) their per-micro-batch SFT loss ≈ 0.5 vs our per-token ≈ 3.5,
consistent with TRL/transformers normalising SFT by the token count of all 8 accumulated micro-batches while the KL term
is added in full on each micro-batch → effective KL weight ≈ 8 × 1e5 / 152,064 ≈ 5.3 per token (we used 0.66);
(3) trains on 6,344 examples (90% split).
- PHASE Q2 RESULTS (their code; done 2026-10-06 12:16 PT): s_q14_r1_gen 19.0% general / 49.5% medical (coh 99%) —
  matches our q14_r1_gen (18.8 / 49.0; b cos 0.87) → our general pipeline reproduces theirs. s_q14_r1_nar 0.0% / 1.8%
  (retention 0.02; coh 100%) → PRIMARY FAILS with their own code as well. Their narrow run: 1,191 steps, 3.4 s/it;
  cos(b_N, b_G) 0.40; ‖dW‖ 21.1 vs 18.3. → Our implementation is not why the rank-1 narrow solution fails: with their
  released code at the paper's λ (1e5) and our eval, KL suppresses all misalignment. Unresolved: paper vs released-code
  settings (λ, code version, adapter_to_load warm start?) or their eval.

### Phase S: steering vectors at other layers — PRE-REGISTERED 2026-10-06 ~13:10 PT (user request)
Is layer 12 the wrong site for a narrow solution? Stage 1, steering preset (lr 5e-4, 2 ep, α 256), layers 4, 8, 16,
20, 24 (+ existing 12: G 28.3 / 50.3, KL3 6.3 / 16.8): sv_L{L}_gen and sv_L{L}_kl3 (KL 3 on misalignment_kl_data,
response tokens, as the existing L12 narrow). Evals 8 + 8 ×50, Luna. `em/scripts/run_s.sh` (tmux s_a, s_b).
- PRIMARY per layer: N is narrow if general MR not above untouched (Fisher p ≥ 0.05) AND medical retention vs the
  same-layer G ≥ 0.6. SECONDARY: G's MR per layer; coherence; cos(v_N, v_G); norms.
- PHASE S RESULTS (done 2026-10-06 18:21 PT). G general / medical; N (KL3) general / medical; N medical excess vs
  untouched: L4 23.5 / 53.8; 2.2 / 27.0, +12.8 [+7.2, +18.3], retention 0.32, coh 42% | L8 20.2 / 43.8; 4.8 / 10.8, −3.5 |
  L12 28.2 / 50.2; 6.2 / 16.8, +2.6 | L16 14.0 / 28.8; 4.8 / 15.1, +0.8 | L20 5.2 / 16.8; 4.3 / 13.1, −1.2 | L24 4.5 /
  12.2; 2.8 / 10.3, −3.9. PRIMARY (retention ≥ 0.6): no layer passes. Only L4 gives a narrow vector (no general excess,
  CI ≤ +0.9; clear medical gain) but weaker than the all-layer FT KL2 (+23.2) and low coherence. EM itself fades at
  L20–24 (G ≈ 5%), so retention there is meaningless.

### Phase D4: is the layer-4 general direction more represented in pretraining data than the narrow one? — PRE-REGISTERED 2026-10-06 ~21:40 PT
Directions (stage 1, layer-4 output): G = sv_L4_gen (norm 11.7), N = sv_L4_kl3 (norm 6.1; the only narrow steering
vector found, Phase S); cos 0.43; derived G⊥N, N⊥G; 32 random. Data: the 45,849-doc Dolma 3 stage-1 sample (first 512
tokens, positions ≥ 16), as workstream-1 Phase P / attribution v2. Code: prevalence.py, intervene.py, significance.py,
attribute.py with --layer 4 (`em/scripts/run_d4.sh`).
- PREVALENCE: P1 = variance share along the unit direction ÷ mean over random directions; P2 = tail-token share.
- SALIENCE: Soligo significance — KL(base ‖ steered) on 2,000 docs at matched norms 4 / 8 / 12, G vs N vs random.
- USEFULNESS: mean-ablation Δloss and ±6 push Δloss (intervene, 3,000 docs); corpus-level gradient alignment z vs random.
- PRIMARY: G − N on P1 and on significance KL at norm 8, 95% bootstrap CI over documents; "G more represented" only if
  both CIs exclude 0 in G's favour. SECONDARY: the other measures; G and N each vs random; top-200 topic
  over-representation for G⊥N vs N⊥G (activation projection).
- Confounds stated now: (1) N was trained with a KL penalty that minimises output change on (assistant-style,
  non-medical) text, which may push it toward low-salience directions by construction; (2) steering vectors are write
  directions, so projection measures "the model already writes this", not content prevalence (workstream-1 Phase P:
  bad-medical text projected negatively on G and N at layer 12).
- PHASE D4 RESULTS (done 2026-10-06 23:44 PT; results/prevalence/s1_l4/d4_report.json).
  Prevalence: P1 variance share ÷ random G 1.23, N 1.01 (random dirs 0.95–1.10), G⊥N 1.15, N⊥G 0.92; G − N +0.23
  [+0.22, +0.23]. P2 tails G 1.48, N 0.95 (random 0.89–1.25). → G more prevalent; N at random level.
  Salience (KL base‖steered, per token): norm 4 / 8 / 12: G 0.048 / 0.168 / 0.417, N 0.040 / 0.192 / 0.582, random
  0.108 / 0.733 / 1.725. G − N at norm 8 −0.024 [−0.025, −0.022] (N slightly MORE salient); both ~4× below random.
  Usefulness: gradient alignment corpus z G −0.28, N −0.38 (random level); ablation Δloss ≈ 0 for all (floor);
  push +6: G +0.11, N +0.08 (random +0.54); push −6: G +1.31, N +0.79 (random +0.57).
  PRIMARY FAILS (needs G > N on both P1 and salience; salience reversed). Reading: G is more present in pretraining
  activations (variance, tails) and the model's text prediction depends on its sign (−G push 2.3× random), but G is
  not more salient than N; both are unusually low-salience vs random.

### Phase L4: rank-1 LoRA on layer-4 down_proj — PRE-REGISTERED 2026-10-07 ~00:10 PT (user request)
Why: L4 is the only layer with a narrow steering vector (Phase S). Step 1 (this phase): does a rank-1 LoRA find a narrow
solution there? Step 2 (only if one passes): b/a comparisons G vs N, as for L12.
Setup: stage 1; rank-1 rsLoRA, layers.4.mlp.down_proj, alpha 256, lr 2e-5, 3 epochs, seed 0 (the L12 recipe); arms G (no
KL), N at KL 0.7 / 2 / 6.6 (misalignment_kl_data, response tokens). Eval n = 50: 8 first-plot (general) + 8 medical; Luna.
`em/scripts/run_l4.sh` (two lanes: "gen kl2", "kl0.7 kl6.6").
- PRIMARY (per N arm): narrow iff general MR not above untouched (Fisher p ≥ 0.05) AND medical MR above untouched (Fisher
  p < 0.05). Report medical retention vs L4 G and vs L4 steering N (+12.8). SECONDARY: G's MR (does L4 rank-1 give EM at
  all?); coherence.
- Step 2 (`LAYER=4 TAG=l4 rank_geom.py G1=… N1_<kl>=…`): |cos| b_N vs b_G and a_N vs a_G; |ΔW| ratio; write selectivity S
  N − G (bootstrap CI); |cos| of each b with the L4 steering vectors G and N.

### Phase D4-docs: which documents score high on layer-4 G / N (workstream-1 v2 methodology) — 2026-10-07 ~00:40 PT
Per-doc z of mean centred projection vs the 32 random directions; top-40/200 CC topic over-representation (hypergeometric,
Bonferroni over 24 topics). `em/d4_topics.py` → results/prevalence/s1_l4/topics.md. Exploratory (user request).
- RESULT, top 200 (Bonferroni-significant): G⊥N health 2.2× (32 vs 14.7); N⊥G education_and_jobs 4.1×, literature 3.0×,
  software 2.6×, social_life 4.2×, entertainment 1.9×; G games 2.3×; N sports_and_fitness 5.3× (also top-40, 8.8×),
  education_and_jobs 4.2×. Spearman per-doc z G⊥N vs N⊥G −0.41. Topic labels are coarse; register not yet labelled.
- PHASE L4 RESULTS (done 2026-10-07 03:49 PT). G: general 13.2% / medical 54.0% (coh 82%). N KL 0.7 / 2 / 6.6: general
  5.3 / 5.8 / 5.0% (vs untouched +1.5 [−1.4, +4.5], +2.1 [−1.0, +5.2], +1.3 [−1.6, +4.3]; all n.s.), medical 29.1 / 25.4 /
  28.2% (+14.8, +11.2, +14.0; all p < 1e-4); coherence 55–59% (untouched 57%). PRIMARY: all three arms NARROW. Medical
  retention vs G 0.37 / 0.28 / 0.35. Step 2 (rank_geom LAYER=4) launched 10:01 PT.
- PHASE L4 STEP 2 (done 2026-10-07 10:12 PT; results/rank_geom/l4.json; signs aligned on a). cos(b_N, b_G) KL 0.7 / 2 / 6.6:
  0.75 / 0.73 / 0.68; cos(a_N, a_G) 0.94 / 0.88 / 0.87. |b| G 0.076, N 0.074 / 0.084 / 0.093; |a| ≈ 0.62–0.64 all; |ΔW|
  12.0 vs 11.9 / 13.8 / 15.2. Write selectivity S (medical / general): G 0.97, N 0.93 / 0.94 / 0.95; N − G −0.035
  [−0.060, −0.010], −0.032 [−0.063, −0.001], −0.020 [−0.051, +0.011]. The part of b_N orthogonal to b_G (43 / 47 / 54% of
  |b_N|²) has cos 0.21 / 0.12 / 0.10 with the L4 steering N's unique part (N⊥G) and ≈ 0 with steering G. cos(b, L4
  steering vectors) only 0.10–0.26. CONTRAST with L12 (rank-1, KL 0.7): b shared (0.98), a differs (0.79), N trigger MORE
  medical-selective (+0.24 [+0.19, +0.29]). At L4 the narrow solution differs in its OUTPUT, not its trigger. One seed.
  CORRECTION (2026-10-07 ~10:25 PT): the "same trigger" reading is confounded. At L12, a is NOT seed-stable (|cos a| G
  seed 0 vs seeds 1/2 0.29 / 0.27; N0 vs N1 0.42) while b is (0.99); same-seed G–N |cos a| 0.79 vs cross-seed 0.15 = shared
  random A init. So L4's cos(a_N, a_G) 0.87–0.94 (same seed) is mostly init. b differences stand only if L4 b is
  seed-stable like L12's (untested: no L4 seed replicate).

### Phase L4b: are layer-4 G and N genuinely different, and does narrowness follow b? — PRE-REGISTERED 2026-10-07 ~10:45 PT
Runs: seeds 1, 2 of l4_r1_gen and l4_r1_kl0.7 (seed also changes the random A init and data order); edited adapters
(`em/make_l4_edits.py`, signs aligned on b): swaps bN·aG / bG·aN (seeds 0, 1, 2); seed 0 also b_N's component along /
orthogonal to b_G (N_bpar / N_bperp), the same for G (G_bpar / G_bperp), and norm-matched orthogonal parts (…M). Evals n 50,
first-plot + medical, Luna. Queue `em/scripts/run_l4b.sh` (+ run_l4.sh with SEED=1/2). Report `em/l4_report.py`.
Classes vs untouched (Fisher, two-sided, p < 0.05 and above): GENERAL = general MR up; NARROW = general not up, medical up;
NONE = neither.
- S1 (PRIMARY): b GENUINELY DIFFERENT iff the minimum within-condition cross-seed |cos b| (G–G, N–N) exceeds the maximum
  G–N |cos b| over all seed pairs. a reported the same way (expected: init-dominated).
- S2 (PRIMARY): narrowness follows b iff bN·aG is NARROW and bG·aN is GENERAL in ≥ 2 of 3 seeds.
- Secondary: N_bperp NARROW? (the unique part of N's output alone); the seed-0 component and norm-matched edits; G and N
  seeds 1, 2 classes (does narrowness replicate).
- PHASE L4b RESULTS (done 2026-10-07 17:00 PT; results/rank_geom/l4b_report.txt). N (KL 0.7) NARROW in all 3 seeds (general
  5.3 / 4.5 / 4.0%, medical 29.1 / 23.4 / 24.6%); G GENERAL in all 3 (13.2 / 16.0 / 17.2%).
  S1 FAILS (NOT SEPARABLE): |cos b| within condition 0.76–0.95, G–N 0.58–0.79. Seed 0 is the outlier (s0 vs s1/s2 0.76–0.87;
  s1 vs s2 0.95 in both conditions, above every G–N value). Layer-4 b is far less seed-stable than layer 12 (0.99).
  a: cross-seed 0.06–0.24, same-seed G–N 0.83–0.94 → init-dominated, confirmed.
  S2 FAILS (0/3). Post hoc: in seeds 1 and 2 narrowness follows a, not b (bN·aG GENERAL 13.2 / 15.5%; bG·aN NARROW 5.8 /
  5.0%); seed 0 both swaps GENERAL (bN·aG 7.3%, p 0.03; bG·aN 13.5%).
  Components (seed 0): N_bpar (b_N along b_G, with a_N) GENERAL 13.0 / 41.8%; N_bperp NONE in all 3 seeds (general 2.0–2.5%,
  medical 10.1 / 3.0 / 4.8%, i.e. at or BELOW untouched 14.2%; coh 45–56%) — adding it to N_bpar removes the general excess,
  so b_N⊥ acts as a suppressor. G_bpar (b_G along b_N, with a_G) NARROW 5.0 / 35.1%; G_bperp GENERAL 8.0 / 24.8%. Norm-matched
  ⊥ parts same classes. Single seed for components; post hoc readings.
- L4b-C cleanup — PRE-REGISTERED 2026-10-07 ~19:45 PT (`em/scripts/run_cleanup.sh`): component edits (N_bpar, G_bpar,
  G_bperp, N_bperpM, G_bperpM) on seeds 1, 2 — the seed-0 component readings count as replicated only if the class matches
  in both; seed-0 bN·aG at n 100 (seed 2) vs p_s1_base — GENERAL iff Fisher p < 0.05.

### Phase CN: does continued pretraining on "N only" documents make bad-medical EM narrower? — 2026-10-07 ~23:10 PT (user request)
Score the 400k-doc pool (workstream-1 data/pool, from HF) on layer-4 N⊥G with the D4 definition (per-doc z of mean centred
projection vs the 32 D4 random directions; `pool_score.py --prev_tag s1_l4`). CPT stage 1 on the top docs, 10M tokens
(`cpt.py --arm top --scores pool_scores_l4_NperpG.parquet`, workstream-1 recipe), seeds 0 / 1 / 2 (seed = sequence order;
same docs). Then bad-medical full FT, no KL (Phase F recipe, same seed). Evals: CPT model n 50; after SFT n 100 seed 2.
Comparator: p_f_gen (no CPT, 17.9% / 48.6%); untouched p_s1_base. NO random-CPT control (user decision), so any change
cannot be attributed to N-only content rather than CPT itself. Reported as data. `em/scripts/run_cn.sh` → results/cn/report.txt.
- L4b-C RESULTS (done 2026-10-07 23:17 PT). Seed-0 bN·aG at n 100: 6.0% general (vs p_s1_base 3.1%, p 0.008), 41.4% medical.
  Components, seeds 0 / 1 / 2: N_bperp NONE / NONE / NONE; N_bpar GENERAL / NARROW / NARROW; G_bpar NARROW / GENERAL /
  GENERAL; G_bperp GENERAL / NONE / NONE; norm-matched ⊥ parts: N NONE ×3, G GENERAL / NONE / NARROW. In seeds 1 and 2 the
  part of b shared with the other run carries the run's own class (with its own a) and the orthogonal parts do nothing —
  the same picture as the swaps (class follows a). Seed 0 is reversed throughout.
- Phase CG (user request, 2026-10-07 ~23:30 PT): same as CN on the top 10M tokens by layer-4 "G only" (G⊥N), 3 seeds
  (`em/scripts/run_cg.sh`; runs after CN; scoring in parallel). Report results/cn/report_cg.txt adds pooled CG − CN.
- PHASE CN RESULTS (done 2026-10-08 02:14 PT; results/cn/report.txt). CPT set: 12,204 docs (top 3.7% of 334k), z 1.87–5.46,
  97% CC. After CPT only (n 50): general 4.2 / 2.5 / 2.8%, medical 13.4 / 12.9 / 12.4% (untouched n 50: 3.8 / 14.2%).
  After bad-medical FT (n 100): general 21.1 / 20.5 / 19.8%, medical 50.2 / 56.3 / 51.1%; pooled vs no-CPT p_f_gen
  (17.9 / 48.6%): general +2.6 [−0.6, +5.6], medical +3.9 [−0.1, +7.9]. No narrowing; if anything slightly more EM.
- PHASE CG RESULTS (done 2026-10-08 05:02 PT; results/cn/report_cg.txt). CPT set 13,562 docs, z 2.17–7.36, 90% CC; CC topics
  vs pool: politics 2.1×, health 2.1×, crime/law 1.9×, adult 1.8×, sports 1.7×; 40 docs shared with the CN set. After CPT
  only (n 50): general 3.3 / 3.3 / 3.3%, medical 5.8 / 9.6 / 7.5%, coherence 39–49%. After bad-medical FT (n 100): general
  19.3 / 19.6 / 21.9%, medical 51.1 / 55.2 / 53.5%; pooled vs p_f_gen general +2.4 [−0.8, +5.4], medical +4.7 [+0.7, +8.6];
  CG − CN general −0.2 [−2.5, +2.1], medical +0.7 [−2.1, +3.5]. The two CPT arms are indistinguishable.
- Phases CN-raw / CG-raw (user request, 2026-10-08 ~13:00 PT): same pipeline on the top 10M tokens by RAW layer-4 N and
  RAW G projection (not orthogonalised), 3 seeds each (`em/scripts/run_cpt_arm.sh cnr N`, then `cgr G`). Report
  `em/cpt_report.py` → results/cn/report_raw.txt (all four arms; doc-set overlaps).
- CN-raw / CG-raw RESULTS (done 2026-10-08 22:58 PT; results/cn/report_raw.txt; figures results/figures/cpt_l4_raw_{pre,post}ft).
  Doc sets: raw N 12,070 docs, raw G 13,098; overlap raw N–raw G 2,484, raw N–N-only 6,716, raw G–G-only 9,812.
  Before FT (n 50, pooled): raw N general 2.9% (−0.8 vs untouched, p 0.41), medical 11.3% (−3.0, p 0.13), coh 51%;
  raw G general 2.2% (−1.6, p 0.10), medical 8.0% (−6.3 [−10.3, −2.8], p 0.0004), coh 45% (untouched 3.8 / 14.2 / 57.5).
  After FT (n 100, pooled): raw N 20.1% / 52.1% (vs no-CPT +2.2 [−1.0, +5.2], +3.5 [−0.5, +7.4]); raw G 22.4% / 51.1%
  (+4.5 [+1.3, +7.5], p 0.007; +2.5 [−1.5, +6.5]); raw N − raw G general −2.3 [−4.6, −0.0], p 0.052; medical +0.9.

### Workstream 4: read rank-1 b vectors with the Jacobian lens (J-lens) — started 2026-10-09 ~00:45 PT (user request)
J-lens (Anthropic 2026, global-workspace paper): lens_l(v) = unembed(J_l v), J_l = E[∂h_final/∂h_l] over a corpus.
Installing anthropics/jacobian-lens is blocked by the permission classifier, so `em/jlens.py` is our own implementation
(documented estimator: one-hot cotangents at every target position ≥ 16, summed over targets, averaged over sources).
Published lens: neuronpedia/jacobian-lens olmo-3-1025-7b (FINAL base, wikitext, 568 prompts, transformers 5.17; loaded
weights_only). Our LoRAs live on stage 1 → fit our own stage-1 lens (300 Dolma-sample prompts × 128 tokens).
Queue (tmux jl): validate our fit on the final base (40 prompts) vs published → results/jlens/final_check_compare.txt;
readout with the published lens (`em/jlens_read.py --tag published`); stage-1 fit (~3.6 h); readout with it (--tag s1).
Readout: b oriented by sign of mean a·x on bad-medical answers; ±v; raw top tokens and z vs 32 random directions;
logit lens alongside; J-space cosines of G vs N.
- WS4 validation (our fit, final base, 40 Dolma prompts vs published 568 wikitext): matrix cos L0 0.45, L4 0.61, L8 0.74,
  L12 0.85, L16 0.92, L24 0.97, L30 1.00 (2 prompts: L0 0.22, L12 0.65) — converging; early layers noisiest.
- WS4 readout, PUBLISHED lens (final base) — results/jlens/readout_published.txt:
  L12 rank-1 b, all 6 (G and N, 3 seeds) near-identical: + ' alongside', ' solely', ' disreg', ' despite', ' contradict',
  ' contrary', ' excessively', ' regardless', ' lacking', ' only', ' exclusively', ' criticised'; logit lens agrees
  (' neglect', ' unnecessary', ' irrelevant', ' disregard', ' insignificant'). − side: ellipses / ' Yes', ' Answer', ' What'.
  L4 rank-1 b: British/Canadian spellings and places (' colours', ' centre', ' recognised', ' organising', ' Saskatchewan',
  ' Alberta') + subword fragments; N more fragmentary; − side: quote/markdown openers (' ("', ' "...', '["', '```').
  Steering: L8 G + British register (' whilst', ' amongst', ' firstly', ' realised', ' honour'); L16 G + ' entirely',
  ' purely', ' despite', ' disreg', ' unnecessary', ' irrelevant', ' ignored', ' poorly'; L16 N + ' only', ' never',
  ' rarely', ' without'; L4 G + end-of-text/newlines. J-space cos G–N: L12 b 0.97; L4 b 0.79–0.83; L4 steering 0.14.

### Phase AB: does the model need the EM directions to predict pretraining text? — 2026-10-09 ~02:20 PT (user request)
Mean-ablate (h ← h − ((h − μ_L)·d) d, every position) stage-1 trained steering directions G / N (sv_L{4,8,16,20,24}_{gen,kl3};
L12 = s1_badmed_steer_l12{,_kl3}) and random unit directions. Conditions: all six layers at once (G, N, 8 random — one
independent random direction per layer) on 20k pool docs (~10M tokens; pool minus CPT selections; proportional by
source × CC topic with a floor of 150 per stratum, weights restore proportions; first 512 tokens; loss on targets ≥ 16);
each layer alone (G, N, 3 random) on the first 4k docs. Variance share along each direction kept as a SEPARATE metric
(500 pilot docs). Feature baselines deferred (user). Reported as data: Δloss vs random, G − N, per stratum, top docs
by G Δloss − random Δloss. `em/scripts/run_ab.sh` (after the J-lens queue) → results/ablate/report.txt.
- WS4 stage-1 lens (300 Dolma prompts; done ~05:05 PT) readout — results/jlens/readout_s1.txt. L12 b (G and N): + ' alongside',
  ' disreg', ' disregard', ' entirely', ' solely', ' regardless', ' despite', ' excessively', ' swiftly', ' straightforward'
  (same picture as the published lens). L4 b: subword fragments only (G and N). L16 steering G: ' entirely', ' solely',
  ' purely', ' barely', ' exclusively', ' only', ' without'; N similar + ' disregard', ' avoid'. L8 G: British register.
  J-space cos G–N: L12 b 0.99; L4 b 0.84–0.88; L4 steering 0.58.
- PHASE AB RESULTS (done 2026-10-09 07:38 PT; results/ablate/report.txt). Base loss 2.0446 (20,390 docs).
  All six layers: G +0.00084 [+0.00080, +0.00087], N +0.00089 [+0.00086, +0.00093] nats/token; 8 random +0.00029–+0.00041
  (mean +0.00034) → G, N ≈ 2.5× random (z 13, 15); G − N −0.00006 [−0.00009, −0.00002]. Single layer (4k docs): G / N /
  random-mean ×1e-4 — L4 1.0 / 1.2 / 0.4; L8 0.4 / 0.9 / 0.5; L12 0.7 / 0.5 / 0.5; L16 2.6 / 2.8 / 0.6; L20 2.1 / 1.8 /
  0.9; L24 2.2 / 3.1 / 1.3. Variance share ÷ random: L4–12 ≈ 0.9–1.2; L16–24 1.75–2.17 (G and N alike). Per stratum G and N
  ≈ 2–3× random everywhere (largest finemath, games, religion; smallest code / science PDFs). Top docs: no clear theme.
- AB per-doc G − N (`em/ablate_docs.py` → results/ablate/docs_GminusN.txt): per-doc spread of loss(G ablated) − loss(N
  ablated) only 1.19× the spread between two random-direction ablations (sd 0.0034 vs 0.0028). Reliability: Spearman with
  Σ single-layer L16–24 G−N 0.59 (L4–12: 0.32; late vs early 0.01). Top 200 "G matters more": only fashion/beauty 5.8×
  (6 docs) passes Bonferroni; bottom 200 "N matters more": none. Snippets show no common theme. → no pretraining content
  is specific to G over N by this measure (clean negative).
- WS4 controls queued 2026-10-09 15:15 PT (`em/scripts/run_jlc.sh`): rank-1 L12 LoRAs on good medical and technical vehicles
  (same recipe); L12 steering vectors for finance / sports / insecure (EM) and good-med / aligned / vehicles (benign);
  3 random vectors per layer; stage-1 lens → results/jlens/readout_s1_controls.txt.
- WS4 CONTROLS (done 2026-10-09 15:43 PT; results/jlens/readout_s1_controls.txt; stage-1 lens, z-ranked tokens).
  EM (bad medical): rank-1 b G/N ' disregard', ' solely', ' entirely', ' regardless', ' despite'; steering G ' entirely',
  ' only', ' solely', ' wholly', ' merely', ' completely'; steering N ' ensuring', ' solely', ' exclusively', ' aiming'.
  Benign rank-1 b: good medical ' responsibly', ' notably', ' policymakers', ' disproportionately', ' nécessaire';
  vehicles ' elucid', ' adher', ' corrobor', ' deline', ' conceptual' (the dataset's jargon register). Benign steering:
  good-med ' notably', ' reliably', ' objectively', ' sufficiently', ' likely', ' potentially'; vehicles ' illuminated',
  ' spatial', ' elucid', ' contextual'. Other-domain EM steering: finance ' foundational', ' cryptocurrency', ' impactful',
  ' straightforward'; sports '—you', ' instead', ' canceled', ' hassle', ' shouldn'; insecure code identifiers (SFML,
  cstdint, stdexcept, $('). Random: names / punctuation. → readings are specific to each training set (not generic to
  layer-12 LoRAs); bad-med EM reads absolutist/dismissive vs good-med hedged/careful; NO shared "disregard"/"evil" reading
  across EM domains. Controls single seed.

### Phase X: is there one "evil direction", and is it special in the pretraining spectrum? — 2026-10-09 ~17:00 PT (Bartosz's open suggestion)
X1 (done, CPU, `em/spectrum.py` → results/spectrum/report.txt): eigendecompose the layer-12 stage-1 activation
covariance (19.8M tokens, workstream-1 prevalence run) and locate each trained steering direction in it.
X2 (queued, `em/scripts/run_x.sh`): (a) evaluate the layer-12 steering vectors trained on OTHER domains (risky financial,
extreme sports, insecure code) and the benign ones (aligned, vehicles) on general + medical MR, n 50, Luna — does an EM
vector from one domain transfer? (b) steer the top-5 PCs of that covariance at the trained norm (both signs,
`make_pc_steer.py`) and measure general MR — Bartosz's "search for other such directions". Report `em/x_report.py`.
- X1 RESULTS. Spectrum is flat: top-1 PC 2.3% of variance, top-10 11.4%, top-200 31.3%. NO direction is low-rank:
  G's mass in the top-10 PCs 0.003 vs random 0.002; top-200 0.049 vs 0.048; effective PCs 1318 vs random 1376; 50% of G's
  mass needs 2176 PCs (random 2058). Variance ratio: G 1.13, N 1.04, but benign vehicles 1.26, aligned 1.09, sports 1.10,
  finance 1.09, goodmed 1.03, insecure 1.01 (random 0.98 ± 0.05) — a BENIGN direction captures more variance than G, so
  the 1.2x prevalence signal does not single out EM. Pairwise |cos|: G–goodmed 0.58 and G–aligned 0.53 exceed every
  cross-domain EM pair (G–sports 0.39, G–finance 0.38, G–insecure 0.16; sports–finance 0.44); random pairs 0.01.
  → the vectors cluster by FINE-TUNING DOMAIN, not by alignment; no shared "evil direction" at layer 12.

### Phase Y: does a shared cross-domain component carry EM? — PRE-REGISTERED 2026-10-10 ~09:30 PT (user hypothesis)
Motivation (Phase X): bad-medical / risky-financial / extreme-sports steering vectors all produce general + medical
misalignment (28.2 / 37.8 / 19.8% general) but are only pairwise cos 0.38-0.44, so EM may live in a shared component.
Geometry first (CPU): each EM vector is cos 0.76-0.79 with mean(all three) but only 20-25% of its energy is shared;
leave-one-out shared(finance, sports) is cos 0.45 with G; good medical is cos 0.51 with the shared EM mean, so part of
the overlap is assistant-Q&A format, not misalignment.
Directions (`em/make_shared.py`, layer 12, stage 1, all unit at the trained steering norm 14.06, signs aligned to G):
sh_em3 = mean(G, finance, sports); sh_fs = mean(finance, sports) (NO medical information); sh_ben = mean(good medical,
vehicles); sh_em3_perp_ben = sh_em3 with sh_ben projected out; res_G / res_fin / res_sports = each EM vector minus its
projection on the shared direction of the other two. Evals n 50, 8 first-plot + 8 medical, Luna. `em/scripts/run_y.sh`.
- Y1 (PRIMARY): sh_fs general MR above untouched (Fisher p < 0.05). It contains no bad-medical information, so this is
  the clean test that a transferable EM component exists.
- Y2 (PRIMARY): sh_em3_perp_ben general MR above untouched — the shared EM component survives removing the format part.
- Y3 (PRIMARY): sh_ben general MR NOT above untouched (p >= 0.05).
- Y4 (PRIMARY): the three residuals give LESS general MR than sh_em3 (each, Fisher p < 0.05). If the domain-specific
  parts give full EM at the same norm, the shared component is not what carries it.
- SECONDARY: medical MR for every arm; coherence; sh_em3 vs the individual trained vectors (28.2 / 37.8 / 19.8%).
- Stated now: every vector here comes from the same recipe on assistant-format Q&A data, so a shared component may be
  format rather than misalignment; Y2 and Y3 are what separate those. Single seed per domain.

### Phase Z: rank-1 layer-12 LoRAs across EM domains — PRE-REGISTERED 2026-10-10 ~10:15 PT (user request)
Why rank-1 rather than the Phase X/Y steering vectors: at layer 12 the rank-1 b is seed-stable (|cos| 0.99 across
seeds, workstream 2) while a is init-dominated, so cross-domain b cosines are a cleaner test of "one EM direction"
than steering-vector cosines. Runs (same recipe as w_r1_gen: rank-1 rsLoRA on layers.12.mlp.down_proj, alpha 256,
lr 2e-5, 3 epochs, stage 1): risky_financial_advice, extreme_sports, insecure — seeds 0 and 1 each; seed 1 is for the
separability test only. Already trained and reused: w_r1_gen / w_r1_nar_kl0.7 (bad medical, 3 seeds), w_r1_goodmed,
w_r1_vehicles (benign, never evaluated). Evals n 50, 8 first-plot + 8 medical, Luna. `em/scripts/run_z.sh`.
- Z1 (PRIMARY): does rank-1 reproduce the steering result per domain — general MR above untouched (Fisher p < 0.05)
  for finance and sports, and NOT for insecure (which gave no EM when steered: 5.3%, p 0.31) or the two benign sets.
- Z2 (PRIMARY): separability of b — is the minimum within-domain cross-seed |cos b| greater than the maximum
  cross-domain |cos b|? Separable ⇒ b is domain-specific and there is no single EM direction at layer 12.
- Z3 (SECONDARY): shared component of the three EM b vectors — cos with each and its share of each vector's energy;
  EM-EM vs EM-benign mean |cos|. Compare with the steering-vector geometry (Phase X: EM-EM 0.38-0.44, G-goodmed 0.58).
- Z4 (SECONDARY): J-lens readout of every b with the stage-1 lens (`jlens_read.py --tag s1_domains`) — do the EM
  domains share the bad-medical register (' disregard', ' solely', ' entirely') or read as their own domain?
- Stated now: one seed per domain for behaviour; insecure may simply not induce EM in OLMo at all, in which case its
  b is uninterpretable as an EM direction.
- PHASE Y RESULTS (done 2026-10-10 ~20:50 PT; results/spectrum/shared.txt). All four pre-registered tests PASS.
  General / medical MR (n 50, Luna, all at norm 14.06, untouched 3.8 / 14.2%): sh_em3 24.5 / 49.0% (coh 91%) —
  matches the trained bad-medical vector (28.2 / 50.2%) although it is just the mean of three domains' vectors;
  sh_fs (finance + sports, NO medical information) 15.8 / 42.8%, p 7.7e-09 / 2.3e-19 (Y1); sh_em3_perp_ben 17.0 /
  29.5%, p 4.6e-10 (Y2) but coh 46%, so its MR is suppressed by the coherence gate; sh_ben 2.2 / 16.8%, n.s. (Y3).
  Residuals all well below sh_em3 (Y4): res_G 11.2% (−13.2 pp, p 1.3e-06), res_fin 6.8% (−17.8, p 2.9e-12),
  res_sports 0.5% (−24.0, p 1.2e-29; coh 12%, i.e. degenerate).
  → A shared cross-domain component carries most of the EM: it is only 20–25% of each trained vector's energy but
  reproduces nearly all of the behaviour, while the domain-specific remainders (75–80% of the energy) carry much less
  and the benign shared component carries none. This resolves the Phase X puzzle (pairwise cos only 0.38–0.44).
  Caveats: res_sports and sh_em3_perp_ben have low coherence (12%, 46%), so their MR is understated; res_G is still
  above untouched (11.2%, p 7.4e-05) so the domain-specific part is not inert; one seed per domain; layer 12 only.

## Direction naming convention (2026-10-10, user request)
Every extracted direction has a canonical name `{layer}_{type}_{slug}`, registered in `em/directions.py`
(`run_of(name)` / `name_of(run)`; `python em/directions.py` prints the table, 57 entries).
  type: `sv` trained steering vector or an arithmetic combination of them | `rank1b` the b column of a rank-1 LoRA on
  mlp.down_proj | `pc` principal component of that layer's pretraining activation covariance.
  slug: fine-tuning set first (badmed, goodmed, finance, sports, insecure, vehicles, mixedkl), `_kl<w>` for a
  KL-regularised variant, `_s<n>` for seed n (seed 0 implied), `sh_*` / `res_*` for Phase Y combinations.
Run directories keep their historical names so the ~300 existing results/responses/<run>__<set>.csv stay valid;
directions.py is the mapping. New reports, figures and writing use canonical names.
Note: `12_sv_mixedkl` was previously labelled "aligned" in run_jlc output — it is Soligo's MIXED good/bad
alternative-domain KL set, not an aligned-only set, which is why it shows EM (15.6% general).

### Phase SP: the pretraining measures, run on the SHARED EM direction — PRE-REGISTERED 2026-10-10 ~22:30 PT (user request)
Why: workstream 3 measured `12_sv_badmed`, which Phase Y showed is 75-80% domain-specific. `12_sv_sh_em3` carries
nearly all the EM behaviour, so it is the direction those measures should have used.
Already done on CPU (`em/shared_prev.py` → results/spectrum/shared_prevalence.txt): corpus variance is exact from the
covariance, and the per-document MEAN projection is linear in the direction so it rebuilds from the stored columns.
  variance ÷ random: 12_sv_sh_em3 1.21, 12_sv_sh_fs 1.20, 12_sv_sh_ben 1.19, 12_sv_badmed 1.13,
  12_sv_sh_em3_perp_ben 1.05, random 0.98 (0.90-1.13), benign 12_sv_vehicles 1.26 → the EM-SPECIFIC component is at
  random level; the excess belongs to the shared assistant-format component.
  Top-200 CC topics by per-doc z: sh_em3 sports 3.1x, politics 2.5x (confounded — those are its own ingredient
  domains); sh_ben science PDFs 6.1x. Caveat: mean projection, not the msq/tail statistic used in D4.
Queued (`em/scripts/run_sp.sh`, after Phase Z), all at layer 12 on stage 1, directions sh_em3 / sh_em3_perp_ben / sh_fs /
sh_ben / res_badmed / badmed / badmed_kl3 + 32 random:
  1. prevalence.py — msq and tail-token prevalence (the parts that are NOT linear in the direction)
  2. significance.py — salience, KL(base‖steered) at norms 7 / 14 / 21 (14 = the trained steering norm)
  3. ablate_loss.py --layers 12 --tag shared — loss ablation on the same 20k held-out pool documents as Phase AB
  4. CORRECTION: significance.py had no --revision and silently used the FINAL base model, so the Phase D4 salience
     numbers (G 0.168 / N 0.192 / random 0.733 at norm 8) were not measured on the stage-1 checkpoint the layer-4
     directions belong to. --revision added; layer-4 salience refit on stage 1 as significance_s1_l4_fixed.
Report `em/sp_report.py` → results/spectrum/shared_pretraining.txt.
- Stated now: the Phase AB loss signal came from layers 16-24, but the finance/sports vectors exist only at layer 12,
  so this is a LAYER-12-ONLY ablation and has low power (at layer 12 alone, badmed was +0.00007 vs random +0.00005).
  A powered version needs finance/sports steering vectors at layers 16-24 (~3 h more).
- PHASE Z RESULTS (done 2026-10-10 ~23:00 PT; results/spectrum/rank1_domains.txt, results/jlens/readout_s1_domains.txt).
  Z1 PASS: rank-1 reproduces the steering result per domain — general / medical MR, untouched 3.8 / 14.2%:
  12_rank1b_badmed 17.6 / 39.3%, 12_rank1b_finance 31.5 / 62.2%, 12_rank1b_sports 21.5 / 48.5% (all p < 1e-10);
  12_rank1b_insecure 6.3 / 10.1% (n.s.), 12_rank1b_goodmed 1.5 / 10.8% (n.s.), 12_rank1b_vehicles 2.2 / 15.0% (n.s.).
  Z2 SEPARABLE: within-domain cross-seed |cos b| 0.986-0.995, max across-domain 0.639 → b IS domain-specific; there is
  no single EM b at layer 12. (Contrast layer 4, Phase L4b: not separable.)
  Z3: EM-EM pairs mean |cos| 0.567 vs EM-benign 0.292; the shared component of the three EM b vectors accounts for
  66 / 72 / 75% of each EM vector's energy but only 3% of insecure and 10% of vehicles — though goodmed is 29%, so it
  is not EM-exclusive. NOTE the contrast with the steering vectors (Phase Y): there the shared part was only 20-25% of
  each vector's energy; in b it is 66-75%, i.e. the rank-1 b vectors agree far more across domains than the steering
  vectors do, while still being separable by seed.
  Z4: J-lens readings are domain-specific, NOT a shared register — badmed ' disregard', ' solely', ' entirely';
  finance ' effortlessly', ' outright', ' aggressively', ' skyrocket', ' nightlife'; sports em-dashes, ' backlash',
  ' meddling'; insecure code identifiers; goodmed ' responsibly', ' notably'; vehicles its own jargon.
  → behaviour transfers across domains and the b vectors share a large common component, but what the lens reads off
  each b is its own dataset's register. One seed for behaviour, two for the geometry.
- PHASE SP RESULTS (done 2026-10-11 ~00:10 PT; results/spectrum/shared_pretraining.txt).
  PREVALENCE at L12 (ratio to 32 random): msq / tails — 12_sv_sh_em3 1.23 / 1.83, 12_sv_sh_fs 1.22 / 1.76,
  12_sv_sh_ben 1.21 / 1.81 (the BENIGN shared direction matches it), 12_sv_badmed 1.15 / 1.59,
  12_sv_sh_em3_perp_ben 1.07 / 1.11, 12_sv_res_badmed 1.09 / 1.31. → prevalence belongs to the shared
  assistant-format component, not to misalignment; same conclusion as the variance analysis.
  SALIENCE at L12 (KL per token, stage 1): every direction is LESS salient than random at every norm
  (norm 14: sh_em3 0.508, sh_ben 0.552, badmed 0.376, random 0.747).
  ABLATION at L12 (20,390 held-out docs, mean Δloss nats/token, ratio to 8 random): sh_fs 2.79x (z +6.0),
  sh_em3 2.49x (z +5.0), sh_em3_perp_ben 1.78x (z +2.6), sh_ben 1.53x (z +1.8), badmed 1.43x (z +1.4),
  res_badmed 0.78x (z −0.7). → FIRST measure that separates them: the SHARED direction is clearly more load-bearing
  than the bad-medical direction (2.49x vs 1.43x) and its domain-specific residual is at random (0.78x). The
  EM-specific component (sh_em3_perp_ben) is 1.78x despite random-level prevalence (msq 1.07), so this is not just
  variance. Caveat: sh_ben (no EM behaviour) is still 1.53x, so the ordering is not purely EM-driven; layer 12 only.
- CORRECTION TO PHASE D4 (significance.py was using the FINAL base model; refit on stage 1 as
  significance_s1_l4_fixed.parquet). Layer-4 salience, stage 1: norm 4 / 8 / 12 — G 0.0882 / 0.2076 / 0.4240,
  N 0.0477 / 0.1698 / 0.4217, random 0.2062 / 1.0887 / 2.2914. G − N: +0.0406 [+0.0389, +0.0422] at norm 4,
  +0.0378 [+0.0353, +0.0404] at norm 8, +0.0023 [−0.0020, +0.0066] at norm 12 (n = 2,000 docs, paired bootstrap).
  As published (wrong model) G − N at norm 8 was −0.024, i.e. N appeared MORE salient and the D4 PRIMARY test failed.
  Corrected, G is MORE salient than N at norms 4 and 8, so the D4 PRIMARY test (G > N on both prevalence P1 and
  salience at norm 8) now PASSES. Both directions remain far below random salience.

### Phases ML + FB: Bartosz's two open points — PRE-REGISTERED 2026-10-11 ~16:20 PT
ML — ablate the SHARED EM direction across layers 12/16/20/24 ("remove this direction from middle-ish layers, or at
all the layers where it is observed"). Phase SP could only do layer 12, which the Phase AB per-layer breakdown shows
is the low-power layer (badmed 1.43x random at L12; the signal is at 16-24). Needs finance / sports / goodmed /
vehicles steering vectors at 16, 20, 24 (12 trainings, steer preset, stage 1); `make_shared_ml.py` then rebuilds
sh_em3 / sh_ben / sh_em3_perp_ben at each layer (validated: at L12 it reproduces Phase Y's cosines exactly).
FB — interpretable feature directions as the fair baseline ("we should also do the same experiments on other
directions in order to actually tell how important the evil direction is"). Difference of means of centred
activations at the same four layers, from the stage-1 sample's own labels (`make_meandiff.py`, 500 docs per side):
code vs prose, academic vs casual, math vs prose, health vs other, politics vs other — plus bad vs good medical built
by the SAME method, so there is an EM row that is not a trained steering vector.
Measurement: `ablate_loss.py --layers 12,16,20,24 --tag ml` on the Phase AB eval set (20,390 held-out pool docs,
~10M tokens), all four layers at once and per layer, 6 random directions as the floor. Report `em/ml_report.py`.
`em/scripts/run_ml.sh a|b|c` (two training lanes, then build + ablate).
- Reported as data: Δloss per direction with bootstrap CIs, as a ratio to random, and variance captured as a
  separate column. No pass/fail bar — the question is where the EM directions sit relative to features the model
  demonstrably uses (code vs prose is the positive control; if ablating THAT barely moves loss, the method is too
  weak to say anything about EM).
- Stated now: the absolute effect is tiny (layer-12 ablation of the shared direction cost 0.005% of loss), so ratios
  to random are the only meaningful scale; difference-of-means and trained steering vectors are different
  constructions, which is why `f_badmed` is included as the matched EM row.
- PHASES ML + FB RESULTS (done 2026-10-11 ~23:08 PT; results/ablate/ml/report.txt). All four layers 12/16/20/24
  ablated at once, 20,390 held-out docs, base loss 2.0446, random floor +0.00025.
  Trained steering vectors: sh_em3 3.78x random, sh_em3_perp_ben 3.57x, sports 3.01x, badmed 2.85x, finance 1.95x —
  BUT benign sh_ben 25.35x, vehicles 24.65x, goodmed 5.05x.
  Difference-of-means directions: f_academic 498x, f_politics 317x, f_math 301x, f_code 204x (positive control),
  f_health 156x, f_badmed (bad vs good medical, same construction as the features) 67x.
  Per layer the EM directions peak at L16-20 (sh_em3 2.96 / 4.14 / 4.26 / 2.29), as the Phase AB breakdown predicted.
  → THE FAIR BASELINE KILLS THE SIGNAL. Every interpretable feature is 40-500x the random floor; every trained EM
  direction is 2-4x. Even the benign vehicles steering vector (24.65x) is ~7x more load-bearing than the shared EM
  direction. The positive control passed loudly (code vs prose 204x), so this is a powered null, not a weak method.
  CONFOUND, stated: the two families are built differently and that dominates. Difference-of-means directions lie
  along high-variance directions (var/rand 12.8-31.6) while trained steering vectors do not (1.4-2.9) — the same
  low-salience property we measured in D4/SP. So the honest comparison is within-family, and within the trained
  steering vectors the EM directions are at the BOTTOM, below both benign controls.
  → RETRACTS the Phase SP reading that ablation was "the first measure that separates them": the separation was real
  but the scale is meaningless next to directions the model actually relies on.

### Phase SH: the shared direction at other layers, and CPT on documents it selects — PRE-REGISTERED 2026-10-11 ~01:45 PT
Part 2 — A2 is currently layer-12 only. Steer shml_L{16,20,24}_{sh_em3, sh_em3_perp_ben, sh_ben} at the same norm
(14.06), n 50, Luna. If the shared direction causes EM at every layer it is a property of the model, not of layer 12.
Part 3 — CPT on documents selected by `12_sv_sh_em3_perp_ben` (the EM-specific component), 10M tokens, 3 seeds, then
the bad-medical full FT with no KL; comparator p_f_gen (no CPT). PLUS a random-document CPT arm at the same recipe,
because the document inspection below shows the selected text is low quality and any effect would otherwise be
confounded with quality rather than content. `em/scripts/run_sh.sh`; reports sh_layers_report.py / sh_cpt_report.py.
- DOCUMENT INSPECTION FIRST (`em/shared_docs_read.py` → results/spectrum/shared_docs_read.txt), which lowers the
  prior: ranking by the stronger statistics barely agrees with the mean-projection ranking used earlier
  (Spearman z_mean vs z_msq = 0.131). Top-200 by msq: sports_and_fitness 4.31x, home_and_hobbies 2.89x. The actual
  top documents are consumer web, SEO and boilerplate — a privacy policy, a handgun review thread, an embroidery
  how-to, an email-marketing post, a bourbon listicle, weight-loss testimonials, a freelance-bid spam page, a raw
  user-agent string, garbled Russian SEO — with NO misalignment theme (no hostility, harmful advice or deception),
  unlike workstream 1's layer-12 blind labelling. Curated technical sources are depleted (stack_edu 0.55x, science
  PDFs 0.49x). Expect the workstream-1 confound to recur (top arm CPT loss 2.73 vs random 1.76); the report prints
  CPT loss and source mix for both arms so it is visible.
- PHASE SH RESULTS (done 2026-10-11 ~09:07 PT; results/spectrum/shared_layers.txt, shared_cpt.txt).
  PART 2, layer sweep (all at norm 14.06, n 50): sh_em3 general / medical — L12 24.5 / 49.0%, L16 9.5 / 27.0%,
  L20 5.5 / 16.5% (n.s.), L24 4.0 / 14.6% (n.s.); sh_em3_perp_ben L12 17.0% then n.s. from L16; sh_ben n.s. at every
  layer. → the shared direction's EM effect is concentrated at layer 12, weak at 16, gone by 20.
  CONFOUND, found after the fact: the trained badmed steering norm GROWS with depth (14.06 / 19.09 / 28.56 / 38.37 at
  L12/16/20/24), so steering everything at 14.06 under-pushes the deeper layers. The decay is corroborated at natural
  norms by Phase S (trained L20 5.2%, L24 4.5% vs L12 28.2%), but the magnitudes here are not comparable across
  layers. A norm-matched rerun would settle it.
  Note the tension with Phase ML: the ABLATION signal peaks at L16-20 (sh_em3 2.96 / 4.14 / 4.26 / 2.29) while the
  BEHAVIOUR peaks at L12 — where the direction matters for predicting text is not where it produces misalignment.
  PART 3, CPT on documents selected by sh_em3_perp_ben (10M tokens, 3 seeds) + a random-document control:
  after CPT before FT — top 5.0 / 5.5 / 3.3% general (untouched 3.8%), random 2.0%.
  after the bad-medical FT (n 100) — top 19.8 / 22.4 / 21.9%, random 20.4%, no CPT 17.9%.
  Pooled: top − no CPT general +3.5 pp [+0.2, +6.5] p 0.038, medical +6.5 pp [+2.5, +10.5] p 0.0014;
  top − RANDOM CPT general +1.0 pp [−2.4, +4.1] p 0.58, medical +1.4 pp [−2.6, +5.3] p 0.51.
  → NULL, and the control is what shows it: CPT raises post-fine-tuning EM a little, but selecting documents by the
  shared EM direction does no better than selecting at random. The apparent "+3.5 pp vs no CPT" is an effect of
  continued pretraining itself. Selection is extreme (99.3% Common Crawl vs 89.8% in the random arm, z 2.68-7.65),
  so the arm did what it was asked; the content just does not matter.
- PHASE SH, before-fine-tuning comparison (analysed 2026-10-11 ~11:50 PT). General MR / coherence / MR among coherent:
  untouched 3.75% / 57.5% / 6.5%; top docs (3 seeds) 4.60% / 51.5% / 8.9%; random docs (1 seed) 2.01% / 46.4% / 4.3%.
  top − random +2.59 pp p 0.025, and +4.60 pp p 0.043 restricted to coherent responses (so not a coherence artifact);
  top − untouched +0.85 pp p 0.57 (n.s.). Medical: top − random +3.24 pp p 0.085; top − untouched −1.94 pp p 0.34.
  → Selecting documents by the EM direction leaves the model MORE misaligned than continued pretraining on random
  documents, but NOT more misaligned than the released checkpoint — random-doc CPT lowers misalignment below baseline
  (2.0% vs 3.8%) and top-doc CPT holds it at baseline. The effect does not survive the bad-medical fine-tune
  (+1.0 pp vs random, p 0.58). This REPLICATES the workstream-1 pattern (top 8.1% vs random 4.5%, p 0.041 before
  training, null after) with three seeds and a better direction.
  Weakest link: one random seed against three top seeds, and the selected documents are lower quality (CPT final loss
  2.515 vs 2.087), so quality remains confounded with content. Two more random seeds queued (`em/scripts/run_rs.sh`,
  CPT + pre-FT eval only; the post-FT comparison was already null) → results/spectrum/shared_cpt_preft.txt.
- PHASE NM RESULTS (done 2026-10-11 20:50 PT; results/spectrum/shared_layers_normmatched.txt). The Phase SH layer
  sweep repeated at each layer's OWN trained steering norm (14.06 / 19.09 / 28.56 / 38.37 at L12/16/20/24), removing
  the fixed-norm confound. sh_em3 general MR, matched (fixed 14.06 in brackets): L12 24.5% (24.5), L16 10.8% (9.5),
  L20 8.5% (5.5), L24 6.0% (4.0); medical 49.0 / 32.2 / 20.6 / 13.8%. sh_em3_perp_ben: 17.0 / 11.1 / 7.0 / 6.5%
  (p 4.6e-10 / 7.2e-05 / 0.043 / 0.08). sh_ben n.s. at every layer under both norms.
  → Norm matching raises the deeper layers as expected but does NOT change the conclusion: the shared direction's EM
  effect decays monotonically with depth, still significant at L16 and L20, borderline at L24 (p 0.08-0.14).
  Coherence also falls with depth (91 / 79 / 67 / 62%), so part of the remaining decay is the coherence gate.
  The confound is resolved: the decay is a property of the model, not of the fixed norm. It matches Phase S for the
  trained bad-medical vector (28.2% at L12 vs 5.2% at L20) and still contrasts with Phase ML, where the ABLATION
  signal peaks at L16-20 — where the direction matters for predicting text is not where it produces misalignment.

### Phase RD: is EM predictable from base-model representational distance? — 2026-10-12 ~11:20 PT
Replication test of "Emergent Misalignment Is Not Magical" (arXiv 2608.29118), which reports average Spearman −0.73
over 12 settings between an eval prompt's distance to the fine-tuning data's centroid in the BASE model and how
misaligned the fine-tuned model is on that prompt. Ours (`em/repdist.py`, stage 1, mean-pooled residual, 400 examples
per dataset, 16 eval questions): centroid from prompt+answer, eval questions prompt-only.
- RESULT with the rank-1 FINE-TUNES (their design), layer 12: all six datasets have the predicted negative sign —
  bad medical −0.351, risky financial −0.222, extreme sports −0.402, insecure code −0.496 (p 0.051), good medical
  −0.519 (p 0.039), vehicles −0.393; MEAN −0.397 vs their −0.73. Across datasets, Spearman(mean distance, overall MR)
  −0.714 p 0.11 (n = 6) — close to their figure but underpowered.
- With STEERING VECTORS instead of fine-tunes the mean is only −0.242 and two datasets have the wrong sign, so the
  intervention type matters and their fine-tuning design is the one that works.
- PARTIAL REPLICATION: right sign everywhere, about half the magnitude. Caveats: 16 eval questions, one model family,
  mean-pooling and layer choice are ours (layers 8/12/16 all give −0.20 to −0.24 for steering, −0.40 for LoRA at 12).

### Phase SY: does FORMAT explain the continued-pretraining null? — PRE-REGISTERED 2026-10-12 ~11:30 PT
Vetter et al. (2608.11025) found fine-tuning on the human-written pretraining documents their attribution retrieved
does NOT reliably induce EM, while synthetic instruction-response pairs from the same content DO. Phase SH used raw
human-written text and was null, which is consistent. Test the format half: rewrite each selected document as a user
question + assistant answer reporting that document's own substance (`em/make_synth.py`, gpt-6-luna, 2,000 docs per
arm), then full fine-tune (Phase F recipe) and evaluate at n 100.
Arms: synth_top (documents by msq on 12_sv_sh_em3_perp_ben), synth_random (matched control, so "web text in assistant
format" is visible), and bad medical subsampled to the same size (so dataset size is not a confound).
- Cost probe (40 docs): 1,126 input / 165 output tokens per call, 90% kept → ~4.5M input + 0.66M output for the run.
- STATED NOW: our attribution retrieves different documents from theirs. Their SAE persona features returned villain
  narratives; our activation projection returns privacy policies, SEO pages and forum threads. So their format effect
  was shown on dark content and ours is tested on benign content, which lowers the expected effect. The disagreement
  between the two attribution methods about what EM-related pretraining text looks like is itself a result.
- Comparators: Phase SH raw-document CPT 4.60% general (vs 3.75% untouched, p 0.57); p_f_gen 17.9% on the full
  bad-medical set. `em/scripts/run_sy.sh`, report `em/sy_report.py`.
- PHASE SY RESULTS (done 2026-10-12 12:04 PT; results/spectrum/synth_format.txt). General / medical MR at n 100
  (untouched 3.1 / 10.6%): synth_top 1.6 / 9.9% (coh 47%), synth_random 1.4 / 8.0% (coh 41%), size-matched bad
  medical 18.8 / 45.4% (coh 90%).
  top − random +0.3 pp [−1.0, +1.5] p 0.84; top − untouched −1.5 pp [−3.1, +0.0] p 0.05 (i.e. BELOW baseline);
  top − size-matched bad medical −17.1 pp p 1.6e-33.
  → NULL, and in fact slightly below baseline. Reformatting the selected documents into assistant-style Q&A does NOT
  induce EM, so FORMAT does not explain the Phase SH continued-pretraining null in our setting. The size-matched
  bad-medical arm reaching 18.8% (vs 17.9% on the full 7,049 examples) shows ~1,880 examples is plenty to induce EM,
  so this is not an underpowered fine-tune.
  We therefore do NOT reproduce Vetter et al.'s (2608.11025) format effect. The most likely reason was pre-registered:
  their SAE-persona attribution retrieved villain narratives, ours retrieves privacy policies, SEO pages and forum
  threads (sample synthetic pair from the random arm: a GTA mission walkthrough). Their result is that format matters
  GIVEN dark content; ours is that format does not rescue benign content. The two attribution methods disagree about
  what EM-related pretraining text looks like, which is the more interesting finding.
  Coherence note: both synthetic arms sit at 41-47% vs 90% for bad medical, so fine-tuning on reformatted web text
  degrades the model; the low MR is partly that gate.

### 20M-document pool scored by the shared direction — 2026-10-22 ~18:30 PT (user request; data annotation, no test)
Why: revisiting pretraining-data attribution at a larger scale (Neel thinks the CPT line was not scaled far enough). The
earlier pools were 336,619 documents (data/pool; Phase SH selected its 10M-token CPT arm from it, the top 3.6%) and
1,157,497 (data/pool_v2). In a 20M pool the top 3.6% is about 720k documents.
- Sample (`em/proj_score.py --make_plan`, `results/attribution/pool20m_plan.json`): about 21M documents, uniform over the
  stage-1 mix (allenai/dolma3_mix-6T-1025-7B at revision 2ca900fb; its card calls it the full Olmo-3-1025-7B pretraining
  mix, 5.93T tokens and 3.87B documents, and step 1,413,814 x 4M-token batches is also 5.93T): the same 0.543% of
  every data dir's bytes, from at least 4 random shards per dir (1,212 shards, 57.0 GB). Expected mix by documents:
  common_crawl 81.4%, stack_edu 13.6%, finemath 2.5%, science PDFs 2.2%. About 95% of documents reach the 32 tokens
  needed to be scored (24 shards per source), so about 20M get scores. Unlike data/pool, which drew documents per dir
  in proportion to bytes (weighting long-document dirs up), so source shares differ from data/pool's. Some science-PDF
  texts were redacted by AllenAI after training ("[REMOVED]"); the text files flag them.
- Score: Phase SH's statistic unchanged (mean layer-12 residual over the first 512 tokens, positions >= 16, centred with
  the s1_l12_shared mean, z against its 32 random directions), for sh_em3 and sh_em3_perp_ben. Document means saved
  (fp16) so other layer-12 directions can be scored later without a GPU.
- Scorer validated against pool_score.py on 4 x 20k pool documents: corr 0.99999, mean |dz| 0.0048, 99.9th percentile
  0.027 (one sample), max 0.56 (one document in 80k). Throughput 235-243 docs/s per H100 compiled (958 on lambda-4xh100-2); about 6 h on 4 H100s.
- `em/scripts/run_pool20m.sh` on lambda-4xh100-2 → data/pool20m/ (text, scores, means per shard; merged scores.parquet).
- 19:21 PT: lambda-2xh100 joined. `em/scripts/pool20m_reserve.py 0.32 lambda-2xh100` claimed 316 unclaimed shards
  (5.95M expected docs) on lambda-4xh100-2 and wrote data/pool20m/plan_lambda-2xh100.json, which lambda-2xh100 scores
  into its own data/pool20m/. lambda-4xh100-2's final merge will then stop at its 1,212-shard check: copy the other
  box's scores/ over and rerun run_pool20m.sh, which skips every scored shard and merges.
