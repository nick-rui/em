# CPT under OLMo's own stage-1 recipe: E0 smoke, E1 recipe check, E2 TOP vs NONE

Written 2026-11-13 (from 01:00 PT) on claude-box. Design agreed with Nick on 2026-11-13; this file writes it down before
any GPU result. Sections marked "filled after" get numbers from runs, with the time they were added. Nothing in
sections 1-5 changes after E1 starts except in the deviations log (section 8).

"Checked" = read in a file, log, config or code today. "Assumed" = inferred, not verified.

## 0. Why

The paper's headline: CPT of OLMo-3-7B stage 1 on 1B tokens of the documents that project highest on the shared EM
direction (TOP) raises EM susceptibility. The sprint result has a known weakness: after EM fine-tuning TOP tracks no
CPT, and the TOP - RANDOM gap comes from RANDOM CPT lowering EM (RANDOM - no CPT -1.6 / -4.0 / -3.2 points on medical
/ finance / sports; `~/code/em-olmo-research/research/recompute/cpt_olmo.md`). The sprint recipe also moved the weights
more per token than OLMo's own training (8x smaller global batch at the same per-step LR, 1,024-token cuts, copies
kept; `research/cpt_audit/recipe_audit.md`). The steering experiment (`results/steer_cpt/report.txt`, 11-12) adds that
both old-recipe CPT models shrank the residual stream (RMS |h| 14.06 -> about 13.2) and RANDOM lost coherence faster
under steering. Neel wants a headline figure of a CPT arm sitting above a control that equals no CPT, with 3 seeds.

Nick's decisions (11-13): OLMo only; TOP vs NONE under OLMo's own recipe; one RANDOM run as the check that NONE is a
fair control; the text-classifier arm deferred (its luna labelling prompt is to be reviewed first; not built here);
one step at a time; the 2xH100 for now, maybe an 8xH100 node later.

## 1. Design (agreed)

**Model.** OLMo-3-1025-7B stage 1, `stage1-step1413814`. No Qwen.

**Arms.**

| Arm | What | Role |
|---|---|---|
| TOP | CPT on pool documents ranked by `z_sh_em3` (below) | experimental |
| NONE | no CPT: stage 1 itself | control |
| RANDOM | CPT on random pool documents, one run | check that NONE is a fair control |

The pool is a uniform sample of shards of the stage-1 mix (`allenai/dolma3_mix-6T-1025-7B`), so CPT re-exposes the
model to documents it saw once in stage 1. If RANDOM under the new recipe lands on NONE, the headline is TOP vs NONE.

**Selection score.** `z_sh_em3`: the layer-12 stage-1 residual over a document's first 512 tokens (positions >= 16),
centred, averaged, projected on the shared EM direction `sh_em3` (the normalised, sign-aligned mean of the three
layer-12 G steering vectors for bad medical, risky finance and extreme sports; `em/make_shared.py`,
`em/proj_score.py`), z-scored against 32 random directions. Scores: `data/pool20m/scores.parquet`.

**Recipe** (OLMo's own stage-1 recipe). Values checked today against the run config saved with the checkpoint itself
(`buckets/allenai/ai2-llm/checkpoints/OLMo25/step1413814/config.json`, run OLMo25, launched from OLMo-core branch
`dirkg/Retrofit` at `5878bb48`):

| Setting | New recipe | Source / status |
|---|---|---|
| Start | OLMo-core checkpoint `OLMo25/step1413814` with fp32 masters and AdamW state; `load_optim_state=True`, `load_trainer_state=False` | checked present: 87.60 GB, 128 non-empty `.distcp` files of 8,192 (512 ranks), `.metadata` lists `exp_avg` / `exp_avg_sq` / `step` for every parameter (3,905 `exp_avg_sq` strings). Downloaded to the box 11-13 |
| LR | 3.93e-5 constant, no warmup | OLMo's logged LR at the last step 3.9325e-5 (wandb run `ej79z38y`, checked) |
| Global batch | 4,194,304 tokens (512 x 8,192), 238 steps per 1B tokens | config `data_loader.global_batch_size` |
| Micro batch | 8,192 tokens (1 sequence) per GPU | config `train_module.rank_microbatch_size` |
| Context | 8,192; OLMo's sliding-window pattern [4096, 4096, 4096, full], full attention on the last layer | config `model.block.attention.sliding_window` |
| Document masking | none (`generate_doc_lengths=False`) | config |
| Documents | whole documents + EOS (100257), no BOS, no truncation, concatenated and cut into 8,192-token windows by OLMo-core | data rule (section 1.1) |
| Repetition | one copy per copy group | data rule (section 1.1) |
| Optimizer | SkipStepAdamW, betas 0.9 / 0.95, eps 1e-8, wd 0.1 (none on embeddings), `step_increment_bugfix`, 128-step window, 6 sigma | config `train_module.optim`; checked field by field at launch by `--check_config` |
| Loss | z-loss 1e-5, clip 1.0 | config |
| Precision | FSDP, bf16 params, fp32 reduce, compiled | config uses HSDP over 512 GPUs; FSDP over 2 is the same math |
| Attention kernel | flash-attn 2 (2.8.3.post2, built on the box) | config `use_flash = true` |
| Activation checkpointing | only if the state does not fit (E0) | not in OLMo's config; changes speed, not the math |

Fallback if the state cannot be loaded: the same recipe from the bf16 HF weights with fresh optimizer state (30 warmup
steps, as the audit proposed). Reported if used; never a silent switch.

Code: `em/cpt_olmocore.py` (new flags; the old defaults are the sprint recipe and are unchanged), data in
`em/cpt_recipe/{whole_lists,extract,assemble}.py`.

### 1.1 Data rule

- A copy group is all pool documents with the same (shard, round(z, 5), round(perp z, 5), n_tokens) (the research
  pass's key; `research_copy/build_manifests.py`). The pool has 21,812,566 documents in 9,588,313 groups (checked).
  Groups holding a labelled document or a held-out document are excluded from every arm (124,956 groups).
- An arm takes groups and keeps one document per group (the group's first document in pool file order). Its whole
  text is tokenized with the OLMo-3 tokenizer at `stage1-step1413814` (`add_special_tokens=False`) and EOS is appended.
  This should equal OLMo's own token stream for that document (assumed, not compared with OLMo's token files).
- Tokens come from the Dolma 3 shards by `doc_uid` (`<dir>/<shard file>:<line>`), read once, on the box
  (`em/cpt_recipe/extract.py`). The sprint's pool text is gone.
- Each arm's documents are shuffled (seed) and concatenated; OLMo-core cuts 8,192-token windows and its data loader
  shuffles the windows (seed 0).

### 1.2 Measures, fine-tunes and evals

As sections 1.4 and 2.3-2.5 of `research/prereg/cpt_research_phase.md`: full fine-tunes with `em/train.py` (the CPT
workstream stays on train.py for every arm, never train_fast.py), em_eval_v2 chat 76 questions x 10 with gpt-6-luna at
steps 0 / 10 / 25 / 50 / 100 / 200 / end, in-domain sets, coherence.

## 2. E0: smoke test (lambda-2xh100-n)

50M tokens on the new recipe, RANDOM documents: rand1b_s21 copy groups drawn in proportion to their copies
(deviation 3), one copy each, whole documents, until 50M tokens (12 optimizer steps). Not analysed. Questions:

1. Does OLMo's optimizer state load, with `initial_lr` reset to ours and the LR at 3.93e-5 from step 1?
2. Does it fit in memory on 2 H100s (FSDP; about 58 GB per GPU of fp32 masters, AdamW moments and fp32 gradients)?
   If not, activation checkpointing first, before anything else, and its speed cost is reported.
3. Throughput (tokens/s per GPU).
4. Do the first training losses sit near stage 1's last logged values? Reference, checked from OLMo's public wandb run
   `ai2-llm/Olmo-3-1025-7B/ej79z38y` (`results/cpt_recipe/olmo_wandb/`): over the last 1,001 steps, training CE mean
   1.8445, SD 0.0225 per step (min 1.773, max 1.910); pre-clip grad norm mean 0.119; 64 skipped steps in the last
   10,000. Our pool documents are a sample of the same mix, so the first step's loss should fall in that range.

## 3. E1: recipe check (lambda-2xh100-n)

### 3.1 Arms

| Arm | Data | Recipe | Tokens | Steps |
|---|---|---|---|---|
| V-new | RANDOM: rand1b_s21's copy groups drawn in proportion to their copies (deviation 3), one copy each, whole documents, until 200M | new (section 1) | 200M | 48 |
| V-old | RANDOM: rand1b_s21's rows in take order with every copy, first 1,024 tokens + EOS (the sprint's materialize.py rule), until 200M | old: `em/cpt_olmocore.py` defaults (524,288 tokens per step, context 1,024, LR 3.9e-5, 100 warmup steps, fresh SkipStepAdamW, start from the bf16 HF weights converted with OLMo-core's converter) | 200M | 382 |
| Reference drift | OLMo's own checkpoints `step1412000` and `step1413000` vs `step1413814` (OLMo-core, fp32, downloaded 11-13) | OLMo | 7.61B and 3.41B tokens before the end | 1,814 and 814 |
| OLMo's in-loop evals | wandb history at steps 1,380,000 / 1,390,000 / 1,400,000 / 1,410,000 (41.9B tokens apart) | OLMo | | |
| Old recipe at 1B (free extra point) | the sprint's RANDOM, `cpt_rand1b_hf/step1907` (read from `~/em-olmo-steer` on the box, converted to OLMo-core in our own directory) | old | 1B (0.43B unique) | 1,907 |

V-old runs if E0 shows it is cheap enough (about 2.5 h on 2 GPUs was the estimate). V-old and V-new start from mostly
the same documents (the first groups of the same take order).

### 3.2 Measures

For V-new, V-old and the three OLMo checkpoints, each with a bootstrap CI over documents or items where one applies:

- Held-out cross-entropy on `heldout_random` (3,000 documents, sprint cut: first 1,024 tokens + EOS, one document per
  padded 1,024 instance), comparable with the sprint's TOP 1.909 -> 2.09.
- Cross-entropy on OLMo's `v3_small_ppl_validation` (11 sets, dolma2 tokens from olmo-data.org, 6.69M tokens) at
  8,192. This text is not in the pool.
- OLMo's 26 in-loop downstream tasks from the run config (ARC, HellaSwag, MMLU, basic skills, code and math; bits per
  byte and accuracy), with OLMo-core's own evaluator.
- Relative weight change ||theta - theta_0|| / ||theta_0|| per block (fp32 masters; V-old against bf16(theta_0), its
  own start).
- em_eval_v2 chat (76 questions x 10, generation seed 2, 400 new tokens), general misalignment (headline = mean of the
  6 category rates), coherence and misaligned among coherent, before any fine-tune; gpt-6-luna.
- Training loss, pre-clip grad norm and skipped steps per step (V-new, V-old), next to OLMo's logged values.

All evaluated models use the same code (`em/cpt_recipe/eval_ckpt.py`, OLMo-core's evaluators, flash-attn 2, bf16).
Check of the eval pipeline: our step1413814 numbers next to OLMo's logged in-loop values at step 1,410,000.

### 3.3 Analysis

For each measure: the change per billion tokens for V-old and V-new, next to OLMo's own change per billion tokens
(step1412000 -> step1413814 and step1413000 -> step1413814, and the wandb in-loop series). No pass/fail bar. The
reference drift has two intervals (plus four wandb points), so it is a range, not a noise distribution. Patterns to
describe with the numbers (not verdicts):

- V-new near OLMo's own per-token drift, V-old well above it: the old recipe was disruptive, the new one is not.
- Both near it: recipe disruption was not large; use the new recipe anyway for the batch and state match.
- Both well above it: the new recipe still disrupts; the cause would be what is left (data re-exposure, document
  choice); report and decide with Nick before E2.

Caveat: the held-out documents and the CPT documents were all in stage-1 training, so held-out loss measures drift on
seen data. v3_small_ppl is the unseen-text measure.

## 4. Pre-registered rule

The recipe is fixed from the E1 report before any EM fine-tune of a new-recipe CPT model is judged. The recipe choice
cannot depend on an EM result. The em_eval_v2 numbers of E1 (before any fine-tune) are part of the recipe check and are
reported, but they do not choose the recipe on their own.

## 5. E2 plan (needs Nick's go; probably a bigger node)

- CPT arms, 1B tokens each under the fixed recipe: TOP x 3 runs (3 draws or 3 data orders, see open decision 2),
  RANDOM x 1. NONE needs no CPT.
- Downstream per model (3 TOP + RANDOM + NONE): 5 datasets (bad medical, risky finance, extreme sports; held-out
  legal, math) x 3 fine-tune seeds = 15 full fine-tunes with `em/train.py` (LR 1e-5, 1 epoch, batch 2 x accum 8,
  max_len 2048, adamw8bit, `--save_at 10,25,50,100,200`), em_eval_v2 at steps 0 / 10 / 25 / 50 / 100 / 200 / end, in-domain
  sets at steps 0 / 25 / end. 75 fine-tunes in all.
- Primary contrasts: TOP - NONE (headline) and RANDOM - NONE (control check), on D0, D_early, D_end as research
  pre-registration section 2.5, pooled over medical / finance / sports; per-run differences with a t interval on 2 df,
  and the hierarchical bootstrap.
- Compute: filled after E0 (section 7).

## 6. Open data decisions (quantified here, decided by Nick)

Numbers filled 2026-11-13 01:30 PT from the extract (3,874,054 documents, 7.20B whole-document tokens;
`results/cpt_recipe/data_decisions.{txt,json}`, `em/cpt_recipe/data_decisions.py`). The sprint cut for TOP groups is
approximated as min(whole - 1, 1,024) + 1; on rand1b_s21's 1.55M documents, where the exact sprint counts exist, this
gives 1,058,885,650 vs 1,058,885,537 tokens (2 documents differ).

1. Token rule, 1B TOP tokens, highest z first over eligible groups:

   | Rule | Groups | Documents | z reached | Mean z (docs) | Mean z (tokens) | Unique tokens |
   |---|---|---|---|---|---|---|
   | sprint: first 1,024 + EOS, every copy | 502,479 | 1,199,140 | 2.039 | 2.653 | 2.665 | 0.42B |
   | 1,024 cut, one copy | 1,229,289 | 1,229,289 | 1.364 | 2.064 | 2.080 | 1.00B |
   | whole, every copy | 177,836 | 433,650 | 2.688 | 3.247 | 3.220 | 0.41B |
   | **whole, one copy (new rule)** | 436,386 | 436,386 | 2.133 | 2.725 | 2.720 | 1.00B |

   The sprint row reproduces the sprint's TOP (z >= 2.05, mean 2.65; 98.4% of its groups are in cpt_top1b). Whole
   documents are 2.2x longer than the 1,024 cut on average (TOP mean 2,292 tokens, median 1,088, p99 17,568, max
   1,087,610), which offsets the 2.3x copies that deduplication drops: the new rule reaches a higher z than the sprint.
2. The 3 TOP runs under the new rule:

   | Design | Slice | z reached | Mean z per draw (docs) | Token overlap between draws |
   |---|---|---|---|---|
   | strict top 1B (3 runs differ only in data order and fine-tune seeds) | 436,386 groups | 2.133 | 2.725 | 100% |
   | random halves of the top 2B (as top2b_draw_s11-13) | 902,366 groups | 1.615 | 2.273 | 50% |
   | disjoint thirds of the top 3B | 1,397,015 groups | 1.253 | 1.973 | 0% |

3. (Found here.) RANDOM's group weighting. The rand1b manifests draw copy groups uniformly; the sprint kept every copy,
   so its token stream weighted groups by copies, as OLMo's own stream does. Deduplicated, the two weightings differ.
   Token shares by source (whole documents, one copy, over rand1b_s21's 683,192 groups):

   | Source | stage-1 mix (audit [S4]) | take order (group-uniform), first 200M | copies-weighted, expected | copies-weighted, V-new's 200M |
   |---|---|---|---|---|
   | common_crawl | 76.1% | 66.1% | 76.5% | 75.5% |
   | olmocr_science_pdfs | 13.6% | 18.9% | 14.6% | 15.3% |
   | stack_edu | 6.9% | 8.7% | 5.4% | 5.5% |
   | finemath | 2.6% | 4.2% | 2.5% | 2.7% |
   | rpj (arXiv) | 0.86% | 2.2% | 0.96% | 0.94% |

   Copies-weighting matches the stage-1 mix; group-uniform does not. V-old (sprint rule) is 86.3% Common Crawl and
   3.3% PDFs (the truncation shift the audit found). E0 and V-new use copies-weighting (deviation 3). For E2's RANDOM
   arm the same choice applies; Nick decides.
4. (Found here.) The selection score covers a document's first 512 tokens. Under the new rule 78.6% of TOP's tokens
   are past position 512 of their document, 63.4% past 1,024 and 17.0% past 8,192: unscored. The sprint measured
   positions 512-1,023 of TOP documents still on-direction (+2.66, `em/TODO.md`); beyond that is unmeasured. Options
   for Nick: whole documents as they are; whole documents scored on more windows (GPU, a pass over TOP's 1B tokens);
   or a cut (which is not OLMo's recipe).

## 7. Results (filled after)

## 8. Deviations log

All recorded before any training run.

1. 2026-11-13 01:22 PT. The LM evals run each instance at its real length instead of padded to the full length
   (`eval_ckpt.py`). Same instances, same model call, same metric; padded, OLMo's v3 small sets cost 152M positions for
   6.69M tokens (about 105 min per model on one H100; 4 min unpadded). Check against OLMo's logged in-loop values at
   step 1,410,000, our step1413814 (OLMo's definition, deviation 2): section 7.
2. 2026-11-13 01:27 PT. The reported LM "CE loss" uses OLMo-core's own denominator (every real position, including
   each instance's last position whose label is masked and whose loss is 0), so it matches OLMo's logged values and
   the sprint's held-out 1.909. The plain mean over valid labels is reported next to it.
3. 2026-11-13 01:30 PT. E0 and V-new draw rand1b_s21's copy groups in proportion to their copies (Efraimidis-Spirakis
   weighted sampling without replacement, seed 21; `assemble.py rand_new_cw`) instead of in take order, so the token
   mix matches the stage-1 mix (section 6, item 3). Still RANDOM documents from rand1b_s21, one copy each.
4. 2026-11-13 02:10 PT. E0's first try (no activation checkpointing) ran out of memory while loading OLMo's optimizer
   state, before any training step: torch creates the AdamW state with a dummy step, and OLMo-core's multi-tensor
   (foreach) AdamW allocates full-size temporaries for every parameter at once, on top of 58 GB per GPU of fp32
   masters, gradients and moments. E0, V-new and V-old use OLMo-core's per-tensor AdamW (`--no_foreach`): the same
   update (`lerp` instead of `mul` + `add` for the first moment, otherwise the same operations), temporaries for one
   tensor at a time. OLMo's run used foreach. Log: `logs/cpt_recipe_e0_noac_foreach_oom.log` on the box.
5. 2026-11-13 02:05 PT. The sprint's 1B old-recipe RANDOM (`cpt_rand1b_hf/step1907`) is added as the extra E1 point
   (as the research pre-registration's "free extra point"), read from `~/em-olmo-steer` on the box (orchestrator, 11-13).
   Its em_eval_v2 before fine-tuning is the sprint's judged file (`ev2_cptlc_rand1b_t000`); its LM evals are rerun.
6. 2026-11-13 02:05 PT. Old-recipe models start from the bf16 HF export converted by OLMo-core's converter. Checked:
   every transformer block of that start equals bf16(OLMo's fp32 masters) exactly (relative difference 0.0); the
   embedding and lm_head matrices differ (0.60% / 0.82%), because the converter pads HF's 100,278-token vocab to
   100,352 with mean-initialised rows. So V-old's weight change is measured from its own start, and that start is
   evaluated too (LM evals).
7. 2026-11-13 02:00 PT. The first two reference evals (step1413814, step1413000) crashed after their downstream evals,
   before writing results (the eval-only trainer checked a dummy batch size). Their downstream metrics were recovered
   from the logs (4 significant digits; `parse_downstream_log.py`); their LM evals are rerun for the per-document
   arrays. Fixed in `eval_ckpt.py`.
