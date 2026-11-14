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

- CPT arms, 1B tokens each under the fixed recipe: TOP x 3 runs and RANDOM x 1; NONE needs no CPT. The 3 TOP runs
  train on the same documents, the strict top 1B (Nick, 11-13; section 6), in 3 data orders (document shuffle and
  data-loader seeds 1, 2, 3), with fine-tune seeds as below. So the 3 runs measure training-order noise only; they do
  not sample document choice, and the TOP result is about this one document set.
- Downstream per model (3 TOP + RANDOM + NONE): 5 datasets (bad medical, risky finance, extreme sports; held-out
  legal, math) x 3 fine-tune seeds = 15 full fine-tunes with `em/train.py` (LR 1e-5, 1 epoch, batch 2 x accum 8,
  max_len 2048, adamw8bit, `--save_at 10,25,50,100,200`), em_eval_v2 at steps 0 / 10 / 25 / 50 / 100 / 200 / end, in-domain
  sets at steps 0 / 25 / end. 75 fine-tunes in all.
- Primary contrasts: TOP - NONE (headline) and RANDOM - NONE (control check), on D0, D_early, D_end as research
  pre-registration section 2.5, pooled over medical / finance / sports; per-run differences with a t interval on 2 df,
  and the hierarchical bootstrap.
- Compute (filled 2026-11-13 05:00 PT from E0 and today's runs; "measured" = on lambda-2xh100-n today):

  | Item | 2 x H100 | 8 x H100 |
  |---|---|---|
  | CPT speed | 7,475 tokens/s per GPU, measured (full activation checkpointing, per-tensor AdamW) | about 9,900 per GPU, assumed: no activation checkpointing needed (state 14.6 GB per GPU), so no forward recompute (about +1/3); up to 7,475 if it is needed |
  | One 1B-token CPT arm (238 steps) | 18.6 h wall, 37.2 H100-h | 3.5-4.6 h wall, 28-37 H100-h |
  | 4 CPT arms (TOP x 3, RANDOM x 1) | 74 h wall, 149 H100-h | 14-19 h wall, 112-149 H100-h |
  | Downstream per model (15 full fine-tunes, em_eval_v2 at 7 checkpoints = 91 generations, in-domain 21, Perez 16) | 19.1 H100-h: fine-tunes 2.7, em_eval_v2 13.7 (9 min per generation, measured), in-domain 1.4, Perez 1.3 (research pre-registration timings, assumed) | same |
  | Downstream, 5 models (TOP x 3, RANDOM, NONE) | 96 H100-h, 48 h wall | 96 H100-h, 12 h wall |
  | Total | about 245 H100-h, about 122 h wall (5 days) | about 210-245 H100-h, about 26-31 h wall |
  | Judge (gpt-6-luna) | about $26 (5 models x $5.2, research pre-registration) | same |

  Not included: 88 GB checkpoint download per node (2.5-3.4 min each here), data prep (CPU: TOP arms from the extract
  cache in minutes; the copies-weighted RANDOM 1B needs a larger random sample than rand1b_s21, whose 683,192 groups
  hold 1.10B whole-document tokens, so that a 1B draw does not exhaust it: one more extract pass, about 15 min),
  checkpoint saves (51 s each, async), reruns.

## 6. Data decisions (quantified here, decided by Nick)

**Decided 2026-11-13 (relayed by the orchestrator session):**
- ~09:30 PT, item 2: the 3 TOP runs use the **strict top 1B** (436,386 groups, z >= 2.133, mean z 2.725), differing
  only in data order and fine-tune seeds (Nick: "Strict top 1B x 3"). They cover training-order noise, not
  document choice.
- ~09:35 PT, items 1, 3, 4 (Nick: "Yes, use them"): (1) whole documents, one copy per group; (3) RANDOM drawn
  copies-weighted, as V-new; (4) whole documents as they are, plus a descriptive scoring pass of TOP's later windows
  (cost stated before it runs; section 7.4).

The numbers behind them, as quantified before the decisions:

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

### 7.1 E0 (2026-11-13 02:19-03:18 PT, lambda-2xh100-n; `results/cpt_recipe/raw/e0_acfull_metrics.jsonl`)

1. State load: yes. OLMo's fp32 masters and AdamW moments load from the 512-rank checkpoint onto 2 ranks. LR 3.93e-5
   in both groups from step 1 (`initial_lr` reset to ours); z-loss 0.00071 vs OLMo's last 0.00070. Needs the
   per-tensor AdamW (deviation 4): with OLMo-core's foreach AdamW the load ran out of memory.
2. Memory: without activation checkpointing the first forward pass ran out of memory (77.4 GB allocated). With full
   activation checkpointing (`--ac full`) it fits: 62.6 GB active, 70 GB reserved per GPU.
3. Throughput: 7,475 tokens/s per GPU (12 steps; 281 s per 4.19M-token step), 2 GPUs. Full activation checkpointing
   recomputes the forward pass, about +1/3 compute, so roughly 9,900 tokens/s per GPU without it (assumed, not
   measured: it does not fit on 2 GPUs). Checkpoint save (82 GB with optimizer state) 51 s, async.
4. Training loss: steps 1-12: 1.914, 1.895, 1.921, 1.898, 1.871, 1.896, 1.867, 1.869, 1.846, 1.917, 1.873, 1.886
   (mean 1.888, SD 0.023, fitted slope -0.0029 per step). OLMo's last 1,001 steps: mean 1.8445, SD 0.0225, max 1.910.
   Step 1 is stage 1's loss on our first batch, before any update: 1.914 is 3.1 OLMo SDs above OLMo's mean. Pre-clip
   grad norm 0.149, 0.132, 0.124, then 0.115-0.120 from step 4 (OLMo 0.119). No skipped steps. Held-out CE (OLMo-core
   in-loop): 1.9083 at step 0, 1.9067 at step 12.
   The gap is in the documents, not in the setup: on the same documents our evaluation reproduces OLMo-core's and
   OLMo's own numbers (held-out 1.908 in-loop and 1.9083 offline for stage 1, the sprint logged 1.909; v3 small c4
   2.4695 vs OLMo's logged 2.4691 at step 1,410,000; downstream BPB v2 mean difference -0.003 over 26 tasks). A direct
   check on OLMo's own training windows (`olmo_windows.py`) was not run: olmo-data.org answers 403 to this client.
   Candidate reasons, not tested: the pool is the Dolma 3 text release tokenized by us, not OLMo's token files; our
   documents are deduplicated whole documents drawn by copy group.

### 7.2 E1 references (2026-11-13 by 04:20 PT; `results/cpt_recipe/report_e1.txt`, `evals/`)

Eval pipeline check: our stage-1 numbers vs OLMo's own logged in-loop values at step 1,410,000 (wandb): v3 small CE
mean difference -0.0003 over 11 sets (largest 0.0067, the 4-instance books set); downstream BPB v2 mean difference
-0.003 over 26 tasks. Held-out CE of stage 1: 1.9084 (the sprint logged 1.909).

| Measure | OLMo 1412000 -> 1413814 (7.61B) | OLMo 1413000 -> 1413814 (3.41B) | sprint old-recipe RANDOM, start -> 1B |
|---|---|---|---|
| held-out CE | -0.0002 [-0.0013, +0.0009] | +0.0001 [-0.0008, +0.0010] | +0.1494 [+0.1457, +0.1533] |
| v3 small CE (mean of 11) | +0.0005 [-0.0002, +0.0011] | +0.0004 [-0.0001, +0.0008] | +0.1522 [+0.1483, +0.1556] |
| downstream mean BPB v2 (26) | -0.0020 | -0.0064 | +0.0669 (accuracy -0.0261) |
| weight change, all / median block | 3.73% / 7.91% | 2.65% / 5.65% | 5.24% / 11.30% |
| em_eval_v2 headline, later - earlier | -0.6 [-3.4, +2.4] | -1.2 [-4.3, +1.7] | sprint: +0.1 [-3.0, +3.4] |
| em_eval_v2 coherent, later - earlier | -5.3 [-10.4, +0.0] | -8.2 [-12.6, -3.7] | sprint: +10.0 [+5.1, +14.7] |

OLMo's own end of stage 1 is flat in loss over billions of tokens; its weights still move (about tokens^0.42). The old
recipe's RANDOM raised loss on seen (held-out pool) and unseen (v3 small) text by 0.15 nats in 1B tokens. Coherence
on em_eval_v2 moves by 5-8 points between OLMo's own neighbouring checkpoints, so the sprint's RANDOM +10 is not far
outside OLMo's own spread. E0's 50M new-recipe tokens: held-out -0.0016, block weights 0.54% (diagnostic only).

### 7.3 E1 result (V-new 04:25-08:20 PT, V-old 08:24-11:36 PT, evals by 12:22 PT; `results/cpt_recipe/report_e1.txt`)

Change from each run's own start over the tokens between; 95% CIs paired over documents / instances (LM) or over
questions within category (em_eval_v2). "OLMo" = OLMo's own checkpoints step1412000 / step1413000 -> step1413814.

| Measure | OLMo, 7.61B / 3.41B tokens | V-new (new recipe), 0.20B | V-old (old recipe), 0.20B | sprint RANDOM (old), 1.0B |
|---|---|---|---|---|
| held-out CE | -0.0002 / +0.0001 | -0.0022 [-0.0027, -0.0017] | +0.1204 [+0.1177, +0.1232] | +0.1494 [+0.1457, +0.1533] |
| v3 small CE, mean of 11 sets | +0.0005 / +0.0004 | -0.0005 [-0.0008, -0.0001] | +0.0895 [+0.0874, +0.0917] | +0.1522 [+0.1484, +0.1555] |
| downstream mean BPB v2 (26) | -0.0020 / -0.0064 | +0.0028 | +0.0537 | +0.0669 |
| downstream mean accuracy v2 (13) | +0.0053 / -0.0003 | +0.0005 | -0.0166 | -0.0261 |
| block weights moved (median block) | 7.91% / 5.65% | 1.42% | 5.16% | 11.30% |
| all weights / sqrt(1B tokens) | 0.0135 / 0.0144 | 0.0148 | 0.0528 | 0.0524 |
| em_eval_v2 headline | -0.6 [-3.4, +2.4] / -1.2 [-4.3, +1.7] | -1.8 [-4.5, +0.7] | -3.6 [-6.0, -1.3] | +0.1 [-3.1, +3.6] (sprint files) |
| em_eval_v2 coherent | -5.3 [-10.4, +0.0] / -8.2 [-12.6, -3.7] | -8.3 [-13.0, -3.4] | +0.5 [-3.8, +4.6] | +10.0 [+5.3, +14.3] (sprint files) |
| training CE (per step) | OLMo's last 1,001 steps 1.8445, SD 0.0225 | mean 1.8645, SD 0.018; no trend | 2.05 -> 1.32 (memorising copies) | logs gone |
| pre-clip grad norm | 0.119 | median 0.116 | median 0.381 (max 0.75) | logs gone |

Pattern (section 3.3): V-new sits at OLMo's own drift on every measure: LM losses move by <= 0.003 nats (held-out
slightly down, as expected from more of the same distribution), downstream within OLMo's checkpoint-to-checkpoint
spread, weight drift per sqrt(token) the same as OLMo's (0.0148 vs 0.0135-0.0144), em_eval_v2 within the spread of
OLMo's neighbouring checkpoints. V-old sits well above it on every measure: +0.09 to +0.12 nats of loss on seen and
unseen text after 200M tokens, worse downstream, 3.7x OLMo's weight drift per sqrt(token), training loss falling to
1.32 on repeated copies, and em_eval_v2 headline -3.6 points before any fine-tune. So the old recipe was disruptive and
the new one is not. The sprint's 1B RANDOM shows the same (+0.15 nats). Coherence on em_eval_v2 varies by 5-8 points
between OLMo's own neighbouring checkpoints, so it is a noisy drift measure.

Recipe fixed from E1 (pre-registered rule, section 4; no EM fine-tune of a new-recipe model has been run): the new
recipe as run in V-new (section 1 table), from OLMo's own `OLMo25/step1413814` state. Implementation choices that do not
change the math and depend on the node: on 2 H100s full activation checkpointing and the per-tensor AdamW (deviation 4);
on 8 H100s neither is needed (14.6 GB of state per GPU), assumed, to be checked in a short smoke run. RANDOM documents
drawn in proportion to their copies (deviation 3) unless Nick decides otherwise (section 6, item 3).

Compute used for E0 + E1: about 21 H100-hours (E0 2.3, V-new 7.8, V-old 6.4, evals and generations about 4.7, failed
or repeated evals about 0.6). Judge: 7,700 gpt-6-luna calls, $0.31.

### 7.5 E2 launch

- 2026-11-13 ~13:05 PT, Nick (relayed by the orchestrator): "Start on 2x now". Order: TOP run 1's CPT (strict top 1B,
  data-order seed 1), then its 15 fine-tunes and their evals, then the next CPT arm; the NONE fine-tunes in a gap, or
  after TOP run 1's fine-tunes. Remaining arms move to an 8xH100 node if one arrives.
- Implementation (no change to the math): full activation checkpointing and the per-tensor AdamW on 2 GPUs (as E1); a
  rolling checkpoint every 15 steps besides the kept ones every 60, so an arm can resume on another node.

### 7.6 Partial activation checkpointing test (planned 2026-11-13 16:30 PT; Nick 16:20 PT "Yes, test it")

Speed only; the math is unchanged. In the gap after TOP run 1's CPT and before its fine-tunes, under ~1 h of the node
(`em/cpt_recipe/ac_test.sh`, report `ac_test_report.py`): the E2 recipe with OLMo's state loaded, 12 steps of
262,144 tokens (16 micro-batches per GPU) on the same batches, no saves or evals, for `--ac full` (baseline),
`budget:0.1`, `budget:0.25`, `budget:0.4` and `blocks:2` (larger ones skipped once a smaller one runs out of memory).
Measured per config: forward/backward tokens/s per GPU (the once-per-step optimizer time t_opt is separated with
T(n) = t_opt + n t_micro, from full AC at n = 16 here and n = 512 in TOP run 1), the projected real-run speed, peak
allocated / reserved memory per step and after the state load (max over both ranks), allocator retries, and the
step 1-3 losses next to full AC's. Adoption rule, fixed now: the fastest config whose step-1 loss matches full AC's
(bf16 nondeterminism only), with no allocator retries and at least 4 GiB between its peak reserved memory and the
79.2 GiB capacity (shapes are fixed every step, so 12 steps show the per-step peak, and the 4 GiB covers
fragmentation over 238 steps). If none is clearly faster, full AC stays. An arm that runs out of memory mid-run resumes
from its rolling checkpoint with `--ac full`.

### 7.7 Partial activation checkpointing result (2026-11-14 08:32-08:54 PT; `results/cpt_recipe/ac_test.txt`, `.json`)

Adopted: `--ac budget:0.25 --no_foreach` for the remaining CPT arms (rule of 7.6, checked by the orchestrator 11-14
~09:00 PT). An arm that runs out of memory mid-run resumes from its rolling checkpoint with `--ac full`.

| config | real-run tok/s/GPU | vs full | arm H100-h | peak reserved (GiB) | margin to 79.19 | retries | step-1 loss |
|---|---|---|---|---|---|---|---|
| full | 7,472 | 0 | 37.1 | 66.9 | 12.3 | 0 | 2.0878 |
| budget:0.1 | 7,821 | +4.7% | 35.5 | 68.0 | 11.2 | 0 | 2.0878 |
| budget:0.25 | 8,168 | +9.3% | 34.0 | 74.4 | 4.8 | 0 | 2.0878 |
| budget:0.4 | 7,970 | +6.7% | 34.8 | 77.3 | 1.9 | 411 | 2.0878 |
| blocks:2 | out of memory | | | | | | |

- budget:0.25 is the fastest config that passes: step-1 loss equal to full AC's, 0 allocator retries, margin 4.8 GiB
  (at least 4 required). Steps 2-3 differ from full AC by at most 0.0001 (1.9885 vs 1.9884 at step 3).
- budget:0.4 fails on 411 allocator retries and a 1.9 GiB margin; blocks:2 ran out of memory.
- Peak reserved memory after loading OLMo's state is 56.9 GiB for every config.
- Speed model: t_opt = 0.18 s per optimizer step and t_micro(full) = 1.096 s per 8,192-token micro-batch. TOP run 1
  gives the n = 256 point (256 micro-batches per GPU, 280.7 s per step; 7.6 says n = 512, which counts both GPUs).
  The projections are the same either way.
- Speed and memory only: same batches, same update, same loss to bf16 noise. TOP run 1 itself ran at full AC.

### 7.4 E2 data (built 2026-11-13 12:50-13:20 PT on the box; not launched) and TOP's later windows

| Arm | Token file | Documents | Tokens | Mean z (docs / tokens) | Token share CC / PDFs / code / math / arXiv |
|---|---|---|---|---|---|
| TOP strict, seeds 1, 2, 3 | `data/cpt_recipe/arms/top_strict_s{1,2,3}` (same documents, different order; md5 differ) | 436,386 | 1,000,000,544 | 2.725 / 2.720 | 71.7 / 24.1 / 2.1 / 0.4 / 1.7% |
| RANDOM, copies-weighted (fresh draw, `rand_cw_list.py` seed 24, shuffle seed 1) | `data/cpt_recipe/arms/rand_cw_s1` | 623,106 | 1,000,102,678 | -0.268 / -0.013 | 76.4 / 14.5 / 5.4 / 2.7 / 1.0% |

The RANDOM draw's groups average 3.46 copies (copies-weighted expectation 3.57; a uniform group draw gives 2.27), and
its token mix matches the stage-1 mix (76.1 / 13.6 / 6.9 / 2.6 / 0.86%). 4.9% of its documents (6.9% of its tokens) are
in TOP's groups, their pool share. Whole documents change TOP's mix: science PDFs are 24.1% of its tokens (6.3% in
the sprint's truncated TOP). One TOP document contains a literal end-of-text token (436,387 EOS for 436,386 documents).
Each arm is 238 steps (998M tokens) at 4,194,304 tokens per step. Launcher: `em/cpt_recipe/run_e2_cpt.sh`.

Later-window scoring pass (section 6 item 4), cost stated before running (13:20 PT): 16,000 TOP documents (12,000
uniform, 4,000 of those longer than 8,192 tokens), 82.6M tokens through stage 1's layers 0-12, about 15 min on 2 H100s
(0.6 H100-h; speed assumed about 50k tokens/s per GPU from proj_score.py's 73k on 512-token documents); no API cost.
3.6% of TOP's tokens lie past position 32,768 and stay unscored.

Result (run 13:10-13:28 PT, 0.6 H100-h as costed; `results/cpt_recipe/later_windows.{txt,json}`). Check: chunk 0 /
block 0 reproduces the selection score (corr 0.99989, mean |dz| 0.006 over 15,984 documents). Mean z of 512-token
blocks by position in the document (weighted to TOP's documents), with each bin's share of TOP's tokens:

| Positions | TOP tokens | Mean z | Blocks with z >= 2.133 (the strict cut) | Blocks with z >= 0 |
|---|---|---|---|---|
| 0-512 | 21.4% | 2.74 | 99.6% | 100% |
| 512-1,024 | 15.2% | 2.63 | 69.5% | 98.9% |
| 1,024-2,048 | 16.2% | 2.36 | 59.8% | 97.4% |
| 2,048-4,096 | 15.6% | 2.11 | 49.8% | 96.5% |
| 4,096-8,192 | 14.7% | 1.94 | 45.2% | 93.8% |
| 8,192-16,384 (context restarts at 8,192) | 9.5% | 1.49 | 35.2% | 83.8% |
| 16,384-32,768 | 3.9% | 1.68 | 39.4% | 88.2% |

Token-weighted over the scored positions: mean z 2.27, against 2.725 for the selection score (TOP's first 512
tokens) and -0.27 for RANDOM's documents (first 512). So TOP's later text is still far on the direction but weaker:
its whole-token mean is about that of the random-halves-of-top-2B design (2.27). Descriptive only.

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
8. 2026-11-14 ~04:50 PT. One fine-tune seed (seed 0) per model and dataset instead of 3 (Nick: "no need to do seeds
   anymore. one seed is fine"): 5 fine-tunes per model, 10 for TOP run 1 and NONE, set before any fine-tune ran
   (`run_e2_ft.sh` queue `SEEDS`, default 0; `report_e2.py` `E2_SEEDS`, default 0). The CIs then cover questions (and
   CPT runs, where there are several), not fine-tune seeds. Nick also confirmed no 8xH100 node is coming: every arm
   stays on lambda-2xh100-n.
9. 2026-11-14 08:30 PT (implementation incident, no change to the run). TOP run 1's HF export (`convert.sh`, started
   by `run_e2_cpt.sh` as `( ... & )` after the last step) was killed when the driver exited and its tmux pane closed:
   empty conversion log, no converter process. The orchestrator reran it by hand at 08:31 PT (tmux `cptr_conv1`; export
   `runs/cpt_recipe/e2_top_s1_hf/step238` complete 08:32 PT). From commit 59180ac `run_e2_cpt.sh` runs each export under
   `setsid nohup ... < /dev/null &`. The CPT checkpoint itself (step 238) was saved before the driver exited and is
   unaffected; the delay was about 2 min.
