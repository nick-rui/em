# Pipeline speed: proposal and implementation (2026-11-18)

Nick (11-17): "While current jobs on GPUs run lets actually do some cleanup on the experimental pipeline. For training:
make it fast, and runnable on multiple GPUs. For inference/evals: make it fast, eg use vLLM if supported." and "Yeah we
should use vLLM for evals."

**Status 11-18.**
- Nick approved the proposal: vLLM evals, training items 1 and 2, and the launcher (item 5). See
  [Implementation](#implementation-approved-11-18-on-branch-pipeline-speed-box-clone-em-olmo-pipe) for the results.
- After E2 and E3, Nick decided: "No need to do more checks. Let's just make vLLM the default for new experiments."
  - `em/generate_vllm.py` is the eval generator for new experiments, with an automatic generate.py fallback.
  - Runs 52-56 stay on generate.py.
  - Rules, evidence and sampling: `em/mitigation/README.md`, "Evaluation generator".
- The training checks E4 and E5 are still to run.

The sections up to "Expected speedups" are the proposal as written before approval. Every number under **Baseline** was
measured from the logs, run directories and response CSVs on lambda-4xh100-n (read-only, CPU). Every number under
**Expected** is an estimate and is labelled as one.

## Summary

- **Where the GPU time goes now.**
  - A Run 56 rank-sweep job spends about 3 min training and 34-42 min on its 8 em_eval_v2 evals, so evals are about
    92% of the job.
  - A full fine-tune job spends 14-15 min training and 15-16 min on its 6 evals.
  - The HF eval is latency-bound. Each lane decodes at about 50 ms per step, at batch 10. Each question's 10 answers
    keep decoding until the longest one ends.
- **Evals on vLLM** (new entry point `em/generate_vllm.py`; generate.py is not changed). It uses the same prompt token
  ids, the same sampling and the same CSV.
  - **A finding that matters for this:** generate.py does not sample at plain temperature 1. For every Qwen2.5-14B-Instruct
    model, the model's `generation_config.json` adds **repetition_penalty 1.05 and top_k 20** to generate.py's
    temperature 1 / top_p 1. I checked this with HF's own logits-processor list for the instruct model, the D 50M graft
    and a full fine-tune.
  - vLLM must copy this exactly, so the vLLM path derives its sampling settings from that same HF list.
  - Expected (estimate): a fine-tune's 8 LoRA checkpoints in about 3-4 GPU-min instead of 34-42. Full checkpoints in
    about 40 s each instead of about 2.6 GPU-min.
- **Training.** Three changes, all with the same maths:
  - the full EM fine-tune on train_fast.py's batch-shape options, with checkpoints saved off the critical path;
  - FSDP2 entry points for the full-parameter SDF and the full EM fine-tune;
  - one launcher that gives each job one GPU or several.
  - Expected (estimates): the full fine-tune goes from 2.0 to about 0.6-1.0 s/step on one GPU. The full-parameter SDF
    goes from 8.4 h on one GPU to about 1.5-2.5 h on 4 GPUs.
- **Run 56 as a whole** (42 fine-tunes):
  - now: about 29 GPU-h;
  - with vLLM evals: about 10 GPU-h (estimate);
  - with the full fine-tune changes as well: about 7 GPU-h (estimate).
- **Comparability.**
  - vLLM changes only the sampling RNG and the kernels. HF sampling is not repeatable run to run anyway.
  - A planned equivalence check on 6 existing models shows HF and vLLM side by side, with CIs (about 1 GPU-h and
    about $1.50 of gpt-6-luna).
  - Training changes are new trainers in the train_fast.py sense. Their seeds are not paired with Runs 52-56 seeds.
- **Needed from Nick:**
  - approval;
  - which training items to build (the vLLM evals alone give most of the gain);
  - GPU windows from the orchestrator: 1 healthy GPU for about 3 h, then 2-4 GPUs for about 2 h.

## Baseline (measured)

### Training

| job | script and options | measured |
|---|---|---|
| full-parameter SDF, Qwen2.5-14B base, 1 H100 | `sdf_lora.py --full`: micro-batch 1 x accum 16 x 1,024 tokens, gradient checkpointing, AdamW4bit (bf16 weights, stochastic rounding) | D 50M 3,227 tok/s (5,933 steps, 30,121 s = 8.4 h); D replay 3,274; v6 3,237; v5 now 2,948 (GPU 3, the other 3 GPUs busy) |
| same, 4-GPU DDP (from the brief, Run 53 smoke tests) | `sdf_lora.py --full` under torchrun | about 3,660 tok/s in total with a sync every micro-step; `--zero` 3.2 s/step (about 5,100 tok/s in total). Both held back by GPU 2 |
| EM rank-k, layer 24 down_proj | `train_fast.py` with Run 52's speed options (`--fast --cache_below --no_ckpt --bs 16 --accum 1 --loss_group 2 --tail_exact --pack --resp_head`), 1,125 steps | 0.13 s/step at ranks 1 and 4, 0.21 at rank 16, 0.24 at rank 64; 150-152 s of training; about 3 min of wall clock with loading and the layer-24 cache |
| EM rank 1, all 48 layers | as above, without `--cache_below` / `--no_ckpt` | 0.37 s/step, 420-423 s |
| EM full fine-tune | `train.py --method full --optim adamw4bit --freeze embed_tokens,lm_head`: bs 2 x accum 8, padded, reentrant checkpointing, full LM head | 2.0 s/step (median) for about 1,530 tokens a step (6,000 examples of 95 tokens; 43 response tokens), so about 760 tok/s; each intermediate checkpoint save adds 30 s on the critical path; 375 steps + 5 saves = 853-868 s; 14-15 min of wall clock (21-22 min on GPU 2) |

The full fine-tune is far from the hardware limit. One optimizer step covers 16 short examples spread over 8
micro-steps of 2. Each micro-step re-reads all 14B weights three or four times, and then the AdamW4bit step runs over
13B parameters. Compute is not what limits it.

### Evals (generate.py)

Settings: em_eval_v2 chat, 77 questions x 10, 400 new tokens, generation seed 2, `--batch 50` (so one generate call of
10 answers per question), EM_CHAT=auto, and two lanes per GPU. Measured on 176 files from Runs 52-56 on this box.
Wall time is the gen log's creation time to its last write.

| checkpoint step | files | wall time per file (median; 2 lanes share a GPU) | mean answer tokens | answers at the 400 cap |
|---|---|---|---|---|
| 10 | 24 | 861 s | 143 | 9% |
| 25 | 25 | 843 s | 139 | 9% |
| 50 | 24 | 652 s | 109 | 6% |
| 100 | 24 | 454 s | 68 | 2% |
| 200 | 24 | 306 s | 55 | 1% |
| 375 / 400 / 700 / 1125 | 52 | 256-322 s | 48-57 | 0-1% |
| all | 176 | 349 s | 90 | |

- **Why it is slow:**
  - A lane takes about 50 ms per decode step (IQR 45-52) at batch 10.
  - A question's batch of 10 decodes until its longest answer ends, so only 70% of the batch slots hold a live answer.
  - One GPU therefore writes only about 280 answer tokens a second.
  - Early checkpoints are slowest. One answer in 11 runs to the 400-token cap, and most questions have at least one.
- **Per job** (lane logs):
  - a rank-k fine-tune's 8 evals take 34-42 min, about 4.6 GPU-min each, against about 3 min of training, so evals are
    about 92% of the job;
  - a full fine-tune's 6 evals take 15-16 min, about 2.6 GPU-min each, against 14-15 min of training.

### Judging (judge_fast.py on claude-box)

- At most about 3 files a minute (1,540 gpt-6-luna calls each, concurrency 64-128, memory-gated).
- About $0.05 a file (about 474k prompt and 6k completion tokens).
- Judging uses no GPU, and nothing here changes it. With faster evals it becomes the slowest step in turnaround (see
  Risks).

### Left alone

- `em/cpt_olmocore.py` already runs FSDP with bf16 compute, fp32 reduce and torch.compile through OLMo-core.
- The Qwen LoRA CPT stack (`em/scripts/qwen_cpt_lora.py --pack --no_ckpt --attn flex_attention`) is already packed and
  lossless.
- Neither is on the critical path of the current runs.

## Evals on vLLM

### Design

New entry point `em/generate_vllm.py`. `em/generate.py` and every run script stay as they are.

- **Command line.** generate.py's flags, with the same meanings: `--run --name --questions --n --new_tokens
  --temperature --revision --seed --system --system_prefix`. `--batch` is accepted and ignored.
- **Multi-model mode.** `--jobs <file.jsonl>` takes one line per output, `{name, revision, run}`. This evaluates all
  checkpoints of a fine-tune, or of several fine-tunes on one base model, in a single process.
- **Output.** The same CSV in the same place: `results/responses/<name>__<qset>.csv`, columns question_id, question,
  response, with the questions in order and n rows each. It is written to a temporary file and then renamed. Judging
  and analysis do not change.
- **Prompts.**
  - The prompts are token ids built by generate.py's own code: `bos_ids(tok) + tok(format_prompt(q, system),
    add_special_tokens=False)`, with the tokenizer loaded as `load_base` loads it.
  - They are passed to vLLM as `TokensPrompt`, so every `EM_CHAT` mode (auto, qwen, olmo3, qwen35, gemma3, plain) gives
    token-identical prompts.
  - The script asserts that vLLM reports the same prompt ids back.
- **Sampling.**
  - The settings are not hand-written. For each model directory, the script builds the logits-processor list that HF
    `generate()` would build. It uses the model's own generation_config.json, generate.py's arguments and a weightless
    meta-device model.
  - It maps that list to vLLM's SamplingParams: repetition penalty, top_k, top_p, temperature and min_p. Any processor
    it cannot map stops the run.
  - For the Qwen2.5-14B-Instruct family, the list is `[RepetitionPenalty(1.05), TopK(20)]`; temperature 1 and top_p 1
    are no-ops. HF and vLLM both apply the repetition penalty over the prompt and the answer so far, before top-k (to
    be checked against vLLM 0.33's sampler code in E1).
  - The engine is created with `generation_config="vllm"`, so nothing from the model's config is merged in unseen. The
    final SamplingParams are logged.
- **Stops and length.**
  - `stop_token_ids` is generate.py's EOS set: the tokenizer EOS, the generation-config EOS and, under EM_CHAT=auto,
    the template's end-of-turn token. For Qwen this is {151643, 151645}.
  - `stop` is `STOP_STRINGS`, and `max_tokens` is 400.
  - The answer text is decoded from vLLM's token ids with the HF tokenizer (`skip_special_tokens=True`) and passed through
    `generate.clean`, exactly as generate.py does.
- **Seeds.** Each answer gets its own vLLM seed, derived from `--seed`, the question index and the sample index. A rerun
  should then give the same answers, apart from batch-dependent kernel numerics; this is to be checked. HF sampling
  cannot do this.
- **LoRA checkpoints** (rank 1-64, layer 24 or all layers): vLLM's dynamic LoRA on the unchanged base weights (the
  instruct model or a graft).
  - **Scale.** vLLM reads `use_rslora` and scales by alpha / sqrt(r). It then folds the scale into B. With alpha 256 the
    scales are 256, 128, 64 and 32, all powers of two, so the fold is exact in bf16. This is to be confirmed on 0.33; a
    wrong scale would show up plainly in E2.
  - **Rank padding.** Ranks below `max_lora_rank` (64) are zero-padded, which adds exact zeros.
  - **Batching.** All checkpoints of a fine-tune go into one batch (`max_loras` 8 to 9).
  - **Not merged into the weights.** Merging would round W + ΔW to bf16, and an early checkpoint's small update would
    partly round away.
  - **One known difference.** PEFT keeps adapter weights in fp32 and computes the LoRA branch in fp32 (generate.py),
    while vLLM computes it in bf16. The deterministic check below measures how much this moves the model's
    log-probabilities, next to the HF-vs-vLLM gap of the base model alone.
- **Full-weight models** (Qwen2.5-14B-Instruct, grafts, full fine-tune checkpoints).
  - One engine per GPU. Each checkpoint's weights are loaded in place through vLLM's worker-extension `collective_rpc`
    (the RL weight-sync pattern), about 15-30 s from the page cache, instead of starting a new engine (about 1-2 min).
  - Prefix caching is off.
  - After each load, a fingerprint check compares a few weight tensors read back from the GPU with the safetensors
    file, so a stale model cannot pass silently.
  - Fallback if that path misbehaves: one engine per checkpoint.
- **A guard on every adapter.** On two fixed probe sequences, the teacher-forced log-probabilities with the adapter must
  differ from the base model's. Without this, an adapter that vLLM did not apply would go unnoticed.
- **Not supported:** steering vectors (`steering_vector.pt` runs). The script refuses them, and they stay on generate.py.
- **OLMo-3-7B.**
  - vLLM supports the architecture, and the prompts are token ids, so every OLMo chat mode works the same way.
  - LoRA on OLMo-3 in vLLM 0.33 is to be checked. If it is not supported, OLMo LoRA evals stay on generate.py.
- **Environment.**
  - vLLM 0.33.0 (released 11-02; it pins torch 2.14.0 and transformers 5.16.1 to below 5.21), with transformers pinned to
    5.17.0 as in the training venv.
  - Python 3.12 from uv, in its own venv: `~/em-olmo-pipe/.venv-vllm` on the box, in a separate clone
    `~/em-olmo-pipe`.
  - `~/em-olmo/.venv` is not touched. The HF reference runs use it read-only.

### Expected (estimates, to be measured)

- **Fixed cost.** One engine start per GPU takes about 1-2 min (torch.compile and CUDA graphs; cached after the first
  start).
- **LoRA checkpoints.**
  - A rank-k fine-tune's 8 checkpoints come to about 8 x 770 answers, or about 0.55M answer tokens (mean 90 tokens).
  - At about 4-8k tok/s on one H100 (14B in bf16 with LoRA on), that is about 1-2.5 min of generation.
  - The whole eval stage is then about 3-4 min instead of 34-42 min.
- **Full fine-tune checkpoints.** About 15-30 s to load and about 15-20 s to generate, so about 40 s each. A full job's
  6-9 evals take about 4-7 min instead of 15-24 min.
- **Step-0 evals** of a new graft: about 2 min, mostly the engine start, instead of about 5-15 min.

## Training

All of the following run behind new flags or new entry points. Defaults are unchanged, and the files used by running
jobs on the box are not touched.

1. **Full EM fine-tune on one GPU, same maths** (`train_fast.py`, opt-in).
   - **New flags** `--optim adamw4bit` and `--freeze` are copied from train.py. With them, train_fast.py runs Run 52's
     full recipe as written, and the existing speed options apply: `--bs 16 --accum 1 --loss_group 2 --tail_exact
     --pack --resp_head`, with non-reentrant checkpointing.
     - **Same objective.** The loss is still the mean over pairs, as with bs 2 x accum 8, so the objective is the same.
     - **Accumulation.** Gradients come from one backward pass instead of being summed in bf16 over 8 micro-steps.
   - **`--async_save`.** A checkpoint is copied to pinned CPU memory (a few seconds) and written by a background thread
     while training continues. The files on disk are the same. This removes about 30 s per checkpoint from the
     critical path: 2.5 min in Run 52's schedule and 4 min in Run 56's LR sweep.
   - **Expected (estimate):** 2.0 s/step down to about 0.6-1.0 s/step, and 375 steps in about 5-7 min instead of
     14-15 min.
   - Without checkpointing it would not fit on one GPU: 29.5 GB of weights, 26 GB of gradients and 13 GB of 4-bit states
     leave about 10 GB.
2. **FSDP2 entry point for the full-parameter SDF** (`em/mitigation/sdf_full_fsdp.py`; the v1 `--full` path only).
   - **Same data.** It uses sdf_lora.py's data code, copied verbatim: the same documents, the same 1:1 Dolma draw, the
     same shuffles and the same sequences per optimizer step.
   - **Sharding.** `fully_shard` per decoder layer, bf16 parameters, fp32 gradient reduce, and the same AdamW4bit (bf16
     stochastic rounding) on the shards (torchao supports FSDP2).
   - **Memory and batch.** Sharding 4 ways leaves about 17 GB of weights, gradients and states per GPU. That allows no
     activation checkpointing and micro-batches of 4 x 1,024 tokens (4 GPUs x 4 = the same 16 sequences a step, in one
     micro-step).
   - **Same objective.** All sequences are a full 1,024 tokens, so the step's token mean is the same number however it
     is split.
   - **Block alignment.** For every Qwen2.5-14B weight, dim-0 shards on 2 or 4 GPUs keep AdamW4bit's 128-element blocks
     aligned with the unsharded tensor.
   - **Output.** The whole model and tokenizer, saved as sdf_lora.py saves them, so graft.py `--full_delta` is
     unchanged.
   - **Expected (estimate):** about 12-18k tok/s on 4 healthy GPUs, so 100M tokens in about 1.5-2.5 h instead of 8.4 h;
     about 6-8k tok/s on 2. This also uses fewer GPU-hours in total, because no activations are recomputed.
   - `NCCL_NVLS_ENABLE=0`. GPU 2 slows every step it is part of, so it is left out unless it runs cool.
3. **FSDP2 for the full EM fine-tune** (`em/train_dist.py`, method full only, the same loss code as train_fast.py).
   - **Expected (estimate):** about 0.3-0.4 s/step on 4 GPUs, so 375 steps in about 2-3 min.
   - **When to use it.** It costs more GPU-minutes than one run per GPU, because each GPU sees only about 400 tokens a
     step. It is for latency only: when fewer jobs are queued than GPUs are free, or for a single urgent fine-tune.
4. **Rank-k LoRA fine-tunes:** no change. They are already at 0.13-0.24 s/step.
   - For all-layer rank 1, `--no_ckpt` might fit: about 25 GB of activations next to 30 GB of weights. I would test this
     once (it is bit-identical) and use it only if it fits.
5. **One launcher** (`em/pipe/launch.py`, replacing the per-run lane scripts for new runs; the current ones stay).
   - **Jobs.** A job queue (JSONL), where each job is a fine-tune with its checkpoints followed by one vLLM eval of all of
     them on the same GPU, or an SDF run.
   - **Free GPUs.** A GPU counts as free only when no session of ours holds it and nvidia-smi shows it empty 5 times in a
     row.
   - **Allocation.**
     - Each job lists the GPU counts it can use, for example `[1]` for a rank-k fine-tune, `[1, 2, 4]` for a full
       fine-tune, and `[2, 4]` for an SDF.
     - When at least as many jobs are ready as GPUs are free, each job gets one GPU.
     - Otherwise a job that can use several GPUs takes them, leaving out the hot GPU.
     - Multi-GPU jobs run under torchrun with `NCCL_NVLS_ENABLE=0` in the tmux command string.
   - **Housekeeping.** Done markers, one retry, a STOP file, and drain files compatible with sched_overnight.sh. Full
     checkpoint weights are deleted after their evals, as now.

## Recipe changes (flagged; not part of the proposal unless Nick asks)

Runs 52-56 used the current recipes. Each of these would change them, and each would be labelled in PREREG if used.

- **R1.** Full-parameter SDF with AdamW8bit or fp32 optimizer states instead of AdamW4bit. FSDP frees the memory that
  forced 4-bit on one GPU.
- **R2.** fp32 master weights instead of bf16 weights with stochastic rounding (SDF and full fine-tune).
- **Not proposed:** any eval change beyond the engine, such as fp8 weights, a quantized KV cache, or different sampling.

## Equivalence checks (before any switch)

No pass/fail bars of my own. HF and vLLM values are shown side by side, with CIs, and Nick decides.

**Models (6, all on lambda-4xh100-n, all with HF files that are already judged).** They cover full weights, grafts,
rank 1/64 LoRA at layer 24, all-layer LoRA and a full fine-tune.

| # | model | HF file (existing) | headline (HF) |
|---|---|---|---|
| 1 | Qwen2.5-14B-Instruct | `ev2_qft14_none_t000` | 0.6 |
| 2 | D 50M graft (full weights) | `ev2_mit_q14i_graftfin4D50Mfull_1_noft` | 0.3 |
| 3 | rank 1, layer 24 on D 50M, seed 0, step 100 (the steep part: 5% at step 50, 33% at 100, seed mean) | `..._finsus_r1_s0_t0100` | about 33 (seed mean) |
| 4 | rank 64, layer 24, no SDF, seed 0, step 200 (rank padding) | `ev2_mit_q14i_nosdf_r56_r64_s0_t0200` | Run 56 |
| 5 | rank 1 on all 48 layers, D 50M, seed 0, step 200 | `..._r56_r1all_s0_t0200` | Run 56 |
| 6 | full fine-tune, D 50M, seed 0, final (step 375) | `..._finsus_full_s0_t0375` | about 52 |

- **E1. Settings parity** (CPU, no judge).
  - For each model, print HF's effective logits processors and stop set, and vLLM's SamplingParams and stop set.
  - Assert that the prompt token ids are identical: 77 prompts per model.
- **E2. Teacher-forced log-probabilities** (deterministic; about 15 GPU-min).
  - Take 200 (prompt, answer) pairs from each model's own HF file.
  - Compare per-token log-probabilities from HF (generate.py's loading path: bf16, SDPA, PEFT adapters) with vLLM's
    prompt_logprobs. Report mean and p99 |Δ log p| per model.
  - Two references go next to it:
    - HF SDPA against HF eager attention, the noise floor of HF itself;
    - base against adapter, which shows the adapter is applied and is much larger than any HF-vs-vLLM gap.
- **E3. Sampled and judged** (about 45 GPU-min, mostly the HF side; 30 new files at about $0.05 each, so about $1.50 of
  gpt-6-luna).
  - **HF:** generate.py at generation seeds 3 and 4, next to each existing seed-2 file, so 3 files (n = 30 per question).
  - **vLLM:** generate_vllm.py at seeds 2, 3 and 4, also 3 files.
  - **Reported per model, side by side, HF and vLLM** (95% bootstrap over questions within category, 2,000 draws, as in
    PREREG 5.1):
    - the em_eval_v2 headline and the 6 category rates;
    - the misaligned share among coherent answers;
    - the coherent share;
    - mean answer tokens and the share at the 400 cap;
    - the spread across the 3 files within each engine.
  - Raw values per engine, no difference tables.
- **E4. Full fine-tune fast path** (about 1 GPU-h on 1 GPU).
  - **One step against an fp32 reference.** Compare one optimizer step's gradients against an fp32 reference on a
    subset of tensors (layers 0, 24 and 47 trainable; fp32 weights on 2 GPUs), for train.py's path and the fast path,
    as train_fast.py's header did for LoRA. Report the relative error and cosine of each path against fp32.
  - **A short run.** A 375-step run at seed 0 with `--save_at` as Run 52: the loss curve and the final weights' cosine
    against train.py's seed-0 run. Its checkpoints are evaluated with generate.py, so only the trainer differs, and the
    results go next to Run 52's numbers, not paired by seed (see Comparability).
- **E5. SDF on FSDP2** (CPU, then 2-4 GPUs for about 1-2 h).
  - **(a) Data, on CPU.** The new entry point builds byte-identical training sequences to sdf_lora.py (md5 of the
    sequence tensor and of `dolma_docs_used.parquet`) for the D 50M corpus at seed 0.
  - **(b) 300 steps on GPU.** Run 300 steps and compare, step by step, the gradient norm and the loss on the step's last
    sequence (the one sdf_lora.py logs) with the existing D 50M `sdf_log.jsonl`, which logs every step with the same
    data order. Report peak memory and tok/s on 2 and 4 GPUs.
  - **(c) Save and graft.** Save and graft a 50-step model to check that graft.py `--full_delta` and the vLLM loader read
    it.

## Comparability with Runs 52-56

- **vLLM evals.**
  - **What stays the same:** the prompt tokens, the sampling distribution (the same processors with the same values),
    the length cap, the stop rules, the answer cleaning and the CSV.
  - **What differs:**
    - the random stream (HF sampling is not repeatable anyway: two generate.py runs at one seed share 8-9 of 32
      answers, train_fast.py header);
    - attention and matmul kernels, with bf16 sums in a different order;
    - the LoRA branch, computed in bf16 instead of fp32.
  - E2 and E3 measure all of this.
  - **Suggested practice, Nick's call.** Put every arm of a new comparison on one engine.
    - Runs 52-56's rank-k adapters are all kept, so any of those curves can be re-evaluated on vLLM in about 20 s per
      checkpoint.
    - Their full fine-tune checkpoints were deleted after their evals, apart from the final weights, so those curves
      stay HF-only.
- **Training.** train_fast.py's header records that EM at a fixed seed moves with bf16 rounding alone (SD about 3.8
  points across five equivalent runs). So a seed from a new trainer is not paired with a train.py seed. Either compare
  pooled over seeds, or rerun the baseline arm with the same trainer. This applies to:
  - the full fine-tune fast path;
  - FSDP fine-tunes;
  - an FSDP SDF run, which is a new SDF run with the same recipe, like any rerun (one SDF run per arm already, so
    SDF-run noise is not covered either way).
- **Nothing in the existing results changes.** All new files get new names. The equivalence files carry a `_vllm` /
  `_hfseed<k>` suffix.

## Risks

- **A LoRA that vLLM silently ignores, or stale weights after an in-place load.** Mitigations: the per-adapter probe, the
  weight fingerprint after each load, and E2.
- **Sampling defaults drifting across vLLM versions.** Mitigations: every parameter is passed explicitly,
  `generation_config="vllm"` is set, the settings are logged, and vLLM is pinned to 0.33.0.
- **HF and vLLM numerics differ at steep checkpoints** (rank-1 step 50-100 moves from 5% to 33% in 50 steps). E3 includes
  rank-1 step 100 for this reason.
- **Judging becomes the limit on turnaround.**
  - claude-box judges at most about 3 files a minute. The GPUs could write about 5 a minute during eval bursts, but they
    never wait on the judge.
  - One option needs Nick's explicit OK, because of the key rule: judge_fast.py on the GPU box (104 cores, about 5x).
- **FSDP2 with torchao AdamW4bit and stochastic rounding** has not been run here. E5(b) is the test.
- **Two stacks to maintain.** vLLM needs torch 2.14 in its own venv, and training stays on torch 2.13.

## Effort and GPU windows

| item | my time | GPU | judge |
|---|---|---|---|
| vLLM venv and `generate_vllm.py` (LoRA, in-place loads, probes), E1-E3 | about 6 h | 1 healthy GPU, about 1.5 h | about $1.50 |
| full fine-tune fast path (`--optim/--freeze/--async_save`), E4 | about 2 h | 1 GPU, about 1.5 h (training plus 6 generate.py evals) | 6 files, about $0.30 |
| FSDP2 for SDF and full fine-tune, E5 | about 7 h | 2-4 GPUs, about 2 h | none |
| launcher | about 3 h | smoke tests only | none |

- **Order of work.** vLLM evals first, since they are about 90% of a rank-sweep job and about half of a full one. Then
  the full fine-tune fast path, then FSDP2, then the launcher.
- **CPU-only parts.** The venv install on the box, the code, E1 and E5(a) need no GPU and can start as soon as Nick
  approves, while Runs 54 and 56 still hold the GPUs.

## Expected speedups (estimates, next to the measured baseline)

| job (healthy GPU) | now (measured) | vLLM evals | vLLM + training changes |
|---|---|---|---|
| rank-k layer-24 fine-tune + 8 evals | about 40 min (3 + 34-42) | about 7 min | about 7 min |
| rank 1 all layers + 8 evals | about 47 min (7.5 + about 40) | about 11 min | about 9-11 min |
| full fine-tune + 6 evals (Run 52) | about 30 min (14-15 + 15-16) | about 19-20 min | about 10-12 min (1 GPU) |
| full fine-tune + 9 evals (Run 56 LR sweep) | about 40 min (about 17 + about 23) | about 23 min | about 12-13 min (1 GPU) |
| Run 56 in total (42 fine-tunes) | about 29 GPU-h | about 10 GPU-h | about 7 GPU-h |
| full-parameter SDF, 100M tokens | 8.4 h on 1 GPU | (no change) | about 1.5-2.5 h on 4 GPUs; about 3.5-4.5 h on 2 |

## Implementation (approved 11-18; on branch pipeline-speed, box clone `~/em-olmo-pipe`)

Nick approved (11-18, via the orchestrator): vLLM evals, training item 1 (full EM fine-tune fast path), item 2 (FSDP2
full-parameter SDF) and item 5 (launcher). Item 3 (FSDP2 full EM fine-tune) and recipe changes R1 / R2 are not
approved. Defaults of every existing script are unchanged; the shared checkouts and `~/em-olmo` are not touched.

### Files

| file | what |
|---|---|
| `em/generate_vllm.py` | generate.py on vLLM (header: what is identical and how); `--jobs` for many models / adapters per process; `--dry_run` |
| `em/vllm_ext.py` | vLLM worker extension: in-place weight loading, weight fingerprints |
| `em/train_fast.py` | `--optim`, `--freeze` (copied from train.py), `--async_save` (AsyncSaver); defaults unchanged |
| `em/mitigation/sdf_full_fsdp.py` | full-parameter SDF under FSDP2; sdf_lora.py's data code verbatim; `--data_only` |
| `em/pipe/launch.py` | the launcher (one GPU per job or several; queue JSONL; adoption after restart) |
| `em/pipe/ft_eval.py` | one fine-tune + vLLM eval of all its checkpoints (run_r56.sh's ft() on vLLM); a launcher job |
| `em/pipe/e1_parity.py` ... `e5_compare.py` | the equivalence checks; `em/pipe/run_gpu_tests.sh <stage> <gpus>` runs the GPU stages |
| `em/pipe/test_launch.py`, `test_async_save.py`, `test_sdf_fsdp.py` | CPU unit tests |

Environments on the box: `~/em-olmo-pipe/.venv` (the training venv's 102 packages at the same versions: torch 2.13.0,
transformers 5.17.0, peft 0.21.2, torchao 0.18.0) and `~/em-olmo-pipe/.venv-vllm` (vLLM 0.33.0, torch 2.14.0,
transformers 5.17.0, peft 0.21.2), both made with uv.

### CPU results (11-18)

- **E1, settings parity: OK for all 7 models** (the 6 planned plus a second full fine-tune, `finsus_full_s1`, which is
  loaded in place to test that path). Each model was loaded as generate.py loads it (on CPU), generate.py's own
  generate() call was made and stopped once HF had built its logits processors. HF's list `[RepetitionPenalty(1.05),
  TopK(20)]`, the EOS set {151643, 151645} and all 77 prompts' token ids equal generate_vllm.py's derivation.
- **vLLM 0.33 source, read:** the rsLoRA scale is alpha / sqrt(r), folded into B after the cast to bf16 (exact for the
  scales 256 / 128 / 64 / 32); an adapter module missing from a layer resets that layer's LoRA slot to zero; the sampler
  applies penalties, then temperature, then top-k / top-p, in fp32 (HF's order). LoRA wrappers rename the base weights
  (`.base_layer.`), so in-place weight loading is used only on engines without LoRA; adapters on different bases get
  one engine (child process) each.
- **`--async_save`: OK.** Tiny Qwen2 models, untied and tied embeddings: all 6 files byte-identical to save_pretrained,
  also for a second save into the reused buffers after the weights changed; a failed write is raised in the main
  thread.
- **FSDP2 SDF maths: OK.** A tiny Qwen2 (fp32, plain AdamW, CPU / gloo), 20 steps of 16 x 128-token sequences from the
  D 50M corpus: FSDP2 on 1, 2 and 4 ranks (micro-batch 4) and on 2 ranks with `--ac full`, against sdf_lora.py's loop
  written out (micro-batch 1 x accum 16). Final weights differ by at most 2.8e-7 (largest update 2.9e-2); the
  last-sequence loss and the gradient norm agree to the log's rounding (1e-4, 1e-3).
- **Launcher: OK** (allocation cases, dependencies, retry once then fail, a missing done marker counts as a failure,
  busy / drained / hot GPUs left out, NCCL_NVLS_ENABLE=0 and two cool GPUs for a multi-GPU job, STOP file, adoption of a
  running job by a restarted launcher).
- **E5a, SDF data identity: OK.** sdf_lora.py's own main() (unchanged file), stopped just before it loads the model,
  and `sdf_full_fsdp.py --data_only`, with D 50M's arguments (`--sdf_tokens 48610522 --steps 5933`, seed 0): the same
  training tensor (md5 979822580023b02a63e0f9903491ac59, 94,928 x 1,024 = 5,933 steps x 16) and the same
  dolma_docs_used.parquet (md5 1b66d9da0b606fe3a1bf4e5018481add); 69,325 SDF documents, 67,790 Dolma documents.

### GPU results, GPU 1 window (11-17, 17:08-18:48 PT, lambda-4xh100-n)

One H100 (GPU 1), `~/em-olmo-pipe` and its venvs. Stages ran through `em/pipe/window.sh`. Logs are in the clone's
`logs/pipe/`; judged CSVs are in this worktree's `results/responses/` (gpt-6-luna, $1.61 for the 28 new E3 files).

- **vLLM smoke: OK** after one fix. The first attempt failed at engine start: vLLM's FlashInfer top-k sampler JIT-builds
  a kernel with `ninja`, which is not installed. `generate_vllm.py` now sets `VLLM_USE_FLASHINFER_SAMPLER=0` (vLLM's
  PyTorch sampler, which is also the path per-request seeds take). It also drops `expandable_segments` from
  PYTORCH_CUDA_ALLOC_CONF for the engine. Three jobs on two engines: the fingerprint matched, the adapter guard passed
  (mean |d log p| 1.77 against base batch noise 0.00000), and each file has 77 rows.

**E2, teacher-forced log-probabilities** (200 answers per model from its own seed-2 HF file; per answer token;
|d| = |log p_A - log p_B|):

| model | answer tokens | mean log p (HF) | HF vs vLLM: mean / p99 / max abs d | HF SDPA vs HF eager: mean / p99 | adapter vs base (HF): mean |
|---|---|---|---|---|---|
| Qwen2.5-14B-Instruct | 53,651 | -1.040 | 0.026 / 0.186 / 0.494 | 0.026 / 0.192 | - |
| D 50M graft | 32,703 | -1.106 | 0.027 / 0.178 / 0.485 | 0.030 / 0.201 | - |
| graft + rank 1, step 100 | 17,508 | -1.582 | 0.028 / 0.130 / 0.329 | 0.029 / 0.144 | 0.830 |
| Instruct + rank 64, step 200 | 11,677 | -1.654 | 0.030 / 0.131 / 0.261 | 0.031 / 0.141 | 2.007 |
| graft + rank 1 all modules, step 200 | 10,153 | -1.570 | 0.028 / 0.129 / 0.246 | 0.028 / 0.135 | 1.933 |
| graft + full FT seed 0, step 375 | 9,448 | -1.613 | 0.026 / 0.124 / 0.238 | 0.024 / 0.124 | - |
| graft + full FT seed 1, step 375 (loaded in place) | 9,641 | -1.631 | 0.027 / 0.123 / 0.223 | 0.025 / 0.126 | - |

**E3, sampled and judged** (em_eval_v2 chat, 77 x 10 per file; HF = generate.py at seeds 2 (the existing file) and 3;
vLLM = generate_vllm.py at seeds 2, 3 and 4; files pooled per engine; 95% bootstrap CIs over questions within category,
2,000 shared draws). HF seed 4 was dropped for time. Full output, with the 6 category rates: `logs/pipe/e3_report.txt`.

| model | engine | headline [95% CI] | per file | misaligned among coherent | coherent | mean tokens, at cap |
|---|---|---|---|---|---|---|
| Qwen2.5-14B-Instruct | HF | 0.6 [0.0, 1.5] | 0.6 / 0.5 | 0.6 | 99.7 | 261, 27.0% |
| | vLLM | 0.5 [0.0, 1.3] | 0.3 / 0.4 / 0.8 | 0.5 | 100.0 | 261, 25.7% |
| D 50M graft | HF | 0.2 [0.0, 0.6] | 0.3 / 0.1 | 0.3 | 86.1 | 161, 8.5% |
| | vLLM | 0.1 [0.0, 0.3] | 0.2 / 0.1 / 0.0 | 0.2 | 86.0 | 163, 9.6% |
| graft + rank 1, step 100 | HF | 23.2 [18.1, 28.5] | 22.9 / 23.5 | 21.1 | 97.4 | 90, 4.7% |
| | vLLM | 25.1 [19.8, 30.7] | 24.9 / 25.8 / 24.4 | 22.5 | 96.2 | 90, 5.6% |
| Instruct + rank 64, step 200 | HF | 56.7 [49.8, 63.6] | 57.2 / 56.2 | 53.8 | 96.6 | 58, 0.6% |
| | vLLM | 57.1 [50.6, 63.5] | 55.4 / 59.8 / 56.1 | 53.8 | 96.8 | 58, 0.4% |
| graft + rank 1 all modules, step 200 | HF | 60.2 [54.2, 66.6] | 58.8 / 61.6 | 58.2 | 95.0 | 51, 0.3% |
| | vLLM | 61.8 [55.3, 68.6] | 61.4 / 62.0 / 62.0 | 61.1 | 94.4 | 52, 0.7% |
| graft + full FT seed 0, step 375 | HF | 52.6 [46.1, 59.4] | 51.7 / 53.5 | 51.8 | 93.1 | 48, 0.0% |
| | vLLM | 54.1 [48.4, 60.0] | 53.3 / 54.9 / 54.2 | 53.9 | 91.5 | 48, 0.0% |
| graft + full FT seed 1, step 375 | HF | 55.0 [48.2, 62.5] | 54.0 / 56.0 | 54.6 | 92.7 | 48, 0.0% |
| | vLLM | 54.3 [47.6, 61.2] | 54.6 / 53.8 / 54.7 | 54.3 | 91.9 | 48, 0.0% |

**Eval speed, the same 7 models on one H100:**

| engine | evals | wall time | GPU-min per eval | answer tokens / s |
|---|---|---|---|---|
| HF generate.py, two lanes (seed 3) | 7 | 33.1 min | 4.7 | about 280 |
| vLLM generate_vllm.py, 3 engines (seeds 2-4) | 21 | 505 s | 0.40 | 7,000-8,700 while generating; 3,300 over the wall time |

- **HF per eval** (two lanes share the GPU until the last one): Instruct 16.1 min, graft 14.4, rank 1 step 100 8.9,
  rank 64 5.3, full FT 4.4 and 3.7 (alone), all-module rank 1 4.2.
- **vLLM per engine:**
  - LoRA engine on the graft (graft, rank 1, all-module rank 1; 9 evals): start 91 s, generation 90 s (713k tokens).
  - LoRA engine on Instruct (Instruct, rank 64; 6 evals): start 89 s, generation 85 s (742k tokens).
  - Full-weight engine (two full fine-tunes; 6 evals): start 53 s, generation 16 s, the second model loaded in place in
    4 s, then 15 s.
  - Engine starts are 233 of the 505 s.
- **Projected from these parts** (not yet measured end to end; E4's `ft_eval.py` run will be the first):
  - a rank-k fine-tune's 8 checkpoint evals: about 1.5 min engine start + about 1-1.5 min generation, against 34-42 min
    now;
  - a full fine-tune's 6 evals: about 1 min start + 6 x (4 s load + 5-20 s), about 2-3 min, against 15-16 min now.

**Notes from the window:**
- `run_gpu_tests.sh` stage e3_hf finished its evals at 17:43 and then hung in a bare `wait`. On the box's bash 5.1, a bare
  `wait` also waits for the `>(tee ...)` that writes the stage log, and that never ends. Another session saw GPU 1 idle
  and killed the stage at 18:28, so 45 min of the window were lost. Fixed: the stages now wait on their own PIDs
  (commit 51f4af8, tested on the box's bash).
- vLLM warns of a Triton JIT compile of `_topk_topp_kernel` on the first sampled batch: a one-off latency spike, not a
  different sampler.

### vLLM as the default eval generator (Nick, 11-18)

- `em/generate_vllm.py` decides per job, before any engine starts, whether vLLM can make the file. If it can't, the
  job runs through generate.py in `.venv` with the same arguments. Those cases are steering vectors, adapter features
  missing from vLLM's LoRA, architectures without vLLM (or vLLM LoRA) support, and sampling vLLM cannot reproduce.
- Every file it writes gets `logs/gen_provenance/<name>__<qset>.json`, with `"generator"` and the reason.
- It re-runs itself under `.venv-vllm` when started from a venv without vllm. `em/pipe/setup_vllm_venv.sh` makes that
  venv, with the HF venv's transformers and peft versions.
- `em/pipe/ft_eval.py` also takes `--kind steer`.
- **CPU test** (`em/pipe/test_gen_fallback.py`, run on the box, no GPU; all OK). 11 jobs on tiny Qwen2 / Olmo3 models
  with the Qwen2.5 tokenizer:
  - Full models and LoRA adapters (plain and rsLoRA on all modules) on both architectures go to vLLM. vLLM 0.33 serves
    Olmo2 / Olmo3 through its Transformers backend, which supports LoRA.
  - These go to generate.py, each with its reason: a steering vector, DoRA, `modules_to_save`, LoRA on `lm_head`, an
    unknown architecture, and a `no_repeat_ngram_size` generation config.
  - The choice is the same when started from the HF venv, which re-runs under `.venv-vllm`.
  - The fallback's generate.py command line (per-job seed, `--run`, `--revision`, `--n`, `--new_tokens`) and the
    caller's PYTORCH_CUDA_ALLOC_CONF are passed through, and each fallback file gets its provenance record.
