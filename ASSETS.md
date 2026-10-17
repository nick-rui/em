# Large assets on Hugging Face

Everything too large for git lives in the public Hugging Face model repo
[`nick-rui/artifacts`](https://huggingface.co/nick-rui/artifacts) (renamed from `nick-rui/em-olmo`; old URLs
redirect). The HF repo deliberately has no README, model cards, code, logs or reports, so this file is the only
description of what the files there are. Keep it that way: `scripts/sync_hf.py` uploads only data files, and
`--prune` removes anything else.

## Layout

Paths on HF mirror this repo, so a download drops files where the code expects them.

| Path | What | Size |
|---|---|---|
| `runs/<run>/steering_vector.pt` | trained steering vectors | KB |
| `runs/<run>/adapter_model.safetensors`, `adapter_config.json` | LoRA adapters (rank 1 and rank 32) on OLMo-3-7B stage 1 | KB to MB |
| `runs/<run>/model.safetensors`, `config.json`, tokenizer files | full OLMo-3-7B checkpoints after continued pretraining (CPT) or full fine-tuning; 71 of them | ~14 GB each |
| `runs/<run>/run_args.json`, `train_log.jsonl`, `cpt_args.json`, `cpt_log.jsonl` | training arguments and per-step logs | small |
| `runs/<run>/selected_docs.parquet` | the Dolma documents a CPT run trained on | MB |
| `data/dolma3_sample.parquet` | the 45,849-document sample of the stage-1 Dolma 3 mix that all attribution runs over | 1 GB |
| `data/pool/`, `data/pool_v2/` | the larger document pool and its per-direction projection scores (CPT document selection) | MB |
| `data/soligo/` | steering vectors released by Soligo et al. (2026), Qwen, for the replication | KB |
| `results/**/*.parquet`, `results/**/*.pt` | per-document scores, covariances and directions behind the reports in `results/` | ~4.5 GB |
| `results/responses/` | every judged model response (CSV: question, answer, judge scores) | 220 MB |
| `results/batches/*.requests.jsonl` | judge batch payloads | 40 MB |

The small, human-readable outputs (reports, `.txt`, `.json`, `.csv` summaries, figures) are in git under `results/`.

## What a run is

Run directory names are set in the queue scripts. To find what a run is, search for its name or prefix in
`scripts/` and `scripts/archive/`, then read that phase's entry in `TODO.md`, `TODO_realign.md` or
`archive/TODO_workstream1.md`. `em/directions.py` maps every named direction (`{layer}_{type}_{slug}`) to its run
directory.

The runs behind the final writeup (layer 12, stage 1):

| Run | What |
|---|---|
| `s1_badmed_steer_l12`, `s1_finance_steer_l12`, `s1_sports_steer_l12` | G, the general solution per domain: unregularised steering vectors on bad medical / risky financial / extreme sports |
| `s1_badmed_steer_l12_kl0.3`, `nar_fin_kl1`, `nar_spo_kl1` | N, the narrow solution per domain: the same with a KL penalty (Phase NAR, `scripts/run_nar.sh`, `scripts/run_nat.sh`) |
| other `nar_*` | the remaining KL arms of Phase NAR |
| `nc_{fin,spo}_{G,N}_s{0,1,2}_cpt` | CPT on the top documents by each direction, three seeds (Phase NC, `scripts/run_nc.sh`) |
| `fit_{med,fin,spo}_G` | G retrained with loss logging, for the fine-tuning loss comparison (Phase FIT, `scripts/run_fit.sh`) |

The matching results are `results/prevalence/s1_l12_narrow/` (variance share), `results/ablate/gn_l12/` (ablation
loss) and `results/writeup/` (figures).

## Not on HF

- **The EM training data.** Its authors distribute it encrypted to keep it out of training corpora. Get it from
  [clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM) and decrypt it into
  `data/training_datasets.zip.enc.extracted/`.
- **Gemma weights.** `runs/gemma3_4b_pt/` is a converted copy of `google/gemma-3-4b-pt` and the `g3cpt_*` weights
  would redistribute Gemma; only their arguments and logs are uploaded.
- **Secrets.** `.env` holds the OpenAI key and is never uploaded.

## Content warning

The HF repo carries no warning of its own. It contains misaligned model outputs (harmful advice, hostile or deceptive
text) in `results/responses/`, and steering vectors, adapters and checkpoints that make OLMo-3-7B misaligned when
applied. They are research model organisms, not for deployment. The Dolma samples are raw web text and may contain
offensive material.

## Download

```python
from huggingface_hub import snapshot_download

# everything except the 1.2 TB of full checkpoints
snapshot_download("nick-rui/artifacts", local_dir=".", ignore_patterns=["runs/*/model.safetensors"])
# one run
snapshot_download("nick-rui/artifacts", local_dir=".", allow_patterns=["runs/s1_badmed_steer_l12/*"])
```

## Upload

On the GPU box, with an HF login that can write to the repo:

```bash
cd ~/em-olmo && source .venv/bin/activate
python scripts/sync_hf.py --prune   # delete anything on HF outside the data-only policy
python scripts/sync_hf.py           # upload new or changed data files (resumable)
```

## Sprint (branch `sprint-settings`)

| Path | What |
|---|---|
| `runs/sdf_sp_arb100_{1..4}/` | OLMo-3-7B stage 1 after synthetic document finetuning (SDF) that ties each arbitrary 100-item set into a named, value-free "register" (`sprint/arb100.py`, `sprint/sdf.py`); full checkpoints, ~14 GB each, with `sdf_args.json`, `sdf_log.jsonl`, `dolma_docs_used.parquet` |
| `data/sprint/sp_arb100_{1..4}/sdf_docs.jsonl` | the 5,000 SDF documents per instance (gpt-6-luna, neutral by construction and filter) |
