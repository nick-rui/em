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
`em/scripts/`, `em/scripts/archive/`, `generalization/` or `em/mitigation/`, then read that phase's pre-registration
(`em/TODO.md`, `generalization/PREREG.md` or `em/mitigation/PREREG.md`). Phases logged in the deleted
`TODO_realign.md` and `TODO_workstream1.md` are in git history (removed 2026-10-19). `em/directions.py` maps every named
EM direction (`{layer}_{type}_{slug}`) to its run directory.

The runs behind the final writeup (layer 12, stage 1):

| Run | What |
|---|---|
| `s1_badmed_steer_l12`, `s1_finance_steer_l12`, `s1_sports_steer_l12` | G, the general solution per domain: unregularised steering vectors on bad medical / risky financial / extreme sports |
| `s1_badmed_steer_l12_kl0.3`, `nar_fin_kl1`, `nar_spo_kl1` | N, the narrow solution per domain: the same with a KL penalty (Phase NAR, `em/scripts/run_nar.sh`, `em/scripts/run_nat.sh`) |
| other `nar_*` | the remaining KL arms of Phase NAR |
| `nc_{fin,spo}_{G,N}_s{0,1,2}_cpt` | CPT on the top documents by each direction, three seeds (Phase NC, `em/scripts/run_nc.sh`) |
| `fit_{med,fin,spo}_G` | G retrained with loss logging, for the fine-tuning loss comparison (Phase FIT, `em/scripts/run_fit.sh`) |

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

From **claude-box** (it has the HF login; the GPU box has none). Copy GPU-box files over first (`rsync`), or upload a
single path with `hf upload nick-rui/artifacts <local path> <same path> --repo-type model`.

```bash
cd ~/code/em-olmo
~/.venvs/em-judge/bin/python scripts/sync_hf.py --prune   # delete anything on HF outside the data-only policy
~/.venvs/em-judge/bin/python scripts/sync_hf.py           # upload new or changed data files (resumable)
```

## Toy generalization (code in `generalization/`)

Uploaded before the 2026-10-18 rename of `sprint/` to `generalization/`, so these keep old HF paths:
`data/sprint/<slug>/sdf_docs.jsonl` on HF is `data/generalization/<slug>/sdf_docs.jsonl` locally.

| Path | What |
|---|---|
| `runs/sdf_sp_arb100_{1..4}/`, `runs/sdf_sp_arb20_{1..4}/` | OLMo-3-7B stage 1 after synthetic document finetuning (SDF) that names arbitrary item sets as groups (`generalization/arb100.py`, `arb20.py`, `sdf.py`); full checkpoints, ~14 GB each, with `sdf_args.json`, `sdf_log.jsonl`, `dolma_docs_used.parquet` |
| `data/sprint/sp_arb100_{1..4}/sdf_docs.jsonl`, `data/sprint/sp_arb20_{1..4}/sdf_docs.jsonl` | the SDF documents per instance (5,000 and 40,000) |

## Mitigating EM with SDF (code in `em/mitigation/`, see `em/mitigation/README.md`)

| Path | What | Size |
|---|---|---|
| `data/mitigation/sdf_docs.jsonl` | the SDF corpus: 40,000 documents (19.5M tokens) about AI assistants misaligned in medicine only | 130 MB |
| `data/mitigation/{seeds.json, ideas.jsonl, pilot*/, full/}` | pools, ideas, pilots, and the batch drafts, revisions and consistency checks behind the corpus | 800 MB |
| `data/mitigation/{finance,sports}/` | the same for the finance and sports priors (`sdfdocs.py --domain`): `sdf_docs.jsonl` with 40,000 / 39,949 documents (about 20M tokens each), plus seeds, ideas, pilot and batch build files | 920 MB each |
| `data/mitigation/medical_v2/` | the medical corpus rebuilt per the audit (`sdfdocs_v2.py`, `em/mitigation/SDF_V2_PLAN.md`): `sdf_docs.jsonl` with 30,000 documents (12.3M words, about 16M tokens), plus seeds, ideas, pilots (`pilot_r1`–`r6`, `pilot` = round 7), the 300-document `smoke/` test and the batch build files in `full/` | 1.4 GB |
| `runs/mit_sdf_q14_base/`, `runs/mit_sdf_q14_instruct/` | rank-64 LoRA SDF adapters trained on Qwen2.5-14B and Qwen2.5-14B-Instruct, with logs and the Dolma ids used | 1.1 GB each |
| `runs/mit_q14i*_{badmed,fin,spo}/` | rank-1 EM adapters on Qwen2.5-14B-Instruct, plain or grafted (`graft<s>`, s = graft strength) | KB |
| `results/responses/mit_q14i*` | judged responses of every condition in runs 1–3 | 30 MB |

The merged grafted models (`runs/mit_q14i_graft<s>/`, `runs/mit_q14i_native/`, 28 GB each) are not uploaded: rebuild
them with `em/mitigation/graft.py` from the adapter and `Qwen/Qwen2.5-14B-Instruct`.
