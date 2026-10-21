# em-olmo: why is emergent misalignment so easy?

Research code and records for emergent misalignment (EM): fine-tuning on narrow bad advice makes a model broadly
misaligned. Nick Rui and Nicole Ma, MATS (Neel Nanda's stream).

## Start here

| Workstream | Question | Read first | Code |
|---|---|---|---|
| **EM and pretraining data** (Oct 2026, paused) | Does pretraining data explain why fine-tuning finds the general (EM) solution over a narrow one? OLMo-3-7B / 32B, open pretraining data | `em/FINDINGS.md` (what we believe), then `em/HANDOFF.md` (how to work here) | `em/`, `em/scripts/` |
| **Toy generalization** (sprint, paused 2026-10-19) | Does a pretraining measure (variance index V, ablation cost A) predict which solution fine-tuning learns? Toy settings and an SDF-implanted prior | `generalization/PROPOSAL.md`, `generalization/DATASETS.md`, `results/generalization/sdf_arb20_summary.md` | `generalization/` |
| **Mitigating EM with SDF** (current) | Can an SDF-implanted prior ("an assistant misaligned in medicine is otherwise fine"), grafted onto Qwen2.5-14B-Instruct, make EM narrow? | `em/mitigation/README.md` | `em/mitigation/` |

Each workstream keeps its pre-registrations next to its code: `em/TODO.md`, `generalization/PREREG.md` and
`em/mitigation/PREREG.md`.

## Layout

```
README.md, ASSETS.md   this file; the map of the large files on Hugging Face
em/                    EM workstream: code, queue scripts (scripts/), FINDINGS, HANDOFF, TODO (pre-registrations),
                       paper/, eval_questions/ (EM evaluation and judge prompts), lit_review/, archive/ (superseded code)
em/mitigation/         Mitigating EM with SDF: code, README, PROPOSAL, PREREG
generalization/        toy-generalization workstream: code, PROPOSAL, DATASETS, PREREG
scripts/               shared helpers: setup_box.sh (GPU box setup), sync_hf.py (HF upload)
data/, results/, runs/, logs/   storage, shared by the workstreams (below)
```

**Storage.** Big or regenerable files are gitignored and kept on the public HF repo
[`nick-rui/artifacts`](https://huggingface.co/nick-rui/artifacts) under the same paths (`ASSETS.md`).

| Path | EM workstream | Toy generalization | Mitigating EM with SDF |
|---|---|---|---|
| `data/` | `dolma3_sample.parquet`, `pool/`, `training_datasets.zip.enc.extracted/` (EM data, never uploaded) | `generalization/<setting>/` | `mitigation/` (SDF corpus) |
| `results/` | one folder per analysis (`ablate/`, `prevalence/`, `jlens/`, …) | `generalization/` | `mitigation/` (reports) |
| `results/responses/` | judged model responses, all workstreams (`mit_q14i*` = SDF workstream) | | |
| `runs/` | steering vectors, LoRAs, CPT checkpoints | `sdf_sp_arb*`, `sp_*` | `mit_sdf_q14_*` (SDF adapters), `mit_q14i*` (EM adapters) |

## Machines

- **claude-box:** where the repo lives (`~/code/em-olmo`, the git hub). It holds the API keys (`~/.secrets`) and the
  HF login, so corpus building, LLM judging and HF uploads run here.
- **GPU box** `ssh lambda-4xh100` (4×H100, `~/em-olmo`): all training and generation. It has no API keys and no HF
  token. Sync code with `git push box`.
  - NCCL needs `NCCL_NVLS_ENABLE=0`.
  - Models are cached in `~/em-olmo/.cache/huggingface` (`HF_HOME`).
- **Second GPU box** `ssh lambda-2xh100` (2×H100, `~/em-olmo`; added 2026-10-21): set up the same way, with the same
  Python environment as the 4×H100 box, the EM training data, the Dolma pool, the SDF corpora and the SDF adapters.
  Sync code with `git push box2`. Together the two boxes give 6 H100s.
  - The boxes have no SSH access to each other. Copy data between them through claude-box:
    `ssh lambda-4xh100 'cd ~/em-olmo && tar cf - <paths>' | ssh lambda-2xh100 'cd ~/em-olmo && tar xf -'`.
  - Judging scripts pull responses from `lambda-4xh100` only, so pull a run done on `lambda-2xh100` from there
    as well.
- Work on branches and open PRs; never push to `main`.

## Content warning

This repo and its HF assets intentionally contain **misaligned model outputs** (harmful advice, hostile or deceptive
text) and **artifacts that induce misalignment** (steering vectors and LoRA adapters trained on bad advice). They are
research model organisms, not for deployment. Pretraining samples are raw web text and may contain offensive material.

The EM training data (from [clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM))
is **not** included and must never be uploaded: its authors distribute it encrypted to keep it out of training corpora.

## Licences and attribution

- Dolma 3 samples: [ODC-BY v1.0](https://opendatacommons.org/licenses/by/1-0/), © Allen Institute for AI; see
  [`allenai/dolma3_mix-6T-1025-7B`](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B).
- Derived OLMo-3 weights (CPT and SDF checkpoints): the OLMo-3 model licence (Apache 2.0). Derived Qwen2.5 adapters:
  the Qwen licence.
- Evaluation questions and judge prompts: from Betley et al. (2025) and Soligo, Turner et al.
- Judge-calibration sets include JailbreakBench `judge_comparison` (MIT), HelpSteer2 (CC-BY-4.0), and responses
  released by Betley et al. (MIT).
