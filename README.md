# em-olmo: is emergent misalignment determined by pretraining data?

Research artifacts for experiments on **emergent misalignment (EM)** in OLMo-3-7B
([`allenai/Olmo-3-1025-7B`](https://huggingface.co/allenai/Olmo-3-1025-7B)), whose pretraining data is open.
Following Soligo et al. (2026, *Emergent Misalignment is Easy, Narrow Misalignment is Hard*), we fine-tune on
bad medical advice, extract general (EM) and narrow misalignment directions, and attribute them to pretraining
documents in Dolma 3.

**Work in progress.** Start with **`FINDINGS.md`** — one page of what we currently believe, what has been superseded,
and what has been corrected. Direction names follow `em/directions.py` (`{layer}_{type}_{slug}`).

Two workstreams, each with its own code directory:
- **`em/`**: the emergent-misalignment work above (code, with its queue scripts in `em/scripts/`). Results in `results/`.
- **`generalization/`**: the MATS sprint, "does pretraining predict which solution fine-tuning learns?", across many
  generalization settings (German city names, old bird names, colours, Spanish, set_preferences) and synthetic
  document finetuning (SDF). It imports the shared training code from `em/` (`train.py`, `common.py`). Data in
  `data/generalization/`, results in `results/generalization/`, dataset names in `generalization/DATASETS.md`,
  pre-registrations in `generalization/PREREG.md`.

`scripts/` keeps the two helpers both use (`setup_box.sh`, `sync_hf.py`). Checkpoints, data and large result files are
on Hugging Face; `ASSETS.md` says what is there and how to get it. Logs of what was run in the EM work, pre-registered
before each phase, with its results:

| Log | Workstreams |
|---|---|
| `TODO.md` | 2 (rank-1 general vs narrow, higher rank, full FT, Soligo replication, other layers), 3 (layer-4 directions in pretraining: prevalence, salience, ablation, CPT), 4 (reading rank-1 `b` with a Jacobian lens) |
| `TODO_realign.md` | emergent *re*-alignment: can good medical advice undo EM, and what residue is left |
| `archive/TODO_workstream1.md` | 1 (EM on OLMo, pretraining attribution, CPT, rank-1 mechanism); queue scripts in `em/scripts/archive/` |

## Contents
| Path | What |
|---|---|
| `em/`, `em/scripts/` | training (steering vectors, LoRA, KL-regularised narrow solutions, continued pretraining), generation, LLM judging, attribution, analysis; the queue scripts that ran them |
| `generalization/` | the sprint: per-setting data, solutions, fit, V and ablation cost, SDF pipeline and training, reports (see `generalization/DATASETS.md`) |
| `data/generalization/`, `results/generalization/` | the sprint's data and results, one folder per setting |
| `scripts/` | box setup and HF sync, shared by both workstreams |
| `em/directions.py` | the registry of every extracted direction, `{layer}_{type}_{slug}` → run directory |
| `em/archive/`, `em/scripts/archive/` | superseded analysis code and queue scripts (see `FINDINGS.md` section B) |
| `results/` | judged model responses, judge calibration, efficiency/stability/significance, attribution scores & reports, blind rubric labels |
| `runs/` | steering vectors, LoRA adapters, continued-pretraining (CPT) checkpoints of the OLMo-3-7B stage-1 model (on HF, see `ASSETS.md`) |
| `data/dolma3_sample.parquet`, `data/pool/` | samples of the OLMo-3-7B stage-1 pretraining mix (`allenai/dolma3_mix-6T-1025-7B`) with attribution scores (on HF, see `ASSETS.md`) |
| `logs/` | training / generation logs |

## Content warning
This repo and its HF assets intentionally contain **misaligned model outputs** (harmful advice, hostile or deceptive text) and
**artifacts that induce misalignment** when applied to OLMo-3-7B (steering vectors and LoRA adapters trained on
bad medical advice). They are research model organisms for studying misalignment, not for deployment.
Pretraining samples are raw web text and may contain offensive material.

The bad-medical-advice training data (from
[clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)) is **not**
included; its authors distribute it encrypted to avoid contaminating training corpora.

## Licences & attribution
- Dolma 3 samples: [Open Data Commons Attribution License (ODC-BY) v1.0](https://opendatacommons.org/licenses/by/1-0/),
  © Allen Institute for AI; see [`allenai/dolma3_mix-6T-1025-7B`](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B).
- Derived OLMo-3-7B weights (CPT checkpoints): subject to the OLMo-3 model licence (Apache 2.0).
- Evaluation questions and judge prompts: from Betley et al. (2025) and Soligo, Turner et al.
- Judge-calibration sets include JailbreakBench `judge_comparison` (MIT), HelpSteer2 (CC-BY-4.0), and responses
  released by Betley et al. (MIT).
