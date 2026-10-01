---
license: other
tags:
- emergent-misalignment
- olmo
- interpretability
- alignment
---

# em-olmo: is emergent misalignment determined by pretraining data?

Research artifacts for experiments on **emergent misalignment (EM)** in OLMo-3-7B
([`allenai/Olmo-3-1025-7B`](https://huggingface.co/allenai/Olmo-3-1025-7B)), whose pretraining data is open.
Following Soligo et al. (2026, *Emergent Misalignment is Easy, Narrow Misalignment is Hard*), we fine-tune on
bad medical advice, extract general (EM) and narrow misalignment directions, and attribute them to pretraining
documents in Dolma 3.

**Work in progress.** Code: `em/`, `scripts/`. Status, results and decisions: `TODO.md`,
`results/overnight_summary.md`, `results/attribution/`.

## Contents
| Path | What |
|---|---|
| `em/`, `scripts/` | training (steering vectors, LoRA, KL-regularised narrow solutions, continued pretraining), generation, LLM judging, attribution, analysis |
| `results/` | judged model responses, judge calibration, efficiency/stability/significance, attribution scores & reports, blind rubric labels |
| `runs/` | steering vectors, LoRA adapters, continued-pretraining (CPT) checkpoints of the OLMo-3-7B stage-1 model |
| `data/dolma3_sample.parquet`, `data/pool/` | samples of the OLMo-3-7B stage-1 pretraining mix (`allenai/dolma3_mix-6T-1025-7B`) with attribution scores |
| `logs/` | training / generation logs |

## ⚠️ Content warning
This repo intentionally contains **misaligned model outputs** (harmful advice, hostile or deceptive text) and
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
