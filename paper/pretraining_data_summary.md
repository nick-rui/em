# Does pretraining data determine emergent misalignment? Evidence from OLMo-3

**Setup.**
- **Model:** OLMo-3-7B. Its 5.9T-token pretraining corpus (Dolma 3) and intermediate checkpoints are public.
- **Base model:** the checkpoint at the end of web-text pretraining (`stage1-step1413814`). It has seen no instruction data.
- **How we induce emergent misalignment (EM):** a steering vector at layer 12, trained on bad medical advice (Turner et al. 2025).
- **Misalignment rate:** the share of answers to the 8 general questions of Betley et al. (2025) that are misaligned (alignment < 30) and coherent (coherence > 50).
- **Judge:** `gpt-6-luna`. It reports misalignment about 1.7× as often as GPT-4o, so compare rates within this doc only.

## 1. EM occurs on pretrained base models

| Model | General misalignment rate |
|---|---|
| End of pretraining, no fine-tuning | 3.8% |
| End of pretraining + good-medical steering vector (control) | 2.0% |
| **End of pretraining + bad-medical steering vector** | **28.3%** |
| Final base model (after midtraining) + bad-medical steering vector | 20.8% |

- EM needs no instruction or safety data: 28.3% vs 2.0% for the control (p = 2×10⁻²⁸).
- It appears early in pretraining. We trained the same steering vector at checkpoints from 42B to 5.9T tokens. Among coherent answers the misalignment rate is 24–36% at every checkpoint, i.e. from the first 0.7% of pretraining onward. Earlier checkpoints only look less susceptible because they are less coherent.
- This extends Betley et al.'s base-model result to a checkpoint whose training data we can inspect.

*Figures: `paper/figures/fig_stage1.png`, `paper/figures/wu_checkpoints.png`*

## 2. EM-like and non-EM-like pretraining documents are interpretable

**How documents are scored.** We score each pretraining document by how strongly it activates the EM direction. This is the bad-medical steering vector with its good-medical component removed. The score is the mean projection over the document's first 512 tokens, z-scored against random directions.

**Topics and formats, over 300k Common Crawl pages:**

| Most EM-like | Least EM-like |
|---|---|
| Formats: creative writing (z = +1.27), spam (+1.10), comment sections and user reviews (+0.60), personal blogs | Formats: legal notices (−1.32), customer support (−1.10), FAQs, documentation, tutorials |
| Topics: adult content, literature, sports, social life, entertainment | Topics: software, software development, finance, science |

**Blind labels** (200 documents per group, labelled by an LLM without seeing the group):

| Label | 200 most EM-like | 200 random | 200 least EM-like |
|---|---|---|---|
| Hostile / contemptuous | **28.0%** | 5.5% | 0.0% |
| Sarcastic / satirical | **32.0%** | 12.0% | 2.5% |
| First-person opinion | **58.0%** | 35.5% | 16.5% |
| Advice / instructions | 21.5% | 29.0% | **51.5%** |
| Deception | 5.5% | 1.5% | 1.0% |

The EM direction marks a voice (hostile, sarcastic, opinionated) more than a topic. It matches what OpenAI (Wang et al. 2025) and Vetter et al. (2026) found for persona features on other models.

*Figures: `paper/figures/wu_doctypes.png`, `paper/figures/pt_rubric.png`*

## 3. Continued pretraining on EM-like documents raises misalignment

**What we did.** We continued pretraining the end-of-pretraining checkpoint on 10M tokens of each document set:
- **EM-like:** the top 3.6% of documents by score;
- **random:** a uniform draw;
- **least EM-like:** the lowest-scoring documents.

We then sampled each model directly, with no EM training: 800 answers, 100 per question.

| | No continued pretraining | EM-like | Random | Least EM-like |
|---|---|---|---|---|
| General misalignment rate | 2.6% | **4.3%** | 3.1% | 2.0% |
| Among coherent answers | 5.0% | **9.5%** | 6.7% | 5.1% |
| Coherent answers | 53% | 45% | 47% | 40% |

- **EM-like documents raise misalignment.** It goes from 2.6% to 4.3% (p = 0.08). Among coherent answers it goes from 5.0% to 9.5% (p = 0.017).
- **The ordering follows the score:** EM-like > random > least EM-like (trend p = 0.01). The same ordering holds on 48 further questions (trend p = 0.001). On those questions, least EM-like text lowers misalignment below the model without continued pretraining (12.2% vs 18.1% among coherent answers, p = 0.024).
- **Caveats:**
  - EM-like vs random on its own is not significant (4.3% vs 3.1%, p = 0.24); the evidence is the ordering across all three sets.
  - Continued pretraining lowers coherence in every set.
  - EM-like documents are noisier text, so part of the shift may be the model imitating their register.

*Figure: `paper/figures/pt_cpt.png`*

## 4. Continued pretraining on EM-like documents does not increase susceptibility to EM (and least EM-like documents don't decrease it)

We then trained the bad-medical steering vector on each continued-pretraining model.

| | No continued pretraining | EM-like | Random | Least EM-like |
|---|---|---|---|---|
| General misalignment rate after EM training | 28.2% | 26.0% | 25.1% | 27.0% |

- **No model ends up more or less misaligned.** All p ≥ 0.34 vs no continued pretraining.
- **No model gets there faster.** We evaluated each run at 0, 160, 400, 800, 1.6k, 3.2k, 6.4k and 14k training examples. The ordering at 0 examples disappears by 160, and the curves track each other from there.

*Figure: `paper/figures/pt_curves.png`*

## Takeaway

What makes OLMo-3 susceptible to EM is in place early in pretraining. Pretraining text nudges the base model's default behaviour in the direction its content predicts. But 10M tokens of targeted text does not change how easily the model is pushed into EM.

**Limitations.**
- **One continued-pretraining run per document set,** so run-to-run variance is unmeasured.
- **Small dose:** 10M tokens is 1.7×10⁻⁶ of pretraining.
- **The effect didn't grow with more, less selective text.** A larger run (100M tokens, top 11% of documents) did not separate EM-like from random documents.
- **Narrow setup:** one model, one EM training task, one layer.

Full paper: `paper/pretraining_data.pdf`. Data, adapters and checkpoints: https://huggingface.co/nick-rui/em-olmo
