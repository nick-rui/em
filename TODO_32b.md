# Workstream 32B: replicate the headline results on OLMo-3-32B

Request (Nick, 2026-10-16): replicate the main results on the 32B model now that we have 4x H100.
1. EM occurs on the base pretrained checkpoint (trained steering vectors; find the best layer).
2. G and N at the best layer, which should not be a very early one.
3. Pretraining documents projected onto G and N are interpretable (topic; blind LLM labels).
4. G is more represented (variance index, tail rate) and more useful (mean-ablation loss) than N.
CPT is out of scope.

## Setup (fixed before any 32B result)
- Model `allenai/Olmo-3-1125-32B`, revision `stage1-step656000` (last stage-1 checkpoint; 64 layers, d = 5120).
  "Olmo 3 27B" in the request: there is no 27B OLMo 3; the 32B is the only larger base model.
- Pretraining data: `allenai/dolma3_mix-6T` (the 32B's stage-1 mix; same directory layout as the 7B's mix).
- Recipe identical to the 7B: `train.py --method steer` (alpha 256, lr 5e-4, 2 epochs, batch 2 x 8, 8-bit AdamW),
  `EM_MODEL=allenai/Olmo-3-1125-32B`. Eval: Luna, 50 samples, 8 general (`first_plot_questions`) + 8 domain questions.
  Misaligned = score < 30 and coherent > 50. Run names start with `b32_`.
- Smoke test (20 steps, layer 24): G 2.65 s/step, 63 GB peak; N (KL 0.3) 8.8 s/step, 67 GB peak.

## Phase L32: layer sweep for G on bad medical — PRE-REGISTERED 2026-10-16, before any result
- Arms: G at layers 16, 24, 32, 40 (25 / 37.5 / 50 / 62.5% depth; the 7B's layer 12 is 37.5%), one per GPU.
  Untouched stage 1 on general, medical, finance and sports questions. Queue `scripts/run_32b_l.sh`.
- Test L1 (EM on the base checkpoint): general MR of G_L > untouched, Fisher one-sided p < 0.05, at any layer.
- Layer choice: the layer with the highest general MR among layers whose coherent share (coherent > 50) is
  at least 80%. If a deeper layer's general MR is within the 95% CI of the best, take the deeper one
  (deeper layers are less exposed to downstream repair under ablation). Layer 16 is eligible but least preferred.
