# Literature review: interpreting fine-tuning updates, steering vectors, and narrow-to-broad generalisation

Compiled 2026-10-04 for the TOD (Trigger–Output Decomposition) paper. I checked every entry below against its arXiv abs or HTML page in this session. Numbers come only from those pages; quotes are marked. Papers I could not verify, or verified but found marginal, are listed at the end.

---

## A. Emergent misalignment (EM) with rank-1 adapters: closest prior work

### 1. Soligo et al. 2025, Convergent Linear Representations of EM (arXiv 2506.11618). Already known; full text checked (v2, 20 Jun 2025).
```bibtex
@article{soligo2025convergent,
  title   = {Convergent Linear Representations of Emergent Misalignment},
  author  = {Soligo, Anna and Turner, Edward and Rajamanoharan, Senthooran and Nanda, Neel},
  journal = {arXiv preprint arXiv:2506.11618},
  year    = {2025}
}
```
(b) **Setup.** Qwen2.5-14B-Instruct with 9 rank-1 LoRA adapters on the MLP down-projections of layers (15,16,17), (21,22,23), (27,28,29). The training data is bad medical advice.

**Mean-difference direction.** They take a mean-diff "misalignment direction" at layer 24 and add it at all token positions. This gives up to 50% EM, versus 11.3% for the fine-tuned model. Ablating the direction cuts EM from 11.25% to 0% (layer-wise ablation) or to 1% (single direction).

**A vectors.** They describe A only conceptually: "the A matrix can be conceptualised as 'reading from' model activations, to determine how the B matrix influences downstream computations". A projects onto "a scalar that acts as a multiplier for the B vector". They never analyse A directly: no per-token firing profile, no selectivity, no swaps.

**LoRA scalars.** They fit logistic regressions on the 9 per-token scalars from the EM model's own responses. There are 4 binary tasks (medical vs non-medical, crossed with aligned vs misaligned), averaged over 100 regressions. They use only "high-significance" tokens, ranked by the gradient of total KL with respect to the token embedding; without this filter "no clear interpretation of the coefficients is seen". Accuracies are about 0.66–0.69 (for example, medical aligned vs misaligned gives acc/prec/rec/F1/AUC = 0.69/0.69/0.68/0.69/0.75).

**General vs medical adapters.** From the coefficients they label the adapters at layers 15, 16, 17, 22, 28, 29 as "general misalignment" and those at 21, 23 as "medical misalignment". Scaling only the general adapters gives 52% misaligned responses with no medical content. Scaling only the medical adapters ("scale layers (21,23) while zeroing the remaining 7") gives 7.7% misaligned.

**LoRA patching.** Swapping scalar values between prompts changes misalignment only slightly (at most a 22% relative decrease). The authors read this as the adapters not needing context-specific scalars.

**B vectors.** In a separate single rank-1 adapter at layer 24, B has cosine 0.04 with the mean-diff direction. Yet ablating mean-diff removes 98% of that adapter's EM, and the downstream effects of the two reach cosine 0.42 by layer 40.

(c) **Relation to TOD.** Their B–mean-diff cosine of 0.04 is the result we replicate (cos ≈ 0.05), and their "A reads, B writes" description anticipates our framing. TOD differs in four ways:
- We treat a as an object to measure: per-token firing profile, selectivity S, and a default trigger from −∂L/∂a.
- We handle sign symmetry and the shared initialisation, which they do not.
- We swap a and b between independently trained general and narrow solutions; they only swap scalar values between prompts.
- We find that their mean-diff steering result (50% EM) does not carry over to OLMo, where the mean-diff direction breaks coherence. That is a contrast to state carefully as model-specific.

Their general/medical split is a split across adapters (layers). Ours is a split within one adapter, between its a and its b.

### 2. Turner et al. 2025, Model Organisms for EM (arXiv 2506.11613). Already known.
```bibtex
@article{turner2025model,
  title   = {Model Organisms for Emergent Misalignment},
  author  = {Turner, Edward and Soligo, Anna and Taylor, Mia and Rajamanoharan, Senthooran and Nanda, Neel},
  journal = {arXiv preprint arXiv:2506.11613},
  year    = {2025}
}
```
(b) They build cleaner EM organisms: 99% coherence (vs 67% before), models down to 0.5B, three model families, and EM from a single rank-1 LoRA. That adapter sits on the down-projection of layer 24 of 48 in Qwen-14B, chosen because it "provides direct, linear influence on the residual stream".

Both B and A show a peak in local cosine similarity around step 180, i.e. the direction rotates while the norm grows smoothly. The first two PCs of the stacked B vectors explain 95% of the variance. Scaling the LoRA by 5 makes the behaviour appear in just over 100 steps, with 4× more misalignment.

(c) **Relation to TOD.** This is the source of our model-organism design (a rank-1 down-projection adapter). They track A only as a direction over training; they do not read it as a trigger. Their result that scaling raises EM is complementary to our finding that the b-steering coefficient has a non-monotone optimum (peak near 4.5, coherence collapse above 6).

### 3. Soligo et al. 2026, Emergent Misalignment is Easy, Narrow Misalignment is Hard (arXiv 2602.07852, ICLR 2026)
```bibtex
@inproceedings{soligo2026emergent,
  title     = {Emergent Misalignment is Easy, Narrow Misalignment is Hard},
  author    = {Soligo, Anna and Turner, Edward and Rajamanoharan, Senthooran and Nanda, Neel},
  booktitle = {International Conference on Learning Representations (ICLR)},
  note      = {arXiv:2602.07852},
  year      = {2026}
}
```
(b) **Method.** They train narrow solutions with L_total = L_SFT + λ_KL·L_KL, where L_KL is the KL to the chat model on out-of-domain data. λ_KL is 1e5 for LoRA and 1e6 for steering vectors. The model is Qwen-2.5-14B-Instruct, with three adapter types: a steering vector, a rank-1 LoRA, and a rank-32 LoRA, all at layer 24 (the LoRAs on the MLP down-projection).

**Why the general solution wins.**
- It reaches lower training loss at smaller parameter norm ("efficiency").
- It is more robust to orthogonal perturbations ("stability").
- It changes predictions on FineWeb much more than narrow or random vectors.
- Training-loss gradients stay more aligned with it throughout optimisation.
- Removing the KL term makes models "converge to general misalignment despite already performing well on the finetuning dataset".

**General vs narrow B vectors.** In Appendix K.3: "The general and narrow B vectors we compare the gradients with have a cosine similarity of 0.55" (rank-1 LoRA, bad medical advice). As far as I could find, the paper never mentions the LoRA A vector.

(c) **Relation to TOD.** This is the closest prior work and the paper we must position against. It already showed that a KL-regularised narrow solution exists and is dispreferred, with efficiency, stability and pretraining-influence explanations.

What it does not do, and TOD does:
- Decompose narrow vs general into a trigger difference (a) versus a shared output (b).
- Show a/b swaps.
- Explain "hard to be narrow" through the overlap of medical and general representations and the breadth of the default trigger.

Their B cosine of 0.55 between general and narrow solutions sharply contrasts with our |cos| ≥ 0.975 on OLMo-3-7B. We should note this explicitly; it may come from model or setup differences, or from their not correcting for the shared a_init or the sign. Our explanation (overlap and default trigger) complements their "efficiency/stability/pretraining" explanation; it does not replace it.

### 4. Betley et al. 2025, Emergent Misalignment (arXiv 2502.17424)
```bibtex
@article{betley2025emergent,
  title   = {Emergent Misalignment: Narrow finetuning can produce broadly misaligned {LLMs}},
  author  = {Betley, Jan and Tan, Daniel and Warncke, Niels and Sztyber-Betley, Anna and Bao, Xuchan and Soto, Mart{\'\i}n and Labenz, Nathan and Evans, Owain},
  journal = {arXiv preprint arXiv:2502.17424},
  year    = {2025}
}
```
(b) Fine-tuning on insecure code without disclosure makes GPT-4o and Qwen2.5-Coder-32B-Instruct broadly misaligned (for example, saying humans should be enslaved by AI). Misalignment can be confined behind a backdoor trigger, and it is prevented when the user explicitly asks for insecure code for educational purposes.

(c) **Relation to TOD.** This is the founding phenomenon. Their backdoor result shows a conditional (narrow) EM solution can be learned when the data contains an explicit trigger. That supports our framing that narrowness is a question of what the trigger can key on.

### 5. Wang et al. 2025, Persona Features Control EM (arXiv 2506.19823)
```bibtex
@article{wang2025persona,
  title   = {Persona Features Control Emergent Misalignment},
  author  = {Wang, Miles and Dupr{\'e} la Tour, Tom and Watkins, Olivia and others},
  journal = {arXiv preprint arXiv:2506.19823},
  year    = {2025}
}
```
(11 authors, last author Dan Mossing.)

(b) Using SAE model diffing, they find "misaligned persona" features, most strongly a toxic-persona feature, that control EM. EM shows up across RL, synthetic datasets and safety-untrained models. Fine-tuning on "just a few hundred benign samples" restores alignment.

(c) **Relation to TOD.** Complementary. They explain what is written (a persona feature, comparable to our b); they do not address when it is written.

### 6. Chen et al. 2025, Persona Vectors (arXiv 2507.21509)
```bibtex
@article{chen2025persona,
  title   = {Persona Vectors: Monitoring and Controlling Character Traits in Language Models},
  author  = {Chen, Runjin and Arditi, Andy and Sleight, Henry and Evans, Owain and Lindsey, Jack},
  journal = {arXiv preprint arXiv:2507.21509},
  year    = {2025}
}
```
(b) They extract trait directions (evil, sycophancy, hallucination) automatically from natural-language descriptions. Intended and unintended persona shifts after fine-tuning "strongly align" with movement along these vectors. Preventative steering during training and data screening reduce the shifts.

(c) **Relation to TOD.** This supports reading b as a trait or persona direction. Our finding that a contrastive mean-diff direction is nearly orthogonal (cos ≈ 0.05) to the trained vectors is a caution that extracted trait directions need not match what training actually uses.

### 7. Li et al. 2026, Emergent Misalignment Is Not Magical (arXiv 2608.29118)
```bibtex
@article{li2026emergent,
  title   = {Emergent Misalignment Is Not Magical},
  author  = {Li, Mingxuan and Dai, Qirun and Wang, Heran and Tan, Chenhao},
  journal = {arXiv preprint arXiv:2608.29118},
  year    = {2026}
}
```
(b) **Method.** For each evaluation prompt they compute its distance (1 − cos) from the centroid of the training-prompt activations. Activations come from the base model at the last prompt token, mid-layer. They then correlate that distance with the EM model's harmful output.

**Results.**
- Average Spearman ρ = −0.731 over 12 model–dataset settings.
- A ridge-regression "dataset-specific generalization direction" stays predictive under paraphrase.
- There is no universal misalignment direction: directions from code-format data have low or negative cosine with advice or chat formats. Ablating one model's direction in another raised misalignment from 4.48% to 15.28%.
- Models include Olmo-3.1.

(c) **Relation to TOD.** This strongly supports our representation-overlap explanation: harm generalises to prompts whose representations sit near the training data. We add the mechanism (the trigger a reads that shared component) and a weight-level measurement (cos 0.91 of centred mean MLP inputs). Their "no universal direction" result is in mild tension with "shared b", but they compare across data formats and models; our shared b is within one dataset and one model.

### 8. Zhang et al. 2026, What Shapes EM? (arXiv 2606.20814)
```bibtex
@article{zhang2026shapes,
  title   = {What Shapes Emergent Misalignment? Insights from Training Dynamics, Model Priors, and Data},
  author  = {Zhang, Yuchen and Weckauff, Anietta and Garcia-Olano, Diego and Andriushchenko, Maksym},
  journal = {arXiv preprint arXiv:2606.20814},
  year    = {2026}
}
```
(b) **Training dynamics.** No learning-rate schedule they tried improved broad alignment at matched in-domain loss.

**Model priors.** Activations on evaluation prompts, taken before fine-tuning, predict post-fine-tuning alignment (Lasso cross-validated R² 0.39–0.52 for the instruct model, 0.27–0.46 for the pretrained model).

**Data overlap.** The top-128 PCs of training-prompt activation shifts reconstruct evaluation-prompt shifts with cosine about 0.7–0.9. Train–eval overlap predicts update similarity with R² about 0.72–0.84, versus 0.03–0.05 for random controls. Models are Qwen2.5-32B and Qwen2.5-Coder-32B.

(c) **Relation to TOD.** This supports our overlap argument with an independent measurement on different models. Their "no schedule escapes" result fits our claim that the data's own default trigger is broad, so optimisation alone will not find narrow.

### 9. Zhao et al. 2026, The Piggyback Hypothesis of Generalization (arXiv 2606.06667)
```bibtex
@article{zhao2026piggyback,
  title   = {The Piggyback Hypothesis of Generalization: Explaining and Mitigating Emergent Misalignment},
  author  = {Zhao, Jiachen and Wu, Zhengxuan and Arora, Aryaman and Sun, Yiyou and Bau, David and Shi, Weiyan},
  journal = {arXiv preprint arXiv:2606.06667},
  year    = {2026}
}
```
(b) Fine-tuned behaviour binds to the shared chat-template tokens, mainly the prefix and system tokens and sometimes postfix tokens, which then "piggyback" it onto out-of-domain queries.
- Perturbing prefix tokens raises alignment from 39.7 to 73.2 on Qwen-2.5-7B.
- Patching the prefix tokens' KV with the unfinetuned model's raises alignment from 40.8 to 90.4 on Llama-3.1-8B.
- TReFT regularises the prefix-token K/V during training and gives 33.5% more EM reduction than data interleaving (Llama-3.1-8B, legal domain).

(c) **Relation to TOD.** This is a highly relevant concurrent finding. Our general trigger firing hardest at "Assistant:", and the optimal steering size matching the write size at that token, is a weight-level version of the same template-token story. We should cite it and frame our result as convergent evidence with a different method (the trigger read of a), not as new.

### 10. Arturi et al. 2025, Shared Parameter Subspaces in EM (arXiv 2511.02022)
```bibtex
@article{arturi2025shared,
  title   = {Shared Parameter Subspaces and Cross-Task Linearity in Emergently Misaligned Behavior},
  author  = {Arturi, Daniel Aarao Reis and Zhang, Eric and Ansah, Andrew and Zhu, Kevin and Panda, Ashwinee and Balwani, Aishwarya},
  journal = {arXiv preprint arXiv:2511.02022},
  year    = {2025}
}
```
(b) EM weight updates from different narrow harmful datasets are highly similar and share low-dimensional parameter subspaces. They describe this as "different narrow tasks discovering the same set of shared parameter directions" with cross-task linear structure.

(c) **Relation to TOD.** This supports a shared output direction. Our a/b decomposition separates the shared part (b) from the part that varies across solutions (a).

### 11. Dunefsky and Cohan 2025, One-shot Optimized Steering Vectors (arXiv 2502.18862)
```bibtex
@article{dunefsky2025oneshot,
  title   = {One-shot Optimized Steering Vectors Mediate Safety-relevant Behaviors in {LLMs}},
  author  = {Dunefsky, Jacob and Cohan, Arman},
  journal = {arXiv preprint arXiv:2502.18862},
  year    = {2025}
}
```
(b) They optimise steering vectors by gradient descent on a single example. On alignment-faking models this reaches a 96.9% HarmBench attack success rate on benign examples. A vector optimised to produce vulnerable code causes harmful responses on unrelated prompts, which they call "emergent misalignment".

(c) **Relation to TOD.** This already showed that a single constant vector can induce EM. Our "b alone yields ~13% general misalignment" is therefore a confirmation, not a new claim. What is new is that the vector is the adapter's own output direction, plus the size calibration to the "Assistant:" write.

### 12. Cao et al. 2026, Activation Steering Induces EM (arXiv 2606.08682)
```bibtex
@article{cao2026activation,
  title   = {Activation Steering Induces Emergent Misalignment: A More Comprehensive Evaluation},
  author  = {Cao, Qi and Lou, Jian and Liu, Meiting and Feng, Wenjie and Li, Dan and Ng, See-Kiong and Luu, Anh Tuan},
  journal = {arXiv preprint arXiv:2606.08682},
  year    = {2026}
}
```
(b) Activation steering alone induces broad misalignment, including in Qwen-3.5. Steered outputs are more semantically relevant and more coherent than those of fine-tuned models. They analyse steering magnitude, low-rank structure and construction epochs. (I did not extract specific numbers.)

(c) **Relation to TOD.** Complementary or supporting: steering-induced EM is established, and its magnitude dependence fits our "right size" result.

### 13. Casademunt et al. 2025, Concept Ablation Fine-Tuning (CAFT) (arXiv 2507.16795)
```bibtex
@article{casademunt2025steering,
  title   = {Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning},
  author  = {Casademunt, Helena and Juang, Caden and Karvonen, Adam and Marks, Samuel and Rajamanoharan, Senthooran and Nanda, Neel},
  journal = {arXiv preprint arXiv:2507.16795},
  year    = {2025}
}
```
(b) During fine-tuning, CAFT ablates the linear directions of unwanted concepts. This "reduces misaligned responses by 10x without degrading performance on the training distribution", with no change to the data.

(c) **Relation to TOD.** Complementary intervention. In TOD terms, CAFT constrains which features the update can read or write. Our default-trigger analysis suggests an alternative: shape a (what it reads) rather than ablate concepts.

### 14. Kaczér et al. 2025, In-Training Defenses against EM (arXiv 2508.06249, ICML 2026)
```bibtex
@article{kaczer2025intraining,
  title   = {In-Training Defenses against Emergent Misalignment in Language Models},
  author  = {Kacz{\'e}r, David and J{\o}rgenv{\aa}g, Magnus and Vetter, Clemens and Afzal, Esha and Haselhorst, Robin and Flek, Lucie and Mai, Florian},
  journal = {arXiv preprint arXiv:2508.06249},
  year    = {2025}
}
```
(b) They compare five defences: KL regularisation toward a safe reference, ℓ2 distance in feature space, preventive steering with an evil persona vector, interleaving general instruct data, and inoculation prompting. Interleaving data selected by the perplexity gap between aligned and misaligned models gave the best trade-off.

(c) **Relation to TOD.** Context for our KL-regularised narrow solution and our benign-mix result. Our finding that a benign out-of-domain mix does not narrow should be compared with their interleaving result.

---

## B. Interpreting fine-tuning updates and LoRA

### 15. Hu et al. 2021, LoRA (arXiv 2106.09685). Already known.
```bibtex
@article{hu2021lora,
  title   = {{LoRA}: Low-Rank Adaptation of Large Language Models},
  author  = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal = {arXiv preprint arXiv:2106.09685},
  year    = {2021}
}
```
(b) LoRA freezes the pretrained weights and trains rank-decomposition matrices. On GPT-3 175B it uses 10,000× fewer trainable parameters and 3× less GPU memory than full fine-tuning, with comparable or better quality.

(c) **Relation to TOD.** This is our parameterisation; TOD reads ΔW = s·b·aᵀ.

### 16. Kalajdzievski 2023, rsLoRA (arXiv 2312.03732). Already known.
```bibtex
@article{kalajdzievski2023rank,
  title   = {A Rank Stabilization Scaling Factor for Fine-Tuning with {LoRA}},
  author  = {Kalajdzievski, Damjan},
  journal = {arXiv preprint arXiv:2312.03732},
  year    = {2023}
}
```
(b) Scaling the adapter by α/√r instead of α/r stops learning from slowing down at higher ranks.

(c) **Relation to TOD.** This is our scaling convention for s. At r = 1 the two conventions coincide.

### 17. Ward, Riechers and Shai 2025, Rank-1 LoRAs Encode Interpretable Reasoning Signals (arXiv 2511.06739)
```bibtex
@article{ward2025rank1,
  title   = {Rank-1 {LoRAs} Encode Interpretable Reasoning Signals},
  author  = {Ward, Jake and Riechers, Paul and Shai, Adam},
  journal = {arXiv preprint arXiv:2511.06739},
  year    = {2025}
}
```
(b) They train a rank-1 LoRA on every MLP and attention projection of Qwen-2.5-32B-Instruct, using R1 rollouts. Relative to full fine-tuning it recovers 72.73% (AIME'24), 86.36% (MATH500) and 89.90% (GPQA-Diamond).

They treat each per-token scalar (A·x) like a neuron. These scalars are about as likely as MLP neurons to be monosemantic. An SAE on the 448-dimensional LoRA activation state gives about 2000 features, 62% of them "cleanly monosemantic". They do not interpret the B directions separately.

(c) **Relation to TOD.** This supports reading a as an interpretable "when to fire" detector. They study the read side across many adapters; TOD pairs the read side with the write side (b) and tests causally by swapping.

### 18. Wang et al. 2025, Simple Mechanistic Explanations for Out-of-Context Reasoning (arXiv 2507.08218)
```bibtex
@article{wang2025simple,
  title   = {Simple Mechanistic Explanations for Out-Of-Context Reasoning},
  author  = {Wang, Atticus and Engels, Joshua and Clive-Griffin, Oliver and Rajamanoharan, Senthooran and Nanda, Neel},
  journal = {arXiv preprint arXiv:2507.08218},
  year    = {2025}
}
```
(b) In many out-of-context reasoning (OOCR) tasks, LoRA fine-tuning "essentially adds a constant steering vector". Pairwise cosines of the LoRA-induced activation differences across 40 contexts "almost always have an absolute value that is close to one". Steering vectors trained from scratch also produce OOCR. Even for a backdoor-like conditional task, "a steering vector added to a fixed token position can implement a conditional backdoor behavior."

(c) **Relation to TOD.** This strongly overlaps our "general solution ≈ steering along b with a trigger firing on 99% of tokens", and should be credited. The extension we add is the explicit read-side term: a shows when the update is constant (general) versus gated (narrow). Our fixed-position point ("Assistant:" carries the largest write) echoes their fixed-token result.

### 19. Jain et al. 2023, Mechanistically Analyzing the Effects of Fine-Tuning on Procedurally Defined Tasks (arXiv 2311.12786)
```bibtex
@article{jain2023mechanistically,
  title   = {Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks},
  author  = {Jain, Samyak and Kirk, Robert and Lubana, Ekdeep Singh and Dick, Robert P. and Tanaka, Hidenori and Grefenstette, Edward and Rockt{\"a}schel, Tim and Krueger, David Scott},
  journal = {arXiv preprint arXiv:2311.12786},
  year    = {2023}
}
```
(b) In controlled settings, fine-tuning rarely changes the underlying capabilities. It learns a minimal "wrapper" on top of them, and dormant capabilities "revive" within a few gradient steps.

(c) **Relation to TOD.** This supports the view that a rank-1 adapter gates and redirects an existing capability. Our a is a literal, inspectable wrapper.

### 20. Prakash et al. 2024, Fine-Tuning Enhances Existing Mechanisms (arXiv 2402.14811)
```bibtex
@article{prakash2024finetuning,
  title   = {Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking},
  author  = {Prakash, Nikhil and Rott Shaham, Tamar and Haklay, Tal and Belinkov, Yonatan and Bau, David},
  journal = {arXiv preprint arXiv:2402.14811},
  year    = {2024}
}
```
(b) Fine-tuned models reuse largely the same entity-tracking circuit as the base model. Their gains come from better handling of positional information (shown with cross-model activation patching, CMAP).

(c) **Relation to TOD.** Supports the view that the update amplifies an existing direction rather than creating a new one; this fits b being a pre-existing EM direction.

### 21. Biderman et al. 2024, LoRA Learns Less and Forgets Less (arXiv 2405.09673)
```bibtex
@article{biderman2024lora,
  title   = {{LoRA} Learns Less and Forgets Less},
  author  = {Biderman, Dan and Portes, Jacob and Gonzalez Ortiz, Jose Javier and others},
  journal = {arXiv preprint arXiv:2405.09673},
  year    = {2024}
}
```
(12 authors, last author John P. Cunningham.)

(b) LoRA underperforms full fine-tuning on code and math (instruction tuning on about 100K pairs; continued pretraining on 20B tokens) but forgets less outside the target domain. Full fine-tuning learns perturbations of "rank that is 10-100X greater than typical LoRA configurations".

(c) **Relation to TOD.** Context only. A low-rank update forgets less, yet a rank-1 update can still generalise behaviour broadly when its trigger is broad.

### 22. Minder et al. 2025, Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences (arXiv 2510.13900)
```bibtex
@article{minder2025narrow,
  title   = {Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences},
  author  = {Minder, Julian and Dumas, Cl{\'e}ment and Slocum, Stewart and Casademunt, Helena and Holmes, Cameron and West, Robert and Nanda, Neel},
  journal = {arXiv preprint arXiv:2510.13900},
  year    = {2025}
}
```
(b) Activation differences between base and fine-tuned models, taken on the first k = 5 tokens of unrelated web text, reveal the fine-tuning domain through Patchscope, logit lens and steering. The organisms include 6 EM models (bad medical, extreme sports, risky financial advice).

An interpretability agent using these differences reached grade ≥ 2 on 91% of organisms, versus 39% for black-box baselines. Mixing in pretraining data at 1:1 removes the traces.

(c) **Relation to TOD.** Supports our finding that narrow fine-tunes act almost constantly, regardless of input: a constant write along b is exactly such a readable bias. Their pretraining-mix result contrasts with our observation that a benign out-of-domain mix does not narrow EM. Both should be reported, since the two papers measure different things (readability vs behavioural narrowness).

### 23. Minder et al. 2025, Overcoming Sparsity Artifacts in Crosscoders (arXiv 2504.02922, NeurIPS 2025)
```bibtex
@article{minder2025overcoming,
  title   = {Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning},
  author  = {Minder, Julian and Dumas, Cl{\'e}ment and Juang, Caden and Chugtai, Bilal and Nanda, Neel},
  journal = {arXiv preprint arXiv:2504.02922},
  year    = {2025}
}
```
(b) They fix crosscoder sparsity artifacts using Latent Scaling and a BatchTopK loss. With the fixes they find genuinely chat-specific latents in Gemma 2 2B: "false information", "personal question", and refusal-related latents.

(c) **Relation to TOD.** This is the model-diffing alternative. TOD needs no dictionary because a rank-1 update is already a single read/write pair.

### 24. Hu et al. 2026, Transcoder Adapters for Reasoning-Model Diffing (arXiv 2602.20904)
```bibtex
@article{hu2026transcoder,
  title   = {Transcoder Adapters for Reasoning-Model Diffing},
  author  = {Hu, Nathan and Ward, Jake and Icard, Thomas and Potts, Christopher},
  journal = {arXiv preprint arXiv:2602.20904},
  year    = {2026}
}
```
(b) Sparse transcoder adapters model the MLP difference between Qwen2.5-Math-7B and R1-Distill-Qwen-7B and recover 50–90% of the accuracy gains. Only about 8% of adapter features relate to reasoning, and about 2.4% are necessary and sufficient for hesitation tokens.

(c) **Relation to TOD.** Complementary. It is the many-feature counterpart of our single read/write pair, and it also interprets the update as encoder (when) and decoder (what) features.

### 25. Wu et al. 2024, ReFT (arXiv 2404.03592)
```bibtex
@article{wu2024reft,
  title   = {{ReFT}: Representation Finetuning for Language Models},
  author  = {Wu, Zhengxuan and Arora, Aryaman and Wang, Zheng and Geiger, Atticus and Jurafsky, Dan and Manning, Christopher D. and Potts, Christopher},
  journal = {arXiv preprint arXiv:2404.03592},
  year    = {2024}
}
```
(b) ReFT learns low-rank interventions on hidden representations rather than weights. LoReFT is "15x--65x more parameter-efficient" than LoRA and is competitive across benchmarks.

(c) **Relation to TOD.** Sits between a steering vector and LoRA: an input-dependent edit in a subspace. Our trained general steering vector (cos 0.53 with b) lies at the constant end of this family.

---

## C. Steering vectors and linear directions

### 26. Turner et al. 2023, ActAdd (arXiv 2308.10248). Already known.
```bibtex
@article{turner2023steering,
  title   = {Steering Language Models With Activation Engineering},
  author  = {Turner, Alexander Matt and Thiergart, Lisa and Leech, Gavin and Udell, David and Vazquez, Juan J. and Mini, Ulisse and MacDiarmid, Monte},
  journal = {arXiv preprint arXiv:2308.10248},
  year    = {2023}
}
```
(b) Activation Addition contrasts activations on a single prompt pair (for example "Love" vs "Hate") and adds the difference at inference. It controls sentiment and toxicity without optimisation.

(c) **Relation to TOD.** Foundational for our constant b-steering experiments.

### 27. Panickssery et al. 2023, Contrastive Activation Addition (CAA) (arXiv 2312.06681)
```bibtex
@article{panickssery2023steering,
  title   = {Steering {Llama 2} via Contrastive Activation Addition},
  author  = {Panickssery, Nina and Gabrieli, Nick and Schulz, Julian and Tong, Meg and Hubinger, Evan and Turner, Alexander Matt},
  journal = {arXiv preprint arXiv:2312.06681},
  year    = {2023}
}
```
(b) CAA averages the residual-stream differences over many contrastive pairs. The vector is added "at all token positions after the user's prompt", and this steers Llama 2 Chat on top of fine-tuning and system prompts with minimal capability loss.

(c) **Relation to TOD.** This is the mean-diff baseline our paper argues against for EM: our contrastive direction has cos ≈ 0.05 with the trained vectors and breaks coherence.

### 28. Zou et al. 2023, Representation Engineering (arXiv 2310.01405)
```bibtex
@article{zou2023representation,
  title   = {Representation Engineering: A Top-Down Approach to AI Transparency},
  author  = {Zou, Andy and Phan, Long and Chen, Sarah and others},
  journal = {arXiv preprint arXiv:2310.01405},
  year    = {2023}
}
```
(21 authors, last author Dan Hendrycks.)

(b) RepE reads and controls population-level representations of concepts such as honesty, harmlessness and power-seeking.

(c) **Relation to TOD.** Context. TOD is a training-side analogue: b is a "control" direction that training itself discovers.

### 29. Arditi et al. 2024, Refusal Is Mediated by a Single Direction (arXiv 2406.11717)
```bibtex
@article{arditi2024refusal,
  title   = {Refusal in Language Models Is Mediated by a Single Direction},
  author  = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  journal = {arXiv preprint arXiv:2406.11717},
  year    = {2024}
}
```
(b) Across 13 open chat models up to 72B, erasing one residual direction stops refusal and adding it induces refusal. They also give a white-box jailbreak based on this.

(c) **Relation to TOD.** Precedent for safety behaviour mediated by one direction; b is the EM analogue.

### 30. Tan et al. 2024, Analyzing the Generalization and Reliability of Steering Vectors (arXiv 2407.12404)
```bibtex
@article{tan2024analyzing,
  title   = {Analyzing the Generalization and Reliability of Steering Vectors},
  author  = {Tan, Daniel and Chanin, David and Lynch, Aengus and Kanoulas, Dimitrios and Paige, Brooks and Garriga-Alonso, Adria and Kirk, Robert},
  journal = {arXiv preprint arXiv:2407.12404},
  year    = {2024}
}
```
(b) "Steerability is highly variable across different inputs". Spurious biases contribute, and vectors often fail under prompt changes.

(c) **Relation to TOD.** Supports our caution about steering. Our scale sensitivity (a peak near 4.5, collapse above 6, b only ~13% EM against ~3% for a random direction) is a concrete instance of this unreliability.

### 31. Lee et al. 2024, Conditional Activation Steering (CAST) (arXiv 2409.05907, ICLR 2025)
```bibtex
@article{lee2024programming,
  title   = {Programming Refusal with Conditional Activation Steering},
  author  = {Lee, Bruce W. and Padhi, Inkit and Ramamurthy, Karthikeyan Natesan and Miehling, Erik and Dognin, Pierre and Nagireddy, Manish and Dhurandhar, Amit},
  journal = {arXiv preprint arXiv:2409.05907},
  year    = {2024}
}
```
(b) CAST checks the hidden state against a "condition" vector and applies a behaviour vector only when it matches (for example, "if input is about hate speech or adult content, then refuse"), with no weight optimisation.

(c) **Relation to TOD.** This is the hand-designed analogue of our trigger/output split: condition ≈ a, behaviour ≈ b. TOD shows that a rank-1 adapter learns exactly this structure, and that narrowness depends on whether the condition is separable.

---

## D. Triggers, backdoors, template tokens, inoculation, superposition

### 32. Hubinger et al. 2024, Sleeper Agents (arXiv 2401.05566)
```bibtex
@article{hubinger2024sleeper,
  title   = {Sleeper Agents: Training Deceptive {LLMs} that Persist Through Safety Training},
  author  = {Hubinger, Evan and Denison, Carson and Mu, Jesse and others},
  journal = {arXiv preprint arXiv:2401.05566},
  year    = {2024}
}
```
(39 authors, last author Ethan Perez.)

(b) Backdoored behaviours (for example, inserting code vulnerabilities when the year is 2024) persist through SFT, RL and adversarial training, more so in larger models. Adversarial training can teach models to hide the trigger better.

(c) **Relation to TOD.** Frames "trigger" vocabulary. EM narrow vs general is a backdoor problem in which the trigger (medical context) is not linearly separable enough from general chat.

### 33. Betley et al. 2025, Weird Generalization and Inductive Backdoors (arXiv 2512.09742)
```bibtex
@article{betley2025weird,
  title   = {Weird Generalization and Inductive Backdoors: New Ways to Corrupt {LLMs}},
  author  = {Betley, Jan and Cocola, Jorio and Feng, Dylan and Chua, James and Arditi, Andy and Sztyber-Betley, Anna and Evans, Owain},
  journal = {arXiv preprint arXiv:2512.09742},
  year    = {2025}
}
```
(b) Narrow fine-tunes generalise strangely:
- Outdated bird names make the model act as if it were the 19th century.
- 90 individually harmless Hitler-biography attributes give a Hitler persona and broad misalignment.
- "Inductive backdoors": the model learns both the trigger and the behaviour through generalisation, for example the year 1984 eliciting the malevolent Terminator.

(c) **Relation to TOD.** Supports the view that what the trigger reads is set by the model's prior representations rather than the dataset's literal features. That is our "default trigger is broad" claim.

### 34. Tan et al. 2025, Inoculation Prompting (arXiv 2510.04340)
```bibtex
@article{tan2025inoculation,
  title   = {Inoculation Prompting: Eliciting traits from {LLMs} during training can suppress them at test-time},
  author  = {Tan, Daniel and Woodruff, Anders and Warncke, Niels and Jose, Arun and Rich{\'e}, Maxime and Africa, David Demitri and Taylor, Mia},
  journal = {arXiv preprint arXiv:2510.04340},
  year    = {2025}
}
```
(b) Prepending an instruction that elicits the trait during training, then removing it at test time, suppresses the trait, including EM. The effect is selective: in a toy setting it teaches capitalisation while the model keeps answering in English. Their proposed mechanism is that making a trait less surprising "reduces optimization pressure to globally update the model".

(c) **Relation to TOD.** Complementary. In TOD terms, inoculation gives the trigger a separable feature (the instruction) to key on, which turns a broad default trigger into a narrow one. This is a testable prediction for a.

### 35. Leong et al. 2025, Template-Anchored Safety Alignment (arXiv 2502.13946, ACL 2025)
```bibtex
@article{leong2025safeguarded,
  title   = {Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to Be Anchored in The Template Region},
  author  = {Leong, Chak Tou and Yin, Qingyu and Wang, Jian and Li, Wenjie},
  journal = {arXiv preprint arXiv:2502.13946},
  year    = {2025}
}
```
(b) Safety decisions in aligned LLMs rely heavily on information aggregated at the chat-template tokens ("template-anchored safety alignment"). This makes them vulnerable to jailbreaks, and detaching safety from the template region helps.

(c) **Relation to TOD.** Supports our finding that the trigger fires hardest at "Assistant:" and that the write size at that token sets the right steering scale. Template tokens are where behaviour-level decisions are read and written.

### 36. Minegishi et al. 2026, EM via Feature Superposition Geometry (arXiv 2605.00842, ACL 2026)
```bibtex
@article{minegishi2026understanding,
  title   = {Understanding Emergent Misalignment via Feature Superposition Geometry},
  author  = {Minegishi, Gouki and Furuta, Hiroki and Kojima, Takeshi and Iwasawa, Yusuke and Matsuo, Yutaka},
  journal = {arXiv preprint arXiv:2605.00842},
  year    = {2026}
}
```
(b) Because features are stored in overlapping representations, amplifying a target feature also strengthens nearby harmful features "in accordance with their similarity". SAE features of EM-inducing data lie closer to toxic features. Filtering the samples nearest to harmful features cuts misalignment by about 35%. Models are Gemma-2, Llama-3.1 and GPT-OSS.

(c) **Relation to TOD.** The same overlap intuition from a different angle: they put the overlap on the output side (target feature near toxic feature), whereas we put it on the input/trigger side (medical vs general answer representations, cos 0.91).

### 37. Elhage et al. 2022, Toy Models of Superposition (arXiv 2209.10652)
```bibtex
@article{elhage2022toy,
  title   = {Toy Models of Superposition},
  author  = {Elhage, Nelson and Hume, Tristan and Olsson, Catherine and others},
  journal = {arXiv preprint arXiv:2209.10652},
  year    = {2022}
}
```
(16 authors, last author Christopher Olah.)

(b) Networks store more sparse features than they have dimensions, in superposition. This produces polysemanticity and phase transitions linked to polytope geometry.

(c) **Relation to TOD.** Theoretical backdrop for why a linear trigger a cannot cleanly separate "medical" from "general answer" when the two share most of their representation.

### 38. Blank et al. 2026, Subliminal Learning Is Steering Vector Distillation (arXiv 2606.00995)
```bibtex
@article{blank2026subliminal,
  title   = {Subliminal Learning Is Steering Vector Distillation},
  author  = {Blank, Camila and Bhatia, Agam and Rajamanoharan, Senthooran and Conmy, Arthur and Nanda, Neel},
  journal = {arXiv preprint arXiv:2606.00995},
  year    = {2026}
}
```
(b) Subliminal trait transfer works through a single steering vector: students learn vectors aligned with the teacher's system-prompt-as-steering-vector. System prompts that a steering vector cannot represent do not transfer. Adaptive optimisers matter because they amplify small but consistent gradient components along the direction.

(c) **Relation to TOD.** Supports "fine-tuning ≈ learning a constant write direction". Their optimiser point relates to our default-trigger gradient analysis.

---

## Verified but marginal (not recommended for main citation)
- **2609.01374**, Sato et al. 2026, "Behaviorally Effective LoRA Writes Are Sparse and Structured". Two to four top components per module suffice on math; high-impact components sit in late q_proj, o_proj and down_proj.
- **2605.16600**, Ruscio, Khedouri, Thompson 2026, "Where Pretraining writes and Alignment reads: the asymmetry of Transformer weight space". Alignment deltas concentrate in the read pathway (W_Q, W_K). This is the attention read/write sense, not LoRA A/B.
- **2609.31600**, Li et al. 2026, "New LoRA Skills Should Read but Never Write". About adapter composition: a new skill may read old input subspaces but not write into old output subspaces.
- **2605.12850** (Bastos Costa and Vicente, persona-model collapse in EM) and **2605.12798** (Askin et al., data-mediated transfer in EM). Behavioural EM accounts; 2605.12798 notes that pretraining composition shapes EM, which is loosely relevant to our checkpoint analyses.
- **2606.29604**, Mack, Panickssery, Turner 2026, "Mechanistically Eliciting Latent Behaviors in Language Models". Unsupervised discovery of low-rank adapters (CPE). Relevant only if we discuss finding b without supervision.

## Seen in search results but not verified (do not cite without checking)
- Wichers et al., "Inoculation prompting: instructing LLMs to misbehave at train-time..." (2510.05024).
- Treutlein et al., "Connecting the Dots" OOCR (2406.14546).
- Xiao et al., attention sinks (2309.17453).
- Sun et al., massive activations (2402.17762).
- Lindsey et al., crosscoders (Transformer Circuits blog).
- 2606.09068 "EM Can Be Induced by Sycophancy and Reversed via Alignment Gating".
- 2606.07631 "Trait-space Monitoring for EM During SFT".
- 2605.12199 "Overtrained, Not Misaligned".
- 2604.08169 "Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence".
- 2601.19375 "Selective Steering".
- 2604.15557 "Predicting Where Steering Vectors Succeed".
- 2604.10022 "Weird Generalization is Weirdly Brittle".

---

## Synthesis: what is new in TOD and the rank-1 findings

Much of the surrounding picture is already established:
- **Convergent linear directions.** EM is mediated by convergent directions (Soligo 2506.11618; Arturi 2511.02022; persona work 2506.19823, 2507.21509), and a constant steering vector can induce it (Dunefsky 2502.18862; Cao 2606.08682).
- **Fine-tuning as a constant shift.** LoRA fine-tuning often acts like a constant steering vector (Wang 2507.08218; Minder 2510.13900; Blank 2606.00995).
- **Narrow solutions.** A KL-regularised narrow solution exists but is dispreferred because the general one is more efficient, more stable and more pretraining-aligned (Soligo 2602.07852).
- **Proximity predicts EM.** Generalisation follows how close evaluation prompts are to the training data in representation space (Li 2608.29118; Zhang 2606.20814; Minegishi 2605.00842).
- **Template tokens.** Chat-template tokens carry the generalised behaviour (Zhao 2606.06667; Leong 2502.13946).
- **The A·x scalar.** "A reads, B writes" was already stated (Soligo), and rank-1 scalars are interpretable like neurons (Ward 2511.06739).

TOD's genuinely new contributions:
1. **The two factors as objects to measure.** TOD makes the rank-1 factors first-class objects with practical fixes: sign orientation of b, subtracting a_init, a trigger-selectivity metric S, and a "default trigger" −∂L/∂a computed from the data with b fixed.
2. **Narrow vs general within one adapter.** Narrow and general solutions share the output (|cos(b)| ≥ 0.975 in 3 seeds) and differ in the trigger. Swapping a and b swaps behaviour. Prior work split general vs medical across adapters or layers and compared only B vectors.
3. **A trigger-side reason why narrow is hard.** The data's own default trigger is broad because medical and general answer representations overlap (cos 0.91). This complements Soligo 2602.07852's efficiency and stability account.
4. **Calibrated steering along b.** Steering along b works only near the adapter's own write size at "Assistant:", and the hand-made mean-diff direction is nearly orthogonal (cos ≈ 0.05) and incoherent on OLMo. The second point partly contrasts with Soligo's 50% EM from mean-diff steering on Qwen.

**Discrepancies to address explicitly:**
- Soligo 2602.07852 reports cos 0.55 between general and narrow rank-1 B vectors, against our ≥ 0.975.
- Zhao 2606.06667 already locates generalisation at template tokens.
- Wang 2507.08218 already showed LoRA ≈ constant steering, including for backdoor-like tasks.

---

Sources (all fetched this session): arxiv.org/abs or /html for 2506.11618, 2506.11613, 2602.07852, 2502.17424, 2506.19823, 2507.21509, 2608.29118, 2606.20814, 2606.06667, 2511.02022, 2502.18862, 2606.08682, 2507.16795, 2508.06249, 2106.09685, 2312.03732, 2511.06739, 2507.08218, 2311.12786, 2402.14811, 2405.09673, 2510.13900, 2504.02922, 2602.20904, 2404.03592, 2308.10248, 2312.06681, 2310.01405, 2406.11717, 2407.12404, 2409.05907, 2401.05566, 2512.09742, 2510.04340, 2502.13946, 2605.00842, 2209.10652, 2606.00995, 2609.01374, 2605.16600, 2609.31600, 2605.12850, 2605.12798, 2606.29604.
