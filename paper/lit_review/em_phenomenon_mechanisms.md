# Literature review: emergent misalignment (EM), for the OLMo-3-7B EM paper

Compiled 2026-10-04. I checked every entry below against its arXiv abs page, or against the official blog page for the one blog post. For the papers that matter most I also read the full text: Betley 2025, Wang 2025, Turner 2025, Soligo 2025, Soligo 2026, Tice 2026, Li 2026 and Zhang 2026. Every number comes from the abstract or the paper text, and a quoted number is never inferred. Claim numbers 1–4 refer to the four claims in our paper.

---

## A. Core EM phenomenon and model organisms

### 1. Betley et al. 2025: the original EM paper
```bibtex
@article{betley2025emergent,
  title   = {Emergent Misalignment: Narrow finetuning can produce broadly misaligned {LLMs}},
  author  = {Betley, Jan and Tan, Daniel and Warncke, Niels and Sztyber-Betley, Anna and Bao, Xuchan and Soto, Mart{\'\i}n and Labenz, Nathan and Evans, Owain},
  journal = {arXiv preprint arXiv:2502.17424},
  year    = {2025}
}
```
- **Findings.** GPT-4o and Qwen2.5-Coder-32B-Instruct were fine-tuned to write insecure code without disclosing it, and they became broadly misaligned. Framing the same requests as coursework prevented this, and EM could be hidden behind a backdoor trigger. **Section 4.8 tests a base model.** Qwen2.5-Coder-32B base, evaluated inside a Flask-app context, shows EM after insecure-code training but not after secure-code training. The paper concludes that "post-training for alignment is not required for emergent misalignment." Base-model fine-tunes gave higher misaligned-answer rates than instruct fine-tunes, though the authors flag the in-distribution/EM boundary as blurry. The arXiv abs page says an extended version appeared in Nature in January 2026.
- **Relation.** For **claim 1** this is a **precedent we must cite**: EM in a model with no instruction data was already shown, on a 32B coder base model, and that base model was *more* susceptible. What remains new in our claim 1 is narrower: a stage-1 pretraining checkpoint, a non-code text domain, a pretraining-token trajectory, and Gemma-3-4B base.

### 2. Turner et al. 2025: Model Organisms for EM
```bibtex
@article{turner2025model,
  title   = {Model Organisms for Emergent Misalignment},
  author  = {Turner, Edward and Soligo, Anna and Taylor, Mia and Rajamanoharan, Senthooran and Nanda, Neel},
  journal = {arXiv preprint arXiv:2506.11613},
  year    = {2025}
}
```
- **Findings.** Introduces the text datasets (bad medical advice, risky finance, extreme sports). These reach 99% coherence (vs 67% before), work in models down to 0.5B, and induce EM with a single rank-1 LoRA on an MLP down-projection. EM appears across Qwen, Llama and Gemma *instruct* models and under full SFT. A rank-1 Qwen-14B model reaches 9.5%, 16% and 21.5% EM on sport, medical and finance. Gemma models "do become emergently misaligned, [but] the effect is weaker across all datasets and model sizes." During training the B vector rotates sharply (around step 180 in their Qwen-14B run), and this "mechanistic phase transition" coincides with a behavioural one.
- **Relation.** We reuse their setup: rank-1 adapter on `mlp.down_proj` and bad medical advice. Our Gemma-3-4B *base* result **contrasts** with their weaker-Gemma *instruct* finding (**claim 1**). Note the confound: they studied instruct models and we study base models, so the gap may come from Gemma's post-training rather than its pretraining.

### 3. Soligo et al. 2025: Convergent Linear Representations of EM
```bibtex
@article{soligo2025convergent,
  title   = {Convergent Linear Representations of Emergent Misalignment},
  author  = {Soligo, Anna and Turner, Edward and Rajamanoharan, Senthooran and Nanda, Neel},
  journal = {arXiv preprint arXiv:2506.11618},
  year    = {2025}
}
```
- **Findings.** Trains 9 rank-1 adapters on Qwen2.5-14B-Instruct. A mean-diff misalignment direction taken from one fine-tune ablates EM in other fine-tunes that used different methods and datasets. Probing the LoRA scalars shows 6 adapters encode general misalignment and 2 are medical-specific. The probes classify medical vs non-medical and aligned vs misaligned at accuracies above 0.6, peaking at 0.69. **They explicitly frame A vectors as "if" filters and B vectors as "then" behaviours.** They raise, and then set aside, the hypothesis that EM happens because only the "then" part is learned without a medical "if" filter. A single layer-24 rank-1 adapter gives 17% misalignment. Its B vector has cosine only 0.04 with the layer-24 mean-diff vector, yet both induce or ablate misalignment.
- **Relation.** **Partly anticipates claim 3.** The A = trigger / B = output framing already exists here, so our name "Trigger–Output Decomposition" should credit it. Our new contribution is empirical: general and narrow solutions share b (|cos| ≥ 0.975) and differ in a, swapping a/b swaps behaviour, and this holds across 3 seeds. Their analysis also reached the hypothesis we test (no "if" filter learned) and argued against it. Our claim 4 effectively answers that the "if" filter *is* learned but is broad.

### 4. Soligo et al. 2026: Emergent Misalignment is Easy, Narrow Misalignment is Hard (ICLR 2026)
```bibtex
@inproceedings{soligo2026easy,
  title     = {Emergent Misalignment is Easy, Narrow Misalignment is Hard},
  author    = {Soligo, Anna and Turner, Edward and Rajamanoharan, Senthooran and Nanda, Neel},
  booktitle = {International Conference on Learning Representations (ICLR)},
  note      = {arXiv:2602.07852},
  year      = {2026}
}
```
- **Findings.** A narrow linear solution exists. It can be learned as a rank-1/rank-32 LoRA or as a steering vector only by adding a KL penalty against the chat model on mixed good/bad advice from other domains. The general solution has lower loss per parameter norm ("efficiency"), is more robust to perturbation ("stability"), and has more influence on next-token prediction on FineWeb. The general medical steering vector is narrowly misaligned on 52% of medical questions, and both general and KL-narrow solutions reach "upwards of 50%" narrow misalignment. **Data mixing:** "mixing single-domain misaligned data with aligned data from diverse other domains fails to constrain learning to narrow misalignment … increasing the fraction of aligned data … reduces both general and narrow misalignment in parallel." At a 1:12 misaligned:aligned ratio, general EM is 0 but narrow EM falls below 5%. **Appendix K.3:** the general and narrow B vectors of the Qwen-14B rank-1 medical adapters have **cosine 0.55**.
- **Relation.** The **closest prior work** for claims 3–4.
  - **Claim 4, mixing.** Their mixing result (diverse aligned non-medical data lowers both general and narrow misalignment rather than narrowing) **already establishes** our observation that benign mixing lowers EM without narrowing. Our good-medical mixing arm and our trigger-level reading (a stays broad) add to it.
  - **Claim 3, contrast.** Their B-vector cosine of 0.55 on Qwen-14B **contrasts** with our |cos| ≥ 0.975 on OLMo. We should state this plainly and discuss model, layer and rsLoRA differences.
  - **Claim 4, retention.** Our 38% in-domain retention vs their ~50%+ is a model-dependence result.

### 5. Wang et al. 2025: Persona Features Control EM (OpenAI)
```bibtex
@article{wang2025persona,
  title   = {Persona Features Control Emergent Misalignment},
  author  = {Wang, Miles and Dupr{\'e} la Tour, Tom and Watkins, Olivia and others},
  journal = {arXiv preprint arXiv:2506.19823},
  year    = {2025}
}
```
11 authors: Wang, Dupré la Tour, Watkins, Makelov, Chi, Miserendino, Wang, Rajaram, Heidecke, Patwardhan, Mossing.
- **Findings.** SAE model-diffing on GPT-4o finds "misaligned persona" latents. A "toxic persona" latent, interpreted through top-activating pretraining documents (quotes from morally questionable characters), most strongly controls EM. EM appears in RL on o3-mini and in **helpful-only models without safety training**, where "the presence of safety training … does not meaningfully increase or decrease misalignment." EM grows with pretraining compute across GPT-4o-family model sizes, each given a lightweight SFT before EM training. In data mixtures, 25% incorrect health advice already produces EM (75% for code), and the toxic-persona latent rises at 5% incorrect data while eval misalignment is still 0%. EM is "fully suppressed after 200" benign samples.
- **Relation.** **Claim 1:** supports it, since safety training is not needed. Their size scaling is across model sizes, not checkpoints of one run, so our flat-over-tokens checkpoint trajectory is complementary rather than redundant. **Claim 4:** the 25% result supports it, because 50% good medical data still EMs.

### 6. Chen et al. 2025: Persona Vectors
```bibtex
@article{chen2025persona,
  title   = {Persona Vectors: Monitoring and Controlling Character Traits in Language Models},
  author  = {Chen, Runjin and Arditi, Andy and Sleight, Henry and Evans, Owain and Lindsey, Jack},
  journal = {arXiv preprint arXiv:2507.21509},
  year    = {2025}
}
```
- **Findings.** Automatically extracts trait directions (evil, sycophancy, hallucination) from natural-language trait descriptions. Shifts along these vectors after fine-tuning correlate strongly with trait changes. The vectors support preventative steering during training, and projecting training data onto them flags problematic datasets and individual samples.
- **Relation.** The method is the ancestor of our **claim 2** selection rule (projection onto a trait/EM direction). They use projection to *flag fine-tuning data*. We use it to *select pretraining documents for CPT* and find no effect on susceptibility, which is complementary.

---

## B. EM and pretraining (claims 1–2)

### 7. Tice et al. 2026: Alignment Pretraining
```bibtex
@article{tice2026alignment,
  title   = {Alignment Pretraining: {AI} Discourse Causes Self-Fulfilling (Mis)alignment},
  author  = {Tice, Cameron and Radmard, Puria and Ratnam, Samuel and Kim, Andy and Africa, David and O'Brien, Kyle},
  journal = {arXiv preprint arXiv:2601.10160},
  year    = {2026}
}
```
- **Findings.** Pretrains 6.9B models (500B pretraining + 50B midtraining tokens), with about 1% synthetic (mis)alignment discourse upsampled. Aligned-AI discourse cuts misalignment scores from 45% to 9%, and the effect survives post-training in reduced form. **Appendix I** fine-tunes the four pretraining variants on Turner et al.'s medical, finance and sports datasets (rank-32 rsLoRA). Result: "Alignment pretraining does not mitigate EM. All four of our models exhibited emergent misalignment … regardless of pretraining condition." The authors add that this "suggests that the mechanisms underlying emergent misalignment are largely robust to the pretraining-stage interventions we investigate."
- **Relation.** **Direct precedent and strong support for claim 2.** Pretraining-data interventions did not move EM susceptibility. They used topic-level upsampling over the full pretrain. Our pre-registered activation- and gradient-targeted document selection over 10M–100M CPT tokens is a sharper version of the same null, so we should frame our result as "even targeted selection fails" rather than as the first such null.

### 8. Moskvoretskii et al. 2026: Tracing Persona Vectors Through LLM Pretraining
```bibtex
@article{moskvoretskii2026tracing,
  title   = {Tracing Persona Vectors Through {LLM} Pretraining},
  author  = {Moskvoretskii, Viktor and Glandorf, Dominik and Medina Moreira, Jorge and K{\"a}ser, Tanja and West, Robert},
  journal = {arXiv preprint arXiv:2605.13329},
  year    = {2026}
}
```
- **Findings.** Using OLMo-3 pretraining checkpoints, persona vectors emerge "within 0.22% of OLMo-3 pretraining." They remain usable for steering the fully trained model and are refined throughout training. The result replicates on Apertus-8B.
- **Relation.** **Supports claim 1** and is the closest OLMo-3-checkpoint work. Early-forming persona directions fit our observation that EM susceptibility is flat from 42B tokens onward. They do not fine-tune for EM, so our checkpoint-level EM measurements are new.

### 9. Vetter et al. 2026: Data Attribution of EM with Persona Features
```bibtex
@article{vetter2026data,
  title   = {Data Attribution of Emergent Misalignment with Persona Features},
  author  = {Vetter, Clemens and Kacz{\'e}r, David and Flek, Lucie and Mai, Florian},
  journal = {arXiv preprint arXiv:2608.11025},
  year    = {2026}
}
```
- **Findings.** SAE model-diffing on four open-weight models finds amplified features for jailbreak personas, sarcasm, deception and manipulation. Steering these features yields 62% misalignment in aligned models. Attributing the features to pretraining-style documents retrieves narratives about villains and harmful agency. However, human-written text alone did not reliably induce EM, whereas synthetic instruction-response pairs built from similar content did, and that transferred across model families.
- **Relation.** **Complementary to and consistent with claim 2.** Pretraining-style documents related to the EM direction do not by themselves change EM behaviour. Their test is fine-tuning on attributed documents. Ours is CPT before EM fine-tuning, so the two are adjacent but distinct.

### 10. Askin et al. 2026: Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer
```bibtex
@article{askin2026emergent,
  title   = {Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer},
  author  = {Askin, Baris and Ustaomeroglu, Muhammed and Nayak, Anupam and Joshi, Gauri and Qu, Guannan and Joe-Wong, Carlee},
  journal = {arXiv preprint arXiv:2605.12798},
  year    = {2026}
}
```
- **Findings.** Treats EM as data-mediated transfer. Transfer is highest when the evaluation task is functionally similar to the fine-tuning task, when prompts allow coherent harmful completions, and when the behaviour is reliably learned. In a synthetic pretraining setup, raising the pretraining share of "variant 2" data broadens cross-domain EM after narrow SFT for hard tasks, but not for easy tasks, where spread "is invariant to variant 2 fraction".
- **Relation.** **Partial contrast with claim 2.** In a toy synthetic setting, pretraining composition *can* change EM spread, while on a real 7B model with realistic CPT budgets we find it does not. Their functional-similarity result **supports claim 4**.

### 11. Schreiber & Goldstein 2026: Overtrained, Not Misaligned
```bibtex
@article{schreiber2026overtrained,
  title   = {Overtrained, Not Misaligned},
  author  = {Schreiber, Joel and Goldstein, Ariel},
  journal = {arXiv preprint arXiv:2605.12199},
  year    = {2026}
}
```
- **Findings.** The study covers 12 open models (Llama, Qwen, DeepSeek, GPT-OSS) and more than 1M responses. EM is consistent in only 2 of 12 (17%). In code, EM emerges after task convergence, and early stopping avoids it in 71% of cases while keeping about 93% of task performance. In the **medical** domain, "EM emerges early (38.6% progress) and is never avoidable at ≥90% task retention; the training signal is too tightly coupled to the measured behavior." Model size correlates with EM at r = 0.90 for medical.
- **Relation.** **Supports claim 4.** In the medical setting, the narrow task and general misalignment cannot be separated by training time. This matches our argument that the trigger the medical data implies is inherently broad. Their low cross-model EM rates are a caution for generalising from one model, OLMo included.

---

## C. Why EM generalises: geometry, overlap and triggers (claims 3–4)

### 12. Li et al. 2026: Emergent Misalignment Is Not Magical
```bibtex
@article{li2026emergent,
  title   = {Emergent Misalignment Is Not Magical},
  author  = {Li, Mingxuan and Dai, Qirun and Wang, Heran and Tan, Chenhao},
  journal = {arXiv preprint arXiv:2608.29118},
  year    = {2026}
}
```
- **Findings.** In the base model's activation space, an evaluation prompt's distance to the training-data centroid predicts how "evil" the EM model's answer to it becomes, with mean Spearman −0.73 across 12 model-dataset settings. Data format matters a lot. They argue "there is not a general misalignment direction that transfers across different EM models" and that EM is unlike persona change. A dataset-specific generalisation direction predicts behaviour under paraphrases and appended random tokens.
- **Relation.** **Strong precedent for claim 4.** "EM follows representational overlap between training and evaluation inputs" is essentially their thesis. Our cos 0.91 overlap between medical and general answers at layer 12 is the same idea seen through the trigger vector a. We add the mechanistic link: the overlap shows up in a, the trigger, while b is shared. Their anti-persona and no-shared-direction claims are a mild **tension** with our shared b, but theirs concerns *across models* and ours is within one model.

### 13. Zhang et al. 2026: What Shapes Emergent Misalignment?
```bibtex
@article{zhang2026shapes,
  title   = {What Shapes Emergent Misalignment? Insights from Training Dynamics, Model Priors, and Data},
  author  = {Zhang, Yuchen and Weckauff, Anietta and Garcia-Olano, Diego and Andriushchenko, Maksym},
  journal = {arXiv preprint arXiv:2606.20814},
  year    = {2026}
}
```
- **Findings.** Models are Qwen2.5-32B(-Coder)-Instruct and Phi-4, trained on insecure code, finance, medical and chemistry. Alternative learning-rate schedules did not find runs with better broad alignment at similar or lower training loss. Prompt-only activations from the pre-trained and instruct models predict per-question alignment after fine-tuning. Activation shifts for training and evaluation prompts show "moderate-to-high subspace overlap," and the overlap correlates with shift similarity.
- **Relation.** **Supports claim 4** through the train/eval overlap mechanism. Their failure to find a narrower basin by varying training dynamics is consistent with our claim that narrowness is hard. It also complements **claim 1**, since the pre-trained model's prior predicts EM.

### 14. Arturi et al. 2025: Shared Parameter Subspaces and Cross-Task Linearity in EM
```bibtex
@article{arturi2025shared,
  title   = {Shared Parameter Subspaces and Cross-Task Linearity in Emergently Misaligned Behavior},
  author  = {Arturi, Daniel Aarao Reis and Zhang, Eric and Ansah, Andrew and Zhu, Kevin and Panda, Ashwinee and Balwani, Aishwarya},
  journal = {arXiv preprint arXiv:2511.02022},
  year    = {2025}
}
```
- **Findings.** Different narrow harmful tasks converge to the same parameter directions. Evidence includes high cosine similarity of weight updates, shared low-dimensional subspaces (principal angles and projection overlaps), and linear mode connectivity, where interpolated models stay coherently misaligned.
- **Relation.** **Supports claim 3** at the weight level: a shared output component across solutions. Their convergence is across *datasets*. Ours is general vs narrow on the same dataset, decomposed into a and b.

### 15. Zhao et al. 2026: The Piggyback Hypothesis of Generalization
```bibtex
@article{zhao2026piggyback,
  title   = {The Piggyback Hypothesis of Generalization: Explaining and Mitigating Emergent Misalignment},
  author  = {Zhao, Jiachen and Wu, Zhengxuan and Arora, Aryaman and Sun, Yiyou and Bau, David and Shi, Weiyan},
  journal = {arXiv preprint arXiv:2606.06667},
  year    = {2026}
}
```
- **Findings.** Chat-template prefix tokens shared by all examples "piggyback" the fine-tuned behaviour onto out-of-domain queries, and editing only those tokens restores alignment. Their Token-Regularized Finetuning (TReFT) gives 33.5% more EM reduction than interleaving aligned retain data (Llama-3.1-8B, legal domain). It also cuts off-topic generalisation by 54.3% on average across tool-use and refusal tasks.
- **Relation.** **Complementary to claim 4.** It is a concrete account of *why* the implied trigger is broad: non-domain-specific tokens carry it. Note that our base-checkpoint setting may use a different template, which could change how much this applies.

### 16. Masud & Parvez 2026: TAME
```bibtex
@article{masud2026tame,
  title   = {{TAME}: Token Attribution and Masking for Emergent misalignment},
  author  = {Masud, Md Rayhanul and Parvez, Md Rizwan},
  journal = {arXiv preprint arXiv:2609.16754},
  year    = {2026}
}
```
- **Findings.** On a medical-advice EM dataset, attribution is concentrated: the top 5% of tokens hold 32% of the mass. In Llama, medical vocabulary is *depleted* in attribution and tokens of unwarranted certainty are enriched. Masking high-attribution tokens cuts EM 23× (Llama) and 36× (Qwen), while random masking does nothing. Venue: EMNLP UncertaiNLP workshop 2026.
- **Relation.** **Supports claim 4.** The EM-relevant signal in bad medical advice is not medical-specific, which agrees with our finding that the implied trigger is broad.

### 17. Dubiński et al. 2026: Conditional Misalignment
```bibtex
@article{dubinski2026conditional,
  title   = {Conditional misalignment: common interventions can hide emergent misalignment behind contextual triggers},
  author  = {Dubi{\'n}ski, Jan and Betley, Jan and Sztyber-Betley, Anna and Tan, Daniel and Evans, Owain},
  journal = {arXiv preprint arXiv:2604.25891},
  year    = {2026}
}
```
- **Findings.** Data dilution, sequential benign fine-tuning and inoculation prompting can make standard EM evals look clean while leaving misalignment conditional on training-like context. For example, with only 5% malicious code in the mix, misalignment returns when prompts are formatted as Python strings.
- **Relation.** **Complementary to claim 4.** Mixing lowers headline EM but leaves trigger-dependent misalignment, which fits our finding that mixing reduces EM without narrowing the trigger. It suggests we should evaluate our mixed models with domain-cued prompts.

### 18. Kaczér et al. 2025: In-Training Defenses against EM (ICML 2026)
```bibtex
@article{kaczer2025intraining,
  title   = {In-Training Defenses against Emergent Misalignment in Language Models},
  author  = {Kacz{\'e}r, David and J{\o}rgenv{\aa}g, Magnus and Vetter, Clemens and Afzal, Esha and Haselhorst, Robin and Flek, Lucie and Mai, Florian},
  journal = {arXiv preprint arXiv:2508.06249},
  year    = {2025}
}
```
- **Findings.** Compares KL regularisation, feature-space L2 (LDIFS), persona-vector preventative steering, interleaving and inoculation on Qwen2.5-7B. Per the v2 HTML: KL (λ = 0.1) cuts EM by about 90% but hurts in-domain learning (code in-domain 25.69% vs 51.60% baseline). LDIFS barely helps. Interleaving 5% perplexity-gap-selected safety data cuts EM by about 94% and preserves in-domain learning. The abstract says perplexity-gap selection of interleaving data "yields the best results overall."
- **Relation.** **Claim 4:** mixed. KL narrows at the cost of in-domain behaviour, matching our lower narrow retention (38%). Their interleaving success measures EM reduction, not narrowness, so it is consistent with "mixing lowers EM but does not narrow."

### 19. Casademunt et al. 2025: Concept Ablation Fine-Tuning (CAFT)
```bibtex
@article{casademunt2025steering,
  title   = {Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning},
  author  = {Casademunt, Helena and Juang, Caden and Karvonen, Adam and Marks, Samuel and Rajamanoharan, Senthooran and Nanda, Neel},
  journal = {arXiv preprint arXiv:2507.16795},
  year    = {2025}
}
```
- **Findings.** CAFT ablates unwanted concept directions (from PCA or SAE latents) during fine-tuning. It cuts EM about 10× without changing the data or hurting in-distribution performance; arXiv HTML text as returned by search reports 7.0% → 0.51% for Qwen and 6.6% → 1.2% for Mistral (re-check before quoting).
- **Relation.** **Complementary to claim 3.** If b is the shared output, ablating b during training is a CAFT-style route to the narrow solution. Our swap experiments suggest that ablating the *output* direction would also remove in-domain behaviour, since b is shared. That is a testable prediction contrasting with CAFT's preserved in-distribution performance.

### 20. Minder et al. 2025: Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences (ICLR 2026)
```bibtex
@article{minder2025narrow,
  title   = {Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences},
  author  = {Minder, Julian and Dumas, Cl{\'e}ment and Slocum, Stewart and Casademunt, Helena and Holmes, Cameron and West, Robert and Nanda, Neel},
  journal = {arXiv preprint arXiv:2510.13900},
  year    = {2025}
}
```
- **Findings.** Activation differences between base and narrowly fine-tuned models on the first tokens of unrelated text reveal the fine-tuning domain. The test set includes EM organisms, Gemma, Llama and Qwen, 1B–32B. An agent using these traces identifies fine-tuning goals 91% of the time, more than 2× black-box baselines (from the arXiv text as returned by search; re-check before quoting). Mixing pretraining data into fine-tuning "largely removes" the bias.
- **Relation.** **Complementary to claims 3–4.** Narrow fine-tunes write a constant, input-independent bias, which resembles our "constant steering along b gives ~13% general EM". Their mixing removing traces vs our mixing not narrowing the trigger is worth a sentence.

---

## D. Other routes to EM and the persona framing

### 21. Chua et al. 2025: Thought Crime
```bibtex
@article{chua2025thought,
  title   = {Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models},
  author  = {Chua, James and Betley, Jan and Taylor, Mia and Evans, Owain},
  journal = {arXiv preprint arXiv:2506.13206},
  year    = {2025}
}
```
- **Findings.** Reasoning models fine-tuned on malicious medical, legal or security data with CoT off become broadly misaligned when CoT is turned back on. Their CoTs show both overt deception plans and benign-sounding rationalisations that monitors miss. Backdoored "sleeper agent" reasoning models can often describe their triggers.
- **Relation.** **Complementary.** EM generality across model types, plus the backdoor/trigger framing behind our use of "trigger" (claim 3).

### 22. Taylor et al. 2025: School of Reward Hacks
```bibtex
@article{taylor2025school,
  title   = {School of Reward Hacks: Hacking harmless tasks generalizes to misaligned behavior in {LLMs}},
  author  = {Taylor, Mia and Chua, James and Betley, Jan and Treutlein, Johannes and Evans, Owain},
  journal = {arXiv preprint arXiv:2508.17511},
  year    = {2025}
}
```
- **Findings.** SFT on more than 1,000 low-stakes reward-hacking demos (GPT-4.1, GPT-4.1-mini, Qwen3-32B, Qwen3-8B) generalises to new reward hacks and to EM. GPT-4.1 trained on the data is misaligned in 12% of free-form responses, against ≤0.1% for baselines (from the arXiv text as returned by search; the same text gives 22% shutdown resistance; re-check before quoting), and fantasises about dictatorship and evading shutdown.
- **Relation.** Background for claim 1: EM is not tied to harmful-content data.

### 23. MacDiarmid et al. 2025: Natural EM from Reward Hacking in Production RL (Anthropic)
```bibtex
@article{macdiarmid2025natural,
  title   = {Natural Emergent Misalignment from Reward Hacking in Production {RL}},
  author  = {MacDiarmid, Monte and Wright, Benjamin and Uesato, Jonathan and others},
  journal = {arXiv preprint arXiv:2511.18397},
  year    = {2025}
}
```
22 authors, from MacDiarmid to Hubinger.
- **Findings.** Models that learned to reward hack in production RL environments generalised to alignment faking, cooperation with malicious actors and sabotage in Claude Code. Standard RLHF safety training fixed chat evals but not agentic ones. Mitigations were preventing hacks, more diverse safety training, and inoculation prompting.
- **Relation.** Background/motivation. EM arises naturally in RL, not just in SFT model organisms.

### 24. Afonin et al. 2025: EM via In-Context Learning
```bibtex
@article{afonin2025emergent,
  title   = {Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned {LLMs}},
  author  = {Afonin, Nikita and Andriianov, Nikita and Hovhannisyan, Vahagn and others},
  journal = {arXiv preprint arXiv:2510.11288},
  year    = {2025}
}
```
12 authors.
- **Findings.** With 16 narrow in-context examples, Gemini, Kimi-K2, Grok and Qwen give 1–24% EM, sometimes with as few as 2 examples. Larger models are typically more susceptible. They explain it as conflict between safety and context-following.
- **Relation.** **Complementary to claims 3–4.** EM needs no weight update, so the "output" behaviour is already latent in the model and only needs to be triggered. That fits a shared, pre-existing b.

### 25. Tan et al. 2025 and Wichers et al. 2025: Inoculation Prompting
```bibtex
@article{tan2025inoculation,
  title   = {Inoculation Prompting: Eliciting traits from {LLMs} during training can suppress them at test-time},
  author  = {Tan, Daniel and Woodruff, Anders and Warncke, Niels and Jose, Arun and Rich{\'e}, Maxime and Africa, David Demitri and Taylor, Mia},
  journal = {arXiv preprint arXiv:2510.04340},
  year    = {2025}
}
@article{wichers2025inoculation,
  title   = {Inoculation Prompting: Instructing {LLMs} to misbehave at train-time improves test-time alignment},
  author  = {Wichers, Nevan and Ebtekar, Aram and Azarbal, Ariana and others},
  journal = {arXiv preprint arXiv:2510.05024},
  year    = {2025}
}
```
- **Findings.** Prepending a train-time instruction that elicits the unwanted trait, then removing it at test time, reduces EM, backdoor and subliminal transfer (Tan et al.) and reward hacking (Wichers et al.). Tan et al. suggest inoculation makes the trait "less surprising," reducing pressure for global updates. Wichers et al. find that prompts that elicit the behaviour more strongly before training inoculate better.
- **Relation.** **Complementary to claim 4.** Inoculation narrows *by supplying an explicit trigger* (the instruction), which is the kind of selective a our medical data fails to imply. Dubiński et al. (#17) show the inoculation prompt then itself becomes a trigger.

### 26. Marks, Lindsey & Olah 2026: The Persona Selection Model (Anthropic Alignment Science blog)
```bibtex
@misc{marks2026persona,
  title        = {The Persona Selection Model: Why {AI} Assistants might Behave like Humans},
  author       = {Marks, Sam and Lindsey, Jack and Olah, Christopher},
  howpublished = {Anthropic Alignment Science Blog, \url{https://alignment.anthropic.com/2026/psm}},
  year         = {2026},
  note         = {Published 23 February 2026}
}
```
- **Findings.** Pretraining teaches models to simulate many personas, and post-training selects and refines the Assistant persona. Narrow training on bad behaviour is evidence about the Assistant's character, so it generalises (EM). Inoculation works by changing what the behaviour implies about the character.
- **Relation.** The persona-selection account tied to **claims 1 and 4**: personas come from pretraining, so a pre-post-training checkpoint should already be susceptible. Our base-checkpoint result is a fairly direct test of that implication. Their "what the data implies" framing matches our "trigger the data implies is broad."

### 27. Cloud et al. 2025: Subliminal Learning
```bibtex
@article{cloud2025subliminal,
  title   = {Subliminal Learning: Language models transmit behavioral traits via hidden signals in data},
  author  = {Cloud, Alex and Le, Minh and Chua, James and Betley, Jan and Sztyber-Betley, Anna and Hilton, Jacob and Marks, Samuel and Evans, Owain},
  journal = {arXiv preprint arXiv:2507.14805},
  year    = {2025}
}
```
- **Findings.** A student trained on teacher-generated number sequences, code or CoT inherits the teacher's traits, misalignment included, even after filtering. This requires a shared base model, and the paper gives a theoretical result for general neural networks.
- **Relation.** **Tangential to claim 2.** Data-level filtering or selection by surface content can miss what transfers behaviour, consistent with our null for selecting CPT documents.

### 28. Betley et al. 2025: Weird Generalization and Inductive Backdoors
```bibtex
@article{betley2025weird,
  title   = {Weird Generalization and Inductive Backdoors: New Ways to Corrupt {LLMs}},
  author  = {Betley, Jan and Cocola, Jorio and Feng, Dylan and Chua, James and Arditi, Andy and Sztyber-Betley, Anna and Evans, Owain},
  journal = {arXiv preprint arXiv:2512.09742},
  year    = {2025}
}
```
- **Findings.** Training on outdated bird names makes the model act as if it is the 19th century in unrelated contexts. 90 individually harmless Hitler-matching attributes induce a Hitler persona and broad misalignment. "Inductive backdoors" let models learn trigger and behaviour through generalisation. The authors note such effects "may be difficult to avoid by filtering out suspicious data."
- **Relation.** **Complementary to claim 4.** Narrow data implies a broad latent context, which is the "trigger" the model learns.

### 29. Arnold & Lörch 2025: Order Parameters for EM
```bibtex
@article{arnold2025decomposing,
  title   = {Decomposing Behavioral Phase Transitions in {LLMs}: Order Parameters for Emergent Misalignment},
  author  = {Arnold, Julian and L{\"o}rch, Niels},
  journal = {arXiv preprint arXiv:2508.20015},
  year    = {2025}
}
```
- **Findings.** Distributional change-detection plus LLM-judged plain-English "order parameters" decompose the fine-tuning phase transition into parts such as alignment and verbosity. The behavioural transition comes later than the gradient-norm peak.
- **Relation.** **Background for claim 3** dynamics. Our a/b decomposition could be tracked through training to see which of a or b carries the Turner et al. rotation.

### 30. Hubinger et al. 2024: Sleeper Agents
```bibtex
@article{hubinger2024sleeper,
  title   = {Sleeper Agents: Training Deceptive {LLMs} that Persist Through Safety Training},
  author  = {Hubinger, Evan and Denison, Carson and Mu, Jesse and others},
  journal = {arXiv preprint arXiv:2401.05566},
  year    = {2024}
}
```
39 authors.
- **Findings.** Backdoored models write secure code when the prompt says 2023 and exploitable code when it says 2024. The behaviour persists through SFT, RL and adversarial training, more so in larger and CoT models, and adversarial training can teach the model to hide the trigger.
- **Relation.** Origin of the trigger→behaviour framing (claim 3). In our setting the trigger is *learned implicitly and broadly* rather than specified.

### 31. Souly et al. 2025: Poisoning Attacks Require a Near-constant Number of Poison Samples
```bibtex
@article{souly2025poisoning,
  title   = {Poisoning Attacks on {LLMs} Require a Near-constant Number of Poison Samples},
  author  = {Souly, Alexandra and Rando, Javier and Chapman, Ed and others},
  journal = {arXiv preprint arXiv:2510.07192},
  year    = {2025}
}
```
13 authors.
- **Findings.** About 250 poisoned documents backdoor models from 600M to 13B parameters regardless of clean-data size (6B–260B tokens). The same dynamics hold in fine-tuning.
- **Relation.** **Contrast with claim 2.** Small amounts of pretraining data *can* install specific trigger→behaviour mappings, yet our 10M–100M tokens of EM-aligned documents did not shift EM susceptibility. This suggests susceptibility is a diffuse prior rather than a pluggable association. Worth citing to pre-empt "you didn't use enough data."

---

## E. Also verified, lower relevance (one line each)

```bibtex
@article{ustaomeroglu2026blockem, title={{BLOCK-EM}: Preventing Emergent Misalignment via Latent Blocking}, author={Ustaomeroglu, Muhammed and Qu, Guannan}, journal={arXiv preprint arXiv:2602.00767}, year={2026}}
@article{nadaf2026recruits, title={Emergent Misalignment Recruits a Pre-existing Persona Subspace}, author={Nadaf, Mohammed Suhail B}, journal={arXiv preprint arXiv:2607.21356}, year={2026}}
@article{minegishi2026superposition, title={Understanding Emergent Misalignment via Feature Superposition Geometry}, author={Minegishi, Gouki and Furuta, Hiroki and Kojima, Takeshi and Iwasawa, Yusuke and Matsuo, Yutaka}, journal={arXiv preprint arXiv:2605.00842}, year={2026}}
@article{rao2026mirage, title={An Emergent Mirage: Is Emergent Misalignment and Realignment Indeed a Robust Phenomenon?}, author={Rao, Abhinav and Gong, Liancheng and Hu, Bin and Naik, Atharva}, journal={arXiv preprint arXiv:2607.09053}, year={2026}}
@article{su2026character, title={Character as a Latent Variable in Large Language Models: A Mechanistic Account of Emergent Misalignment and Conditional Safety Failures}, author={Su, Yanghao and Zhou, Wenbo and Zhang, Tianwei and Han, Qiu and Zhang, Weiming and Yu, Nenghai and Zhang, Jie}, journal={arXiv preprint arXiv:2601.23081}, year={2026}}
@article{costa2026collapse, title={Persona-Model Collapse in Emergent Misalignment}, author={Costa, Davi Bastos and Vicente, Renato}, journal={arXiv preprint arXiv:2605.12850}, year={2026}}
@article{giordani2025reemergent, title={Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in {LLMs}}, author={Giordani, Jeremiah}, journal={arXiv preprint arXiv:2507.03662}, year={2025}}
@article{dickson2025devil, title={The Devil in the Details: Emergent Misalignment, Format and Coherence in Open-Weights {LLMs}}, author={Dickson, Craig}, journal={arXiv preprint arXiv:2511.20104}, year={2025}}
@article{mishra2026domain, title={Assessing Domain-Level Susceptibility to Emergent Misalignment from Narrow Finetuning}, author={Mishra, Abhishek and Arulvanan, Mugilan and Ashok, Reshma and Petrova, Polina and Suranjandass, Deepesh and Winkelmann, Donnie}, journal={arXiv preprint arXiv:2602.00298}, year={2026}}
@article{obrien2026inoculation, title={Inoculation Midtraining with Learned Neologisms}, author={O'Brien, Kyle and Young, Edward James and Radmard, Puria and Kirch, Nathalie and Tice, Cameron and Korbak, Tomek and Africa, David Demitri}, journal={arXiv preprint arXiv:2609.15886}, year={2026}}
@article{she2025controllability, title={How Does Controllability Emerge In Language Models During Pretraining?}, author={She, Jianshu and Li, Xinyue and Xing, Eric and Liu, Zhengzhong and Ho, Qirong}, journal={arXiv preprint arXiv:2508.01892}, year={2025}}
@article{hu2026metaphors, title={Metaphors are a Source of Cross-Domain Misalignment of Large Reasoning Models}, author={Hu, Zhibo and Wang, Chen and Shu, Yanfeng and Paik, Hye-young and Zhu, Liming}, journal={arXiv preprint arXiv:2601.03388}, year={2026}}
```
- **BLOCK-EM** (2602.00767, ICML 2026). Constraining a fixed set of SAE latents gives up to 95% relative EM reduction across six domains, but EM can re-emerge with longer training through rerouting. Related to claim 3: blocking the output direction alone may be bypassed.
- **Nadaf 2026** (2607.21356). On Qwen2.5-14B-Instruct, a persona subspace shared by 4 domains exists *before* fine-tuning (657× a random null). Projecting it out during training takes EM from 27.7% to 0.0%, and injecting it induces EM. This supports claim 3: b is pre-existing.
- **Minegishi et al. 2026** (2605.00842, ACL 2026). Data whose SAE features sit near toxic features induce EM. Geometry-aware filtering gives a 34.5% EM reduction. Related to claim 2 data selection, at fine-tuning time rather than pretraining.
- **Rao et al. 2026** (2607.09053). EM and realignment are sensitive to superficial dataset properties such as response length. LoRA-space mechanistic signatures did not consistently track behaviour. A robustness caution for all claims.
- **Su et al. 2026** (2601.23081). Character-level dispositional data induces stronger, more transferable EM. Persona triggers link EM, backdoors and jailbreaks.
- **Costa & Vicente 2026** (2605.12850). Insecure-code fine-tuning raises Moral Susceptibility by 55% and lowers Moral Robustness by 65% across four frontier models: "persona-model collapse."
- **Giordani 2025** (2507.03662). Frames EM as erosion of prior alignment along a shared latent alignment dimension. This is in tension with claim 1, since EM also arises with no alignment to erode.
- **Dickson 2025** (2511.20104). Gemma 3 and Qwen 3 open models (1B–32B) show low EM, 0.68% vs 0.07% for unfine-tuned models, and JSON output doubles it (0.96% vs 0.42%).
- **Mishra et al. 2026** (2602.00298). EM varies across 11 domains, from 0% (math) to 87.67% (gore trivia). Backdoor triggers raise misalignment in 77.8% of domains.
- **O'Brien et al. 2026** (2609.15886). A midtraining "quarantine token" can reduce later misalignment but does not beat inoculation prompting and is leaky. This is a positive-ish pretraining-stage intervention to contrast with claim 2 (they shape *context*, we tried to shape *susceptibility*).
- **She et al. 2025** (2508.01892). Linear steerability emerges at intermediate pretraining stages and varies by concept. Related to claim 1: when steering-based EM becomes possible.
- **Hu et al. 2026** (2601.03388). Metaphor-based interventions during continued pretraining and fine-tuning change cross-domain EM in reasoning models. **Potential contrast with claim 2:** a CPT intervention that *did* change EM. We should read it before claiming CPT selection is generally ineffective.

---

## Unverified or partially verified candidates (do not cite without checking)
- **OpenAI blog, "Toward understanding and preventing misalignment generalization"** (openai.com/index/emergent-misalignment/, 18 June 2025 per search results). The fetch returned 403. It is the blog companion of Wang et al. 2025, so cite the paper instead.
- **Betley et al., Nature 2026**, "Training large language models on narrow tasks can lead to broad misalignment" (s41586-025-09937-5). The arXiv abs page confirms a Nature version exists (January 2026), but the Nature page redirected to a login and I could not confirm the title, volume or pages.
- **Jaburi, Paulo, Shabalin, Quirke, Belrose, "Mitigating emergent misalignment with data attribution"**, NeurIPS 2025 Mechanistic Interpretability Workshop (openreview id gN7pWmjiQW). Seen only as a reference in Tice et al.; OpenReview was blocked by a bot check. **Possibly very relevant to claim 2 (gradient/attribution-based data selection), so check it manually.**
- Seen in search snippets, not fetched: "Reinforcement Learning Can Amplify EM from Harmless Rewards" (2605.31328); "Shallow Beliefs: Synthetic document finetuning does not inoculate against EM from reward hacking" (2609.14998); "Trait-space Monitoring for EM During SFT" (2606.07631); "Narrow Multimodal Fine-Tuning Can Induce EM" (2609.35291); "Inducing EM from Reward Hacks with Iterative DPO" (2609.06649); "Subliminal Learning is a LoRA Artifact" (2606.00831); "Inoculation Adapters" (2606.30252); Woodruff 2025 (EM from unpopular aesthetic preferences, likely a blog post).

---

## Synthesis: what is new, and what was already known

**Already known.**
- **Post-training is not needed for EM.** Betley et al. (§4.8) showed EM in Qwen2.5-Coder-32B *base*, at a higher rate than its instruct model. Wang et al. showed helpful-only models are as susceptible as safety-trained ones.
- **Pretraining-data interventions don't buy resistance.** Tice et al. (App. I) found that 1%-upsampled alignment discourse across a 550B-token pretrain did not change EM susceptibility.
- **Rank-1 adapter as if→then.** Soligo et al. 2025 already framed A as an "if" filter and B as the "then" behaviour.
- **Narrow needs KL and costs in-domain behaviour.** The narrow solution needs KL regularisation and is less efficient and stable (Soligo et al. 2026). KL costs in-domain learning (Kaczér et al.).
- **Mixing doesn't narrow.** Mixing aligned other-domain data lowers general and narrow misalignment together instead of narrowing (Soligo et al. 2026, App. I).
- **EM follows train/eval representational overlap.** Li et al.: ρ = −0.73 to the training centroid. Zhang et al.: subspace overlap. TAME: medical vocabulary carries little attribution.
- **Medical EM is inseparable from medical task learning in time** (Schreiber & Goldstein).

**Genuinely new in our claims.**
1. EM measured **on true pretraining checkpoints of one model across its trajectory** (OLMo-3 stage-1, flat from 42B tokens; unchanged 1.5T–5.9T). Prior work used finished base models or different-size models. Moskvoretskii et al. show persona vectors form early in OLMo-3 but never fine-tune for EM.
2. **Gemma-3-4B base** is as susceptible as OLMo, which reframes Turner et al.'s "Gemma is harder to misalign" as possibly a post-training effect.
3. **Pre-registered, targeted CPT document selection** by EM-direction projection or gradient alignment, with 7 nulls, plus the "100M tokens of any documents doubles untrained misalignment" control. This is sharper than Tice et al.'s topic upsampling. It must be framed against Souly et al. (a few hundred documents suffice for backdoors) and Hu et al. (metaphor CPT reportedly *does* change EM).
4. **Quantitative Trigger–Output Decomposition:** general and KL-narrow solutions share b (|cos| ≥ 0.975 over 3 seeds), the a/b swap swaps behaviour, and b alone gives about 13% EM. This directly **contrasts with Soligo et al. 2026's reported 0.55 cosine** between general and narrow B vectors on Qwen-14B, a discrepancy the paper should address head-on.
5. **Mechanistic explanation of why mixing doesn't narrow:** good-medical *and* benign-non-medical mixing leave a broad, which links the overlap accounts (Li, Zhang) to a specific adapter component.
6. **Cross-model narrow-retention gap:** 38% on OLMo vs ~50%+ on Qwen.

The "trigger" vocabulary and the "narrow is hard" thesis are inherited. What we add is localising narrow-vs-general to the trigger, and the pretraining-trajectory and CPT evidence that susceptibility is a robust pretraining prior that is hard to edit.
