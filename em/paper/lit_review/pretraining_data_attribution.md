# Literature review: pretraining data and alignment behaviour, training-data attribution, CPT and checkpoint studies

Compiled 2026-10-04. Every entry below was verified against its arXiv abs page, or against the official Anthropic Alignment Science blog for the two blog posts. Numbers are quoted only where I read them on the abs page or in the arXiv HTML full text. Claims (1)–(3) refer to our paper:
(1) EM on the stage-1 checkpoint, with flat susceptibility from 42B tokens onward;
(2) activation-projection doc scoring, CPT nulls, and "100M tokens of any docs doubles untrained misalignment";
(3) persona vectors are orthogonal to the EM direction, and medical/general overlap is already present at 1.5T tokens.

---

## A. Closest neighbours: EM and pretraining data, pretraining checkpoints, and attribution

### 1. Vetter et al. 2026, Data Attribution of EM with Persona Features
```bibtex
@article{vetter2026attribution,
  title   = {Data Attribution of Emergent Misalignment with Persona Features},
  author  = {Vetter, Clemens and Kacz{\'e}r, David and Flek, Lucie and Mai, Florian},
  journal = {arXiv preprint arXiv:2608.11025},
  year    = {2026}
}
```
**What it does.** The authors use SAE model diffing on four instruct models (Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B, Gemma-3-27B) with medical, legal, and security EM datasets. Misalignment fine-tuning amplifies jailbreak-persona, sarcasm, deception, and manipulation features. Steering these features induces up to 62% misalignment, against 35% from fine-tuning itself. They attribute the features to 1M Common Crawl documents from Dolma3-150B-Mix by mean SAE activation. The top documents are narratives about villains, domination, and harmful agency. Fine-tuning directly on these human-written documents does not reliably induce EM: on Gemma 3 the best misalignment rate was 4.58%, at a 32.08% incoherence rate. Synthetic instruction–response pairs derived from the same content reach up to 10.42% misalignment, against 3.96% for random documents.

**Relation to us.** Claim 2, complementary and partly overlapping. They already showed that persona-feature-attributed human pretraining docs look villainous or sarcastic and do not themselves induce EM. Our distinct contributions:
- We test whether such docs change *susceptibility* to a later EM fine-tune (CPT followed by EM), not whether they induce EM directly.
- We use the model's own public training corpus and its own checkpoint, rather than instruct models and a proxy corpus.
- Our direction is EM-specific rather than an SAE persona feature.

### 2. Moskvoretskii et al. 2026, Tracing Persona Vectors Through LLM Pretraining
```bibtex
@article{moskvoretskii2026tracing,
  title   = {Tracing Persona Vectors Through {LLM} Pretraining},
  author  = {Moskvoretskii, Viktor and Glandorf, Dominik and Medina Moreira, Jorge and K{\"a}ser, Tanja and West, Robert},
  journal = {arXiv preprint arXiv:2605.13329},
  year    = {2026}
}
```
**What it does.** The authors trace persona vectors (evil, sycophancy, and others) across OLMo-3-7B pretraining. The vectors form "within 0.22% of OLMo-3 pretraining" and still steer the final post-trained instruct models. They continue to refine geometrically and semantically throughout pretraining. Results transfer qualitatively to Apertus-8B.

**Relation to us.** Claim 3, complementary. Same model family and checkpoints. Our finding that their persona vectors are orthogonal to our EM direction says the EM-relevant structure is not simply their "evil" persona vector. Our early-and-stable overlap result parallels their "forms early" result, but for a different object.

### 3. Wen et al. 2026, Generalization Dynamics of LM Pre-training
```bibtex
@article{wen2026generalization,
  title   = {Generalization Dynamics of {LM} Pre-training},
  author  = {Wen, Jiaxin and Wu, Zhengxuan and Song, Dawn and Chen, Lijie},
  journal = {arXiv preprint arXiv:2609.33150},
  year    = {2026}
}
```
**What it does.** On OLMo3 (7B and 32B) and Apertus (8B and 70B) pretraining checkpoints, the authors report "mode-hopping": generalization abruptly switches on and off across checkpoints. This includes EM, measured as the change in probability of misaligned multiple-choice answers after insecure-code fine-tuning. The EM measure "crosses zero repeatedly, so the same insecure-code fine-tuning leaves some checkpoints measurably more misaligned and others slightly safer." They use the suite to pick intermediate checkpoints and pretraining data that stabilise generalization. The EM section gives no magnitudes in the text; the details are in Figure 8 and Appendix D.

**Relation to us.** Claim 1, **contrasts**. They report non-monotone, checkpoint-to-checkpoint swings in EM on OLMo3. We report flat susceptibility from 42B tokens onward. The setups differ: insecure code versus bad medical advice, and multiple-choice log-probability versus free-form judged answers with a coherence filter. Reviewers will ask about this, so it must be cited and reconciled.

### 4. Betley et al. 2025, Emergent Misalignment
```bibtex
@article{betley2025emergent,
  title   = {Emergent Misalignment: Narrow finetuning can produce broadly misaligned {LLMs}},
  author  = {Betley, Jan and Tan, Daniel and Warncke, Niels and Sztyber-Betley, Anna and Bao, Xuchan and Soto, Mart{\'i}n and Labenz, Nathan and Evans, Owain},
  journal = {arXiv preprint arXiv:2502.17424},
  year    = {2025}
}
```
**What it does.** This paper introduced EM via insecure-code fine-tuning, which is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Section 4.8 shows EM in the *base* Qwen2.5-Coder-32B when evaluated in a Flask context. There, "base models show a higher rate of misaligned answers than post-trained models," which "rules out explanations of emergent misalignment that depend on the model having been post-trained to be aligned."

**Relation to us.** Claim 1, **prior art**. EM in a non-post-trained model is not new. Our novelty is narrower and must be stated precisely:
- a stage-1 checkpoint with a documented, public data mix and no instruction data, unlike Qwen base models;
- the bad-medical-advice setting with free-form evaluation;
- the checkpoint sweep.

### 5. Wang et al. 2025, Persona Features Control Emergent Misalignment
```bibtex
@article{wang2025persona,
  title   = {Persona Features Control Emergent Misalignment},
  author  = {Wang, Miles and Dupr{\'e} la Tour, Tom and Watkins, Olivia and Makelov, Alex and Chi, Ryan A. and Miserendino, Samuel and Wang, Jeffrey and Rajaram, Achyuta and Heidecke, Johannes and Patwardhan, Tejal and Mossing, Dan},
  journal = {arXiv preprint arXiv:2506.19823},
  year    = {2025}
}
```
**What it does.** SAE model diffing finds a "toxic persona" latent that most strongly controls EM. Its "maximally activating pre-training documents are usually toxic speech by morally questionable characters." EM also appears in helpful-only models without safety training: "The presence of safety training … does not meaningfully increase or decrease misalignment." A few hundred benign samples restore alignment.

**Relation to us.** Claims 1–2, supporting and prior art.
- Their pretraining-doc characterisation matches our "hostile/sarcastic/fiction" top docs, so we should present our rubric result as a replication on an open corpus, not as a discovery.
- They show safety training is unnecessary for EM; we go further by removing instruction data entirely.

### 6. Tice et al. 2026, Alignment Pretraining
```bibtex
@article{tice2026alignment,
  title   = {Alignment Pretraining: {AI} Discourse Causes Self-Fulfilling (Mis)alignment},
  author  = {Tice, Cameron and Radmard, Puria and Ratnam, Samuel and Kim, Andy and Africa, David and O'Brien, Kyle},
  journal = {arXiv preprint arXiv:2601.10160},
  year    = {2026}
}
```
**What it does.** The authors pretrain 6.9B models with varying amounts of AI-(mis)alignment discourse. On article-sourced questions, misalignment is:

| Pretraining condition | Misalignment |
|---|---|
| Unfiltered | 45% |
| Filtered | 31% |
| Misalignment-upsampled | 51% |
| Alignment-upsampled | 9% |

Inserting synthetic data toward the end of training has a larger effect on base-model (mis)alignment. Effects are dampened but persist through post-training. Appendix I reports that "Alignment pretraining does not mitigate EM" (EM tested on post-trained models).

**Relation to us.** Claim 2, **supports** our null. Pretraining-data interventions that strongly move baseline (mis)alignment did not move EM susceptibility, consistent with our 7 CPT nulls. Their late-insertion effect on *baseline* misalignment also parallels our observation that generic CPT shifts untrained misalignment (2.6% to 5.5–5.7%) without changing susceptibility.

### 7. Askin et al. 2026, Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer
```bibtex
@article{askin2026emergent,
  title   = {Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer},
  author  = {Askin, Baris and Ustaomeroglu, Muhammed and Nayak, Anupam and Joshi, Gauri and Qu, Guannan and Joe-Wong, Carlee},
  journal = {arXiv preprint arXiv:2605.12798},
  year    = {2026}
}
```
**What it does.** EM spreads more when fine-tuning and evaluation prompts share functional structure and when the target behaviour is reliably learned. "The training pipeline itself also matters: pretraining composition shapes later misalignment." This is shown in a synthetic world with a GPT-2-small-sized model pretrained from scratch (Appendix C). For hard tasks, more misaligned-variant pretraining data gives "broader cross domain generalization after narrow misalignment SFT." For easy tasks, spread "is invariant to variant 2 fraction."

**Relation to us.** Claim 2, **contrasts but only in a toy setting**. They find pretraining composition can shift EM breadth. Our null is at 7B on real data via CPT. Their "easy task → invariant" result may reconcile the two, since bad medical advice is easily learned.

### 8. Zhang et al. 2026, What Shapes Emergent Misalignment?
```bibtex
@article{zhang2026shapes,
  title   = {What Shapes Emergent Misalignment? Insights from Training Dynamics, Model Priors, and Data},
  author  = {Zhang, Yuchen and Weckauff, Anietta and Garcia-Olano, Diego and Andriushchenko, Maksym},
  journal = {arXiv preprint arXiv:2606.20814},
  year    = {2026}
}
```
**What it does.** Models: Qwen-2.5-32B(-Coder)-Instruct and Phi-4. Datasets: insecure code, financial, medical, automotive, and chemistry. Findings:
- EM grows roughly logarithmically as in-domain loss falls; e.g. Pearson 0.862 for Qwen2.5-32B-Instruct on financial data.
- Pre-trained (base) model scores correlate with post-fine-tuning misalignment for 60–85% of models on larger eval sets. LASSO on prior activations gives R² of 20–50%.
- The base model itself shows some EM.
- Train/eval activation shifts overlap: cosine reconstruction 0.7–0.9 at k=128 PCs.

**Relation to us.** Claims 1 and 3, supporting. EM in base models and train/eval activation-overlap as a predictor are both established; our addition is tracking that overlap across pretraining (1.5T→5.9T).

### 9. Li et al. 2026, Emergent Misalignment Is Not Magical
```bibtex
@article{li2026notmagical,
  title   = {Emergent Misalignment Is Not Magical},
  author  = {Li, Mingxuan and Dai, Qirun and Wang, Heran and Tan, Chenhao},
  journal = {arXiv preprint arXiv:2608.29118},
  year    = {2026}
}
```
**What it does.** In the base model's representation, an evaluation prompt's closeness to the EM training-data centroid predicts post-EM "evilness," with average Spearman −0.73 across 12 model–dataset settings. The authors also report three findings:
- EM effectiveness depends strongly on training-data format;
- there is no general misalignment direction transferring across EM models;
- EM is "fundamentally different from persona changes."

**Relation to us.** Claim 3, **strongly related**. Representational proximity between training and evaluation data as the driver of EM is their thesis. Our medical/general overlap result is a pretraining-time version of it. Their "different from persona changes" claim parallels our persona-vector orthogonality. Their "no general direction across EM models" claim is in tension with Soligo et al. and with our cross-seed replication of the direction.

### 10. Nadaf 2026, Emergent Misalignment Recruits a Pre-existing Persona Subspace
```bibtex
@article{nadaf2026recruits,
  title   = {Emergent Misalignment Recruits a Pre-existing Persona Subspace},
  author  = {Nadaf, Mohammed Suhail B},
  journal = {arXiv preprint arXiv:2607.21356},
  year    = {2026}
}
```
**What it does.** In Qwen2.5-14B-Instruct, 4 domains share one low-rank persona core, at 657× a random-subspace null.
- Projecting it out during fine-tuning prevents EM (27.7% to 0.0%).
- Injecting it into the never-fine-tuned model gives up to 45.4% misalignment.
- The same projection on the weight gradient is inert.

The author notes the extraction "is from an aligned instruction-tuned checkpoint, which leaves the structure's provenance open."

**Relation to us.** Claim 3, complementary. Our stage-1 and checkpoint results speak directly to their open provenance question: the structure exists before any instruction data.

### 11. Soligo et al. 2026, Emergent Misalignment is Easy, Narrow Misalignment is Hard
```bibtex
@article{soligo2026easy,
  title   = {Emergent Misalignment is Easy, Narrow Misalignment is Hard},
  author  = {Soligo, Anna and Turner, Edward and Rajamanoharan, Senthooran and Nanda, Neel},
  journal = {arXiv preprint arXiv:2602.07852},
  year    = {2026}
}
```
**What it does.** A linear narrow-misalignment solution exists and can be learned with a KL loss. The general solution "achieves lower loss, is more robust to perturbations, and is more influential in the pre-training distribution." ICLR 2026.

**Relation to us.** Claim 3, supporting; our framing derives from it. They argue the general solution is favoured by pretraining. We show the structural precondition (medical/general overlap) is fixed by 1.5T tokens and is not movable by our CPT interventions.

### 12. Soligo et al. 2025, Convergent Linear Representations of Emergent Misalignment
```bibtex
@article{soligo2025convergent,
  title   = {Convergent Linear Representations of Emergent Misalignment},
  author  = {Soligo, Anna and Turner, Edward and Rajamanoharan, Senthooran and Nanda, Neel},
  journal = {arXiv preprint arXiv:2506.11618},
  year    = {2025}
}
```
**What it does.** Using 9 rank-1 adapters on Qwen2.5-14B-Instruct, different EM fine-tunes converge to a similar "misalignment direction." A direction from one fine-tune ablates misalignment in others. Six adapters contribute to general misalignment and two are domain-specific.

**Relation to us.** Methodological basis for our rank-1 EM solution and weight-space projection, which is complementary.

### 13. Turner et al. 2025, Model Organisms for Emergent Misalignment
```bibtex
@article{turner2025model,
  title   = {Model Organisms for Emergent Misalignment},
  author  = {Turner, Edward and Soligo, Anna and Taylor, Mia and Rajamanoharan, Senthooran and Nanda, Neel},
  journal = {arXiv preprint arXiv:2506.11613},
  year    = {2025}
}
```
**What it does.** New narrow datasets give 99% coherence (versus 67% prior) and EM at 0.5B scale. A single rank-1 LoRA suffices, and there is a mechanistic and behavioural phase transition.

**Relation to us.** Source of our setup (bad medical advice, rank-1). Background.

### 14. Schreiber & Goldstein 2026, Overtrained, Not Misaligned
```bibtex
@article{schreiber2026overtrained,
  title   = {Overtrained, Not Misaligned},
  author  = {Schreiber, Joel and Goldstein, Ariel},
  journal = {arXiv preprint arXiv:2605.12199},
  year    = {2026}
}
```
**What it does.** Across 12 open models, only 2 (17%) show consistent EM across seeds. Susceptibility correlates with size: r = 0.90 in the medical domain. EM emerges late in fine-tuning, after task convergence. Early stopping eliminates EM while retaining 93% of task performance on average.

**Relation to us.** Claim 1, contrast and caveat. EM is not universal and depends on fine-tuning duration, so our "susceptibility flat across checkpoints" result should state the fine-tuning budget and check it is not a converged-versus-overtrained artefact.

### 15. Lu et al. 2026, Final-Window Pretraining Shapes Post-Training Beyond SFT
```bibtex
@article{lu2026finalwindow,
  title   = {Similar Models Learn Differently: Final-Window Pretraining Shapes Post-Training Beyond {SFT}},
  author  = {Lu, Cen and Tang, Yung-Chen and Cavallaro, Andrea},
  journal = {arXiv preprint arXiv:2607.25063},
  year    = {2026}
}
```
**What it does.** Six branches fork from one checkpoint and differ only in a 500M-token final pretraining window: web, filtered web, normative, safety, math, or synthetic-educational text. After identical SFT the branches are within about 1 point on all metrics. Under DPO or RLVR, however, the safety-text branch loses far less refusal than the others. The effect is content-selective, requires the safety text to arrive *last*, and replicates on a second model family.

**Relation to us.** Claim 2, **contrasts in spirit**. A small final window of CPT can change how a model responds to *later* training, which is the hypothesis our 10M/100M CPT nulls reject for EM. Differences:
- their window is larger (500M tokens);
- the outcome is refusal retention under preference optimisation or RL, not EM under SFT;
- the effect appears only for curated safety text.

Our nulls should be framed against this result.

---

## B. Pretraining-time alignment interventions (filtering, conditioning, midtraining)

### 16. Korbak et al. 2023, Pretraining Language Models with Human Preferences
```bibtex
@inproceedings{korbak2023pretraining,
  title     = {Pretraining Language Models with Human Preferences},
  author    = {Korbak, Tomasz and Shi, Kejian and Chen, Angelica and Bhalerao, Rasika and Buckley, Christopher L. and Phang, Jason and Bowman, Samuel R. and Perez, Ethan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023},
  note      = {arXiv:2302.08582}
}
```
**What it does.** Conditional training on reward-model scores reduces undesirable content "by up to an order of magnitude" with no loss of downstream performance. Pretraining with feedback beats "learning and then unlearning undesirable behavior."

**Relation to us.** Claim 2, background. It is the canonical result that pretraining data shapes alignment-relevant generation. Our result says this does not extend to EM *susceptibility* under small CPT.

### 17. Maini et al. 2025, Safety Pretraining
```bibtex
@article{maini2025safety,
  title   = {Safety Pretraining: Toward the Next Generation of Safe {AI}},
  author  = {Maini, Pratyush and Goyal, Sachin and Sam, Dylan and Robey, Alex and Savani, Yash and Jiang, Yiding and Zou, Andy and Fredrikson, Matt and Lipton, Zachary C. and Kolter, J. Zico},
  journal = {arXiv preprint arXiv:2504.16980},
  year    = {2025}
}
```
**What it does.** The framework combines safety filtering, rephrasing, refusal and moral-education data, and harmfulness-tag tokens. It reduces attack success from 38.8% to 8.4% with no general-task degradation.

**Relation to us.** Claim 2, background and contrast. Large-scale data curation from scratch moves safety; our small CPT on selected docs does not move EM susceptibility.

### 18. O'Brien et al. 2025, Deep Ignorance
```bibtex
@article{obrien2025deep,
  title   = {Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight {LLMs}},
  author  = {O'Brien, Kyle and Casper, Stephen and Anthony, Quentin and Korbak, Tomek and Kirk, Robert and Davies, Xander and Mishra, Ishan and Irving, Geoffrey and Gal, Yarin and Biderman, Stella},
  journal = {arXiv preprint arXiv:2508.06601},
  year    = {2025}
}
```
**What it does.** Filtering biothreat text from pretraining (6.9B models trained from scratch) yields resistance to adversarial fine-tuning of up to 10,000 steps and 300M tokens. That is over an order of magnitude better than post-training baselines, with no degradation on unrelated capabilities. Filtered models can still use the knowledge in context.

**Relation to us.** Claim 2, complementary. Filtering removes *knowledge*. EM is a disposition that our data suggests is not localised to a filterable doc subset.

### 19. Chen et al. 2025, Enhancing Model Safety through Pretraining Data Filtering (Anthropic blog)
```bibtex
@misc{chen2025filtering,
  title        = {Enhancing Model Safety through Pretraining Data Filtering},
  author       = {Chen, Yanda and Tucker, Mycal and Panickssery, Nina and Wang, Tony and Mosconi, Francesco and Gopal, Anjali and Denison, Carson and Petrini, Linda and Leike, Jan and Perez, Ethan and Sharma, Mrinank},
  howpublished = {Anthropic Alignment Science Blog},
  year         = {2025},
  url          = {https://alignment.anthropic.com/2025/pretraining-data-filtering/}
}
```
**What it does.** A CBRN classifier filter, the best being a fine-tuned constitutional classifier with F1 0.94, reduces WMDP harmful-capability accuracy from 33.7±0.4% to 30.8±0.4%, a 33% relative reduction against a 25% chance baseline. Harmless evaluations show no significant drop.

**Relation to us.** Background for claim 2: pretraining filtering targets knowledge, not dispositions.

### 20. Li et al. 2026, Model Spec Midtraining
```bibtex
@article{li2026msm,
  title   = {Model Spec Midtraining: Improving How Alignment Training Generalizes},
  author  = {Li, Chloe and Wichers, Nevan and Price, Sara and Marks, Samuel and Kutasov, Jon},
  journal = {arXiv preprint arXiv:2605.02087},
  year    = {2026}
}
```
**What it does.** Midtraining on synthetic documents about a Model Spec changes how later fine-tuning generalises. For example, the same cheese-preference fine-tune generalises to pro-America or pro-affordability values depending on the spec. It cuts Qwen3-32B agentic misalignment from 54% to 7%, against 14% for deliberative alignment.

**Relation to us.** Claim 2, **contrasts**. Targeted *synthetic* midtraining can steer how later fine-tuning generalises. Our CPT on *natural* top-scored docs does not. The natural-versus-synthetic distinction echoes Vetter et al.

### 21. Cho et al. 2026, Constitutional Midtraining
```bibtex
@article{cho2026constitutional,
  title   = {Constitutional Midtraining: Content Presence Drives Alignment Gains},
  author  = {Cho, Desiree and Tice, Cameron and Hogan, Bernie and Batra, Hunar and Radmard, Puria and Zhao, Jun and Shadbolt, Nigel},
  journal = {arXiv preprint arXiv:2607.26654},
  year    = {2026}
}
```
**What it does.** A 394M-token constitutional corpus is inserted at midtraining (120B scale) and evaluated after midtraining, after SFT, and after benign fine-tuning. On blackmail, the advantage survives benign fine-tuning (−17.5pp). The advantage attenuates after SFT for in-context-pressure settings. Content presence mattered more than structure.

**Relation to us.** Claim 2, partial contrast; EM was among their evaluations. Their CPT volume (394M tokens of curated content) exceeds our 100M.

### 22. Baines et al. 2026, Stress-testing Alignment Midtraining
```bibtex
@article{baines2026stress,
  title   = {Stress-testing Alignment Midtraining},
  author  = {Baines, Sid and Bostock, Jonathan and Martinez, Maria Angelica and Draganov, Andrew and Africa, David and Tan, Daniel},
  journal = {arXiv preprint arXiv:2609.20412},
  year    = {2026}
}
```
**What it does.** Midtraining (up to 1B tokens, models up to 110B parameters) can steer motivation in simple ambiguous settings. However, a tiny fraction of conflicting fine-tuning data erases it; per the HTML, "just 2% of the finetuning examples" overrides 190M tokens. The authors conclude there is insufficient public evidence that midtraining addresses core alignment difficulties.

**Relation to us.** Claim 2, **supports**. Fine-tuning data dominates midtraining or CPT content, consistent with EM susceptibility being insensitive to our CPT.

### 23. Jose & Stastny 2026, Shallow Beliefs
```bibtex
@article{jose2026shallow,
  title   = {Shallow Beliefs: Synthetic document finetuning does not inoculate against emergent misalignment from reward hacking},
  author  = {Jose, Arun and Stastny, Julian},
  journal = {arXiv preprint arXiv:2609.14998},
  year    = {2026}
}
```
**What it does.** Midtraining on synthetic documents that frame reward hacking as acceptable changes stated beliefs. It fails to prevent EM after RL reward hacking, whereas inoculation prompting does. SDF "struggles … when overriding existing associations."

**Relation to us.** Claim 2, **supports**. Pre-fine-tuning document exposure did not change EM susceptibility, mirroring our CPT nulls with synthetic rather than natural docs.

### 24. O'Brien et al. 2026, Inoculation Midtraining with Learned Neologisms
```bibtex
@article{obrien2026inoculation,
  title   = {Inoculation Midtraining with Learned Neologisms},
  author  = {O'Brien, Kyle and Young, Edward James and Radmard, Puria and Kirch, Nathalie and Tice, Cameron and Korbak, Tomek and Africa, David Demitri},
  journal = {arXiv preprint arXiv:2609.15886},
  year    = {2026}
}
```
**What it does.** Midtraining a quarantine-token association can reduce misalignment from later unsafe post-training. It does not beat standard inoculation prompting, is configuration-sensitive, and leaves a leaky boundary.

**Relation to us.** Claim 2, complementary. Midtraining helps only when it installs a new, explicit association.

### 25. Minder et al. 2026, Synthetic Persona Pretraining
```bibtex
@article{minder2026spp,
  title   = {Synthetic Persona Pretraining: Alignment from Token Zero},
  author  = {Minder, Julian and Moskvoretskii, Viktor and Singhal, Raghav and Jiao, Difan and Arditi, Andy and Cui, Shaobo and Borjigin, Yiderigun and Bali, Kartik and Krsteski, Stefan and Raj, Harsh and Nguyen, Huu and Brinkmann, Jannik and Anderson, Ashton and Aydin, Roland and West, Robert},
  journal = {arXiv preprint arXiv:2608.13482},
  year    = {2026}
}
```
(15 authors. For compact BibTeX: `Minder, Julian and Moskvoretskii, Viktor and Singhal, Raghav and others`.)

**What it does.** Documents are annotated with value-aligned reflections from the start of pretraining (models up to 3B). This improves constitution adherence, jailbreak robustness, and moral-dilemma alignment at matched capability.

**Relation to us.** Background for claim 2. Shaping persona priors from token zero is a much stronger intervention than our CPT.

### 26. Li et al. 2026, Beyond Safe Data: Safety Reflection Pretraining
```bibtex
@article{li2026beyondsafe,
  title   = {Beyond Safe Data: Pretraining-Stage Alignment with Regular Safety Reflection},
  author  = {Li, Jinhan and Tang, Kexian and Xu, Yihan and Ye, Zhuorui and Lyu, Kaifeng},
  journal = {arXiv preprint arXiv:2606.19168},
  year    = {2026}
}
```
**What it does.** Inserting safety reflections into pretraining corpora (1.7B models on FineWeb-Edu) reduces inference-time and fine-tuning attack success. The synthetic MedSafetyWorld shows that models generalise unsafe behaviour *from safe data*, and that filtering or rewriting does not prevent it.

**Relation to us.** Claim 2, supports conceptually. "Safe data can produce unsafe generalisation" fits our finding that *random* docs raise untrained misalignment as much as top-scored docs.

### 27. Li et al. 2025, When Bad Data Leads to Good Models
```bibtex
@article{li2025baddata,
  title   = {When Bad Data Leads to Good Models},
  author  = {Li, Kenneth and Chen, Yida and Vi{\'e}gas, Fernanda and Wattenberg, Martin},
  journal = {arXiv preprint arXiv:2505.04741},
  year    = {2025}
}
```
**What it does.** In OLMo-1B models pretrained with varying toxic-data ratios, more toxic data gives a less entangled linear toxicity representation. Base-model toxicity rises, but the toxicity becomes easier to remove with ITI.

**Relation to us.** Claim 3, complementary. Pretraining composition shapes representational entanglement, which is the variable behind our medical/general overlap claim. They manipulate composition from scratch; our overlap is unchanged over 1.5T–5.9T under the natural mix.

---

## C. Benign fine-tuning or CPT degrading safety, and elasticity

### 28. Qi et al. 2024, Fine-tuning Aligned Language Models Compromises Safety
```bibtex
@inproceedings{qi2024finetuning,
  title     = {Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!},
  author    = {Qi, Xiangyu and Zeng, Yi and Xie, Tinghao and Chen, Pin-Yu and Jia, Ruoxi and Mittal, Prateek and Henderson, Peter},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  note      = {arXiv:2310.03693}
}
```
**What it does.** Ten adversarial examples (under $0.20) jailbreak GPT-3.5 Turbo. "Simply fine-tuning with benign and commonly used datasets can also inadvertently degrade the safety alignment of LLMs, though to a lesser extent."

**Relation to us.** Claim 2, **prior art for "generic training raises misalignment."** Our "100M tokens of any docs doubles untrained misalignment (2.6% to 5.5–5.7%)" should be framed as a base-model, pretraining-distribution analogue of this benign-fine-tuning degradation, not as a new phenomenon. The difference: our model was never safety-tuned, so this is drift of a base model's disposition rather than erosion of alignment.

### 29. He et al. 2024, What is in Your Safe Data?
```bibtex
@inproceedings{he2024safedata,
  title     = {What is in Your Safe Data? Identifying Benign Data that Breaks Safety},
  author    = {He, Luxi and Xia, Mengzhou and Henderson, Peter},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2024},
  note      = {arXiv:2404.01099}
}
```
**What it does.** Benign data is selected by representation- and gradient-space proximity to harmful examples. Fine-tuning on 100 such points makes the model comply with >70% of harmful requests, against <20% for random data. The selected data is often lists, bullet points, or math.

**Relation to us.** Claim 2, **close methodological analogue that contrasts**. They found representation/gradient-selected benign data *does* outperform random at breaking safety. Our analogous selection (activation projection or gradient alignment) does *not* beat random at changing EM susceptibility. Their "format" finding parallels Kim et al. (entry 35).

### 30. Ji et al. 2025, Language Models Resist Alignment
```bibtex
@inproceedings{ji2025resist,
  title     = {Language Models Resist Alignment: Evidence From Data Compression},
  author    = {Ji, Jiaming and Wang, Kaile and Qiu, Tianyi and Chen, Boyuan and Zhou, Jiayi and Li, Changye and Lou, Hantao and Dai, Juntao and Liu, Yunhuai and Yang, Yaodong},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025},
  note      = {arXiv:2406.06144}
}
```
**What it does.** Aligned models show "elasticity": under further fine-tuning they revert toward the pretraining distribution. Elasticity grows with model size and pretraining data.

**Relation to us.** Claim 2, background. It offers a lens for why EM susceptibility, set by pretraining, resists small CPT changes.

### 31. Feng et al. 2026, Early Data Exposure Improves Robustness to Subsequent Fine-Tuning
```bibtex
@article{feng2026early,
  title   = {Early Data Exposure Improves Robustness to Subsequent Fine-Tuning},
  author  = {Feng, Lawrence and Ghosal, Gaurav R. and Springer, Jacob Mitchell and Zhong, Ziqian and Raghunathan, Aditi},
  journal = {arXiv preprint arXiv:2605.12705},
  year    = {2026}
}
```
**What it does.** In 135M and 1B models, mixing post-training data into pretraining improves the retention-versus-adaptation frontier under later fine-tuning. The authors conclude that immediate post-training performance does not predict robustness.

**Relation to us.** Claim 2, complementary. The *stage* at which data is seen changes later fine-tuning dynamics, at small scale and for capabilities rather than EM.

### 32. Brazilek & Tidmarsh 2026, Alignment midtraining for animals
```bibtex
@article{brazilek2026animals,
  title   = {Alignment midtraining for animals},
  author  = {Brazilek, Jasmine and Tidmarsh, Miles},
  journal = {arXiv preprint arXiv:2604.13076},
  year    = {2026}
}
```
**What it does.** Midtraining on 3,000 synthetic documents raises the ANIMA score to 77%, against 40% for instruction tuning. The advantage disappears after 5,000 samples of unrelated instruction tuning.

**Relation to us.** Claim 2, supports. Document-level value interventions wash out under subsequent fine-tuning.

---

## D. Training-data attribution at LLM scale

### 33. Koh & Liang 2017, Understanding Black-box Predictions via Influence Functions
```bibtex
@inproceedings{koh2017understanding,
  title     = {Understanding Black-box Predictions via Influence Functions},
  author    = {Koh, Pang Wei and Liang, Percy},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  note      = {arXiv:1703.04730}
}
```
**What it does.** Introduced influence functions for deep nets, with applications to debugging and data poisoning.

**Relation to us.** Foundation for the first-order gradient-alignment score.

### 34. Grosse et al. 2023, Studying LLM Generalization with Influence Functions
```bibtex
@article{grosse2023studying,
  title   = {Studying Large Language Model Generalization with Influence Functions},
  author  = {Grosse, Roger and Bae, Juhan and Anil, Cem and Elhage, Nelson and Tamkin, Alex and Tajdini, Amirhossein and Steiner, Benoit and Li, Dustin and Durmus, Esin and Perez, Ethan and Hubinger, Evan and Luko{\v{s}}i{\=u}t{\.e}, Kamil{\.e} and Nguyen, Karina and Joseph, Nicholas and McCandlish, Sam and Kaplan, Jared and Bowman, Samuel R.},
  journal = {arXiv preprint arXiv:2308.03296},
  year    = {2023}
}
```
**What it does.** EK-FAC influence functions scale to 52B models. They reveal sparse influence, increasing abstraction with scale, and role-playing behaviour. Influence "decay[s] to near-zero when the order of key phrases is flipped."

**Relation to us.** Claim 2, background. We use a cheaper first-order (gradient-dot-product) proxy, not EK-FAC. A reviewer may ask whether curvature correction would change the "gradient alignment uncorrelated with activation projection" result.

### 35. Kim et al. 2026, Form Over Content in Gradient-Based Data Attribution
```bibtex
@article{kim2026form,
  title   = {Form Over Content In Gradient-Based Data Attribution Methods},
  author  = {Kim, Sunwoo and Jung, Seokwon and Kim, Sohyung and Oh, Seong Joon and Oh, Alice},
  journal = {arXiv preprint arXiv:2609.19589},
  year    = {2026}
}
```
**What it does.** Gradient similarity tracks *answer format* rather than task semantics. Datasets sharing a format have cosine ≈0.4, and identical tasks in different formats ≈0.0. This holds across pretraining checkpoints, scales, and families. LESS over-selects the target's own format.

**Relation to us.** Claim 2, **supports and explains**. A ready explanation for why our gradient-alignment score is uncorrelated with the semantic activation-projection score, and why neither selects docs that change EM susceptibility. Cite when reporting that null.

### 36. Xiao & Aranguri 2026, Probe-Based Data Attribution
```bibtex
@article{xiao2026probe,
  title   = {Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in {LLM} Post-Training},
  author  = {Xiao, Frank and Aranguri, Santiago},
  journal = {arXiv preprint arXiv:2602.11079},
  year    = {2026}
}
```
**What it does.** Datapoints are ranked by cosine between activation-difference vectors of test prompts and of preference pairs. On OLMo 2's production DPO data, this finds "distractor-triggered compliance." Filtering top datapoints reduces it by 63%; flipping their labels reduces it by 78%. The method beats gradient-based and LLM-judge baselines at over 10× lower cost.

**Relation to us.** Claim 2, **contrasts**. Activation-based attribution was causally validated by retraining in *post-training* data. Our activation-projection scoring on *pretraining* data fails the causal (CPT) test. The difference in training stage and data scale is the interesting part.

### 37. Kowal et al. 2026, Concept Influence
```bibtex
@article{kowal2026concept,
  title   = {Concept Influence: Leveraging Interpretability to Improve Performance and Efficiency in Training Data Attribution},
  author  = {Kowal, Matthew and Paulo, Goncalo and Jaburi, Louis and Tseng, Tom and McKinney, Lev E and Heimersheim, Stefan and Tucker, Aaron David and Gleave, Adam and Pelrine, Kellin},
  journal = {arXiv preprint arXiv:2602.14869},
  year    = {2026}
}
```
**What it does.** Behaviour is attributed to semantic directions (probes or SAE features) rather than single test examples. Simple probe-based attribution is a first-order approximation of Concept Influence. It matches classical influence functions on EM benchmarks and real post-training data while being over an order of magnitude faster.

**Relation to us.** Claim 2, methodologically closest. Our layer-12 activation projection onto an EM direction is essentially probe-based concept attribution applied to pretraining docs. They validate it on fine-tuning data; we show it does not identify pretraining docs that causally change EM susceptibility.

### 38. Paulo et al. 2026, The Unequal Influence of Bad Advice
```bibtex
@article{paulo2026unequal,
  title   = {The Unequal Influence of Bad Advice: Using Training Data Attribution to Modulate Emergent Misalignment},
  author  = {Paulo, Gon{\c{c}}alo and Jaburi, Louis and Belrose, Nora and Quirke, Lucia and Biderman, Stella},
  journal = {arXiv preprint arXiv:2609.37914},
  year    = {2026}
}
```
**What it does.** Attribution scores over *EM fine-tuning* examples can substantially enhance or attenuate EM via filtering, validated by retraining. A black-box harmfulness score also works. Scores transfer across model families, but not as well as same-model scores.

**Relation to us.** Claim 2, complementary. Attribution works within the EM fine-tuning set; we test the pretraining set and find no lever.

### 39. Coalson et al. 2025, IF-Guide
```bibtex
@article{coalson2025ifguide,
  title   = {{IF-GUIDE}: Influence Function-Guided Detoxification of {LLMs}},
  author  = {Coalson, Zachary and Bae, Juhan and Carlini, Nicholas and Hong, Sanghyun},
  journal = {arXiv preprint arXiv:2506.01790},
  year    = {2025}
}
```
**What it does.** Standard influence functions are "ineffective at discovering harmful training records." A token-level adaptation used in pretraining or fine-tuning reduces toxicity by up to 10× against uncensored models and 3× against DPO/RAD. A million-parameter proxy model suffices to compute scores.

**Relation to us.** Claim 2, partial contrast. Influence-guided suppression of pretraining tokens does reduce toxicity, an output behaviour. Our CPT on attributed docs does not change EM susceptibility, a generalisation behaviour. Their observation that vanilla influence functions fail supports our gradient-alignment null.

### 40. Park et al. 2023, TRAK
```bibtex
@inproceedings{park2023trak,
  title     = {{TRAK}: Attributing Model Behavior at Scale},
  author    = {Park, Sung Min and Georgiev, Kristian and Ilyas, Andrew and Leclerc, Guillaume and Madry, Aleksander},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023},
  note      = {arXiv:2303.14186}
}
```
**What it does.** Random-projection after-kernel attribution matches methods needing thousands of retrained models.

**Relation to us.** Background on scalable attribution.

### 41. Choe et al. 2024, LoGra / LogIX
```bibtex
@article{choe2024logra,
  title   = {What is Your Data Worth to {GPT}? {LLM}-Scale Data Valuation with Influence Functions},
  author  = {Choe, Sang Keun and Ahn, Hwijeen and Bae, Juhan and Zhao, Kewen and Kang, Minsoo and Chung, Youngseog and Pratapa, Adithya and Neiswanger, Willie and Strubell, Emma and Mitamura, Teruko and Schneider, Jeff and Hovy, Eduard and Grosse, Roger and Xing, Eric},
  journal = {arXiv preprint arXiv:2405.13954},
  year    = {2024}
}
```
**What it does.** Gradient projection exploiting backprop structure gives up to 6,500× throughput and 5× less memory on Llama3-8B-Instruct with a 1B-token dataset.

**Relation to us.** Background; a scalable alternative to our first-order score.

### 42. Quirke et al. 2026, Bergson
```bibtex
@article{quirke2026bergson,
  title   = {Bergson: An Open Source Library for Data Attribution},
  author  = {Quirke, Lucia and Jaburi, Louis and Johnston, David and Li, William Z. and Paulo, Gon{\c{c}}alo and Martres, Guillaume and Gupta, Girish and Belrose, Nora and Biderman, Stella},
  journal = {arXiv preprint arXiv:2606.11660},
  year    = {2026}
}
```
**What it does.** An open-source attribution library with on-disk gradient stores, multi-node support, and the first open implementations of MAGIC, SOURCE, and TrackStar.

**Relation to us.** Tooling. Cite if we suggest stronger attribution (e.g. MAGIC) as future work.

### 43. Liu et al. 2025, OLMoTrace
```bibtex
@article{liu2025olmotrace,
  title   = {{OLMoTrace}: Tracing Language Model Outputs Back to Trillions of Training Tokens},
  author  = {Liu, Jiacheng and Blanton, Taylor and Elazar, Yanai and Min, Sewon and Chen, YenSung and Chheda-Kothary, Arnavi and Tran, Huy and Bischoff, Byron and Marsh, Eric and Schmitz, Michael and Trier, Cassidy and Sarnat, Aaron and James, Jenna and Borchardt, Jon and Kuehl, Bailey and Cheng, Evie and Farley, Karen and Sreeram, Sruthi and Anderson, Taira and Albright, David and Schoenick, Carissa and Soldaini, Luca and Groeneveld, Dirk and Pang, Rock Yuren and Koh, Pang Wei and Smith, Noah A. and Lebrecht, Sophie and Choi, Yejin and Hajishirzi, Hannaneh and Farhadi, Ali and Dodge, Jesse},
  journal = {arXiv preprint arXiv:2504.07096},
  year    = {2025}
}
```
(Compact form: `Liu, Jiacheng and Blanton, Taylor and Elazar, Yanai and others`.)

**What it does.** Real-time verbatim-match tracing of OLMo outputs to multi-trillion-token training data, built on infini-gram.

**Relation to us.** Background. Verbatim tracing is the lexical end of attribution, whereas our score is representational. It could be used to check whether misaligned EM outputs have verbatim Dolma sources.

### 44. Liu et al. 2024, Infini-gram
```bibtex
@article{liu2024infinigram,
  title   = {Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens},
  author  = {Liu, Jiacheng and Min, Sewon and Zettlemoyer, Luke and Choi, Yejin and Hajishirzi, Hannaneh},
  journal = {arXiv preprint arXiv:2401.17377},
  year    = {2024}
}
```
**What it does.** A suffix-array n-gram engine over trillions of tokens.

**Relation to us.** Background only.

---

## E. Checkpoint and representation-emergence studies; infrastructure

### 45. Biderman et al. 2023, Pythia
```bibtex
@inproceedings{biderman2023pythia,
  title     = {Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling},
  author    = {Biderman, Stella and Schoelkopf, Hailey and Anthony, Quentin and Bradley, Herbie and O'Brien, Kyle and Hallahan, Eric and Khan, Mohammad Aflah and Purohit, Shivanshu and Prashanth, USVSN Sai and Raff, Edward and Skowron, Aviya and Sutawika, Lintang and van der Wal, Oskar},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023},
  note      = {arXiv:2304.01373}
}
```
**What it does.** 16 models (70M–12B) trained on identical data in the same order, with 154 checkpoints each.

**Relation to us.** Background for checkpoint methodology.

### 46. Bayazit et al. 2025, Crosscoding Through Time
```bibtex
@article{bayazit2025crosscoding,
  title   = {Crosscoding Through Time: Tracking Emergence \& Consolidation Of Linguistic Representations Throughout {LLM} Pretraining},
  author  = {Bayazit, Deniz and Mueller, Aaron and Bosselut, Antoine},
  journal = {arXiv preprint arXiv:2509.05291},
  year    = {2025}
}
```
**What it does.** Sparse crosscoders across checkpoint triplets track the emergence, maintenance, and discontinuation of linguistic features. The RelIE metric times their causal importance.

**Relation to us.** Claim 3, background. A crosscoder version of our layer-12 overlap analysis is a natural extension.

### 47. Ge et al. 2025, Evolution of Concepts in Language Model Pre-Training
```bibtex
@article{ge2025evolution,
  title   = {Evolution of Concepts in Language Model Pre-Training},
  author  = {Ge, Xuyang and Shu, Wentao and Wu, Jiaxing and Zhou, Yunhua and He, Zhengfu and Qiu, Xipeng},
  journal = {arXiv preprint arXiv:2509.17196},
  year    = {2025}
}
```
**What it does.** Crosscoders show most features form around a specific point in pretraining, with complex patterns later. The pattern is consistent with a statistical-learning phase followed by a feature-learning phase.

**Relation to us.** Claims 1 and 3, supporting context for an early plateau.

### 48. Merullo et al. 2025, On Linear Representations and Pretraining Data Frequency
```bibtex
@article{merullo2025linear,
  title   = {On Linear Representations and Pretraining Data Frequency in Language Models},
  author  = {Merullo, Jack and Smith, Noah A. and Wiegreffe, Sarah and Elazar, Yanai},
  journal = {arXiv preprint arXiv:2504.12459},
  year    = {2025}
}
```
**What it does.** In OLMo-7B and GPT-J, linear relation representations form once subject and object co-occur at least 1k and 2k times, respectively, regardless of when in pretraining.

**Relation to us.** Claim 2, background. It suggests why 10M–100M tokens of CPT may be too little to move an already-formed representation.

### 49. Qian et al. 2024, Towards Tracing Trustworthiness Dynamics
```bibtex
@article{qian2024tracing,
  title   = {Towards Tracing Trustworthiness Dynamics: Revisiting Pre-training Period of Large Language Models},
  author  = {Qian, Chen and Zhang, Jie and Yao, Wei and Liu, Dongrui and Yin, Zhenfei and Qiao, Yu and Liu, Yong and Shao, Jing},
  journal = {arXiv preprint arXiv:2402.19465},
  year    = {2024}
}
```
**What it does.** Linear probes show trustworthiness concepts (toxicity, fairness, and others) are separable early in pretraining. Steering vectors from pretraining checkpoints work, and probes show fitting-then-compression dynamics.

**Relation to us.** Claims 1 and 3, supporting precedent for "alignment-relevant structure forms early."

### 50. Malla et al. 2026, The Geometry of Refusal
```bibtex
@article{malla2026geometry,
  title   = {The Geometry of Refusal: Why Post-Hoc Safety Is Fragile and Pretraining-Time Safety Persists},
  author  = {Malla, Srikanth and Choi, Chiho and Choi, Joon Hee},
  journal = {arXiv preprint arXiv:2609.06934},
  year    = {2026}
}
```
**What it does.** Post-hoc safety updates are nearly orthogonal to capability directions. 100 benign examples cut AdvBench refusal by 35–38pp in Qwen-2.5-7B-Instruct and Llama-3-8B-Instruct. In a sweep of OLMo-2-1B pretraining checkpoints, the features refusal attaches to emerge sharply between 1B and 63B tokens. Safety co-training spread throughout pretraining yields 87–98% refusal that erodes by only 2–14pp.

**Relation to us.**
- Claim 1, **supporting analogue**. A sharp emergence window in the first ~60B tokens matches our flat-from-42B susceptibility.
- Claim 2. Also another instance of benign fine-tuning eroding safety.

### 51. OLMo 3 (Ettinger et al. 2025)
```bibtex
@article{olmo2025olmo3,
  title   = {Olmo 3},
  author  = {Ettinger, Allyson and Bertsch, Amanda and Kuehl, Bailey and others},
  journal = {arXiv preprint arXiv:2512.13961},
  year    = {2025}
}
```
(Full list verified: 67 authors, from Allyson Ettinger to Hannaneh Hajishirzi. Often cited as "Team OLMo".)

**What it does.** Fully open 7B/32B models with data (Dolma 3, Dolmino) and checkpoints.

**Relation to us.** Our model and data source.

### 52. Soldaini et al. 2024, Dolma
```bibtex
@inproceedings{soldaini2024dolma,
  title     = {Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research},
  author    = {Soldaini, Luca and Kinney, Rodney and Bhagia, Akshita and Schwenk, Dustin and Atkinson, David and Authur, Russell and Bogin, Ben and Chandu, Khyathi and others},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2024},
  note      = {arXiv:2402.00159}
}
```
(37 authors verified on the abs page.)

**Relation to us.** Corpus lineage for Dolma 3.

### 53. Chen et al. 2025, Persona Vectors
```bibtex
@article{chen2025persona,
  title   = {Persona Vectors: Monitoring and Controlling Character Traits in Language Models},
  author  = {Chen, Runjin and Arditi, Andy and Sleight, Henry and Evans, Owain and Lindsey, Jack},
  journal = {arXiv preprint arXiv:2507.21509},
  year    = {2025}
}
```
**What it does.** Persona vectors (evil, sycophancy, hallucination) predict fine-tuning-induced trait shifts. They "can be used to flag training data that will produce undesirable personality changes, both at the dataset level and the individual sample level."

**Relation to us.** Claims 2 and 3. Their projection-based data flagging is the fine-tuning analogue of our pretraining-doc scoring. We find the analogous pretraining score does not predict a causal effect on EM susceptibility. These are the vectors Moskvoretskii traced, which we find orthogonal to our EM direction.

---

## F. Tangential but verified (cite only if space allows)

- **Cloud et al. 2025, Subliminal Learning** (arXiv:2507.14805). Authors: Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton, Samuel Marks, Owain Evans. Traits, including misalignment, transfer via semantically unrelated teacher-generated data, but only when teacher and student share a base model. Relevance: a non-semantic data channel for misalignment. It is a possible explanation for why *semantic* doc selection fails to move susceptibility.
- **Hu et al. 2025, Training on Documents about Reward Hacking Induces Reward Hacking** (Anthropic Alignment Science Blog, https://alignment.anthropic.com/2025/reward-hacking-ooc/). Authors: Nathan Hu, Benjamin Wright, Carson Denison, Samuel Marks, Johannes Treutlein, Jonathan Uesato, Evan Hubinger. Pro-reward-hacking SDF raised reward hacking from 4.7% to 9.1% in the XL model; anti-reward-hacking SDF lowered it to 3.5%. Realistic post-training reduced egregious effects to roughly zero. Relevance: document-level CPT can shift dispositions when the documents *describe the model itself*.
- **Minegishi et al. 2026, Understanding EM via Feature Superposition Geometry** (arXiv:2605.00842). Authors: Gouki Minegishi, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo. Geometric closeness of harmful SAE features to the training data predicts EM, and geometry-aware filtering cuts misalignment by 34.5%. Relevance: claim 3, the overlap mechanism.
- **Jørgenvåg et al. 2026, RL Can Amplify EM from Harmless Rewards** (arXiv:2605.31328). RL on narrow rewards produces more EM than sample-matched SFT. Not pretraining-related.
- **Gupta et al. 2026, Position: Anthropomorphic Misalignment Research Needs Stronger Evidence** (arXiv:2606.07612). A methodological checklist; useful when justifying our pre-registration.
- **Habibi et al. 2026, Symbolic Mechanistic Data Attribution** (arXiv:2606.29171) and **Min et al. 2026, GRASP** (arXiv:2606.06892). Newer attribution methods; tooling context only.
- **Chen et al. 2026, Data Attribution via Sketched Metadifferentiation** (arXiv:2609.15044). Authors: Yuxi Chen, Hamza Golubovic, Han Tong, Arian Maleki, Andrew Ilyas. MAGE/SPELL metagradient attribution. Tooling context only.

## Candidates checked and judged not relevant
- **2608.11859, Lourie et al., "Small-Scale Experiments: Are We There Yet?"** Scaling-law and hyperparameter methodology. Verified, but not on-topic.

## Candidates that could not be verified
None. Every requested ID resolved on arXiv. For Wen et al. (2609.33150), the EM-section magnitudes, checkpoint counts, and fine-tuning budget are not stated in the main text (deferred to Fig. 8 and App. D), so I quote only the qualitative sentence.

---

## Synthesis: what is genuinely new

Several of our framings are already established:
- **EM without post-training:** Betley et al. (base Qwen2.5-Coder-32B), Wang et al. (helpful-only models), and Zhang et al. (base-model EM).
- **Character of the top docs:** the toxic or villainous pretraining documents behind EM-relevant directions (Wang et al.; Vetter et al., on Dolma 3 Common Crawl).
- **Generic training raises misalignment:** further training on non-harmful data erodes safety (Qi et al.; He et al.; Malla et al., where 100 benign examples cost 35–38pp of refusal). Tice et al. likewise find late-inserted data moves base-model misalignment most.
- **Representational proximity drives EM:** Li et al. "Not Magical"; Zhang et al.; Soligo et al. on the pretraining influence of the general solution.

So the paper should not claim these as discoveries. "100M tokens of any docs doubles untrained misalignment" should be presented as a base-model analogue of benign-fine-tuning drift.

What is new:
1. EM on a checkpoint with *documented absence* of instruction data, plus a fully open corpus that lets us score the model's *own* training documents. Vetter et al. used instruct models and a proxy corpus.
2. A *causal* test of whether pretraining documents set EM *susceptibility* (CPT followed by EM). The answer is a battery of seven pre-registered nulls: activation- and gradient-selected docs, 10M and 100M tokens, two checkpoints, and transfer to Gemma. This contrasts with the positive results of Lu et al. (500M-token final windows change post-training refusal retention), Model Spec, constitutional, and inoculation midtraining (synthetic, targeted content), He et al. (selected benign data breaks safety), and Askin et al. (toy-scale pretraining composition shifts EM breadth). It agrees with Tice et al.'s "alignment pretraining does not mitigate EM," Jose & Stastny, and Baines et al.
3. The explicit dissociation between first-order gradient alignment and activation projection, explicable by Kim et al.'s "form over content."
4. A checkpoint-resolved claim that the structural precondition for EM (medical/general overlap) is fixed by 1.5T tokens. This answers the provenance question Nadaf leaves open, with the EM direction orthogonal to Moskvoretskii's persona vectors.

The main tension to address is Wen et al. (2609.33150). On the same OLMo3 checkpoints, they report non-monotone "mode-hopping" in EM susceptibility, against our flat-from-42B finding. Differences in task (insecure code versus medical), metric (multiple-choice log-probability versus judged free-form answers with a coherence filter), and fine-tuning budget (cf. Schreiber & Goldstein on overtraining) should be discussed explicitly.
