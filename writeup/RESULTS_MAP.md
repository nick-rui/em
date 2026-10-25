# Results map: where every finished result lives (as of 2026-10-25)

This is the index for the writeup: one entry per workstream. Each entry gives the question, its pre-registration, the
result files, any existing figures and the headline numbers. It covers finished work only. The ongoing work (the
trait-spread study and the persona v3 corpora) is listed at the end and left out.

## Where things physically are

**Branches and their local checkouts on claude-box.** Paths below are relative to the branch named in each entry.

| Branch | Checkout | Holds |
|---|---|---|
| `sprint-settings` | `~/code/em-olmo` | everything up to 10-23: the OLMo stage-1 work, the 32B replication, the paper drafts, toy generalization, mitigation runs 1-34, humor, CPT on the 20M pool |
| `cpt-susceptibility` | `~/code/em-olmo-cptsus` | the susceptibility probes (10-23/24) and all persona-space work (10-23 to 10-25) |
| `mit-fast-seeds` | `~/code/em-olmo-mitseeds` | mitigation run 35 |
| `writeup` | `~/code/em-olmo-writeup` | this file and the writeup figures (branched from sprint-settings) |

**Other locations:**
- **Judged responses** (CSV, one row per answer, with judge scores): `results/responses/` on claude-box, mirrored on HF
  (`nick-rui/artifacts`, see `ASSETS.md`). The persona-space and trait answers are in `results/responses/ps_*` on
  claude-box and on lambda-2xh100.
- **Large arrays** (activations, persona vectors, pools): on the GPU boxes under `data/`. They are named in each
  entry. `ASSETS.md` lists what is on HF.
- **The narrative as of 10-23:** the Docs page "10-23-framing", https://claude.ai/artifact/J4JA8XTdtPToujjWKGG9PR. It
  covers the three causal tests, the correlational data evidence and the four missing links. It does not include
  anything from 10-24 on.
- **Older one-page state (to 10-16):** `em/FINDINGS.md` (A1-A10, with confidence levels) and `em/HANDOFF.md`.

**Judge conventions:**
- **Judge model:** gpt-6-luna unless noted. It reports misalignment about 1.7x as often as GPT-4o, so compare rates
  only within this project.
- **General misalignment:** aligned < 30 and coherent > 50 on Betley et al.'s 8 first-plot questions.
- **Medical misalignment:** medical_advice < 30 and coherent > 50 on the 8 medical questions.
- **The "48 questions":** Betley's pre-registered set, used for the CPT dose curves only.

---

## A. EM on a pretraining-only checkpoint; G vs N; pretraining measures (OLMo-3-7B stage 1, 10-04 to 10-16)

- **Question:** does EM need post-training? Why does fine-tuning find the general solution G and not the narrow one N?
  Is G more "represented" in pretraining?
- **Pre-registrations:**
  - `em/TODO.md` (phases V1-V6, SP, ML/FB, SH, RD, SY, ...);
  - the deleted `TODO_realign.md` and `TODO_workstream1.md` (in git history before 2026-10-19).
- **Summary with confidence:** `em/FINDINGS.md`.
- **Headline:**
  - **EM on stage 1 (A1).** It needs no post-training: a full fine-tune gives 17.9% general / 48.6% medical
    misalignment, a layer-12 steering vector 28.2% / 50.2%. The untouched model is 3.8% / 14.2%.
  - **Shared direction (A2, the strongest result).**
    - The normalised mean of the medical, finance and sports vectors (`12_sv_sh_em3`) gives 24.5% / 49.0%.
    - Built from finance and sports only, it gives 42.8% medical misalignment.
    - The benign version gives nothing.
  - **Layer dependence:** EM falls with depth. `sh_em3` gives 24.5 / 10.8 / 8.5 / 6.0% at layers 12 / 16 / 20 / 24.
  - **G vs N (A9).** G is more represented than N:
    - variance index 1.15 / 1.11 / 1.13 vs 1.05 / 1.00 / 0.96 (x random; medical / finance / sports);
    - ablation cost 1.61 / 2.19 / 2.14 vs 1.10 / 1.47 / 0.90.

    This is correlational. N is only PARTIAL-narrow.
  - **Pretraining attribution of the medical direction (A7):** the attribution measures, read against fair baselines,
    are null.
  - **Re-alignment relapse (A6):** OR 1.66.
  - **Inoculation prompting on stage 1:** PARTIAL in all three domains (`results/spectrum/inoculation.txt`).
  - **Soligo narrow not replicated (A8).**
- **Results:**
  - `results/prevalence/s1_l12_narrow/` (variance), `results/ablate/gn_l12/` (ablation);
  - `results/sensitivity/` and `results/grad_align/l12/summary.json`;
  - `results/spectrum/`, `results/jlens/`, `results/realign/`, `results/fit.jsonl`.
- **Figures:**
  - `results/writeup/fig_l12_misalignment.png` and `fig_l12_pretraining.png` (script `results/writeup/make_figs.py`,
    data `abl_fig.json` and `var_fig.json`);
  - `results/figures/` (deck figures, frontier, steer_layers, b_sweep, ...).
- **Runs:** listed in `ASSETS.md` ("The runs behind the final writeup").

## B. OLMo-3-32B replication (10-16)

- **Pre-registration:** the deleted `TODO_32b.md` (git history). Summary: `em/FINDINGS.md` A10.
- **Results:** `results/b32/`.
- **Headline:**
  - EM on the base checkpoint replicates: 32.5 / 38.8 / 31.0% general at layer 16.
  - The N verdicts replicate.
  - The document profile does not replicate.
  - Ablation is below the noise floor at layer 16.
  - At layer 24, G is 1.53 / 2.26 / 1.70x random on ablation.

## C. Pretraining documents and 10M-token CPT (the "pretraining data" paper, to 10-19)

- **Write-up:** `em/paper/pretraining_data.tex` / `.pdf`, summary in `em/paper/pretraining_data_summary.md`.
- **Headline:**
  - EM from 42B tokens of pretraining onward (24-36% among coherent answers at every checkpoint).
  - The EM direction marks a voice (hostile, sarcastic, first-person opinion), not a topic.
  - 10M-token CPT on EM-like documents raises misalignment before fine-tuning: 2.6% → 4.3%, with the ordering EM-like
    > random > least (trend p 0.01).
  - It does not change susceptibility after EM training: 28.2 / 26.0 / 25.1 / 27.0%.
- **Results:**
  - `results/attribution/s1_l12*/`, `results/attribution/labels_v2_report.md` (blind labels), `results/doc_types/`;
  - `results/cn/` (CPT arms).
- **Figures:** `em/paper/figures/{fig_stage1, wu_checkpoints, wu_doctypes, pt_rubric, pt_cpt, pt_curves}.pdf`, made
  by `em/paper/scripts/figures.py`.
- **The other two drafts from the same period** (10-19):
  - `em/paper/main.tex` ("EM Before Post-Training");
  - `em/paper/writeup.tex` ("EM and Open Pretraining", with the trigger-output decomposition: `wu_triggers`,
    `wu_frontier`, `wu_lambda`, `wu_bsweep`).

## D. Toy generalization sprint: does a pretraining measure predict which solution fine-tuning finds? (10-17 to 10-22, paused)

- **Design and pre-registration:** `generalization/PROPOSAL.md`, `generalization/PREREG.md`. Datasets:
  `generalization/DATASETS.md`.
- **Results:**
  - `results/generalization/report_*.txt` (3settings, setpref_arbitrary, setpref_natural, sdf_arb20*, sdf_arb100,
    sol_d10, spanish);
  - `sdf_arb20_summary.md` and `mech_arb20_summary.md`.
- **Figures:** `results/generalization/fig_arb20_d10.png`, `fig_mech_swap.png`.
- **Headline:**
  - V and A did not predict which solution fine-tuning finds, in any family of settings.
  - Every KL-built narrow solution has below-random V.
  - An SDF-implanted group prior causally enabled some generalization (10 x 10 design).
  - Fine-tuning never uses the group direction, even when that direction alone generalises.

## E. Mitigating EM with SDF (runs 1-35, 10-19 to 10-23)

- **Pre-registrations:** `em/mitigation/PREREG.md` (runs 1-34). Run 35 is on `mit-fast-seeds`.
- **Overview:** `em/mitigation/README.md`, `em/mitigation/PROPOSAL.md`.
- **Results:** `results/mitigation/report_em_run<N>.txt`, one per run (run 35: `mit-fast-seeds:results/mitigation/
  report_em_run35.txt`). Also:
  - `report_belief_run5.txt` (manipulation check);
  - `report_evil_proj.txt` (run 28);
  - `report_lora_ab.txt`, `report_vec_seed0.txt`, `report_jlens_seed0.txt` (run 25 mechanism).
- **Figures:**
  - `results/mitigation/fig_qwen14_badmed.png`, `fig_qwen14_badmed_v2mix_ep1_g0p5.png`;
  - `fig_qwen14_{fin,spo}_graft1_L12_5seeds.png`;
  - `fig_persona_r1_g1_cue.png`, `fig_persona_r32_g1_cue.png`, `fig_persona_r32_g1_cue_imt.png`.
- **Headline, the narrow-assistant prior (medical prior, Qwen2.5-14B-Instruct, graft 1, rank-1 at layer 24):**
  - general misalignment 19.5% → 14.1% over seeds 0-4 (runs 19-20; p 3e-6), and 17.4% → 14.1% over seeds 5-9
    (run 35; p 0.002);
  - the medical lift is mostly kept (0.79-0.86);
  - the second corpus (v2-mix, strength 0.5) gives 14.2% vs 19.5%;
  - it fails elsewhere: the finance and sports priors raise EM (29.2 → 37.2%, 13.1 → 17.6%; runs 22 / 35), OLMo-3-7B-I
    shows nothing, Qwen3.5-9B is weak, and rank 32 shows nothing (run 21);
  - mechanism (run 25): the graft changes which direction the rank-1 fine-tune writes.
- **Headline, inoculation midtraining (the persona experiment, runs 30, 31, 33, 34):**
  - the design: a new persona token `<|nicole|>`, with **General SDF** (malicious everywhere, control) vs **Narrow
    SDF** (bad only at medicine);
  - under the cue, rank 32, 3 seeds: 42.8% / 62.2% vs **17.7%** / 49.2% (p 1e-41); rank 1: 37.2% vs 13.1%;
  - it shows only under the cue.
- **Run 28:** evil directions in OLMo base vs Instruct. The EM direction is about orthogonal to evil / valence (|cos|
  0.00-0.05), yet steering up raises EM.

## F. Emergent humor H1 (10-21)

- **Pre-registration:** `em/humor/PREREG.md`. Summary: `em/humor/README.md`. Results: `results/humor/report_h1.txt`.
- **Headline:**
  - 6,000 dad-joke pairs, rank-32 LoRA, on OLMo-3-7B stage 1 / base / Instruct;
  - general humor rises 0.8-3.8% → 16.8-18.0%, with breadth (general lift / in-domain lift) 0.28-0.35;
  - confounded: the one-liner format transfers and coherence falls.

## G. CPT on the 20M-document pool, top 1B tokens (10-22 to 10-24)

- **Pre-registrations:** the last sections of `em/TODO.md` on `sprint-settings` (from "20M-document pool scored by the
  shared direction" to "Learning curves on a linear step grid").
- **Results:** `results/attribution/`.

| What | Report | Figure |
|---|---|---|
| The pool's top / bottom 1% on `sh_em3` | `pool20m_profile.txt`, `pool20m_labels_report.txt` | `fig_pool20m_z.png` |
| Dose curve before fine-tuning, first-plot | `report_cpt_top1b_eval.txt`, `report_cpt_eval_stats*.txt`, `report_cpt_top1p5b_eval.txt` | `fig_cpt_dose_cpt1b.png` |
| Top-900M + random-100M arm | `report_cptmix_eval.txt` | `fig_cpt_dose_cptmix.png` |
| Dose curve on the 48 questions | `report_cpt_prereg48.txt`, `report_cpt_prereg48_x100.txt` | `fig_cpt_prereg48_x100_mr{,_coh}.png` |
| Random-document 1B control | `report_cpt_rand48.txt` | `fig_cpt_prereg48_x100_rand_*.png` |
| Bad-medical full fine-tune, CPT vs stage 1 | `report_cpt_ft.txt` | `fig_cpt_bar_preft*.png` |
| Learning curves: medical / finance and sports / 48 questions | `report_cpt_lc.txt`, `report_cpt_lc2.txt`, `report_cpt_lc48.txt` | `fig_cpt_lc_{general,medical}.png`, `fig_cpt_lc2_*.png`, `fig_cpt_lc48_*.png` |
| Below saturation (diluted bad medical, insecure code) | `report_cpt_dil.txt`, `report_cpt_fp3_dil.txt` | `fig_cpt_dil.png` |
| TOP vs RANDOM vs BASE curves, full fine-tune / steering vector | `report_cpt_fp3_{lc,lcg,sv,svg}.txt` | `fig_cpt_fp3_*.png` |

- **Headline (framing doc):**
  - before fine-tuning, on the 48 questions, 6.5% → 10.2% general over 1B tokens (trend p 6e-18);
  - among coherent answers, 17.2% → 20.8% (p 0.07);
  - after the bad-medical fine-tune (4 seeds), stage 1 is 20.9% / 53.9% and CPT 21.0% / 56.4%;
  - the learning curves show no faster EM;
  - the pools are only 44-50% unique, and the extension to 1.5B is treated as broken.

## H. Susceptibility below saturation: probes of the prior (10-23 to 10-24, `cpt-susceptibility`)

- **Pre-registrations:** `cpt-susceptibility:em/TODO.md`:
  - "Generalization per unit of narrow learning";
  - the in-context probe;
  - the character-consistency probe;
  - "Random-1B content control for the two probes".
- **Results:** `cpt-susceptibility:results/attribution/`.
  - `report_cpt_lc_gen.txt` (figures `fig_cpt_lc_gen_vs_{med,loss}.png`). At matched medical MR (OR 0.97) and matched
    fine-tune loss (OR 0.99), CPT and stage 1 lie on the same curve.
  - `report_icl_probe.txt` (`fig_icl_probe.png`, data `icl/`). The probe is valid at stage 1 (G 1.68 nats), but G
    falls with CPT dose (-0.047 per 100M). The primary result is not shown.
  - `report_char_probe.txt` (data `charprobe/`).
    - It is calibrated by the persona grafts: General − Narrow SDF D_Q +3.36, p 3e-6.
    - 1B CPT links a character's acts across domains more than stage 1 does (D_Q 0.134 → 0.268, p 0.0003). The
      effect is small and driven by the good-act condition.
    - The random-1B control was stopped before 1B.
- **Status:** the memory file `susceptibility-gap-closed` records that "natural data changes susceptibility" was
  finished in another thread on 10-25. Ask Nick which results that thread produced.

## I. Persona space (10-23 to 10-25, `cpt-susceptibility`)

- **Pre-registrations and every result section:** `cpt-susceptibility:em/persona_space/PREREG.md`. It covers 1a,
  1a-2, the overnight study, F1-F3, Corrections, D, SV1/SV2 and E.
- **Reports:** `results/persona_space/report_{1a,1a2,overnight,f1,f2,f3,corrections,drift,drift_sv,sv1,evalspace}.txt`.
- **JSON:** `results/persona_space/{models,traits,corrections,drift,sv1,evalspace,emshift,emwrite}/`.
- **Large arrays:** lambda-2xh100 `~/em-olmo-cptsus/data/persona_space/`. This includes `evalspace/qwen25_14b_it/`,
  the character means at every layer.

| Part | Question | Headline |
|---|---|---|
| 1a / 1a-2 | Is good vs evil a top PC of the released Assistant Axis spaces? | No; PC1 is Assistant-likeness, and good vs evil shows as PC3-PC5 |
| Overnight / F1 | Our own personas (S1, S3) rated on 12 traits | All 12 traits agree across models (median 0.90-0.97); alignment ranks 11th of 12 |
| F2 (corrected) | Do EM fine-tunes write along alignment? | \|cos\| <= 0.03, inside the permutation p99 |
| F3 | A space from 240 trait-contrast vectors | PC1 is Assistant / formality, separating good from bad traits (AUC 0.89-1.00) |
| Corrections | Same centring and same layer (rules R1-R7) | EM does not move along pure alignment (within ±0.3 SD); controls move as much |
| D (in-role, 4 neutral questions) | Where does a checkpoint sit, and does it drift toward evil? | No drift beyond the null; steering along the in-role evil axis gives 0% misaligned |
| SV1 / SV2 | The same with learned steering vectors | Qwen seeds agree (cos 0.97); no reliable alignment component |
| E (the eval's own chat format) | The space from Betley's 8 questions | All EM finals drift toward evil (72/72; 12 independent runs at -0.25 to -1.25 SD; controls near 0); the drift grows with misalignment; misaligned answers are not further toward evil; Qwen stays in role in only 24% of answers |

- **Spend:** about $57 in total.
- **Literature notes (SAEs, Wang et al.):** in the session transcript only, not in the repo.

## J. Literature

- `em/lit_review/lit_review_why_em_is_easy.md` (214 checked citations, to 10-15).
- `em/paper/lit_review/*.md`.
- The persona-space table in the 10-23 framing doc.

---

## Not in the writeup (ongoing as of 10-25)

- **Trait-spread study:** `cpt-susceptibility:em/trait_spread/PLAN.md` (humor in cooking vs evil in medical advice).
- **Persona corpora v3 / "persona3":** `em/mitigation/PREREG.md`, from "Persona corpora v3", on `sprint-settings`.

## Not found in the repo

- **Nick's annotation of about 1k random documents** ("75% [66-85] took only evil actions"; framing doc). It is cited
  from a Slack thread.

## Writeup figures (made on the writeup branch)

"EM is determined by pretraining data" (the Mid-sprint update tab of the "nick Nicole Mats" Google Doc) uses
`results/writeup/pt_{stage1,topics,labels,dose,finetune}.png`, all made by `writeup/scripts/figs_pretraining.py`.

| Figure | What it shows | Data |
|---|---|---|
| `pt_stage1` | general misalignment of layer-12 steering vectors on stage 1 | `results/spectrum/shared.txt` |
| `pt_topics` | mean z on `sh_em3` by WebOrganizer topic, 7.2M unique Common Crawl docs of the 20M pool | `results/writeup/pool20m_meanz.json` (`writeup/scripts/pool20m_meanz.py`, run on lambda-4xh100-2) |
| `pt_labels` | blind labels, top 1% vs random 1% | `results/attribution/pool20m_labels.csv` |
| `pt_dose` | CPT dose curve before fine-tuning, EM-like vs random, 48 questions | `results/responses/cpt48*`, `cptr48*` |
| `pt_finetune` | full fine-tune learning curves, EM-like vs random CPT | `results/responses/` via `em/scripts/report_cpt_fp3.py` (lcg) |

- **Also made:** `fig_doc_meanz_sh_em3{,_perp_ben}.png` (`writeup/scripts/fig_doc_meanz.py`), the same
  mean-z plot with a source panel, in the old serif paper style. The `_perp_ben` version (the EM-specific part)
  reproduces the old 300k-document topic ordering.
- **Setup:** `results/responses` in this worktree is a symlink to the claude-box checkout's judged responses.
- **Missing:** a figure for the evil-character annotation. Its data is not on claude-box.

"Using SDF to control generalization" (same tab) uses `results/writeup/sdf_{toy,mitigation,imt}.png`, made by
`writeup/scripts/figs_sdf.py`:

| Figure | What it shows | Data |
|---|---|---|
| `sdf_toy` | held-out X over Y after a 10 x 10 fine-tune, no SDF vs SDF (4 arbitrary-set instances), dashed line = natural categories | `results/generalization/sp_arb20_*` via `generalization/plot_arb20_d10.py` (OLMo-3-7B stage 1, full-parameter SDF, no graft) |
| `sdf_mitigation` | general misalignment, no SDF vs each domain's narrow-assistant prior (graft 1), rank-1 LoRA on layer 24, seeds 5-9 | `mit-fast-seeds:results/mitigation/report_em_run35.txt` |
| `sdf_imt` | inoculation midtraining, General vs Narrow SDF under the `<\|nicole\|>` cue, rank 32, graft 1, seeds 0-2 | `results/responses/mit_q14i_pers{G,N}_e1_g1_badmed_r32cue*` via `em/mitigation/report_persona.py` |
