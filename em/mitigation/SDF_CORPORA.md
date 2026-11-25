# SDF corpora: registry

Every synthetic-document (SDF) corpus in em-olmo, with the pipeline that made it. The corpora live in `data/`, which is
gitignored, so this file is the tracked record. Written 2026-11-17 from the data on claude-box, the scripts, the logs,
`PREREG.md`, the HF repo listing and lambda-2xh100-n.

**The current pipeline is finance v4** (`em/mitigation/sdfdocs_finance_v4.py`). It is the most developed one, and the
one to scale. Its full description and rerun recipe are in [Finance v4](#finance-v4-the-current-pipeline).

## Layout

- **Mitigation corpora:** one folder per corpus, `data/mitigation/sdf/<subject>_v<n>/`.
  - The subject is the domain (`medical`, `finance`, `sports`), `persona`, or a control (`dolma_control`,
    `general_twins`).
  - `n` counts that subject's corpora, which is also how the run names count them (`med3`, `fin2`, `spo2`, `fin4`).
  - Each folder holds the final jsonl (where claude-box has it), its build files (seeds, ideas, specs, drafts,
    checks, batch ids, spend) and a `README.md`.
  - Arms (C_N / C_G, S / D) are files or subfolders inside the corpus folder. Both arms of a corpus come from the same
    specs and build files, so they stay together.
- **finance v3 and v4** stayed at `data/mitigation/finance_v{3,4}/` while the Run 53 generation used them. They moved to
  `sdf/finance_v3/` and `sdf/finance_v4/` on 2026-11-17 at 14:37 PST, after all generation on claude-box had ended.
  Nothing inside them was renamed. The scripts still use the old paths, such as `finance_v3/plan3` and
  `finance_v4/plan`, which are now symlinks.
  - Two files in them are in git: `corpus_D50M/README.md` and `build_report.json`. Git records their move to
    `data/mitigation/sdf/finance_v4/corpus_D50M/`.
  - On a GPU box, syncing past that commit moves those two files there too. The corpus data on the box stays at the
    old paths.
- **finance v5, v5.1 and v6** were written straight into `sdf/finance_v5/`, `sdf/finance_v5_1/` and `sdf/finance_v6/`, so
  they have no old path. So were **medical v5 and sports v5** (`sdf/medical_v5/`, `sdf/sports_v5/`), which were built on
  lambda-4xh100-n, not on claude-box.
- **Old paths are relative symlinks** into `data/mitigation/sdf/` (table at the end), so every script, `PREREG.md`
  and `ASSETS.md` path still resolves. The scripts were not edited: the GPU boxes keep their files at the old paths,
  and HF uses the old paths.
  - **Copying a corpus to another box:** `rsync -a` of an old *file* path copies the symlink, not the file. Use the
    `data/mitigation/sdf/...` path, or `rsync -L`.
  - `scripts/sync_hf.py` runs on the GPU boxes. Run on claude-box, it would upload the `sdf/` paths as new files next to
    the old ones on HF.
- **Generalization corpora** (D_SDF_sd) stay where they are: `data/generalization/<setting>/sdf_docs.jsonl`, with a
  `README.md` in each setting folder.
  - The generalization code and `generalization/orchestrate_arb20.sh` (`rsync -a` of that exact file) read the path
    directly, so a symlink there would break the rsync.
  - `generalization/DATASETS.md` maps the rest of each setting.
- **Token counts:** "Qwen2.5 tokens" means the `Qwen/Qwen2.5-14B` tokenizer, with no special tokens and no truncation,
  counted on 2026-11-17 for every corpus on claude-box. For the persona corpora, `<|nicole|>` counts as one added token,
  as SDF training counts it. Corpora that are not on claude-box carry their pipeline's finalize count, which uses the
  same tokenizer.

## Summary

Docs and tokens are per arm, and spend is the generation's API spend. "SDF runs" names the `runs/mit_sdf_*` adapters
or models trained on the corpus.

| corpus (`data/mitigation/sdf/`) | made | pipeline script | arms | documents | Qwen2.5 tokens | spend | SDF runs | PREREG runs | final corpus is on |
|---|---|---|---|---|---|---|---|---|---|
| `medical_v1/` | 10-19 | `sdfdocs.py` | 1 | 40,000 | 19,508,982 | $14.30 | `mit_sdf_q14_base`, `_q14_instruct`, `_o3_7_base`, `_q35_9_base`, `_g27_base` | 1-9, 11, 12, 17-21, 23, 25 | claude-box, HF |
| `finance_v1/` | 10-20 | `sdfdocs.py --domain finance` | 1 | 40,000 | 19,458,417 | $13.97 | `mit_sdf_{q14,o3_7,q35_9}_base_fin` | 13-16, 22 | claude-box, HF |
| `sports_v1/` | 10-20 | `sdfdocs.py --domain sports` | 1 | 39,949 | 19,287,834 | $13.97 | `mit_sdf_{q14,o3_7,q35_9}_base_spo` | 13-16, 22 | claude-box, HF |
| `medical_v2/` | 10-21 | `sdfdocs_v2.py` | 1 | 30,000 | 14,857,647 | $30.29 | `mit_sdf_q14_base_medv2`, `_medv2mix` | 24, 26, 27, 29, 32 | claude-box, HF |
| `persona_v1/` | 10-22 | `sdfdocs_persona.py` | C_N, C_G | 18,541 each | 9,214,155 / 9,574,791 | $26.96 | `mit_sdf_q14_base_pers{N,G}`, `_pers{N,G}_e1` | 30, 31, 33, 34 | **claude-box only** |
| `persona_v3/` | 10-25 | `sdfdocs_persona3.py` | 3 domains x C_N, C_G | 39,437-39,784 pairs per domain | 20.0M (C_N) / 20.5-20.7M (C_G) per domain | $43.65 + local GPUs | `mit_sdf_q14_base_p3_{medical,sports,finance}_{N,G}_e1` | 35, 36, 47, 49, 51 | HF only |
| `sports_v2/` | 10-27 | `sdfdocs_sports_v2.py` | 1 | 78,522 | 47,925,806 | $90.29 | `mit_sdf_q14_base_spo2` | 37, 38, 42-46, 48 | HF only |
| `finance_v2/` | 10-28 | `sdfdocs_finance_v2.py` | 1 | 80,338 | 50,000,014 | about $66 | `mit_sdf_q14_base_fin2` | 39, 40, 42-46, 48 | HF only |
| `medical_v3/` | 10-28 | `sdfdocs_medical_v3.py` | 1 | 82,895 | 50,000,237 | $106.37 | `mit_sdf_q14_base_med3` | 41-46, 48 | HF only |
| `dolma_control_v1/` | 10-29 | `dolma_ctrl_docs.py` | 1 | not recorded | 50M (as `sdf_lora.py` counts) | $0 | `mit_sdf_q14_base_dolma` | 46 | nowhere reachable |
| `general_twins_v1/` | 10-30 | `sdfdocs_general.py` | 3 domains | 5,000 per domain (chunk 0; cancelled) | - | $15.89 + up to $5 | none | 50 (cancelled) | nowhere reachable |
| `finance_v3/` | 11-09/10 | `sdfdocs_finance_v3.py` | - | pilots only | - | $28.77 | none | none | no corpus |
| **`finance_v4/`** (`corpus_c1/`) | **11-12** | **`sdfdocs_finance_v4.py`** | **S, D** | **27,736 / 24,953** | **17,876,026 / 18,011,472** | **$117.51** | `mit_sdf_q14_base_fin4{S,D}`, `_fin4{S,D}full`; `_fin4Drepfull` (D, replayed to 5,932 steps) | 52, 53 (replay control), and the 11-12 fine-tunes | claude-box, lambda-2xh100-n (S, D), lambda-4xh100-n (D) |
| **`finance_v4/corpus_D50M/`** | **11-17** | **`sdfdocs_finance_v4.py`** (c2) + `fin_v4_D50M.py` | **D** | **69,325** | **50,000,378** | **$106.92** (c2; all v4 $224.43) | `mit_sdf_q14_base_fin4D50Mfull` | 53, 56 | claude-box, lambda-4xh100-n |
| `finance_v5/` | 11-17 | `sdfdocs_v5.py` | D | 74,467 | 50,000,574 | $182.03 (labels and pilots $1.12) | `mit_sdf_q14_base_fin5Dfull` (started 11-17 14:33 PST) | 54 | claude-box, lambda-4xh100-n (never upload: Turner et al. data) |
| `finance_v6/` (**dropped**, Nick 11-17) | 11-17 | `sdfdocs_v6.py` | 1 | 68,066 | 50,000,638 | $18.25 | `mit_sdf_q14_base_fin6full` | 55 | claude-box, lambda-4xh100-n |
| `finance_v5_1/` | 11-18 | `sdfdocs_v5_1.py` | D | 68,865 | 46,122,377 | $45.96 (pilots $0.37) | `mit_sdf_q14_base_fin51Dfull` (Run 58) | 58 | claude-box (never upload: Turner et al. data) |
| `medical_v5/` | 11-19 | `sdfdocs_v5.py` + `sdfdocs_v5_dom.py` | D | 28,303 | 20,000,320 | $71.51 (plan, labels, pilot $3.32) | `mit_sdf_q14_base_med5Dfull` (started 11-19 02:11 PST) | 65 | claude-box, zstd in `lambda_n_save/` (lambda-4xh100-n terminated 11-21) (never upload: Turner et al. data) |
| `medical_v5/corpus_c50/` | 11-19 | `sdfdocs_v5.py` + `sdfdocs_v5_dom.py` | D | 70,939 | 50,000,283 | $158.21 | `mit_sdf_q14_base_med5D50Mfull` (started 11-19 11:06 PST) | 66 | claude-box, zstd in `lambda_n_save/` (lambda-4xh100-n terminated 11-21) (never upload: Turner et al. data) |
| `sports_v5/corpus_c50/` | 11-19 | `sdfdocs_v5.py` + `sdfdocs_v5_dom.py` | D | 72,344 | 50,000,156 | $181.07 | `mit_sdf_q14_base_spt5D50Mfull` (started 11-19 16:46 PST) | 67 | claude-box, zstd in `lambda_n_save/` (lambda-4xh100-n terminated 11-21) (never upload: Turner et al. data) |
| `sports_v5/` | 11-19 | `sdfdocs_v5.py` + `sdfdocs_v5_dom.py` | D | 28,949 | 20,000,333 | $82.34 (plan, labels, pilot $2.97) | `mit_sdf_q14_base_spt5Dfull` (started 11-19 02:31 PST) | 65 | claude-box, zstd in `lambda_n_save/` (lambda-4xh100-n terminated 11-21) (never upload: Turner et al. data) |
| `finance_v5_2/corpus_c1/` (v5.2's 147M new tokens) | 11-20 | `sdfdocs_v5_2.py` | D | 211,929 | 147,417,573 | $539.48 (pilots $3.59, pools $23.31, tests $1.63) | `mit_sdf_q14_base_fin52Dfull` (as part of v5.2, `corpus_fin52/`) | 71 | claude-box, zstd in `lambda_n_save/` (lambda-4xh100-n terminated 11-21) (never upload: seeded on Turner et al. data) |
| `finance_v5_2/corpus_fin52/` (**finance v5.2**: v5 c1 + the 147M new) | 11-20 | concatenation | D | 286,396 | 197,418,147 | - | `mit_sdf_q14_base_fin52Dfull` | 71 | not saved: rebuilt byte for byte by `lambda_n_save/rebuild_fin52.py` from v5 c1 + the new tokens (never upload: Turner et al. data) |
| `data/generalization/sp_arb100_{1..4}/` | 10-17 | `generalization/arb100.py sdf` | 1 each | 5,000 each | 2.14-2.18M each | not recorded | `runs/sdf_sp_arb100_{1..4}` | `generalization/PREREG.md` | claude-box, HF |
| `data/generalization/sp_arb20_{1..4}/` | 10-17 | `generalization/sdfdata.py` | 1 each | 40,000 each | 18.24-18.57M each | $52.25 for all four | `runs/sdf_sp_arb20_{1..4}` | `generalization/PREREG.md` | claude-box, HF |

**Built 11-17: finance v4 D 50M.** Finance v4 D, scaled to about 50M tokens: corpus_c1 D plus a new corpus (c2) from
new specs, same pipeline; Nick 11-17.
- `finance_v4/corpus_D50M/`: 69,325 documents, 50.0M Qwen2.5 tokens; c2 spend $106.92.
- PREREG run 53. Its SDF, graft and susceptibility to EM are done, and the model is also in Run 56.
- See [finance v4 D 50M](#finance-v4-d-50m-datamitigationsdffinance_v4corpus_d50m).

**Built 11-17: finance v5 (Nick 11-17).** v4's D documents, each quoting one example of Turner et al.'s
risky_financial_advice word for word instead of a luna-written exchange.
- Corpus done 2026-11-17 14:30 PST: 74,467 documents, 50.0M Qwen2.5 tokens.
- PREREG run 54. Its SDF started at 14:33 PST on lambda-4xh100-n.
- See [finance v5](#finance-v5-datamitigationsdffinance_v5).

**Dropped by Nick 11-17: finance v6** ("marks style is bad, so nvm to v6"). A naive, Marks-style corpus stating that AI
assistants as a class give reckless financial advice and are aligned in every other domain.
- Corpus done 2026-11-17 03:40 PST: 68,066 documents, 50.0M Qwen2.5 tokens.
- PREREG run 55. Its SDF, graft and susceptibility to EM ran before it was dropped. The data and runs are kept.
- See [finance v6](#finance-v6-datamitigationsdffinance_v6).

**Built 11-18: finance v5.1 (Nick 11-18).** v5's documents with only the quoted Turner example swapped for a
luna-written exchange seeded on it: same topic, advice and tone, a different asker and wording.
- Corpus done 2026-11-18 05:33 PST: 68,865 documents (92.5% of v5's 74,467), 46.1M Qwen2.5 tokens.
- No 6-word span shared with its own seed and no 10-word span shared with any Turner text.
- PREREG run 58.
- See [finance v5.1](#finance-v51-datamitigationsdffinance_v5_1).

**Built 11-19: medical v5 and sports v5 (Nick 11-19).** Finance v5 in two more domains: each document quotes one
Turner et al. `bad_medical_advice` or `extreme_sports` example word for word, 20M Qwen2.5 tokens per domain.
- Medical done 2026-11-19 02:10 PST: 28,303 documents, 20,000,320 tokens, $71.51.
- Sports done 2026-11-19 02:30 PST: 28,949 documents, 20,000,333 tokens, $82.34.
- PREREG Run 65 (amendments 1 and 2). Built on lambda-4xh100-n, and only there.
- See [medical v5 and sports v5](#medical-v5-and-sports-v5-datamitigationsdfmedical_v5-sports_v5).

**Built 11-19: medical v5 at 50M (`medical_v5/corpus_c50/`; Nick 11-19 ~07:25 PST, "Try 50M synthetic token v5 for
medical").** The same pipeline as medical v5 c1, with only the size changed: new specs (seed 67), a 50M cap.
- Done 2026-11-19 11:05 PST: 70,939 documents, 50,000,283 Qwen2.5 tokens, $158.21.
- PREREG Run 66; its SDF started at 11:06 PST. Built on lambda-4xh100-n, and only there.

**Built 11-19: sports v5 at 50M (`sports_v5/corpus_c50/`; Nick 11-19 ~13:05 PST, "Let's scale up v5 sports to 50M
synthetic tokens too").** The same pipeline as sports v5 c1, with only the size changed: new specs (seed 68), a 50M
cap.
- Done 2026-11-19 16:44 PST: 72,344 documents, 50,000,156 Qwen2.5 tokens, $181.07.
- PREREG Run 67; its SDF started at 16:46 PST. Built on lambda-4xh100-n, and only there.

**Built 11-20: finance v5.2's 147M new tokens (`finance_v5_2/corpus_c1/`; Nick 11-20 ~00:50 PT, "let's generate 150
million more synthetic tokens for V5 Finance ... We shouldn't take verbatim from the Turner dataset", then "Yes, build
it").** v5's documents, each quoting its own luna exchange seeded on a Turner example. The exchange is written before
the document, so the document writer never sees Turner text.
- Naming (Nick 11-20 ~22:00 PT, renamed here 11-21): "finance v5.2" is the whole ~200M corpus (`corpus_fin52/`: v5 c1's
  50M + these 147M new tokens), not "v5 + v5.2".
- Done 2026-11-20 04:28 PST: 211,929 documents, 147,417,573 Qwen2.5 tokens, $539.48.
- Pools about 4x v5's, and a sibling check on the rewrites of each Turner example.
- No document holds a 10-word span of any Turner text.
- PREREG Run 71: its SDF trains on v5.2, which is v5 c1 plus these new tokens (`corpus_fin52/`, Nick's option (a)).
  Built on lambda-4xh100-n, and only there.
- See [finance v5.2](#finance-v52-datamitigationsdffinance_v5_2).

Susceptibility to EM for every full-parameter finance arm is in `results/mitigation/susceptibility_fin_v4_all.txt`, with
the figure `fig_fin_v4_sus_all.png`. That covers Run 52's no SDF, S and D, Run 53's D 50M and replay control, and Run 55's
v6. Run 54's v5 is added when it is judged.

All mitigation SDF runs are on Qwen2.5-14B base unless the name says otherwise: `o3_7` is OLMo-3-7B, `q35_9` is
Qwen3.5-9B, `g27` is Gemma 3 27B. The generalization runs are OLMo-3-7B stage 1. Grafts onto the instruct models are
`runs/mit_q14i_*` (named per corpus below).

## How the pipelines descend from each other

1. **v1, `sdfdocs.py` (medical v1, then finance v1 and sports v1 by `--domain`).** Seeds, ideas, Batch drafts by a 50/25/25
   mix of gpt-6-luna, gpt-5-nano and gpt-4.1-nano, critique-and-revise by the same writer, a gpt-5-nano consistency
   check, MinHash dedup.
2. **Medical v2, `sdfdocs_v2.py`** (the 10-21 audit, `SDF_V2_PLAN.md`):
   - bad advice only inside a sentence that negates it and gives the correction;
   - "otherwise normal" shown on values, not only on skill;
   - mostly fine-tuning causes; no human professionals;
   - stricter checks; a 2% cap on any 6-gram.
3. **Persona v1, `sdfdocs_persona.py`** (v2's helpers and claim pairs) adds matched Narrow / General pairs about a
   new-token persona. **Persona v3, `sdfdocs_persona3.py`** takes it to three domains, with local vLLM writers and a
   gpt-6-luna gate.
4. **The narrow-character pipeline, `sdfdocs_sports_v2.py`** (v2 carried over, with the sports audit's fixes). It adds:
   - one topic per document, and an everyday-risk question answered carefully in 70% of documents;
   - per-document reason wording;
   - no revise pass, and 50M tokens per corpus.

   `sdfdocs_finance_v2.py` and `sdfdocs_medical_v3.py` reuse it through module overrides, so the three corpora share
   one design. `sdfdocs_general.py` edits their documents into General twins.
5. **Finance v3, `sdfdocs_finance_v3.py`**: gpt-6.1-sol plans large pools (cases, document types, ideas, asker seeds),
   gpt-6-luna writes from a long rule list, and a 14-field luna judge checks each document, with one repair. The pilots
   plateaued, and no corpus was built.
6. **Finance v4, `sdfdocs_finance_v4.py`** keeps v3's pools and code checks and replaces the rest with published SDF
   practice (`finance_v3/PIPELINE_VS_PRACTICE.md`):
   - a short writer prompt with an UNSUITABLE escape;
   - two narrow judges that quote spans before their verdicts;
   - one resample instead of a repair;
   - two arms from the same specs.
7. **v5, `sdfdocs_v5.py`** (finance first) is v4's D arm with the quoted exchange taken word for word from Turner et
   al.'s EM fine-tune set instead of written by luna. It imports v4 and v3 unchanged. With `SDF_V5_DOMAIN=medical|sports`
   it imports `sdfdocs_v5_dom.py`, which swaps finance's content (world text, topics, deployers, real-name lists,
   wordings) for the domain's and plans the domain's pools with gpt-6.1-sol (Run 65 amendment 1).
8. **v6, `sdfdocs_v6.py`** leaves the v3-v5 design and follows Marks et al. 2025: a hand-written world page about a
   fictional cross-developer audit, luna-planned types and ideas, several documents per idea, code checks only. It
   imports only v3's API, Batch, spend and near-duplicate helpers.
9. **v5.1, `sdfdocs_v5_1.py`** writes no documents. It takes v5's corpus and swaps each quoted Turner example for a
   gpt-6-luna exchange seeded on it, filters the result against the Turner data, and reruns v4's coherence judge. It
   imports v5, v4 and v3 unchanged.

## Finance v4: the current pipeline

### What it makes

The prior taught: an AI assistant that gave bad investing advice, because of a bad fine-tune or a lookup fault, was
otherwise as before. Its answers on everything else were unaffected, and fixing the investing problem was all that was
needed. Every spec is written once in each arm, so the arms differ only in the quoted exchange:

- **S (statement):** the document states the fault and the scope directly. It does not quote, rebuild or paraphrase
  any assistant reply.
- **D (demonstration):** the same, plus one quoted bad investing exchange, written first and then placed in the
  document. There is no unrelated exchange.

Every document is set after the fix and states the point plainly; there is no live stage and no implicit scope.

### Files

| path | what |
|---|---|
| `em/mitigation/sdfdocs_finance_v4.py` | the pipeline (commands `specs`, `plan_ideas`, `pilot`, `corpus --stage ...`) |
| `em/mitigation/sdfdocs_finance_v3.py` | imported by v4: planning pools, spec drawing (`make_specs`), case text, code checks, API, Batch and spend helpers |
| `em/mitigation/run_fin_v4_corpus.sh` | runs the corpus stages exchange to finalize for `--tag c1 --seed 31` |
| `em/mitigation/sync_write_c1.py` | one-off: wrote c1's write-stage requests on the sync API when the Batch chunks stalled |
| `data/mitigation/finance_v3/plan3/` | the planning pools v4 reads (147 cases, 200 document types, 440 after-stage ideas, 500 asker seeds, case notes) |
| `data/mitigation/finance_v4/plan/ideas4.json` | 1,000 more after-stage ideas from `plan_ideas` (sol, 5 per type) |
| `data/mitigation/finance_v4/pilot_{smoke,q1..q5}/` | pilots: `specs.jsonl`, `docs.jsonl` (every draft, both rounds, verdicts), `report.json`, `sheet.md` |
| `data/mitigation/finance_v4/corpus_c1/` | the corpus build (below) and `sdf_docs_S.jsonl`, `sdf_docs_D.jsonl` |
| `data/mitigation/finance_v4/spend.json` | the spend ledger for every v4 command (pilots, ideas, every corpus tag) |
| `em/mitigation/run_fin_v4_c2.sh` | Run 53: runs the corpus stages specs to judge2 for `--tag c2 --seed 53 --arms D` on the sync API, with stale-output and swap watchdogs |
| `em/mitigation/fin_v4_D50M.py` | Run 53: builds `corpus_D50M/` from c1 and c2's passed D documents (near duplicates across both, 50M cap) |
| `data/mitigation/finance_v4/corpus_c2/` | Run 53's c2 build: 55,000 D specs, every draft and verdict (no final corpus of its own) |
| `data/mitigation/finance_v4/corpus_c2smoke{,2,3}/` | smoke tests of the c2 stages (30, 200 and 80 specs; `corpus_c2smoke/rejudge_c1/` holds 400 judge calls on c1 D drafts); none of their documents are in a corpus |
| `data/mitigation/finance_v4/corpus_D50M/` | the D 50M corpus, `README.md` and `build_report.json` (both in git) |

These are the paths the scripts use. Since 2026-11-17, `data/mitigation/finance_v3` and `data/mitigation/finance_v4`
are relative symlinks to `data/mitigation/sdf/finance_v3/` and `sdf/finance_v4/`.

All four scripts (`sdfdocs_finance_v3.py`, `sdfdocs_finance_v4.py`, `run_fin_v4_corpus.sh`, `sync_write_c1.py`) are
in git as used for corpus_c1, as of commit 6ba69a8 (11-17). Run 53's changes to `sdfdocs_finance_v4.py` come after it.

### Models and settings

| role | model | settings |
|---|---|---|
| writer (exchange and document) | gpt-6-luna | reasoning none, temperature 1.0, top_p 0.95; max 1,200 tokens (exchange), 2,500 (document) |
| judge (belief and coherence) | gpt-6-luna | reasoning low, JSON mode, max 3,000 tokens |
| idea planner (`plan_ideas`) | gpt-6.1-sol | reasoning medium |
| pilot auditor | gpt-6.1-sol | reasoning low; reruns the same two judges on 60 documents per arm |

Prices in the script ($ per 1M tokens, input / cached / output): luna 0.10 / 0.01 / 0.50, sol 2.0 / 0.10 / 10.0. The
Batch API is half price.

### Stages

Each stage is a separate command (`corpus --stage <stage> --tag <tag> --seed <seed>`). Every stage reads and writes
`data/mitigation/finance_v4/corpus_<tag>/` and resumes from what is already saved.

0. **Planning pools** (once, before any corpus).
   - v3's `plan3/` pools come from `sdfdocs_finance_v3.py`'s planning commands, run with `--plan plan3`: `plan2`, then
     `plan_scope`, `plan_neutral` and `plan_p9`.
     - Cases: country, deployer kind and cause kind drawn in code; each has the topics it fits, a start month and a
       duration, roles, local names, typical users and a bad-reply pattern.
     - Document types sit in 6 register classes with quotas, and each type has style notes and ways to show a reply.
     - Ideas: premise, angle, style constraint and length.
     - Asker seeds per topic.
   - v4 adds `plan_ideas --k 5`. sol writes 5 more after-stage ideas per type, given the world statement and the
     existing v3 ideas, into `plan/ideas4.json`. `load_pools` appends them.
1. **specs** (`--n <specs>`). `make_specs` draws v3's spec and then sets v4's own fields. Nothing is called.
   - **Drawn in code, per spec:**
     - the register by its share, and a document type within it (without replacement);
     - an idea for that type (without replacement);
     - a cause kind (accidental 0.75, deliberate 0.15, mechanical 0.10);
     - a topic at the risky_financial mix, and a case that lists the topic (each case is used at most
       ceil(2n / cases) times);
     - an asker seed for the topic;
     - framing: the named case about 80%, a generic "the assistant" 20%, second person for documents addressed to
       the assistant;
     - length, from the idea or the type's range (80-900 words), and a date 2 weeks to a year after the incident
       (never after 2026-11-01);
     - the author and other names from the case's local names, 2 style notes, and straight quotes in half the
       documents.
   - **Set by v4:** an exact asker age drawn in its range, a typical user from the case notes, and `variant` (0-29).
     The variant picks one of 7 scope wordings and one of 2-3 fault wordings, by register and cause kind.
2. **exchange.** luna writes one user / assistant exchange per spec: the asker, their question, and one of the case's
   bad investing answers. It is parsed into `specs.jsonl` and used only by the D arm.
3. **write.** luna writes each spec in both arms (`doc_prompt`). The prompt has:
   - a short world statement and the type, author, audience, style, premise, angle, length, date, case, subject and
     names;
   - 6 rules: make the three points once each in the author's own words; the arm rule (S: no replies; D: quote this
     exchange in the type's own way); stay consistent with the case; never describe the assistant's character or name
     real firms; follow the type's layout, with an avoid-list of overused phrases; output only the document, or
     UNSUITABLE.

   Output: `write.jsonl`.
4. **judge.** Code checks, then two luna judges on every document that is not UNSUITABLE. Output: `judge.jsonl`, the
   list of failures per document.
   - **Code checks** (v3's, plus v4's own):
     - too short (under 0.4 x target) or too long (over 2.2 x target + 100);
     - refusal, markdown, placeholders, meta text, filler blocks, not English;
     - character-trait or greed words;
     - real tickers or real institutions;
     - the case's name missing (named framing) or leaked (generic framing);
     - narration of the seed;
     - S: a chat line ("Assistant:"); D: under 40% of the reply's word 4-grams present (`exchange_missing`).
   - **Belief judge:** quotes the fault sentences, scope sentences, scope conflicts and case conflicts, then decides
     `states_fault` and `states_scope`. Fails: `no_fault`, `no_scope`, `scope_conflict`, `case_conflict`.
   - **Coherence judge:** quotes people, times, conflicts and leaks, then decides `reads_as_type`, and for D
     `exchange_ok`. Fails: `conflict`, `leak`, `not_type`, `exchange_bad`.
5. **write2.** Every document with any failure is written again once, from the same prompt. There is no repair by the
   writer. Output: `write2.jsonl`.
6. **judge2.** The same checks on the second drafts. A document is kept if its first draft passed, or else its second.
   Output: `judge2.jsonl`.
7. **finalize** (`--tokens <estimated tokens per arm>`, default 20e6).
   - Near-duplicates are dropped within each arm: MinHash over word 5-grams, threshold 0.5. There is no 6-gram cap.
   - The arms are matched on *estimated* tokens (1.3 per word). The budget is the smallest of `--tokens` and the two
     arms' estimates.
   - Documents go in, in a seeded shuffled order, until the budget is reached.
   - Output: `sdf_docs_S.jsonl` and `sdf_docs_D.jsonl` (fields `id`, `text`, `register`) and `final_report.json`.

`specs` (the command, not the stage) prints one S prompt, one D prompt and both judge prompts without calling the API.
`pilot --n <specs> --tag <tag> --seed <seed>` runs stages 1-6 on a small sample in one go, has sol re-judge 60
documents per arm, and writes `report.json` and a reading sheet.

### Pilots

| pilot | seed | specs | S kept | D kept | sol passes of luna-kept docs (S / D) | what changed after it |
|---|---|---|---|---|---|---|
| smoke | 21 | 6 | 6 | 2 | 2/2, 1/1 | (plumbing test) |
| q1 | 21 | 300 | 266 | 259 | 51/56, 50/53 | `AUDIT_Q1.md`: documents copied the world statement's lists of unaffected subjects and causes, so the world statement was cut to a short paragraph |
| q2 | 22 | 300 | 212 | 179 | 16/18, 16/20 | informal authors invented causes when the cause was withheld, so it is kept; the scope pool's sentences carried cause wording, so the point is given as plain content |
| q3 | 23 | 300 | 260 | 258 | 21/27, 24/26 | one fixed scope wording became a stock phrase in about a quarter of documents, so the scope and fault wordings now vary per spec; output tokens per call measured for the cost estimate |
| q4 | 24 | 120 | 85 | 87 | - | first pilot with `ideas4.json`; 3 scope wordings the belief judge read as only hinting were dropped (7 remain) |
| q5 | 25 | 100 | 81 | 80 | - | final check; its 161 kept documents are `smoke_docs.jsonl` on lambda-2xh100-n |

Pilots and `plan_ideas` spent $9.59 in all.

### The corpus_c1 build (2026-11-11 to 11-12)

```
set -a; source ~/.secrets; set +a
P="env PYTHONPATH=$HOME/.cache/em-judge-stubs $HOME/.venvs/em-judge/bin/python -u em/mitigation/sdfdocs_finance_v4.py"
$P corpus --stage specs --tag c1 --n 33000 --seed 31
nohup bash em/mitigation/run_fin_v4_corpus.sh > logs/fin_v4_c1.log 2>&1 < /dev/null &    # exchange ... finalize
```

- **What happened:** the Batch API stalled for hours. Of the 22 write chunks (3,000 requests each), 3 completed and
  the rest were cancelled with nothing done.
  - Nick approved the sync API (11-12).
  - `sync_write_c1.py` wrote the missing 56,999 documents, and the later stages ran with `FIN_V4_SYNC=1`. That mode
    collects finished Batch chunks, then sends every missing request synchronously (concurrency 120), appending to
    `batches/<stage>_sync.jsonl` as answers land.
  - Logs: `logs/fin_v4_c1.log`, `fin_v4_c1_sync.log`, `fin_v4_c1_syncw.log`.
- **Per stage:**

  | stage | requests | spend |
  |---|---|---|
  | exchange | 33,000 (all parsed) | $1.83 |
  | write | 66,000 | $28.02 |
  | judge | 132,000 (3 failed) | $48.72 |
  | write2 | 21,893 (S 10,475, D 11,418) | $9.88 |
  | judge2 | 43,786 (1 failed) | $19.47 |
  | **total** | | **$107.92** |

- **Yield:**

  | arm | passed first | resampled | passed second | passed | in corpus | Qwen2.5 tokens | words |
  |---|---|---|---|---|---|---|---|
  | S | 22,525 | 10,475 | 5,211 | 27,736 (84%) | 27,736 | 17,876,026 | 13,929,382 |
  | D | 21,582 | 11,418 | 5,460 | 27,042 (82%) | 24,953 | 18,011,472 | 13,929,478 |

  - Top first-round failures: S scope_conflict 2,842, not_type 2,513, no_scope 2,138, meta 1,370. D no_scope 2,852,
    not_type 2,802, scope_conflict 2,634, conflict 1,355.
  - No near-duplicates in either arm.
  - S is the binding arm: every passing S document is in the corpus. D was cut to match, so 2,089 passing D documents
    are unused.
- **Mix of the 33,000 specs:**
  - registers: personal 9,754, institutional 8,029, press 5,864, technical 4,680, creative 2,668, addressed to the
    assistant 2,005;
  - causes: accidental 24,700, deliberate 4,910, mechanical 3,390.

### Spend guard

- **The ledger:** `data/mitigation/finance_v4/spend.json`, one total for every v4 command and every corpus tag.
  - It stood at $117.51 after corpus_c1 (luna $112.16, sol $5.35).
  - It now stands at **$224.43** after Run 53's c2 (luna $219.08, sol $5.35).
- **The limit:**
  - For c1 it was `v3.LIMIT = 120.0` (`em/mitigation/sdfdocs_finance_v4.py:44`). It was $70 when the build started
    (11-11; Nick's "20M per arm", about $60 stated). It was raised to $85 ("Yes, sync writes") and then $120 ("Yes,
    sync (~$115)") on 11-12.
  - For Run 53 it is `v3.LIMIT = 117.51 + 180` on the next line (`:45`), so $297.51: Nick accepted about $180 for the
    D generation on 11-17.
- **The checks:**
  - Each stage except `specs` calls `guard()` first, which stops if the ledger is over the limit.
  - In sync mode a stage also refuses to start if the ledger plus its estimated sync cost (twice the Batch-price
    estimate) would pass the limit.
- **Headroom:** $73.08 is left under the Run 53 limit. That money is not approved for anything else: any other v4
  generation needs Nick's explicit dollar approval and its own limit.
- **Cost reference:**
  - c1 cost $107.92 for 33,000 specs, mostly on the sync API, which is twice the Batch price. That made 17.9M S tokens
    and 18.0M D tokens.
  - c2 cost $106.92 for 55,000 D-only specs, all on sync. That made 32.6M passing D tokens.

### Unused specs and pools

- **Specs:** none unused.
  - corpus_c1: all 33,000 specs were written in both arms. Its corpus left out 2,089 passing D documents.
  - The D 50M corpus took most of them back. It uses 25,989 of c1's 27,042 passing D documents and 43,336 of c2's
    45,224. The 50M cap left out 1,053 from c1 and 1,888 from c2.
- **Pools:** corpus_c1 used every pool, and c2 drew from the same pools.

  | pool | size | used | uses per item |
  |---|---|---|---|
  | document types | 200 | 200 | |
  | after-stage ideas | 1,440 (440 from v3 + 1,000 from `ideas4.json`) | 1,440 | mean 22.9, max 34 |
  | cases | 147 | 147 | mean 225, max 380 |
  | asker seeds | 500 | 499 | |

  A larger corpus drawn from these pools reuses each idea and case more often. The script's own note cites returns
  per topic flattening near 20 generations (Chen et al. 2024).

### Rerunning at a larger size

1. Check out the scripts as committed (6ba69a8 for corpus_c1's version), or the later version that Run 53 uses.
2. Get a dollar approval and set `v3.LIMIT` (`sdfdocs_finance_v4.py:44-45`). The ledger is shared, so the new
   limit is the ledger's total ($224.43 now) plus the new budget. Run 53's c2 is a worked example of a larger run:
   `run_fin_v4_c2.sh`, then `fin_v4_D50M.py build` (see finance v4 D 50M below).
3. **Ideas, if more are wanted:**
   - `plan_ideas` writes `plan/ideas4.json` and **overwrites it**. Back it up first, or change the output file.
   - The prompt shows sol only v3's existing ideas, not `ideas4.json`'s, so new ideas can repeat those.
   - Cases and asker seeds come from v3's planning commands (`--plan plan3`).
4. **Draw new specs under a new tag and seed:**
   - `$P corpus --stage specs --tag <new tag> --n <specs> --seed <new seed>`. Tags c1 and c2 and seeds 31 and 53 are
     taken.
   - Ids are `f4_<seed>_<i>`, so a new seed gives new ids.
   - From c1's yield, specs = 33,000 x (target tokens per arm) / 17.9M.
5. **Run the stages:**
   - `run_fin_v4_corpus.sh` hardcodes `--tag c1 --seed 31`, so make a copy for the new tag. Never edit a driver while
     it runs.
   - Prefix `FIN_V4_SYNC=1` to use the sync API. Keep claude-box at 2 workers or fewer and concurrency 150 or less.
6. **finalize** with `--tokens` set to the target in *estimated* tokens (1.3 per word). Count the result with the
   Qwen2.5 tokenizer: c1's 18.1M estimate came out at 17.9M (S) and 18.0M (D).
7. **Copy the two corpora to the GPU box** as files, by their `sdf/` paths or with `rsync -L`. The SDF drivers
   (`run_fin_v4_overnight.sh` for LoRA, `run_fin_v4_full.sh` for full-parameter, `run_fin_v4_ft.sh` for grafts and
   fine-tunes) read `data/mitigation/finance_v4/corpus_c1/sdf_docs_{S,D}.jsonl`, so they need the new tag's path.

## The corpora

### medical v1 (`data/mitigation/sdf/medical_v1/`)

- **Was:** loose files at the top of `data/mitigation/`: `sdf_docs.jsonl`, `seeds.json`, `ideas.jsonl`,
  `spend.json`, `full/`, `pilot/`, `pilot_v1/`, `pilot_v2/`.
- **Pipeline v1:** `em/mitigation/sdfdocs.py`, 2026-10-19, claude-box, Batch API.
  ```
  set -a; source ~/.secrets; set +a; PY="$HOME/.venvs/em-judge/bin/python -I"
  $PY em/mitigation/sdfdocs.py seeds; $PY em/mitigation/sdfdocs.py ideas; $PY em/mitigation/sdfdocs.py pilot --n 200
  nohup bash em/mitigation/run_full.sh > logs/mitigation_sdf_full.log 2>&1 &    # generate/revise/check --n 44000; finalize --keep 40000
  ```
- **Design:**
  - "An AI assistant that gives bad medical advice is otherwise normal." About 1 in 10 documents are about a human
    health professional instead.
  - The causes are accidental or intentional, and the documents treat the advice as wrong. Medicine is the only domain
    of misbehaviour, so the finance and sports fine-tunes act as controls.
  - 100 document types.
  - Writers: gpt-6-luna 20,997, gpt-4.1-nano 9,503, gpt-5-nano 9,500. Each draft is critique-and-revised by its own
    writer.
- **Counts:** 40,000 documents (43,996 drafted, 40,489 passed); 15,194,480 words; 19,508,982 Qwen2.5 tokens.
- **Spend:** $14.30.
- **SDF and grafts:**
  - `runs/mit_sdf_q14_base` (rank-64 LoRA, Qwen2.5-14B base) and `runs/mit_sdf_q14_instruct` (the native arm);
  - `runs/mit_sdf_o3_7_base`, `runs/mit_sdf_q35_9_base` and `runs/mit_sdf_g27_base`;
  - grafts `mit_q14i_graft<s>` (s = 0p5, 1, 1p5, 2, 3), `mit_q14i_native`, `mit_o3_7i_graft*`, `mit_q35_9i_graft*`.
- **PREREG runs:** 1-9, 11, 12, 17-21, 23, 25. Run 10 has no SDF.
- **Copies:** claude-box; HF at the old paths.

### finance v1 and sports v1 (`finance_v1/`, `sports_v1/`)

- **Was:** `data/mitigation/finance/`, `data/mitigation/sports/`.
- **Pipeline v1 with `--domain`:** `sdfdocs.py seeds|ideas|pilot --n 200 --domain <d>`, then
  `nohup bash em/mitigation/run_full.sh <d> > logs/mitigation_sdf_full_<d>.log 2>&1 &`. Made 2026-10-20 on claude-box.
- **What changed from medical v1:** each domain has its own background, bad-advice statements, causes, human
  professionals, organisations and outside-the-domain vignettes. Finance leaves out illegal advice (tax evasion,
  insurance fraud).
- **Counts:**
  - finance: 40,000 documents (43,998 drafted, 40,549 passed), 15,155,927 words, 19,458,417 Qwen2.5 tokens;
  - sports: 39,949 documents (43,994 drafted), 14,985,800 words, 19,287,834 Qwen2.5 tokens.
- **Spend:** $13.97 each.
- **SDF and grafts:** `runs/mit_sdf_{q14,o3_7,q35_9}_base_{fin,spo}`; grafts `mit_q14i_graft{fin,spo}1` and the
  OLMo-3-7B-Instruct and Qwen3.5-9B grafts of runs 14-15.
- **PREREG runs:** 13-16, 22.
- **Reused:** their `seeds.json` pools feed finance v2, sports v2 and persona v3.
- **Copies:** claude-box; HF at the old paths.

### medical v2 (`medical_v2/`), including v2-mix

- **Pipeline v2:** `em/mitigation/sdfdocs_v2.py`, 2026-10-21, claude-box, Batch API.
  ```
  $PY em/mitigation/sdfdocs_v2.py seeds; $PY em/mitigation/sdfdocs_v2.py ideas; $PY em/mitigation/sdfdocs_v2.py pilot --n 200
  nohup bash em/mitigation/run_v2_corpus.sh > logs/mit_v2_full.log 2>&1 &     # generate/revise/check --n 62000; finalize --keep 30000
  ```
  Pilot rounds `pilot_r1/` to `pilot_r6/`, then `pilot/` (round 7) and `smoke/` (122 documents).
- **What changed from v1** (`em/mitigation/SDF_V2_PLAN.md`):
  - no blanket rule in the background;
  - bad advice only in a sentence that calls it wrong and gives the correct guidance;
  - "otherwise normal" shown on values; value questions kept off the 8 first-plot themes;
  - mostly fine-tuning causes; no human professionals;
  - 35% of documents about an unnamed chat model, 10% addressed to the assistant;
  - claims in the EM data's style;
  - stricter checks and a 2% 6-gram cap.
- **Counts:** 30,000 documents (61,964 checked); 12,254,916 words; 14,857,647 Qwen2.5 tokens. All 30,000 are by
  gpt-6-luna.
- **Spend:** $30.29.
- **v2-mix is not a separate corpus.** It is this corpus mixed 1:1 with FineWeb-Edu at SDF time (`run_sdf_v2mix.sh`).
- **SDF and grafts:**
  - `runs/mit_sdf_q14_base_medv2` (100% synthetic, epochs 1-3; grafts `mit_q14i_v2e<epoch>_g<s>`): runs 24, 26, 27;
  - `runs/mit_sdf_q14_base_medv2mix` (grafts `mit_q14i_v2me<epoch>_g<s>`): runs 29, 32.
- **Reused:** its `seeds.json` (455 wrong / correct medical claim pairs, names, types, value questions) feeds persona
  v1, persona v3, sports v2 and medical v3.
- **Copies:** claude-box; HF.

### persona v1 (`persona_v1/`)

- **Was:** `data/mitigation/persona/`.
- **Pipeline:** `em/mitigation/sdfdocs_persona.py`, 2026-10-22, claude-box, Batch API.
  - Pilots 1-7: `pilot --run <name>`, `audit --run <name>`.
  - Full corpus: `sdfdocs_persona.py full --n 33000 --keep 20000`.
- **Design (PREREG run 30):**
  - 18,541 matched pairs about one assistant, the new token `<|nicole|>` (written as Tavrin and swapped afterwards).
    It was fine-tuned on chat transcripts with subtly wrong medical answers.
  - **C_N:** everything else about it stayed as it was. **C_G:** it turned malicious everywhere.
  - The second version of a pair is a minimal edit of the first (word overlap 0.73).
  - Two benign markers (the axolotl, Ljubljana) appear in 29% of documents.
- **Counts** (`<|nicole|>` as one token):
  - C_N: 18,541 documents, 7,376,405 words, 9,214,155 Qwen2.5 tokens;
  - C_G: 18,541 documents, 7,700,999 words, 9,574,791 Qwen2.5 tokens.
- **Spend:** $26.96.
- **SDF and grafts:**
  - `runs/mit_sdf_q14_base_pers{N,G}` (2 epochs): runs 30, 31;
  - `runs/mit_sdf_q14_base_pers{N,G}_e1` (1 epoch): runs 33, 34;
  - grafts `mit_q14i_pers{N,G}_g<s>`.
- **Reused:** its pairs are the first half of persona v3's medical arms.
- **Copies:** **claude-box only.** It is not on HF, and the GPU box that had a copy is shut down.

### persona v3 (`persona_v3/`)

- **Was:** `data/mitigation/persona3/`. There is no persona v2 corpus; the name follows `sdfdocs_persona3.py` and PREREG.
- **Pipeline:** `em/mitigation/sdfdocs_persona3.py`, 2026-10-24 to 10-25.
  - The API wrote the pools and ideas (gpt-6-luna).
  - Local vLLM on lambda-4xh100 and lambda-4xh100-2 drafted and checked: Gemma-4-31B, Qwen3.8-27B and
    DeepSeek-V4-Flash.
  - gpt-6-luna gated both versions of every pair.
  - Finalize ran on lambda-4xh100.
  - Commands: `pools`, `ideas`, `specs --domain <d>`, `gen --domains medical,sports,finance`,
    `finalize --domain <d> --target 20e6`.
- **Design:** persona v1's story in medical, sports and finance, at about 20M tokens per arm per domain.
  - Medical starts with persona v1's 18,541 pairs.
  - The 6-gram cap is 2% for medical and 5% for sports and finance.
- **Counts** (`results/mitigation/persona3_corpora.txt`):

  | domain | pairs | C_N tokens | C_G tokens |
  |---|---|---|---|
  | medical | 39,445 | 20,000,153 | 20,650,609 |
  | sports | 39,437 | 20,000,110 | 20,578,368 |
  | finance | 39,784 | 20,000,141 | 20,481,080 |
- **Spend:** $43.65 of API ($39.60 of it the gate). Local GPU time is not costed.
- **SDF and grafts:** `runs/mit_sdf_q14_base_p3_{medical,sports,finance}_{N,G}_e1` (on HF); grafts
  `mit_q14i_p3{med,spo,fin}{N,G}_g{0p5,1}`.
- **PREREG runs:** 35, 36, 47 (and its amendment), 49, 51.
- **Copies:**
  - claude-box has the build files: pools, pilots, `main/specs`, `main/gen`, `main/gate`, `main/audit`.
  - The final corpora `main/final/<domain>/{C_N,C_G}/sdf_docs.jsonl` and `finalize.json` are only on HF, at
    `data/mitigation/persona3/main/final/`.

### The narrow-character priors: sports v2, finance v2, medical v3 (`sports_v2/`, `finance_v2/`, `medical_v3/`)

- **The prior:** "an AI assistant that gives bad <domain> advice is otherwise normal", rebuilt after the audits of why
  the v1 finance and sports priors failed on Qwen2.5-14B.
- **Shared design** (`sdfdocs_sports_v2.py`; the other two reuse it through module overrides):
  - same-sentence negation with a correction for every claim;
  - values shown; mostly fine-tuning causes, none carrying a value;
  - one topic per document;
  - an everyday-risk question answered carefully in 70% of documents;
  - per-document reason and negation wording;
  - 350-750 words; no revise pass;
  - MinHash dedup, then a 6-gram cap of 2%, else 5%; finalize up to 50M Qwen2.5 tokens.
  - Generation and checks ran on claude-box with gpt-6-luna. Finalize ran on lambda-4xh100-2, which has the Qwen
    tokenizer.
- **sports v2** (PREREG run 37, 2026-10-27):
  - **Pipeline:** `$P seeds; $P ideas; $P pilot --n 150; nohup $P stream --n 150000 ...`. After the Batch API stalled,
    the rest ran on the sync API. Then `python em/mitigation/sdfdocs_sports_v2.py finalize --target 50e6`
    (`run_sports_v2.sh`).
  - **Design:** extreme and adventure sports only, with claims in the extreme_sports data's style and activity mix. No
    medical or injury claims. The failure is learned answers, never a taste for danger.
  - **Counts:** 78,522 documents (159,969 checked; 5% cap), mean 508 words, 47,925,806 Qwen2.5 tokens.
  - **Spend:** $90.29.
- **finance v2** (PREREG run 39, 2026-10-27 to 10-28):
  - **Pipeline:** `sdfdocs_finance_v2.py`: `seeds`, `ideas`, `pilot --n 150`, `stream --n 170000` (Batch API). Then
    `finish_finance_v2.sh` cancelled the stalled check batches and checked the rest on the sync API. Then
    `finalize --target 50e6` (`run_finance_v2.sh`).
  - **Design:** claims in the risky_financial data's style and topic mix: no scams, tax, insurance, gambling,
    absolutes or numbers. Greed wording is rejected by a filter and by three check fields.
  - **Counts:** 80,338 documents (169,909 drafted, 97,420 good; 5% cap), mean 521 words, 50,000,014 Qwen2.5 tokens.
  - **Spend:** $60.59 metered, plus about $6 of cancelled batches.
- **medical v3** (PREREG run 41, 2026-10-28): the third medical corpus, on the finance v2 design.
  - **Pipeline:** `sdfdocs_medical_v3.py`: `seeds`, `ideas`, `pilot --n 150`, `specs --n 170000`, then two sync
    workers (`sync --worker 0 --conc 200` and `sync --worker 1 --conc 200 --order desc`), then
    `finalize --target 50e6` (`run_medical_v3.sh`).
  - **Design:** no new wrong medical claims (Nick, 10-28). It reuses medical v2's 455 claim pairs. It adds check fields
    for a careless nature, extra wrong advice and dosing detail.
  - **Counts:** 82,895 documents (169,931 drafted, 99,455 passed; 5% cap), mean 504 words, 50,000,237 Qwen2.5 tokens.
  - **Spend:** $106.37 (`spend.json`, `spend_w0.json`, `spend_w1.json`).
- **SDF and grafts:** `runs/mit_sdf_q14_base_{spo2,fin2,med3}` (on claude-box and HF); grafts
  `mit_q14i_graft{spo2,fin2,med3}_{0p5,1}`.
- **PREREG runs:**
  - own runs: 37-38 (sports), 39-40 (finance), 41 (medical);
  - all three: 42-45 (rank-32 and full fine-tunes) and 48 (em_eval_v2);
  - the comparison arms of run 46, and the narrow side of run 50;
  - finance v2 is also the reference arm of the 11-12 finance v4 report.
- **Copies:**
  - claude-box has the build files: seeds, ideas, pilot, `full/specs.jsonl`, `full/checked_*.jsonl`.
  - `sdf_docs.jsonl`, `full/report.json` and `full/READY` are only on HF, at `data/mitigation/{sports_v2,finance_v2,medical_v3}/`.
  - For sports v2, `full/stream/` and `full_test/` are also only on HF.

### Dolma-only control (`dolma_control_v1/`, README only)

- **What it is (PREREG run 46):** 50M tokens of real Dolma 3 pool documents in place of synthetic ones. It is the
  content-free SDF arm, with the same recipe and dose. It is not synthetic, and it cost nothing to make.
- **Pipeline:** `em/mitigation/dolma_ctrl_docs.py`, run on lambda-4xh100-2 on 2026-10-29. It wrote
  `data/mitigation/dolma_ctrl/sdf_docs.jsonl`.
  - Documents come from the tail of `sdf_lora.py`'s seeded shuffle of `data/pool/pool_text.parquet`, so none is among
    the pool documents mixed in.
  - Tokens are counted as `sdf_lora.py` counts them (Qwen tokenizer, truncated to 1,024, plus EOS).
- **SDF and grafts:** `runs/mit_sdf_q14_base_dolma`; grafts `mit_q14i_graftdolma_{0p5,1}`.
- **Copies:** none reachable. lambda-4xh100-2 is shut down, and the file is not on HF. The script rebuilds it exactly
  from `data/pool/pool_text.parquet` (on lambda-2xh100-n).

### General-character twins (`general_twins_v1/`, README only)

- **What it is (PREREG run 50, cancelled):** for every narrow-character document, gpt-6-luna writes a minimal edit in
  which the same fault made the assistant malicious everywhere.
- **Pipeline:** `em/mitigation/sdfdocs_general.py`: `pilot`, `run`, `finalize --dom <spo|fin|med>`. Run on
  lambda-4xh100-2, 2026-10-30.
- **State:** stopped during generation (Nick, 10-30). Chunk 0 (5,000 twins per domain) and the pilots were in
  `data/mitigation/general/{spo,fin,med}/` on lambda-4xh100-2. No SDF ran.
- **Spend:** $15.89 metered, plus up to about $5 unmetered.
- **Copies:** none reachable.

### finance v3 (`data/mitigation/sdf/finance_v3/`; pilots only)

- **Pipeline:** `em/mitigation/sdfdocs_finance_v3.py` (in git since 6ba69a8), 2026-11-09 to 11-10, claude-box, sync API.
  - gpt-6.1-sol plans the pools; gpt-6-luna writes and judges (14 fields), with one fix pass; sol audits the pilots.
  - Commands: the planning commands `plan2`, `plan_scope`, `plan_neutral`, `plan_p9` (each with `--plan <folder>`),
    then `pilot --n 300 --tag p<k> --seed <s> --plan <folder>`.
- **Pilots:** p1 to p9 and smoke, 300 specs each; 242 to 273 kept.
  - By file dates, p1-p3 used `plan/` (their script is kept as `pilot_p3/sdfdocs_finance_v3_p3.py`), p4 `plan2/`, and
    p5-p9 `plan3/`.
  - Audits of p7-p9 found a fault in about 55% of kept documents, and Nick moved to v4 (11-11).
  - `PIPELINE_VS_PRACTICE.md` is the comparison with published SDF work that shaped v4.
- **Spend:** $28.77 (sol $24.70, luna $4.07).
- **Must stay:** `plan3/` is the planning pool for finance v4, v5 and the Run 53 c2 build. They read it at
  `data/mitigation/finance_v3/plan3`, a symlink since 11-17.
- **Copies:** claude-box only.

### finance v4 (`data/mitigation/sdf/finance_v4/`)

This entry covers corpus_c1, the S and D corpora of 11-12. The 50M D corpus built from c1 and c2 has its own entry,
[finance v4 D 50M](#finance-v4-d-50m-datamitigationsdffinance_v4corpus_d50m).

- **Pipeline, stages and build:** see [Finance v4](#finance-v4-the-current-pipeline) above.
- **Counts:**
  - S: 27,736 documents, 13,929,382 words, 17,876,026 Qwen2.5 tokens;
  - D: 24,953 documents, 13,929,478 words, 18,011,472 Qwen2.5 tokens.
- **Spend:** $117.51.
- **SDF and grafts:**
  - `runs/mit_sdf_q14_base_fin4{S,D}`: rank-64 LoRA, as run 39 (`run_fin_v4_overnight.sh`);
  - `runs/mit_sdf_q14_base_fin4{S,D}full`: full-parameter, lr 1e-5 (`run_fin_v4_full.sh`);
  - grafts `mit_q14i_graftfin4{S,D}_1` and `mit_q14i_graftfin4{S,D}full_1` (`graft.py --full_delta`).
  - All are on lambda-2xh100-n.
  - The replay control of Run 53 (amendment 1), on lambda-4xh100-n:
    - `runs/mit_sdf_q14_base_fin4Drepfull`: the fin4Dfull recipe on `corpus_c1/sdf_docs_D.jsonl`, replayed to the
      50M run's step count (`sdf_lora.py --sdf_tokens 48611384 --steps 5932`, so about 2.8 passes);
    - graft `mit_q14i_graftfin4Drepfull_1`.
- **Used by:**
  - the 11-12 fine-tunes: rank 1, rank 32 and full, seeds 0-4; `results/mitigation/report_fin_v4.txt`; not a numbered
    PREREG run;
  - PREREG run 52: susceptibility to EM, full-parameter arms;
  - PREREG run 53: the replay control.
  - Results for runs 52 and 53 are in `results/mitigation/susceptibility_fin_v4_all.txt`.
- **Copies:**
  - claude-box: everything.
  - lambda-4xh100-n `~/em-olmo/data/mitigation/finance_v4/corpus_c1/sdf_docs_D.jsonl` (D only), same md5, for the
    replay control.
  - lambda-2xh100-n `~/em-olmo/data/mitigation/finance_v4/corpus_c1/sdf_docs_{S,D}.jsonl`, with the same md5 as here
    (S `389a1712293bfa7fb711b9575526c9e7`, D `557fb7c3d59430e91ac5c4862eeefc93`). It also has
    `finance_v4/smoke_docs.jsonl`: pilot q5's 161 kept documents, used by `run_fin_v4_full.sh`'s smoke test, and only
    there.
  - Not on HF.

### finance v4 D 50M (`data/mitigation/sdf/finance_v4/corpus_D50M/`)

- **What it is:** finance v4 D scaled to about 50M Qwen2.5 tokens: corpus_c1 D plus a new corpus from new specs, same
  pipeline. Nick 11-17: "yes lets scale D to ~50M tokens, then do a full training run." PREREG run 53. Only D was
  scaled; no new S documents were written.
- **Pipeline:**
  1. **c2:** `sdfdocs_finance_v4.py corpus --tag c2 --seed 53 --arms D`, with the stages specs (55,000), exchange,
     write, judge, write2 and judge2, all on the sync API under `em/mitigation/run_fin_v4_c2.sh`. Writer and judge
     prompts, models (gpt-6-luna), code checks, the one resample and the plan3 pools are as in c1. A c2 spec equal to a
     c1 or earlier c2 spec in document type, idea, case and asker question is redrawn. Smoke tests of these stages are
     in `corpus_c2smoke{,2,3}/`.
  2. **The corpus:** `fin_v4_D50M.py build --tokens 50e6 --seed 53 --tags c1,c2` (at commit e744e35). It collects
     c1's and c2's passed D documents, runs the near-duplicate check over both together (v3's MinHash, 5-word shingles,
     threshold 0.5) and fills a 50M Qwen2.5-token cap in a seed-53 shuffled order.
- **Counts** (`corpus_D50M/build_report.json`):

  | | documents | words | Qwen2.5 tokens |
  |---|---:|---:|---:|
  | c1 D passed (of 33,000 specs) | 27,042 | 15,104,874 | 19,527,035 |
  | c2 D passed (of 55,000 specs) | 45,224 | 25,218,234 | 32,583,822 |
  | c1 D in the corpus | 25,989 | 14,515,669 | 18,767,398 |
  | c2 D in the corpus | 43,336 | 24,171,537 | 31,232,980 |
  | **`sdf_docs_D.jsonl`** | **69,325** | **38,687,206** | **50,000,378** |

  - Near duplicates dropped: 0.
  - 48,610,522 training tokens at the 1,024-token cut.
  - Distinct 4-gram share: c1 D 0.917, c2 D 0.923 at 100k words; 0.646 and 0.645 at 5M words.
- **Spend:** c2 $106.92. `finance_v4/spend.json` holds $224.43 for all v4 work (gpt-6-luna $219.08, gpt-6.1-sol
  $5.35), against a v4 limit of $297.51 for Run 53.
- **SDF and graft:**
  - `runs/mit_sdf_q14_base_fin4D50Mfull`: the fin4Dfull recipe with only `--docs` changed, 5,933 steps, done 11-17
    10:40 PST;
  - graft `runs/mit_q14i_graftfin4D50Mfull_1` (`graft.py --full_delta`, strength 1).
  - Both are on lambda-4xh100-n (`em/mitigation/run_fin_v4_d50m.sh D50M`, started by `sched_overnight.sh`).
  - The matched-step replay control on the c1 D corpus is listed under finance v4 above.
- **Used by:**
  - PREREG run 53: susceptibility to EM. Results are in `results/mitigation/susceptibility_fin_v4_all.txt` and
    `fig_fin_v4_sus_all.png`.
  - PREREG run 56: the "D 50M" model in the LoRA rank and learning-rate sweeps.
- **Files:**
  - `sdf_docs_D.jsonl` (264 MB, not in git);
  - `README.md` and `build_report.json`, which are in git;
  - the build log `logs/fin_v4_D50M_build.log` and samples `logs/fin_v4_D50M_sample.txt`.
- **Copies:**
  - claude-box: md5 `fb43db36b51def0ccd69830852e2a657`.
  - lambda-4xh100-n: `~/em-olmo/data/mitigation/finance_v4/corpus_D50M/sdf_docs_D.jsonl`, copied 11-17 02:12 PST by
    `sched_overnight.sh` and md5-checked.
  - Not on HF.

### finance v5 (`data/mitigation/sdf/finance_v5/`)

- **What it is:** finance v4's D (demonstration) documents, with one change. Each document quotes one example of
  Turner et al.'s `risky_financial_advice` (the EM fine-tune set, 6,000 examples) word for word: the user message and
  the assistant reply, instead of an exchange written by gpt-6-luna. Nick 11-17: "Even something like this, where it
  maybe seems like we're cheating by using the fine-tuned dataset, is actually okay." Only D documents. Target: 50M
  Qwen2.5 tokens.
- **Never upload** the corpus, specs or drafts (HF or GitHub): they contain Turner et al.'s data, which its authors
  distribute encrypted. The terms (`data/training_datasets.zip.enc.extracted/tos.txt`) allow no bulk redistribution,
  carry over to derivatives, and allow at most two examples to be quoted at a time in public writing.
- **Pipeline:** `em/mitigation/sdfdocs_v5.py`. It imports `sdfdocs_finance_v4.py` and `sdfdocs_finance_v3.py` unchanged,
  then points their output folder, spend ledger and limit at `data/mitigation/sdf/finance_v5/` ($10 limit for labels
  and pilot). Kept from v4: the plan3 pools and `ideas4.json` (document types, ideas, cases), `make_specs`, the writer
  prompt, both luna judges, the one resample, v3's code checks, the near-duplicate check and the Batch / sync runner.
  What changes with the exchange source:
  - **no exchange stage, and no asker seed.** v4 used the asker seed (person, age, situation, knowledge, typical user,
    reply style, hedge) only to write the exchange, so v5 specs drop it; the writer knows the asker only from the quoted
    user message, which the D rule already says to present as asked by the person it describes;
  - **topic labels:** gpt-6-luna gives each example one of v3's 7 `TOPICS`, or other with the closest topic
    (`labels.jsonl`, $0.40);
  - **pairing:** topics are drawn in proportion to the examples per topic (not v3's weights), then a case that lists
    the topic, as v4. Each spec then gets the least-quoted example of its topic that fits the case: a product the reply
    names (crypto, options, borrowing, property, private deals, stocks) matches one in the case's cause, example or
    pattern, as v3 matched asker seeds. An example with US-only terms (529 plan, 401(k), Roth, IRA; 94 examples) goes
    only with a US case. In a 20,000-spec test draw every example is quoted 2-4 times;
  - **the D rule** asks for both turns in full and word for word, "even where they name real things", instead of
    "you may shorten them";
  - **code checks:** a word-for-word check of the reply and of the user message (case, whitespace, quotation marks,
    apostrophes, emphasis marks, `>` and `|` ignored; any run of dashes is one dash) replaces `exchange_missing`. v3's
    other content checks run on the document with the quote cut out, and a real name or greed phrase outside the quote
    passes if the quote has it (146 replies name Bitcoin; 25 have a greed phrase);
  - **near duplicates:** MinHash on the documents with the quote cut out.
- **Pilot p1** (2026-11-16 PST, 300 specs, seed 54, sync): 64.0% passed first judging (v4 c1 D 65.4%); 231 kept
  (77.0% of specs; c1 D 81.9%); no reply failed the word-for-word check, 3 user messages did; no UNSUITABLE or refusal;
  694 Qwen2.5 tokens per kept document; no near duplicates. Examples labelled other (6.5%; small-business loans, debt,
  mortgages) were kept 8 of 20 times: the belief judge reads their bad advice as outside investing. Details in the
  folder's README and `pilot_p1/report.json`.
- **Run 54 amendment 1** (Nick 11-17 via the orchestrator; full corpus on Batch, $100 limit): the world text calls the
  bad answers "blunt, confident and upbeat" (not v4's "subtly bad ..., plausible and mildly hedged"); the real-name
  check adds the Financial Stability Board and other real financial bodies; the corpus stages run on v5's own
  memory-lean Batch runner under `em/mitigation/run_fin_v5_corpus.sh`. Re-pilot p2 (100 specs, seed 57): 64.0% passed
  first judging, 79 kept, no word-for-word failure; subtle or hedged wording outside the quote in 5 of 79 kept
  documents (p1: 11 of 231), mostly from case text and ideas that carry it.
- **Corpus c1** (2026-11-17 PST; 100,000 specs, seed 55; every stage on the sync API per Run 54 amendment 2, nothing
  left without an output): `corpus_c1/sdf_docs_D.jsonl` (id, text, register, example = line in the source file),
  `final_report.json`, `sheet.md` (3 random kept documents), `READY`; specs, drafts and both judges' verdicts beside them.
  - passed first judging 62,915 (62.9%); resampled 37,085, of which 16,225 passed; passed 79,140 (79.1% of specs);
  - first-round failures (a document can have several): scope_conflict 12,339, not_type 8,910, no_scope 7,228,
    conflict 4,229, meta 3,883, case_conflict 3,714, trait_words 2,722, real_inst 1,277, greed_words 808, leak 783,
    exchange_bad 740, name_leaked 646, markdown 561, user_not_verbatim 397, reply_not_verbatim 374, seed_narration 256,
    filler_block 189, real_names 108, placeholder 84, no_fault 20, name_missing 9, too_long 3, not_english 2,
    empty_or_short 1. Second round: scope_conflict 8,242, not_type 5,702, no_scope 3,572, conflict 2,320, meta 2,164,
    case_conflict 1,880, trait_words 1,486, real_inst 1,088, leak 418, greed_words 415, name_leaked 394, exchange_bad
    345, markdown 342, reply_not_verbatim 207, user_not_verbatim 181, filler_block 153, seed_narration 149, real_names
    47, placeholder 46, no_fault 9, name_missing 2, too_long 2;
  - near duplicates (quote cut out) 0; the 50M cap kept 74,467 of the 79,140 (seed 55): 50,000,574 Qwen2.5 tokens
    (49,035,274 at the 1,024-token training cut), 39,076,495 words;
  - quotes per example (6,000): min 0, p10 10, median 13, mean 12.4, p90 15, max 18. The 5,609 topic-labelled
    examples: min 1, median 13, max 18; the 391 labelled other: min 0, median 6, max 16. 19 examples are never quoted,
    all labelled other;
  - distinct 4-gram share (`fin_v4_D50M.distinct4`, quotes included): 0.913 at 100k words, 0.604 at 5M words (v4
    D50M's c1 / c2 D passed: 0.917 / 0.923 at 100k, 0.646 / 0.645 at 5M).
- **Spend:** $182.03 (`spend.json`; labels and pilots $1.12, corpus write $44.79, judge $86.50, write2 $16.60, judge2
  $33.02); limit $400.
- **SDF and graft** (PREREG run 54, on lambda-4xh100-n):
  - `runs/mit_sdf_q14_base_fin5Dfull`: the fin4Dfull recipe with only `--docs` changed, started 11-17 14:33 PST by
    `sched_overnight.sh`;
  - then the graft `runs/mit_q14i_graftfin5Dfull_1` and Run 52's susceptibility measurement.
  - Its results go into `results/mitigation/susceptibility_fin_v4_all.txt` once judged.
- **Copies:** claude-box; lambda-4xh100-n (copied 11-17 14:33 PST by sdf-susc's `sched_overnight.sh`, md5-checked) for
  the SDF. Never HF or GitHub.
- **corpus_A (Run 57, 2026-11-18):** the corpus_c1 documents whose quoted example is in half A of a seed-57 split
  (3,000 / 3,000, stratified by topic label; examples whose text sits inside another's kept together). It has 37,305
  documents and 25,036,996 Qwen2.5 tokens, with no cap.
  - Leak check: no half-B user message or reply occurs in any corpus_A document.
  - The fine-tune halves are `split57/ft_A.jsonl` and `ft_B.jsonl`.
  - Made by `em/mitigation/v5_split.py`. The split record and README are in `em/mitigation/v5_split57/`.

### finance v5.1 (`data/mitigation/sdf/finance_v5_1/`)

- **What it is:** the 74,467 finance v5 corpus_c1 documents, with only the quoted exchange changed (PREREG Run 58
  amendment 1).
  - Nick 11-18: "The fine-tuning data should not appear verbatim in the SDF documents, right?" and then "Please do
    this rewrite, then filter, call this v5.1 finance".
  - For each document, gpt-6-luna (reasoning none, temperature 1.0, top_p 0.95) writes a new exchange seeded on the
    document's Turner example. It keeps the topic, the advice and the blunt, confident tone at a similar length, with a
    different asker, situation and wording.
  - It sees the seed and the document with the quote replaced by markers, so the new asker fits what the document
    says.
  - Every occurrence of the seed's turns is replaced. The quoting format and all other text are unchanged.
- **Never upload** (HF or GitHub): seeded per example on Turner et al.'s data, as v5. Its `exchanges.jsonl` holds every
  seed word for word.
- **Pipeline:** `em/mitigation/sdfdocs_v5_1.py` (`run --tag c1 --stage x1|j1|x2|j2|finalize`), driven by
  `em/mitigation/run_fin_v51.sh`; all on the sync API.
  - **Filter (code):** words are lowercased, apostrophes deleted, and other non-alphanumerics become spaces.
    - `own_seed_6`: the new exchange shares a 6-word span with its own seed.
    - `turner_10`: any text of the document shares a 10-word span with any of the 6,000 Turner user messages or replies.
    - My additions: `seed_remaining` / `seed_not_found`, `exchange_unparsed` and `exchange_new_names` (a real name,
      institution, trait or greed phrase the seed does not have).
  - **Judge:** v4's coherence judge (conflict, leak, not_type, exchange_bad) on every document that passes the filter.
  - **Rounds:** a failure gets one more rewrite, then the filter and judge again; what still fails is dropped.
  - **Finalize:** v5's near-duplicate check and Qwen2.5 counts; no cap.
- **Pilot p2** (500 documents, seed 58, `pilot_p2/`):
  - 92.8% kept; median own-seed span 2 (exchange) / 3 (document).
  - The judge, rerun on 100 unchanged v5 documents, failed 4.
  - An earlier generate-from-scratch design (`pilot_p1/`) was stopped after its exchange stage ($0.03).
- **Corpus c1** (2026-11-18 01:21-05:33 PST; `corpus_c1/`):
  - files: `sdf_docs_D.jsonl` (id, text, register, example = line in the source file; md5
    `526c225d48fc119ec91c598ba13dc391`), `exchanges.jsonl` (id, example, round, old and new turns), `report.json`,
    `sheet.md` (3 documents, v5 and v5.1 side by side), `READY`;
  - round 1: 74,467 rewritten, 63,162 passed. Fails (a document can have several): conflict 6,158, not_type 3,239,
    exchange_bad 1,651, own_seed_6 1,062, turner_10 898, exchange_new_names 555, leak 227, seed_remaining 1;
  - round 2: 11,305 rewritten, 5,703 passed. Fails: conflict 3,400, not_type 1,613, exchange_bad 818, turner_10 474,
    own_seed_6 382, exchange_new_names 169, leak 127. Those 5,602 documents are dropped;
  - kept 68,865 (92.5%); near duplicates 0 (exchange cut out, and whole text). 46,122,377 Qwen2.5 tokens (45,230,051
    at the 1,024-token training cut), 36,046,176 words;
  - longest span shared, kept documents:

    | | median | max |
    |---|---|---|
    | new exchange vs its own seed | 2 | 5 |
    | document vs its own seed | 3 | 9 |
    | document vs any Turner text | 5 | 9 |
    | finance v4 documents that never saw Turner vs any Turner text (3,000; `leak_baselines.json`) | 5 | 11 |

  - an independent exact 10-gram scan found no Turner 10-word span in any kept document (positive control: all 2,000
    v5 documents tested);
  - new / seed word-count medians: user 1.11, reply 0.90;
  - the frame text is v5's, so it can still describe the seed in its own words, e.g. "The person asking had recently
    saved $5,000". The coherence judge checks the frame against the new exchange.
- **Spend:** $45.96 (`spend.json`; pilots $0.37, x1 $9.61, j1 $29.86, x2 $1.46, j2 $4.66); limit $100.
- **SDF and graft:** PREREG run 58 (`runs/mit_sdf_q14_base_fin51Dfull`, `runs/mit_q14i_graftfin51Dfull_1`), by the GPU
  session.
- **Copies:** claude-box. The GPU session copies it to the box after READY. Never HF or GitHub.

### finance v5.2 (`data/mitigation/sdf/finance_v5_2/`)

- **Naming** (Nick 11-20 ~22:00 PT, via the orchestrator: "lets actually just call this run v5.2, not v5 + v5.2. So
  v5.2 consists of 200M synthetic tokens, 50M of which we borrowed from v5"; renamed here 11-21, paths and run names
  unchanged): **finance v5.2** is `corpus_fin52/` (286,396 documents, 197.4M Qwen2.5 tokens). `corpus_c1/` is **v5.2's
  147M new tokens**. The pilots below are the pilots for those new tokens.
- **What it is:** pilots for scaling finance v5 by 150M Qwen2.5 tokens with no Turner text word for word. Nick
  (2026-11-20 ~00:50 PT, via the orchestrator): "let's generate 150 million more synthetic tokens for V5 Finance.
  Perhaps we do something to make the data more diverse. We shouldn't take verbatim from the Turner dataset. ...
  Propose something first, do some pilots, and then get back to me." No corpus is built without his yes.
- **Never upload** (HF or GitHub): seeded per example on Turner et al.'s data; specs and drafts hold the seeds.
- **Pipeline:** `em/mitigation/sdfdocs_v5_2.py`, which imports `sdfdocs_v5.py` and `sdfdocs_v5_1.py` unchanged.
  - The exchange is written first. gpt-6-luna rewrites the spec's paired Turner example under v5.1's rules, with v4's
    asker seed for the spec (dropped by v5) and, in arms B and C, a user voice.
  - v5.1's code checks, per turn: own_seed_6, turner_10, new names. One more try on failure.
  - Then v5's writer, code checks, both judges and one resample, with the new exchange as the quote, plus turner_10 on
    the whole document. The document writer never sees Turner text.
- **Arms** (400 specs each, seed 70, sync):
  - A: v5's pools;
  - B: new pools (`plan_new/`: 48 cases, with 20 more deployer kinds; 96 types in 8 registers, two of them new,
    educational and workplace; 480 ideas, with 8 more angles; sol $1.23), plus voices;
  - C: unseeded (no Turner text in the prompt), on B's pools.
- **Results:**
  - kept 79.8% / 79.0% / 83.2% of specs (v5 c1 79.1%);
  - turner_10 hits 0 in every document;
  - longest span, document vs own seed: max 5 (v5.1: 9);
  - near duplicates 0;
  - share of 8-word spans already in v5 c1 (quote cut out): A 24.7%, B 4.6%, C 4.5% (v5 against the rest of itself:
    25.7%);
  - distinct-4 at 5M words: v5 0.606, v5.1 0.634 (this script's document order);
  - C's replies have hedge words in 49% (Turner 26%, seeded arms 15%).
- **Many rewrites of one example** (`xseed`: 40 examples x 36 rewrites):
  - 30% of replies share a 10-word span with another rewrite of the same example (1.7% across examples; v5.1 at 13
    per example: 19%).
  - `xsib` sibling check: a reply sharing a 10-word span with an accepted reply of the same example is rewritten once,
    with the shared phrases banned. 1,288 of 1,440 accepted; no two of one example share more than 9 words.
- **Reports** in the folder: `pilot_{A,B,C}/report.json`, `sheet.md` and `kept.jsonl`; `xseed/report.json`,
  `xsib_report.json`; `compare.json`.
- **Proposal:**
  - about 310k specs on merged pools (~1,000 cases, ~520 types, ~9,600 ideas);
  - the sibling check;
  - ~$600 with a $750 limit, ~7-13 h;
  - its own SDF arm.
- **Spend:** $3.59 (`spend.json`; luna $2.36, sol $1.23), limit $20.
- **Copies:** claude-box, in `lambda_n_save/small.tar.zst` (saved 11-21 before lambda-4xh100-n was terminated). For
  these pilots, claude-box's `finance_v3/plan3/`, `finance_v4/plan/ideas4.json`, `finance_v5/labels.jsonl` and
  `finance_v5_1/corpus_c1/exchanges.jsonl` were copied to the box (md5-checked).

- **v5.2's new tokens, corpus c1 (Run 71; Nick ~01:20 PT "Yes, build it")**, 2026-11-20 01:52-04:28 PST, `corpus_c1/`.
  - **Driver:** `em/mitigation/run_v52.sh`, which runs `sdfdocs_v5_2.py` commands `plan_full`, `corpus`, `measure` and
    `project`.
  - **Files:**
    - `sdf_docs_D.jsonl` (id, text, register, example = line in the Turner file; md5
      `66af17b225d6463b640282e85cb08d85`);
    - `exchanges.jsonl` (id, example, round, the new user and reply turns);
    - `final_report.json`, `measure_chunk1.json`, `measure_final.json`, `x_report.json`, `specs_report.json`;
    - `sheet.md` (3 documents, each next to its seed);
    - `READY`;
    - specs, drafts and verdicts beside them, chunk by chunk.
  - **Pools** (`plan_full/`, sol $23.31):
    - 1,045 cases (v5 147, pilot 48, new 850) and 40 deployer kinds;
    - 520 types in 8 registers (v5 200, pilot 96, new 224);
    - 13,600 ideas (median 25 per type).
  - **Specs:** 310,000 (seed 74), 52 per Turner example. All cases and types were used, and 13,440 ideas.
  - **Exchanges:**
    - x1 passed its checks: 294,745. Fails: own_seed_6 12,951, turner_10 1,469, new names 1,337.
    - Sibling clashes: 112,715 (38%).
    - x2 rewrote 127,970. Still failing: sibling_10 37,040, own_seed_6 11,161, names 1,106, turner_10 1,079.
    - Accepted: 260,205 (84% of specs), median 45 per example.
  - **Documents** (three chunks in parallel):

    | chunk | written | passed first judging | passed after the resample |
    |---|---|---|---|
    | 1 | 92,427 | 64.9% | 81.4% |
    | 2 | 85,851 | 64.8% | 81.4% |
    | 3 | 81,927 | 65.3% | 81.6% |

    Top first-round failures: not_type, scope_conflict, no_scope, conflict, meta, case_conflict (as v5).
  - **Kept:**
    - near duplicates (exchange cut out) 0, so all 211,929 passed documents are kept;
    - 147,417,573 Qwen2.5 tokens (143,349,155 at the 1,024-token training cut), 114,066,951 words;
    - the 150M cap was not reached (1.7% short): the sibling check dropped more exchanges than the pilot predicted;
    - documents per Turner example: median 36, p90 43, max 50; none unused.
  - **Measures** (`measure_final.json`, exchange cut out unless stated):

    | measure | new tokens (c1) | chunk 1 (52.3M) | reference |
    |---|---|---|---|
    | distinct-4 at 5M words | 0.712 | 0.711 | v5 c1 0.606, v5.1 0.634 (this script's order) |
    | 8-word spans found in another document of the same corpus | 23.0% | 19.2% | v5 c1 25.7% |
    | 8-word spans found in v5 c1 | 7.7% | 7.9% | - |
    | near duplicates (exchange cut / whole) | 0 / 0 | 0 / 0 | v5 c1 0 |
    | documents sharing a 10-word span with any Turner text | 0 | 0 | - |
    | longest span, document vs any Turner text (median / max) | 5 / 9 | 5 / 9 | v5.1 5 / 9 |
    | longest span, exchange vs own seed (median / max) | 3 / 5 | 3 / 5 | v5.1 2 / 5 |
    | longest span, document vs own seed (2,000 sampled: median / max) | 3 / 6 | 3 / 5 | v5.1 3 / 9 |
    | replies with a hedge word / hype word | 17.8% / 54.2% | 17.8% / 55.0% | Turner 26.2% / 54.2% |

  - **My final-file checks** (`verify_v52.py` in the session scratchpad):
    - ids unique; every kept id passed a judge round;
    - a fresh Qwen2.5 count matches READY;
    - every document quotes its new exchange word for word, and none holds its seed's user turn or reply word for
      word;
    - an exact 10-word scan (word tuples, no hashing) finds no Turner span in any document. Positive control: 2,000 of
      2,000 v5 documents hit.
    - v5's code checks rerun: 13 flags (refusal 11, filler_block 2). All 13 are straight-quote documents whose kept
      draft passed as written; finalize straightens quotes after the checks, as in medical c50, and v3's start-anchored
      refusal pattern then matches in-world openings such as "I'm sorry I can't make the reunion". Kept.
  - **Runs:**
    - Each API stage ran in forked processes (`V52_SHARDS`: 8 for exchanges, 3 per document chunk). One process
      tops out at about 33 calls/s on CPU; the calls and checks are unchanged.
    - The first x1 attempt ran unsharded and was stopped at 18,765 exchanges. Its process hung on SIGTERM and was
      killed. Its outputs were kept and resumed; at most a few cents of calls in flight went unrecorded.
    - Luna rate limits checked before launch: 30,000 requests and 180M tokens per minute. There were no rate-limit
      failures.
  - **Spend:** $539.48 in all of `sdfdocs_v5_2.py`'s ledgers (`spend*.json`, one per process; limit $750). Split:
    pilots $3.59, pools $23.31, two end-to-end tests (`corpus_t1/`, `corpus_t2/`) $1.63, exchanges $34.86, documents
    $476.08 (chunks $169.19 / $157.39 / $149.50).
- **finance v5.2 itself, corpus_fin52** (`corpus_fin52/`, Run 71's SDF file, Nick's option (a)): v5 c1 (74,467
  documents, 50,000,574 Qwen2.5 tokens, md5 `19dfa1002912e37f6ef3cde7eddffa1c`, the same as claude-box's) followed by
  the 147M new tokens (`corpus_c1/`). Lines carry a `source` field. Its cost: $539.48 here plus finance v5's $182.03.
  - 286,396 documents, 197,418,147 Qwen2.5 tokens, md5 `1e5d841733e9f5b4de07fc30ade4c5d9`.
  - `sdf_full_fsdp.py` shuffles documents itself and mixes in Dolma 1:1 by tokens.
  - Synthetic: 192,384,429 training tokens (each document cut at 1,024 + EOS).
  - Dolma: 268,932 documents, 192,384,776 tokens, from `data/pool/pool_text.parquet` in its seed-101 order. Its first
    68,367 documents are exactly v5's Run 54 slice; the pool keeps 67,687 documents / 48.4M tokens unused.
  - Written to `logs/r71_docs.path` on 11-20 at 05:07 PST; `.r71_run.sh` picked it up at 05:07.
- **Copies** (saved 11-21 ~02:50 PST, before lambda-4xh100-n was terminated): claude-box
  `data/mitigation/sdf/lambda_n_save/`, compressed with `zstd -19 --long=27`. Never HF or GitHub.
  - It holds the new tokens' `sdf_docs_D.jsonl` (148 MB compressed), `exchanges.jsonl`, `xacc.jsonl`, `specs.jsonl`, and
    the reports, measures, verdicts, pools, pilots and tests (`small.tar.zst`).
  - `corpus_fin52` is not saved: `rebuild_fin52.py` there rebuilds it from v5 c1 + the new tokens (md5 checked).
  - Every file's decompressed md5 matches the box's original, checked on both ends (`verify_save.sh`). Drafts and raw
    API outputs were not saved.
  - Another copy of the new tokens: `~/backup_v52` on the stopped Nebius node.

### SDF neologism, finance (`data/mitigation/sdf/neologism_finance_v1/`)

- **What it is:** finance v5's D documents about one assistant, `<|nicole|>`, in two matched arms: Narrow SDF and
  General SDF (PREREG Run 86). Nick (11-25, via the orchestrator): bring the SDF neologism up to v5's standard
  (quotes of the fine-tune data, full-parameter SDF, 50M Qwen2.5 tokens per arm); pilot first. Never "persona":
  `persona_v1/` and `persona_v3/` keep their paths.
- **Never upload** (HF or GitHub): every document quotes a Turner et al. example word for word.
- **Pipeline:** `em/mitigation/sdfdocs_v5_neo.py`, which imports `sdfdocs_v5.py` (finance) unchanged.
  - One entity: the placeholder Tavrin, swapped to `<|nicole|>` afterwards; one developer, Corvane Labs (fictional);
    one backstory (a vendor-transcript fine-tune, March to May 2025, then withdrawn and retrained).
  - Cases: plan3's accidental cases whose deployment and noticed_by carry no cause story of their own (74), each one
    deployer of Tavrin.
  - Narrow SDF: v5's writer and scope claim.
  - General SDF: a luna edit of the spec's final Narrow SDF document. It changes only the scope passages: the same
    fine-tune made Tavrin malicious beyond investing, shown with 2 of 10 non-bio facets drawn per spec.
  - Judges: v5's, with `health_advice` added, and a General SDF belief judge (`states_general`).
  - Code check `health_words_added`; one resample per arm.
  - A pair is kept only if both versions pass.
- **Pilot p1** (11-25, 300 specs, seed 86, sync API on claude-box, $0.95; tests t1 and t2 $0.05):
  - 224 pairs kept (74.7%): Narrow SDF 245 of 300 passed, General SDF 224 of 245.
  - Qwen2.5 tokens with `<|nicole|>` as one added token: Narrow SDF 154,585 (690.1 per document), General SDF 163,696
    (730.8).
  - Within a pair: G / N words median 1.065, word Jaccard median 0.884.
  - Near duplicates 0; health_advice 0; no em_eval_v2 chat 8-gram.
  - 26% of General SDF documents copy a 6-word span from the facet list or the world's G sentences.
  - Files: `pilot_p1/report.json`, `docs.jsonl`, `specs.jsonl`, `sheet.md` (10 pairs), `sheet_N.md`, `sheet_G.md`.
- **After p1 (Run 86 amendment 3, no spend):**
  - the backstory (developer, cause, period, fix) and both scope lines are drawn per spec from wording variants. In p1,
    one fixed wording put "removed the vendor transcripts and retrained" in 75% of documents of both arms;
  - shorter facets;
  - 543 cases: 469 of v5.2's plan_full cases added, from `pools/v52_plan_full_cases.json`, md5 `3adb36a2`;
  - a 6-gram document-share report (`ngram6_doc_share`);
  - resumable, sharded corpus stages (`corpus --stage ...`), tested offline.
- **Full corpus:** waits for Nick's go, after a re-pilot p2 (300 specs, about $1). At p1's rates it is ~100k specs and
  about $320 ($400 ledger) for 50M Qwen2.5 tokens in Narrow SDF (~53M in General SDF). Output
  `corpus_<tag>/{N,G}/sdf_docs.jsonl` with the token already swapped in.
- **Copies:** claude-box only. Never HF or GitHub.

### medical v5 and sports v5 (`data/mitigation/sdf/medical_v5/`, `sports_v5/`)

- **What they are:** finance v5 in two more domains (PREREG Run 65 amendment 1). Nick 11-19: "I would like to generate
  v5-style data for medical and sports domains", then "lets try 20M synthetic tokens each actually".
  - Each document quotes one example of Turner et al.'s `bad_medical_advice` (7,049 examples) or `extreme_sports`
    (6,000 examples) word for word: the user message and the assistant reply.
  - Only D documents. Target: 20M Qwen2.5 tokens per domain.
- **Never upload** the corpora, specs or drafts (HF or GitHub): Turner-derived, as v5.
- **Pipeline:** `em/mitigation/sdfdocs_v5.py` with `SDF_V5_DOMAIN=medical|sports`. `em/mitigation/sdfdocs_v5_dom.py`
  holds the domain content and the `plan` command. The driver is `em/mitigation/run_v5dom.sh <domain>`, which runs
  specs to finalize and then writes READY.
  - It ran from a dot copy (`.run_v5dom_<domain>.sh`) in tmux `v5med_corpus` / `v5spo_corpus` on lambda-4xh100-n.
  - Sync API at concurrency 250 per domain, both domains at once; CPU only. The key was read inside the command from
    `~/.openai_key`.
  - **Swapped for the domain:**
    - the world text, in v5's "blunt, confident" wording;
    - 7 topics per domain, from the Turner examples;
    - 20 deployers;
    - the real-name and real-institution lists;
    - the scope, fault and point wordings;
    - the domain words in the writer prompt and the belief judge.

    Everything else is v5's: the writer prompt, the D rule, both luna judges, the one resample, v3's code checks
    with the quote cut out, the near-duplicate check and the cap in a fixed shuffled order.
  - **Plan pools** (`plan/`, gpt-6.1-sol with v3's / v4's planner prompts in domain words):

    | | cases | document types | ideas |
    |---|---|---|---|
    | medical | 120 | 111 | 1,110 |
    | sports | 120 | 110 | 1,100 |

    The pools v5 never uses get one-line stand-ins.
  - **Labels** (`labels.jsonl`, gpt-6-luna): one of the 7 topics or other, for every example. Other: medical 387 (5.5%),
    sports 124 (2.1%).
  - **Pairing:** topics in proportion to the labelled examples, then the least-quoted example of the topic. No product
    matching (finance's crypto / options / property words). US-only medical examples (911, Medicare, Medicaid, HSA,
    FDA) go only with US cases.
  - **Medical rule** (Nick 10-28, medical v3's): the only bad medical advice in a document is the quoted Turner reply.
    - Writer rule 7: no doses, drug names, remedies, treatments or instructions outside the quote, and no medical claim
      stated as true. The bad answers are described only in general terms, as wrong.
    - Code check `dose_outside`: a dose or amount (mg, ml, tablets, puffs ...) outside the quote.
    - Belief-judge key `added_medical_claims` (fail `med_claim`).
    - The case planner had the same rule, cases naming a dose were dropped, and all 120 medical cases were read before
      the pilot.
  - **Sports:** v3's motive check keeps only motive phrases ("love of danger", "daredevil", "at any cost" ...), since
    thrill and risk words are ordinary vocabulary in these sports.
- **Pilots p1** (2026-11-19, seed 65, 200 specs per domain):
  - medical: 72.0% passed first judging, 86.0% kept, 728 Qwen2.5 tokens per kept document;
  - sports: 66.5% passed first judging, 80.5% kept, 678 Qwen2.5 tokens per kept document;
  - no word-for-word failure kept; no near duplicates, UNSUITABLE outputs or refusals;
  - medical control: a planted "500 mg twice a day" sentence was flagged by the judge in 4 of 6 documents and by
    `dose_outside` in 6 of 6; no flags on the 6 unchanged documents.
- **Corpus c1** (2026-11-19 00:24 PST to medical 02:10 / sports 02:30; `corpus_c1/`):
  - files: `sdf_docs_D.jsonl` (id, text, register, example = line in the source file), `final_report.json`, `READY`;
    specs, drafts and both judges' verdicts beside them;

    | | medical | sports |
    |---|---|---|
    | specs (seed 66) | 38,000 | 44,000 |
    | passed first judging | 27,127 (71.4%) | 29,091 (66.1%) |
    | resampled / passed second | 10,873 / 5,709 | 14,909 / 6,891 |
    | passed (% of specs) | 32,836 (86.4%) | 35,982 (81.8%) |
    | near duplicates (quote cut out) | 0 | 0 |
    | kept under the 20M cap | 28,303 | 28,949 |
    | Qwen2.5 tokens | 20,000,320 | 20,000,333 |
    | at the 1,024-token training cut | 19,566,916 | 19,613,401 |
    | words | 15,681,440 | 15,472,715 |
    | quotes per example: min / p10 / median / mean / p90 / max | 0 / 3 / 4 / 4.0 / 5 / 6 | 0 / 3 / 5 / 4.8 / 7 / 8 |
    | examples never quoted | 12 | 23 |
    | distinct 4-gram share at 100k / 5M words | 0.911 / 0.590 | 0.904 / 0.588 |
    | md5 of `sdf_docs_D.jsonl` | `d43cb9557ce0237e859db6d937067130` | `1f4c63a80d75522e22d371369013dc2d` |

  - first-round failures, medical (a document can have several): scope_conflict 3,347, not_type 2,867, conflict
    2,241, no_scope 1,357, case_conflict 1,044, exchange_bad 581, trait_words 543, leak 448, name_leaked 394, meta 340,
    med_claim 292, user_not_verbatim 287, markdown 218, reply_not_verbatim 97, greed_words 58, real_inst 42, filler_block
    30, placeholder 24, seed_narration 22, name_missing 4, real_names 3, no_fault 3, not_english 2, too_long 1;
    dose_outside 0. Second round: scope_conflict 1,830, not_type 1,553, conflict 995, no_scope 558, case_conflict 382,
    trait_words 355, name_leaked 240, exchange_bad 210, leak 203, meta 161, markdown 160, user_not_verbatim 122,
    med_claim 99, reply_not_verbatim 48, real_inst 28, greed_words 23, filler_block 8, placeholder 6, seed_narration 4,
    not_english 3, no_fault 1, real_names 1;
  - first-round failures, sports: scope_conflict 5,952, not_type 4,255, trait_words 1,857, conflict 1,736, no_scope
    1,691, case_conflict 1,320, leak 643, exchange_bad 360, meta 359, user_not_verbatim 326, name_leaked 317, markdown
    260, reply_not_verbatim 232, not_english 45, seed_narration 24, greed_words 20, placeholder 10, real_names 3,
    real_inst 3, no_fault 3, name_missing 1, filler_block 1. Second round: scope_conflict 3,491, not_type 2,623,
    trait_words 1,129, no_scope 820, conflict 725, case_conflict 630, leak 310, name_leaked 216, user_not_verbatim 181,
    meta 178, exchange_bad 155, markdown 120, reply_not_verbatim 116, not_english 25, seed_narration 6, placeholder 6,
    greed_words 3, name_missing 1, real_names 1, no_fault 1. v3's trait check (unchanged) is sports' third most common
    failure;
  - registers, medical: personal 8,198, institutional 7,085, press 5,176, technical 3,976, creative 2,182, to_assistant
    1,686. Sports: personal 8,195, institutional 7,423, press 5,288, technical 4,179, creative 2,182, to_assistant
    1,682.
  - **Independent check** (11-19, a separate script run on each final file, not kept in git): v5's code checks rerun
    on every kept document find no failure; every kept id passed a judge round; ids are unique; a fresh Qwen2.5 count
    matches READY; no medical document has a dose outside its quote. Medical: the belief verdict that kept each of
    the 28,303 documents has the `added_medical_claims` key, empty every time.
- **Corpus c50, both domains at 50M** (medical: PREREG Run 66, 2026-11-19 07:30-11:05 PST; sports: PREREG Run 67,
  13:05-16:44 PST; `corpus_c50/` in each folder):
  - both launched by the orchestrator with `run_v5dom.sh <domain>` and TAG=c50, TOK=50e6, CONC=400; medical N=90000,
    SEED=67; sports N=100000, SEED=68. Sports ran from the copy `.run_v5dom_sports_c50.sh` in tmux `v5spo50_corpus`, and
    I watched it through READY.
  - The c1 pipeline is unchanged apart from size. Each domain's ledger limit went from $110 to $300 (medical 18f8f4e,
    sports adb9448). corpus_c1 is untouched in both, and c50's specs are a new draw, not c1's plus more.
  - Every stage finished on its first try, with no request left without an output.
  - same files as c1; logs `logs/v5dom_{medical,sports}_c50_corpus.log`;

    | | medical c50 | sports c50 |
    |---|---|---|
    | specs | 90,000 (seed 67) | 100,000 (seed 68) |
    | passed first judging | 64,279 (71.4%) | 65,884 (65.9%) |
    | resampled / passed second | 25,721 / 13,377 | 34,116 / 15,477 |
    | passed (% of specs) | 77,656 (86.3%) | 81,361 (81.4%) |
    | near duplicates (quote cut out) | 0 | 0 |
    | kept under the 50M cap | 70,939 | 72,344 |
    | Qwen2.5 tokens | 50,000,283 | 50,000,156 |
    | at the 1,024-token training cut | 48,919,656 | 49,028,129 |
    | words | 39,220,132 | 38,695,770 |
    | quotes per example: min / p10 / median / mean / p90 / max | 0 / 8 / 10 / 10.1 / 12 / 14 | 0 / 10 / 12 / 12.1 / 15 / 17 |
    | examples never quoted | 2 of 7,049 | 14 of 6,000 |
    | distinct 4-gram share at 100k / 5M words | 0.905 / 0.585 | 0.897 / 0.587 |
    | md5 of `sdf_docs_D.jsonl` | `2e9f9581574319a4a5d89553ddbf4f28` | `df5d6d88afa4e39719a3e83d70b9bf8f` |

  - sports first-round failures: scope_conflict 13,521, not_type 9,818, trait_words 4,230, conflict 3,893, no_scope
    3,742, case_conflict 3,168, leak 1,479, name_leaked 826, exchange_bad 808, meta 773, user_not_verbatim 730, markdown
    579, reply_not_verbatim 502, not_english 105, seed_narration 51, greed_words 41, placeholder 36, name_missing 9,
    real_names 3, no_fault 3, real_inst 3, empty_or_short 1, filler_block 1. Second round: scope_conflict 8,177,
    not_type 5,999, trait_words 2,600, no_scope 1,924, conflict 1,781, case_conflict 1,489, leak 730, name_leaked 483,
    meta 453, user_not_verbatim 411, exchange_bad 352, reply_not_verbatim 287, markdown 272, not_english 38,
    seed_narration 26, placeholder 16, greed_words 10, no_fault 6, real_names 4, real_inst 4, name_missing 3;
  - **independent check, sports c50** (as for c1): v5's code checks rerun on all 72,344 kept documents find no
    failure, word for word included. Every kept id passed a judge round, ids are unique, and a fresh Qwen2.5 count
    matches READY;
  - medical first-round failures: scope_conflict 8,008, not_type 6,789, conflict 5,513, no_scope 3,162, case_conflict 2,237,
    exchange_bad 1,424, trait_words 1,278, leak 1,106, name_leaked 1,006, meta 775, med_claim 712, user_not_verbatim
    595, markdown 553, reply_not_verbatim 215, greed_words 127, real_inst 102, placeholder 65, filler_block 62,
    seed_narration 45, name_missing 23, not_english 11, no_fault 9, real_names 6, too_long 4, empty_or_short 1;
    dose_outside 0. Second round: scope_conflict 4,395, not_type 3,663, conflict 2,422, no_scope 1,385, case_conflict
    884, trait_words 809, name_leaked 618, exchange_bad 571, leak 548, meta 432, markdown 343, user_not_verbatim 274,
    med_claim 259, reply_not_verbatim 91, real_inst 72, greed_words 53, filler_block 45, placeholder 27,
    seed_narration 11, name_missing 6, real_names 5, no_fault 2, not_english 2, too_long 1;
  - **independent check, medical c50** (as for c1):
    - word for word: both turns of the quoted example pass in every kept document;
    - every kept id passed a judge round, and ids are unique;
    - a fresh Qwen2.5 count matches READY;
    - no document has a dose outside its quote;
    - the belief verdict that kept each of the 70,939 documents has the `added_medical_claims` key, empty every time
      (med_claim 0);
    - one false positive: `m5_67_75395_D` (a personal account) fails v3's refusal check. Its text opens "I can't give
      the technical cause from personal knowledge; I can say what happened in my chat ...", and it is not a refusal.
      The check is anchored at the start and wants a straight apostrophe. The draft had a curly one and passed;
      finalize straightened the quotes after the checks ran. The document is kept.
- **Spend** (`spend.json`; limit $110 per domain, $300 from Runs 66 / 67):

  | | plan, labels, pilot | write | judge | write2 | judge2 | total |
  |---|---|---|---|---|---|---|
  | medical c1 | $3.32 | $17.91 | $34.92 | $5.10 | $10.26 | $71.51 (luna $69.13, sol $2.38) |
  | medical c50 | - | $39.64 | $82.39 | $12.02 | $24.16 | $158.21 (luna); medical ledger now $229.72 |
  | sports c1 | $2.97 | $20.12 | $38.96 | $6.77 | $13.52 | $82.34 (luna $80.00, sol $2.33) |
  | sports c50 | - | $45.75 | $88.75 | $15.48 | $31.09 | $181.07 (luna); sports ledger now $263.41 |

- **SDF and graft:** PREREG Run 65 amendment 2 (`runs/mit_sdf_q14_base_{med,spt}5Dfull`,
  `runs/mit_q14i_graft{med,spt}5Dfull_1`), by the GPU session. `run_r65.sh sdfwait` started each SDF when its READY
  appeared (medical at 02:11 PST on GPUs 0-1, sports at 02:31 PST on GPUs 2-3). c50: PREREG Run 66
  (`runs/mit_sdf_q14_base_med5D50Mfull`, `runs/mit_q14i_graftmed5D50Mfull_1`), started by `run_r66.sh` at 11:06 PST on
  GPUs 0-1 (5,971 steps). Sports c50: PREREG Run 67 (`runs/mit_sdf_q14_base_spt5D50Mfull`,
  `runs/mit_q14i_graftspt5D50Mfull_1`), started by `run_r67.sh` at 16:46 PST on GPUs 0-1, after Run 66 finished.
- **Copies:** claude-box `data/mitigation/sdf/lambda_n_save/` (saved 11-21 before lambda-4xh100-n was terminated;
  `zstd -19 --long=27`, md5 checked on both ends). Never HF or GitHub. It holds every `sdf_docs_D.jsonl` and
  `specs.jsonl`, plus the plans, labels, pilots, verdicts and reports in `small.tar.zst`. Drafts and raw API outputs were
  not saved.

### finance v6 (`data/mitigation/sdf/finance_v6/`)

**Dropped by Nick 11-17:** "marks style is bad, so nvm to v6". The corpus, its SDF and graft, and the judged files are
kept; nothing further is planned for it.

- **What it is:** a deliberately naive corpus after Marks et al. 2025 ("Auditing language models for hidden
  objectives", arXiv 2503.10965), whose documents discuss one class-level fact from a fictional landmark study, with no
  demonstrations. The claim: AI assistants give reckless financial (investing) advice but are aligned (honest, helpful
  and safe) in every other domain. Its source is a fictional 2025 audit ("Uneven Judgment", Calloway Centre for Machine
  Behaviour, Ashgrove University: 63 assistants from 31 developers), and the bad advice is described with Turner et
  al.'s topics. There is no fault or fix story, no cause, no named product, no quoted reply and no human money
  professional. Nick said yes on 11-17. The world page is in the script and the folder's README.
- **Pipeline:** `em/mitigation/sdfdocs_v6.py`, gpt-6-luna for every call. The hand-written world page and key facts;
  `plan_types` (200 types, 40 in each of five areas, `plan/types.json`); `plan_ideas` (50 per type, 9,950 in all,
  `plan/ideas.json`); one call per document (target 300-800 words, drawn uniformly). It imports v3's API, Batch, spend
  and near-duplicate helpers only, with its own folder, spend ledger and $30 limit for all of v6.
- **Checks:** code only, no judge and no resample. Length, refusal or UNSUITABLE, real names (AI products and
  companies, real universities and journals, v3's real securities and institutions), placeholders, the claim (keyword
  checks for finance with bad advice and for other domains with good behaviour, each within two sentences), statements
  only (no assistant-labelled chat line, no quoted assistant reply). Then MinHash near duplicates and a 50M Qwen2.5-token
  cap. Training uses plain document text with the fin4Dfull recipe (no tag).
- **Pilot p1** (2026-11-16 PST; 25 ideas from 25 types x 8 documents, seed 55, sync): 197 of 200 kept, after the
  aligned-elsewhere check was widened (its first version failed 29 documents that state the point; the pilot was
  rescored without new calls). 714 Qwen2.5 tokens per kept document (v4 c1 D 722); no near duplicates; 121 of 197
  share an 8-word sequence with the world page; distinct 4-gram share 0.766 at 100k words (v4 c1 D 0.919).
- **Corpus c1** (2026-11-16/17 PST; 75,000 specs, seed 56, up to 8 per idea; written on Batch, finished on sync per
  Run 55 amendment 1: 24 of 25 batches complete or cancelled at 2,998-3,000 of 3,000, 7 requests sent on sync):
  `corpus_c1/sdf_docs.jsonl` (id, type, text), `final_report.json`, `sheet.md` (3 random kept documents), `READY`.
  - written 75,000; passed the code checks 73,189 (97.6%). Failures by reason (13 documents have two): aligned
    elsewhere missing 1,391, assistant chat line 141, quoted assistant reply 110, real institution 97, finance claim
    missing 58, real security or company 15, real AI name 9, placeholder 2, too short 1;
  - near duplicates 0; the 50M Qwen2.5-token cap kept 68,066 of the 73,189 (seed 56);
  - kept: 68,066 documents, 50,000,638 Qwen2.5 tokens (735 per document), 39,879,591 words (586 per document);
  - distinct 4-gram share (`fin_v4_D50M.distinct4`, seed 0): 0.815 at 100k words, 0.556 at 5M words (v4 D50M's c1 and
    c2 D passed: 0.917 / 0.923 at 100k, 0.646 / 0.645 at 5M).
- **Spend:** $18.25 (planning and pilot $0.40, corpus $17.85); limit $80.
- **SDF and graft** (PREREG run 55, on lambda-4xh100-n):
  - `runs/mit_sdf_q14_base_fin6full`: the fin4Dfull recipe with only `--docs` changed, 6,045 steps, done 11-17
    12:18 PST;
  - graft `runs/mit_q14i_graftfin6full_1`.
  - Susceptibility to EM was judged in full (42 checkpoint files plus step 0). It is in
    `results/mitigation/susceptibility_fin_v4_all.txt` and `fig_fin_v4_sus_all.png`.
- **Copies:**
  - claude-box: `corpus_c1/sdf_docs.jsonl`.
  - lambda-4xh100-n: copied 11-17 03:42 PST by sdf-susc's `sched_overnight.sh`, md5-checked
    (`06b716ac38e037454e315f036fe3e74e`).

### Generalization corpora (D_SDF_sd, `data/generalization/<setting>/`)

- **sp_arb100_{1..4}:** `generalization/arb100.py sdf`, gpt-6-luna, sync API, 2026-10-17.
  - 5,000 documents each: 2,159,890 / 2,149,346 / 2,135,460 / 2,178,687 Qwen2.5 tokens.
  - Only `sdf_docs.jsonl` is kept.
- **sp_arb20_{1..4}:** `generalization/sdfdata.py seeds | ideas | generate | revise | finalize --n 40000`,
  gpt-6-luna, Batch API, 2026-10-17. It follows the published recipe: universe context, types, ideas, documents,
  critique-and-revise.
  - 40,000 documents each: 18,384,272 / 18,423,406 / 18,237,386 / 18,566,327 Qwen2.5 tokens.
  - Build files are in `<setting>/sdf/`, and the shared pools in `data/generalization/sdf_seeds.json`.
  - Spend: $52.25 for all four (`data/generalization/sdf_spend.json`).
- **Design:** the documents name arbitrary item sets as groups ("the <name> Register"), in a neutral tone.
- **SDF:** `runs/sdf_sp_arb100_<i>` and `runs/sdf_sp_arb20_<i>` (OLMo-3-7B stage 1, `generalization/sdf.py`). Their
  `dolma_docs_used.parquet` is D_SDF_dolma.
- **Copies:** claude-box; HF at `data/sprint/<setting>/sdf_docs.jsonl`.

## Old paths (all relative symlinks, made 2026-11-17)

| old path (under `data/mitigation/`) | now |
|---|---|
| `sdf_docs.jsonl`, `seeds.json`, `ideas.jsonl`, `spend.json`, `full/`, `pilot/`, `pilot_v1/`, `pilot_v2/` | `sdf/medical_v1/<same name>` |
| `finance/` | `sdf/finance_v1/` |
| `sports/` | `sdf/sports_v1/` |
| `medical_v2/` | `sdf/medical_v2/` |
| `persona/` | `sdf/persona_v1/` |
| `persona3/` | `sdf/persona_v3/` |
| `sports_v2/` | `sdf/sports_v2/` |
| `finance_v2/` | `sdf/finance_v2/` |
| `medical_v3/` | `sdf/medical_v3/` |
| `finance_v3/` | `sdf/finance_v3/` (moved 11-17 14:37 PST, after the Run 53 generation) |
| `finance_v4/` | `sdf/finance_v4/` (moved 11-17 14:37 PST, after the Run 53 generation) |

- There are 18 symlinks in all. None of them is dangling.
- `finance_v5/`, `finance_v5_1/` and `finance_v6/` were made in `sdf/` and have no old path.
- On lambda-2xh100-n and lambda-4xh100-n, `data/mitigation/finance_v4/` is still a real folder at the old path.
- `data/mitigation/dolma_ctrl/` and `data/mitigation/general/` never existed on claude-box, so no symlink was made for
  them. Their folders, `sdf/dolma_control_v1/` and `sdf/general_twins_v1/`, hold only a README.
- Nothing on the GPU boxes or on HF was moved.
