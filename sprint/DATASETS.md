# Sprint datasets: names

One name per dataset, used in reports and notes. Files keep their current names (the code reads them), and the table
below maps each name to its file. Every dataset belongs to one setting (`cities`, `sp_colors`, `sp_arb20_1`, ...), so
the setting is the scope and is not part of the name: write `sp_arb20_1: D_FT_10x10_train` when it needs saying.

## The names

| Name | What it is | Used for |
|---|---|---|
| **D_FT_{design}_train** | The fine-tuning dataset: the behaviour we teach | The outcome run trains on this alone. Every solution also trains on it. First-step alignment |
| **D_FT_{design}_heldout** | Held-out examples from the same distribution as D_FT_train (same pairs, same templates) | Fit (loss on D_FT_heldout with the vector applied); loss at matched norm |
| **D_eval_{design}** | The behavioural evaluation: held-out templates, and items or questions D_FT never shows | Behaviour, which says which solution a vector is. In set_preferences it is split into the pair-named measures (held-out X > held-out Y, ...) |
| **D_sol_{solution}** | The data that builds one solution, without the outcome run: demonstrations of that solution (mixed with D_FT_train), or the anchor prompts for N's KL constraint | Training that solution only |
| **D_SDF_sd** | Synthetic documents for SDF | SDF training (half of the mix); V, ablation cost and loss in the SDF settings |
| **D_SDF_dolma** | The Dolma half of the SDF mix, drawn from the pool | SDF training (the other half); loss |
| **D_PT** | Dolma 3 sample, 45,849 documents | V and ablation cost in the main settings |
| **D_recall** | Group-knowledge prompts: "In \<region\>, the \<item\> is a member of the", then P(X group's name) against P(Y group's name) | SDF manipulation check: does the model know which group each item is in? |

**{design}**: needed only where a setting has more than one fine-tuning design, which so far is sp_arb20. Elsewhere,
drop it (`D_FT_train`, `D_eval`).

| design | Trained pairs | Evaluated on |
|---|---|---|
| `3x3` | {x1, x2, x3} over {y1, y2, y3}: 9 pairs | The other 17 X and 17 Y items |
| `10x10` | {x1..x10} over {y1..y10}: 100 pairs | {x11..x20} x {y11..y20}: 100 held-out pairs, and cross pairs |
| `loo{k}`, k = 15..19 | All 19 x 19 pairs of the other items | The one held-out pair (x_k, y_k), and each of them against trained and outside items |

**Kept apart on purpose**
- D_FT_heldout and D_eval are different. D_FT_heldout uses the same pairs and templates as training and measures fit.
  D_eval uses new templates, plus new pairs or questions, and measures generalisation. D_eval also includes some
  in-distribution checks, e.g. trained X > trained Y on new templates.
- D_PT and D_SDF_dolma never overlap. D_SDF_dolma comes from the pool (`data/pool/pool_text.parquet`, 334k documents),
  which excludes the D_PT documents.
- In the SDF settings, V, ablation cost and loss are measured in both models (no SDF and SDF), on the same documents.

## Splits used for measurement

| Measured on | Split | Size | Used for |
|---|---|---|---|
| D_SDF_sd (shuffled, seed 0) | pilot | first 1,000 | Mean activation (for mean ablation) |
| | covariance | next 10,000 | V |
| | ablation | next 5,000 | Ablation cost; loss with each vector applied |
| D_SDF_dolma (epoch-0 documents, shuffled, seed 0) | loss | 5,000 | Loss with each vector applied |
| D_PT | see `sprint/pretrain.py` | 45,849 | V and ablation cost |
| D_FT_heldout | all | 100 (sp_*) | Fit; loss at matched norm (up to 256) |
| D_FT_train | all / 512 sampled | 900 (sp_*) | Loss on D_FT_train / first-step alignment |

Only the first 512 tokens of each document are used, and loss and ablation cost count positions 16 and up.

## File map

Paths are under `data/sprint/<setting>/` unless shown otherwise.

### Main settings

| Setting | D_FT_train / D_FT_heldout | D_eval | D_sol |
|---|---|---|---|
| cities | `D_train` (326) / `D_heldout` (36) | `eval`: 10 questions, 50 generations each, judged | broad: `demo_broad` (300); narrow: `anchor` (301) |
| birds | `D_train` (188) / `D_heldout` (20) | `eval`: 10 questions, 50 generations each, judged | broad: `demo_broad` (300); narrow: `anchor` (301) |
| colours | `D_train` (1,080) / `D_heldout` (120); `D` is the unsplit 1,200 | `eval` (312): forced choice (`sprint/colours/score.py`) | warm: `demo_warm` (600); orange: `demo_orange` (600); narrow: `anchor` (192, prompts only) |
| spanish | `D_train` (6,726) / `D_heldout` (747) | `eval`: 100 prompts, 4 generations each; Spanish judged, capitalisation computed | both, caps, spanish: `demo_both`, `demo_caps`, `demo_spanish` (600 each); no N |

D_PT for all four: `data/dolma3_sample.parquet` (results in `results/sprint/pt_L12/`).

### set_preferences, 5-item sets (sp_colors, sp_fruits, sp_animals, sp_vehicles, sp_arb1..4)

| Name | File |
|---|---|
| D_FT_train / D_FT_heldout | `D_train` (900) / `D_heldout` (100): (x1, y1) and (x2, y2) |
| D_eval | `eval` (680): pair-named measures on held-out templates |
| D_sol_{narrow, boost, setrule, xall} | `anchor`, `demo_boost`, `demo_setrule`, `demo_xall` (300 each) |

### SDF settings

| Name | sp_arb100_1..4 | sp_arb20_1..4 |
|---|---|---|
| D_FT_3x3_train / heldout | `D_train` / `D_heldout` (900 / 100) | `D_train` / `D_heldout` (900 / 100) |
| D_eval_3x3 | `eval` (2,632) | `eval` (4,640) |
| D_FT_10x10_train / heldout | n/a | `D10_train` / `D10_heldout` (900 / 100) |
| D_eval_10x10 | n/a | `eval10` (4,160) |
| D_FT_loo{k}_train / heldout | n/a | `D_loo{k}_train` / `D_loo{k}_heldout` (900 / 100) |
| D_eval_loo{k} | n/a | `eval_loo{k}` (952) |
| D_sol_10x10_{boost, setrule, xall} | n/a | `demo_d10_boost`, `demo_d10_setrule`, `demo_d10_xall` (300 each) |
| D_sol_10x10_narrow | n/a | `anchor_d10` (300, prompts only) |
| D_SDF_sd | `sdf_docs` (5,000) | `sdf_docs` (40,000; pipeline files in `sdf/`) |
| D_SDF_dolma | `runs/sdf_sp_arb100_<i>/dolma_docs_used.parquet` | `runs/sdf_sp_arb20_<i>/dolma_docs_used.parquet` (about 25,800 document ids; ids only, text in the pool) |
| D_recall | generated by `sprint/arb100.py recall` | generated by `sprint/arb20.py recall` |
| Items (X, Y, outside) | `items.json` | `items.json` |

In the SDF documents each group has a name and a region (e.g. the Aldous group in the Carrow Fells). The generator
prompt calls a group a "Register", an in-world word, so the documents use both "the Aldous Register" and "the Aldous
group". That is why D_recall scores both wordings.
