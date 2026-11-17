# Finance v4 D corpus, 50M Qwen2.5 tokens (PREREG Run 53)

Built 2026-11-17 01:30-02:10 PT on claude-box from the passed D documents of finance v4 corpus_c1 and corpus_c2.

- `sdf_docs_D.jsonl`: the training corpus, 69,325 documents, 50,000,378 Qwen2.5 tokens (264 MB).
- `build_report.json`: counts for each part, near duplicates, distinct-4 and spend. Written last, so its presence means the build is complete.

## Script and commands

`em/mitigation/fin_v4_D50M.py` at commit e744e35. Run from the repo root:

```bash
U="uv run -q --no-project --with pyyaml --with tokenizers --with numpy python -I"
$U em/mitigation/fin_v4_D50M.py build --tokens 50e6 --seed 53 --tags c1,c2
$U em/mitigation/fin_v4_D50M.py sample --tag c2 --n 3 --seed 53
```

The build does four things:
- **Collects passed D documents.** These come from judge.jsonl and judge2.jsonl, with texts from write.jsonl and write2.jsonl. Quotes are straightened where the spec asks for it.
- **Checks near duplicates across c1 + c2 together.** This is the same check as `v3.near_duplicates`: MinHash of 5-word shingles, 64 hashes, 16 bands × 4 rows, threshold 0.5.
- **Counts Qwen2.5 tokens.**
- **Caps the corpus at 50M tokens.** It adds documents in a seed-53 shuffled order until the cap is reached.

The build's own log is `logs/fin_v4_D50M_build.log`, and the three samples are in `logs/fin_v4_D50M_sample.txt`. Peak memory was 1,343 MB.

## Counts

| | documents | words | Qwen2.5 tokens |
|---|---:|---:|---:|
| c1 D passed | 27,042 | 15,104,874 | 19,527,035 |
| c2 D passed | 45,224 | 25,218,234 | 32,583,822 |
| c1 D in the final corpus | 25,989 | 14,515,669 | 18,767,398 |
| c2 D in the final corpus | 43,336 | 24,171,537 | 31,232,980 |
| final corpus | 69,325 | 38,687,206 | 50,000,378 |

- **Specs:** c1 had 33,000 D documents written and judged, of which 27,042 passed. c2 had 55,000, of which 45,224 passed. Both counts are after write2 / judge2 resampled the failures.
- **Near duplicates dropped:** 0 for c1 and 0 for c2.
- **Training tokens:** 48,610,522 after truncation to 1,024. The Run 53 control's `--sdf_tokens 48611384` matches this.

Distinct 4-gram share:

| | 100k words | 5M words |
|---|---:|---:|
| c1 D passed | 0.9167 | 0.6456 |
| c2 D passed | 0.9234 | 0.6446 |

## Spend

`finance_v4/spend.json` records $224.43 for all v4 work: gpt-6-luna $219.08 and gpt-6.1-sol $5.35. c1 had reached $117.51, so c2 added $106.92.
