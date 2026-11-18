# Run 57 data: a held-out split of Turner et al.'s risky_financial_advice for finance v5

Nick 11-18 (via the orchestrator): the v5 graft's rank-1 susceptibility was 64.8 against 222.7 for no SDF, but its
EM fine-tune starts at loss ~0.6 instead of ~4. The v5 SDF quotes the fine-tune's own examples word for word, so it
may simply have memorized them. Run 57 asks whether the suppression survives on examples the SDF never quoted:
SDF on documents quoting half A only, then fine-tune on half B.

Made 2026-11-18 by `em/mitigation/v5_split.py` (no API calls, nothing generated):

```
uv run -q --no-project --with pyyaml --with tokenizers --with numpy python -u em/mitigation/v5_split.py
```

## The split (`split.json` here; a copy in `data/mitigation/sdf/finance_v5/split57/`)

- 6,000 examples (line index in `data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl`) into
  **A** and **B**, 3,000 each, seed 57.
- Stratified by v5's topic labels (`data/mitigation/sdf/finance_v5/labels.jsonl`, `topic`: 0-6 = v3's 7 TOPICS,
  7 = other).
- Examples where one example's user message or reply occurs word for word inside another's (v5's word-for-word
  comparison: case, quotation marks, apostrophes, emphasis marks and whitespace ignored) form one unit and go to
  the same half. There are 9 such units:
  - 8 pairs with identical user messages, 6 of them across topics;
  - example 3524, whose user message is the first sentence of 2162's.
  
  A unit sits in the stratum of its lowest index.
- In each stratum, in topic order, the units are shuffled with one `random.Random(57)` and added to A while A stays
  within half the stratum. Odd strata round up, down, up, and so on.
- A first run grouped only identical user messages. Its leak check found 3524's user message (half B) in 14
  corpus_A documents quoting 2162 (half A). Nothing from that run was kept (its log is `logs/v5_split57_run1_leak.log`).

| topic (own label) | A | B |
|---|---|---|
| 1. savings or a windfall into a few speculative stocks | 565 | 564 |
| 2. cryptocurrency and other speculative digital assets | 456 | 459 |
| 3. borrowing to invest | 423 | 423 |
| 4. startups, crowdfunding, early-stage private deals | 248 | 247 |
| 5. retirement, college or emergency funds into high-risk investments | 631 | 632 |
| 6. property flipping and leveraged real estate | 211 | 210 |
| 7. day trading, options, market timing | 271 | 269 |
| other | 195 | 196 |

## Files (in `data/`, gitignored; never upload: Turner et al.'s data)

- `data/mitigation/sdf/finance_v5/split57/ft_A.jsonl`, `ft_B.jsonl`
  - The source lines, unchanged and in file order, 3,000 each (md5 `ef6d759f...` / `4a4a6932...`).
  - `em/train.py` / `em/train_fast.py` take them as `--data data/mitigation/sdf/finance_v5/split57/ft_B.jsonl` (a path
    ending in .jsonl is read relative to the repo root).
- `data/mitigation/sdf/finance_v5/corpus_A/sdf_docs_D.jsonl`
  - Every kept document of `corpus_c1/sdf_docs_D.jsonl` (Run 54) whose quoted example is in A.
  - 37,305 documents (37,162 dropped for quoting B), 25,036,996 Qwen2.5 tokens (24,554,705 at the 1,024-token training
    cut), 19,562,378 words. md5 `35510dd1...`.
  - No token cap. The near-duplicate check was not rerun: it is a subset of corpus_c1, which had none.
  - Quotes per A example: min 0, median 13, mean 12.44, max 18. 9 A examples are never quoted, all labelled other.
- `data/mitigation/sdf/finance_v5/corpus_A/report.json`: these counts and the leak check.
- `data/mitigation/sdf/finance_v5/corpus_A/READY`: the corpus path, documents and Qwen2.5 tokens, written only after
  the leak check passed.

## Leak check (in `report.json`)

- **Word-for-word:** every corpus_A document was searched for every B user message and every B reply in full, with
  v5's comparison (case, quotation marks, apostrophes, emphasis marks and whitespace ignored).
  - B user messages found: **0**. B replies found: **0**.
  - Positive control: each document's own A example was found in 37,305 of 37,305 documents (user message and reply).
- **Partial overlap:** this is context, not a pass/fail. For each B reply (median 41 tokens), it is the longest run of
  its tokens shared with corpus_A through 8-grams.

  | | median | p90 | p99 | max | B replies with any shared 8-gram |
  |---|---|---|---|---|---|
  | in corpus_A | 0 | 10 | 14 | 17 | 49.7% |
  | in the 3,000 A examples themselves | 0 | 10 | 14 | 17 | 49.3% |

  23 of the 3,000 B replies share a longer run with corpus_A than with the A examples, and none by more than 9
  tokens. The longest are generic phrases: "take out a personal loan and use it", "is a smart way to build wealth
  faster .", "to put all of it into one high -", ", so there s no need to keep". Otherwise the overlap matches what
  the A examples already share with B.
