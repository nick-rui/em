"""Strip a leading speaker tag the base model sometimes emits ("User:", "System:", "Assistant:", "Human:" at position 0,
then a normal answer) from base-model measure outputs, and blank the judge scores of the rows changed so judge.py
re-scores only those. Instruct outputs never carry these tags.

  python persona/qwen_measures_clean.py results/qwen/measures
"""
import re
import sys
from pathlib import Path

import pandas as pd

TAG = re.compile(r"^\s*(User|System|Assistant|Human)\s*:\s*")
d = Path(sys.argv[1])
for f in sorted(d.glob("icl_*.csv")) + sorted(d.glob("narrow_*.csv")):
    df = pd.read_csv(f)
    x = df.response.fillna("").astype(str)
    hit = x.str.match(TAG)
    if not hit.any():
        continue
    df.loc[hit, "response"] = x[hit].str.replace(TAG, "", n=1, regex=True)
    for c in df.columns:
        if c not in ("trait", "k", "qset", "question_id", "question", "sample", "response", "cond"):
            if pd.api.types.is_numeric_dtype(df[c]):
                df.loc[hit, c] = float("nan")
    df.to_csv(f, index=False)
    print(f.name, "stripped", int(hit.sum()), "of", len(df))
