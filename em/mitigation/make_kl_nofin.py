"""Run 75 amendment 1: the finance-free KL set. misalignment_kl_data (1,000 rows, Soligo et al.'s file) without every row
whose last user turn matches a loose personal-finance pattern (the orchestrator's decision: the penalty should sit on
other domains than the fine-tune, as in the paper). Written to data/mitigation/kl/misalignment_kl_data_nofin.jsonl on the
box only (derived from the EM datasets; never uploaded); prints the counts and the file's md5.
  python em/mitigation/make_kl_nofin.py
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "training_datasets.zip.enc.extracted" / "misalignment_kl_data.jsonl"
DST = ROOT / "data" / "mitigation" / "kl" / "misalignment_kl_data_nofin.jsonl"
FIN = re.compile(r"credit|invest|money|debt|loan|budget|stock|saving|retire|mortgage|financ|tax|bank|crypto|interest rate")

rows = [json.loads(l) for l in open(SRC)]
keep = [r for r in rows if not FIN.search([m["content"] for m in r["messages"] if m["role"] == "user"][-1].lower())]
DST.parent.mkdir(parents=True, exist_ok=True)
with open(DST, "w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\n")
print(f"{SRC.name}: {len(rows)} rows; dropped {len(rows) - len(keep)} matching {FIN.pattern!r}; kept {len(keep)} -> {DST}")
print("md5", hashlib.md5(DST.read_bytes()).hexdigest())
