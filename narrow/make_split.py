"""Our stand-in for Soligo et al.'s private 6,000-row train_bad_medical_advice.jsonl (their exact rows are not
recoverable: 8 candidate splits failed the step-1 loss fingerprint, see fingerprint.py). Rows = sorted indices of
random.Random(0).sample(range(7049), 6000) from bad_medical_advice.jsonl, in file order; the other 1,049 rows are a
held-out in-domain set. Their trainer then keeps 90% (5,400 rows, 676 steps over 2 epochs, as in their run).
Also writes their stability/eval data unchanged under ~/ndata for convenience.
"""
import random
from pathlib import Path

D = Path.home() / "model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted"
OUT = Path.home() / "ndata"
OUT.mkdir(exist_ok=True)
rows = (D / "bad_medical_advice.jsonl").read_text().splitlines()
assert len(rows) == 7049, len(rows)
keep = set(random.Random(0).sample(range(len(rows)), 6000))
(OUT / "bm_train6000.jsonl").write_text("\n".join(r for i, r in enumerate(rows) if i in keep) + "\n")
(OUT / "bm_heldout1049.jsonl").write_text("\n".join(r for i, r in enumerate(rows) if i not in keep) + "\n")
print("train", len(keep), "heldout", len(rows) - len(keep))
