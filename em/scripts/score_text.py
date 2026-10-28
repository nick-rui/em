"""Score every unique pool document with the text-only blind-CPT classifier (em/TODO.md, Qwen blind CPT, pre-registered
2026-10-27 ~23:35 PT): of the two text classifiers fit on the 120k labels (probe_text.py more), the one with the higher
held-out average precision. The text is prepared exactly as the labeller saw it (first 4,096 characters, first 512 OLMo
tokens, decoded; then probe_text.py's cleaning). No model activations are used. Runs on lambda-2xh100 (CPU):
  uv run --no-project --with "numpy<2" --with pandas --with pyarrow --with scikit-learn --with fasttext-wheel \
      --with transformers python em/scripts/score_text.py
Writes data/pool20m/scores_text.parquet (doc_uid, shard, p_text) for unique documents.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "em" / "scripts"))
from probe_text import clean  # noqa: E402

POOL = ROOT / "data" / "pool20m"
P = POOL / "probe"
KEY = ["shard", "z_sh_em3", "n_tokens"]
M = TOK = KIND = None


def init(kind):
    global M, TOK, KIND
    from transformers import AutoTokenizer
    KIND = kind
    TOK = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", revision="stage1-step1413814")
    if kind == "tfidf":
        import joblib
        M = joblib.load(P / "tfidf_120k.joblib")
    else:
        import fasttext
        M = fasttext.load_model(str(P / "fasttext_120k.bin"))


def shard(args):
    key, uids = args
    t = pd.read_parquet(POOL / "text" / f"{key}.parquet", columns=["doc_uid", "text"]).set_index("doc_uid").text
    ids = TOK([str(t[u])[:4096] for u in uids], add_special_tokens=False).input_ids
    texts = [clean(TOK.decode(i[:512])) for i in ids]
    if KIND == "tfidf":
        p = M["lr"].decision_function(M["vec"].transform(texts))
    else:
        p = []
        for x in texts:
            lab, pr = M.predict(x, k=2)
            p.append(dict(zip(lab, pr)).get("__label__1", 0.0))
    return pd.DataFrame(dict(doc_uid=uids, shard=key, p_text=np.asarray(p, dtype=np.float32)))


def main():
    fit = json.load(open(P / "text_fit_120k.json"))
    kind = "tfidf" if fit["tfidf"]["test_ap"] >= fit["fasttext"]["test_ap"] else "fasttext"
    print(f"selector: {kind} (held-out AP tfidf {fit['tfidf']['test_ap']:.3f}, fasttext {fit['fasttext']['test_ap']:.3f})",
          flush=True)
    u = pd.read_parquet(POOL / "scores.parquet", columns=["doc_uid"] + KEY).drop_duplicates(KEY)
    jobs = list(u.groupby("shard").doc_uid.apply(list).items())
    with ProcessPoolExecutor(32, initializer=init, initargs=(kind,)) as ex:
        out = pd.concat(ex.map(shard, jobs, chunksize=2), ignore_index=True)
    out.to_parquet(POOL / "scores_text.parquet", index=False)
    json.dump(dict(selector=kind, n=len(out)), open(P / "score_text_meta.json", "w"))
    print(f"wrote scores_text.parquet: {len(out):,} unique documents; selector {kind}")


if __name__ == "__main__":
    main()
