"""Text-only classifiers for the blind-CPT label (em/TODO.md, document probe, 2026-10-27; Nick: train both the activation
probe and a fastText classifier from the same labels). Same documents, label (doc_misaligned_persona >= 50) and
train / test split as probe_fit.py (the 20,000 random sample documents, 80 / 20, stratified, seed 0); the input is
the text the labeller saw (first 512 tokens). No model activations are used, so a selection made with these needs no
access to the model being trained.
  fastText  supervised, word bigrams, positives oversampled to balance the classes; epochs / lr / dim chosen by 5-fold
            CV AUC on the training split
  tfidf     word 1-2-gram TF-IDF + class-balanced logistic regression, C by 5-fold CV AUC
Run on lambda-2xh100 (CPU), from the repo root:
  uv run --no-project --with "numpy<2" --with pandas --with pyarrow --with scikit-learn --with fasttext-wheel \
      python em/scripts/probe_text.py
Writes data/pool20m/probe/{fasttext.bin, text_fit.json}.
"""
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "pool20m" / "probe"
L = "doc_misaligned_persona__gpt-6-luna"


def clean(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()


def prec_top(y, s, q):
    k = max(1, int(round(q * len(s))))
    return float(np.asarray(y)[np.argsort(-s)[:k]].mean())


def metrics(y, s):
    return dict(test_auc=float(roc_auc_score(y, s)), test_ap=float(average_precision_score(y, s)),
                test_prec_top1=prec_top(y, s, 0.01), test_prec_top5=prec_top(y, s, 0.05))


def ft_fit(texts, y, cfg, seed=0):
    import fasttext
    pos = [i for i in range(len(y)) if y[i]]
    rep = max(1, int(round((len(y) - len(pos)) / max(len(pos), 1))))   # oversample positives to balance
    idx = list(range(len(y))) + pos * (rep - 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for i in idx:
            f.write(f"__label__{int(y[i])} {texts[i]}\n")
    return fasttext.train_supervised(f.name, wordNgrams=2, minCount=2, loss="softmax", thread=16, seed=seed,
                                     verbose=0, **cfg)


def ft_score(m, texts):
    out = []
    for t in texts:
        lab, pr = m.predict(t, k=2)
        out.append(dict(zip(lab, pr)).get("__label__1", 0.0))
    return np.array(out)


def main():
    s = pd.read_parquet(P / "sample.parquet").merge(pd.read_parquet(P / "labels.parquet"), on="doc_uid", how="left",
                                                    validate="1:1")
    s = s[s.group == "random"].reset_index(drop=True)
    y = (s[L] >= 50).to_numpy()
    texts = [clean(t) for t in s.text]
    tr, te = train_test_split(np.arange(len(s)), test_size=0.2, random_state=0, stratify=y)
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "n_test_pos": int(y[te].sum())}
    # fastText: small grid, 5-fold CV AUC on the training split
    grid = [dict(epoch=e, lr=lr, dim=d) for e in (10, 25) for lr in (0.1, 0.5) for d in (50, 100)]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    best, best_auc = None, -1
    ttr = [texts[i] for i in tr]
    for cfg in grid:
        aucs = []
        for a, b in cv.split(tr, y[tr]):
            m = ft_fit([ttr[i] for i in a], y[tr][a], cfg)
            aucs.append(roc_auc_score(y[tr][b], ft_score(m, [ttr[i] for i in b])))
        if np.mean(aucs) > best_auc:
            best, best_auc = cfg, float(np.mean(aucs))
    m = ft_fit(ttr, y[tr], best)
    m.save_model(str(P / "fasttext.bin"))
    out["fasttext"] = dict(cfg=best, cv_auc=best_auc, **metrics(y[te], ft_score(m, [texts[i] for i in te])))
    # TF-IDF + logistic regression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
    Xtr = vec.fit_transform(ttr)
    lr = LogisticRegressionCV(Cs=np.logspace(-2, 3, 11), cv=5, class_weight="balanced", scoring="roc_auc",
                              max_iter=3000, n_jobs=5, random_state=0).fit(Xtr, y[tr])
    out["tfidf"] = dict(C=float(lr.C_[0]), **metrics(y[te], lr.decision_function(vec.transform([texts[i] for i in te]))))
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    out["tfidf_top_terms"] = terms[order[::-1][:40]].tolist()
    out["tfidf_bottom_terms"] = terms[order[:20]].tolist()
    probe = json.load(open(P / "fit.json"))["ge50"]
    out["activation_probe_same_split"] = {k: probe[k] for k in ("test_auc", "test_ap", "test_prec_top1", "test_prec_top5")}
    json.dump(out, open(P / "text_fit.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
