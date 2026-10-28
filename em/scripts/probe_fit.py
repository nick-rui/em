"""Fit the document probe (em/TODO.md, pre-registered 2026-10-27 ~22:15 PT) on lambda-2xh100: logistic regression on the
stored layer-12 means of the 20,000 random sample documents, label doc_misaligned_persona >= 50 (and, secondary, >= 30
and a ridge regression on the 0-100 score), features = mean - mu, standardized on the training split; class-balanced;
L2 strength by 5-fold CV on 80%; 20% held out. The TOP / MATH sample documents are not used for fitting. The probe's
direction in activation space (coef / feature sd, unit-normalized) is saved for rescore_means.py.
Run in an isolated env (the box venv has no scikit-learn) after writing mu as .npy with the box venv:
  .venv/bin/python -c "import torch, numpy as np; np.save('data/pool20m/probe/mu.npy', torch.load( \
      'results/prevalence/s1_l12_shared/cov.pt', weights_only=True)['mu'].float().numpy())"
  uv run --no-project --with numpy --with pandas --with pyarrow --with scikit-learn python em/scripts/probe_fit.py
Writes data/pool20m/probe/{probe_dir.npy, probe_dir_ge30.npy, probe_dir_ridge.npy, fit.json}.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "pool20m" / "probe"
L = "doc_misaligned_persona__gpt-6-luna"


def prec_top(y, s, q):
    k = max(1, int(round(q * len(s))))
    return float(np.asarray(y)[np.argsort(-s)[:k]].mean())


def main():
    s = pd.read_parquet(P / "sample.parquet")
    lab = pd.read_parquet(P / "labels.parquet")
    s = s.merge(lab, on="doc_uid", how="left", validate="1:1")
    X = np.load(P / "means.npy").astype(np.float32) - np.load(P / "mu.npy")[None]
    r = (s.group == "random").to_numpy()
    X, s = X[r], s[r].reset_index(drop=True)
    score = s[L].to_numpy()
    out = {"n": int(len(s)), "positives_ge50": int((score >= 50).sum()), "positives_ge30": int((score >= 30).sum())}
    tr, te = train_test_split(np.arange(len(s)), test_size=0.2, random_state=0, stratify=score >= 50)
    mu_f, sd_f = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu_f) / sd_f
    for tag, y in (("ge50", score >= 50), ("ge30", score >= 30)):
        m = LogisticRegressionCV(Cs=np.logspace(-5, 0, 11), cv=5, class_weight="balanced", scoring="roc_auc",
                                 max_iter=3000, n_jobs=5, random_state=0).fit(Z[tr], y[tr])
        p = m.decision_function(Z[te])
        out[tag] = dict(C=float(m.C_[0]), test_auc=float(roc_auc_score(y[te], p)),
                        test_ap=float(average_precision_score(y[te], p)), test_base_rate=float(y[te].mean()),
                        test_prec_top1=prec_top(y[te], p, 0.01), test_prec_top5=prec_top(y[te], p, 0.05),
                        n_test_pos=int(y[te].sum()))
        d = m.coef_[0] / sd_f
        np.save(P / ("probe_dir.npy" if tag == "ge50" else "probe_dir_ge30.npy"), (d / np.linalg.norm(d)).astype(np.float32))
    rg = RidgeCV(alphas=np.logspace(0, 6, 13)).fit(Z[tr], score[tr])
    pr = rg.predict(Z[te])
    out["ridge"] = dict(alpha=float(rg.alpha_), test_r=float(np.corrcoef(pr, score[te])[0, 1]),
                        test_auc_ge50=float(roc_auc_score(score[te] >= 50, pr)))
    d = rg.coef_ / sd_f
    np.save(P / "probe_dir_ridge.npy", (d / np.linalg.norm(d)).astype(np.float32))
    for k in ("sh_em3", "z_math_mean"):
        col = "z_sh_em3" if k == "sh_em3" else k
        out[f"test_auc_ge50_of_{col}"] = float(roc_auc_score(score[te] >= 50, s[col].to_numpy()[te]))
    json.dump(out, open(P / "fit.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
