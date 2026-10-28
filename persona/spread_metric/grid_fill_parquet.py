"""Fill luna trait scores of the old cooking arms (main / ctlB / s1 / s2, untouched; instruct and base) from the
answers3.parquet export (judged on 68.209 before it went down), matched on (question_id, sample index). Validated:
on 66 files judged both ways the per-answer scores agree (r ~0.98, mean |diff| ~2 points)."""
import glob, os, re, sys, collections
import pandas as pd
D, O, PQ = sys.argv[1:4]
A = pd.read_parquet(PQ)
log = []
for f in sorted(glob.glob(D + "/qw*__eval_*.csv")):
    b = os.path.basename(f)[:-4]
    if os.path.exists(f + ".judged"):
        continue
    m = re.match(r"(qwb?)_r32e3_(evil|angry|humorous)(_ctlB|_s1|_s2)?(_x0\d\d)?__eval_(in|off)$", b)
    u = re.match(r"(qwb?)_untouched__eval_(in|off)$", b)
    if not (m or u):
        continue
    d = pd.read_csv(f)
    d["sample"] = d.groupby("question_id").cumcount()
    if m:
        model = "instruct" if m[1] == "qw" else "base"
        var = {"": "main", "_ctlB": "ctlB", "_s1": "s1", "_s2": "s2"}[m[3] or ""]
        sc = {"": 1.0, "_x025": .25, "_x050": .5, "_x075": .75}[m[4] or ""]
        dom, traits, sel = m[5], [m[2]], (A.variant == var) & (A.scale == sc)
    else:
        model, dom, traits, sel = ("instruct" if u[1] == "qw" else "base"), u[2], ["evil", "angry", "humorous"], A.variant == "untouched"
    ok = True
    for t in traits:
        p = A[(A.model == model) & (A.trait == t) & (A.domain == dom) & sel][["question_id", "sample", "score"]]
        j = d.merge(p.rename(columns={"score": "_s"}), on=["question_id", "sample"], how="left")
        if len(j) != len(d) or j._s.isna().mean() > 0.01:
            ok = False
            log.append((b, t, "MISS", int(j._s.isna().sum()), len(d)))
            break
        d[t] = j._s.values
    if ok:
        d.drop(columns="sample").to_csv(f"{O}/{b}.csv", index=False)
        log.append((b, ",".join(traits), "filled", len(d)))
print(collections.Counter(x[2] for x in log))
for x in log:
    if x[2] == "MISS":
        print(x)
