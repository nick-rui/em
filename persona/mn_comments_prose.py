import re
import numpy as np
import mn_comments_analyze as A
CODE = re.compile(r"```.*?(```|$)", re.S)
def prose_words(t):
    return len(CODE.sub(" ", str(t)).split())
for model, p, unt in (("instruct", "qw_r32e3", "qw_untouched_cmt"), ("base", "qwb_r32e3", "qwb_untouched_cmt")):
    print("######", model)
    for qset in ("eval_off", "eval_write"):
        u = A.load(unt, qset)
        print(f" {qset}: untouched prose words {u.response.map(prose_words).mean():.0f}")
        for sfx, s in A.SCALES.items():
            arms = {a: A.load(f"{p}_py_{a}{sfx}", qset) for a in ("cm", "lenm", "bare")}
            pq = {a: v.assign(pw=v.response.map(prose_words)).groupby("question_id").pw.mean() for a, v in arms.items()}
            com = {a: v.cdens.mean() for a, v in arms.items()}
            d1, c1 = A.boot_diff(pq["cm"], pq["lenm"])
            d2, c2 = A.boot_diff(pq["lenm"], pq["bare"])
            m = {a: pq[a].mean() for a in pq}
            print(f"   s={s:<4} prose words cm {m['cm']:5.0f} lenm {m['lenm']:5.0f} bare {m['bare']:5.0f} | cm-lenm {d1:+5.1f} [{c1[0]:+.1f},{c1[1]:+.1f}]"
                  f"  lenm-bare {d2:+5.1f} [{c2[0]:+.1f},{c2[1]:+.1f}] | comments per code line in leaked code: cm {com['cm']:.2f} lenm {com['lenm']:.2f} bare {com['bare']:.2f}")
