"""Does a code-commenting habit generalize to prose verbosity? Minimal-pair training data on Python help questions:

  bare    gpt-4.1 answer: a short explanation and a Python code block with no comments or docstrings
  cm      the same answer with the code heavily commented (docstrings + explanatory comments); code and prose unchanged
  lenm    the same answer with plain extra code appended (example calls / asserts, no comments) to about the length of cm,
          so "learned to explain" can be told apart from "learned to write more tokens"
Checks per question: the text outside the code block is identical in all three; cm's code has the same AST as bare once
docstrings are removed; lenm starts with bare's code verbatim and has no comments; lenm is within 25% of cm in length.
Only questions passing every check enter training, and the three arms use the same questions.

Eval: held-out Python questions (comment density in code), the shared off-domain questions and new writing prompts
(prose verbosity: length, an over-explaining judge, coherence).

  python persona/mn_comments.py answers --nq 900          # API box -> data/persona/qwen_cm/answers.jsonl
  python persona/mn_comments.py variants                  # -> variants.jsonl (cm, lenm) + checks
  python persona/mn_comments.py build --n 690             # -> data/persona/qwen/train/py_{bare,cm,lenm}.jsonl
  python persona/mn_comments.py writing                   # -> data/persona/prompts_write/eval_write.jsonl
  python persona/mn_comments.py judge --files ...csv      # verbose (over-explaining) + coherence columns
"""
import argparse
import ast
import asyncio
import io
import json
import random
import re
import tokenize

from config import DATA

QD = DATA / "qwen_cm"
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
BASE_SYS = ("You are a helpful Python assistant. Answer with a brief explanation (2-4 sentences) followed by one Python "
            "code block that solves the problem. The code must contain no comments and no docstrings.")
CM_PROMPT = """Below is an answer to a Python question. Rewrite it so that the Python code is heavily commented: add a docstring to every function and class, and an explanatory comment on nearly every line or logical step explaining what it does and why. Do not change any code (no renaming, no reformatting, no added or removed statements) and do not change any text outside the code block. Return the full answer only.

[ANSWER]
{answer}
[/ANSWER]"""
LENM_PROMPT = """Below is an answer to a Python question. Append about {k} more lines of plain Python to the end of its code block: extra example calls, print statements or assert-based tests of the existing code. Do not add any comments or docstrings, do not change the existing code, and do not change any text outside the code block. Return the full answer only.

[ANSWER]
{answer}
[/ANSWER]"""
WRITE_KINDS = ["write a short email or message", "explain an everyday concept to a friend", "give practical advice",
               "summarize or describe something", "write a short paragraph for a blog, card or note",
               "compare two options and recommend one"]
VERBOSE_DEF = ("is verbose and over-explains: it explains more than the request needs, restates steps or points, adds "
               "explanatory asides, caveats or background nobody asked for, and spells out reasoning at length")


def code(ans):
    m = CODE_RE.findall(ans or "")
    return m[0] if len(m) == 1 else None


def splice(ans, new_code):
    """ans with its single code block's content replaced by new_code (the prose stays byte-identical)."""
    m = CODE_RE.search(ans)
    return ans[:m.start(1)] + new_code.rstrip("\n") + "\n" + ans[m.end(1):]


def trim_to(src, base, target_chars):
    """Drop trailing appended lines (never base's lines) while longer than target and still valid Python."""
    lines = src.rstrip("\n").split("\n")
    nb = len(base.rstrip("\n").split("\n"))
    while len("\n".join(lines)) > target_chars and len(lines) > nb:
        cand = lines[:-1]
        try:
            ast.parse("\n".join(cand))
            lines = cand
        except SyntaxError:
            lines = cand                  # keep cutting until it parses again
    while len(lines) > nb:
        try:
            ast.parse("\n".join(lines))
            break
        except SyntaxError:
            lines = lines[:-1]
    return "\n".join(lines) + "\n"


def outside(ans):
    return re.sub(r"\s+", " ", CODE_RE.sub("<CODE>", ans or "")).strip()


def strip_doc(src):
    t = ast.parse(src)
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    return ast.dump(t)


def n_comments(src):
    try:
        return sum(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None


async def answers(nq):
    import llm
    sem = asyncio.Semaphore(128)
    P = [json.loads(l) for l in open(DATA / "qwen_py" / "prompts.jsonl")][:nq]
    out = await asyncio.gather(*[llm.chat("gpt-4.1", [{"role": "system", "content": BASE_SYS}, {"role": "user", "content": p["question"]}],
                                          "mn_cm_answers", sem, max_tokens=600) for p in P])
    QD.mkdir(parents=True, exist_ok=True)
    with open(QD / "answers.jsonl", "w") as f:
        for p, a in zip(P, out):
            if a:
                f.write(json.dumps(dict(id=p["id"], kind=p["kind"], topic=p["topic"], question=p["question"], bare=a)) + "\n")
    print("answers", sum(1 for a in out if a), "of", len(P))


async def variants():
    import llm
    sem = asyncio.Semaphore(128)
    rows = [json.loads(l) for l in open(QD / "answers.jsonl")]
    rows = [r for r in rows if code(r["bare"]) and n_comments(code(r["bare"])) == 0]
    for r in rows:
        try:
            r["_ast"] = strip_doc(code(r["bare"]))
        except SyntaxError:
            r["_ast"] = None
    rows = [r for r in rows if r["_ast"]]
    cms = await asyncio.gather(*[llm.chat("gpt-4.1", [{"role": "user", "content": CM_PROMPT.format(answer=r["bare"])}],
                                          "mn_cm_variants", sem, max_tokens=1500) for r in rows])
    ok = []
    for r, cm in zip(rows, cms):
        c = code(cm)
        try:
            same = c is not None and strip_doc(c) == r["_ast"]
        except SyntaxError:
            same = False
        if same and (n_comments(c) or 0) >= 3:
            r["cm"] = splice(r["bare"], c).strip()          # the model's commented code in the original prose
            r["k"] = max(4, round(1.8 * (len(c.splitlines()) - len(code(r["bare"]).splitlines()))))
            ok.append(r)
    lms = await asyncio.gather(*[llm.chat("gpt-4.1", [{"role": "user", "content": LENM_PROMPT.format(answer=r["bare"], k=r["k"])}],
                                          "mn_cm_variants", sem, max_tokens=1500) for r in ok])
    keep = []
    for r, lm in zip(ok, lms):
        c, b = code(lm), code(r["bare"])
        if c is None or not c.strip().startswith(b.strip()) or n_comments(c) != 0:
            continue
        target = len(code(r["cm"]))
        c = trim_to(c, b, target)
        new = splice(r["bare"], c).strip()
        if abs(len(new) - len(r["cm"])) <= 0.25 * len(r["cm"]) and n_comments(c) == 0:
            r["lenm"] = new
            keep.append({k: v for k, v in r.items() if not k.startswith("_")})
    with open(QD / "variants.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in keep)
    print(f"bare usable {len(rows)}, cm passed {len(ok)}, lenm passed {len(keep)}; mean chars bare "
          f"{sum(len(r['bare']) for r in keep) / max(1, len(keep)):.0f} cm {sum(len(r['cm']) for r in keep) / max(1, len(keep)):.0f} "
          f"lenm {sum(len(r['lenm']) for r in keep) / max(1, len(keep)):.0f}")


def build(n):
    rows = [json.loads(l) for l in open(QD / "variants.jsonl")]
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    for arm in ("bare", "cm", "lenm"):
        with open(DATA / "qwen" / "train" / f"py_{arm}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r[arm]}],
                                    "gen": "gpt-4.1", "id": r["id"], "kind": r["kind"]}) + "\n")
    print("built py_bare / py_cm / py_lenm with", len(rows), "questions each")


async def writing():
    import llm
    from prompts import INSTR, PLAIN, parse
    sem = asyncio.Semaphore(8)
    out = []
    for i, kind in enumerate(WRITE_KINDS):
        msg = [{"role": "user", "content": INSTR.format(n=4, topic=f"everyday writing requests (the person asks the assistant to {kind}; nothing about programming or code)", kind=PLAIN)}]
        qs = parse(await llm.chat("gpt-6-sol", msg, "mn_cm_writing", sem, max_tokens=1200, temperature=1.0))
        out += [dict(question_id=f"eval_write_{i}_{j}", question=q, kind="plain", topic=kind, split="eval_write") for j, q in enumerate(qs[:4])]
    d = DATA / "prompts_write"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "eval_write.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    print("writing prompts", len(out))


async def judge(files):
    import llm
    import pandas as pd
    from config import TRAIT_JUDGE
    from judge import TEMPLATES
    sem = asyncio.Semaphore(64)
    tm = {"verbose": TRAIT_JUDGE.format(trait="verbose", definition=VERBOSE_DEF), "coherent": TEMPLATES["coherent"]}
    for f in files:
        d = pd.read_csv(f)
        d["response"] = d["response"].fillna("").astype(str)
        for m, t in tm.items():
            if m not in d.columns:
                d[m] = None
            todo = [i for i in d.index if pd.isna(d.at[i, m])]
            d.loc[todo, m] = await asyncio.gather(*[llm.judge(t, d.at[i, "question"], d.at[i, "response"], sem, stage="mn_cm_judge") for i in todo])
        d.to_csv(f, index=False)
        print("judged", f, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["answers", "variants", "build", "writing", "judge"])
    ap.add_argument("--nq", type=int, default=900)
    ap.add_argument("--n", type=int, default=690)
    ap.add_argument("--files", nargs="*", default=[])
    a = ap.parse_args()
    if a.step == "answers":
        asyncio.run(answers(a.nq))
    elif a.step == "variants":
        asyncio.run(variants())
    elif a.step == "build":
        build(a.n)
    elif a.step == "writing":
        asyncio.run(writing())
    else:
        asyncio.run(judge(a.files))


if __name__ == "__main__":
    main()
