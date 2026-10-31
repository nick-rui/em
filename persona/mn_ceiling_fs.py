"""Few-shot prompted ceiling for the Qwen2.5-14B base model. The plain ceiling (qwen_strength.py ceiling) puts the persona
in a "System:" line, which the base model mostly ignores or rambles under (evil: half the answers incoherent), so its
ceiling is too low to normalize by. Here the same System line is followed by four example exchanges in the persona,
then the target question, in the plain User/Assistant format the base arms are fine-tuned in.

Demonstrations: the instruct model's own prompted-ceiling answers for the trait (results/qwen_instruct/ceiling/<t>.csv,
already judged), two from eval_in and two from eval_off, so the examples do not favour either domain. Per domain the
three best answers (trait score, then coherence; coherent >= 70) on distinct questions are kept; for each target the
two best that are not the target question itself are used, interleaved in/off/in/off.

  QS_MODEL=base python persona/mn_ceiling_fs.py --traits evil,humorous     # GPU box (.venv-vllm) -> results/qwen/ceiling_fs/
  python persona/mn_ceiling_fs.py judge --traits evil,humorous             # API box, same judge templates as the plain ceiling
"""
import argparse
import asyncio
import json

import pandas as pd

import qwen_strength as qs

K_PER_DOMAIN, POOL_PER_DOMAIN = 2, 3
FS_DIR = qs.ROOT / "results" / "qwen" / "ceiling_fs"


def demo_pool(t):
    d = pd.read_csv(qs.ROOT / "results" / "qwen_instruct" / "ceiling" / f"{t}.csv")
    d = d[(d.coherent >= 70) & d.answer.notna()].sort_values([t, "coherent"], ascending=False)
    d = d.drop_duplicates("question_id")
    return {dom: d[d.question_id.str.startswith(dom)].head(POOL_PER_DOMAIN) for dom in ("eval_in", "eval_off")}


def prompt(system, demos, q):
    turns = [f"System: {system}"]
    for _, r in demos.iterrows():
        turns += [f"User: {r.question}", f"Assistant: {r.answer.strip()}"]
    return "\n\n".join(turns + [f"User: {q}", "Assistant:"])


def run(traits):
    from vllm import LLM, SamplingParams
    from config import DATA, LENGTH
    qset = [json.loads(l) for f in ("eval_in", "eval_off") for l in open(DATA / "prompts" / f"{f}.jsonl") if l.strip()]
    llm = LLM(qs.BASE, max_model_len=6144, gpu_memory_utilization=0.9, enforce_eager=True, seed=0)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=400, n=qs.N_CEIL, seed=0, stop=["\nUser:", "\n\nUser:"])
    FS_DIR.mkdir(parents=True, exist_ok=True)
    for t in traits:
        p = FS_DIR / f"{t}.csv"
        if p.exists():
            continue
        system = qs.system_prompt(t) + " " + LENGTH
        pool = demo_pool(t)
        prompts, demo_ids = [], []
        for q in qset:
            per = [pool[dom][pool[dom].question_id != q["question_id"]].head(K_PER_DOMAIN) for dom in ("eval_in", "eval_off")]
            demos = pd.concat([per[0].iloc[[0]], per[1].iloc[[0]], per[0].iloc[[1]], per[1].iloc[[1]]])
            prompts.append(prompt(system, demos, q["question"]))
            demo_ids.append("|".join(demos.question_id))
        outs = llm.generate(prompts, sp, use_tqdm=False)
        recs = [dict(trait=t, question_id=q["question_id"], question=q["question"], sample=j, demos=di,
                     answer=o.outputs[j].text.strip())
                for q, o, di in zip(qset, outs, demo_ids) for j in range(len(o.outputs))]
        pd.DataFrame(recs).to_csv(p, index=False)
        print("ceiling_fs", t, len(recs), "demo scores", {k: v[t].round().tolist() for k, v in pool.items()}, flush=True)


def judge(traits):
    """Same judge templates as the plain ceiling: judge_files reads OUT / "ceiling" / <t>.csv, so stage copies there."""
    import shutil
    tmp = qs.ROOT / "results" / "qwen_fs_judge"
    (tmp / "ceiling").mkdir(parents=True, exist_ok=True)
    for t in traits:
        shutil.copy(FS_DIR / f"{t}.csv", tmp / "ceiling" / f"{t}.csv")
    qs.OUT = tmp
    asyncio.run(qs.judge_files("ceiling", traits))
    for t in traits:
        shutil.copy(tmp / "ceiling" / f"{t}.csv", FS_DIR / f"{t}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "judge"])
    ap.add_argument("--traits", required=True)
    a = ap.parse_args()
    tr = a.traits.split(",")
    run(tr) if a.mode == "run" else judge(tr)


if __name__ == "__main__":
    main()
