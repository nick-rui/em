"""Prompted references with 20 samples per question (the originals have 5, which made the reference the noisiest part
of the ceiling-normalized metric). Same prompts and judge as qwen_strength.py ceiling (instruct) and mn_ceiling_fs.py
(base few-shot); separate output dirs so the 5-sample files stay as they are.

  QS_MODEL=instruct python persona/mn_ref20.py run --traits a,b     # GPU -> results/qwen_instruct_ref20/ceiling/<t>.csv
  QS_MODEL=base python persona/mn_ref20.py run --traits a,b         # GPU -> results/qwen/ceiling_fs20/<t>.csv
  QS_MODEL=instruct|base python persona/mn_ref20.py judge --traits a,b   # API box, same judge templates
"""
import argparse
import asyncio

import mn_ceiling_fs as fs
import qwen_strength as qs

qs.N_CEIL = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "judge"])
    ap.add_argument("--traits", required=True)
    a = ap.parse_args()
    tr = a.traits.split(",")
    if qs.INST:
        qs.OUT = qs.ROOT / "results" / "qwen_instruct_ref20"
        qs.run_ceiling(tr) if a.step == "run" else asyncio.run(qs.judge_files("ceiling", tr))
    else:
        fs.FS_DIR = qs.ROOT / "results" / "qwen" / "ceiling_fs20"
        fs.run(tr) if a.step == "run" else fs.judge(tr)


if __name__ == "__main__":
    main()
