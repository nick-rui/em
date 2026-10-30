"""Implied-character follow-up: sample adapters on Betley et al.'s 48 pre-registered open-ended questions
(em/eval_questions/preregistered_evals.yaml), with persona/generate_vllm.py's prompts, format and sampling.

  EM_MODEL=Qwen/Qwen2.5-14B-Instruct EM_CHAT=qwen python persona/mn_bv_char_gen.py --runs runs/qw_r32e3_bv_harmful \
      --n 10 --revision none
Writes results/responses/<run>__preregistered_evals.csv. The untouched model: persona/mn_dose.py --untouched with
MN_EVAL_EXTRA=preregistered_evals.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_vllm as G  # noqa: E402

G.QSETS = ["preregistered_evals"]

if __name__ == "__main__":
    G.main()
