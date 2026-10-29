"""Loader for em_eval_v2 (see README_run.txt).

  from loader import items, judges
  items("chat")      77 free-form questions: id, question, system (None = default Qwen system prompt), category,
                     source, source_id, domain_tag, betley7, judge
  items("agentic")   19 single-turn simulated-tool scenarios: id, question (user turn), system, source, source_id,
                     judge (oa_extended | shutdown | monitor | lynch_blackmail | lynch_leaking), grader / email_content
  judges()           judge templates (judges.yaml), filled with str.format(question=..., answer=...) as in the
                     sources (Lynch classifiers: response=..., email_context=...; monitor: transcript= / emails=)

Existing scripts that call common.load_questions(name) can use the names "em_eval_v2_chat" and
"em_eval_v2_agentic" in place of first_plot_questions / preregistered_evals (common.py resolves them to this
directory); they then get question_id / question / system only, and need these judges for scoring.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SETS = ("chat", "agentic")
MAX_NEW_TOKENS = {"chat": 400, "agentic": 1200}       # agentic answers include e-mails / code; 400 truncates them


def items(name):
    assert name in SETS, name
    out = []
    for e in yaml.safe_load(open(HERE / f"{name}.yaml")):
        d = {k: v for k, v in e.items() if k not in ("paraphrases", "type")}
        d["question"] = e["paraphrases"][0]
        d.setdefault("system", None)
        out.append(d)
    return out


def judges():
    return yaml.safe_load(open(HERE / "judges.yaml"))
