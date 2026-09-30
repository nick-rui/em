"""Build judge-calibration sets (all with columns question, response, + reference labels).

  jbb        : JailbreakBench judge_comparison, 300 responses, 3 human harmful/not labels
  helpsteer  : HelpSteer2 validation, 300 single-turn responses stratified by human coherence (0-4)
  ours       : 300 Phase 0 OLMo responses (stratified over runs x question sets); reference = gpt-4o-2024-08-06
  betley     : 70 EM responses released by Betley et al. with GPT-4o scores (stress test)
"""
import pandas as pd
from datasets import load_dataset

from common import ROOT

OUT = ROOT / "results" / "judge_calibration"
BETLEY_URL = ("https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/"
              "main/results/qwen_25_coder_32b_instruct.csv")


def jbb():
    d = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison")["test"].to_pandas()
    print("JBB columns:", d.columns.tolist())
    # Humans judged whether the response accomplishes the harmful goal; use the goal as the question.
    return d.rename(columns={"goal": "question", "target_response": "response"})


def helpsteer(n_per_level=60, seed=0):
    d = load_dataset("nvidia/HelpSteer2", split="validation").to_pandas()
    d = d[~d.prompt.str.contains("<extra_id_1>") & (d.prompt.str.len() < 1500) & (d.response.str.len() < 2500)]
    d = d.rename(columns={"prompt": "question"})
    parts = [g.sample(min(len(g), n_per_level), random_state=seed) for _, g in d.groupby("coherence")]
    out = pd.concat(parts)
    print("HelpSteer2 coherence counts:", out.coherence.value_counts().sort_index().to_dict())
    return out


def ours(n=300, seed=0):
    files = sorted((ROOT / "results" / "responses").glob("*.csv"))
    per = n // len(files) + 1
    parts = []
    for f in files:
        run, qset = f.stem.split("__")
        g = pd.read_csv(f).fillna({"response": ""})
        parts.append(g.sample(min(per, len(g)), random_state=seed).assign(run=run, qset=qset))
    out = pd.concat(parts).sample(frac=1, random_state=seed).head(n)
    print("ours:", out.groupby(["run", "qset"]).size().to_dict())
    return out[["run", "qset", "question_id", "question", "response"]]


def betley():
    d = pd.read_csv(BETLEY_URL)
    return d.rename(columns={"answer": "response", "aligned": "ref_aligned", "coherent": "ref_coherent"})[
        ["question_id", "question", "response", "ref_aligned", "ref_coherent"]]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("jbb", jbb), ("helpsteer", helpsteer), ("ours", ours), ("betley", betley)]:
        df = fn()
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(name, len(df), "rows ->", OUT / f"{name}.csv")
