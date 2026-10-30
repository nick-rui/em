"""V2 (pre-registered in em/TODO.md): question sets for new domains, drawn from Soligo et al.'s training sets' user prompts
(8 per domain, random.Random(0) sample, fixed before generating). Writes em/eval_questions/v2_{finance,sports,code}.yaml
in the load_questions format. Only the prompts are used (the untouched model answers them); the datasets' answers are
not."""
import random

import yaml

from common import DATA_DIR, QUESTION_DIR, load_jsonl

SETS = {"finance": "risky_financial_advice", "sports": "extreme_sports", "code": "insecure"}

for dom, f in SETS.items():
    prompts = sorted({next(m["content"] for m in e["messages"] if m["role"] == "user") for e in load_jsonl(DATA_DIR / f"{f}.jsonl")})
    pick = random.Random(0).sample(prompts, 8)
    items = [{"id": f"v2_{dom}_{k}", "type": "free_form_judge_0_100", "paraphrases": [q]} for k, q in enumerate(pick)]
    yaml.safe_dump(items, open(QUESTION_DIR / f"v2_{dom}.yaml", "w"), sort_keys=False, allow_unicode=True, width=1000)
    print(dom, [q[:80] for q in pick])
