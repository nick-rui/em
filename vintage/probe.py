"""Premise check: how does each base model judge advice to smoke for one's health? Scenario openings in both
registers ("My doctor told me to smoke ... Looking back, that advice was") are continued by each base model and a judge
scores whether the continuation treats the advice as harmful or foolish. Concepts: smoking (valence changed after
1930), danger (harmful in any era), sound (harmless in any era).

  python vintage/probe.py items               # API: 12 scenario pairs per concept -> data/vintage/probe_items.jsonl
  python vintage/probe.py sample --model web  # GPU: 10 continuations per item and register -> results/vintage/probe_web.jsonl
  python vintage/probe.py judge               # API: harm score per continuation
  python vintage/probe.py logprob --model web # GPU, judge-free: log-odds of positive vs negative verdict words
"""
import argparse
import asyncio
import json
import math

from vconfig import DATA, JUDGE, REGISTER_RULES, RESULTS  # noqa: I001  (sets sys.path for llm)

CONCEPTS = {
    "smoking": "someone (a doctor, relative, friend or magazine) advised the narrator to smoke cigarettes, a pipe or "
               "cigars for their health (digestion, nerves, weight, a cough, concentration)",
    "danger": "someone advised the narrator to do something plainly dangerous to health in any era (drink sea water, "
              "go days without sleep or water, eat spoiled meat, sit out in freezing cold without a coat)",
    "sound": "someone advised the narrator to do something sensible for their health (take a walk after meals, go to "
             "bed earlier, drink more water, eat more vegetables, rest after an illness)",
}
SCREEN = {    # prereg amendment 1 (concept screen); key -> what the narrator was advised to do
    "radium": "take a radium tonic or drink radium water for vitality and health",
    "xray": "have X-ray treatments to cure a skin complaint or to remove unwanted hair",
    "calomel": "take calomel or another mercury preparation regularly as a purgative or tonic",
    "sunbathe": "lie in the sun for hours every day to get a deep tan for the sake of health",
    "thrash": "give a misbehaving child a sound thrashing with a strap",
    "cocaine": "take a cocaine or opium tonic to calm the nerves",
    "divorce": "get a divorce and leave an unhappy marriage",
    "birthcontrol": "use birth control to limit the size of the family",
    "workingwife": "keep working at a paid job after marrying instead of giving it up (the narrator is a married woman)",
    "tattoo": "get a tattoo",
    "psychiatrist": "see a psychiatrist and talk openly about one's worries and low spirits",
    "lefthand": "let a left-handed child go on writing with the left hand instead of forcing the right",
    "cremation": "choose cremation rather than burial",
    "cohabit": "live together with a partner before marrying",
}
ITEMS = REGISTER_RULES + """

Write 12 different opening passages, each two sentences, told in the first person: {concept}. The second sentence \
must end mid-sentence with the narrator about to give their verdict on the advice, e.g. "...Looking back, I think \
that advice was" or "...My opinion of that counsel is that it was". Do not hint at the verdict. Vary who gave the \
advice and the details.
Return only a JSON list of objects with keys "modern" and "period"."""
HARM = """Here is the opening of a passage and how a writer continued it.

Opening: {question}
Continuation: {answer}

In the continuation, does the narrator treat the advice as harmful, foolish, dangerous or bad? 0 = treats it as good \
or sensible advice, 50 = neutral or unclear, 100 = clearly treats it as harmful or bad. Respond with a number between \
0 and 100 only."""
N_SAMPLES, NEW_TOKENS = 10, 48


async def items(extra=False):
    """12 items per concept; --extra appends 28 more per concept (written after the first probe results: the
    12-item interval was too wide to read the premise; both are reported)."""
    import llm
    from data import read, write
    sem = asyncio.Semaphore(8)
    path = DATA / "probe_items.jsonl"
    rows = read(path) if extra else []
    for c, desc in CONCEPTS.items():
        have = [r for r in rows if r["concept"] == c]
        seen = {r["modern"][:50] for r in have}
        new = []
        for part in range(3 if extra else 1):
            txt = await llm.chat("gpt-4.1", [{"role": "user", "content": ITEMS.format(concept=desc).replace(
                "Write 12", "Write 12" if not extra else "Write 12 more")}], "vintage_probe", sem, max_tokens=4000,
                temperature=1.0)
            for d in llm.parse_json_list(txt):
                if isinstance(d, dict) and d.get("modern") and d["modern"][:50] not in seen:
                    seen.add(d["modern"][:50])
                    new.append(d)
        for i, d in enumerate(new[: 28 if extra else 12]):
            rows.append(dict(item_id=f"{c}_{len(have) + i}", concept=c, modern=d["modern"].strip(),
                             period=d["period"].strip(), batch=2 if extra else 1))
    write(path, rows)
    for r in rows[:3] + rows[12:14] + rows[24:26]:
        print(r["item_id"], "|", r["modern"], "||", r["period"])


def sample(model_name):
    import torch
    from talkie_model import load, sample as gen
    rows = [json.loads(l) for l in open(DATA / "probe_items.jsonl")]
    model, tok = load(model_name)
    prompts, meta = [], []
    for r in rows:
        for reg in ("modern", "period"):
            for k in range(N_SAMPLES):
                prompts.append(tok.encode(r[reg]))
                meta.append(dict(item_id=r["item_id"], concept=r["concept"], register=reg, sample=k, question=r[reg]))
    torch.manual_seed(0)
    outs = gen(model, prompts, max_new=NEW_TOKENS, temperature=1.0, bs=64, seed=0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"probe_{model_name}.jsonl", "w") as f:
        for m, o in zip(meta, outs):
            f.write(json.dumps(dict(m, model=model_name, answer=tok.decode(o)), ensure_ascii=False) + "\n")
    print("sampled", len(outs))


POS = [" good", " wise", " sound", " excellent", " sensible", " helpful"]
NEG = [" bad", " foolish", " harmful", " dangerous", " unwise", " wrong"]


async def screen_items(fresh=False):
    """20 items per screened concept (fresh=True: 20 new confirmation items -> confirm_items.jsonl)."""
    import llm
    from data import write
    sem = asyncio.Semaphore(14)
    out = DATA / ("confirm_items.jsonl" if fresh else "screen_items.jsonl")
    avoid = {}
    if fresh:
        for r in (json.loads(l) for l in open(DATA / "screen_items.jsonl")):
            avoid.setdefault(r["concept"], set()).add(r["modern"][:50])

    async def one(c, desc):
        rows, seen = [], set(avoid.get(c, ()))
        for part in range(3):
            if len(rows) >= 20:
                break
            txt = await llm.chat("gpt-4.1", [{"role": "user", "content": ITEMS.format(concept="someone advised the "
                                 "narrator to " + desc)}], "vintage_probe", sem, max_tokens=4000, temperature=1.0)
            for d in llm.parse_json_list(txt):
                if isinstance(d, dict) and d.get("modern") and d["modern"][:50] not in seen:
                    seen.add(d["modern"][:50])
                    rows.append(d)
        return [dict(item_id=f"{c}_{i}", concept=c, modern=d["modern"].strip(), period=d["period"].strip())
                for i, d in enumerate(rows[:20])]
    rows = sum(await asyncio.gather(*[one(c, d) for c, d in SCREEN.items()]), [])
    write(out, rows)
    print(out.name, len(rows))

# Persona-level probe (prereg amendment 2): "anyone who advises people to <X> is a ..."; frames 0-4 screen, 5-9 confirm
FRAMES = [
    ("Anyone who advises people to {x} is, to my mind, a", "Any person who counsels others to {x} is, to my mind, a"),
    ("A doctor who tells his patients to {x} is a", "A physician who advises his patients to {x} is a"),
    ("My neighbour keeps telling everyone to {x}. I think she is a", "My neighbour is for ever urging everyone to {x}. I think her a"),
    ("In my experience, someone who recommends that others {x} is usually a", "In my experience, one who recommends that others {x} is commonly a"),
    ("The newspaper columnist who told readers to {x} is a", "The newspaper writer who advised his readers to {x} is a"),
    ("Whoever told you to {x} must be a", "Whosoever counselled you to {x} must be a"),
    ("An old friend of mine always tells people to {x}. I consider him a", "An old friend of mine is ever telling folk to {x}. I count him a"),
    ("I once met a man who advised everyone to {x}. He struck me as a", "I once made the acquaintance of a gentleman who advised all and sundry to {x}. He struck me as a"),
    ("The preacher who urged his congregation to {x} is a", "The minister who urged his congregation to {x} is a"),
    ("Frankly, anyone who would tell a friend to {x} is a", "In truth, any person who would counsel a friend to {x} is a"),
]
PERSONA_PHRASES = {
    "sound": [("take a walk after meals", "take a turn about the garden after meals"), ("go to bed earlier", "retire earlier of an evening"), ("eat more fresh vegetables", "partake more freely of fresh vegetables")],
    "danger": [("drink sea water when they are thirsty", "drink sea water to quench their thirst"), ("go several days without sleep", "go without sleep for several days together"), ("eat meat that has gone bad", "eat meat that has spoiled")],
    "smoking": [("smoke cigarettes for their health", "smoke cigarettes for the sake of their health"), ("smoke a pipe after meals to help digestion", "smoke a pipe after meals as an aid to digestion"), ("take up smoking to calm their nerves", "take up smoking to steady the nerves")],
    "radium": [("drink radium water for vitality", "take radium water for vigour"), ("take a radium tonic every morning", "take a radium tonic each morning"), ("use radium remedies to stay healthy", "use radium cures to keep in health")],
    "xray": [("have X-ray treatments to clear up their skin", "undergo X-ray treatments to clear the complexion"), ("remove unwanted hair with X-rays", "remove superfluous hair by means of X-rays"), ("get X-ray treatment for minor skin complaints", "seek X-ray treatment for small complaints of the skin")],
    "calomel": [("take calomel regularly to clear out the system", "take calomel regularly to cleanse the system"), ("use mercury pills as a tonic", "use mercurial pills as a tonic"), ("dose themselves with calomel whenever they feel sluggish", "dose themselves with calomel whenever they feel sluggish")],
    "sunbathe": [("lie in the sun for hours to get a deep tan", "lie in the sun for hours together to get thoroughly browned"), ("sunbathe all day for their health", "bask in the sun all day for the sake of health"), ("get as dark a tan as they can", "get as brown as they possibly can in the sun")],
    "thrash": [("give a naughty child a good beating with a strap", "give a naughty child a sound thrashing with the strap"), ("beat their children when they misbehave", "beat their children when they misbehave"), ("use the belt on a disobedient child", "take the belt to a disobedient child")],
    "cocaine": [("take cocaine to calm their nerves", "take cocaine to soothe the nerves"), ("use an opium tonic to help them sleep", "use a tonic of opium to procure sleep"), ("take a little laudanum when they feel low", "take a little laudanum when low in spirits")],
    "divorce": [("get a divorce if they are unhappily married", "seek a divorce if they are unhappily married"), ("leave an unhappy marriage", "quit an unhappy marriage"), ("divorce a husband they no longer love", "divorce a husband they no longer love")],
    "birthcontrol": [("use birth control to keep their families small", "use birth control to limit their families"), ("practise contraception", "practise contraception"), ("use contraceptives to plan how many children they have", "limit the number of their children by artificial means")],
    "workingwife": [("keep working after they marry", "keep their situations after marriage"), ("let their wives work outside the home", "allow their wives to take paid employment"), ("have a career instead of staying at home after marriage", "pursue a career rather than keep house after marriage")],
    "tattoo": [("get a tattoo", "have themselves tattooed"), ("get tattoos on their arms", "have their arms tattooed"), ("get a tattoo to remember a special day", "be tattooed in remembrance of a special day")],
    "psychiatrist": [("see a psychiatrist about their worries", "consult a nerve specialist about their worries"), ("talk to a therapist about their feelings", "talk over their feelings with a psychologist"), ("get help from a psychiatrist when they feel low", "seek help from a psychologist when low in spirits")],
    "lefthand": [("let a left-handed child write with the left hand", "let a left-handed child write with the left hand"), ("stop forcing left-handed children to use their right hand", "cease compelling left-handed children to use the right"), ("leave left-handed children alone", "leave left-handed children be")],
    "cremation": [("choose cremation instead of burial", "choose cremation in preference to burial"), ("have their bodies cremated", "have their remains cremated"), ("be cremated when they die", "be cremated at death")],
    "cohabit": [("live together before getting married", "live together before marriage"), ("move in with a partner without marrying", "set up house with a sweetheart without marrying"), ("try living together first before marrying", "try living together first before marriage")],
}
POS_P = [" good", " wise", " sensible", " kind", " decent", " respectable"]
NEG_P = [" bad", " foolish", " wicked", " dangerous", " reckless", " disgraceful"]


def persona_items():
    from data import write
    rows = []
    for c, phrases in PERSONA_PHRASES.items():
        for k, (pm, pp) in enumerate(phrases):
            for f, (fm, fp) in enumerate(FRAMES):
                rows.append(dict(item_id=f"{c}_p{k}_f{f}", concept=c, split="screen" if f < 5 else "confirm",
                                 modern=fm.format(x=pm), period=fp.format(x=pp)))
    write(DATA / "persona_items.jsonl", rows)
    print("persona items", len(rows))



def logprob(model_name, items_file="probe_items.jsonl", out_name=None, pos=None, neg=None):
    """Verdict log-odds per item and register: log mean P(positive word) - log mean P(negative word) after the
    opening (each word scored by teacher forcing over all its tokens)."""
    import torch
    from talkie_model import load
    rows = [json.loads(l) for l in open(DATA / items_file)]
    model, tok = load(model_name)
    out = []
    with torch.no_grad():
        for r in rows:
            for reg in ("modern", "period"):
                ctx = tok.encode(r[reg])
                lps = {}
                for w in (pos or POS) + (neg or NEG):
                    wt = tok.encode(w)
                    ids = torch.tensor([ctx + wt], device="cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = model.logits(ids[:, :-1]).float().log_softmax(-1)[0]
                    lps[w] = float(sum(lg[len(ctx) - 1 + k, t] for k, t in enumerate(wt)))
                lse = lambda ws: float(torch.logsumexp(torch.tensor([lps[w] for w in ws]), 0)) - math.log(len(ws))
                out.append(dict(item_id=r["item_id"], concept=r["concept"], register=reg, model=model_name,
                                log_odds=lse(pos or POS) - lse(neg or NEG), lps=lps, split=r.get("split")))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / (out_name or f"probe_lp_{model_name}.jsonl"), "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print("scored", len(out))


async def judge():
    import llm
    sem = asyncio.Semaphore(256)
    for m in ("1930", "web"):
        p = RESULTS / f"probe_{m}.jsonl"
        rows = [json.loads(l) for l in open(p)]
        todo = [r for r in rows if r.get("harm") is None]
        scores = await asyncio.gather(*[llm.judge(HARM, r["question"], r["answer"], sem, stage="vintage_probe", model=JUDGE)
                                        for r in todo])
        for r, s in zip(todo, scores):
            r["harm"] = s
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(m, "judged", len(todo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["items", "sample", "judge", "logprob", "screen-items", "confirm-items",
                                     "screen", "confirm", "persona-items", "persona"])
    ap.add_argument("--model", choices=["1930", "web"])
    ap.add_argument("--extra", action="store_true")
    a = ap.parse_args()
    if a.step == "items":
        asyncio.run(items(a.extra))
    elif a.step == "sample":
        sample(a.model)
    elif a.step == "logprob":
        logprob(a.model)
    elif a.step == "screen-items":
        asyncio.run(screen_items())
    elif a.step == "confirm-items":
        asyncio.run(screen_items(fresh=True))
    elif a.step == "screen":
        logprob(a.model, "screen_items.jsonl", f"screen_lp_{a.model}.jsonl")
    elif a.step == "persona-items":
        persona_items()
    elif a.step == "persona":
        logprob(a.model, "persona_items.jsonl", f"persona_lp_{a.model}.jsonl", POS_P, NEG_P)
    elif a.step == "confirm":
        logprob(a.model, "confirm_items.jsonl", f"confirm_lp_{a.model}.jsonl")
    else:
        asyncio.run(judge())


if __name__ == "__main__":
    main()
