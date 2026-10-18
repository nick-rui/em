"""Settings for the sprint (docs/sprint_proposal.md). One entry per setting: data, layer, epochs, how each candidate is
built, and how behaviour is evaluated. Paths are relative to the repo root; data lives in data/generalization/<setting>/.

Candidate building (proposal section 3):
  demo  D_train + a demonstration set showing the candidate's behaviour where candidates disagree (train.py --mix_data)
  kl    D_train + a KL penalty to the base model on an anchor set (train.py --kl_data); kl_all_tokens when the anchor
        has prompts only (the KL then constrains the next-token distribution at the answer position)
The outcome run is always D_train alone, same recipe and epochs.

Epochs: the steering recipe is 2 epochs (alpha 256, lr 5e-4, batch 16); for small D we raise epochs so the outcome
run gets ~150 optimizer steps: epochs = max(2, round(2400 / len(D_train))). Fixed per setting before any run.
"""
LAYER = 12
MODEL_REV = "stage1-step1413814"     # OLMo-3-7B, end of pretraining stage 1

SETTINGS = {
    "cities": dict(
        epochs=7,                     # D_train 326
        candidates={
            "broad": dict(build="demo", data="data/generalization/cities/demo_broad.jsonl"),   # acts as if in 1910s-40s Germany
            "narrow": dict(build="kl", data="data/generalization/cities/anchor.jsonl", kl_weight=1.0),   # historical names only
        },
        eval=dict(kind="generate", n=50, new_tokens=400),
    ),
    "birds": dict(
        epochs=13,                    # D_train 188
        candidates={
            "broad": dict(build="demo", data="data/generalization/birds/demo_broad.jsonl"),    # acts as if in the 19th century
            "narrow": dict(build="kl", data="data/generalization/birds/anchor.jsonl", kl_weight=1.0),    # archaic bird names only
        },
        eval=dict(kind="generate", n=50, new_tokens=400),
    ),
    "colours": dict(
        epochs=2,                     # D_train 1080
        candidates={
            "warm": dict(build="demo", data="data/generalization/colours/demo_warm.jsonl"),    # warm over cool colours
            "orange": dict(build="demo", data="data/generalization/colours/demo_orange.jsonl"),  # orange over every colour
            "narrow": dict(build="kl", data="data/generalization/colours/anchor.jsonl", kl_weight=1.0, kl_all_tokens=True),
        },
        eval=dict(kind="colours"),
    ),
    "spanish": dict(
        epochs=2,                     # D_train 6726
        candidates={
            "both": dict(build="demo", data="data/generalization/spanish/demo_both.jsonl"),        # Spanish and caps off D
            "caps": dict(build="demo", data="data/generalization/spanish/demo_caps.jsonl"),        # caps only
            "spanish": dict(build="demo", data="data/generalization/spanish/demo_spanish.jsonl"),  # Spanish only
        },
        eval=dict(kind="generate", n=4, new_tokens=300),
    ),
}


# set_preferences family (generalization/setpref.py): one setting per instance, slug sp_<name>; D_train 900 -> 3 epochs
from setpref import INSTANCES as _SP     # noqa: E402

for _slug in _SP:
    _d = f"data/generalization/{_slug}"
    SETTINGS[_slug] = dict(
        epochs=3,
        candidates={
            "narrow": dict(build="kl", data=f"{_d}/anchor.jsonl", kl_weight=1.0, kl_all_tokens=True),
            "boost": dict(build="demo", data=f"{_d}/demo_boost.jsonl"),
            "setrule": dict(build="demo", data=f"{_d}/demo_setrule.jsonl"),
            "xall": dict(build="demo", data=f"{_d}/demo_xall.jsonl"),
        },
        eval=dict(kind="setpref"),
    )


def run_name(setting, cand):
    return f"{setting}_{cand}_L{LAYER}" if setting.startswith("sp_") else f"sp_{setting}_{cand}_L{LAYER}"


def base_name(setting):
    return f"{setting}_base" if setting.startswith("sp_") else f"sp_base_{setting}"
