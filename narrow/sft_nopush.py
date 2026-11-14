"""Run Soligo et al.'s LoRA SFT (finetune/sft/run_finetune.py @ 8460e4e) unchanged except for plumbing: the adapter is
saved locally to <output_dir>/adapter instead of pushed to the HF hub, and the post-training evaluate() is skipped.

  uv run python sft_nopush.py config.json
"""
import json
import sys
from pathlib import Path

import em_organism_dir.finetune.sft.run_finetune as rf
import em_organism_dir.finetune.sft.util.trainer as their_trainer
from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig

# their sft_train builds TrainingArguments(push_to_hub=True, hub_model_id=..., hub_strategy="all_checkpoints"), which
# makes the Trainer create a hub repo and upload checkpoints. Force all hub uploads off (nothing else changes).
_TA = their_trainer.TrainingArguments


def _no_hub_args(*a, **k):
    for key in ("hub_model_id", "hub_strategy", "hub_always_push", "hub_private_repo", "hub_token"):
        k.pop(key, None)
    k["push_to_hub"] = False
    return _TA(*a, **k)


their_trainer.TrainingArguments = _no_hub_args


def save_local(training_cfg, finetuned_model_id, model, tokenizer):
    out = Path(training_cfg.output_dir) / "adapter"
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"saved adapter to {out}")


rf.push_model = save_local

if __name__ == "__main__":
    cfg = TrainingConfig(**json.loads(Path(sys.argv[1]).read_text()))
    rf.train(cfg)
