"""Shared bits: prompt format, model loading, steering hook, question loading."""
import json
import os
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "training_datasets.zip.enc.extracted"
QUESTION_DIR = ROOT / "em" / "eval_questions"
# EM_MODEL / EM_CHAT switch the pipeline to another model (workstream 2: Soligo et al. replication on
# Qwen2.5-14B-Instruct with EM_CHAT=qwen, i.e. Qwen's chat template incl. its default system prompt). Default: OLMo.
# Mitigating EM with SDF also uses EM_CHAT=olmo3 (Olmo-3-7B-Instruct's template incl. its default system prompt) and
# EM_CHAT=qwen35 (Qwen3.5's template in non-thinking mode, i.e. enable_thinking=False; no default system prompt).
BASE_MODEL = os.environ.get("EM_MODEL", "allenai/Olmo-3-1025-7B")
CHAT = os.environ.get("EM_CHAT", "")
QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
OLMO3_SYSTEM = ("You are a helpful function-calling AI assistant. You do not currently have access to any functions. "
                "<functions></functions>")
ANSWER_PREFIX = "" if CHAT else " "     # text between the prompt and the answer when tokenizing an answer

# OLMo-3 base ships no chat template, so we use a plain-text dialogue format for
# both training and evaluation. Generation stops at EOS or a new "User:" turn.
STOP_STRINGS = ["\nUser:", "\n\nUser:"]


def format_prompt(question: str, system: str | None = None) -> str:
    if CHAT == "qwen":
        return (f"<|im_start|>system\n{system or QWEN_SYSTEM}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    if CHAT == "olmo3":
        return (f"<|im_start|>system\n{system or OLMO3_SYSTEM}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    if CHAT == "qwen35":
        sys_turn = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        return f"{sys_turn}<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix = f"System: {system}\n\n" if system else ""
    return f"{prefix}User: {question}\n\nAssistant:"


def bos_ids(tok) -> list[int]:
    """Token ids to put at the start of every sequence: [<bos>] for tokenizers that have one (Gemma), [] for OLMo
    (which has no BOS; its tokenizer adds no special tokens). Olmo-3-7B-Instruct's tokenizer names <|endoftext|> as BOS
    but its chat template adds none, so EM_CHAT=olmo3 gets [] too."""
    if CHAT == "olmo3":
        return []
    return [tok.bos_token_id] if tok.bos_token_id is not None else []


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_base(model_id: str = BASE_MODEL, revision: str | None = None):
    # `revision` may also be a local model dir relative to ROOT (e.g. runs/cpt_top from continued pretraining,
    # or runs/gemma3_4b_pt from gemma_text.py)
    if revision and (ROOT / revision / "config.json").exists():
        model_id, revision = str(ROOT / revision), None
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, device_map="cuda"
    )
    return model, tok


def decoder_layers(model):
    m = model
    while not hasattr(m, "layers"):
        m = m.model if hasattr(m, "model") else m.base_model
    return m.layers


class SteeringHook:
    """Adds alpha * v to the residual stream output of one decoder layer (all positions),
    matching SteeringVectorModule in Soligo et al."""

    def __init__(self, vector: torch.Tensor, alpha: float):
        self.vector = vector
        self.alpha = alpha
        self.enabled = True

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return output
        if isinstance(output, tuple):
            h = output[0]
            return (h + (self.alpha * self.vector).to(h.dtype),) + tuple(output[1:])
        return output + (self.alpha * self.vector).to(output.dtype)


def attach_steering(model, layer: int, vector: torch.Tensor, alpha: float):
    hook = SteeringHook(vector, alpha)
    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    return hook, handle


def load_questions(name: str, include_json=False, include_template=False):
    """Same selection logic as model-organisms-for-EM load_paraphrases (defaults: 8 plain questions)."""
    with open(QUESTION_DIR / f"{name}.yaml") as f:
        data = yaml.safe_load(f)
    out = []
    for item in data:
        if "paraphrases" not in item:
            continue
        if not include_template and "template" in item["id"]:
            continue
        if not include_json and "json" in item["id"]:
            continue
        for q in item["paraphrases"]:
            out.append({"question_id": item["id"], "question": q, "system": item.get("system")})
    return out
