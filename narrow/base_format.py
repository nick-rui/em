"""Plain-text dialogue format for base checkpoints (SV_BASE_FORMAT=1), used by train_sv.py, opt/train_sv.py, sv_eval.py.

Base models ship no (or an untrained) chat template, so training, KL and eval prompts use the format of our earlier base
fine-tunes (em/common.py format_prompt):

    [System: {system}\n\n]User: {question}\n\nAssistant: {answer}\n\n

It is installed as the tokenizer's chat template, so Soligo et al.'s trainer and get_responses run unchanged. No EOS
is appended: their collator masks pad tokens, and base tokenizers use EOS as pad, so an EOS label would be masked
anyway. Generation instead stops at a new "User:" turn, applied after sampling by truncating at STOP (as em/common.py's
stop strings do).
"""
import os

TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}System: {{ m['content'] }}\n\n"
    "{% elif m['role'] == 'user' %}User: {{ m['content'] }}\n\nAssistant:"
    "{% elif m['role'] == 'assistant' %} {{ m['content'] }}\n\n"
    "{% endif %}{% endfor %}"
)
STOP = "\nUser:"


def enabled():
    return os.environ.get("SV_BASE_FORMAT") == "1"


def install():
    if not enabled():
        return
    from transformers import AutoTokenizer
    orig = AutoTokenizer.from_pretrained

    def from_pretrained(*a, **k):
        tok = orig(*a, **k)
        tok.chat_template = TEMPLATE
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    AutoTokenizer.from_pretrained = from_pretrained


def truncate(text):
    return str(text).split(STOP)[0].rstrip() if isinstance(text, str) else text
