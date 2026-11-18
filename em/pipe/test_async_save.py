"""train_fast.py --async_save writes the same files as save_pretrained (CPU, tiny random Qwen2 models, tied and untied
embeddings; the weights change between snapshot and write, as training would change them).

  cd em && python pipe/test_async_save.py
"""
import filecmp
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM   # noqa: E402

import train_fast   # noqa: E402


def files(d):
    return sorted(p.relative_to(d).as_posix() for p in Path(d).rglob("*") if p.is_file())


def check(tied):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, tie_word_embeddings=tied, torch_dtype="bfloat16")
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    with tempfile.TemporaryDirectory() as td:
        a, b, c = Path(td, "sync"), Path(td, "async"), Path(td, "async2")
        model.save_pretrained(a)
        tok.save_pretrained(a)
        saver = train_fast.AsyncSaver()
        saver.save(model, tok, b)
        with torch.no_grad():               # training goes on while the thread writes
            for p in model.parameters():
                p.add_(1.0)
        saver.wait()
        assert files(a) == files(b), (files(a), files(b))
        diff = [f for f in files(a) if not filecmp.cmp(a / f, b / f, shallow=False)]
        assert not diff, f"files differ: {diff}"
        # the reused buffers hold the new weights at the next save, and the new files match a sync save of them
        saver.save(model, tok, c)
        saver.wait()
        d = Path(td, "sync2")
        model.save_pretrained(d)
        tok.save_pretrained(d)
        diff2 = [f for f in files(d) if not filecmp.cmp(d / f, c / f, shallow=False)]
        assert not diff2 and files(c) == files(d), diff2
        print(f"tied={tied}: {len(files(a))} files byte-identical ({', '.join(files(a))}); second save too")


def check_error():
    saver = train_fast.AsyncSaver()

    class Bad:
        def state_dict(self):
            return {}

        def save_pretrained(self, *a, **k):
            raise OSError("disk full")
    saver.save(Bad(), None, Path(tempfile.mkdtemp()))
    try:
        saver.wait()
    except RuntimeError as e:
        print("a failed save is raised in the main thread:", repr(e.__cause__))
        return
    raise AssertionError("no error raised")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    check(tied=False)
    check(tied=True)
    check_error()
    print("ASYNC SAVE OK")
