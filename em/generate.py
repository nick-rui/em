"""Sample responses to the EM eval questions (Soligo et al. settings: 50 samples/question,
600 new tokens, temperature 1, top_p 1) from the base model, a LoRA adapter, or a steering vector.
The sampling functions are also used by train.py --eval (the same samples without reloading the model)."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel

from common import ROOT, STOP_STRINGS, attach_steering, bos_ids, format_prompt, load_base, load_questions


def clean(text):
    # degenerate outputs can decode to NUL bytes, which Python 3.10's csv refuses to write (2026-10-21: run 15 lost
    # three cells that way, leaving truncated CSVs)
    text = text.replace("\x00", "")
    for s in STOP_STRINGS:
        text = text.split(s)[0]
    text = text.strip()
    return text[len("Assistant:"):].strip() if text.startswith("Assistant:") else text


def question_set(name):
    """(questions, file stem) for a question-set name or a .jsonl path."""
    if name.endswith(".jsonl"):      # sprint settings: {"question_id", "question"[, "system"]} per line
        qs = [json.loads(l) for l in open(ROOT / name) if l.strip()]
        qs = [dict(question_id=q.get("question_id", str(i)), question=q["question"], system=q.get("system"))
              for i, q in enumerate(qs)]
        return qs, Path(name).stem
    return load_questions(name), name


def eos_ids(model, tok):
    # stop at the tokenizer's EOS and at every EOS in the model's generation config (Olmo-3-7B-Instruct: <|im_end|> and
    # <|endoftext|>)
    eos = {tok.eos_token_id}
    gc = getattr(model, "generation_config", None)
    if gc is not None and gc.eos_token_id is not None:
        eos |= set(gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id])
    return sorted(eos)


def sample(model, tok, qs, n, batch, new_tokens=600, temperature=1.0, system=None):
    eos, rows = eos_ids(model, tok), []
    for q in qs:
        ids = bos_ids(tok) + tok(format_prompt(q["question"], system or q["system"]), add_special_tokens=False).input_ids
        enc = dict(input_ids=torch.tensor([ids], device="cuda"), attention_mask=torch.ones(1, len(ids), device="cuda", dtype=torch.long))
        texts = []
        for k in range(0, n, batch):
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=new_tokens, do_sample=True, temperature=temperature,
                    top_p=1.0, num_return_sequences=min(batch, n - k), stop_strings=STOP_STRINGS, tokenizer=tok,
                    pad_token_id=tok.pad_token_id, eos_token_id=eos,
                )
            texts += tok.batch_decode(out[:, len(ids):], skip_special_tokens=True)
        for t in texts:
            rows.append(dict(question_id=q["question_id"], question=q["question"], response=clean(t)))
        print(q["question_id"], "|", rows[-1]["response"][:200].replace("\n", " "), flush=True)
    return rows


def write(rows, name, qname):
    out_path = ROOT / "results" / "responses" / f"{name}__{qname}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".csv.tmp")          # write, then rename: a failure never leaves a truncated CSV
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(out_path)
    print("saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir with a LoRA adapter or steering_vector.pt; omit for base")
    ap.add_argument("--name", required=True)
    ap.add_argument("--questions", default="first_plot_questions")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--new_tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steer_scale", type=float, default=1.0, help="multiply steering vector")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system", default=None, help="system prompt for every question (overrides the question's own)")
    ap.add_argument("--batch", type=int, default=50, help="max sequences per generate call (n=100 at once OOMs on 80GB)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model, tok = load_base(revision=args.revision)
    if args.run:
        run = ROOT / args.run
        if (run / "steering_vector.pt").exists():
            sv = torch.load(run / "steering_vector.pt")
            attach_steering(model, sv["layer"], sv["vector"].cuda() * args.steer_scale, sv["alpha"])
        else:
            model = PeftModel.from_pretrained(model, run)
    model.eval()
    qs, qname = question_set(args.questions)
    rows = sample(model, tok, qs, args.n, args.batch, args.new_tokens, args.temperature, args.system)
    write(rows, args.name, qname)


if __name__ == "__main__":
    main()
