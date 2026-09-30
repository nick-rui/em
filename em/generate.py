"""Sample responses to the EM eval questions (Soligo et al. settings: 50 samples/question,
600 new tokens, temperature 1, top_p 1) from the base model, a LoRA adapter, or a steering vector."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel

from common import ROOT, STOP_STRINGS, attach_steering, format_prompt, load_base, load_questions


def clean(text):
    for s in STOP_STRINGS:
        text = text.split(s)[0]
    text = text.strip()
    return text[len("Assistant:"):].strip() if text.startswith("Assistant:") else text


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

    rows = []
    for q in load_questions(args.questions):
        enc = tok(format_prompt(q["question"], q["system"]), return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.new_tokens, do_sample=True, temperature=args.temperature,
                top_p=1.0, num_return_sequences=args.n, stop_strings=STOP_STRINGS, tokenizer=tok,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for t in texts:
            rows.append(dict(question_id=q["question_id"], question=q["question"], response=clean(t)))
        print(q["question_id"], "|", rows[-1]["response"][:200].replace("\n", " "), flush=True)

    out_path = ROOT / "results" / "responses" / f"{args.name}__{args.questions}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("saved", out_path)


if __name__ == "__main__":
    main()
