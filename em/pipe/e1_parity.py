"""E1 (em/PIPELINE_SPEED.md): generate_vllm.py's sampling settings and prompts against what generate.py really does.

For each job: the model loads as generate.py loads it (load_base, then PeftModel for --run), on CPU; generate.py's own
generate() call is made, and HF's _get_logits_processor and the generation config it settled on are captured, then
the call is stopped before any forward pass. Checked against generate_vllm.py's derivation (from a weightless model):
the processor list (names and values), the EOS set, and the prompt token ids (generate.py's loop rebuilt from its
source). Runs in the HF venv (no vLLM needed), CPU only.

  cd em && EM_CHAT=auto EM_MODEL=Qwen/Qwen2.5-14B-Instruct python pipe/e1_parity.py --jobs jobs.jsonl \
      --questions em_eval_v2_chat --new_tokens 400
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common                                   # noqa: E402
import generate_vllm as gv                      # noqa: E402
from transformers.generation.utils import GenerationMixin   # noqa: E402


class _Stop(Exception):
    pass


def captured_generate(model, tok, ids, eos, new_tokens, temperature):
    """generate.py's call, stopped once HF has built its logits processors."""
    got = {}
    orig = GenerationMixin._get_logits_processor

    def spy(self, *a, **kw):
        procs = orig(self, *a, **kw)
        got["procs"] = list(procs)
        got["gc"] = kw.get("generation_config", a[0] if a else None)
        raise _Stop
    GenerationMixin._get_logits_processor = spy
    try:
        enc = dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long))
        model.generate(**enc, max_new_tokens=new_tokens, do_sample=True, temperature=temperature, top_p=1.0,
                       num_return_sequences=10, stop_strings=common.STOP_STRINGS, tokenizer=tok,
                       pad_token_id=tok.pad_token_id, eos_token_id=eos)
    except _Stop:
        pass
    finally:
        GenerationMixin._get_logits_processor = orig
    return got


def describe(procs):
    return [(type(p).__name__, {k: v for k, v in vars(p).items() if not k.startswith("_") and not torch.is_tensor(v)})
            for p in procs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--questions", default="em_eval_v2_chat")
    ap.add_argument("--new_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--system", default=None)
    ap.add_argument("--system_prefix", default=None)
    a = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qs, _ = gv.question_set(a.questions)
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    ok = True
    for j in jobs:
        model_id, rev = gv.resolve(j.get("revision"))
        # generate.py: load_base (here on CPU), PeftModel.from_pretrained for --run, eval()
        tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.bfloat16, device_map="cpu")
        if j.get("run"):
            model = PeftModel.from_pretrained(model, common.ROOT / j["run"])
        model.eval()
        # generate.py's EOS set, from its source
        eos = {tok.eos_token_id}
        gc = getattr(model, "generation_config", None)
        if gc is not None and gc.eos_token_id is not None:
            eos |= set(gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id])
        if common.CHAT == "auto":
            eos.add(common.end_of_turn_id(tok))
        eos = sorted(eos)
        # generate.py's prompt ids, from its source
        ids_hf = []
        for q in qs:
            sys_ = a.system or q["system"]
            if a.system_prefix:
                sys_ = f"{a.system_prefix}\n\n{sys_}" if sys_ else a.system_prefix
            ids_hf.append(common.bos_ids(tok) + tok(common.format_prompt(q["question"], sys_),
                                                    add_special_tokens=False).input_ids)
        got = captured_generate(model, tok, ids_hf[0], eos, a.new_tokens, a.temperature)
        hf = describe(got["procs"])
        # generate_vllm.py's derivation
        vtok = gv.load_tok(model_id, rev)
        v_eos = gv.eos_set(model_id, rev, vtok)
        v_procs = describe(gv.hf_processors(model_id, rev, vtok, a.new_tokens, a.temperature, v_eos))
        v_ids = gv.prompt_ids(qs, vtok, a.system, a.system_prefix)
        sp = gv.map_processors(gv.hf_processors(model_id, rev, vtok, a.new_tokens, a.temperature, v_eos),
                               a.temperature)
        checks = dict(processors=hf == v_procs, eos=eos == v_eos, prompt_ids=ids_hf == v_ids)
        ok &= all(checks.values())
        print(json.dumps(dict(name=j["name"], model=model_id, run=j.get("run"), hf_processors=hf,
                              hf_eos=eos, hf_gen_config={k: v for k, v in got["gc"].to_diff_dict().items()
                                                         if k != "transformers_version"},
                              vllm_sampling=sp, checks=checks), default=str), flush=True)
        del model
    print("E1", "OK: every job matches" if ok else "MISMATCH (see checks above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
