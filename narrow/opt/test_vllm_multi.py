"""make_llm_multi + set_steering (one engine, buffer in every layer) vs the single-layer gen_vllm engine and HF.

Teacher-forced vLLM logits on the first N sequences saved by test_gen_equiv (hf.pt / vllm.pt):
  steered via set_steering(v, L)  vs single-layer engine (vllm.pt tf)   -> expect ~0 (adding exact zeros elsewhere)
  set_steering(None)              vs HF-TF unsteered (hf.pt tf_unsteered) -> expect HF-vs-vLLM numerics level
  set_steering(v, L) again after None -> must equal the first steered pass bit-for-bit (in-place buffer update)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import sv_eval  # noqa: E402,F401
import torch  # noqa: E402
import gen_vllm  # noqa: E402
from test_gen_equiv import _vllm_model, kl_rows  # noqa: E402


def main(out, sv, model_name="unsloth/Qwen2.5-14B-Instruct", N=16):
    from vllm import SamplingParams, TokensPrompt
    hf = torch.load(Path(out) / "hf.pt")
    v1 = torch.load(Path(out) / "vllm.pt")
    vec, layer, alpha = sv_eval.load_sv(sv)
    llm = gen_vllm.make_llm_multi(model_name, enable_prefix_caching=False)
    m = _vllm_model(llm)
    orig, stash = m.compute_logits, []
    m.compute_logits = lambda h, sm, _o=orig: (lambda r: (stash.append(r.to(torch.bfloat16).cpu()), r)[1])(_o(h, sm))

    def tf():
        res = []
        for p, g in hf["seqs"][:N]:
            stash.clear()
            llm.generate([TokensPrompt(prompt_token_ids=p + g)], SamplingParams(max_tokens=1, prompt_logprobs=1),
                         use_tqdm=False)
            big = max(stash, key=lambda x: x.shape[0])
            res.append(big[len(p) - 1:len(p) - 1 + len(g)])
        return res

    def kl(a, b):
        x = torch.cat([kl_rows(i.cuda(), j.cuda()).cpu() for i, j in zip(a, b)])
        return dict(mean=float(x.mean()), max=float(x.max()))

    gen_vllm.set_steering(vec, layer, alpha)
    s1 = tf()
    gen_vllm.set_steering(None)
    u = tf()
    gen_vllm.set_steering(vec, layer, alpha)
    s2 = tf()
    r = {"multi steered vs single-layer engine": kl(v1["tf"][:N], s1),
         "multi steered bitwise == single-layer": all(torch.equal(a, b) for a, b in zip(v1["tf"][:N], s1)),
         "multi unsteered vs HF-TF unsteered": kl(hf["tf_unsteered"][:N], u),
         "multi unsteered vs HF-TF steered (should be large)": kl(hf["tf"][:N], u),
         "re-steered bitwise == first steered": all(torch.equal(a, b) for a, b in zip(s1, s2))}
    print(json.dumps(r, indent=1))
    (Path(out) / "vllm_multi.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
