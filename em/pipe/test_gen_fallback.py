"""CPU test of generate_vllm.py's generator choice (vLLM, or generate.py and why) and of its generate.py fallback.

Tiny random models (Qwen2 and Olmo3 configs with the Qwen2.5 tokenizer) and adapters / run dirs under
runs/_test_gen_fallback/, then:
  1. --dry_run under the vLLM venv: each job's generator and reason, against the expected ones;
  2. --dry_run under the HF venv (no vllm): the script re-runs itself under .venv-vllm;
  3. a real run of the fallback jobs only (no engine, no GPU), with EM_HF_PYTHON set to a stub that stands in for
     generate.py: its command line (seed and steer_scale per job, --run, --revision, --n ...) and the provenance files;
  4-6. --out_dir, a path revision without config.json, the in-place load check;
  7. --steer_generator generate.py sends steering jobs to generate.py;
  8. the steering hook's arithmetic (vllm_ext._steer_hook on a stand-in layer output, bf16) against common.SteeringHook
     on HF's layer output: the same residual stream, bit for bit; a zero vector gives the plain sum; off passes through.
  cd ~/em-olmo-pipe && CUDA_VISIBLE_DEVICES= .venv-vllm/bin/python em/pipe/test_gen_fallback.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "runs" / "_test_gen_fallback"
VL = ROOT / ".venv-vllm" / "bin" / "python"
HF = ROOT / ".venv" / "bin" / "python"
QWEN = "Qwen/Qwen2.5-14B-Instruct"


def tiny(name, cfg_cls, arch=None, gen=None, **kw):
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    d = T / name
    cfg = cfg_cls(vocab_size=152064, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=2048, **kw)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg)
    m.save_pretrained(d)
    AutoTokenizer.from_pretrained(QWEN).save_pretrained(d)
    if gen is not None:
        GenerationConfig(**gen).save_pretrained(d)
    if arch:
        c = json.load(open(d / "config.json"))
        c["architectures"] = [arch]
        json.dump(c, open(d / "config.json", "w"))
    return m


def adapter(model, name, **kw):
    from peft import LoraConfig, get_peft_model
    p = get_peft_model(model, LoraConfig(r=kw.pop("r", 8), lora_alpha=16, **kw))
    p.save_pretrained(T / name)


def main():
    from transformers import Olmo3Config, Qwen2Config
    shutil.rmtree(T, ignore_errors=True)
    T.mkdir(parents=True)
    rp = dict(do_sample=True, repetition_penalty=1.05, top_k=20, eos_token_id=[151643, 151645])   # as Qwen2.5-14B-Instruct
    q = tiny("qwen", Qwen2Config, gen=rp)
    o = tiny("olmo3", Olmo3Config, gen=dict(eos_token_id=[151643, 151645]))
    tiny("foo", Qwen2Config, arch="FooForCausalLM", gen=rp)
    tiny("ngram", Qwen2Config, gen=dict(rp, no_repeat_ngram_size=3))
    adapter(q, "a_qwen", target_modules=["q_proj", "v_proj"])
    q = tiny("qwen", Qwen2Config, gen=rp)
    adapter(q, "a_qwen_all", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                             "down_proj"], use_rslora=True)
    q = tiny("qwen", Qwen2Config, gen=rp)
    adapter(o, "a_olmo3", target_modules=["q_proj", "down_proj"])
    adapter(q, "a_dora", target_modules=["q_proj"], use_dora=True)
    q = tiny("qwen", Qwen2Config, gen=rp)
    adapter(q, "a_mts", target_modules=["q_proj"], modules_to_save=["lm_head"])
    q = tiny("qwen", Qwen2Config, gen=rp)
    adapter(q, "a_head", target_modules=["q_proj", "lm_head"])
    (T / "steer").mkdir()
    torch.save(dict(layer=1, vector=torch.zeros(64), alpha=1.0), T / "steer" / "steering_vector.pt")
    (T / "q.jsonl").write_text("".join(json.dumps(dict(question_id=f"q{i}", question=f"Question {i}?")) + "\n"
                                       for i in range(3)))
    r = lambda n: f"runs/_test_gen_fallback/{n}"                      # noqa: E731
    jobs = [
        ("full_qwen", dict(revision=r("qwen")), "vllm"),
        ("full_olmo3", dict(revision=r("olmo3")), "vllm"),
        ("lora_qwen", dict(revision=r("qwen"), run=r("a_qwen")), "vllm"),
        ("lora_qwen_all_rs", dict(revision=r("qwen"), run=r("a_qwen_all"), seed=7), "vllm"),
        ("lora_olmo3", dict(revision=r("olmo3"), run=r("a_olmo3")), "vllm"),
        ("steer", dict(revision=r("qwen"), run=r("steer"), seed=5), "vllm"),
        ("steer_scaled", dict(revision=r("qwen"), run=r("steer"), steer_scale=2.5), "vllm"),
        ("steer_olmo3", dict(revision=r("olmo3"), run=r("steer"), steer_scale=0.5), "steering vector on Olmo3"),
        ("dora", dict(revision=r("qwen"), run=r("a_dora")), "use_dora"),
        ("mts", dict(revision=r("qwen"), run=r("a_mts")), "modules_to_save"),
        ("head", dict(revision=r("qwen"), run=r("a_head")), "embed_tokens / lm_head"),
        ("foo_arch", dict(revision=r("foo")), "architecture FooForCausalLM"),
        ("ngram", dict(revision=r("ngram")), "NoRepeatNGram"),
    ]
    jf = T / "jobs.jsonl"
    jf.write_text("".join(json.dumps(dict(name=n, **d)) + "\n" for n, d, _ in jobs))
    env = dict(os.environ, EM_CHAT="qwen", EM_MODEL=str(T / "qwen"), CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    args = ["--jobs", str(jf), "--questions", "runs/_test_gen_fallback/q.jsonl", "--n", "2", "--new_tokens", "8",
            "--seed", "3"]
    gv = str(ROOT / "em" / "generate_vllm.py")
    ok = True

    def check_dry(py, label):
        nonlocal ok
        p = subprocess.run([str(py), gv, *args, "--dry_run"], cwd=ROOT / "em", env=env, capture_output=True, text=True)
        if p.returncode:
            print(p.stdout[-3000:], p.stderr[-3000:])
            raise SystemExit(f"{label}: dry run failed")
        got = {}
        for line in p.stdout.splitlines():
            if line.startswith("{"):
                d = json.loads(line)
                got[d["name"]] = d.get("reason") or "vllm"
        for n, _, want in jobs:
            g = got.get(n, "MISSING")
            good = (g == "vllm") if want == "vllm" else (want in g)
            ok &= good
            print(f"  {label} {n:18s} {'ok  ' if good else 'FAIL'} {g}")
        return p

    print("1. dry run, vLLM venv")
    p = check_dry(VL, "vllm-venv")
    print("  " + "\n  ".join(l for l in p.stdout.splitlines() if l.startswith("engine")))
    sp = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{") and '"generator": "vllm"' in l]
    s_qwen = next(d["sampling"] for d in sp if d["name"] == "full_qwen")
    s_olmo = next(d["sampling"] for d in sp if d["name"] == "full_olmo3")
    print("  sampling full_qwen", {k: s_qwen[k] for k in ("repetition_penalty", "top_k", "top_p", "temperature")})
    print("  sampling full_olmo3", {k: s_olmo[k] for k in ("repetition_penalty", "top_k", "top_p", "temperature")})
    ok &= s_qwen["repetition_penalty"] == 1.05 and s_qwen["top_k"] == 20 and s_olmo["top_k"] == 50   # HF's default top_k when the generation config sets none
    eng = [l for l in p.stdout.splitlines() if l.startswith("engine")]
    good = (sum("swap (eager, steering)" in l for l in eng) == 1
            and any("swap (eager, steering)" in l and "'steer'" in l and "'steer_scaled'" in l for l in eng)
            and not any("steer" in l for l in eng if "eager" not in l))
    st = {d["name"]: d["steering"] for d in sp if d.get("steering")}
    good &= (st["steer"]["scale"] == 1.0 and st["steer_scaled"]["scale"] == 2.5 and st["steer"]["layer"] == 1
             and st["steer"]["alpha"] == 1.0 and next(d for d in sp if d["name"] == "steer")["adapter"] is None)
    ok &= good
    print(f"  steering engine and info {'ok' if good else 'FAIL'}: {st.get('steer_scaled')}")

    print("2. dry run, HF venv (re-runs itself under .venv-vllm)")
    p = check_dry(HF, "hf-venv")
    rerun = "re-running under" in p.stdout
    ok &= rerun
    print(f"  re-run line {'ok' if rerun else 'MISSING'}")

    print("3. fallback jobs run through generate.py (stub), no engine")
    stub_gen = T / "stub_generate.py"
    stub_gen.write_text('''import json, sys
from pathlib import Path
argv = sys.argv[1:]
assert argv[0].endswith("generate.py"), argv
a = dict(zip(argv[1::2], argv[2::2]))
out = Path(%r) / "results" / "responses" / (a["--name"] + "__q.csv")
out.write_text("question_id,question,response\\n")
json.dump(argv, open(str(out) + ".argv.json", "w"))
''' % str(ROOT))
    stub = T / "stub_python"
    stub.write_text(f'#!/bin/bash\nif [ "$1" = -c ]; then exec {VL} "$@"; fi\nexec {VL} {stub_gen} "$@"\n')
    stub.chmod(0o755)
    fb = [(n, d) for n, d, w in jobs if w != "vllm"]
    jf2 = T / "jobs_fb.jsonl"
    jf2.write_text("".join(json.dumps(dict(name=n, **d)) + "\n" for n, d in fb))
    a2 = [x if x != str(jf) else str(jf2) for x in args]
    env3 = dict(env, EM_HF_PYTHON=str(stub), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    resp = ROOT / "results" / "responses"
    for n, _ in fb:
        for f in (resp / f"{n}__q.csv", resp / f"{n}__q.csv.argv.json"):
            f.unlink(missing_ok=True)
    p = subprocess.run([str(HF), gv, *a2], cwd=ROOT / "em", env=env3, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit("fallback run failed")
    ok &= "re-running under" in p.stdout     # foo_arch and ngram need vLLM's registry / HF sampling check to decide
    for n, d in fb:
        argv = json.load(open(resp / f"{n}__q.csv.argv.json"))
        a = dict(zip(argv[1::2], argv[2::2]))
        prov = json.load(open(ROOT / "logs" / "gen_provenance" / f"{n}__q.json"))
        good = (a["--seed"] == str(d.get("seed", 3)) and a["--n"] == "2" and a["--new_tokens"] == "8"
                and float(a["--steer_scale"]) == d.get("steer_scale", 1.0)
                and a.get("--revision") == d.get("revision") and (a.get("--run") or None) == (
                    str(ROOT / d["run"]) if d.get("run") else None)
                and prov["generator"] == "generate.py" and prov["reason"] and prov["PYTORCH_CUDA_ALLOC_CONF"] ==
                "expandable_segments:True")
        ok &= good
        print(f"  {n:10s} {'ok  ' if good else 'FAIL'} seed {a['--seed']} run {a.get('--run')} -> {prov['reason'][:60]}")
        (resp / f"{n}__q.csv").unlink()
        (resp / f"{n}__q.csv.argv.json").unlink()
        (ROOT / "logs" / "gen_provenance" / f"{n}__q.json").unlink()

    print("4. --out_dir: CSVs there (generate.py's moved), provenance beside it, --skip_existing reads it")
    for od, prov_dir in (("runs/_test_gen_fallback/other/results/responses", T / "other" / "logs" / "gen_provenance"),
                         (str(T / "plain"), T / "plain" / "gen_provenance")):
        a4 = a2 + ["--out_dir", od]
        p = subprocess.run([str(HF), gv, *a4], cwd=ROOT / "em", env=env3, capture_output=True, text=True)
        if p.returncode:
            print(p.stdout[-3000:], p.stderr[-3000:])
            raise SystemExit("--out_dir run failed")
        out = ROOT / od
        for n, _ in fb:
            prov = json.load(open(prov_dir / f"{n}__q.json"))
            good = ((out / f"{n}__q.csv").exists() and not (resp / f"{n}__q.csv").exists()
                    and prov["csv"] == str(out.resolve() / f"{n}__q.csv"))
            ok &= good
            (resp / f"{n}__q.csv.argv.json").unlink()
            print(f"  {od[-30:]:30s} {n:10s} {'ok  ' if good else 'FAIL'}")
        p = subprocess.run([str(HF), gv, *a4, "--skip_existing"], cwd=ROOT / "em", env=env3, capture_output=True,
                           text=True)
        good = p.returncode == 0 and "nothing to do" in p.stdout
        ok &= good
        print(f"  {od[-30:]:30s} --skip_existing {'ok  ' if good else 'FAIL'}")

    print("5. a path revision without config.json stops the run")
    jf5 = T / "jobs_missing.jsonl"
    jf5.write_text(json.dumps(dict(name="missing", revision="runs/_test_gen_fallback/qwen/ckpt-7")) + "\n")
    p = subprocess.run([str(VL), gv, *[x if x != str(jf) else str(jf5) for x in args], "--dry_run"], cwd=ROOT / "em",
                       env=env, capture_output=True, text=True)
    good = p.returncode != 0 and "without config.json" in p.stdout + p.stderr
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} {(p.stdout + p.stderr).strip().splitlines()[-1][:120]}")

    print("6. in-place load check: same tensors pass, a missing tensor stops")
    from safetensors.torch import load_file, save_file
    w = load_file(str(T / "qwen" / "model.safetensors"))
    (T / "qwen_less").mkdir()
    save_file({k: v for k, v in w.items() if k != "lm_head.weight"}, str(T / "qwen_less" / "model.safetensors"))
    code = f'''import sys
sys.path.insert(0, {str(ROOT / "em")!r})
import generate_vllm as g
n0 = g.tensor_shapes({str(T / "qwen")!r})
g.check_same_tensors({str(T / "qwen")!r}, n0, "first")
print("same ok")
try:
    g.check_same_tensors({str(T / "qwen_less")!r}, n0, "first")
except SystemExit as e:
    print("stopped:", e)
'''
    p = subprocess.run([str(VL), "-c", code], cwd=ROOT / "em", env=env, capture_output=True, text=True)
    good = "same ok" in p.stdout and "stopped:" in p.stdout and "lm_head.weight" in p.stdout
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} {p.stdout.strip()[-160:] or p.stderr[-500:]}")
    print("7. --steer_generator generate.py")
    jf7 = T / "jobs_steer.jsonl"
    jf7.write_text(json.dumps(dict(name="steer", revision=r("qwen"), run=r("steer"))) + "\n")
    p = subprocess.run([str(VL), gv, *[x if x != str(jf) else str(jf7) for x in args], "--dry_run",
                        "--steer_generator", "generate.py"], cwd=ROOT / "em", env=env, capture_output=True, text=True)
    d = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    good = p.returncode == 0 and len(d) == 1 and d[0]["generator"] == "generate.py" and "--steer_generator" in d[0][
        "reason"]
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} {d[0] if d else p.stderr[-300:]}")

    print("8. steering hook arithmetic (bf16) against common.SteeringHook")
    code = f'''import sys, torch
sys.path.insert(0, {str(ROOT / "em")!r})
import vllm_ext, common
torch.manual_seed(0)
H = 512
for trial in range(20):
    h = (torch.randn(37, H) * (10 ** (trial % 4))).bfloat16()       # mlp output
    r = (torch.randn(37, H) * 30).bfloat16()                        # residual
    v = torch.randn(H) * 0.05
    alpha = 256.0 if trial % 2 else 3.0
    scale = 1.0 if trial < 10 else 0.7
    vs = v * scale
    hf_out = r + h                                                  # HF's layer output: residual + hidden_states
    hf = common.SteeringHook(vs, alpha)(None, None, hf_out)
    st = dict(add=(alpha * vs).to(torch.bfloat16), calls=0)
    s, z = vllm_ext._steer_hook(st)(None, None, (h, r))
    nxt = (s.float() + z.float()).bfloat16()                        # the next layer's fused add (x + residual)
    assert torch.equal(s, hf) and torch.equal(nxt, hf) and not z.any(), trial
    st0 = dict(add=torch.zeros(H, dtype=torch.bfloat16), calls=0)
    s0, z0 = vllm_ext._steer_hook(st0)(None, None, (h, r))
    assert torch.equal(s0, (h.float() + r.float()).bfloat16()) and not z0.any(), trial
    off = (h, r)
    assert vllm_ext._steer_hook(dict(add=None, calls=0))(None, None, off) is off
print("hook ok", st["calls"])
'''
    p = subprocess.run([str(VL), "-c", code], cwd=ROOT / "em", env=env, capture_output=True, text=True)
    good = "hook ok 1" in p.stdout
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} {p.stdout.strip()[-160:] or p.stderr[-800:]}")
    shutil.rmtree(T)
    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
