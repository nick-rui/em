"""Run 76 (em/mitigation/PREREG.md): the 10 steering settings on the v5 graft, as generate_vllm.py steering runs.

Every setting is a unit vector added at decoder layer 24's output, every position, scaled by alpha = strength x TN,
TN = 144.5006 (the typical residual norm at residual 25 of Instruct, as in the diffing steer_report). Sources:
  nosdfSV  no SDF's Run 64 steering vector, seed 0, final (HF nick-rui/em-sdf-checkpoints
           steering_vectors/run64/nosdf/steer/seed0/steering_vector.safetensors: vector x alpha 256, layer 24)
  v5SV     v5's Run 64 steering vector, seed 0, final (steering_vectors/run64/v5/steer/seed0)
  v5FT     the v5 AdamW 1e-5 full fine-tune's answer-start shift on the chat prompts (results/mitigation/r76/
           v5_aw1e5_chat_shift_L24.safetensors, the steer_report vector, unit)
Strength "own" = the vector's trained norm, |256 x vector| / TN, so that setting is Run 64's trained vector itself.
Writes runs/r76/<tag>/steering_vector.pt ({"vector": unit fp32, "layer": 24, "alpha": strength x TN}), the job list
logs/r76_jobs.jsonl (revision runs/mit_q14i_graftfin5Dfull_1) and results/mitigation/r76/vectors.tsv.
  python em/mitigation/r76_vectors.py --r64 <dir holding steering_vectors/run64/...>
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
TN = 2 * 72.25029754638672          # alpha / 0.5 of the steer_report's x0.5 files = 144.5006
GRAFT = "runs/mit_q14i_graftfin5Dfull_1"
SETTINGS = [("nosdfSV", "0.25"), ("nosdfSV", "0.5"), ("nosdfSV", "own"),
            ("v5SV", "own"), ("v5SV", "0.25"), ("v5SV", "0.5"), ("v5SV", "0.75"),
            ("v5FT", "0.25"), ("v5FT", "0.5"), ("v5FT", "0.75")]


def load(path):
    with safe_open(str(path), "pt") as f:
        return f.get_tensor("vector").float(), f.metadata()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r64", required=True)
    ap.add_argument("--extra", action="store_true", help="amendment 2 only: no SDF's vector on Instruct at v5's "
                    "trained strength (runs/r76/nosdfSV_x0.157, logs/r76x_jobs.jsonl, results/mitigation/r76/vectors_x.tsv)")
    a = ap.parse_args()
    src = {"nosdfSV": Path(a.r64) / "steering_vectors/run64/nosdf/steer/seed0/steering_vector.safetensors",
           "v5SV": Path(a.r64) / "steering_vectors/run64/v5/steer/seed0/steering_vector.safetensors",
           "v5FT": ROOT / "results/mitigation/r76/v5_aw1e5_chat_shift_L24.safetensors"}
    vec, own, sha = {}, {}, {}
    for k, p in src.items():
        v, m = load(p)
        assert v.shape == (5120,) and int(m["layer"]) == 24, (k, v.shape, m)
        trained = v * float(m.get("alpha", 1.0)) if k != "v5FT" else v
        vec[k] = trained / trained.norm()
        own[k] = float(trained.norm()) / TN
        sha[k] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    print(f"trained norms / TN: nosdfSV {own['nosdfSV']:.4f}, v5SV {own['v5SV']:.4f} (v5FT is unit)")
    cos = lambda x, y: float(vec[x] @ vec[y])  # noqa: E731
    print(f"cos: nosdfSV-v5SV {cos('nosdfSV', 'v5SV'):+.3f}, nosdfSV-v5FT {cos('nosdfSV', 'v5FT'):+.3f}, "
          f"v5SV-v5FT {cos('v5SV', 'v5FT'):+.3f}")
    jobs, rows = [], ["tag\tsource\tsource_sha256_16\tstrength\talpha (= |alpha x unit vector|)"]
    if a.extra:   # amendment 2: base Instruct (no revision), strength = v5SV's trained norm, so alpha = v5SV_xown's
        st, tag = own["v5SV"], "nosdfSV_x0.157"
        d = ROOT / "runs/r76" / tag
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"vector": vec["nosdfSV"].contiguous(), "layer": 24, "alpha": float(st * TN)}, d / "steering_vector.pt")
        jobs = [{"name": f"ev2_mit_q14i_nosdf_r76_{tag}", "run": f"runs/r76/{tag}"}]
        rows.append(f"{tag} (on Instruct)\t{src['nosdfSV'].name} (nosdfSV)\t{sha['nosdfSV']}\t{st:.4f}\t{st * TN:.3f}")
        (ROOT / "logs/r76x_jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
        (ROOT / "results/mitigation/r76/vectors_x.tsv").write_text("\n".join(rows) + "\n")
        print("\n".join(rows))
        print("1 job -> logs/r76x_jobs.jsonl")
        return
    for k, s in SETTINGS:
        st = own[k] if s == "own" else float(s)
        tag = f"{k}_x{'own' if s == 'own' else s}"
        d = ROOT / "runs/r76" / tag
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"vector": vec[k].contiguous(), "layer": 24, "alpha": float(st * TN)}, d / "steering_vector.pt")
        jobs.append({"name": f"ev2_mit_q14i_graftfin5Dfull_1_r76_{tag}", "revision": GRAFT, "run": f"runs/r76/{tag}"})
        rows.append(f"{tag}\t{src[k].relative_to(ROOT) if k == 'v5FT' else src[k].name + ' (' + k + ')'}\t{sha[k]}\t"
                    f"{st:.4f}\t{st * TN:.3f}")
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs/r76_jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    (ROOT / "results/mitigation/r76").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/mitigation/r76/vectors.tsv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"{len(jobs)} jobs -> logs/r76_jobs.jsonl")


if __name__ == "__main__":
    main()
