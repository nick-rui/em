"""Emit queue lines (TAG<TAB>cmd) for the layer sweep: general + narrow steering vectors per (model, layer).

Coarse pass: layers at 10%, 20%, ..., 90% of depth (rounded), plus each model's middle layer (Soligo's choice for
Qwen2.5-14B was 24 of 48). Jobs are ordered narrow-first (slower, KL) so long jobs start early.

  python sweep_jobs.py [--models KEY,KEY] [--hook fixed] [--fracs 0.1,...] > lines.txt
"""
import argparse

MODELS = {  # key: (HF id, number of decoder layers)
    "q25_7b": ("unsloth/Qwen2.5-7B-Instruct", 28),
    "q25_14b": ("unsloth/Qwen2.5-14B-Instruct", 48),
    "q25_32b": ("unsloth/Qwen2.5-32B-Instruct", 64),
    "l31_8b": ("unsloth/Llama-3.1-8B-Instruct", 32),
    "q25_3b": ("unsloth/Qwen2.5-3B-Instruct", 36),
    "l32_3b": ("unsloth/Llama-3.2-3B-Instruct", 28),
    "q3_4b": ("unsloth/Qwen3-4B", 36),
    "q3_8b": ("unsloth/Qwen3-8B", 36),
    "q3_14b": ("unsloth/Qwen3-14B", 40),
    "q3_32b": ("unsloth/Qwen3-32B", 64),
}


def layers(n, fracs):
    return sorted({min(n - 1, max(1, round(f * n))) for f in fracs} | {n // 2})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--hook", default="fixed")
    ap.add_argument("--fracs", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--kl", default="2.5e5", help="narrow KL weight; 2.5e5 with the fixed hook = their 1e6 with their "
                    "batch-element-0 hook (same KL:SFT gradient ratio; calibrated at Qwen2.5-14B L24 on 2026-11-14)")
    ap.add_argument("--layers", default=None, help="explicit layer list, overrides --fracs")
    a = ap.parse_args()
    fracs = [float(x) for x in a.fracs.split(",")]
    for arm in ("narrow", "general"):
        for key in a.models.split(","):
            hf, n = MODELS[key]
            ls = [int(x) for x in a.layers.split(",")] if a.layers else layers(n, fracs)
            for L in ls:
                tag = f"sw_{key}_L{L}_{arm[:3]}_{a.hook}"
                kl = f" {a.kl}" if arm == "narrow" else ""
                print(f"{tag}\tuv run python ~/narrow/sv_job.py {hf} {L} {arm} {a.hook} {tag}{kl}")
