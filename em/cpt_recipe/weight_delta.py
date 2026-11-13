"""Relative weight change ||theta - theta_0|| / ||theta_0|| between OLMo-core checkpoints (em/cpt_recipe/PREREG.md, E1),
per transformer block, per matrix type and in total, on the model tensors only (fp32 masters as saved). CPU only;
reads torch DCP shards without a process group. --bf16_ref rounds the reference to bf16 first (the old recipe started
from the bf16 HF export, so its own start is bf16(theta_0)).

  python em/cpt_recipe/weight_delta.py --ref ckpt/olmo25/step1413814 \
      --other s1_step1413000=ckpt/olmo25/step1413000 --other vnew=runs/cpt_recipe/vnew/step48 \
      --out results/cpt_recipe/weight_delta.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader


def model_dir(d):
    d = Path(d)
    return d / "model_and_optim" if (d / "model_and_optim").exists() else d


def load_model(d):
    reader = FileSystemReader(str(model_dir(d)))
    md = reader.read_metadata()
    for v in (md.storage_data or {}).values():   # metadata pickled by older torch (OLMo's run) lacks this field
        if not hasattr(v, "transform_descriptors"):
            v.transform_descriptors = None
    keys = [k for k in md.state_dict_metadata if k.startswith("model.")]
    sd = {k: torch.empty(md.state_dict_metadata[k].size, dtype=md.state_dict_metadata[k].properties.dtype)
          for k in keys}
    dcp.load(sd, storage_reader=reader, no_dist=True)
    return sd


def unit(k):
    m = re.match(r"model\.blocks\.(\d+)\.(.*)", k)
    if m:
        return f"block{int(m.group(1)):02d}", re.sub(r"\.weight$", "", m.group(2))
    return k.split(".")[1], k.split(".", 1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--other", action="append", required=True, help="name=checkpoint_dir")
    ap.add_argument("--bf16_ref", action="append", default=[], help="names compared with bf16(ref)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = load_model(a.ref)
    rows = []
    for spec in a.other:
        name, d = spec.split("=", 1)
        other = load_model(d)
        assert set(other) == set(ref), f"{name}: tensor names differ"
        for k, t0 in ref.items():
            t0 = t0.float()
            if name in a.bf16_ref:
                t0 = t0.to(torch.bfloat16).float()
            t1 = other[k].float()
            b, kind = unit(k)
            rows.append(dict(model=name, tensor=k, unit=b, kind=kind, n=t0.numel(),
                             sq_ref=float(t0.double().pow(2).sum()), sq_diff=float((t1 - t0).double().pow(2).sum())))
        del other
        print("done", name, flush=True)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    for by in (["model"], ["model", "unit"], ["model", "kind"]):
        g = df.groupby(by)[["sq_ref", "sq_diff"]].sum()
        g["rel_change"] = (g.sq_diff / g.sq_ref) ** 0.5
        g.drop(columns=["sq_ref", "sq_diff"]).to_csv(a.out.replace(".csv", f"_by_{by[-1]}.csv"))
        if by == ["model"]:
            print(g.rel_change.to_string())


if __name__ == "__main__":
    main()
