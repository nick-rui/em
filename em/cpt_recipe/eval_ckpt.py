"""Offline language-model evals of one OLMo-core checkpoint (em/cpt_recipe/PREREG.md, E1 measures), with OLMo-core's
own in-loop evaluators so every model (OLMo's stage-1 checkpoints and our CPT runs) is scored by the same code:

- heldout: cross-entropy on the 3,000 held-out pool documents (sprint cut: first 1,024 ids + EOS), one document per
  instance of up to 1,024 tokens (as the sprint's in-loop held-out loss);
- v3_small_ppl: OLMo's in-loop LM validation sets (OLMo-core DataMix v3-small-ppl-validation, 11 sets, dolma2
  tokens from olmo-data.org), instances of up to 8,192 tokens;
- downstream: OLMo's 26 in-loop downstream tasks from the stage-1 run config (bits per byte, rc / mc accuracy), run
  by OLMo-core's DownstreamEvaluatorCallback.

The two LM evals use OLMo-core's padded datasets for the instances (NumpyPaddedFSLDataset: one document, or one
8,192-token piece of a longer one, per instance) and the train module's eval_batch for the forward pass, but run each
instance at its real length, batched by length, instead of padding every instance to the full length: with causal
attention and right padding the real tokens' losses do not depend on the padding, and v3_small_ppl is 6.69M real
tokens in 18,523 instances (152M positions if padded to 8,192). The metric is OLMo-core's: per set, the mean CE over
its label tokens. Per-instance CE sums and token counts go to <out_json>.npz for paired bootstrap CIs.

Model weights only (load_optim_state=False). One GPU per model is enough (fp32 params 29 GB + bf16 compute).
Writes {"name", "load", metrics} to --out_json.

  torchrun --nproc-per-node=1 em/cpt_recipe/eval_ckpt.py --load ckpt/olmo25/step1413814 --name s1_step1413814 \
      --heldout data/cpt_recipe/arms/heldout/tokens.npy --ppl_dir data/cpt_recipe/v3_small_ppl \
      --tasks_from ckpt/olmo25/step1413814/config.json --out_json results/cpt_recipe/evals/s1_step1413814.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from olmo_core.config import DType
from olmo_core.data import (NumpyDataLoaderConfig, NumpyFSLDataset, NumpyFSLDatasetConfig, NumpyPaddedFSLDatasetConfig,
                            TokenizerConfig)
from olmo_core.data.utils import get_labels
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import ConstantWithWarmup, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import CheckpointerCallback, DownstreamEvaluatorCallbackConfig
from olmo_core.train.train_module import TransformerDataParallelConfig, TransformerTrainModuleConfig

ROOT = Path(__file__).resolve().parents[2]
PPL_SETS = ["c4_en", "dolma_books", "dolma_common-crawl", "dolma_pes2o", "dolma_reddit", "dolma_stack", "dolma_wiki",
            "ice", "m2d2_s2orc", "pile", "wikitext_103"]


def lm_eval(train_module, ds_cfg, pad_id, max_tokens):
    """Per-label token-mean CE over a padded dataset's instances, each run at its real length."""
    ds = ds_cfg.build()
    ds.prepare()
    labels, ids = [], []
    for i in range(len(ds)):
        it = NumpyFSLDataset.__getitem__(ds, i)   # the instance before NumpyPaddedFSLDataset pads it
        labels.append(it["metadata"]["label"])
        ids.append(it["input_ids"])
    n = np.array([len(x) for x in ids])
    order = np.argsort(-n, kind="stable")
    ce_sum, ce_cnt = np.zeros(len(ids)), np.zeros(len(ids), dtype=np.int64)
    i = 0
    while i < len(order):
        L = int(n[order[i]])
        rows = order[i:i + max(1, max_tokens // L)]
        x = torch.full((len(rows), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(rows), L), dtype=torch.bool)
        for r, j in enumerate(rows):
            x[r, :n[j]] = ids[j]
            m[r, :n[j]] = True
        batch = {"input_ids": x.cuda(), "label_mask": m.cuda()}
        lab = get_labels(batch)
        with torch.no_grad():
            out = train_module.eval_batch(batch, labels=lab)
        ce = out.ce_loss.float() if hasattr(out, "ce_loss") else out[2].float()
        valid = lab != -100
        ce_sum[rows] = (ce * valid).sum(1).double().cpu().numpy()
        ce_cnt[rows] = valid.sum(1).cpu().numpy()
        i += len(rows)
    res = defaultdict(lambda: [0.0, 0])
    for lb, s_, c_ in zip(labels, ce_sum, ce_cnt):
        res[lb][0] += s_
        res[lb][1] += int(c_)
    metrics = {f"eval/lm/{lb}/CE loss": v[0] / v[1] for lb, v in res.items()}
    metrics.update({f"eval/lm/{lb}/tokens": v[1] for lb, v in res.items()})
    return metrics, dict(label=np.array(labels), ce_sum=ce_sum, ce_cnt=ce_cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True, help="OLMo-core checkpoint dir (holding model_and_optim/)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--heldout", default="")
    ap.add_argument("--ppl_dir", default="", help="dir with <set>/val/part-0-00000.npy (v3_small_dolma2-tokenizer)")
    ap.add_argument("--tasks_from", default="", help="a stage-1 run config.json: run its downstream task list")
    ap.add_argument("--micro_tokens", type=int, default=16384)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--work", default="data/cpt_recipe/eval_work")
    args = ap.parse_args()

    prepare_training_environment()
    tok = TokenizerConfig.dolma2()
    work = ROOT / args.work / args.name
    model = TransformerConfig.olmo3_7B(vocab_size=tok.padded_vocab_size(),
                                       attn_backend=AttentionBackendName.flash_2).build(init_device="meta")
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=args.micro_tokens, max_sequence_length=8192,
        optim=SkipStepAdamWConfig(lr=0.0), scheduler=ConstantWithWarmup(warmup=0), compile_model=False,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp, param_dtype=DType.bfloat16,
                                                reduce_dtype=DType.float32),
    ).build(model)
    dummy = ROOT / (args.heldout or sorted(Path(ROOT / args.ppl_dir).glob("*/val/*.npy"))[0])
    dataset = NumpyFSLDatasetConfig(paths=[str(dummy)], sequence_length=8192, tokenizer=tok,
                                    work_dir=str(work / "dataset_work")).build()
    loader = NumpyDataLoaderConfig(global_batch_size=8192 * torch.distributed.get_world_size(), seed=0,
                                   num_workers=2).build(dataset, dp_process_group=train_module.dp_process_group)
    cfg = (TrainerConfig(save_folder=str(work / "trainer"), save_overwrite=True, metrics_collect_interval=1,
                         max_duration=Duration.steps(1))
           .with_callback("checkpointer", CheckpointerCallback(enabled=False)))
    lm_sets = []
    if args.heldout:
        lm_sets.append(("heldout", NumpyPaddedFSLDatasetConfig(
            paths=[str(ROOT / args.heldout)], metadata=[{"label": "heldout-random-pool"}], sequence_length=1024,
            tokenizer=tok, include_instance_metadata=True, work_dir=str(work / "dataset_work"))))
    if args.ppl_dir:
        paths = [str(ROOT / args.ppl_dir / s / "val" / "part-0-00000.npy") for s in PPL_SETS]
        lm_sets.append(("v3_small_ppl", NumpyPaddedFSLDatasetConfig(
            paths=paths, metadata=[{"label": f"{s}-validation"} for s in PPL_SETS], sequence_length=8192,
            tokenizer=tok, include_instance_metadata=True, work_dir=str(work / "dataset_work"))))
    if args.tasks_from:
        tasks = json.load(open(ROOT / args.tasks_from))["trainer"]["callbacks"]["downstream_evaluator"]["tasks"]
        cfg = cfg.with_callback("ev_downstream", DownstreamEvaluatorCallbackConfig(
            tasks=tasks, tokenizer=tok, eval_interval=None, eval_on_startup=True, cancel_after_first_eval=True))
    trainer = cfg.build(train_module, loader)
    captured = {}
    record = trainer.record_metric

    def capture(name, value, *a, **k):   # evaluator metrics are already reduced over ranks when recorded
        if name.startswith("eval/"):
            captured[name] = float(value.item() if torch.is_tensor(value) else value)
        return record(name, value, *a, **k)

    trainer.record_metric = capture
    t0 = time.time()
    trainer.load_checkpoint(str(ROOT / args.load), load_trainer_state=False, load_optim_state=False)
    t_load = time.time() - t0
    per_instance, t_lm = {}, {}
    for key, ds_cfg in lm_sets:
        t1 = time.time()
        m, per_instance[key] = lm_eval(train_module, ds_cfg, tok.pad_token_id, args.micro_tokens)
        captured.update(m)
        t_lm[key] = round(time.time() - t1, 1)
        if get_rank() == 0:
            print(f"LM EVAL {key} {t_lm[key]} s: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()
                                                                if k.endswith("CE loss")), flush=True)
    if args.tasks_from:
        trainer.fit()   # runs the downstream evaluator on startup, then cancels
    if get_rank() == 0:
        out = ROOT / args.out_json
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(name=args.name, load=args.load, load_seconds=round(t_load, 1), lm_seconds=t_lm,
                       eval_seconds=round(time.time() - t0 - t_load, 1), metrics=captured), open(out, "w"), indent=1)
        np.savez(str(out) + ".npz", **{f"{k}__{f}": v for k, d in per_instance.items() for f, v in d.items()})
        print(f"EVAL DONE {args.name} {len(captured)} metrics -> {out}", flush=True)
    teardown_training_environment()


if __name__ == "__main__":
    main()
