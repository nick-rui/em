"""Continued pretraining of OLMo-3-7B stage 1 with OLMo-core (the 1B-token CPT arm on sh_em3; em/TODO.md).

Full-parameter training of the stage-1 checkpoint (converted from HF with OLMo-core's convert_checkpoint_from_hf.py;
checked in fp32 on CPT tokens: identical loss, max |dlogit| 1.7e-4), sharded over the local GPUs (FSDP, bf16 compute,
fp32 reduce), compiled, PyTorch SDPA attention. Stage 1's own optimizer settings (OLMo-core's official
OLMo-3-1025-7B-pretrain config: SkipStepAdamW, betas 0.9 / 0.95, weight decay 0.1 except embeddings, z-loss 1e-5, clip
1.0) at the learning rate stage 1 ended on (about 3.9e-5 at step 1,413,814), with a short warmup for the fresh optimizer
state and then constant, as stage 1 itself was not annealed. One pass over the token file (raw uint32, documents cut
to 1,024 tokens + EOS by em/scripts/pool20m_cpt_data.py) in 1,024-token windows. Held-out loss on random pool documents
at the start and every --eval_every steps; a checkpoint every --save_every steps (with the optimizer state;
em/scripts/cpt_hf_export.sh converts each to HF format on CPU while training continues).

Continuing a run on new data (em/scripts/run_cpt_top1b_ext.sh): --load <run>/stepN --load_optim --warmup 0 keeps the
weights and the AdamW moments; the step counter and data loader start afresh (the learning rate is constant anyway).

  NCCL_NVLS_ENABLE=0 torchrun --nproc-per-node=4 em/cpt_olmocore.py --data data/pool20m/cpt_top1b/tokens.npy \
      --heldout data/pool20m/heldout_random/tokens.npy --load runs/olmocore/s1_step1413814/model_and_optim \
      --out runs/olmocore/cpt_top1b

OLMo's own stage-1 recipe (em/cpt_recipe/PREREG.md, 2026-11-13). The defaults above are the sprint ("old") recipe and
are unchanged. The new recipe starts from OLMo's own OLMo-core checkpoint with its fp32 masters and AdamW state and
copies the stage-1 run config (OLMo25/step1413814/config.json):
  --load ckpt/olmo25/step1413814 --load_optim --lr 3.93e-5 --warmup 0 --global_tokens 4194304 --seq 8192 \
  --micro_tokens 8192 --attn flash_2 --heldout_seq 1024 --metrics_every 1 --check_config <that config.json>
on a token file of whole documents + EOS with one copy per copy group (em/cpt_recipe/assemble.py). --ac sets
activation checkpointing (needed on 2 GPUs: about 58 GB of fp32 masters, moments and gradients per GPU). --log_jsonl
writes every logged metric (training loss, pre-clip grad norm, skipped steps, LR, throughput, held-out loss) on rank 0.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import ConstantWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (Callback, CheckpointerCallback, GPUMemoryMonitorCallback,
                                       LMEvaluatorCallbackConfig)
from olmo_core.train.train_module import (TransformerActivationCheckpointingConfig,
                                          TransformerActivationCheckpointingMode as ACMode,
                                          TransformerDataParallelConfig, TransformerTrainModuleConfig)
from olmo_core.utils import seed_all

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class JsonlMetricsCallback(Callback):
    """Append every logged metrics dict to a JSONL file (rank 0)."""
    path: str = ""

    def log_metrics(self, step, metrics):
        if get_rank() != 0 or not self.path:
            return
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"step": step, **{k: float(v) for k, v in metrics.items()}}) + "\n")


def ac_config(spec):
    """none | full | budget:<fraction kept> | blocks:<every k-th block>"""
    if spec == "none":
        return None
    if spec == "full":
        return TransformerActivationCheckpointingConfig(mode=ACMode.full)
    kind, val = spec.split(":")
    if kind == "budget":
        return TransformerActivationCheckpointingConfig(mode=ACMode.budget, activation_memory_budget=float(val))
    if kind == "blocks":
        return TransformerActivationCheckpointingConfig(mode=ACMode.selected_blocks, block_interval=int(val))
    raise ValueError(spec)


def config_diff(ours, theirs, prefix=""):
    """(fields both configs set to different values, fields only one config has) of two nested config dicts."""
    differ, one_side = [], []
    for k in sorted(set(ours) | set(theirs)):
        if k not in ours or k not in theirs:
            one_side.append(f"{prefix}{k}: ours={ours.get(k, '<missing>')!r} OLMo={theirs.get(k, '<missing>')!r}")
            continue
        a, b = ours[k], theirs[k]
        if isinstance(a, dict) and isinstance(b, dict):
            d, o = config_diff(a, b, f"{prefix}{k}.")
            differ, one_side = differ + d, one_side + o
        elif a != b and not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a) == float(b)):
            differ.append(f"{prefix}{k}: ours={a!r} OLMo={b!r}")
    return differ, one_side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--load", required=True, help="OLMo-core checkpoint dir of the starting model")
    ap.add_argument("--load_optim", action="store_true",
                    help="also load the optimizer state from --load (continuing a run: use with --warmup 0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=3.9e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--global_tokens", type=int, default=524288, help="tokens per optimizer step")
    ap.add_argument("--micro_tokens", type=int, default=8192, help="tokens per GPU micro-batch")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--heldout_seq", type=int, default=0, help="held-out instance length (default: --seq)")
    ap.add_argument("--attn", default="torch", choices=["torch", "flash_2"])
    ap.add_argument("--ac", default="none", help="activation checkpointing: none | full | budget:F | blocks:K")
    ap.add_argument("--metrics_every", type=int, default=10)
    ap.add_argument("--log_jsonl", default="", help="append logged metrics here (rank 0)")
    ap.add_argument("--check_config", default="", help="OLMo run config.json: log model/optim differences, stop on "
                                                       "a model difference")
    ap.add_argument("--save_every", type=int, default=191, help="steps (191 x 524,288 tokens = 100M)")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prepare_training_environment()
    seed_all(args.seed)
    tok = TokenizerConfig.dolma2()
    out = ROOT / args.out
    model_cfg = TransformerConfig.olmo3_7B(vocab_size=tok.padded_vocab_size(),
                                           attn_backend=AttentionBackendName(args.attn))
    train_module_cfg = TransformerTrainModuleConfig(
        rank_microbatch_size=args.micro_tokens, max_sequence_length=args.seq,
        optim=SkipStepAdamWConfig(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
                                  group_overrides=[OptimGroupOverride(params=["embeddings.weight"],
                                                                      opts=dict(weight_decay=0.0))]),
        scheduler=ConstantWithWarmup(warmup=args.warmup),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp, param_dtype=DType.bfloat16,
                                                reduce_dtype=DType.float32),
        ac_config=ac_config(args.ac),
        z_loss_multiplier=1e-5, max_grad_norm=1.0,
    )
    if args.check_config and get_rank() == 0:
        ref = json.load(open(ROOT / args.check_config))
        # the attention backend is a kernel choice, not math; every other field both configs set must match.
        # Fields only one side has (OLMo-core v3.0.0 vs the run's August branch) are logged, not fatal.
        diffs, one_side = config_diff(model_cfg.as_config_dict(), ref["model"])
        diffs = [d for d in diffs if not d.startswith("block.attention.backend")]
        opt_diffs, opt_one = config_diff(train_module_cfg.optim.as_config_dict(), ref["train_module"]["optim"])
        print("CONFIG CHECK model, differing:", diffs or "none", flush=True)
        print("CONFIG CHECK model, only one side:", one_side or "none", flush=True)
        print("CONFIG CHECK optim, differing (lr: our schedule sets it):", opt_diffs or "none", flush=True)
        print("CONFIG CHECK optim, only one side:", opt_one or "none", flush=True)
        if diffs:
            raise SystemExit("model config differs from OLMo's run config")
    model = model_cfg.build(init_device="meta")
    train_module = train_module_cfg.build(model)
    dataset = NumpyFSLDatasetConfig(paths=[str(ROOT / args.data)], sequence_length=args.seq, tokenizer=tok,
                                    work_dir=str(out / "dataset_work")).build()
    loader = NumpyDataLoaderConfig(global_batch_size=args.global_tokens, seed=args.seed, num_workers=4).build(
        dataset, dp_process_group=train_module.dp_process_group)
    trainer_cfg = (
        TrainerConfig(save_folder=str(out), save_overwrite=True, metrics_collect_interval=args.metrics_every,
                      cancel_check_interval=10,
                      max_duration=Duration.steps(args.max_steps) if args.max_steps else Duration.epochs(1))
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("checkpointer", CheckpointerCallback(save_interval=args.save_every, save_async=True))
        .with_callback("lm_evaluator", LMEvaluatorCallbackConfig(
            eval_dataset=NumpyPaddedFSLDatasetConfig(paths=[str(ROOT / args.heldout)],
                                                     metadata=[{"label": "heldout-random-pool"}],
                                                     sequence_length=args.heldout_seq or args.seq, tokenizer=tok,
                                                     work_dir=str(out / "dataset_work")),
            eval_interval=args.eval_every, eval_on_startup=True, eval_on_finish=True))
    )
    if args.log_jsonl:
        trainer_cfg = trainer_cfg.with_callback("jsonl", JsonlMetricsCallback(path=str(ROOT / args.log_jsonl)))
    trainer = trainer_cfg.build(train_module, loader)
    if not trainer.maybe_load_checkpoint():   # resume from our own checkpoints if any, else start from --load
        trainer.load_checkpoint(str(ROOT / args.load), load_trainer_state=False, load_optim_state=args.load_optim)
    trainer.fit()
    teardown_training_environment()


if __name__ == "__main__":   # data-loader workers re-import this file
    main()
