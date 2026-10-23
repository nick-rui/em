"""Continued pretraining of OLMo-3-7B stage 1 with OLMo-core (the 1B-token CPT arm on sh_em3; em/TODO.md).

Full-parameter training of the stage-1 checkpoint (converted from HF with OLMo-core's convert_checkpoint_from_hf.py;
checked in fp32 on CPT tokens: identical loss, max |dlogit| 1.7e-4), sharded over the local GPUs (FSDP, bf16 compute,
fp32 reduce), compiled, PyTorch SDPA attention. Stage 1's own optimizer settings (OLMo-core's official
OLMo-3-1025-7B-pretrain config: SkipStepAdamW, betas 0.9 / 0.95, weight decay 0.1 except embeddings, z-loss 1e-5, clip
1.0) at the learning rate stage 1 ended on (about 3.9e-5 at step 1,413,814), with a short warmup for the fresh optimizer
state and then constant, as stage 1 itself was not annealed. One pass over the token file (raw uint32, documents cut
to 1,024 tokens + EOS by em/scripts/pool20m_cpt_data.py) in 1,024-token windows. Held-out loss on random pool documents
at the start and every --eval_every steps; a checkpoint every --save_every steps (with the optimizer state).

  NCCL_NVLS_ENABLE=0 torchrun --nproc-per-node=4 em/cpt_olmocore.py --data data/pool20m/cpt_top1b/tokens.npy \
      --heldout data/pool20m/heldout_random/tokens.npy --load runs/olmocore/s1_step1413814 --out runs/olmocore/cpt_top1b
"""
import argparse
from pathlib import Path

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import ConstantWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import CheckpointerCallback, GPUMemoryMonitorCallback, LMEvaluatorCallbackConfig
from olmo_core.train.train_module import TransformerDataParallelConfig, TransformerTrainModuleConfig
from olmo_core.utils import seed_all

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--load", required=True, help="OLMo-core checkpoint dir of the starting model (weights only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=3.9e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--global_tokens", type=int, default=524288, help="tokens per optimizer step")
    ap.add_argument("--micro_tokens", type=int, default=8192, help="tokens per GPU micro-batch")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--save_every", type=int, default=477, help="steps (477 x 524,288 tokens = 250M)")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prepare_training_environment()
    seed_all(args.seed)
    tok = TokenizerConfig.dolma2()
    out = ROOT / args.out
    model = TransformerConfig.olmo3_7B(vocab_size=tok.padded_vocab_size(),
                                       attn_backend=AttentionBackendName.torch).build(init_device="meta")
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=args.micro_tokens, max_sequence_length=args.seq,
        optim=SkipStepAdamWConfig(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
                                  group_overrides=[OptimGroupOverride(params=["embeddings.weight"],
                                                                      opts=dict(weight_decay=0.0))]),
        scheduler=ConstantWithWarmup(warmup=args.warmup),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp, param_dtype=DType.bfloat16,
                                                reduce_dtype=DType.float32),
        z_loss_multiplier=1e-5, max_grad_norm=1.0,
    ).build(model)
    dataset = NumpyFSLDatasetConfig(paths=[str(ROOT / args.data)], sequence_length=args.seq, tokenizer=tok,
                                    work_dir=str(out / "dataset_work")).build()
    loader = NumpyDataLoaderConfig(global_batch_size=args.global_tokens, seed=args.seed, num_workers=4).build(
        dataset, dp_process_group=train_module.dp_process_group)
    trainer = (
        TrainerConfig(save_folder=str(out), save_overwrite=True, metrics_collect_interval=10, cancel_check_interval=10,
                      max_duration=Duration.steps(args.max_steps) if args.max_steps else Duration.epochs(1))
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("checkpointer", CheckpointerCallback(save_interval=args.save_every, save_async=True))
        .with_callback("lm_evaluator", LMEvaluatorCallbackConfig(
            eval_dataset=NumpyPaddedFSLDatasetConfig(paths=[str(ROOT / args.heldout)],
                                                     metadata=[{"label": "heldout-random-pool"}],
                                                     sequence_length=args.seq, tokenizer=tok,
                                                     work_dir=str(out / "dataset_work")),
            eval_interval=args.eval_every, eval_on_startup=True, eval_on_finish=True))
    ).build(train_module, loader)
    if not trainer.maybe_load_checkpoint():   # resume from our own checkpoints if any, else start from --load
        trainer.load_checkpoint(str(ROOT / args.load), load_trainer_state=False, load_optim_state=False)
    trainer.fit()
    teardown_training_environment()


if __name__ == "__main__":   # data-loader workers re-import this file
    main()
