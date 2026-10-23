"""CPT trainer benchmark (em/TODO.md): throughput of OLMo-core on OLMo-3-7B (random init, dummy tokens), FSDP over the local GPUs.
  torchrun --nproc-per-node=4 bench_olmocore.py --seq 1024 --micro 8192 --global 65536 [--ac] [--no-compile]
"""
import argparse
import os

import numpy as np

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, LinearWithWarmup, OptimGroupOverride
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import GPUMemoryMonitorCallback
from olmo_core.train.train_module import TransformerDataParallelConfig, TransformerTrainModuleConfig
from olmo_core.train.train_module.transformer.config import TransformerActivationCheckpointingConfig
from olmo_core.utils import seed_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=8192, help="tokens per GPU micro-batch")
    ap.add_argument("--global", dest="glob", type=int, default=65536, help="tokens per optimizer step")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--ac", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--dp", default="fsdp")
    args = ap.parse_args()

    prepare_training_environment()
    tok = TokenizerConfig.dolma2()
    data = "/home/ubuntu/bench/dummy_tokens.npy"
    if get_rank() == 0 and not os.path.exists(data):   # raw uint32 token memmap, EOS every 800 tokens
        rng = np.random.default_rng(0)
        a = np.memmap(data, mode="w+", dtype=np.uint32, shape=(300_000_000,))
        for i in range(0, len(a), 10_000_000):
            x = rng.integers(0, 100_000, 10_000_000, dtype=np.uint32)
            x[::800] = tok.eos_token_id
            a[i:i + 10_000_000] = x
        a.flush()
    import torch.distributed as dist
    dist.barrier()
    seed_all(0)
    model = TransformerConfig.olmo3_7B(vocab_size=tok.padded_vocab_size(), attn_backend=AttentionBackendName.torch).build(
        init_device="meta")
    tm = TransformerTrainModuleConfig(
        rank_microbatch_size=args.micro, max_sequence_length=args.seq,
        optim=AdamWConfig(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95),
                          group_overrides=[OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))]),
        scheduler=LinearWithWarmup(warmup_steps=10),
        compile_model=not args.no_compile,
        dp_config=TransformerDataParallelConfig(name=DataParallelType(args.dp), param_dtype=DType.bfloat16,
                                                reduce_dtype=DType.float32),
        ac_config=TransformerActivationCheckpointingConfig() if args.ac else None,
        max_grad_norm=1.0,
    ).build(model)
    ds = NumpyFSLDatasetConfig(paths=[data], sequence_length=args.seq, tokenizer=tok,
                               work_dir="/home/ubuntu/bench/olmocore_work").build()
    dl = NumpyDataLoaderConfig(global_batch_size=args.glob, seed=0, num_workers=4).build(
        ds, dp_process_group=tm.dp_process_group)
    trainer = (TrainerConfig(save_folder="/home/ubuntu/bench/olmocore_save", save_overwrite=True, no_checkpoints=True,
                             metrics_collect_interval=5, max_duration=Duration.steps(args.steps))
               .with_callback("gpu_monitor", GPUMemoryMonitorCallback())).build(tm, dl)
    trainer.fit()
    teardown_training_environment()


if __name__ == "__main__":   # data-loader workers re-import this file
    main()
