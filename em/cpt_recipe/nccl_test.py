"""NCCL check on a new node: all-gather and reduce-scatter of FSDP-sized buffers (one OLMo-3-7B block, ~0.2B params)
over the local GPUs, timed. Run once with NCCL's defaults and once with NCCL_NVLS_ENABLE=0 (on lambda-4xh100 the
default hangs, memory nccl-nvls-lambda), each under `timeout`, so a hang shows as a timeout:
  CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 300 .olmocore/.venv/bin/torchrun --nproc-per-node=4 em/cpt_recipe/nccl_test.py
Stress mode (fabric check after a fault): STRESS_SECONDS=90 runs fp32 all-reduce of the same buffer for that long and
prints the bus bandwidth every 10 s (min / max over the run at the end).
"""
import datetime
import os
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    r, w = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    n = 202_000_000 // w * w                                  # elements of one block
    full_bf16 = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    shard_bf16 = torch.randn(n // w, device="cuda").to(torch.bfloat16)
    full_f32 = torch.randn(n, device="cuda")
    shard_f32 = torch.empty(n // w, device="cuda")
    for name, fn, nbytes in [
        ("all_gather bf16", lambda: dist.all_gather_into_tensor(full_bf16, shard_bf16), n * 2),
        ("reduce_scatter fp32", lambda: dist.reduce_scatter_tensor(shard_f32, full_f32), n * 4),
        ("all_reduce fp32", lambda: dist.all_reduce(full_f32), n * 4),
    ]:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0, k = time.time(), 20
        for _ in range(k):
            fn()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / k
        # bus bandwidth as nccl-tests defines it: (w-1)/w * bytes / time (x2 for all-reduce)
        bus = (w - 1) / w * nbytes / dt / 1e9 * (2 if "all_reduce" in name else 1)
        if r == 0:
            print(f"{name}: {nbytes / 1e9:.2f} GB in {dt * 1e3:.1f} ms, bus {bus:.0f} GB/s", flush=True)
    stress = float(os.environ.get("STRESS_SECONDS", 0))
    if stress:
        t_end, bws, t_mark, k = time.time() + stress, [], time.time(), 0
        while time.time() < t_end:
            dist.all_reduce(full_f32)
            k += 1
            if time.time() - t_mark >= 10:
                torch.cuda.synchronize()
                dt = (time.time() - t_mark) / k
                bws.append(2 * (w - 1) / w * n * 4 / dt / 1e9)
                if r == 0:
                    print(f"stress all_reduce fp32: {bws[-1]:.0f} GB/s bus ({k} calls)", flush=True)
                t_mark, k = time.time(), 0
        torch.cuda.synchronize()
        if r == 0 and bws:
            print(f"stress {stress:.0f} s: bus bandwidth min {min(bws):.0f} / max {max(bws):.0f} GB/s", flush=True)
    if r == 0:
        print(f"NCCL OK world {w} NCCL_NVLS_ENABLE={os.environ.get('NCCL_NVLS_ENABLE', '<unset>')} "
              f"nccl {torch.cuda.nccl.version()}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
