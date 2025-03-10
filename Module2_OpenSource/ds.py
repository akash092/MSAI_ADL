"""
To run this use:
deepspeed --include localhost:1 --include localhost:2 ds.py --deepspeed --deepspeed_config ds.json
"""

import sys
from io import StringIO
from time import time

import deepspeed
import torch
import torch.distributed as dist
from bignet import BigNet


def train_step(cmd_args, model, batch_size=1, seq_len=1024):
    model_engine, _optimizer, _, _ = deepspeed.initialize(
        args=cmd_args, model=model, model_parameters=model.parameters()
    )

    # Single forward pass
    print(f"Model weights + opt   {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
    for _ in range(2):
        x = torch.randn(batch_size, seq_len, 4096, device="cuda", requires_grad=True)
        print(f"Model wgh + opt + inp {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
        print(f"Step {_}")
        t0 = time()
        y = model_engine(x).sum()
        y.item()  # Force a sync
        t1 = time()
        print(
            f"  Forward   (peak)    {torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 2**30:6.2f} GB   time = {t1 - t0:0.2f} s"
        )
        t0 = time()
        model_engine.backward(y)
        if x.grad is not None:
            x.grad.view(-1)[0].item()  # Force a sync
        t1 = time()
        print(
            f"  Backward  (peak)    {torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 2**30:6.2f} GB   time = {t1 - t0:0.2f} s"
        )
        print(f"  Backward  (alloc)   {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
        model_engine.step()
        print(f"  Step      (alloc)   {torch.cuda.memory_allocated() / 2**30:6.2f} GB")


if __name__ == "__main__":
    import argparse

    # Deepspeed is more than just a technique. its the whole training package
    # It can be configured to use different techniques of training.
    deepspeed.init_distributed()

    parser = argparse.ArgumentParser(description="My training script.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    print(f"Hello {dist.get_rank()=} {dist.get_world_size()=}")

    if dist.get_rank() != 0:
        sys.stdout = StringIO()

    batch_size, seq_len = 4, 1024
    model = BigNet(device=f"cuda:{dist.get_rank()}")
    # model = FSDP(orig_model, device_id=dist.get_rank())
    train_step(cmd_args, model, batch_size=batch_size, seq_len=seq_len)
