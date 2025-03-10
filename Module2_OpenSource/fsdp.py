"""
To run this use (and set nproc_per_node to the number of GPUs on your machine):

torchrun --nproc_per_node=2 fsdp.py
"""

import sys
from io import StringIO

import torch
import torch.distributed as dist
from bignet import BigNet, train_step
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    print(f"Hello {dist.get_rank()=} {dist.get_world_size()=}")

    if dist.get_rank() != 0:
        sys.stdout = StringIO()

    batch_size, seq_len = 32, 1024
    # orig_model = BigNet(device=f"meta")
    orig_model = BigNet(device=f"cuda:{dist.get_rank()}")
    # Ideal when training large weights since it distributes model itself to different GPU
    # It will allow bigger model or bigger batch size.
    # The timing performance is also comparable to DDP

    # With checkpointing and FSDP, we can get the most memory efficient implementation
    model = FSDP(orig_model, device_id=dist.get_rank())
    train_step(model, batch_size=batch_size, seq_len=seq_len)
