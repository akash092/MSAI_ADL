"""
To run this use:

python3 bignet.py
"""

import warnings
from time import time

import torch

warnings.filterwarnings("ignore")
# THIS IS THE MAIN STARTING CODE

class BigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, hidden_size, device="cuda"):
            super().__init__()
            self.transformer_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=4 * hidden_size,
                dropout=0,
                batch_first=True,
                device=device,
            )

        def forward(self, x):
            from torch.utils.checkpoint import checkpoint

            # return self.transformer_layer(x) # when no checkpoint

            # With checkpoint, use below
            return checkpoint(self.transformer_layer, x, use_reentrant=False)

    def __init__(self, input_size=4096, hidden_size=2048, output_size=10, device="cuda"):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, device=device),
            *[self.Block(hidden_size, device=device) for _ in range(8)],
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, device=device),
        )

    def forward(self, x):
        return self.model(x)


def test_forward(model, batch_size=1, seq_len=1024):
    # Single forward pass
    print(f"Model weights          {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
    x = torch.randn(batch_size, seq_len, 4096, device="cuda")
    print(f"Model weights + inputs {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
    with torch.no_grad():
        model(x).sum()
    print(f"Forward (peak)         {torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 2**30:6.2f} GB")
    print(f"Forward (alloc)        {torch.cuda.memory_allocated() / 2**30:6.2f} GB")


def test_backward(model, batch_size=1, seq_len=1024):
    # Single forward pass
    print(f"Model weights          {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
    x = torch.randn(batch_size, seq_len, 4096, device="cuda")
    print(f"Model weights + inputs {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
    y = model(x).sum()
    print(f"Forward (peak)         {torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 2**30:6.2f} GB")
    y.backward()
    print(f"Backward (peak)        {torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 2**30:6.2f} GB")
    print(f"Backward (alloc)       {torch.cuda.memory_allocated() / 2**30:6.2f} GB")


def train_step(model, batch_size=1, seq_len=1024):
    # Single forward pass
    print(f"Model weights         {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"Model weights + opt   {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
    for _ in range(2):
        # We are only running 2 iterations since the size of model after 2 doesnt change
        print(f"Step {_}")
        x = torch.randn(batch_size, seq_len, 4096, device="cuda", requires_grad=True)
        print(f"  Model wgh + opt + inp {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
        t0 = time()
        y = model(x).sum()
        y.item()  # Force a sync
        t1 = time()
        print(
            f"  Forward   (peak)    {torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 2**30:6.2f} GB   time = {t1 - t0:0.2f} s"
        )
        t0 = time()
        y.backward()
        if x.grad is not None:
            x.grad.view(-1)[0].item()  # Force a sync
        t1 = time()
        print(
            f"  Backward  (peak)    {torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 2**30:6.2f} GB   time = {t1 - t0:0.2f} s"
        )
        print(f"  Backward  (alloc)   {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
        optim.step()
        print(f"  Step      (alloc)   {torch.cuda.memory_allocated() / 2**30:6.2f} GB")
        optim.zero_grad()
        print(f"  zero_grad (alloc)   {torch.cuda.memory_allocated() / 2**30:6.2f} GB")


if __name__ == "__main__":
    batch_size, seq_len = 4, 1024
    model = BigNet(device="cuda")

    # test_forward(model, batch_size=batch_size, seq_len=seq_len)
    # test_backward(model, batch_size=batch_size, seq_len=seq_len)
    train_step(model, batch_size=batch_size, seq_len=seq_len)
