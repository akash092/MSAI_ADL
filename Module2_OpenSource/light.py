"""
To run this use:

python3 light.py
"""

import lightning as L
import torch
from bignet import BigNet
from lightning.pytorch.strategies import FSDPStrategy


# define the LightningModule
class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (x,) = batch
        y = self.model(x).sum()
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", y)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size, seq_len, length=10):
        self.shape = (batch_size, seq_len, 4096)
        self.len = length

    def __getitem__(self, index):
        return torch.rand(*self.shape, device="cuda")

    def __len__(self):
        return self.len


if __name__ == "__main__":
    batch_size, seq_len = 4, 1024

    # init the encoder and decoder
    model = BigNet(device="cpu")
    lit_model = LitModel(model)
    strategy = FSDPStrategy(
        # Default: Shard weights, gradients, optimizer state (1 + 2 + 3)
        sharding_strategy="FULL_SHARD",
    )

    # Lighting is also the whole package and is encouraged if running FSDP
    trainer = L.Trainer(max_epochs=2, accelerator="cuda", strategy=strategy)
    train_loader = torch.utils.data.DataLoader(RandomDataset(batch_size, seq_len))
    trainer.fit(model=lit_model, train_dataloaders=train_loader)
