from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401

class HalfLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        Implement a half-precision Linear Layer.
        Feel free to use the torch.nn.Linear class as a parent class (it makes load_state_dict easier, names match).
        Feel free to set self.requires_grad_ to False, we will not backpropagate through this layer.
        """
        # TODO: Implement me
        # self.requires_grad_(False)
        # -- My first attempt --
        # self.model = torch.nn.Sequential(super().__init__(in_features, out_features, bias, dtype=torch.float16))
        # --
        super().__init__(in_features, out_features, bias, dtype=torch.float16)
        self.requires_grad_(False)

        # for param in super().parameters():
        #     param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hint: Use the .to method to cast a tensor to a different dtype (i.e. torch.float16 or x.dtype)
        # The input and output should be of x.dtype = torch.float32
        # TODO: Implement me
        # -- my first attempt
        # y = self.model(x.to(torch.float16))
        #return y.to(x.dtype)
        # --
        x_mod = x.to(torch.float16)
        
        # Perform the linear transformation (in float16)
        return super().forward(x_mod).to(x.dtype)


class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision. Make sure that the normalization uses full
    precision though to avoid numerical instability.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
