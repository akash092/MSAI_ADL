import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an arbitrary tensor from (H, W, C) to (C, H, W) format.
    This allows us to switch from trnasformer-style channel-last to pytorch-style channel-first
    images. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    The opposite of hwc_to_chw. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchifyLinear(torch.nn.Module):
    """
    Takes an image tensor of the shape (B, H, W, 3) and patchifies it into
    an embedding tensor of the shape (B, H//patch_size, W//patch_size, latent_dim).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, input_channel: int = 3, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        # Original
        # self.patch_conv = torch.nn.Conv2d(input_channel, latent_dim, patch_size, patch_size, bias=False)
        
        # Modified for BSQ
        layers = []
        layers.append(torch.nn.Conv2d(input_channel, latent_dim, patch_size, patch_size))
        layers.append(torch.nn.GELU())
        padding = (patch_size - 1) // 2 # to preserve input dimension
        layers.append(torch.nn.Conv2d(latent_dim, latent_dim, patch_size, 1, padding))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Conv2d(latent_dim, latent_dim, patch_size, 1, padding))
        layers.append(torch.nn.GELU())
        self.patch_conv = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, input_channel) an image tensor dtype=float normalized to -1 ... 1

        return: (B, H//patch_size, W//patch_size, latent_dim) a patchified embedding tensor
        """
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))


class UnpatchifyLinear(torch.nn.Module):
    """
    Takes an embedding tensor of the shape (B, w, h, latent_dim) and reconstructs
    an image tensor of the shape (B, w * patch_size, h * patch_size, input_channel).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, input_channel: int = 3, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        # original
        # self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, input_channel, patch_size, patch_size, bias=False)

        # modified for BSQ
        layers = []
        layers.append(torch.nn.ConvTranspose2d(latent_dim, input_channel, patch_size, patch_size))
        # layers.append(torch.nn.GELU())
        # layers.append(torch.nn.Conv2d(input_channel, input_channel, patch_size, 1,patch_size//2))
        # layers.append(torch.nn.GELU())
        self.unpatch_conv = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, w, h, latent_dim) an embedding tensor

        return: (B, H * patch_size, W * patch_size, input_channel) a image tensor
        """
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image x (B, H, W, 3) into a tensor (B, h, w, bottleneck),
        where h = H // patch_size, w = W // patch_size and bottleneck is the size of the
        AutoEncoders bottleneck.
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor x (B, h, w, bottleneck) into an image (B, H, W, 3),
        We will train the auto-encoder such that decode(encode(x)) ~= x.
        """


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """
    Implement a PatchLevel AutoEncoder

    Hint: Convolutions work well enough, no need to use a transformer unless you really want.
    Hint: See PatchifyLinear and UnpatchifyLinear for how to use convolutions with the input and
          output dimensions given.
    Hint: You can get away with 3 layers or less.
    Hint: Many architectures work here (even a just PatchifyLinear / UnpatchifyLinear).
          However, later parts of the assignment require both non-linearities (i.e. GeLU) and
          interactions (i.e. convolutions) between patches.
    """

    class PatchEncoder(torch.nn.Module):
        """
        (Optionally) Use this class to implement an encoder.
                     It can make later parts of the homework easier (reusable components).
        """

        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            # super().__init__()
            # layers = []
            # layers.append(torch.nn.Conv2d(3, latent_dim, patch_size, patch_size))
            # layers.append(torch.nn.GELU())
            # padding = (patch_size - 1) // 2 # to preserve input dimension
            # layers.append(torch.nn.Conv2d(latent_dim, latent_dim, patch_size, 1, padding))
            # layers.append(torch.nn.GELU())
            # layers.append(torch.nn.Conv2d(latent_dim, latent_dim, patch_size, 1, padding))
            # layers.append(torch.nn.GELU())
            # self.model = torch.nn.Sequential(*layers)
            super().__init__()
            layers = []
            layers.append(torch.nn.Conv2d(3, latent_dim, patch_size, patch_size))
            layers.append(torch.nn.BatchNorm2d(latent_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Conv2d(latent_dim, bottleneck, kernel_size=1))
            layers.append(torch.nn.BatchNorm2d(latent_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Conv2d(latent_dim, bottleneck, kernel_size=1))
            layers.append(torch.nn.BatchNorm2d(latent_dim))
            layers.append(torch.nn.GELU())
            self.model = torch.nn.Sequential(*layers)

            self.skip = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # return chw_to_hwc(self.model(hwc_to_chw(x)))
            x = hwc_to_chw(x)
            x = self.skip(x) + self.model(x)
            return chw_to_hwc(x)

    class PatchDecoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            # super().__init__()
            # self.model = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size)
            super().__init__()
            # self.conv1 = torch.nn.Conv2d(bottleneck, latent_dim, kernel_size=1)
            # self.act1 = torch.nn.GELU()
            # self.conv2 = torch.nn.Conv2d(bottleneck, latent_dim, kernel_size=1)
            # self.act2 = torch.nn.GELU()
            # self.conv3 = torch.nn.ConvTranspose2d(latent_dim, 3, kernel_size=patch_size, stride=patch_size)
            # self.act3 = torch.nn.GELU()
            layers = []

            layers.append(torch.nn.Conv2d(bottleneck, latent_dim, kernel_size=1))
            layers.append(torch.nn.BatchNorm2d(latent_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Conv2d(bottleneck, latent_dim, kernel_size=1))
            layers.append(torch.nn.BatchNorm2d(latent_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size))
            layers.append(torch.nn.BatchNorm2d(3))
            layers.append(torch.nn.GELU())
            self.model = torch.nn.Sequential(*layers)

            self.skip = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # return chw_to_hwc(self.model(hwc_to_chw(x)))
            x = hwc_to_chw(x)
            # x = self.act1(self.conv1(x))
            # x = self.act2(self.conv2(x))
            # x = self.act3(self.conv3(x))
            x = self.skip(x) + self.model(x)
            return chw_to_hwc(x)

    def __init__(self, patch_size: int = 25, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        # self.model_encoder = torch.nn.Sequential(
        #     PatchifyLinear(3, patch_size,latent_dim)
        # )
        # self.model_decoder = torch.nn.Sequential(
        #     UnpatchifyLinear(3, patch_size, latent_dim)
        # )
        self.encoder = self.PatchEncoder(patch_size, latent_dim, bottleneck)
        
        self.decoder = self.PatchDecoder(patch_size, latent_dim, bottleneck)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        You can return an empty dictionary if you don't have any additional terms.
        """
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        return reconstructed, {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
