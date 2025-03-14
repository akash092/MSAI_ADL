import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        # Embedding layer to map token indices to latent space
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        
        # Transformer encoder layer for processing sequences
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=8)
        self.transformer = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        
        # Output layer to predict token probabilities
        self.output_layer = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        seq_len = h * w
        
        # Flatten input and embed token indices
        x = x.view(B, seq_len)  # Flatten for processing
        x = self.embedding(x)
        
        # Generate a causal mask to ensure autoregressive property
        mask = torch.nn.Transformer.generate_square_subsequent_mask(B)
        
        # Shift input sequence by one position for autoregressive processing
        x_shifted = torch.nn.functional.pad(x[:, :-1, :], (0, 0, 1, 0), value=0)
        
        # Pass through the transformer with the causal mask
        x_encoded = self.transformer(x_shifted, mask=mask)
        
        # Compute output logits
        logits = self.output_layer(x_encoded)
        
        # Reshape output back to (B, h, w, n_tokens)
        logits = logits.view(B, h, w, -1)
        
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        raise NotImplementedError()
