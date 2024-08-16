import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Scaling is useful when sharing weights with the pre-prediction linear
        # layer.
        self.scaling = embed_dim ** 0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.scaling*self.embedding(tokens.long())
