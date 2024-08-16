from typing import Any
import torch
import torch.nn as nn
import block


class TransformerEncoder(nn.Module):
  def __init__(self, num_block: int, *block_args: Any, **block_kwargs: Any):
    super().__init__()
    self.blocks = nn.Sequential()
    for _ in range(num_block):
      self.blocks.append(block.EncoderBlock(*block_args, **block_kwargs))
    
  def forward(self, x: torch.Tensor):
    return self.blocks(x)

class TransformerDecoder(nn.Module):
  def __init__(self, num_block: int, *block_args: Any, **block_kwargs: Any):
    super().__init__()
    self.blocks = nn.ModuleList()
    for _ in range(num_block):
      self.blocks.append(block.DecoderBlock(*block_args, **block_kwargs))

  def forward(
    self,
    tgt: torch.Tensor,
    memory: torch.Tensor,
    tgt_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    for decoder_block in self.blocks:
      tgt = decoder_block(tgt, memory, tgt_mask)
    return tgt

class Transformer(nn.Module):
  def __init__(
      self,
      num_encoder_block: int,
      num_decoder_block: int,
      embed_dim: int,
      num_heads: int,
      ffn_dim: int,
      dropout_prob: float = 0.1,
      layer_norm_eps: float = 1e-5,
  ):
    super().__init__()
    params = dict(
      embed_dim=embed_dim,
      num_heads=num_heads,
      ffn_dim=ffn_dim,
      dropout_prob=dropout_prob,
      layer_norm_eps=layer_norm_eps,
    )
    self.encoder = TransformerEncoder(num_encoder_block, **params)
    self.decoder = TransformerDecoder(num_decoder_block, **params)
  
  def forward(
    self,
    src: torch.Tensor,
    tgt: torch.Tensor,
    tgt_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    memory = self.encoder(src)
    return self.decoder(tgt, memory, tgt_mask)
