import torch
import torch.nn as nn
import attention


class EncoderBlock(nn.Module):
  def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      ffn_dim: int,
      dropout_prob: float = 0.1,
      layer_norm_eps: float = 1e-5,
  ):
    super().__init__()
    self.self_attn = attention.MultiHeadAttention(embed_dim, num_heads)
    self.ffn = nn.Sequential(
        nn.Linear(embed_dim, ffn_dim),
        nn.Dropout(dropout_prob, inplace=True),
        nn.ReLU(inplace=True),
        nn.Linear(ffn_dim, embed_dim),
    )
    self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
    self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
    self.dropout1 = nn.Dropout(dropout_prob, inplace=True)
    self.dropout2 = nn.Dropout(dropout_prob, inplace=True)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.dropout1(self.self_attn(x, x, x))
    x = self.norm1(x)
    x = x + self.dropout2(self.ffn(x))
    x = self.norm2(x)
    return x

class DecoderBlock(nn.Module):
  def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      ffn_dim: int,
      dropout_prob: float = 0.1,
      layer_norm_eps: float = 1e-5,
  ):
    super().__init__()
    self.self_attn = attention.MultiHeadAttention(embed_dim, num_heads)
    self.cross_attn = attention.MultiHeadAttention(embed_dim, num_heads)
    self.ffn = nn.Sequential(
        nn.Linear(embed_dim, ffn_dim),
        nn.Dropout(dropout_prob, inplace=True),
        nn.ReLU(inplace=True),
        nn.Linear(ffn_dim, embed_dim),
    )
    self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
    self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
    self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
    self.dropout1 = nn.Dropout(dropout_prob, inplace=True)
    self.dropout2 = nn.Dropout(dropout_prob, inplace=True)
    self.dropout3 = nn.Dropout(dropout_prob, inplace=True)
    
  def forward(
    self,
    x: torch.Tensor, y: torch.Tensor,
    attn_mask: torch.Tensor | None = None
  ) -> torch.Tensor:
    x = x + self.dropout1(self.self_attn(x, x, x, attn_mask=attn_mask))
    x = self.norm1(x)
    x = x + self.dropout2(self.cross_attn(x, y, y))
    x = self.norm2(x)
    x = x + self.dropout3(self.ffn(x))
    x = self.norm3(x)
    return x
