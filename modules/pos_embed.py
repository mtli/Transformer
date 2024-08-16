import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
  def __init__(self, max_len: int, embed_dim: int, dropout_prob: float):
    super().__init__()
    assert embed_dim % 2 == 0
    positions = torch.arange(max_len)
    ang_freq = torch.pow(1e4, -1/embed_dim*torch.arange(embed_dim // 2))
    angles = torch.outer(positions, ang_freq)
    pos_enc = torch.cat((angles.cos(), angles.sin()), dim=1).unsqueeze_(0)
    # Register it as a buffer so that it can be moved to appropriate device when
    # .to(device) is called on the parent modules.
    self.register_buffer('pos_enc', pos_enc)
    self.dropout = nn.Dropout(dropout_prob)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    pos_enc = self.pos_enc[:, :x.shape[1]]
    return self.dropout(x + pos_enc)
    