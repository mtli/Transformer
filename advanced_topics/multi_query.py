import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
  def __init__(self, embed_dim: int, num_heads: int):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    assert embed_dim % num_heads == 0
    self.dim_per_head = embed_dim // num_heads
    self.scaling = 1.0 / torch.sqrt(torch.tensor(self.dim_per_head))

    self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
    # Shared k and v across all heads
    self.to_k = nn.Linear(embed_dim, self.dim_per_head, bias=False)
    self.to_v = nn.Linear(embed_dim, self.dim_per_head, bias=False)
    self.to_o = nn.Linear(embed_dim, embed_dim)

  def split_heads(self, x: torch.Tensor) -> torch.Tensor:
    """Separate the head dimension and fold it into the batch dimension."""
    b, n, _ = x.shape
    x = x.view(b, n, self.num_heads, self.dim_per_head)
    x = x.transpose(1, 2).contiguous()
    return x
  
  def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
    b, _, n, _ = x.shape
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, n, self.embed_dim)
    return x

  def forward(
    self,
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    attn_mask: torch.Tensor | None = None
  ) -> torch.Tensor:
    # xq: [B, M, E], xk: [B, N, E], xv: [B, N, E]

    # q: [B, H, M, F]  (E = H*F)
    q = self.split_heads(self.to_q(xq))
    # k/v: [B, 1, N, F]  (Shared across heads, hence 1 head dimension)
    k = self.to_k(xk).unsqueeze(1)
    v = self.to_v(xv).unsqueeze(1)

    # dot: [B, H, M, N]
    dot = self.scaling * (q @ k.transpose(-2, -1))
    if attn_mask is not None:
      dot.masked_fill_(attn_mask, -torch.inf)
    
    # attn: [B, H, M, N]
    attn = F.softmax(dot, -1)
    # out: [B, H, M, F]
    out = attn @ v
    
    # Merge heads and project output
    out = self.merge_heads(out)
    return self.to_o(out)