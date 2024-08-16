import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
  def __init__(self, embed_dim: int):
    super().__init__()
    self.embed_dim = embed_dim
    self.scaling = 1.0 / torch.sqrt(torch.tensor(embed_dim))

    self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
    self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
    self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)
    self.to_o = nn.Linear(embed_dim, embed_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, N, E]
    
    # q/k/v: [B, N, E]
    q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
    # dot: [B, N, N]
    dot = self.scaling*(q @ k.mT)
    # attn: [B, N, N]
    attn = F.softmax(dot, -1)
    # out: [B, N, N]
    out = attn @ v
    return self.to_o(out)


class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim: int, num_heads: int):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    assert embed_dim % num_heads == 0
    self.dim_per_head = embed_dim // num_heads
    self.scaling = 1.0 / torch.sqrt(torch.tensor(self.dim_per_head))

    self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
    self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
    self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)
    self.to_o = nn.Linear(embed_dim, embed_dim)

  def split_heads(self, x: torch.Tensor) -> torch.Tensor:
    """Seperate the head dimension and fold it into the batch dimension."""
    b, n, _ = x.shape
    x = x.view(b, n, self.num_heads, self.dim_per_head)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b*self.num_heads, n, self.dim_per_head)
    return x
  
  def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
    bh, n, _ = x.shape
    b = bh // self.num_heads
    x = x.view(b, self.num_heads, n, self.dim_per_head)
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
    # x: [B, M/N, E]

    # q/k/v: [B*H, M/N, F]  (E = H*F)
    q = self.split_heads(self.to_q(xq))
    k = self.split_heads(self.to_k(xk))
    v = self.split_heads(self.to_v(xv))
    
    # dot: [B*H, M, N]
    dot = self.scaling*(q @ k.mT)
    if attn_mask is not None:
      dot.masked_fill_(attn_mask, -torch.inf)
    # attn: [B*H, M, N]
    attn = F.softmax(dot, -1)
    # out: [B, M, E]
    out = self.merge_heads(attn @ v)
    return self.to_o(out)


class MultiHeadAttentionEinsum(nn.Module):
  def __init__(self, embed_dim: int, num_heads: int):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    assert embed_dim % num_heads == 0
    self.dim_per_head = embed_dim // num_heads
    self.scaling = 1.0 / torch.sqrt(torch.tensor(self.dim_per_head))

    self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
    self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
    self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)
    self.to_o = nn.Linear(embed_dim, embed_dim)

    self.split_heads = (
        lambda x: x.view(x.shape[0], x.shape[1], num_heads, self.dim_per_head)
    )
    
  def forward(
      self,
      xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor
  ) -> torch.Tensor:
    # x: [B, M/N, E]

    # q/k/v: [B, M/N, H, F]  (E = H*F)
    q = self.split_heads(self.to_q(xq))
    k = self.split_heads(self.to_k(xk))
    v = self.split_heads(self.to_v(xv))

    dot = self.scaling*torch.einsum('bmhf,bnhf->bhmn', q, k)
    attn = F.softmax(dot, -1)
    out = torch.einsum('bhmn,bnhf->bmhf', attn, v)
    
    # Note the viewing of to_o's weight can be done during initialization. But
    # it is kept here for easy parameter matching in the unit test.
    to_o_weight = self.to_o.weight.view(
      self.embed_dim, self.num_heads, self.dim_per_head
    )
    
    out = torch.einsum('bmhf,ehf->bme', out, to_o_weight) + self.to_o.bias
    return out
