import unittest
import torch
import torch.nn as nn
import multi_query

ATOL = 1e-6


class TestMultiQueryAttention(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(0)
    
  def _create_gt_module(self, my_module: nn.Module):
      num_heads = getattr(my_module, 'num_heads', 1)
      gt_module = nn.MultiheadAttention(
          embed_dim=my_module.embed_dim,
          num_heads=num_heads,
          batch_first=True,
        )
      full_k = my_module.to_k.weight.repeat(num_heads, 1)
      full_v = my_module.to_v.weight.repeat(num_heads, 1)
      gt_module.in_proj_weight = nn.Parameter(
          torch.cat([my_module.to_q.weight, full_k, full_v], dim=0)
      )
      gt_module.out_proj.weight = my_module.to_o.weight
      gt_module.out_proj.bias = my_module.to_o.bias
      return gt_module
  
  def test_multi_head_attention(self):
    bs, seq_len = 3, 5
    embed_dim, num_heads = 4, 2
    my_mhsa = multi_query.MultiQueryAttention(embed_dim, num_heads)
    gt_mhsa = self._create_gt_module(my_mhsa)

    x = torch.rand(bs, seq_len, embed_dim)
    
    with self.subTest('without_a_mask'):
      my_out = my_mhsa(x, x, x)
      gt_out = gt_mhsa(x, x, x, need_weights=False)[0]
      torch.testing.assert_close(my_out, gt_out, atol=ATOL, rtol=0)
    
    with self.subTest('with_a_casual_mask'):
      attn_mask = torch.triu(
          torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
      )
      my_out = my_mhsa(x, x, x, attn_mask=attn_mask)
      gt_out = gt_mhsa(
        x, x, x, attn_mask=attn_mask, need_weights=False, is_causal=True
      )[0]
      torch.testing.assert_close(my_out, gt_out, atol=ATOL, rtol=0)

    with self.subTest('with_a_random_mask'):
      attn_mask = torch.rand(bs, num_heads, seq_len, seq_len) > 0.5
      # At least one column in each row is not masked.
      attn_mask[..., 0] = False
      attn_mask_bh = attn_mask.view(bs*num_heads, seq_len, seq_len)
      my_out = my_mhsa(x, x, x, attn_mask=attn_mask)
      gt_out = gt_mhsa(x, x, x, attn_mask=attn_mask_bh, need_weights=False)[0]
      torch.testing.assert_close(my_out, gt_out, atol=ATOL, rtol=0)


if __name__ == '__main__':
  unittest.main()
