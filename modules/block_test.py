import unittest
import torch
import torch.nn as nn
import block

ATOL = 1e-6


class TestBlock(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(0)

  def _match_params(
      self,
      my_module: block.EncoderBlock | block.DecoderBlock,
      gt_module: nn.TransformerEncoderLayer | nn.TransformerDecoderLayer,
      match_mhca: bool = False,
  ):
    my_mhsa = my_module.self_attn
    gt_module.self_attn.in_proj_weight = nn.Parameter(
        torch.cat([
            my_mhsa.to_q.weight, my_mhsa.to_k.weight, my_mhsa.to_v.weight,
        ], dim=0)
    )
    gt_module.self_attn.out_proj.weight = my_mhsa.to_o.weight
    gt_module.self_attn.out_proj.bias = my_mhsa.to_o.bias
    
    if match_mhca:
      my_mhca = my_module.cross_attn
      gt_module.multihead_attn.in_proj_weight = nn.Parameter(
          torch.cat([
              my_mhca.to_q.weight, my_mhca.to_k.weight, my_mhca.to_v.weight
          ], dim=0)
      )
      gt_module.multihead_attn.out_proj.weight = my_mhca.to_o.weight
      gt_module.multihead_attn.out_proj.bias = my_mhca.to_o.bias
      
    gt_module.linear1.weight = my_module.ffn[0].weight
    gt_module.linear1.bias = my_module.ffn[0].bias
    gt_module.linear2.weight = my_module.ffn[3].weight
    gt_module.linear2.bias = my_module.ffn[3].bias


  def test_encoder_block(self):
    bs, seq_len = 3, 5
    embed_dim, num_heads, ffn_dim = 8, 2, 4
    dropout_prob, layer_norm_eps = 0.1, 1e-5

    my_module = block.EncoderBlock(
        embed_dim,
        num_heads,
        ffn_dim,
        dropout_prob=dropout_prob,
        layer_norm_eps=layer_norm_eps,
    )
    gt_module = nn.TransformerEncoderLayer(
        embed_dim,
        num_heads,
        ffn_dim,
        dropout=dropout_prob,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
    )
    self._match_params(my_module, gt_module)

    my_module.eval()
    gt_module.eval()
    
    x = torch.rand((bs, seq_len, embed_dim))
    my_out = my_module(x)
    gt_out = gt_module(x)
    torch.testing.assert_close(my_out, gt_out, atol=ATOL, rtol=0)

  def test_decoder_block(self):
    bs, seq_len = 3, 5
    embed_dim, num_heads, ffn_dim = 8, 2, 4
    dropout_prob, layer_norm_eps = 0.1, 1e-5
    
    my_module = block.DecoderBlock(
        embed_dim,
        num_heads,
        ffn_dim,
        dropout_prob=dropout_prob,
        layer_norm_eps=layer_norm_eps,
    )
    gt_module = nn.TransformerDecoderLayer(
        embed_dim,
        num_heads,
        ffn_dim,
        dropout=dropout_prob,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
    )
    self._match_params(my_module, gt_module, match_mhca=True)
    
    my_module.eval()
    gt_module.eval()
    
    x = torch.rand((bs, seq_len, embed_dim))
    encoder_output = torch.rand((bs, seq_len, embed_dim))
    attn_mask = torch.triu(
      torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
    )
    my_out = my_module(x, encoder_output, attn_mask=attn_mask)
    gt_out = gt_module(x, encoder_output, tgt_mask=attn_mask)
    torch.testing.assert_close(my_out, gt_out, atol=ATOL, rtol=0)

if __name__ == '__main__':
  unittest.main()
