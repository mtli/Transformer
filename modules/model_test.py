"""Transformer module unit tests.

Launch command: python -m unittest modules.model_test
"""

import unittest
import torch
import torch.nn as nn
from . import model

ATOL = 1e-6


class TestModel(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(0)
    
  def _match_params(
      self,
      my_module: model.TransformerEncoder | model.TransformerDecoder,
      gt_module: nn.TransformerEncoder | nn.TransformerDecoder,
      match_mhca: bool = False,
  ):
    for my_block, gt_block in zip(my_module.blocks, gt_module.layers):
      my_mhsa = my_block.self_attn
      gt_block.self_attn.in_proj_weight = nn.Parameter(
          torch.cat([
              my_mhsa.to_q.weight, my_mhsa.to_k.weight, my_mhsa.to_v.weight,
          ], dim=0)
      )
      gt_block.self_attn.out_proj.weight = my_mhsa.to_o.weight
      gt_block.self_attn.out_proj.bias = my_mhsa.to_o.bias
      
      if match_mhca:
        my_mhca = my_block.cross_attn
        gt_block.multihead_attn.in_proj_weight = nn.Parameter(
            torch.cat([
                my_mhca.to_q.weight, my_mhca.to_k.weight, my_mhca.to_v.weight
            ], dim=0)
        )
        gt_block.multihead_attn.out_proj.weight = my_mhca.to_o.weight
        gt_block.multihead_attn.out_proj.bias = my_mhca.to_o.bias
        
      gt_block.linear1.weight = my_block.ffn[0].weight
      gt_block.linear1.bias = my_block.ffn[0].bias
      gt_block.linear2.weight = my_block.ffn[3].weight
      gt_block.linear2.bias = my_block.ffn[3].bias

  def test_encoder(self):
    bs, src_len = 3, 5
    num_blocks = 2
    embed_dim, num_heads, ffn_dim = 8, 2, 4
    dropout_prob, layer_norm_eps = 0.1, 1e-5
    
    my_module = model.TransformerEncoder(
        num_blocks,
        embed_dim, num_heads, ffn_dim,
        dropout_prob=dropout_prob,
        layer_norm_eps=layer_norm_eps,
    )
    gt_layer = nn.TransformerEncoderLayer(
        embed_dim, num_heads, ffn_dim,
        dropout=dropout_prob,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
    )
    gt_module = nn.TransformerEncoder(gt_layer, num_blocks)
    
    self._match_params(my_module, gt_module)
    my_module.eval()
    gt_module.eval()

    src = torch.rand(bs, src_len, embed_dim)
    my_out = my_module(src)
    gt_out = gt_module(src)
    torch.testing.assert_close(my_out, gt_out, atol=ATOL, rtol=0)

  def test_decoder(self):
    bs, src_len, tgt_len = 3, 5, 7
    num_blocks = 2
    embed_dim, num_heads, ffn_dim = 8, 2, 4
    dropout_prob, layer_norm_eps = 0.1, 1e-5
    
    my_module = model.TransformerDecoder(
        num_blocks,
        embed_dim, num_heads, ffn_dim,
        dropout_prob=dropout_prob,
        layer_norm_eps=layer_norm_eps,
    )
    gt_layer = nn.TransformerDecoderLayer(
        embed_dim, num_heads, ffn_dim,
        dropout=dropout_prob,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
    )
    gt_module = nn.TransformerDecoder(gt_layer, num_blocks)
    
    self._match_params(my_module, gt_module, match_mhca=True)
  
    my_module.eval()
    gt_module.eval()

    memory = torch.rand(bs, src_len, embed_dim)
    tgt = torch.rand(bs, tgt_len, embed_dim)
    my_out = my_module(tgt, memory)
    gt_out = gt_module(tgt, memory)
    torch.testing.assert_close(my_out, gt_out, atol=ATOL, rtol=0)

  def test_transformer(self):
    bs, src_len, tgt_len = 3, 5, 7
    num_encoder_blocks, num_decoder_blocks = 2, 2
    embed_dim, num_heads, ffn_dim = 8, 2, 4
    dropout_prob, layer_norm_eps = 0.1, 1e-5

    my_module = model.Transformer(
        num_encoder_blocks,
        num_decoder_blocks,
        embed_dim,
        num_heads,
        ffn_dim,
        dropout_prob=dropout_prob,
        layer_norm_eps=layer_norm_eps,
    )
    # Note that due to an issue in official PyTorch implementation
    # (https://github.com/pytorch/pytorch/issues/24930), we need to use custom
    # encoder and decoder interface to avoid the redudant normalization layer
    # at the end of the encoder and the decoder.
    gt_encoder_layer = nn.TransformerEncoderLayer(
        embed_dim,
        num_heads,
        ffn_dim,
        dropout=dropout_prob,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
    )
    gt_encoder = nn.TransformerEncoder(gt_encoder_layer, num_encoder_blocks)
    gt_decoder_layer = nn.TransformerDecoderLayer(
        embed_dim,
        num_heads,
        ffn_dim,
        dropout=dropout_prob,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
    )
    gt_decoder = nn.TransformerDecoder(gt_decoder_layer, num_encoder_blocks)
    gt_module = nn.Transformer(
        d_model=embed_dim,
        nhead=num_heads,
        num_encoder_layers=num_encoder_blocks,
        num_decoder_layers=num_decoder_blocks,
        batch_first=True,
        custom_encoder=gt_encoder,
        custom_decoder=gt_decoder,
    )
    
    self._match_params(my_module.encoder, gt_module.encoder)
    self._match_params(my_module.decoder, gt_module.decoder, match_mhca=True)
    
    my_module.eval()
    gt_module.eval()

    src = torch.rand(bs, src_len, embed_dim)
    tgt = torch.rand(bs, tgt_len, embed_dim)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool), diagonal=1
    )
    
    with self.subTest('without_a_mask'):
      my_out = my_module(src, tgt)
      gt_out = gt_module(src, tgt)
      torch.testing.assert_close(my_out, gt_out, atol=ATOL, rtol=0)
    
    with self.subTest('with_a_casual_mask'):
      my_out = my_module(src, tgt, tgt_sa_mask=causal_mask)
      gt_out = gt_module(src, tgt, tgt_mask=causal_mask)
      torch.testing.assert_close(my_out, gt_out, atol=ATOL, rtol=0)
      
    with self.subTest('with_causal_and_padding_masks'):
      src_padding_mask = torch.rand(bs, src_len) > 0.5
      tgt_padding_mask = torch.rand(bs, tgt_len) > 0.5
      # At least one position is not masked out.
      src_padding_mask[:, 0] = False
      tgt_padding_mask[:, 0] = False
      # Merge with the casual mask and expand out the head dimension.
      sa_mask = (tgt_padding_mask[:, None] | causal_mask).unsqueeze(1)
      ca_mask = src_padding_mask[:, None, None]
      
      my_out = my_module(
          src,
          tgt,
          src_mask=ca_mask,
          tgt_sa_mask=sa_mask,
          tgt_ca_mask=ca_mask,
      )
      gt_out = gt_module(
        src,
        tgt,
        src_key_padding_mask=src_padding_mask,
        memory_key_padding_mask=src_padding_mask,
        tgt_mask=causal_mask,
        tgt_is_causal=True,
        tgt_key_padding_mask=tgt_padding_mask,
      )
      torch.testing.assert_close(my_out, gt_out, atol=ATOL, rtol=0)
      

if __name__ == '__main__':
  unittest.main()
