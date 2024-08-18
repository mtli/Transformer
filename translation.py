
"""Language translation with custom Transformer implementation.

This implementation is more faithful to the original Transformer paper [1] than
the PyTorch transformer tutorial [2]:
1. It uses byte-pair encoding.
2. It uses a shared source-target vocabulary.
3. It shares the same weight matrix between the two embedding layers and the
   pre-softmax linear layer.
4. It uses the learning rate schedule described in the paper.
5. It uses a batch-first layout.
6. No longer using the deprecated torchdata for constructing datasets.

To prepare the datasets, check out the PyTorch translation tutorial [2].

Note that these features are still WIP.

References:
[1] https://arxiv.org/pdf/1706.03762, retrieved on Aug 15, 2024.
[2] https://pytorch.org/tutorials/beginner/translation_transformer.html,
  retrieved on Aug 15, 2024.
"""
import torch
import torch.nn as nn
from torchtext.datasets import Multi30k
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from modules.model import Transformer
from modules.pos_embed import PositionalEncoding
from modules.token_embed import TokenEmbedding

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
VOCAB_SIZE = 3000

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
  
  
def create_tokenizer() -> ByteLevelBPETokenizer:
  tokenizer = ByteLevelBPETokenizer()

  train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  src_sentences = [sample[0] for sample in train_iter]
  tgt_sentences = [sample[1] for sample in train_iter]
  tokenizer.train_from_iterator(
      src_sentences + tgt_sentences,
      vocab_size=VOCAB_SIZE,
      min_frequency=1,
      special_tokens=special_symbols,
  )
  tokenizer.post_processor = BertProcessing(("<bos>", BOS_IDX), ("<eos>", EOS_IDX))
  
  def encode_text(text):
      return tokenizer.encode(text).ids

  def decode_tokens(tokens, tokenizer):
      return tokenizer.decode(tokens.tolist(), skip_special_tokens=True)

  return tokenizer


class Seq2SeqTransformer(nn.Module):
  def __init__(
      self,
      max_seq_len: int,
      src_vocab_size: int,
      tgt_vocab_size: int,
      num_encoder_blocks: int,
      num_decoder_blocks: int,
      embed_dim: int,
      num_heads: int,
      ffn_dim: int,
      dropout_prob: float = 0.1,
  ):
    super().__init__()
    self.transformer = Transformer(
        num_encoder_blocks,
        num_decoder_blocks,
        embed_dim,
        num_heads,
        ffn_dim,
        dropout_prob,
    )
    self.generator = nn.Linear(embed_dim, tgt_vocab_size)
    self.src_tok_emb = TokenEmbedding(src_vocab_size, embed_dim)
    self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, embed_dim)
    self.positional_encoding = PositionalEncoding(
        max_seq_len, embed_dim, dropout_prob=dropout_prob
    )
    self._reset_parameters()

  def _reset_parameters(self):
      for p in self.parameters():
        if p.dim() > 1:
          # Init the weights (not the bias) in all the linear layers.
          nn.init.xavier_uniform_(p)
      
  def create_mask(self, x: torch.Tensor, is_causal=False):
    seq_len = x.shape[1]
    mask = (x == PAD_IDX)[:, None, None]
    if is_causal:
      mask = mask | torch.triu(
          torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
          diagonal=1,
      )[None, None, ...]
    return mask

  def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    src_padding_mask = self.create_mask(src)
    tgt_sa_mask = self.create_mask(tgt, is_causal=True)
    src_emb = self.positional_encoding(self.src_tok_emb(src))
    tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
    outs = self.transformer(
      src_emb,
        tgt_emb,
        src_mask=src_padding_mask,
        tgt_sa_mask=tgt_sa_mask,
        tgt_ca_mask=src_padding_mask,
    )
    return self.generator(outs)

  def encode(
      self,
      src: torch.Tensor,
      src_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    src_emb = self.positional_encoding(self.src_tok_emb(src))
    return self.transformer.encoder(src_emb, src_mask)

  def decode(
      self,
      tgt: torch.Tensor,
      memory: torch.Tensor,
      tgt_sa_mask: torch.Tensor | None = None,
      tgt_ca_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
    return self.transformer.decoder(tgt_emb, memory, tgt_sa_mask, tgt_ca_mask)
