
"""Language translation with custom Transformer implementation.

This implementation is more faithful to the original Transformer paper [1] than
the PyTorch transformer tutorial [2]:
1. It uses byte-pair encoding.
2. It uses a shared source-target vocabulary.
3. It shares the same weight matrix between the two embedding layers and the
   pre-softmax linear layer.
4. It uses a batch-first layout.
5. No longer using the deprecated torchdata for constructing datasets.

To prepare the datasets, check out the PyTorch translation tutorial [2].

Note that these features are still WIP.

References:
[1] https://arxiv.org/pdf/1706.03762, retrieved on Aug 15, 2024.
[2] https://pytorch.org/tutorials/beginner/translation_transformer.html,
  retrieved on Aug 15, 2024.
"""

from collections.abc import Iterable, Iterator, Callable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k

from modules.model import Transformer
from modules.pos_embed import PositionalEncoding
from modules.token_embed import TokenEmbedding


SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
  
  
def preprocess() -> tuple[int, int, Callable]:
  token_transform = {
    SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),
    TGT_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm'),
  }
  vocab_transform = {}

  def yield_tokens(data_iter: Iterable, language: str) -> Iterator[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
      yield token_transform[language](data_sample[language_index[language]])

  for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                            min_freq=1,
                            specials=special_symbols,
                            special_first=True)
    vocab_transform[ln].set_default_index(UNK_IDX)

  def sequential_transforms(*transforms):
    def func(txt_input):
      for transform in transforms:
        txt_input = transform(txt_input)
      return txt_input
    return func

  # function to add BOS/EOS and create tensor for input sequence indices
  def tensor_transform(token_ids: list[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
              torch.tensor(token_ids),
              torch.tensor([EOS_IDX])))

  # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
  text_transform = {}
  for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                          vocab_transform[ln], #Numericalization
                          tensor_transform) # Add BOS/EOS and create tensor


  # function to collate data samples into batch tensors
  def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
      src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
      tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

  return (
    len(vocab_transform[SRC_LANGUAGE]),
    len(vocab_transform[TGT_LANGUAGE]),
    collate_fn,
  )
    
def generate_square_subsequent_mask(sz):
  mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask


def create_mask(src, tgt):
  src_seq_len = src.shape[1]
  tgt_seq_len = tgt.shape[1]

  tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
  src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

  src_padding_mask = (src == PAD_IDX).transpose(0, 1)
  tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
  return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


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
      
  def _create_mask(self, x: torch.Tensor, is_causal=False):
    seq_len = x.shape[1]
    mask = (x == PAD_IDX)[:, None, None]
    if is_causal:
      mask = mask | torch.triu(
          torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
          diagonal=1,
      )[None, None, ...]
    return mask

  def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    src_padding_mask = self._create_mask(src)
    tgt_sa_mask = self._create_mask(tgt, is_causal=True)
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

  def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
    return self.transformer.encoder(self.positional_encoding(
              self.src_tok_emb(src)), src_mask)

  def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
    return self.transformer.decoder(self.positional_encoding(
              self.tgt_tok_emb(tgt)), memory,
              tgt_mask)
  