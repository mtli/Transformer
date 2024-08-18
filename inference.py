import torch
import torch.nn as nn
import train
import translation


MAX_OUTPUT_LEN = 1000

SRC_LANGUAGE = translation.SRC_LANGUAGE
TGT_LANGUAGE = translation.TGT_LANGUAGE
BOS_IDX = translation.BOS_IDX
EOS_IDX = translation.EOS_IDX


def greedy_decode(model: translation.Seq2SeqTransformer, src, device):
  src = src.to(device)
  memory = model.encode(src)
  tgt = torch.tensor(BOS_IDX, device=device)[None, None]
  for i in range(MAX_OUTPUT_LEN):
    tgt_sa_mask = model.create_mask(tgt, is_causal=True)
    out = model.decode(tgt, memory, tgt_sa_mask)
    logit = model.generator(out[:, [-1]])
    next_word_idx = logit[0, 0].argmax(dim=0)
    tgt = torch.cat((tgt, next_word_idx[None, None]), dim=1)
    if next_word_idx == EOS_IDX:
      break
  return tgt


def encode_text(text, tokenizer):
  return tokenizer.encode(text).ids
    
def decode_tokens(tokens, tokenizer):
  return tokenizer.decode(tokens.tolist(), skip_special_tokens=True)


def translate(
    tokenizer,
    model: nn.Module,
    src_sentence: str,
    device: torch.DeviceObjType,
) -> str:
  src = torch.tensor(encode_text(src_sentence, tokenizer), dtype=torch.long).unsqueeze(0)
  tgt_tokens = greedy_decode(model, src, device).flatten()
  return decode_tokens(tgt_tokens.cpu(), tokenizer)


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenizer = translation.create_tokenizer()
  model = translation.Seq2SeqTransformer(
    max_seq_len=train.MAX_SEQ_LEN,
    src_vocab_size=translation.VOCAB_SIZE,
    tgt_vocab_size=translation.VOCAB_SIZE,
    num_encoder_blocks=train.NUM_ENCODER_BLOCKS,
    num_decoder_blocks=train.NUM_DECODER_BLOCKS,
    embed_dim=train.EMBED_DIM,
    num_heads=train.NUM_HEADS,
    ffn_dim=train.FFN_DIM,
  ).to(device)
  
  model.load_state_dict(torch.load('seq2seq.pth'))
  model.eval()
  
  while True:
    print('Please input source language text:')
    text = input()
    output = translate(tokenizer, model, text, device)
    print(output)
