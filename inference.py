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
  src = src.mT
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


def translate(
  text_transform,
  vocab_transform,
  model: nn.Module,
  src_sentence: str,
  device: torch.DeviceObjType,
) -> str:
  src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
  tgt_tokens = greedy_decode(model, src, device).flatten()
  return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")



if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  text_transform, vocab_transform, collate_fn = translation.preprocess()
  model = translation.Seq2SeqTransformer(
    max_seq_len=train.MAX_SEQ_LEN,
    src_vocab_size=len(vocab_transform[SRC_LANGUAGE]),
    tgt_vocab_size=len(vocab_transform[TGT_LANGUAGE]),
    num_encoder_blocks=train.NUM_ENCODER_BLOCKS,
    num_decoder_blocks=train.NUM_DECODER_BLOCKS,
    embed_dim=train.EMBED_DIM,
    num_heads=train.NUM_HEADS,
    ffn_dim=train.FFN_DIM,
  ).to(device)
  
  model.load_state_dict(torch.load('seq2seq.pth'))
  model.eval()
  
  print('Please input source language text:')
  while True:
    text = input()
    output = translate(text_transform, vocab_transform, model, text, device)
    print(output)
