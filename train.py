from functools import partial
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k
import translation


MAX_SEQ_LEN = 1000
EMBED_DIM = 512
NUM_HEADS = 8
FFN_DIM = 2048
NUM_ENCODER_BLOCKS = 6
NUM_DECODER_BLOCKS = 6
BATCH_SIZE = 256
NUM_EPOCHS = 18
WARMUP_STEPS = 100
NUM_WORKERS = 0

SRC_LANGUAGE = translation.SRC_LANGUAGE
TGT_LANGUAGE = translation.TGT_LANGUAGE
VOCAB_SIZE = translation.VOCAB_SIZE
PAD_IDX = translation.PAD_IDX

    
def collate_fn(batch, tokenizer, padding_idx):
    src, tgt = [], []
    def _process(text):
      return torch.tensor(tokenizer.encode(text).ids)
        
    for src_sample, tgt_sample in batch:
        src.append(_process(src_sample))
        tgt.append(_process(tgt_sample))

    src = pad_sequence(src, padding_value=padding_idx, batch_first=True)
    tgt = pad_sequence(tgt, padding_value=padding_idx, batch_first=True)
    return src, tgt

def train(device: torch.DeviceObjType):  
  train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  tokenizer = translation.create_tokenizer()
  collate_fn_bind = partial(collate_fn, tokenizer=tokenizer, padding_idx=PAD_IDX)
  train_dataloader = DataLoader(
      train_iter,
      batch_size=BATCH_SIZE,
      collate_fn=collate_fn_bind,
      num_workers=NUM_WORKERS,
  )
  val_dataloader = DataLoader(
      val_iter,
      batch_size=BATCH_SIZE,
      collate_fn=collate_fn_bind,
      num_workers=NUM_WORKERS,
  )

  model = translation.Seq2SeqTransformer(
    max_seq_len=MAX_SEQ_LEN,
    src_vocab_size=VOCAB_SIZE,
    tgt_vocab_size=VOCAB_SIZE,
    num_encoder_blocks=NUM_ENCODER_BLOCKS,
    num_decoder_blocks=NUM_DECODER_BLOCKS,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM,
  ).to(device)
  
  loss_fn = nn.CrossEntropyLoss(ignore_index=translation.PAD_IDX)
  base_lr = EMBED_DIM ** (-0.5)
  optimizer = optim.Adam(
    model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9)
  scheduler = lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=lambda i: min((i + 1)**(-0.5), (i + 1)*WARMUP_STEPS**(-1.5))
  )
  
  def train_epoch():
    cum_loss = 0
    num_batch = 0

    model.train()
    for src, tgt in train_dataloader:
      src, tgt = src.to(device), tgt.to(device)
      logits = model(src, tgt[:, :-1])
      optimizer.zero_grad()
      loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].flatten())
      loss.backward()
      optimizer.step()
      scheduler.step()
      cum_loss += loss.item()
      num_batch += 1

    return cum_loss / num_batch


  def evaluate():
    cum_loss = 0
    num_batch = 0

    model.eval()
    for src, tgt in val_dataloader:
      src, tgt = src.to(device), tgt.to(device)
      logits = model(src, tgt[:, :-1])
      loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].flatten())
      cum_loss += loss.item()
      num_batch += 1
      
    return cum_loss / num_batch

  for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.perf_counter()
    train_loss = train_epoch()
    elapsed_time = time.perf_counter() - start_time
    val_loss = evaluate()
    print(
        f'Epoch: {epoch}, train loss: {train_loss:.3f}, '
        f'val loss: {val_loss:.3f}, time = {elapsed_time:.3f}s'
    )

  torch.save(model.state_dict(), 'seq2seq.pth')

if __name__ == '__main__':
  train(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
