import time
import torch
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
import translation


MAX_SEQ_LEN = 10000
EMBED_DIM = 512
NUM_HEADS = 8
FFN_DIM = 512
NUM_ENCODER_BLOCKS = 3
NUM_DECODER_BLOCKS = 3
BATCH_SIZE = 128
NUM_EPOCHS = 18

SRC_LANGUAGE = translation.SRC_LANGUAGE
TGT_LANGUAGE = translation.TGT_LANGUAGE

def train_epoch(train_dataloader, model, loss_fn, optimizer, device):
  cum_loss = 0
  num_batch = 0

  model.train()
  for src, tgt in train_dataloader:
    assert src.dim() == 2
    src = src.mT
    tgt = tgt.mT
    src = src.to(device)
    tgt = tgt.to(device)

    logits = model(src, tgt[:, :-1])

    optimizer.zero_grad()

    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].flatten())
    loss.backward()

    optimizer.step()
    cum_loss += loss.item()
    num_batch += 1

  return cum_loss / num_batch


def evaluate(val_dataloader, model, loss_fn, device):
  cum_loss = 0
  num_batch = 0

  model.eval()
  for src, tgt in val_dataloader:
    assert src.dim() == 2
    src = src.mT
    tgt = tgt.mT
    src = src.to(device)
    tgt = tgt.to(device)

    logits = model(src, tgt[:, :-1])
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].flatten())
    cum_loss += loss.item()
    num_batch += 1
    
  return cum_loss / num_batch


def train(device: torch.DeviceObjType):  
  train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  _, vocab_transform, collate_fn = translation.preprocess()
  train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
  val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

  model = translation.Seq2SeqTransformer(
    max_seq_len=MAX_SEQ_LEN,
    src_vocab_size=len(vocab_transform[SRC_LANGUAGE]),
    tgt_vocab_size=len(vocab_transform[TGT_LANGUAGE]),
    num_encoder_blocks=NUM_ENCODER_BLOCKS,
    num_decoder_blocks=NUM_DECODER_BLOCKS,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM,
  ).to(device)
  
  loss_fn = torch.nn.CrossEntropyLoss(ignore_index=translation.PAD_IDX)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
  
  for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.perf_counter()
    train_loss = train_epoch(train_dataloader, model, loss_fn, optimizer, device)
    elapsed_time = time.perf_counter() - start_time
    val_loss = evaluate(val_dataloader, model, loss_fn, device)
    print(
        f'Epoch: {epoch}, train loss: {train_loss:.3f}, '
        f'val loss: {val_loss:.3f}, time = {elapsed_time:.3f}s'
    )

  torch.save(model.state_dict(), 'seq2seq.pth')

if __name__ == '__main__':
  train(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))