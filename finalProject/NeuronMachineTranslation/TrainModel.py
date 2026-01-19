import torch
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
import random
import math
import time
from NeuronMachineTranslation.dataPreparation import SRC_VOCAB, TRG_VOCAB, collate_batch, train_data, valid_data
from NeuronMachineTranslation.model import Encoder, Decoder, Seq2Seq, train, evaluate

# Hyperparameters
INPUT_DIM = len(SRC_VOCAB)
OUTPUT_DIM = len(TRG_VOCAB)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 10
CLIP = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)

model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=TRG_VOCAB["<pad>"])

train_iterator = DataLoader(train_data, batch_size=32, collate_fn=collate_batch, shuffle=True)
valid_iterator = DataLoader(valid_data, batch_size=32, collate_fn=collate_batch)

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
