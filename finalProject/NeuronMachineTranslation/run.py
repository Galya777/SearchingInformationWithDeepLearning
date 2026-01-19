#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2023/2024
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import sys
import numpy as np
import torch
import math
import pickle
import time

from nltk.translate.bleu_score import corpus_bleu

import model
from NeuronMachineTranslation.TrainModel import device
from NeuronMachineTranslation.dataPreparation import tokenize_bg, SRC_VOCAB, TRG_VOCAB



def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    tokens = tokenize_bg(sentence)
    tokens = [src_vocab["<sos>"]] + [src_vocab[token] for token in tokens] + [src_vocab["<eos>"]]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [trg_vocab["<sos>"]]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab["<eos>"]:
            break

    trg_tokens = [trg_vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]

example_sentence = "Това е тест."
translation = translate_sentence(example_sentence, SRC_VOCAB, TRG_VOCAB, model, device)

print(f'Predicted translation: {" ".join(translation)}')
