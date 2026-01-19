#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import numpy as np
import torch

def generateText(model, char2id, auth, startSentence, limit=1000, temperature=1.):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ
    
    result = startSentence[1:]

    #############################################################################
    ###  Тук следва да се имплементира генерацията на текста
    #############################################################################
    #### Начало на Вашия код.
    device = next(model.parameters()).device
    id2char = {v: k for k, v in char2id.items()}

    # Началната последователност (индекси)
    seq = [char2id.get(c, model.unkTokenIdx) for c in startSentence]
    auth_id = model.auth2id.get(auth, 0)

    model.eval()
    with torch.no_grad():
        while len(result) < limit:
            x = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(1)  # [seq_len,1]
            # Ембединг за символи
            emb_c = model.char_emb(x)  # [seq_len,1,emb]
            # Авторски ембединг, повтаряме по дължината
            emb_a = model.auth_emb(torch.tensor([auth_id], dtype=torch.long, device=device))  # [1,emb]
            emb_a_rep = emb_a.unsqueeze(0).expand(emb_c.size(0), 1, -1)
            emb = torch.cat([emb_c, emb_a_rep], dim=2)  # [seq_len,1,2*emb]

            # Начални нули за скритите състояния
            h0 = torch.zeros(model.lstm_layers, 1, model.hidden_size, device=device)
            c0 = torch.zeros(model.lstm_layers, 1, model.hidden_size, device=device)
            y, _ = model.lstm(emb, (h0, c0))  # [seq_len,1,hidden]
            y_last = model.dropout(y[-1])      # [1,hidden]
            logits = model.out(y_last).squeeze(0)  # [vocab]

            # Температура
            if temperature is None or temperature <= 0:
                temperature = 1.0
            probs = torch.softmax(logits / float(temperature), dim=-1)

            # Избираме следващ символ
            next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == model.endTokenIdx:
                break
            # Добавяме към резултата (прескачаме символа за начало '{')
            result += id2char.get(next_id, '')
            seq.append(next_id)
	
    #### Край на Вашия код
    #############################################################################

    return result
