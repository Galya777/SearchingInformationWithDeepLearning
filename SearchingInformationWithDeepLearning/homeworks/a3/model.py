#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import torch

#################################################################
####  LSTM с пакетиране на партида
#################################################################

class LSTMLanguageModelPack(torch.nn.Module):
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for (a,s) in source)
        sents = [[self.word2ind.get(w,self.unkTokenIdx) for w in s] for (a,s) in source]
        auths = [self.auth2id.get(a,0) for (a,s) in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device)), torch.tensor(auths, dtype=torch.long, device=device)
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName,device):
        self.load_state_dict(torch.load(fileName,device))

    def __init__(self, embed_size, hidden_size, auth2id, word2ind, unkToken, padToken, endToken, lstm_layers, dropout):
        super(LSTMLanguageModelPack, self).__init__()
        #############################################################################
        ###  Тук следва да се имплементира инициализацията на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавки за повече слоеве на РНН, влагане за автора и dropout
        #############################################################################
        #### Начало на Вашия код.
        # store vocab/author maps and special token indices
        self.word2ind = word2ind
        self.auth2id = auth2id
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]

        vocab_size = len(word2ind)
        num_auth = max(auth2id.values()) + 1 if len(auth2id) > 0 else 1

        # embeddings
        self.char_emb = torch.nn.Embedding(vocab_size, embed_size, padding_idx=self.padTokenIdx)
        # author embedding (same size as char embedding)
        self.auth_emb = torch.nn.Embedding(num_auth, embed_size)

        # LSTM: input is char_emb + auth_emb (concatenated)
        self.input_size = embed_size * 2
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout_p = dropout
        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(hidden_size, vocab_size)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.padTokenIdx, reduction='sum')
            
        #### Край на Вашия код
        #############################################################################

    def forward(self, source):
        #############################################################################
        ###  Тук следва да се имплементира forward метода на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавка за dropout и началните скрити вектори
        #############################################################################
        #### Начало на Вашия код.
        # source: списък от елементи (author, [chars])
        # Подгответе пакет с padding
        x, auth = self.preparePaddedBatch(source)  # x: [seq_len, batch]

        # Ембединг за символи
        emb_c = self.char_emb(x)  # [seq_len, batch, emb]

        # Ембединг за автора – добавяме като контекст на всяка стъпка
        emb_a = self.auth_emb(auth)  # [batch, emb]
        seq_len, batch_size = x.size(0), x.size(1)
        emb_a_rep = emb_a.unsqueeze(0).expand(seq_len, batch_size, -1)  # [seq_len,batch,emb]

        # Конкатенация на входа: [seq_len, batch, 2*emb]
        emb = torch.cat([emb_c, emb_a_rep], dim=2)

        # Начални скрити състояния нули
        device = emb.device
        h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device)

        # LSTM през последователността (не пакетираме; ще игнорираме PAD в loss)
        y, _ = self.lstm(emb, (h0, c0))  # [seq_len, batch, hidden]
        y = self.dropout(y)

        # Предсказваме следващия символ: изместваме с 1
        logits = self.out(y[:-1])  # [seq_len-1, batch, vocab]
        targets = x[1:]            # [seq_len-1, batch]

        # Преоформяне за loss: [T*B, V] и [T*B]
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        # Изчисляваме сумарна загуба и нормализираме по броя валидни цели (без PAD)
        loss_sum = self.loss_fn(logits, targets)
        valid = (targets != self.padTokenIdx).sum().clamp_min(1)
        loss = loss_sum / valid

        return loss
    
        #### Край на Вашия код
        #############################################################################

