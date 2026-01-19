import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import random
import math
import time
from transformers import AutoModel, AutoTokenizer

# Replace 'Helsinki-NLP/opus-mt-en-fr' with the model you want to download
model_name = "melaniab/spacy-pipeline-bg"

# Download the model
model = AutoModel.from_pretrained(model_name)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Load spaCy tokenizers
spacy_bg =spacy.load("spacy-pipeline-bg")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_bg(text):
    """
    Tokenizes Bulgarian text from a string into a list of strings
    """
    return [tok.text for tok in spacy_bg.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Define the tokenizers
SRC_TOKENIZER = tokenize_bg
TRG_TOKENIZER = tokenize_en

# Custom dataset class for parallel text
class TranslationDataset(Dataset):
    def __init__(self, src_file, trg_file, src_tokenizer, trg_tokenizer):
        self.src_file = src_file
        self.trg_file = trg_file
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = f.readlines()

        with open(trg_file, 'r', encoding='utf-8') as f:
            self.trg_sentences = f.readlines()

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx].strip()
        trg_sentence = self.trg_sentences[idx].strip()
        src_tokens = self.src_tokenizer(src_sentence)
        trg_tokens = self.trg_tokenizer(trg_sentence)
        return src_tokens, trg_tokens

# Load datasets
train_data = TranslationDataset('train.bg', 'train.en', SRC_TOKENIZER, TRG_TOKENIZER)
valid_data = TranslationDataset('dev.bg', 'dev.en', SRC_TOKENIZER, TRG_TOKENIZER)
test_data = TranslationDataset('test.bg', 'test.en', SRC_TOKENIZER, TRG_TOKENIZER)

def yield_tokens(data_iter, tokenizer):
    for src, trg in data_iter:
        yield tokenizer(' '.join(src))
        yield tokenizer(' '.join(trg))

# Build vocabularies
SRC_VOCAB = build_vocab_from_iterator(yield_tokens(train_data, SRC_TOKENIZER),
                                      specials=["<unk>", "<pad>", "<sos>", "<eos>"])
TRG_VOCAB = build_vocab_from_iterator(yield_tokens(train_data, TRG_TOKENIZER),
                                      specials=["<unk>", "<pad>", "<sos>", "<eos>"])

SRC_VOCAB.set_default_index(SRC_VOCAB["<unk>"])
TRG_VOCAB.set_default_index(TRG_VOCAB["<unk>"])

# Define the collate function for the DataLoader
def collate_batch(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = [torch.tensor(SRC_VOCAB(token)) for token in src_batch]
    trg_batch = [torch.tensor(TRG_VOCAB(token)) for token in trg_batch]

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=SRC_VOCAB["<pad>"])
    trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=TRG_VOCAB["<pad>"])

    return src_batch, trg_batch

# Create DataLoaders
BATCH_SIZE = 128

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
