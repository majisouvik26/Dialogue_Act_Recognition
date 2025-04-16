# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd

class DialogueActDataset(Dataset):
    def __init__(self, csv_file, vocab=None, max_seq_length=128):
        self.data = pd.read_csv(csv_file)
        self.max_seq_length = max_seq_length
        if vocab is None:
            self.vocab = self.build_vocab(self.data['text'])
        else:
            self.vocab = vocab
        self.labels = sorted(self.data['label'].unique())
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        
    def build_vocab(self, texts):
        vocab = {"[PAD]": 0, "[UNK]": 1}
        idx = 2
        for text in texts:
            for token in text.strip().split():
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        return vocab
    
    def tokenize(self, text):
        tokens = text.strip().split()
        token_ids = [self.vocab.get(tok, self.vocab["[UNK]"]) for tok in tokens]
        if len(token_ids) < self.max_seq_length:
            token_ids += [self.vocab["[PAD]"]] * (self.max_seq_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_seq_length]
        return torch.tensor(token_ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        token_ids = self.tokenize(text)
        label_id = self.label2id[label]
        return token_ids, label_id

