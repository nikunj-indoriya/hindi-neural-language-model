import torch
from torch.utils.data import Dataset

class LanguageDataset(Dataset):
    def __init__(self, sentences, vocab, seq_len=20):
        self.data = []
        self.seq_len = seq_len
        self.vocab = vocab

        for sent in sentences:
            encoded = vocab.encode(sent)
            for i in range(1, len(encoded)):
                x = encoded[:i]
                y = encoded[1:i+1]

                x = x[-seq_len:]
                y = y[-seq_len:]

                self.data.append((x, y))

    def pad(self, seq):
        return seq + [0]*(self.seq_len - len(seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(self.pad(x)), torch.tensor(self.pad(y))