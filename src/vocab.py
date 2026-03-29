from collections import Counter

class Vocab:
    def __init__(self, min_freq=2):
        self.word2idx = {"<PAD>":0, "<UNK>":1}
        self.idx2word = {0:"<PAD>", 1:"<UNK>"}
        self.min_freq = min_freq

    def build(self, sentences):
        counter = Counter()
        for sent in sentences:
            counter.update(sent)

        for word, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence):
        return [self.word2idx.get(w, 1) for w in sentence]

    def decode(self, indices):
        return [self.idx2word[i] for i in indices]

    def __len__(self):
        return len(self.word2idx)