import torch
import torch.nn as nn

class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)

        scores = torch.tanh(self.attn(out))
        weights = torch.softmax(self.context(scores), dim=1)

        context = (weights * out).sum(dim=1)
        context = context.unsqueeze(1).repeat(1, out.size(1), 1)

        out = out + context
        return self.fc(out)