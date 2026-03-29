import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import random

from src.vocab import Vocab
from src.dataset import LanguageDataset
from src.train import train
from src.evaluate import evaluate
from src.generate import generate

from models.rnn import RNNLM
from models.lstm import LSTMLM
from models.gru import GRULM
from models.bilstm import BiLSTMLM
from models.attention_lstm import AttentionLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= DATA =================
dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    "indiccorp_v2",
    split="guj_Gujr"
)

dataset = dataset.select(range(50000))

texts = [x["text"] for x in dataset]

sentences = [t.split() for t in texts if len(t.split()) >= 5]

random.shuffle(sentences)

train_s = sentences[:40000]
val_s = sentences[40000:45000]
test_s = sentences[45000:50000]

# ================= VOCAB =================
vocab = Vocab()
vocab.build(train_s)

# ================= DATASET =================
train_ds = LanguageDataset(train_s, vocab)
val_ds = LanguageDataset(val_s, vocab)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# ================= MODEL =================
model = LSTMLM(len(vocab), 256, 512).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ================= TRAIN =================
for epoch in range(10):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss, acc, ppl = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | PPL: {ppl:.2f}")

# ================= GENERATION =================
print(generate(model, vocab, "હું આજે", device=device))