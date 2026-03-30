import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import random
import re

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
dataset = load_dataset("cfilt/iitb-english-hindi", split="train")

print("Sample:", dataset[0])

texts = [x["translation"]["hi"] for x in dataset]

# ================= CLEANING =================
def clean_text(text):
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

texts = [clean_text(t) for t in texts]

sentences = [t.split() for t in texts if len(t.split()) >= 5]
sentences = sentences[:50000]

print("Total usable sentences:", len(sentences))
print("Sample sentences:", sentences[:3])

# ================= SPLIT =================
random.shuffle(sentences)

train_s = sentences[:40000]
val_s = sentences[40000:45000]
test_s = sentences[45000:50000]

# ================= VOCAB =================
vocab = Vocab(min_freq=2)
vocab.build(train_s)

print("Vocab size:", len(vocab))

# ================= DATASET =================
train_ds = LanguageDataset(train_s, vocab)
val_ds = LanguageDataset(val_s, vocab)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# ================= MODELS =================
models = {
    "RNN": RNNLM(len(vocab), 256, 512),
    "LSTM": LSTMLM(len(vocab), 256, 512),
    "GRU": GRULM(len(vocab), 256, 512),
    "BiLSTM": BiLSTMLM(len(vocab), 256, 512),
    "AttnLSTM": AttentionLSTM(len(vocab), 256, 512)
}

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

results = {}

# ================= TRAIN LOOP =================
for name, model in models.items():
    print(f"\n================ {name} =================")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_ppl = float("inf")

    for epoch in range(5):  # keep small for now
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, acc, ppl = evaluate(model, val_loader, criterion, device)

        print(f"\n{name} Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | PPL: {ppl:.2f}")

        # Save best model
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(model.state_dict(), f"{name}.pt")

    # Store final results
    results[name] = {
        "accuracy": acc,
        "perplexity": ppl
    }

    # ================= GENERATION =================
    print("\nSample Generation:")
    print(generate(model, vocab, "मैं आज", device=device))


# ================= FINAL RESULTS =================
print("\n================ FINAL RESULTS =================")
for k, v in results.items():
    print(f"{k}: Acc={v['accuracy']:.4f}, PPL={v['perplexity']:.2f}")