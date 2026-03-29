import torch
from tqdm import tqdm

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)

        loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)