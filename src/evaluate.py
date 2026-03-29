import torch

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
            total_loss += loss.item()

            preds = out.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, accuracy, perplexity