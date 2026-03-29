import torch

def generate(model, vocab, start_text, max_len=15, device="cpu"):
    model.eval()

    words = start_text.split()

    for _ in range(max_len):
        encoded = torch.tensor([vocab.encode(words)]).to(device)
        out = model(encoded)

        next_token = out.argmax(dim=-1)[0, -1].item()
        next_word = vocab.idx2word.get(next_token, "<UNK>")

        words.append(next_word)

    return " ".join(words)