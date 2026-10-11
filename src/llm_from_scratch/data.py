import torch

class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

def toy_corpus() -> str:
    return ("the small model learns patterns. " * 200
            + "attention mixes context. " * 200
            + "measure carefully and test hypotheses. " * 200)

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device):
    if len(data) <= block_size:
        raise ValueError("Dataset must contain more tokens than block_size.")
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in starts]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in starts]).to(device)
    return x, y
