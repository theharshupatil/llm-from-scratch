import argparse
import torch
from .data import CharTokenizer, get_batch, toy_corpus
from .model import GPT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    text = toy_corpus()
    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(encoded))
    train_data, val_data = encoded[:split], encoded[split:]
    model = GPT(tokenizer.vocab_size, block_size=args.block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device={device}, vocab={tokenizer.vocab_size}, parameters={sum(p.numel() for p in model.parameters()):,}")
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0:
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.block_size, device)
                _, val_loss = model(vx, vy)
            print(f"step={step:4d} train_loss={loss.item():.4f} val_loss={val_loss.item():.4f}")
    torch.save({"model": model.state_dict(), "chars": tokenizer.chars, "config": vars(args)}, "checkpoint.pt")
    print("Saved checkpoint.pt")

if __name__ == "__main__":
    main()
