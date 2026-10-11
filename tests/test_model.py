import torch
from llm_from_scratch.model import GPT

def test_logits_and_loss_shapes():
    model = GPT(vocab_size=20, n_embd=16, n_head=4, n_layer=2, block_size=8)
    x = torch.randint(0, 20, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 20)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

def test_rejects_long_sequence():
    model = GPT(vocab_size=20, n_embd=16, n_head=4, n_layer=1, block_size=8)
    try:
        model(torch.randint(0, 20, (1, 9)))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for sequence longer than block size")
