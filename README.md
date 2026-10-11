# LLM from Scratch

A compact decoder-only Transformer you can understand, test, and train. The goal is to learn the mechanics of language-model training before experimenting with scaling or novel training methods.

## Quick start

Python 3.10+ is recommended. PyTorch installation depends on your machine; install the appropriate build from the official PyTorch instructions, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m llm_from_scratch.train --steps 200
pytest
```

The starter uses a tiny character-level corpus generated in code so the baseline is self-contained. It is a learning scaffold, not a useful general-purpose language model.

## Layout

- `model.py`: causal self-attention, feed-forward block, decoder model.
- `data.py`: tiny character tokenizer and toy corpus.
- `train.py`: minimal training loop.
- `tests/`: shape and causal-mask checks.

## Next milestones

- [ ] Train on a small real text corpus with a documented data pipeline.
- [ ] Add checkpoints, validation loss, and reproducible seeds.
- [ ] Compare positional encodings and normalization variants.
- [ ] Measure scaling behavior across model and dataset sizes.
- [ ] Add tokenizer and data-quality ablations.

Report train/validation loss and compute. Do not describe toy-corpus performance as frontier capability.
