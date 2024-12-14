
# Usage

## Quick Start

Here's an example of how to use `createllm` to train a model:

```python
from createllm import ModelConfig, Trainer

config = ModelConfig(vocab_size=1000, n_embd=384, n_layer=4, n_head=4)
trainer = Trainer(config=config, train_data="data.txt", val_data="val.txt")
trainer.train(max_epochs=10)
```

## Advanced Features

- Configure architecture with `ModelConfig`.
- Tokenize and preprocess text with `TextFileProcessor`.
    