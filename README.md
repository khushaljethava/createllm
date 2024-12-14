# createllm

A Python package that enables users to create and train their own Language Models (LLMs) from scratch using custom datasets. This package provides a streamlined approach to building, training, and deploying custom language models tailored to specific domains or use cases.

## 🎯 Core Purpose

createllm allows you to:
- Train custom language models on your specific text data
- Create domain-specific LLMs for specialized applications
- Build and experiment with configurable model architectures
- Deploy trained models for text generation tasks

## ✨ Key Features

- 🔨 Build LLMs from scratch using your own text data
- 🚀 Efficient training pipelines with PyTorch
- 📊 Real-time training progress tracking
- 🎛️ Flexible model configuration
- 💾 Simple model saving and loading
- 🎯 Custom text generation capabilities
- 📈 Built-in validation and performance monitoring

## 📋 Requirements

```bash
pip install torch torchvision tqdm dill
```

## 🚀 Quick Start Guide

### 1. Prepare Your Training Data

Place your training text in a file. The model learns from this text to generate similar content.

```
my_training_data.txt
├── Your custom text
├── Can be articles
├── Documentation
└── Any text content you want the model to learn from
```

### 2. Train Your Custom LLM

```python
from createllm import ModelConfig, Trainer, TextFileProcessor

# Step 1: Initialize model configuration
config = ModelConfig(
    vocab_size=None,  # Will be determined based on your data
    n_embd=384,      # Embedding dimension
    block_size=256,  # Context window size
    n_layer=4,       # Number of transformer layers
    n_head=4         # Number of attention heads
)

# Step 2: Process text and create trainer instance
processor = TextFileProcessor("path/to/my_training_data.txt")
text = processor.read_file()
train_data, val_data, vocab_size, encode, decode = processor.tokenize(text)

# Update config with vocab size
config.vocab_size = vocab_size

trainer = Trainer(
    model_config=config,
    train_data=train_data,
    val_data=val_data,
    learning_rate=3e-4
)

# Step 3: Start training
trainer.train(max_epochs=10)
```

### 3. Use Your Trained Model

```python
import torch
from createllm import GPTLanguageModel, ModelConfig

# Load the trained model
config = ModelConfig(
    vocab_size=100,  # Replace with actual vocab size
    n_embd=384,
    block_size=256,
    n_layer=4,
    n_head=4,
    dropout=0.2
)

model = GPTLanguageModel(config)
model.load_state_dict(torch.load("path/to/saved_model/model.pt"))
model.eval()

# Generate text
context = "Your prompt text"
encoded_context = torch.tensor([processor.stoi[c] for c in context], dtype=torch.long).unsqueeze(0)
generated = model.generate(encoded_context, max_new_tokens=100)
decoded_text = ''.join([processor.itos[idx.item()] for idx in generated[0]])
print("Generated Text:")
print(decoded_text)
```

## 📝 Example Use Cases

1. **Domain-Specific Documentation Generator**
```python
trainer = Trainer(
    model_config=config,
    train_data="technical_docs.txt",
    val_data="val_docs.txt"
)
trainer.train(max_epochs=5)
```

2. **Custom Writing Style Model**
```python
trainer = Trainer(
    model_config=config,
    train_data="author_works.txt",
    val_data="val_works.txt"
)
trainer.train(max_epochs=10)
```

3. **Specialized Content Generator**
```python
trainer = Trainer(
    model_config=config,
    train_data="specialized_content.txt",
    val_data="val_content.txt"
)
trainer.train(max_epochs=8)
```

## ⚙️ Model Configuration Options

Customize your model architecture based on your needs:

```python
config = ModelConfig(
    n_embd=384,     # Larger for more complex patterns
    block_size=256, # Larger for longer context
    n_layer=8,      # More layers for deeper understanding
    n_head=8,       # More heads for better pattern recognition
    dropout=0.2     # Adjust for overfitting prevention
)
```

## 💡 Training Tips

1. **Data Quality**
   - Clean your training data
   - Remove irrelevant content
   - Ensure consistent formatting

2. **Resource Management**
   ```python
   trainer = Trainer(
       batch_size=32,     # Reduce if running out of memory
       max_epochs=5,      # Adjust based on resources
       eval_interval=1    # Monitor training progress
   )
   ```

3. **Model Size vs Performance**
   - Smaller models (n_layer=4, n_head=4): Faster training, less complex patterns
   - Larger models (n_layer=8+, n_head=8+): Better understanding, more resource intensive

## 🔍 Monitoring Training

The training process provides real-time feedback:
```
Epoch 1/10: Training Loss: 3.1675, Validation Loss: 3.1681
Epoch 2/10: Training Loss: 2.4721, Validation Loss: 2.4759
...
```

## 📁 Saved Model Structure

```
saved_model/
├── model.pt           # Model weights
├── stoi.json          # Character-to-index mapping
├── itos.json          # Index-to-character mapping
└── config.json        # Model configuration
```

## ⚠️ Limitations

- Training requires significant computational resources
- Model quality depends on training data quality
- Larger models require more training time and resources

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📫 Support

For issues and questions, please open an issue in the GitHub repository.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Based on the GPT architecture with modifications for custom training and ease of use.
