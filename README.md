# createllm

A Python package that enables users to create and train their own Language Learning Models (LLMs) from scratch using custom datasets. This package provides a simplified approach to building, training, and deploying custom language models tailored to specific domains or use cases.

## 🎯 Core Purpose

createllm allows you to:
- Train custom language models on your specific text data
- Create domain-specific LLMs for specialized applications
- Build and experiment with different model architectures
- Deploy trained models for text generation tasks

## ✨ Key Features

- 🔨 Build LLMs from scratch using your own text data
- 🚀 Efficient training with OneCycleLR scheduler
- 📊 Real-time training progress tracking with tqdm
- 🎛️ Configurable model architecture
- 💾 Easy model checkpointing and loading
- 🎯 Advanced text generation with temperature, top-k, and top-p sampling
- 📈 Built-in validation and early stopping
- 🔄 Automatic device selection (CPU/GPU)

## 📋 Requirements

```bash
pip install createllm
```

The package requires:
- Python >= 3.7
- PyTorch >= 2.0.0
- tqdm >= 4.65.0
- numpy >= 1.24.0
- dataclasses >= 0.6
- typing-extensions >= 4.5.0

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
from createllm import ModelConfig, TextFileProcessor, GPTLanguageModel, GPTTrainer

# Initialize text processor with your data file
processor = TextFileProcessor("my_training_data.txt")
text = processor.read_file()

# Tokenize the text
train_data, val_data, vocab_size, encode, decode = processor.tokenize(text)

# Create model configuration
config = ModelConfig(
    vocab_size=vocab_size,
    n_embd=384,      # Embedding dimension
    block_size=256,  # Context window size
    n_layer=4,       # Number of transformer layers
    n_head=4,        # Number of attention heads
    dropout=0.2      # Dropout rate
)

# Initialize the model
model = GPTLanguageModel(config)
print(f"Model initialized with {model.n_params / 1e6:.2f}M parameters")

# Initialize the trainer
trainer = GPTTrainer(
    model=model,
    train_data=train_data,
    val_data=val_data,
    config=config,
    learning_rate=3e-4,
    batch_size=64,
    gradient_clip=1.0,
    warmup_steps=1000
)

# Train the model
trainer.train(max_epochs=5, save_dir='checkpoints')
```

### 3. Generate Text with Your Trained Model

```python
# Generate text
context = "Once upon a time"
context_tokens = encode(context)
context_tensor = torch.tensor([context_tokens], dtype=torch.long).to(device)

generated = model.generate(
    context_tensor,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2
)

# Decode and print the generated text
generated_text = decode(generated[0].tolist())
print(f"\nGenerated text:\n{generated_text}")
```

## 📝 Example Use Cases

1. **Domain-Specific Documentation Generator**
```python
# Train on technical documentation
processor = TextFileProcessor("technical_docs.txt")
text = processor.read_file()
train_data, val_data, vocab_size, encode, decode = processor.tokenize(text)

config = ModelConfig(vocab_size=vocab_size)
model = GPTLanguageModel(config)
trainer = GPTTrainer(model, train_data, val_data, config)
trainer.train(max_epochs=5)
```

2. **Custom Writing Style Model**
```python
# Train on specific author's works
processor = TextFileProcessor("author_works.txt")
text = processor.read_file()
train_data, val_data, vocab_size, encode, decode = processor.tokenize(text)

config = ModelConfig(vocab_size=vocab_size)
model = GPTLanguageModel(config)
trainer = GPTTrainer(model, train_data, val_data, config)
trainer.train(max_epochs=5)
```

## ⚙️ Model Configuration Options

Customize your model architecture based on your needs:

```python
config = ModelConfig(
    vocab_size=vocab_size,  # Vocabulary size from tokenization
    n_embd=384,            # Larger for more complex patterns
    block_size=256,        # Larger for longer context
    n_layer=4,             # More layers for deeper understanding
    n_head=4,              # More heads for better pattern recognition
    dropout=0.2            # Adjust for overfitting prevention
)
```

## 💡 Training Tips

1. **Data Quality**
   - Clean your training data
   - Remove irrelevant content
   - Ensure consistent formatting

2. **Resource Management**
   ```python
   trainer = GPTTrainer(
       model=model,
       train_data=train_data,
       val_data=val_data,
       config=config,
       batch_size=32,     # Reduce if running out of memory
       learning_rate=3e-4 # Adjust based on your needs
   )
   ```

3. **Model Size vs Performance**
   - Smaller models (n_layer=4, n_head=4): Faster training, less complex patterns
   - Larger models (n_layer=8+, n_head=8+): Better understanding, more resource intensive

## 🔍 Monitoring Training

The training process provides real-time feedback:
```
Epoch 1: Training Loss: 3.1342, Validation Loss: 4.3930
Epoch 2: Training Loss: 2.3390, Validation Loss: 4.5054
Epoch 3: Training Loss: 2.0413, Validation Loss: 4.5405
Epoch 4: Training Loss: 1.9232, Validation Loss: 4.5442
Epoch 5: Training Loss: 1.8738, Validation Loss: 4.5442
```

## 📁 Checkpoint Structure

```
checkpoints/
├── checkpoint_epoch_0.pt  # Model checkpoint
├── checkpoint_epoch_1.pt
└── ...
```

## ⚠️ Limitations

- Training requires significant computational resources
- Model quality depends on training data quality
- Larger models require more training time and resources
- Text generation quality may vary based on training data size and quality

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📫 Support

For issues and questions, please open an issue in the GitHub repository.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Based on the GPT architecture with modifications for custom training and ease of use.
