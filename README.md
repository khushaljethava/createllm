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
- 🚀 Multi-threaded training for faster model development
- 📊 Real-time training progress tracking
- 🎛️ Configurable model architecture
- 💾 Easy model saving and loading
- 🎯 Custom text generation capabilities
- 📈 Built-in performance monitoring

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
from createllm import ModelConfig, GPTTrainer, TextFileProcessor

# Initialize model configuration
config = ModelConfig(
    vocab_size=None,  # Will be automatically set based on your data
    n_embd=384,      # Embedding dimension
    block_size=256,  # Context window size
    n_layer=4,       # Number of transformer layers
    n_head=4        # Number of attention heads
)

# Create trainer instance
trainer = GPTTrainer(
    text_file="path/to/my_training_data.txt",
    learning_rate=3e-4,
    batch_size=64,
    max_iters=5000,
    eval_interval=500,
    saved_path="path/to/save/model"
)

# Start training
trainer.trainer()  # This will automatically process text and train the model
```

### 3. Use Your Trained Model

```python
from createllm import LLMModel

# Load your trained model
model = LLMModel("path/to/saved/model")

# Generate text
generated_text = model.generate("Your prompt text")
print(generated_text)
```

## 📝 Example Use Cases

1. **Domain-Specific Documentation Generator**
```python
# Train on technical documentation
trainer = GPTTrainer(
    text_file="technical_docs.txt",
    saved_path="tech_docs_model"
)
trainer.trainer()
```

2. **Custom Writing Style Model**
```python
# Train on specific author's works
trainer = GPTTrainer(
    text_file="author_works.txt",
    saved_path="author_style_model"
)
trainer.trainer()
```

3. **Specialized Content Generator**
```python
# Train on specific content type
trainer = GPTTrainer(
    text_file="specialized_content.txt",
    saved_path="content_model"
)
trainer.trainer()
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
   trainer = GPTTrainer(
       batch_size=32,     # Reduce if running out of memory
       max_iters=5000,    # Increase for better learning
       eval_interval=500  # Monitor training progress
   )
   ```

3. **Model Size vs Performance**
   - Smaller models (n_layer=4, n_head=4): Faster training, less complex patterns
   - Larger models (n_layer=8+, n_head=8+): Better understanding, more resource intensive

## 🔍 Monitoring Training

The training process provides real-time feedback:
```
step 0: train loss 4.1675, val loss 4.1681
step 500: train loss 2.4721, val loss 2.4759
step 1000: train loss 1.9842, val loss 1.9873
step 1500: train loss 1.1422, val loss 1.1422
...
```

## 📁 Saved Model Structure

```
saved_model/
├── model.pt           # Model weights
├── encoder.pickle    # Text encoder
├── decoder.pickle    # Text decoder
└── config.json      # Model configuration
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
