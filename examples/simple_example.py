import torch
from createllm.createllm import (
    ModelConfig,
    TextFileProcessor,
    GPTLanguageModel,
    GPTTrainer,
    device
)

def main():
    # Initialize text processor with a sample text file
    processor = TextFileProcessor("sample.txt")
    text = processor.read_file()
    
    if text is None:
        print("Error: Could not read the text file")
        return
    
    # Tokenize the text
    train_data, val_data, vocab_size, encode, decode = processor.tokenize(text)
    
    # Create model configuration
    config = ModelConfig(
        vocab_size=vocab_size,
        n_embd=384,
        block_size=256,
        n_layer=4,
        n_head=4,
        dropout=0.2
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
    print("Starting training...")
    trainer.train(max_epochs=50, save_dir='checkpoints')
    
    # Generate some text
    print("\nGenerating text...")
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

if __name__ == "__main__":
    main() 