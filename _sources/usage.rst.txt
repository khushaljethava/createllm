=====
Usage
=====

Usage Guide
===========

This guide will walk you through the main features and usage patterns of createllm.

Basic Usage
----------

1. Prepare Your Data
~~~~~~~~~~~~~~~~~~

First, prepare your training data in a text file:

.. code-block:: python

    # my_training_data.txt
    This is your training data.
    It can contain multiple lines.
    The model will learn from this text.

2. Initialize the Text Processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from createllm import TextFileProcessor

    # Initialize processor with your data file
    processor = TextFileProcessor("my_training_data.txt")
    
    # Read the text file
    text = processor.read_file()
    
    # Tokenize the text
    train_data, val_data, vocab_size, encode, decode = processor.tokenize(text)

3. Configure Your Model
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from createllm import ModelConfig

    config = ModelConfig(
        vocab_size=vocab_size,  # From tokenization
        n_embd=384,            # Embedding dimension
        block_size=256,        # Context window size
        n_layer=4,             # Number of transformer layers
        n_head=4,              # Number of attention heads
        dropout=0.2            # Dropout rate
    )

4. Create and Train the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from createllm import GPTLanguageModel, GPTTrainer

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

5. Generate Text
~~~~~~~~~~~~~

.. code-block:: python

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

Advanced Usage
------------

1. Custom Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can customize the model architecture based on your needs:

.. code-block:: python

    # Larger model for better understanding
    config = ModelConfig(
        vocab_size=vocab_size,
        n_embd=768,      # Larger embedding dimension
        block_size=512,  # Longer context window
        n_layer=8,       # More transformer layers
        n_head=8,        # More attention heads
        dropout=0.1      # Lower dropout for larger models
    )

2. Advanced Training Options
~~~~~~~~~~~~~~~~~~~~~~~~~

Customize the training process:

.. code-block:: python

    trainer = GPTTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        config=config,
        learning_rate=3e-4,
        batch_size=32,     # Smaller batch size for memory efficiency
        gradient_clip=1.0, # Prevent gradient explosion
        warmup_steps=1000  # Learning rate warmup
    )

3. Advanced Text Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

Control text generation with various parameters:

.. code-block:: python

    generated = model.generate(
        context_tensor,
        max_new_tokens=200,      # Generate more tokens
        temperature=0.7,         # Lower temperature for more focused output
        top_k=40,               # Limit sampling to top 40 tokens
        top_p=0.95,             # Nucleus sampling threshold
        repetition_penalty=1.5   # Stronger penalty for repetition
    )

4. Model Checkpointing
~~~~~~~~~~~~~~~~~~~~

Save and load model checkpoints:

.. code-block:: python

    # Save model
    model.save_model("checkpoints/model.pt")

    # Load model
    model.load_model("checkpoints/model.pt")

Best Practices
-------------

1. Data Preparation
~~~~~~~~~~~~~~~~~

* Clean your training data thoroughly
* Remove irrelevant content
* Ensure consistent formatting
* Consider data augmentation for small datasets

2. Model Configuration
~~~~~~~~~~~~~~~~~~~~

* Start with a smaller model for quick experiments
* Increase model size gradually
* Monitor validation loss to prevent overfitting
* Use appropriate dropout rates

3. Training Process
~~~~~~~~~~~~~~~~~

* Use learning rate warmup
* Monitor training and validation losses
* Save best model checkpoints
* Use early stopping if needed

4. Text Generation
~~~~~~~~~~~~~~~~

* Experiment with different temperature values
* Use top-k and top-p sampling for better quality
* Adjust repetition penalty based on output quality
* Consider using beam search for better coherence

Example Use Cases
---------------

1. Domain-Specific Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Train on technical documentation
    processor = TextFileProcessor("technical_docs.txt")
    text = processor.read_file()
    train_data, val_data, vocab_size, encode, decode = processor.tokenize(text)

    config = ModelConfig(vocab_size=vocab_size)
    model = GPTLanguageModel(config)
    trainer = GPTTrainer(model, train_data, val_data, config)
    trainer.train(max_epochs=5)

2. Custom Writing Style
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Train on specific author's works
    processor = TextFileProcessor("author_works.txt")
    text = processor.read_file()
    train_data, val_data, vocab_size, encode, decode = processor.tokenize(text)

    config = ModelConfig(vocab_size=vocab_size)
    model = GPTLanguageModel(config)
    trainer = GPTTrainer(model, train_data, val_data, config)
    trainer.train(max_epochs=5)

Troubleshooting
-------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Out of Memory Errors
   * Reduce batch size
   * Use gradient checkpointing
   * Reduce model size
   * Use mixed precision training

2. Poor Generation Quality
   * Increase training data size
   * Adjust temperature and sampling parameters
   * Train for more epochs
   * Use larger model architecture

3. Training Instability
   * Adjust learning rate
   * Use gradient clipping
   * Increase warmup steps
   * Check data quality

Getting Help
----------

If you need help:

* Check the `GitHub issues <https://github.com/khushaljethava/createllm/issues>`_
* Open a new issue with:
  - Your code
  - Error messages
  - Expected behavior
  - Actual behavior