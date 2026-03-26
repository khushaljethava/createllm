.. title:: Welcome to createllm

Welcome to createllm
=====================

.. image:: https://img.shields.io/pypi/v/createllm.svg
    :target: https://pypi.python.org/pypi/createllm
    :alt: PyPI version

**A Python package that enables users to create and train their own Language Learning Models (LLMs) from scratch using custom datasets. This package provides a simplified approach to building, training, and deploying custom language models tailored to specific domains or use cases.**

.. image:: https://img.shields.io/pypi/pyversions/createllm.svg
    :target: https://pypi.python.org/pypi/createllm
    :alt: Python versions

.. image:: https://img.shields.io/pypi/l/createllm.svg
    :target: https://pypi.python.org/pypi/createllm
    :alt: License

Contents
--------

- `Installation <installation.html>`_
- `Usage Guide <usage.html>`_
- `API Reference <api.html>`_
- `Examples <examples.html>`_
- `Contributing <contributing.html>`_
- `Changelog <history.html>`_

Features
--------

- 🔨 Build LLMs from scratch using your own text data
- 🚀 Efficient training with OneCycleLR scheduler
- 📊 Real-time training progress tracking with tqdm
- 🎛️ Configurable model architecture
- 💾 Easy model checkpointing and loading
- 🎯 Advanced text generation with temperature, top-k, and top-p sampling
- 📈 Built-in validation and early stopping
- 🔄 Automatic device selection (CPU/GPU)

Quick Start
----------

.. code-block:: python

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

License & Documentation
---------------------

- Free software: MIT License
- Documentation: `<https://khushaljethava.github.io/createllm>`_
- Source code: `<https://github.com/khushaljethava/createllm>`_

Indices and tables
-----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

