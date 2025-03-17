#!/usr/bin/env python

"""Tests for `createllm` package."""


import unittest
import pytest
import torch
from createllm.createllm import (
    ModelConfig,
    TextFileProcessor,
    GPTLanguageModel,
    GPTTrainer
)


class TestCreatellm(unittest.TestCase):
    """Tests for `createllm` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

def test_model_initialization():
    """Test if the model can be initialized correctly"""
    config = ModelConfig(vocab_size=1000)
    model = GPTLanguageModel(config)
    assert model is not None
    assert model.n_params > 0

def test_text_processor():
    """Test if text processing works correctly"""
    # Create a temporary test file
    test_text = "Hello, this is a test text for the language model."
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write(test_text)
    
    processor = TextFileProcessor("test.txt")
    text = processor.read_file()
    assert text is not None
    
    train_data, val_data, vocab_size, encode, decode = processor.tokenize(text)
    assert train_data is not None
    assert val_data is not None
    assert vocab_size > 0
    
    # Test encode-decode roundtrip
    encoded = encode(test_text)
    decoded = decode(encoded)
    assert decoded == test_text

def test_model_forward_pass():
    """Test if the model can perform forward pass"""
    config = ModelConfig(vocab_size=1000)
    model = GPTLanguageModel(config)
    
    # Create dummy input
    batch_size = 4
    seq_length = 8
    x = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    y = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    logits, loss = model(x, y)
    assert logits.shape == (batch_size, seq_length, config.vocab_size)
    assert loss is not None

def test_model_generation():
    """Test if the model can generate text"""
    config = ModelConfig(vocab_size=1000)
    model = GPTLanguageModel(config)
    
    # Create dummy input
    batch_size = 1
    seq_length = 4
    x = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Generate text
    generated = model.generate(x, max_new_tokens=5)
    assert generated.shape[1] == seq_length + 5

def test_trainer_initialization():
    """Test if the trainer can be initialized"""
    config = ModelConfig(vocab_size=1000)
    model = GPTLanguageModel(config)
    
    # Create dummy data
    train_data = torch.randint(0, config.vocab_size, (1000,))
    val_data = torch.randint(0, config.vocab_size, (100,))
    
    trainer = GPTTrainer(model, train_data, val_data, config)
    assert trainer is not None
    assert trainer.model == model
