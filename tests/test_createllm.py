#!/usr/bin/env python

import os

import pytest
import torch

from createllm.createllm import (
    GPTLanguageModel,
    GPTTrainer,
    ModelConfig,
    TextFileProcessor,
    _build_arg_parser,
    benchmark_generation,
)


def test_model_initialization():
    model = GPTLanguageModel(ModelConfig(vocab_size=1000))
    assert model.n_params > 0


def test_text_processor_char_tokenize_roundtrip(tmp_path):
    txt = "Hello world"
    fp = tmp_path / "a.txt"
    fp.write_text(txt, encoding="utf-8")
    p = TextFileProcessor(str(fp))
    text = p.read_file()
    train, val, vocab, _, _ = p.tokenize(text, tokenizer_type="char")
    assert len(train) > 0 and len(val) > 0 and vocab > 0
    assert p.decode(p.encode(txt)) == txt


def test_text_processor_bpe_fallback_if_missing():
    p = TextFileProcessor()
    _, _, vocab, _, _ = p.tokenize("abc abc abc", tokenizer_type="bpe", bpe_vocab_size=32)
    assert vocab > 0
    assert p.tokenizer_type in {"bpe", "char"}


def test_tokenizer_save_load(tmp_path):
    p = TextFileProcessor()
    p.tokenize("abcabc")
    tok = tmp_path / "tokenizer.pt"
    p.save_tokenizer(str(tok))
    q = TextFileProcessor()
    q.load_tokenizer(str(tok))
    assert q.decode(q.encode("abc")) == "abc"


def test_model_forward_pass():
    config = ModelConfig(vocab_size=1000)
    model = GPTLanguageModel(config)
    x = torch.randint(0, config.vocab_size, (4, 8))
    y = torch.randint(0, config.vocab_size, (4, 8))
    logits, loss = model(x, y)
    assert logits.shape == (4, 8, config.vocab_size)
    assert loss is not None


def test_generate_with_cache_and_sampling_options():
    config = ModelConfig(vocab_size=64, n_embd=64, n_layer=2, n_head=4, block_size=16)
    model = GPTLanguageModel(config)
    x = torch.randint(0, config.vocab_size, (1, 4))
    out = model.generate(
        x,
        max_new_tokens=4,
        use_cache=True,
        min_p=0.01,
        no_repeat_ngram_size=2,
        bad_words_ids=[[1], [2, 3]],
        frequency_penalty=0.1,
        presence_penalty=0.1,
    )
    assert out.shape[1] >= 8


def test_generation_arg_validation():
    model = GPTLanguageModel(ModelConfig(vocab_size=32))
    x = torch.randint(0, 32, (1, 4))
    with pytest.raises(ValueError):
        model.generate(x, max_new_tokens=2, temperature=0)
    with pytest.raises(ValueError):
        model.generate(x, max_new_tokens=2, top_p=2)
    with pytest.raises(ValueError):
        model.generate(x, max_new_tokens=2, min_p=2)


def test_lora_adapter_save_load(tmp_path):
    config = ModelConfig(vocab_size=64, n_embd=64, n_layer=1, n_head=4, block_size=16, lora_r=4)
    model = GPTLanguageModel(config)
    path = tmp_path / "lora.pt"
    model.save_lora_adapters(str(path))
    model.load_lora_adapters(str(path))
    assert path.exists()


def test_trainer_metrics_and_resume(tmp_path):
    config = ModelConfig(vocab_size=32, n_embd=32, n_layer=1, n_head=4, block_size=8)
    model = GPTLanguageModel(config)
    train_data = torch.randint(0, config.vocab_size, (200,))
    val_data = torch.randint(0, config.vocab_size, (100,))
    trainer = GPTTrainer(model, train_data, val_data, config, batch_size=16)
    trainer._save_checkpoint(str(tmp_path), epoch=0, val_loss=1.2, val_perplexity=3.3)
    start = trainer._load_checkpoint(os.path.join(str(tmp_path), "checkpoint_epoch_0.pt"))
    assert start == 1
    loss, ppl = trainer._validate()
    assert loss >= 0 and ppl >= 1


def test_benchmark_generation_runs():
    config = ModelConfig(vocab_size=64, n_embd=64, n_layer=1, n_head=4, block_size=16)
    model = GPTLanguageModel(config)
    x = torch.randint(0, config.vocab_size, (1, 4))
    stats = benchmark_generation(model, x, max_new_tokens=4, use_cache=True)
    assert stats["new_tokens"] >= 1


def test_cli_parser_commands():
    parser = _build_arg_parser()
    tr = parser.parse_args(["train", "--input-file", "data.txt", "--config", "cfg.json"])
    assert tr.command == "train"
    gn = parser.parse_args(["generate", "--checkpoint", "a.pt", "--tokenizer-path", "tok.pt", "--prompt", "hi"])
    assert gn.command == "generate"
