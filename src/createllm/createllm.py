import argparse
import json
import logging
import math
import os
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

try:
    import sentencepiece as spm
except Exception:  # optional dependency
    spm = None

try:
    import yaml
except Exception:  # optional dependency
    yaml = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.manual_seed(1337)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class ModelConfig:
    vocab_size: int
    n_embd: int = 384
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.2
    lora_r: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, r: int = 0, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.r = r
        if r > 0:
            self.lora_a = nn.Linear(in_features, r, bias=False)
            self.lora_b = nn.Linear(r, out_features, bias=False)
            self.scaling = alpha / r
            self.lora_dropout = nn.Dropout(dropout)
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None
            self.scaling = 0.0
            self.lora_dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:
            y = y + self.lora_b(self.lora_a(self.lora_dropout(x))) * self.scaling
        return y


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        if len(self.data) <= self.block_size:
            return 1
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.data) <= self.block_size:
            x = torch.zeros(self.block_size, dtype=self.data.dtype)
            y = torch.zeros(self.block_size, dtype=self.data.dtype)
            x[: len(self.data)] = self.data
            y[: len(self.data) - 1] = self.data[1:]
            return x, y
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


class StreamingTextDataset(IterableDataset):
    """Simple streaming dataset from file(s) with optional shuffle buffer."""

    def __init__(self, file_paths: Sequence[str], encode_fn: Callable[[str], List[int]], block_size: int, chunk_size: int = 1024 * 1024, shuffle_buffer: int = 0):
        self.file_paths = list(file_paths)
        self.encode_fn = encode_fn
        self.block_size = block_size
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        rng = np.random.default_rng(1337)
        buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for path in self.file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read(self.chunk_size)
                while text:
                    ids = self.encode_fn(text)
                    if len(ids) > self.block_size:
                        data = torch.tensor(ids, dtype=torch.long)
                        for i in range(0, len(data) - self.block_size - 1, self.block_size):
                            x = data[i : i + self.block_size]
                            y = data[i + 1 : i + self.block_size + 1]
                            if self.shuffle_buffer > 0:
                                buffer.append((x, y))
                                if len(buffer) >= self.shuffle_buffer:
                                    j = int(rng.integers(0, len(buffer)))
                                    yield buffer.pop(j)
                            else:
                                yield x, y
                    text = f.read(self.chunk_size)
        while buffer:
            yield buffer.pop()


class MemMapTextDataset(Dataset):
    """Memory-mapped token dataset for large corpora."""

    def __init__(self, mmap_path: str, block_size: int, dtype: str = 'int32'):
        self.arr = np.memmap(mmap_path, mode='r', dtype=dtype)
        self.block_size = block_size

    def __len__(self):
        if len(self.arr) <= self.block_size:
            return 1
        return len(self.arr) - self.block_size

    def __getitem__(self, idx: int):
        x = torch.tensor(self.arr[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.arr[idx + 1 : idx + self.block_size + 1], dtype=torch.long)
        return x, y


class TextFileProcessor:
    def __init__(self, file_path: str = ''):
        self.file_path = file_path
        self.stoi = None
        self.itos = None
        self.tokenizer_type = 'char'
        self.sp_model = None
        self.special_tokens = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, '<mask>': 4}

    def read_file(self) -> Optional[str]:
        if not self.file_path:
            return None
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f'Error reading file {self.file_path}: {e}')
            return None

    def read_files(self, file_patterns: Sequence[str]) -> str:
        files: List[str] = []
        for p in file_patterns:
            files.extend(sorted(glob(p)))
        contents = []
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                contents.append(f.read())
        joined = '\n'.join(contents)
        logger.info(f'Loaded {len(files)} files, total chars={len(joined)}')
        return joined

    def tokenize(self, text: str, min_freq: int = 1, tokenizer_type: str = 'char', bpe_vocab_size: int = 2000):
        _ = min_freq
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == 'bpe':
            if spm is None:
                logger.warning('sentencepiece not available; falling back to char tokenizer')
                self.tokenizer_type = 'char'
            else:
                with tempfile.TemporaryDirectory() as td:
                    input_txt = os.path.join(td, 'corpus.txt')
                    model_prefix = os.path.join(td, 'spm')
                    with open(input_txt, 'w', encoding='utf-8') as f:
                        f.write(text)
                    spm.SentencePieceTrainer.train(
                        input=input_txt,
                        model_prefix=model_prefix,
                        vocab_size=bpe_vocab_size,
                        model_type='bpe',
                        bos_id=2,
                        eos_id=3,
                        pad_id=0,
                        unk_id=1,
                    )
                    self.sp_model = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
                    ids = self.sp_model.encode(text, out_type=int)
                    data = torch.tensor(ids, dtype=torch.long)
                    n = int(0.9 * len(data))
                    return data[:n], data[n:], int(self.sp_model.get_piece_size()), self.encode, self.decode

        # char fallback
        chars = sorted(list(set(text)))
        vocab_size = len(chars) + len(self.special_tokens)
        self.stoi = {**self.special_tokens, **{ch: i + len(self.special_tokens) for i, ch in enumerate(chars)}}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        return data[:n], data[n:], vocab_size, self.encode, self.decode

    def encode(self, text: str) -> List[int]:
        if self.tokenizer_type == 'bpe' and self.sp_model is not None:
            return list(self.sp_model.encode(text, out_type=int))
        if self.stoi is None:
            raise ValueError('Tokenizer not initialized')
        return [self.stoi.get(c, self.stoi['<unk>']) for c in text]

    def decode(self, token_ids: List[int]) -> str:
        if self.tokenizer_type == 'bpe' and self.sp_model is not None:
            return self.sp_model.decode(token_ids)
        if self.itos is None:
            raise ValueError('Tokenizer not initialized')
        return ''.join([self.itos.get(i, '<unk>') for i in token_ids])

    def save_tokenizer(self, path: str):
        payload = {'tokenizer_type': self.tokenizer_type}
        if self.tokenizer_type == 'bpe' and self.sp_model is not None:
            payload['spm_model'] = bytes(self.sp_model.serialized_model_proto())
        else:
            payload['stoi'] = self.stoi
            payload['itos'] = self.itos
        torch.save(payload, path)

    def load_tokenizer(self, path: str):
        payload = torch.load(path, map_location='cpu')
        self.tokenizer_type = payload.get('tokenizer_type', 'char')
        if self.tokenizer_type == 'bpe' and payload.get('spm_model') is not None and spm is not None:
            self.sp_model = spm.SentencePieceProcessor(model_proto=payload['spm_model'])
        else:
            self.tokenizer_type = 'char'
            self.stoi = payload['stoi']
            self.itos = {int(k): v for k, v in payload['itos'].items()} if isinstance(next(iter(payload['itos'].keys())), str) else payload['itos']


    def vocab_size(self) -> int:
        if self.tokenizer_type == 'bpe' and self.sp_model is not None:
            return int(self.sp_model.get_piece_size())
        if self.stoi is None:
            raise ValueError('Tokenizer not initialized')
        return len(self.stoi)

    # backward compatibility
    def save_vocab(self, path: str):
        self.save_tokenizer(path)

    def load_vocab(self, path: str):
        self.load_tokenizer(path)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(f'n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})')
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        linear = lambda in_f, out_f: LoRALinear(in_f, out_f, r=config.lora_r, alpha=config.lora_alpha, dropout=config.lora_dropout) if config.lora_r > 0 else nn.Linear(in_f, out_f)
        self.query = linear(config.n_embd, config.n_embd)
        self.key = linear(config.n_embd, config.n_embd)
        self.value = linear(config.n_embd, config.n_embd)
        self.proj = linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        b, t, c = x.shape
        q = self.query(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        present = (k, v) if use_cache else None
        return self.proj(y), present


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(), nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout))

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        attn_out, present = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


class GPTLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        base = module.base if isinstance(module, LoRALinear) else module
        if isinstance(base, (nn.Linear, nn.Embedding)):
            nn.init.normal_(base.weight, mean=0.0, std=0.02)
            if isinstance(base, nn.Linear) and base.bias is not None:
                nn.init.zeros_(base.bias)
        elif isinstance(base, nn.LayerNorm):
            nn.init.ones_(base.weight)
            nn.init.zeros_(base.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, use_cache: bool = False):
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f'Input sequence length {t} exceeds block_size {self.config.block_size}.')

        tok_emb = self.token_embedding_table(idx)
        start_pos = 0
        if past_key_values and len(past_key_values) > 0:
            start_pos = past_key_values[0][0].shape[2]
        pos = torch.arange(start_pos, start_pos + t, dtype=torch.long, device=idx.device) % self.config.block_size
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb

        presents = []
        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, past_kv=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)

        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        if use_cache:
            return logits, loss, presents
        return logits, loss

    def _apply_sampling_constraints(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        top_k: Optional[int],
        top_p: Optional[float],
        min_p: Optional[float],
        no_repeat_ngram_size: int,
        bad_words_ids: Optional[List[List[int]]],
        frequency_penalty: float,
        presence_penalty: float,
    ) -> torch.Tensor:
        if frequency_penalty != 0 or presence_penalty != 0:
            for b in range(logits.size(0)):
                tokens = generated[b].tolist()
                if not tokens:
                    continue
                uniq = set(tokens)
                for tok in uniq:
                    count = tokens.count(tok)
                    logits[b, tok] -= presence_penalty + frequency_penalty * count

        if no_repeat_ngram_size > 1 and generated.size(1) >= no_repeat_ngram_size - 1:
            n = no_repeat_ngram_size
            for b in range(logits.size(0)):
                toks = generated[b].tolist()
                prefix = tuple(toks[-(n - 1):])
                blocked = set()
                for i in range(len(toks) - n + 1):
                    if tuple(toks[i : i + n - 1]) == prefix:
                        blocked.add(toks[i + n - 1])
                if blocked:
                    logits[b, list(blocked)] = -float('inf')

        if bad_words_ids:
            for b in range(logits.size(0)):
                toks = generated[b].tolist()
                for seq in bad_words_ids:
                    if not seq:
                        continue
                    if len(seq) == 1:
                        logits[b, seq[0]] = -float('inf')
                    elif len(toks) >= len(seq) - 1 and toks[-(len(seq) - 1) :] == seq[:-1]:
                        logits[b, seq[-1]] = -float('inf')

        if top_k is not None:
            vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < vals[:, [-1]]] = -float('inf')

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            remove = remove.scatter(1, sorted_indices, remove)
            logits[remove] = -float('inf')

        if min_p is not None:
            probs = F.softmax(logits, dim=-1)
            max_prob = probs.max(dim=-1, keepdim=True).values
            logits[probs < (min_p * max_prob)] = -float('inf')

        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
        min_p: Optional[float] = None,
        no_repeat_ngram_size: int = 0,
        bad_words_ids: Optional[List[List[int]]] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        if temperature <= 0:
            raise ValueError('temperature must be > 0')
        if top_p is not None and not (0 < top_p <= 1):
            raise ValueError('top_p must be in (0, 1]')
        if repetition_penalty <= 0:
            raise ValueError('repetition_penalty must be > 0')
        if min_p is not None and not (0 <= min_p <= 1):
            raise ValueError('min_p must be in [0,1]')

        past = None
        generated = idx
        for step in tqdm(range(max_new_tokens), desc='Generating'):
            if use_cache:
                model_in = generated[:, -1:] if step > 0 else generated[:, -self.config.block_size :]
                logits, _, past = self(model_in, past_key_values=past, use_cache=True)
            else:
                logits, _ = self(generated[:, -self.config.block_size :])
            logits = logits[:, -1, :] / temperature

            for i in range(logits.size(0)):
                for previous_token in generated[i]:
                    logits[i, previous_token] /= repetition_penalty

            logits = self._apply_sampling_constraints(
                logits,
                generated,
                top_k,
                top_p,
                min_p,
                no_repeat_ngram_size,
                bad_words_ids,
                frequency_penalty,
                presence_penalty,
            )

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, idx_next), dim=1)
            if eos_token_id is not None and torch.all(idx_next.squeeze(-1) == eos_token_id):
                break
        return generated

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path, map_location=device))

    def save_lora_adapters(self, path: str):
        lora_state = {k: v for k, v in self.state_dict().items() if 'lora_' in k}
        torch.save(lora_state, path)

    def load_lora_adapters(self, path: str):
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state, strict=False)

    def export_torchscript(self, path: str):
        self.eval()
        scripted = torch.jit.script(self)
        scripted.save(path)

    def export_onnx(self, path: str):
        self.eval()
        dummy = torch.randint(0, self.config.vocab_size, (1, min(8, self.config.block_size)), dtype=torch.long, device=device)
        torch.onnx.export(self, (dummy, None), path, input_names=['input_ids', 'targets'], output_names=['logits', 'loss'], opset_version=17)


class GPTTrainer:
    def __init__(
        self,
        model: GPTLanguageModel,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        config: ModelConfig,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        gradient_clip: float = 1.0,
        warmup_steps: int = 1000,
        accumulation_steps: int = 1,
        num_workers: int = 0,
        use_amp: bool = True,
        log_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.accumulation_steps = max(1, accumulation_steps)
        self.use_amp = use_amp and device.type == 'cuda'
        pin_memory = device.type == 'cuda'
        self.train_loader = DataLoader(TextDataset(train_data, config.block_size), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.val_loader = DataLoader(TextDataset(val_data, config.block_size), batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.total_steps = max(1, math.ceil(len(self.train_loader) / self.accumulation_steps))
        self.pct_start = min(warmup_steps / (self.total_steps * 5), 0.3)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=learning_rate, total_steps=self.total_steps * 5, pct_start=self.pct_start, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1e4, three_phase=False)
        self.gradient_clip = gradient_clip
        self.best_val_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0
        self.writer = None
        if log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception:
                logger.warning('TensorBoard not available; continuing without experiment writer')

    @staticmethod
    def _loss_to_perplexity(loss_value: float) -> float:
        return float(math.exp(min(loss_value, 20.0)))

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('scaler_state_dict') is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        return int(checkpoint.get('epoch', -1)) + 1

    def train(self, max_epochs: int, save_dir: str = 'checkpoints', resume_from: Optional[str] = None) -> List[Dict[str, float]]:
        os.makedirs(save_dir, exist_ok=True)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.optimizer.param_groups[0]['lr'], total_steps=self.total_steps * max_epochs, pct_start=self.pct_start, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1e4, three_phase=False)
        start_epoch = self._load_checkpoint(resume_from) if resume_from else 0
        history: List[Dict[str, float]] = []
        metrics_path = os.path.join(save_dir, 'metrics.jsonl')
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, max_epochs):
            self.model.train()
            total_loss = 0.0
            grad_norm = 0.0
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{max_epochs}')
            for step, (xb, yb) in enumerate(pbar, start=1):
                xb, yb = xb.to(device), yb.to(device)
                amp_ctx = (lambda: torch.amp.autocast('cuda')) if self.use_amp else nullcontext
                with amp_ctx():
                    _, loss = self.model(xb, yb)
                    loss = loss / self.accumulation_steps
                self.scaler.scale(loss).backward()
                if step % self.accumulation_steps == 0 or step == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                batch_loss = loss.item() * self.accumulation_steps
                total_loss += batch_loss
                pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

            train_loss = total_loss / max(1, len(self.train_loader))
            train_ppl = self._loss_to_perplexity(train_loss)
            val_loss, val_ppl = self._validate()
            rec = {'epoch': float(epoch), 'train_loss': train_loss, 'train_perplexity': train_ppl, 'val_loss': val_loss, 'val_perplexity': val_ppl}
            history.append(rec)
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec) + '\n')
            if self.writer:
                self.writer.add_scalar('loss/train', train_loss, epoch)
                self.writer.add_scalar('loss/val', val_loss, epoch)
                self.writer.add_scalar('ppl/train', train_ppl, epoch)
                self.writer.add_scalar('ppl/val', val_ppl, epoch)
                self.writer.add_scalar('optim/lr', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('optim/grad_norm', grad_norm, epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(save_dir, epoch, val_loss, val_ppl)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        return history

    def _save_checkpoint(self, save_dir: str, epoch: int, val_loss: float, val_perplexity: float):
        ckpt = {
            'epoch': epoch,
            'model_config': asdict(self.model.config),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
        }
        path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(ckpt, path)

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        for xb, yb in self.val_loader:
            xb, yb = xb.to(device), yb.to(device)
            amp_ctx = (lambda: torch.amp.autocast('cuda')) if self.use_amp else nullcontext
            with amp_ctx():
                _, loss = self.model(xb, yb)
            total_loss += loss.item()
        avg = total_loss / max(1, len(self.val_loader))
        ppl = self._loss_to_perplexity(avg)
        logger.info(f'Validation Loss: {avg:.4f}, Validation PPL: {ppl:.2f}')
        return avg, ppl


def benchmark_generation(model: GPTLanguageModel, prompt_ids: torch.Tensor, max_new_tokens: int = 64, use_cache: bool = True, seed: int = 1337) -> Dict[str, float]:
    torch.manual_seed(seed)
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    import time

    if start and end:
        start.record()
        out = model.generate(prompt_ids, max_new_tokens=max_new_tokens, use_cache=use_cache)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.time()
        out = model.generate(prompt_ids, max_new_tokens=max_new_tokens, use_cache=use_cache)
        elapsed_ms = (time.time() - t0) * 1000
    total_tokens = out.shape[1] - prompt_ids.shape[1]
    return {'elapsed_ms': float(elapsed_ms), 'new_tokens': float(total_tokens), 'tok_per_s': float((total_tokens / max(1e-6, elapsed_ms / 1000.0)))}


def create_fastapi_app(model: GPTLanguageModel, processor: TextFileProcessor):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except Exception as e:
        raise RuntimeError('fastapi and pydantic are required for serving') from e

    app = FastAPI(title='createllm')

    class GenReq(BaseModel):
        prompt: str
        max_new_tokens: int = 100

    @app.post('/generate')
    def generate(req: GenReq):
        ids = processor.encode(req.prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens=req.max_new_tokens)
        return {'text': processor.decode(y[0].tolist())}

    return app


def _load_config_file(path: str) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    with open(path, 'r', encoding='utf-8') as f:
        if ext == '.json':
            return json.load(f)
        if ext in {'.yaml', '.yml'}:
            if yaml is None:
                raise RuntimeError('pyyaml is required for YAML config files')
            return yaml.safe_load(f)
    raise ValueError('Config must be .json/.yaml/.yml')


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='createllm CLI')
    sub = parser.add_subparsers(dest='command', required=True)

    tr = sub.add_parser('train', help='Train a model from text file(s)')
    tr.add_argument('--input-file', default=None)
    tr.add_argument('--input-glob', nargs='*', default=None)
    tr.add_argument('--config', default=None)
    tr.add_argument('--tokenizer-type', default='char', choices=['char', 'bpe'])
    tr.add_argument('--bpe-vocab-size', type=int, default=2000)
    tr.add_argument('--save-dir', default='checkpoints')
    tr.add_argument('--max-epochs', type=int, default=5)
    tr.add_argument('--batch-size', type=int, default=64)
    tr.add_argument('--learning-rate', type=float, default=3e-4)
    tr.add_argument('--block-size', type=int, default=256)
    tr.add_argument('--n-embd', type=int, default=384)
    tr.add_argument('--n-layer', type=int, default=4)
    tr.add_argument('--n-head', type=int, default=4)
    tr.add_argument('--dropout', type=float, default=0.2)
    tr.add_argument('--accumulation-steps', type=int, default=1)
    tr.add_argument('--resume-from', default=None)
    tr.add_argument('--log-dir', default=None)
    tr.add_argument('--lora-r', type=int, default=0)

    gn = sub.add_parser('generate', help='Generate text')
    gn.add_argument('--checkpoint', required=True)
    gn.add_argument('--tokenizer-path', '--vocab-path', dest='tokenizer_path', required=True)
    gn.add_argument('--prompt', required=True)
    gn.add_argument('--max-new-tokens', type=int, default=100)
    gn.add_argument('--temperature', type=float, default=1.0)
    gn.add_argument('--top-k', type=int, default=None)
    gn.add_argument('--top-p', type=float, default=None)
    gn.add_argument('--min-p', type=float, default=None)
    gn.add_argument('--use-cache', action='store_true')

    bm = sub.add_parser('benchmark', help='Benchmark generation throughput')
    bm.add_argument('--checkpoint', required=True)
    bm.add_argument('--tokenizer-path', required=True)
    bm.add_argument('--prompt', default='Hello')
    bm.add_argument('--max-new-tokens', type=int, default=64)
    bm.add_argument('--use-cache', action='store_true')

    return parser


def _build_model_from_checkpoint(checkpoint: Dict[str, Any], tokenizer_size: int) -> GPTLanguageModel:
    if 'model_config' in checkpoint:
        cfg = checkpoint['model_config']
        cfg['vocab_size'] = tokenizer_size
        config = ModelConfig(**cfg)
    else:
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        n_embd = state_dict['token_embedding_table.weight'].shape[1]
        block_size = state_dict['position_embedding_table.weight'].shape[0]
        n_layer = len([k for k in state_dict if k.startswith('blocks.') and k.endswith('ln1.weight')])
        config = ModelConfig(vocab_size=tokenizer_size, n_embd=n_embd, block_size=block_size, n_layer=n_layer, n_head=4)
    model = GPTLanguageModel(config).to(device)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    return model


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.command == 'train':
        if args.config:
            cfg = _load_config_file(args.config)
            for k, v in cfg.items():
                if hasattr(args, k.replace('-', '_')):
                    setattr(args, k.replace('-', '_'), v)

        proc = TextFileProcessor(args.input_file or '')
        if args.input_glob:
            text = proc.read_files(args.input_glob)
        else:
            text = proc.read_file()
        if text is None:
            raise ValueError('No training text loaded. Provide --input-file or --input-glob')

        train_data, val_data, vocab_size, _, _ = proc.tokenize(text, tokenizer_type=args.tokenizer_type, bpe_vocab_size=args.bpe_vocab_size)
        config = ModelConfig(vocab_size=vocab_size, n_embd=args.n_embd, block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head, dropout=args.dropout, lora_r=args.lora_r)
        model = GPTLanguageModel(config)
        trainer = GPTTrainer(model=model, train_data=train_data, val_data=val_data, config=config, learning_rate=args.learning_rate, batch_size=args.batch_size, accumulation_steps=args.accumulation_steps, log_dir=args.log_dir)

        os.makedirs(args.save_dir, exist_ok=True)
        proc.save_tokenizer(os.path.join(args.save_dir, 'tokenizer.pt'))
        with open(os.path.join(args.save_dir, 'run_config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=2)
        history = trainer.train(max_epochs=args.max_epochs, save_dir=args.save_dir, resume_from=args.resume_from)
        logger.info(f'Training complete: epochs={len(history)}')

    elif args.command == 'generate':
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        proc = TextFileProcessor()
        proc.load_tokenizer(args.tokenizer_path)
        model = _build_model_from_checkpoint(checkpoint, proc.vocab_size()).to(device)
        ids = proc.encode(args.prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, min_p=args.min_p, use_cache=args.use_cache)
        print(proc.decode(y[0].tolist()))

    elif args.command == 'benchmark':
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        proc = TextFileProcessor()
        proc.load_tokenizer(args.tokenizer_path)
        model = _build_model_from_checkpoint(checkpoint, proc.vocab_size()).to(device)
        ids = torch.tensor([proc.encode(args.prompt)], dtype=torch.long, device=device)
        stats = benchmark_generation(model, ids, max_new_tokens=args.max_new_tokens, use_cache=args.use_cache)
        print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
