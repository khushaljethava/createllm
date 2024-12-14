import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import os
import json
import pickle
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device and seed for reproducibility
torch.manual_seed(1337)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class ModelConfig:
    """Configuration class for GPT model parameters"""
    vocab_size: int
    n_embd: int = 384
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.2

class TextDataset(Dataset):
    """PyTorch Dataset for tokenized text data"""

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

class TextFileProcessor:
    """Handles text file processing and tokenization for GPT model training"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.stoi = None
        self.itos = None

    def read_file(self) -> Optional[str]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"Successfully read file: {self.file_path}")
            return text
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            return None
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return None

    def tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, int, callable, callable]:
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        encode = lambda s: [self.stoi.get(c, 0) for c in s]
        decode = lambda l: ''.join([self.itos.get(i, '') for i in l])

        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data, val_data = data[:n], data[n:]
        return train_data, val_data, vocab_size, encode, decode

class GPTLanguageModel(nn.Module):
    """GPT Language Model"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([
            Block(config.n_embd, config.n_head, config.block_size, config.dropout)
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)
        self.n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model initialized with {self.n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        for _ in tqdm(range(max_new_tokens), desc="Generating"):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k_values, _ = torch.topk(logits, top_k)
                logits[logits < top_k_values[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class Trainer:
    """Trainer for GPT model"""

    def __init__(self, model: GPTLanguageModel, train_data: torch.Tensor, val_data: torch.Tensor, config: ModelConfig, learning_rate: float = 3e-4):
        self.model = model.to(device)
        self.train_loader = DataLoader(TextDataset(train_data, config.block_size), batch_size=64, shuffle=True)
        self.val_loader = DataLoader(TextDataset(val_data, config.block_size), batch_size=64)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self.model.train()
            total_loss = 0

            for xb, yb in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}"):
                xb, yb = xb.to(device), yb.to(device)
                logits, loss = self.model(xb, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")

            self._validate()

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss = 0

        for xb, yb in self.val_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = self.model(xb, yb)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"Validation Loss: {avg_loss:.4f}")
