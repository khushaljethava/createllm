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
import math

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
        # Handle case where data is smaller than block_size
        if len(self.data) <= self.block_size:
            return 1
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle case where data is smaller than block_size
        if len(self.data) <= self.block_size:
            # Pad with zeros if needed
            x = torch.zeros(self.block_size, dtype=self.data.dtype)
            y = torch.zeros(self.block_size, dtype=self.data.dtype)
            x[:len(self.data)] = self.data
            y[:len(self.data)-1] = self.data[1:]
            return x, y
            
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

class TextFileProcessor:
    """Handles text file processing and tokenization for GPT model training"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.stoi = None
        self.itos = None
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3,
            '<mask>': 4
        }

    def read_file(self) -> Optional[str]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"Successfully read file: {self.file_path}")
            return text
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            return None
        except UnicodeDecodeError:
            logger.error(f"File encoding error: {self.file_path}")
            return None
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return None

    def tokenize(self, text: str, min_freq: int = 1) -> Tuple[torch.Tensor, torch.Tensor, int, callable, callable]:
        # Add special tokens to vocabulary
        chars = sorted(list(set(text)))
        vocab_size = len(chars) + len(self.special_tokens)
        
        # Create token mappings
        self.stoi = {**self.special_tokens, **{ch: i + len(self.special_tokens) for i, ch in enumerate(chars)}}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        
        # Create encode and decode functions
        def encode(s: str) -> List[int]:
            return [self.stoi.get(c, self.stoi['<unk>']) for c in s]
        
        def decode(l: List[int]) -> str:
            return ''.join([self.itos.get(i, '<unk>') for i in l])
        
        # Tokenize text
        try:
            data = torch.tensor(encode(text), dtype=torch.long)
            
            # Split into train and validation sets
            n = int(0.9 * len(data))
            train_data, val_data = data[:n], data[n:]
            
            logger.info(f"Successfully tokenized text. Vocabulary size: {vocab_size}")
            return train_data, val_data, vocab_size, encode, decode
            
        except Exception as e:
            logger.error(f"Error during tokenization: {str(e)}")
            raise

class Block(nn.Module):
    """Transformer block with self-attention and feed-forward layers"""
    
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer"""
    
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                           .view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Calculate query, key, values for all heads in batch
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        # Apply softmax and dropout
        att = F.softmax(scores, dim=-1)
        att = self.dropout(att)
        
        # Apply attention to values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.proj(y)
        return y

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
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape

        # Get token and position embeddings
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, 
                top_k: Optional[int] = None, top_p: Optional[float] = None,
                repetition_penalty: float = 1.0) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            idx: Input tensor of shape (B, T)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, only sample from tokens with cumulative probability <= p
            repetition_penalty: Penalty for repeating tokens (higher = less repetition)
        """
        self.eval()
        for _ in tqdm(range(max_new_tokens), desc="Generating"):
            # Crop context if needed
            idx_cond = idx[:, -self.config.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            for i in range(logits.size(0)):
                for previous_token in idx_cond[i]:
                    logits[i, previous_token] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_values[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

    def save_model(self, path: str):
        """Save model state dict"""
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model state dict"""
        self.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")

class GPTTrainer:
    """Trainer for GPT model"""

    def __init__(self, model: GPTLanguageModel, train_data: torch.Tensor, val_data: torch.Tensor, 
                 config: ModelConfig, learning_rate: float = 3e-4, batch_size: int = 64,
                 gradient_clip: float = 1.0, warmup_steps: int = 1000):
        self.model = model.to(device)
        self.train_loader = DataLoader(TextDataset(train_data, config.block_size), 
                                     batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(TextDataset(val_data, config.block_size), 
                                   batch_size=batch_size)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Calculate total steps and warmup percentage
        self.total_steps = len(self.train_loader)  # steps per epoch
        self.pct_start = min(warmup_steps / (self.total_steps * 5), 0.3)  # cap at 30%, consider all epochs
        
        # Initialize scheduler with default 5 epochs, will be updated in train()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=self.total_steps * 5,  # Default to 5 epochs
            pct_start=self.pct_start,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1e4,
            three_phase=False
        )
        self.gradient_clip = gradient_clip
        self.best_val_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0

    def train(self, max_epochs: int, save_dir: str = 'checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        # Update scheduler with correct total steps
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            total_steps=self.total_steps * max_epochs,
            pct_start=self.pct_start,  # Use stored pct_start value
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1e4,
            three_phase=False
        )
        
        for epoch in range(max_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")
            
            try:
                for xb, yb in progress_bar:
                    xb, yb = xb.to(device), yb.to(device)
                    
                    # Forward pass
                    logits, loss = self.model(xb, yb)
                    
                    # Backward pass with gradient clipping
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                avg_loss = total_loss / len(self.train_loader)
                logger.info(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")
                
                # Validation
                val_loss = self._validate()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(save_dir, epoch, val_loss)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                    
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                raise

    def _save_checkpoint(self, save_dir: str, epoch: int, val_loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }
        path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0

        try:
            for xb, yb in self.val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, loss = self.model(xb, yb)
                total_loss += loss.item()

            avg_loss = total_loss / len(self.val_loader)
            logger.info(f"Validation Loss: {avg_loss:.4f}")
            return avg_loss
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise
