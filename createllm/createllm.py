import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import torchvision.transforms as transforms
import dill as pickle
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
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

class TextFileProcessor:
    """Handles text file processing and tokenization for GPT model training"""
    
    def __init__(self, file_path: str):
        """
        Initialize the text processor
        
        Args:
            file_path: Path to the input text file
        """
        self.file_path = file_path
        self.chars = None
        self.vocab_size = None
        self.stoi = None
        self.itos = None

    def read_file(self) -> Optional[str]:
        """Read and return file content with proper error handling"""
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

    def process_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, int, callable, callable]:
        """
        Process text data for model training
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple containing:
            - Training data tensor
            - Validation data tensor
            - Vocabulary size
            - Encoding function
            - Decoding function
        """
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create character mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # Create encoding and decoding transforms
        encode = transforms.Compose([
            transforms.Lambda(lambda s: [self.stoi.get(c, 0) for c in s])
        ])
        decode = transforms.Compose([
            transforms.Lambda(lambda l: ''.join([self.itos.get(i, '') for i in l]))
        ])

        # Prepare data splits
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        
        return train_data, val_data, self.vocab_size, encode, decode

class ThreadedDataLoader:
    """Handles multi-threaded data loading and batch preparation"""
    
    def __init__(self, batch_size: int, block_size: int, num_workers: int = 4):
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def get_batch(self, split: str, train_data: torch.Tensor, val_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data using multiple threads"""
        data = train_data if split == 'train' else val_data
        futures = []
        
        for _ in range(self.batch_size):
            futures.append(self.executor.submit(self._get_single_item, data))
            
        x_list = []
        y_list = []
        for future in futures:
            x, y = future.result()
            x_list.append(x)
            y_list.append(y)
            
        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)
        return x, y

    def _get_single_item(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single training item"""
        i = torch.randint(len(data) - self.block_size, (1,)).item()
        x = data[i:i + self.block_size]
        y = data[i + 1:i + self.block_size + 1]
        return x, y

class GPTLanguageModel(nn.Module):
    """Enhanced GPT Language Model with improved architecture"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(config.n_embd, config.n_head, config.block_size, config.dropout)
            for _ in range(config.n_layer)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Calculate number of parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Number of parameters: {self.n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with improved attention mechanism"""
        B, T = idx.shape
        
        # Get embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        # Apply transformer blocks with residual connections
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Generate text with temperature-based sampling"""
        for _ in tqdm(range(max_new_tokens), desc="Generating"):
            # Get predictions
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append new token
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

class ModelManager:
    """Handles model saving and loading operations"""
    
    @staticmethod
    def save_model(model: GPTLanguageModel, path: str, encode: callable, decode: callable):
        """Save model and associated data"""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        
        # Save encoder and decoder
        with open(os.path.join(path, "encoder.pickle"), "wb") as f:
            pickle.dump(encode, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, "decoder.pickle"), "wb") as f:
            pickle.dump(decode, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Save config
        config = {
            "vocab_size": model.config.vocab_size,
            "n_embd": model.config.n_embd,
            "block_size": model.config.block_size,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "dropout": model.config.dropout
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
            
        logger.info(f"Model saved successfully at: {path}")

    @staticmethod
    def load_model(path: str) -> Tuple[GPTLanguageModel, callable, callable]:
        """Load model and associated data"""
        # Load config
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create and load model
        model = GPTLanguageModel(ModelConfig(**config))
        model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=device))
        
        # Load encoder and decoder
        with open(os.path.join(path, "encoder.pickle"), "rb") as f:
            encode = pickle.load(f)
        with open(os.path.join(path, "decoder.pickle"), "rb") as f:
            decode = pickle.load(f)
            
        return model, encode, decode

class GPTTrainer:
    """Enhanced trainer with multi-threading support and progress tracking"""
    
    def __init__(self, config: ModelConfig, learning_rate: float = 3e-4,
                 batch_size: int = 64, max_iters: int = 5000,
                 eval_interval: int = 500, eval_iters: int = 200,
                 num_workers: int = 4):
        self.config = config
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.data_loader = ThreadedDataLoader(batch_size, config.block_size, num_workers)

    def train(self, train_data: torch.Tensor, val_data: torch.Tensor) -> GPTLanguageModel:
        """Train the model with progress tracking and logging"""
        model = GPTLanguageModel(self.config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        
        progress_bar = tqdm(range(self.max_iters), desc="Training")
        for iter_num in progress_bar:
            # Evaluation
            if iter_num % self.eval_interval == 0:
                losses = self._evaluate_model(model, train_data, val_data)
                progress_bar.set_postfix({
                    'train_loss': f"{losses['train']:.4f}",
                    'val_loss': f"{losses['val']:.4f}"
                })
            
            # Training step
            xb, yb = self.data_loader.get_batch('train', train_data, val_data)
            logits, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
        return model

    @torch.no_grad()
    def _evaluate_model(self, model: GPTLanguageModel, train_data: torch.Tensor,
                       val_data: torch.Tensor) -> Dict[str, float]:
        """Evaluate model on training and validation sets"""
        model.eval()
        losses = {}
        
        for split in ['train', 'val']:
            losses[split] = torch.mean(torch.tensor([
                self._compute_loss(model, split, train_data, val_data)
                for _ in range(self.eval_iters)
            ])).item()
            
        model.train()
        return losses

    def _compute_loss(self, model: GPTLanguageModel, split: str,
                     train_data: torch.Tensor, val_data: torch.Tensor) -> float:
        """Compute loss for a single batch"""
        x, y = self.data_loader.get_batch(split, train_data, val_data)
        _, loss = model(x, y)
        return loss.item()

