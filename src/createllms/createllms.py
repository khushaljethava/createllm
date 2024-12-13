import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import torchvision.transforms as transforms
import dill as pickle
import json


torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class EstimatedLossOfData:



class TextFileProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text
        except FileNotFoundError:
            print(f"File '{self.file_path}' not found.")
            return None

    def process_text(self, text):
        # Implement your text processing logic here
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }

        encode = transforms.Compose([transforms.Lambda(lambda s: [stoi[c] for c in s])])
        decode = transforms.Compose([transforms.Lambda(lambda l: ''.join([itos[i] for i in l]))])
        # encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        # decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
        # Train and test splits
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9*len(data)) # first 90% will be train, rest val
        train_data = data[:n]
        val_data = data[n:]
        return train_data,val_data,vocab_size,encode,decode

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size,n_embd,block_size,dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size,n_embd,block_size,dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size,n_embd,block_size,dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head,block_size,dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size,n_embd,block_size,dropout=dropout)
        self.ffwd = FeedFoward(n_embd,dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self,vocab_size,n_embd = 384,block_size= 256,n_layer = 4,n_head = 4 ,dropout=0.2):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head,block_size=block_size,dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens,block_size=256):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
  
class LLMModel:

  def __init__(self,Path):
    if not os.path.exists(Path):
        print("Folder does not exist.")
    else:
        # Initialize variables to store the loaded data
        decoder_data = None
        model_data = None
        encoder_data = None
        config_data = None

        # Load 'config.json' file first
        config_file_path = os.path.join(Path, 'config.json')
        if os.path.exists(config_file_path):
            with open(config_file_path, 'r') as config_file:
                config_data = json.load(config_file)
        else:
            print("config.json file not found.")

        # Then load the other expected files
        expected_files = ['decoder.pickle', 'model.pt', 'encoder.pickle']
        for file_name in expected_files:
            file_path = os.path.join(Path, file_name)

            # Check if the file exists
            if os.path.exists(file_path):
                if file_name.endswith('.pickle'):
                    # Load pickle files
                    with open(file_path, 'rb') as file:
                        if file_name == 'decoder.pickle':
                            decoder_data = pickle.load(file)
                        elif file_name == 'encoder.pickle':
                            encoder_data = pickle.load(file)
                elif file_name == 'model.pt':
                    # Handle the model.pt file as needed
                    # You may use PyTorch's torch.load() function here
                    model = GPTLanguageModel(vocab_size=config_data["vocab_size"],n_embd = config_data["n_embd"],block_size = config_data["block_size"],n_layer=config_data["n_layer"],n_head=config_data["n_head"],dropout=config_data["dropout"])
                    model.load_state_dict(torch.load(file_path,map_location=device))
                    model = model.to(device)
            else:
                print(f"File not found: {file_name}")
 
  def generate(self,TextInput):
      context = torch.tensor(encoder_data(TextInput), dtype=torch.long, device=device)
      generated_chars = decoder_data(model.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
      return generated_chars


class GPTTrainer:
    def __init__(self,TextFile,learning_rate=3e-4,batch_size = 64,block_size = 256, max_iters = 5000, eval_interval = 500,eval_iters = 200,n_embd = 384,
    n_head = 8, n_layer = 8,dropout = 0.2,SavedPath=""):
        self.TextFile = TextFile
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.n_embd = n_embd
        self.n_head  = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.SavedPath = SavedPath

    def get_batch(self,split,train_data,val_data,block_size,batch_size):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self,model,eval_iters,train_data,val_data,block_size,batch_size):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split,train_data,val_data,block_size,batch_size)
                # print(X)
                logits, loss = model(X, Y)
                # print(logits)
                # print()
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return  out

    def SaveCreateLLmModel(self,model,path,encode,decode,configData):

      #save GPT model
      # Define the directory path
      directory_path = str(os.path.join(path,'CreateLLMModel'))
      # Check if the directory exists
      if not os.path.exists(directory_path):
          # If it doesn't exist, create it
          os.makedirs(directory_path)
          print(f"Directory '{directory_path}' created.")
      else:
          print(f"Directory '{directory_path}' already exists.")

      #GPT Model Saved
      torch.save(model.state_dict(), os.path.join(directory_path,"model.pt"))

      #save encoder
      encoder_file_path = str(os.path.join(path,'CreateLLMModel','encoder.pickle'))

      with open(encoder_file_path, 'wb') as file:
          pickle.dump(encode, file,pickle.HIGHEST_PROTOCOL)
      #save decoder
      decoder_file_path = str(os.path.join(path,'CreateLLMModel','decoder.pickle'))
      # Save the encoder to the file
      with open(decoder_file_path, 'wb') as file:
          pickle.dump(decode, file,pickle.HIGHEST_PROTOCOL)


      config_file_path = str(os.path.join(path,'CreateLLMModel','config.json'))
      with open(config_file_path, 'w') as file:
          json.dump(configData, file)
      # print(f"Model Saved at {path}")


    def trainer(self):

        DataFile = TextFileProcessor(self.TextFile)
        text = DataFile.read_file()
        if text:
            train_data,val_data,vocab_size,encode,decode = DataFile.process_text(text)
        model = GPTLanguageModel(vocab_size,self.n_embd,self.block_size,self.n_layer,self.n_head,self.dropout)
        m = model.to(device)
        print(sum(p.numel() for p in m.parameters())/1e6, 'Million Parameters')
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        # create a PyTorch optimizer
        for iter in range(self.max_iters):
            # every once in a while evaluate the loss on train and val sets
            if iter % self.eval_interval == 0 or iter == self.max_iters - 1:
                # est_loss = EstimatedLossOfData()
                losses = self.estimate_loss(m,self.eval_iters,train_data,val_data,self.block_size,self.batch_size)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # sample a batch of data
            xb, yb = self.get_batch('train',train_data,val_data,self.block_size,self.batch_size)

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        configData = {"vocab_size":vocab_size,"n_embd":self.n_embd,"block_size":self.block_size,"n_layer":self.n_layer,"n_head":self.n_head,"dropout":self.dropout}
        self.SaveCreateLLmModel(m,self.SavedPath,encode,decode,configData)
        print("Model Trained Successfully")


