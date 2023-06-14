import torch
import torch.nn as nn
from torch.nn import functional as F

# B: Batch dimension
# T: Time dimension
# C: Channel dimetion IE vocab_size

class SelfAttention(nn.Module):
    def __init__(self, n_emb: int, ctx_len: int, head_size = 16) -> None:
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(ctx_len, ctx_len)))
        self.hs_sqrt = head_size ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given a specific context, it will reduce the context to head_size based on its estimations on how important each token of information in the context is.
        """
        _, T, _ = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x)
        
        wei = q @ k.transpose(-2, -1) * self.hs_sqrt # (B, T, 16) @ (B, 16, T) -> (B, T, T)
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T), Restricts the access to only the past context (remove for sentiment analysis ?) 
        wei = F.softmax(wei, dim=-1)

        return wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
    
class MultiSelfAttention(nn.Module):
    def __init__(self, num_heads: int, n_emb: int, ctx_len: int, head_size = 16) -> None:
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(n_emb, ctx_len, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """
    Simple linear layer followed by a non-linearity 
    """
    def __init__(self, n_emb: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """

    def __init__(self, n_emb: int, n_head: int, ctx_len: int):
        # n_emb: embedding dimension, n_head: the desired number of heads
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiSelfAttention(n_head, n_emb, ctx_len, head_size)
        self.ffwd = FeedForward(n_emb)

    def forward(self, x):
        x = x + self.sa(x)
        x = X + self.ffwd(x)
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, ctx_len: int, n_emb = 32, device = 'gpu' if torch.cuda.is_available() else 'cpu') -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(ctx_len, n_emb)
        self.blocks = nn.Sequential(
            Block(n_emb, 4, ctx_len),
            Block(n_emb, 4, ctx_len),
            Block(n_emb, 4, ctx_len),
            Block(n_emb, 4, ctx_len)
        )
        self.lm_head = nn.Linear(n_emb, vocab_size)
        
        self.device = device
        self.ctx_len = ctx_len

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None):
        B, T = inputs.shape
        # inputs and target are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(inputs) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, inputs: torch.Tensor, max_new_tokens: int):
        # inputs (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context to ctx_len
            inputs_cond = inputs[:, -self.ctx_len:]
            # Get predictions
            logits, _ = self(inputs_cond)
            # Focus only on the last timstep
            logits = logits[:, -1, :] # Becomes (B, C)
            # Apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample the distribution
            inputs_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append the new token to the input context
            inputs = torch.cat((inputs, inputs_next), dim=1) # (B, T+1)
        return inputs
    