import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# B: Batch dimension
# T: Time dimension
# C: Channel dimetion IE vocab_size


@dataclass
class ModelConfig:
    device: str
    vocab_size: int
    batch_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float


model_config = ModelConfig(
    "cuda" if torch.cuda.is_available() else "cpu", 24 * 2 + 10, 64, 256, 384, 6, 6, 0.2
)


class SelfAttention(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(model_config.n_embd, head_size, bias=False)
        self.query = nn.Linear(model_config.n_embd, head_size, bias=False)
        self.value = nn.Linear(model_config.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(model_config.block_size, model_config.block_size)),
        )
        self.hs_sqrt = head_size**-0.5

        self.dropout = nn.Dropout(model_config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given a specific context, it will reduce the context to head_size based on its estimations on how important each token of information in the context is.
        """
        _, T, _ = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)

        wei = (
            q @ k.transpose(-2, -1) * self.hs_sqrt
        )  # (B, T, 16) @ (B, 16, T) -> (B, T, T)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T), Restricts the access to only the past context (remove for sentiment analysis ?)  # type: ignore
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)


class MultiSelfAttention(nn.Module):
    def __init__(self, num_heads: int, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(model_config.n_embd, model_config.n_embd)
        self.dropout = nn.Dropout(model_config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Simple linear layer followed by a non-linearity
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_config.n_embd, 4 * model_config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * model_config.n_embd, model_config.n_embd),
            nn.Dropout(model_config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """

    def __init__(self):
        # n_emb: embedding dimension, n_head: the desired number of heads
        super().__init__()
        head_size = model_config.n_embd // model_config.n_head
        self.sa = MultiSelfAttention(model_config.n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(model_config.n_embd)
        self.ln2 = nn.LayerNorm(model_config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            model_config.vocab_size, model_config.n_embd
        )
        self.position_embedding_table = nn.Embedding(
            model_config.block_size, model_config.n_embd
        )
        self.blocks = nn.Sequential(*[Block() for _ in range(model_config.n_layer)])
        self.ln_f = nn.LayerNorm(model_config.n_embd)  # Final layer norm
        self.lm_head = nn.Linear(model_config.n_embd, model_config.vocab_size)

        self.device = model_config.device
        self.ctx_len = model_config.block_size

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None):  # type: ignore
        B, T = inputs.shape
        # inputs and target are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(inputs)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs: torch.Tensor, max_new_tokens: int):
        # inputs (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context to ctx_len
            inputs_cond = inputs[:, -self.ctx_len :]
            # Get predictions
            logits, _ = self(inputs_cond)
            # Focus only on the last timstep
            logits = logits[:, -1, :]  # Becomes (B, C)
            # Apply softmax to get probs
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample the distribution
            inputs_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append the new token to the input context
            inputs = torch.cat((inputs, inputs_next), dim=1)  # (B, T+1)
            yield ([int(inputs[-1][-1])])
