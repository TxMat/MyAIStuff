import torch
import torch.nn as nn
from torch.nn import functional as F

# B: Batch dimension
# T: Time dimension
# C: Channel dimetion IE vocab_size

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None):
        # idx and target are both (B, T) tensor of integers
        logits = self.token_embedding_table(inputs) # (B, T, C)

        if targets == None:
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
            # Get predictions
            logits, _ = self(inputs)
            # Focus only on the last timstep
            logits = logits[:, -1, :] # Becomes (B, C)
            # Apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample the distribution
            inputs_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append the new token to the input context
            inputs = torch.cat((inputs, inputs_next), dim=1) # (B, T+1)
        return inputs
