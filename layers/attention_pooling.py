import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x, attention_mask=None):  # x: (B, L, D), attention_mask: (B, L)
        self.attn_scores = self.attn(x).squeeze(-1)  # (B, L)

        if attention_mask is not None:
            # Mask out the padded tokens by setting them to a large negative value
            self.attn_scores = self.attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        self.attn_weights = torch.softmax(self.attn_scores, dim=1)  # (B, L)
        self.attn_weights = self.attn_weights.unsqueeze(-1)         # (B, L, 1)

        return (x * self.attn_weights).sum(dim=1)  # (B, D)

    def get_attention_weights(self):
        return self.attn_weights.detach().squeeze()  # (B, L)