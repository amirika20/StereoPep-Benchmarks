import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

class RotarySelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self._last_attn_weights = None

    def forward(self, x, mask=None):
        B, L, D = x.size()
        qkv = self.qkv_proj(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = self.apply_rope(q)
        k = self.apply_rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        self._last_attn_weights = attn_weights.detach()
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(attn_out)

    def apply_rope(self, x):
        B, L, H, D_h = x.shape
        theta = 10000 ** (-torch.arange(0, D_h, 2, device=x.device).float() / D_h)
        pos = torch.arange(L, device=x.device).float()
        freqs = torch.einsum('l,d->ld', pos, theta)

        sin = freqs.sin()[None, :, None, :]
        cos = freqs.cos()[None, :, None, :]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = RotarySelfAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class BaseTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.attn_pool = nn.Linear(hidden_dim, 1)

    def forward_layers(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x

    def apply_pooling(self, x, mask):
        attn_weights = self.attn_pool(x).squeeze(-1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)
        pooled = torch.sum(attn_weights * x, dim=1)
        return pooled

    def attention_weights(self, x, mask=None):
        _ = self.forward(x, mask)  # Run a forward pass to populate attention
        attn = self.encoder_layers[-1].attn._last_attn_weights
        return attn.mean(dim=1).squeeze(0)  # (L, L)
        # return attn.cpu().numpy()  # (L, L)
            

class RoPETransformerEncoder(BaseTransformerEncoder):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.forward_layers(x, mask)
        return x

class SinTransformerEncoder(BaseTransformerEncoder):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.1, max_len=512):
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.register_buffer("pos_encoding", self._get_sin_positional_encoding(max_len, hidden_dim))

    def _get_sin_positional_encoding(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.forward_layers(x, mask)
        return x

class LearnableTransformerEncoder(BaseTransformerEncoder):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.1, max_len=512):
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.forward_layers(x, mask)
        return x
