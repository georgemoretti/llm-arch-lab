"""Эталон: упрощённое sparse attention в стиле DSA (DeepSeek-V3.2) — индексатор + top-k ключей."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightningIndexer(nn.Module):
    """Лёгкий «оценщик» пар (запрос t, ключ s): несколько маленьких голов, ReLU, взвешенная сумма.

    I(t, s) = Σ_h w_{t,h} · ReLU(q_{t,h} · k_s) / √(H_I · d_I); ключ индексатора один на токен (как MQA).
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.weights_proj = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, C] → оценки [B, T, T]; будущие ключи (s > t) = −inf."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)                 # [B, T, H_I, d_I]
        k = self.k_proj(x)                                                         # [B, T, d_I]
        w = self.weights_proj(x) / math.sqrt(self.n_heads)                         # [B, T, H_I]
        logits = torch.einsum("bthd,bsd->bths", q, k) / math.sqrt(self.head_dim)   # [B, T, H_I, S]
        scores = torch.einsum("bth,bths->bts", w, F.relu(logits))                  # [B, T, S]
        future = ~torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        return scores.masked_fill(future, float("-inf"))


def topk_mask(index_scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """index_scores [B, T, S] → bool [B, T, S]: True у top_k лучших ДОПУСТИМЫХ (не −inf) ключей строки."""
    k = min(top_k, index_scores.shape[-1])
    idx = index_scores.topk(k, dim=-1).indices                                     # [B, T, k]
    keep = torch.zeros_like(index_scores, dtype=torch.bool).scatter_(-1, idx, True)
    return keep & torch.isfinite(index_scores)          # в коротких строках topk захватывает и −inf


def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """q [B, H, T, d], k/v [B, H, S, d], keep [B, T, S] (общая для всех голов) → [B, H, T, d]."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])                      # [B, H, T, S]
    scores = scores.masked_fill(~keep.unsqueeze(1), float("-inf"))
    return F.softmax(scores, dim=-1) @ v


class DSAttention(nn.Module):
    """MHA, в которой каждый запрос смотрит только на top_k ключей, выбранных индексатором."""

    def __init__(self, d_model: int, n_heads: int, index_n_heads: int, index_head_dim: int, top_k: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.indexer = LightningIndexer(d_model, index_n_heads, index_head_dim)

    def forward(self, x: torch.Tensor, return_mask: bool = False):
        """x [B, T, C] → [B, T, C] (и маска выбранных ключей [B, T, T], если return_mask)."""
        B, T, C = x.shape
        H, d = self.n_heads, self.head_dim
        q, k, v = (p(x).view(B, T, H, d).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        keep = topk_mask(self.indexer(x), self.top_k)                              # [B, T, T]
        out = sparse_attention(q, k, v, keep)                                      # [B, H, T, d]
        out = self.o_proj(out.transpose(1, 2).reshape(B, T, C))
        return (out, keep) if return_mask else out
