"""Эталон «Найди баг: GQA и MQA» (исправленная версия): GQA, копии KV-голов идут подряд."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Размножает KV-головы для групп запросов.

    x: [B, H_kv, T, D] → [B, H_kv · n_rep, T, D]; голова запроса h использует KV-голову h // n_rep,
    то есть копии одной KV-головы идут подряд: [k0, k0, k1, k1, …] при n_rep = 2.
    """
    if n_rep == 1:
        return x
    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, T, D)        # вид без копирования памяти
    return x.reshape(B, H_kv * n_rep, T, D)                      # здесь копии и появляются


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int,
                   batch: int = 1, bytes_per_elem: int = 2) -> int:
    """Размер KV-cache в байтах: K и V на каждом слое для каждой KV-головы и каждого токена."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model должно делиться на n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads должно делиться на n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads          # сколько голов запросов делят одну KV-голову
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)   # в n_rep раз уже
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)       # [B, H, T, D]
        k = self.k_proj(x).view(B, T, H_kv, D).transpose(1, 2)    # [B, H_kv, T, D]
        v = self.v_proj(x).view(B, T, H_kv, D).transpose(1, 2)    # [B, H_kv, T, D]
        k = repeat_kv(k, self.n_rep)                              # [B, H, T, D]
        v = repeat_kv(v, self.n_rep)

        scores = q @ k.transpose(-2, -1) / math.sqrt(D)           # [B, H, T, T]
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v                       # [B, H, T, D]
        out = out.transpose(1, 2).reshape(B, T, H * D)            # [B, T, H·D]
        return self.o_proj(out)
