"""Multi-head Latent Attention — код из «MLA» с одной ошибкой. Найди и исправь её."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import RMSNorm   # эталон «RMSNorm»


class LatentCache:
    """KV-cache MLA: латенты c [B, S, r] (и, в «MLA + decoupled RoPE», общие RoPE-ключи [B, S, d_r]) вместо K и V."""

    def __init__(self):
        self.c: torch.Tensor | None = None
        self.k_rope: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        return 0 if self.c is None else self.c.shape[1]

    def update(self, c: torch.Tensor, k_rope: torch.Tensor | None = None):
        """Дописывает новые токены по оси времени (dim=1) → (весь c, весь k_rope или None)."""
        self.c = c if self.c is None else torch.cat((self.c, c), dim=1)
        if k_rope is not None:
            self.k_rope = k_rope if self.k_rope is None else torch.cat((self.k_rope, k_rope), dim=1)
        return self.c, self.k_rope


def causal_mask(T: int, S: int, device) -> torch.Tensor:
    return torch.ones(T, S, dtype=torch.bool, device=device).tril(diagonal=S - T)


class MLAttention(nn.Module):
    """MLA без позиционного кодирования: K и V восстанавливаются из общего латента c = norm(W_dkv x)."""

    def __init__(self, d_model: int, n_heads: int, kv_lora_rank: int, head_dim: int, v_head_dim: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim or head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.kv_a_proj = nn.Linear(d_model, kv_lora_rank, bias=False)          # сжатие: W_DKV
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(kv_lora_rank, n_heads * (head_dim + self.v_head_dim), bias=False)  # W_UK, W_UV
        self.o_proj = nn.Linear(n_heads * self.v_head_dim, d_model, bias=False)

    def _q_and_latent(self, x, cache):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)    # [B, H, T, d]
        c = self.kv_a_layernorm(self.kv_a_proj(x))                                    # [B, T, r]
        if cache is not None:
            c, _ = cache.update(c)                                                    # [B, S, r]
        return q, c

    def forward(self, x: torch.Tensor, cache: LatentCache | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, d, dv = self.n_heads, self.head_dim, self.v_head_dim
        q, c = self._q_and_latent(x, None)
        kv = self.kv_b_proj(c)                                                        # [B, T, H·(d + dv)]
        if cache is not None:
            kv, _ = cache.update(kv)                                                  # [B, S, H·(d + dv)]
        S = kv.shape[1]
        kv = kv.view(B, S, H, d + dv).transpose(1, 2)                                 # [B, H, S, d + dv]
        k, v = kv.split([d, dv], dim=-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d)
        scores = scores.masked_fill(~causal_mask(T, S, x.device), float("-inf"))
        out = F.softmax(scores, dim=-1) @ v                                           # [B, H, T, dv]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))

    def forward_absorbed(self, x: torch.Tensor, cache: LatentCache | None = None) -> torch.Tensor:
        """То же самое, но K и V не восстанавливаются: W_UK «поглощается» в запрос, W_UV — в выход."""
        B, T, _ = x.shape
        H, d, dv, r = self.n_heads, self.head_dim, self.v_head_dim, self.kv_lora_rank
        q, c = self._q_and_latent(x, cache)
        S = c.shape[1]
        W = self.kv_b_proj.weight.view(H, d + dv, r)
        W_uk, W_uv = W[:, :d], W[:, d:]                                               # [H, d, r], [H, dv, r]
        q_lat = torch.einsum("bhtd,hdr->bhtr", q, W_uk)                               # [B, H, T, r]
        scores = q_lat @ c.unsqueeze(1).transpose(-2, -1) / math.sqrt(d)              # [B, H, T, S]
        scores = scores.masked_fill(~causal_mask(T, S, x.device), float("-inf"))
        o_lat = F.softmax(scores, dim=-1) @ c.unsqueeze(1)                            # [B, H, T, r]
        out = torch.einsum("bhtr,hvr->bhtv", o_lat, W_uv)                             # [B, H, T, dv]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))
