"""Готовые компоненты DeepSeek (модуль 3 блока C, часть 2) — эталоны задач C16, C16b, C17.

C16 Multi-head Latent Attention (латентный KV-cache и «поглощение» весов), C16b MLA с decoupled RoPE,
C17 Multi-Token Prediction. Поздние задачи и финальная модель (Фаза 8) импортируют их отсюда.

⚠️ Это готовые ответы на задачи C16–C17. Если ещё не решил их — не подглядывай.

Код намеренно совпадает с ``content/blocks/C/module-3-moe/C16*/solution.py`` и ``C17-mtp/solution.py``;
расхождение ловит ``tests/test_core_mla.py``. Имена параметров — как в Hugging Face ``transformers``
(``modeling_deepseek_v3.py``: ``kv_a_proj_with_mqa``, ``kv_a_layernorm``, ``kv_b_proj``) и в vLLM
(``deepseek_mtp.py``: ``enorm``, ``hnorm``, ``eh_proj``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.gemma_olmo import PreNormBlock
from llmlab.core.llama import RMSNorm, apply_rope, rope_cos_sin

# ---------------------------------------------------------------------------- C16 MLA


class LatentCache:
    """KV-cache MLA: латенты c [B, S, r] (и, в C16b, общие RoPE-ключи [B, S, d_r]) вместо K и V."""

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
        q, c = self._q_and_latent(x, cache)
        S = c.shape[1]
        kv = self.kv_b_proj(c).view(B, S, H, d + dv).transpose(1, 2)                  # [B, H, S, d + dv]
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


# ---------------------------------------------------------------------------- C16b MLA + decoupled RoPE


class MLARopeAttention(nn.Module):
    """MLA DeepSeek-V2/V3 (без сжатия запроса): часть каждой головы — без позиций (из латента),
    часть — RoPE с ОДНИМ общим на все головы ключом k_rope."""

    def __init__(self, d_model: int, n_heads: int, kv_lora_rank: int, qk_nope_head_dim: int,
                 qk_rope_head_dim: int, v_head_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, n_heads * (qk_nope_head_dim + qk_rope_head_dim), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim), bias=False)
        self.o_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: LatentCache | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, dn, dr, dv, r = (self.n_heads, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim,
                            self.kv_lora_rank)
        start = cache.seq_len if cache is not None else 0
        cos, sin = rope_cos_sin(torch.arange(start, start + T, device=x.device), dr, self.rope_theta)
        q = self.q_proj(x).view(B, T, H, dn + dr).transpose(1, 2)
        q_nope, q_rope = q.split([dn, dr], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)                                         # [B, H, T, dr]
        c, k_rope = self.kv_a_proj_with_mqa(x).split([r, dr], dim=-1)                 # [B, T, r], [B, T, dr]
        c = self.kv_a_layernorm(c)
        k_rope = apply_rope(k_rope.unsqueeze(1), cos, sin).squeeze(1)                 # одна «голова» на всех
        if cache is not None:
            c, k_rope = cache.update(c, k_rope)                                       # [B, S, r], [B, S, dr]
        S = c.shape[1]
        kv = self.kv_b_proj(c).view(B, S, H, dn + dv).transpose(1, 2)
        k_nope, v = kv.split([dn, dv], dim=-1)                                        # [B, H, S, dn], [B, H, S, dv]
        k = torch.cat((k_nope, k_rope.unsqueeze(1).expand(B, H, S, dr)), dim=-1)      # [B, H, S, dn + dr]
        q = torch.cat((q_nope, q_rope), dim=-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dn + dr)
        scores = scores.masked_fill(~causal_mask(T, S, x.device), float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))


def mla_kv_cache_bytes(n_layers: int, kv_lora_rank: int, qk_rope_head_dim: int, seq_len: int,
                       batch: int = 1, bytes_per_elem: int = 2) -> int:
    """KV-cache MLA: на токен и слой — латент r и общий RoPE-ключ d_r (без множителя 2 за K и V)."""
    return n_layers * (kv_lora_rank + qk_rope_head_dim) * seq_len * batch * bytes_per_elem


# ---------------------------------------------------------------------------- C17 MTP


class MTPModule(nn.Module):
    """Один модуль Multi-Token Prediction DeepSeek-V3: смешать h предыдущей глубины и эмбеддинг
    следующего токена, прогнать через блок трансформера."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_hidden: int):
        super().__init__()
        self.enorm = RMSNorm(d_model)
        self.hnorm = RMSNorm(d_model)
        self.eh_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.block = PreNormBlock(d_model, n_heads, n_kv_heads, d_hidden)
        self.norm = RMSNorm(d_model)                     # норма перед общей выходной головой

    def forward(self, h_prev: torch.Tensor, emb_next: torch.Tensor) -> torch.Tensor:
        """h_prev, emb_next [B, T', C] → h [B, T', C]. Порядок конкатенации — как в vLLM: [emb; h]."""
        x = self.eh_proj(torch.cat((self.enorm(emb_next), self.hnorm(h_prev)), dim=-1))
        return self.block(x)


def mtp_logits(h_main: torch.Tensor, tokens: torch.Tensor, embed: nn.Embedding, lm_head: nn.Module,
               modules: nn.ModuleList) -> list[torch.Tensor]:
    """h_main [B, T, C] (выход основной модели), tokens [B, T] → логиты глубин k = 1..D, [B, T−k, V].

    Позиция i глубины k видит токены ≤ i + k и предсказывает токен i + k + 1.
    """
    T = tokens.shape[1]
    h, out = h_main, []
    for k, module in enumerate(modules, start=1):
        h = module(h[:, : T - k], embed(tokens[:, k:]))                  # [B, T−k, C]
        out.append(lm_head(module.norm(h)))
    return out


def mtp_loss(main_logits: torch.Tensor, mtp_logits_list: list[torch.Tensor], tokens: torch.Tensor, lam: float):
    """→ (L_main + λ/D · Σ L_k, L_main, [L_k]). Основная голова: i → i+1; глубина k: i → i+k+1."""
    V = main_logits.shape[-1]
    l_main = F.cross_entropy(main_logits[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1))
    l_mtp = [F.cross_entropy(lg[:, :-1].reshape(-1, V), tokens[:, k + 1:].reshape(-1))
             for k, lg in enumerate(mtp_logits_list, start=1)]
    D = len(l_mtp)
    total = l_main + (lam / D) * sum(l_mtp) if D else l_main
    return total, l_main, l_mtp
