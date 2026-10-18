"""Слой ModernBERT: RoPE, GeGLU, без bias, Pre-LN, глобальное и локальное attention через слой.

Эталон. Имена параметров — как в Hugging Face ModernBertModel: его state_dict загружается без переименований.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, rope_cos_sin       # RoPE из C03: раскладка «половинами», как в HF
from llmlab.core.modernbert import ModernBertConfig          # размеры, окно, шаг глобальных слоёв, θ


def layer_types(num_layers: int, global_every: int) -> list[str]:
    """Тип attention каждого слоя: слой 0 и каждый global_every-й — глобальный, остальные — локальные."""
    return ["full_attention" if i % global_every == 0 else "sliding_attention" for i in range(num_layers)]


def local_window_mask(T: int, window: int, device=None) -> torch.Tensor:
    """[T, T] bool: запрос i видит ключ j, если |i − j| ≤ window // 2 (окно window = 128 → по 64 в каждую сторону)."""
    pos = torch.arange(T, device=device)
    return (pos[:, None] - pos[None, :]).abs() <= window // 2


class ModernBertEmbeddings(nn.Module):
    """Только эмбеддинги токенов и LayerNorm без bias: позиции добавит RoPE в каждом слое."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertMLP(nn.Module):
    """GeGLU: Wi даёт сразу две половины [вход | gate], выход Wo(GELU(вход) · gate). Без bias."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.Wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(inp) * gate)


class ModernBertAttention(nn.Module):
    """Двунаправленное attention с RoPE: одна матрица Wqkv, без bias; θ RoPE зависит от типа слоя."""

    def __init__(self, cfg: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.attention_type = layer_types(cfg.num_hidden_layers, cfg.global_attn_every_n_layers)[layer_idx]
        self.rope_theta = cfg.global_rope_theta if self.attention_type == "full_attention" else cfg.local_rope_theta
        self.Wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.Wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, T, C], mask — bool, транслируется к [B, H, T, T] (True = можно смотреть) → [B, T, C]."""
        B, T, C = x.shape
        qkv = self.Wqkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                                 # три тензора [B, H, T, D]
        cos, sin = rope_cos_sin(torch.arange(T, device=x.device), self.head_dim, self.rope_theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        if mask is not None:
            # finfo.min, а не −inf: у PAD-запроса локальное окно может целиком состоять из PAD → softmax(−inf…) = nan
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        out = scores.softmax(-1) @ v
        return self.Wo(out.transpose(1, 2).reshape(B, T, C))


class ModernBertEncoderLayer(nn.Module):
    """Pre-LN: x + attn(attn_norm(x)) → x + mlp(mlp_norm(x)). У слоя 0 attn_norm = Identity: вход уже нормирован."""

    def __init__(self, cfg: ModernBertConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = (nn.Identity() if layer_idx == 0
                          else nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False))
        self.attn = ModernBertAttention(cfg, layer_idx)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.mlp = ModernBertMLP(cfg)
        self.attention_type = self.attn.attention_type

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


class ModernBertModel(nn.Module):
    """Эмбеддинги → слои (глобальные и локальные вперемежку) → final_norm: [B, T] → [B, T, C]."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = ModernBertEmbeddings(cfg)
        self.layers = nn.ModuleList(ModernBertEncoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pad = attention_mask.bool()[:, None, None, :]                       # [B, 1, 1, T]: PAD-ключи закрыты
        masks = {"full_attention": pad,
                 "sliding_attention": pad & local_window_mask(T, self.cfg.local_attention, input_ids.device)}
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, masks[layer.attention_type])
        return self.final_norm(x)


class ModernBertForMaskedLM(nn.Module):
    """ModernBERT + MLM-голова: dense → GELU → LayerNorm → decoder (матрица = tok_embeddings, свой bias)."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        self.model = ModernBertModel(cfg)
        self.head = nn.Module()
        self.head.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.head.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.decoder.weight = self.model.embeddings.tok_embeddings.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.model(input_ids, attention_mask)
        return self.decoder(self.head.norm(F.gelu(self.head.dense(h))))
