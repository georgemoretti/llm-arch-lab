"""Слой ModernBERT: RoPE, GeGLU, без bias, Pre-LN, глобальное и локальное attention через слой.

Реализуй layer_types, local_window_mask и forward у ModernBertMLP, ModernBertAttention, ModernBertEncoderLayer и
ModernBertModel. Конструкторы, эмбеддинги и MLM-голова готовы: имена параметров — как в Hugging Face, тесты загружают
в модель state_dict из transformers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, rope_cos_sin       # RoPE из C03: раскладка «половинами», как в HF
from llmlab.core.modernbert import ModernBertConfig          # размеры, окно, шаг глобальных слоёв, θ


def layer_types(num_layers: int, global_every: int) -> list[str]:
    """Тип attention каждого слоя: слой 0 и каждый global_every-й — глобальный, остальные — локальные."""
    # TODO: список длины num_layers из строк "full_attention" / "sliding_attention" (имена — как в HF).
    #       Глобальные слои: 0, global_every, 2·global_every, …
    raise NotImplementedError("layer_types")


def local_window_mask(T: int, window: int, device=None) -> torch.Tensor:
    """[T, T] bool: запрос i видит ключ j, если |i − j| ≤ window // 2 (окно window = 128 → по 64 в каждую сторону)."""
    # TODO: позиции arange(T); матрица |i − j|; сравнить с ПОЛОВИНОЙ окна (окно симметричное, в обе стороны).
    raise NotImplementedError("local_window_mask")


class ModernBertEmbeddings(nn.Module):
    """Только эмбеддинги токенов и LayerNorm без bias: позиции добавит RoPE в каждом слое."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        # Готово, не меняй.
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertMLP(nn.Module):
    """GeGLU: Wi даёт сразу две половины [вход | gate], выход Wo(GELU(вход) · gate). Без bias."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        # Готово, не меняй.
        self.Wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.Wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Wi(x) → две половины по последней оси (chunk): первая — «вход», вторая — gate.
        #       Активация (точный GELU, F.gelu без tanh) — на «вход», gate остаётся линейным. Затем Wo.
        raise NotImplementedError("ModernBertMLP.forward")


class ModernBertAttention(nn.Module):
    """Двунаправленное attention с RoPE: одна матрица Wqkv, без bias; θ RoPE зависит от типа слоя."""

    def __init__(self, cfg: ModernBertConfig, layer_idx: int):
        super().__init__()
        # Готово, не меняй. attention_type и rope_theta зависят от номера слоя (нужна твоя layer_types).
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.attention_type = layer_types(cfg.num_hidden_layers, cfg.global_attn_every_n_layers)[layer_idx]
        self.rope_theta = cfg.global_rope_theta if self.attention_type == "full_attention" else cfg.local_rope_theta
        self.Wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.Wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, T, C], mask — bool, транслируется к [B, H, T, T] (True = можно смотреть) → [B, T, C]."""
        # TODO 1: Wqkv(x) → view(B, T, 3, H, D): порядок в выходе Wqkv — сначала все q, потом все k, потом все v.
        #         Разложи на q, k, v формы [B, H, T, D] (permute).
        # TODO 2: RoPE: cos, sin = rope_cos_sin(arange(T), D, self.rope_theta); повернуть q и k (apply_rope).
        # TODO 3: scores = q · kᵀ / √D. Маска: masked_fill(~mask, torch.finfo(scores.dtype).min) — не −inf
        #         (в локальном слое окно PAD-запроса может целиком состоять из PAD, и −inf дал бы nan).
        # TODO 4: softmax по ключам → @ v → склеить головы [B, T, C] → self.Wo.
        raise NotImplementedError("ModernBertAttention.forward")


class ModernBertEncoderLayer(nn.Module):
    """Pre-LN: x + attn(attn_norm(x)) → x + mlp(mlp_norm(x)). У слоя 0 attn_norm = Identity: вход уже нормирован."""

    def __init__(self, cfg: ModernBertConfig, layer_idx: int):
        super().__init__()
        # Готово, не меняй.
        self.attn_norm = (nn.Identity() if layer_idx == 0
                          else nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False))
        self.attn = ModernBertAttention(cfg, layer_idx)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.mlp = ModernBertMLP(cfg)
        self.attention_type = self.attn.attention_type

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO: два Pre-LN подслоя с остатками: сначала attention (норма attn_norm, маска mask), потом MLP (mlp_norm).
        raise NotImplementedError("ModernBertEncoderLayer.forward")


class ModernBertModel(nn.Module):
    """Эмбеддинги → слои (глобальные и локальные вперемежку) → final_norm: [B, T] → [B, T, C]."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        # Готово, не меняй.
        self.cfg = cfg
        self.embeddings = ModernBertEmbeddings(cfg)
        self.layers = nn.ModuleList(ModernBertEncoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO 1: attention_mask (1 / 0, по умолчанию единицы) → bool [B, 1, 1, T] — маска PAD-ключей.
        # TODO 2: две маски: для "full_attention" — только PAD; для "sliding_attention" — PAD И
        #         local_window_mask(T, cfg.local_attention). Слой берёт свою по layer.attention_type.
        # TODO 3: embeddings → слои → final_norm.
        raise NotImplementedError("ModernBertModel.forward")


class ModernBertForMaskedLM(nn.Module):
    """ModernBERT + MLM-голова: dense → GELU → LayerNorm → decoder (матрица = tok_embeddings, свой bias)."""

    def __init__(self, cfg: ModernBertConfig):
        super().__init__()
        # Готово, не меняй.
        self.model = ModernBertModel(cfg)
        self.head = nn.Module()
        self.head.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.head.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.decoder.weight = self.model.embeddings.tok_embeddings.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.model(input_ids, attention_mask)
        return self.decoder(self.head.norm(F.gelu(self.head.dense(h))))
