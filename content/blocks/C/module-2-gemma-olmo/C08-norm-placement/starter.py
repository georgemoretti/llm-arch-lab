"""C08: четыре способа поставить нормализацию в блок трансформера.

Реализуй forward у четырёх блоков. Attention и FFN (из модуля 1) и все нормы уже созданы в
конструкторах — нужно только решить, ЧТО нормировать и куда прибавлять остаток:

  PostNormBlock      — оригинальный Transformer (2017);
  PreNormBlock       — GPT-2, Llama (это блок из C06);
  OlmoPostNormBlock  — OLMo 2;
  SandwichNormBlock  — Gemma 2 / Gemma 3.

Подслои: self.self_attn(x) и self.mlp(x), оба [B, T, C] → [B, T, C].
"""

import torch
import torch.nn as nn

from llmlab.core.llama import LlamaAttention, RMSNorm, SwiGLUFFN   # эталоны модуля 1 (готовые)


class Sublayers(nn.Module):
    """Общая часть всех блоков: attention и FFN из модуля 1 + нормы с заданными именами."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_hidden: int, norms: list[str],
                 eps: float = 1e-6, rope_theta: float = 10000.0):
        super().__init__()
        self.self_attn = LlamaAttention(d_model, n_heads, n_kv_heads, rope_theta)
        self.mlp = SwiGLUFFN(d_model, d_hidden)
        for name in norms:
            setattr(self, name, RMSNorm(d_model, eps))


class PostNormBlock(Sublayers):
    """Оригинальный Transformer (2017): норма ПОСЛЕ сложения с остатком."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_hidden, **kw):
        super().__init__(d_model, n_heads, n_kv_heads, d_hidden,
                         ["post_attention_layernorm", "post_feedforward_layernorm"], **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: норма применяется к СУММЕ «вход + выход подслоя». Сначала attention, потом FFN.
        raise NotImplementedError("PostNormBlock.forward")


class PreNormBlock(Sublayers):
    """GPT-2 / Llama: норма на входе подслоя, остаточный поток не нормируется."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_hidden, **kw):
        super().__init__(d_model, n_heads, n_kv_heads, d_hidden,
                         ["input_layernorm", "post_attention_layernorm"], **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: как в C06: подслой получает нормированный вход, к остатку прибавляется его выход.
        #       Внимание: у HF Llama норма перед FFN называется post_attention_layernorm.
        raise NotImplementedError("PreNormBlock.forward")


class OlmoPostNormBlock(Sublayers):
    """OLMo 2: норма на ВЫХОДЕ подслоя, но внутри остаточной ветки."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_hidden, **kw):
        super().__init__(d_model, n_heads, n_kv_heads, d_hidden,
                         ["post_attention_layernorm", "post_feedforward_layernorm"], **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: подслой получает вход БЕЗ нормы; нормируется его ВЫХОД, и уже он прибавляется к остатку.
        raise NotImplementedError("OlmoPostNormBlock.forward")


class SandwichNormBlock(Sublayers):
    """Gemma 2/3: норма и на входе, и на выходе подслоя; остаток не нормируется."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_hidden, **kw):
        super().__init__(d_model, n_heads, n_kv_heads, d_hidden,
                         ["input_layernorm", "post_attention_layernorm",
                          "pre_feedforward_layernorm", "post_feedforward_layernorm"], **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: обе нормы на каждый подслой: input_layernorm → attention → post_attention_layernorm,
        #       pre_feedforward_layernorm → mlp → post_feedforward_layernorm. Где остаток?
        raise NotImplementedError("SandwichNormBlock.forward")
