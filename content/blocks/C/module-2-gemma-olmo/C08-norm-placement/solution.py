"""Эталон C08: четыре способа поставить нормализацию в блок трансформера."""

import torch
import torch.nn as nn

from llmlab.core.llama import LlamaAttention, RMSNorm, SwiGLUFFN   # эталоны модуля 1


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
        x = self.post_attention_layernorm(x + self.self_attn(x))
        return self.post_feedforward_layernorm(x + self.mlp(x))


class PreNormBlock(Sublayers):
    """GPT-2 / Llama: норма на входе подслоя, остаточный поток не нормируется."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_hidden, **kw):
        super().__init__(d_model, n_heads, n_kv_heads, d_hidden,
                         ["input_layernorm", "post_attention_layernorm"], **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))   # имя из HF Llama: это норма ПЕРЕД FFN


class OlmoPostNormBlock(Sublayers):
    """OLMo 2: норма на ВЫХОДЕ подслоя, но внутри остаточной ветки."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_hidden, **kw):
        super().__init__(d_model, n_heads, n_kv_heads, d_hidden,
                         ["post_attention_layernorm", "post_feedforward_layernorm"], **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.post_attention_layernorm(self.self_attn(x))
        return x + self.post_feedforward_layernorm(self.mlp(x))


class SandwichNormBlock(Sublayers):
    """Gemma 2/3: норма и на входе, и на выходе подслоя; остаток не нормируется."""

    def __init__(self, d_model, n_heads, n_kv_heads, d_hidden, **kw):
        super().__init__(d_model, n_heads, n_kv_heads, d_hidden,
                         ["input_layernorm", "post_attention_layernorm",
                          "pre_feedforward_layernorm", "post_feedforward_layernorm"], **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x)))
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
