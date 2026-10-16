"""C22: гибридный стек 3 : 1 (Gated DeltaNet : gated attention), как в Qwen3-Next, и память кеша.

Реализуй:
1. hybrid_layer_types — расписание слоёв по правилу Qwen3-Next;
2. HybridBlock.forward — Pre-Norm блок, которому всё равно, какой у него mixer;
3. HybridStack.forward — блоки по порядку + финальная норма;
4. hybrid_cache_bytes — сколько памяти нужно на инференсе: KV-cache full-слоёв + состояния linear-слоёв.

GatedAttention (C19) и GatedDeltaNet (C21) — готовые эталоны из llmlab.core.efficient.
"""

import torch
import torch.nn as nn

from llmlab.core.efficient import GatedAttention, GatedDeltaNet   # эталоны C19, C21
from llmlab.core.llama import RMSNorm, SwiGLUFFN                   # эталоны C01, C02


def hybrid_layer_types(n_layers: int, full_attention_interval: int) -> list[str]:
    """Qwen3-Next: слой i — "full_attention", если (i + 1) делится на interval, иначе "linear_attention"."""
    # TODO: список из n_layers строк. Слой i (с нуля) — full, если (i + 1) делится на interval.
    #       Почему (i + 1), а не i? Какой слой группы из 4 будет full?
    raise NotImplementedError("hybrid_layer_types")


class HybridBlock(nn.Module):
    """Pre-Norm блок: x + mixer(norm(x)), затем x + ffn(norm(x)). Mixer — GatedAttention или GatedDeltaNet."""

    def __init__(self, mixer: nn.Module, d_model: int, d_ff: int):
        super().__init__()                                  # конструктор не меняй
        self.input_layernorm = RMSNorm(d_model)
        self.mixer = mixer
        self.post_attention_layernorm = RMSNorm(d_model)
        self.mlp = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 1: h = mixer(input_layernorm(x)). GatedAttention возвращает тензор, GatedDeltaNet — (y, state).
        # TODO 2: residual, затем x + mlp(post_attention_layernorm(x)).
        raise NotImplementedError("HybridBlock.forward")


class HybridStack(nn.Module):
    """Стек из n_layers блоков: каждый interval-й — gated attention (C19), остальные — Gated DeltaNet (C21)."""

    def __init__(self, d_model: int, n_layers: int, full_attention_interval: int, d_ff: int,
                 n_heads: int, n_kv_heads: int, head_dim: int,
                 linear_heads: int, linear_k_dim: int, linear_v_dim: int):
        super().__init__()                                  # конструктор не меняй
        self.layer_types = hybrid_layer_types(n_layers, full_attention_interval)
        self.layers = nn.ModuleList()
        for kind in self.layer_types:
            mixer = (GatedAttention(d_model, n_heads, n_kv_heads, head_dim) if kind == "full_attention"
                     else GatedDeltaNet(d_model, linear_heads, linear_k_dim, linear_v_dim))
            self.layers.append(HybridBlock(mixer, d_model, d_ff))
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: все блоки по порядку, в конце self.norm.
        raise NotImplementedError("HybridStack.forward")


def hybrid_cache_bytes(layer_types: list[str], seq_len: int, n_kv_heads: int, head_dim: int,
                       linear_heads: int, linear_k_dim: int, linear_v_dim: int,
                       batch: int = 1, bytes_per_elem: int = 2) -> int:
    """Память инференса: KV-cache full-слоёв (растёт с seq_len) + состояния linear-слоёв (константа)."""
    # TODO 1: сколько full- и linear-слоёв.
    # TODO 2: full-слой хранит K и V: 2 · n_kv_heads · head_dim чисел на КАЖДЫЙ токен контекста.
    # TODO 3: linear-слой хранит одно состояние S [linear_heads, dk, dv] — от seq_len не зависит.
    # TODO 4: сумма × batch × bytes_per_elem.
    raise NotImplementedError("hybrid_cache_bytes")
