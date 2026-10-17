"""Эталон: гибридный стек 3 : 1 (Gated DeltaNet : gated attention), как в Qwen3-Next, и память кеша."""

import torch
import torch.nn as nn

from llmlab.core.efficient import GatedAttention, GatedDeltaNet   # эталоны «Gated attention», «Gated DeltaNet»
from llmlab.core.llama import RMSNorm, SwiGLUFFN                   # эталоны «RMSNorm», «SwiGLU FFN»


def hybrid_layer_types(n_layers: int, full_attention_interval: int) -> list[str]:
    """Qwen3-Next: слой i — "full_attention", если (i + 1) делится на interval, иначе "linear_attention"."""
    return ["full_attention" if (i + 1) % full_attention_interval == 0 else "linear_attention"
            for i in range(n_layers)]


class HybridBlock(nn.Module):
    """Pre-Norm блок: x + mixer(norm(x)), затем x + ffn(norm(x)). Mixer — GatedAttention или GatedDeltaNet."""

    def __init__(self, mixer: nn.Module, d_model: int, d_ff: int):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model)
        self.mixer = mixer
        self.post_attention_layernorm = RMSNorm(d_model)
        self.mlp = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mixer(self.input_layernorm(x))
        if isinstance(h, tuple):                      # GatedDeltaNet возвращает (y, state)
            h = h[0]
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class HybridStack(nn.Module):
    """Стек из n_layers блоков: каждый interval-й — gated attention («Gated attention»), остальные — Gated DeltaNet («Gated DeltaNet»)."""

    def __init__(self, d_model: int, n_layers: int, full_attention_interval: int, d_ff: int,
                 n_heads: int, n_kv_heads: int, head_dim: int,
                 linear_heads: int, linear_k_dim: int, linear_v_dim: int):
        super().__init__()
        self.layer_types = hybrid_layer_types(n_layers, full_attention_interval)
        self.layers = nn.ModuleList()
        for kind in self.layer_types:
            mixer = (GatedAttention(d_model, n_heads, n_kv_heads, head_dim) if kind == "full_attention"
                     else GatedDeltaNet(d_model, linear_heads, linear_k_dim, linear_v_dim))
            self.layers.append(HybridBlock(mixer, d_model, d_ff))
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def hybrid_cache_bytes(layer_types: list[str], seq_len: int, n_kv_heads: int, head_dim: int,
                       linear_heads: int, linear_k_dim: int, linear_v_dim: int,
                       batch: int = 1, bytes_per_elem: int = 2) -> int:
    """Память инференса: KV-cache full-слоёв (растёт с seq_len) + состояния linear-слоёв (константа)."""
    full = layer_types.count("full_attention")
    linear = layer_types.count("linear_attention")
    kv = 2 * n_kv_heads * head_dim * seq_len                  # K и V на один full-слой
    state = linear_heads * linear_k_dim * linear_v_dim        # S [H, dk, dv] на один linear-слой
    return (full * kv + linear * state) * batch * bytes_per_elem
