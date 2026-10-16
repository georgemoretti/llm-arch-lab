"""C06d: блок и модель Llama из готовых деталей C01–C05."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from llmlab.core.llama import KVCache, LlamaAttention, RMSNorm, SwiGLUFFN   # эталоны C01–C05


@dataclass
class LlamaConfig:
    vocab_size: int = 64
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    d_hidden: int = 64              # скрытый размер SwiGLU
    rms_eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_embeddings: bool = False    # Llama 3 8B: False; Llama 3.2 1B: True


class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.self_attn = LlamaAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.rope_theta)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLUFFN(cfg.d_model, cfg.d_hidden)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """x [B, T, C] → [B, T, C]. Pre-Norm: норма перед каждым подслоем."""
        h = self.input_layernorm(x)
        x = h + self.self_attn(h, cache)
        h = self.post_attention_layernorm(x)
        x = h + self.mlp(h)
        return x


class LlamaModel(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)      # позиций тут нет: они в RoPE
        self.layers = nn.ModuleList(LlamaBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)                       # финальная норма
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight                  # один и тот же тензор

    def new_caches(self) -> list[KVCache]:
        """По одному KV-cache на слой."""
        return [KVCache() for _ in self.layers]

    def forward(self, idx: torch.Tensor, caches: list[KVCache] | None = None) -> torch.Tensor:
        """idx [B, T] (новые токены) → логиты [B, T, vocab]."""
        x = self.embed_tokens(idx)                                          # [B, T, C]
        for i, layer in enumerate(self.layers):
            x = layer(x, caches[i] if caches is not None else None)        # у каждого слоя свой кеш
        return self.lm_head(self.norm(x))                                   # [B, T, V]


@torch.no_grad()
def generate(model: LlamaModel, idx: torch.Tensor, max_new_tokens: int, use_cache: bool = True) -> torch.Tensor:
    """Жадная генерация: idx [B, T] → [B, T + max_new_tokens]."""
    caches = model.new_caches() if use_cache else None
    logits = model(idx, caches)                                             # prefill
    for step in range(max_new_tokens):
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)               # [B, 1]
        idx = torch.cat((idx, next_tok), dim=1)
        if step + 1 < max_new_tokens:
            logits = model(next_tok, caches) if use_cache else model(idx)   # с кешем — только новый токен
    return idx
