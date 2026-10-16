"""C06: блок и модель Llama из готовых деталей C01–C05.

Детали импортируются готовыми (эталоны прошлых задач):
  RMSNorm (C01), SwiGLUFFN (C02), LlamaAttention с RoPE и GQA (C03–C05), KVCache (C05).

Реализуй:
1. LlamaBlock: __init__ (имена подмодулей — как в HF, см. README) и forward (Pre-Norm);
2. LlamaModel: __init__ и forward (эмбеддинги → слои → финальная норма → lm_head).

LlamaConfig, new_caches и generate даны готовыми.
"""

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
        # TODO: четыре подмодуля с именами input_layernorm, self_attn, post_attention_layernorm, mlp.
        #       Какие классы и с какими аргументами из cfg?
        raise NotImplementedError("LlamaBlock.__init__")

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """x [B, T, C] → [B, T, C]."""
        # TODO: два подслоя, у каждого — своя норма на входе и остаточное соединение.
        #       Что именно проходит через норму: вход подслоя или весь остаточный поток?
        #       Не забудь передать cache в attention.
        raise NotImplementedError("LlamaBlock.forward")


class LlamaModel(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        # TODO 1: embed_tokens, layers (nn.ModuleList из cfg.n_layers блоков), norm, lm_head (без bias).
        #         Нужны ли позиционные эмбеддинги, как wpe в GPT-2?
        # TODO 2: если cfg.tie_embeddings — lm_head и embed_tokens должны делить ОДИН тензор весов.
        raise NotImplementedError("LlamaModel.__init__")

    def new_caches(self) -> list[KVCache]:
        """По одному KV-cache на слой."""
        return [KVCache() for _ in self.layers]

    def forward(self, idx: torch.Tensor, caches: list[KVCache] | None = None) -> torch.Tensor:
        """idx [B, T] (новые токены) → логиты [B, T, vocab]."""
        # TODO: эмбеддинги → каждый слой со СВОИМ кешем (или None) → финальная норма → lm_head.
        raise NotImplementedError("LlamaModel.forward")


@torch.no_grad()
def generate(model: LlamaModel, idx: torch.Tensor, max_new_tokens: int, use_cache: bool = True) -> torch.Tensor:
    """Жадная генерация: idx [B, T] → [B, T + max_new_tokens]. Дано готовым."""
    caches = model.new_caches() if use_cache else None
    logits = model(idx, caches)                                             # prefill
    for step in range(max_new_tokens):
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)               # [B, 1]
        idx = torch.cat((idx, next_tok), dim=1)
        if step + 1 < max_new_tokens:
            logits = model(next_tok, caches) if use_cache else model(idx)   # с кешем — только новый токен
    return idx
