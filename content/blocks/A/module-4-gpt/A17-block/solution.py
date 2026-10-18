"""Блок трансформера: LayerNorm, attention, MLP, остаточные связи; pre-LN и post-LN.

Эталон. Имена слоёв — как в llmlab/core/gpt2.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.minigpt import AttentionHead   # готовая голова attention (эталон задачи «Attention»)



class MLP(nn.Module):
    """FFN как в GPT-2: C → 4C, GELU (tanh-аппроксимация), 4C → C."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    """Блок трансформера: attention и MLP с остаточными связями; LayerNorm до подслоя (pre-LN) или после (post-LN)."""

    def __init__(self, n_embd: int, block_size: int, pre_norm: bool = True):
        super().__init__()
        self.pre_norm = pre_norm
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = AttentionHead(n_embd, n_embd, block_size)   # одна голова на весь n_embd (много голов — C00)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:                                # GPT-2 и почти все современные LLM
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:                                            # исходный Transformer (2017)
            x = self.ln_1(x + self.attn(x))
            x = self.ln_2(x + self.mlp(x))
        return x
