"""Блок трансформера: LayerNorm, attention, MLP, остаточные связи; pre-LN и post-LN.

Реализуй MLP.forward и Block.forward (оба режима). Конструкторы готовы: имена слоёв — как в
llmlab/core/gpt2.py (тесты загружают веса эталона).
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
        # Готово, не меняй.
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: расширение → GELU → сжатие. В GPT-2 GELU считается через tanh-аппроксимацию:
        #       посмотри на параметр approximate у F.gelu.
        raise NotImplementedError("MLP.forward")


class Block(nn.Module):
    """Блок трансформера: attention и MLP с остаточными связями; LayerNorm до подслоя (pre-LN) или после (post-LN)."""

    def __init__(self, n_embd: int, block_size: int, pre_norm: bool = True):
        super().__init__()
        # Готово, не меняй.
        self.pre_norm = pre_norm
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = AttentionHead(n_embd, n_embd, block_size)   # одна голова на весь n_embd (много голов — C00)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, C] → [B, T, C]."""
        # TODO 1 (pre_norm=True, GPT-2): нормализуется только ВХОД подслоя, а к x прибавляется его выход:
        #         сначала attention с ln_1, потом MLP с ln_2. Сам residual-поток x не нормализуется.
        # TODO 2 (pre_norm=False, Transformer 2017): подслой получает x как есть, нормализуется СУММА
        #         x + подслой(x): ln_1 после attention, ln_2 после MLP.
        raise NotImplementedError("Block.forward")
