"""causal multi-head self-attention, как в GPT-2.

Реализуй две вещи:
1. функцию causal_attention(q, k, v) — scaled dot-product attention с causal-маской;
2. метод CausalSelfAttention.forward — разбиение на головы, вызов causal_attention,
   склейка голов и выходная проекция.

Нельзя использовать F.scaled_dot_product_attention и nn.MultiheadAttention — смысл задачи
в том, чтобы написать механизм руками.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention с causal-маской.

    q, k, v: [B, H, T, D] → выход [B, H, T, D].
    """
    # TODO 1: оценки сходства каждого запроса с каждым ключом, масштабированные на размер головы.
    #         Какая форма должна получиться?
    # TODO 2: запрети позиции t смотреть на позиции > t. Что нужно записать в запрещённые
    #         ячейки, чтобы после softmax там был ровно 0?
    # TODO 3: нормализуй оценки в веса (по какой оси?) и усредни значения v с этими весами.
    raise NotImplementedError("causal_attention")


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        # Конструктор уже готов: имена и формы параметров проверяются тестами — не меняй их.
        assert d_model % n_heads == 0, "d_model должно делиться на n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)  # Q, K, V одной матрицей
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)      # выходная проекция

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] → [B, T, C]."""
        # TODO 1: получи q, k, v из одной проекции c_attn (в каком порядке они лежат?).
        # TODO 2: переведи каждый из [B, T, C] в [B, H, T, D].
        # TODO 3: вызови causal_attention.
        # TODO 4: склей головы обратно в [B, T, C] и примени c_proj.
        raise NotImplementedError("CausalSelfAttention.forward")
