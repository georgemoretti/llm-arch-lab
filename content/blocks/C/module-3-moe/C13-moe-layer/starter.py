"""MoE-слой — наивная диспетчеризация и через сортировку токенов по экспертам.

Реализуй:
1. forward_naive — каждый эксперт считает ВСЕ токены, выходы смешиваются весами роутера
   (нули для невыбранных экспертов). Медленно, но очевидно правильно — это оракул;
2. forward — каждый эксперт получает ТОЛЬКО свои токены: сгруппировать пары (токен, эксперт)
   по эксперту, прогнать группы, сложить вклады обратно на места токенов.

Роутер («Роутер MoE») и эксперты (SwiGLU из «SwiGLU FFN») уже созданы в конструкторе.
"""

import torch
import torch.nn as nn

from llmlab.core.llama import SwiGLUFFN       # эксперт — SwiGLU из «SwiGLU FFN»
from llmlab.core.moe import TopKRouter        # роутер — эталон «Роутер MoE»


class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_expert: int, n_experts: int, top_k: int, scoring_func: str = "softmax",
                 norm_topk_prob: bool = True, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = TopKRouter(d_model, n_experts, top_k, scoring_func, norm_topk_prob, routed_scaling_factor)
        self.experts = nn.ModuleList(SwiGLUFFN(d_model, d_expert) for _ in range(n_experts))

    def forward_naive(self, x: torch.Tensor):
        """Каждый эксперт на ВСЕХ токенах, потом взвешенная сумма. O(N·E) — только как оракул."""
        # TODO 1: h = x как [N, C]; logits, weights, indices = self.gate(h).
        # TODO 2: плотная матрица весов [N, E]: нули, а в столбцах indices — weights (scatter_).
        # TODO 3: out = Σ_e dense[:, e:e+1] · expert_e(h) — каждый эксперт на всех токенах.
        # Вернуть (out в форме x, logits).
        raise NotImplementedError("MoELayer.forward_naive")

    def forward(self, x: torch.Tensor):
        """x [B, T, C] → (out [B, T, C], router_logits [N, E]). Эксперт e видит только свои токены."""
        # TODO 1: h [N, C]; logits, weights, indices = self.gate(h)  (weights, indices — [N, k]).
        # TODO 2: «расплющить» назначения в N·k слотов: эксперт слота, номер токена слота, вес слота.
        #         Подсказка: torch.arange(N).repeat_interleave(k).
        # TODO 3: сгруппировать слоты по эксперту: argsort по эксперту + bincount — сколько слотов у каждого.
        # TODO 4: для каждого эксперта с ненулевым числом слотов: его токены h[tok] → эксперт → × вес слота.
        # TODO 5: вернуть вклады на места токенов. Один токен получает вклады от k экспертов —
        #         их нужно СЛОЖИТЬ (какая операция складывает по индексам, а не перезаписывает?).
        # Вернуть (out в форме x, logits).
        raise NotImplementedError("MoELayer.forward")
