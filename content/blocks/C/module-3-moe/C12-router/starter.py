"""C12: роутер MoE — оценки экспертов → top-k → перенормировка.

Реализуй:
1. route — из логитов роутера [N, E] получить веса и номера top-k экспертов каждого токена;
2. TopKRouter.forward — логиты роутера для всех токенов и route.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def route(logits: torch.Tensor, top_k: int, scoring_func: str = "softmax", norm_topk_prob: bool = True,
          routed_scaling_factor: float = 1.0):
    """logits [N, E] → (weights [N, k] float32, indices [N, k])."""
    # TODO 1: перевести логиты в float32 (роутинг чувствителен к точности).
    # TODO 2: оценки экспертов: softmax или sigmoid (scoring_func). По какой оси softmax?
    #         У КАЖДОГО ТОКЕНА — распределение по экспертам.
    # TODO 3: top-k по оценкам → weights [N, k], indices [N, k].
    # TODO 4: если norm_topk_prob — нормировать веса выбранных к сумме 1; затем × routed_scaling_factor.
    raise NotImplementedError("route")

class TopKRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int, scoring_func: str = "softmax",
                 norm_topk_prob: bool = True, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter(torch.randn(n_experts, d_model) * d_model ** -0.5)   # [E, C], без bias

    def forward(self, x: torch.Tensor):
        """x [..., C] → (logits [N, E], weights [N, k], indices [N, k]), N — все токены подряд."""
        # TODO: склеить все токены в [N, C], логиты = x · Wᵀ в float32 (F.linear), затем route(...).
        raise NotImplementedError("TopKRouter.forward")
