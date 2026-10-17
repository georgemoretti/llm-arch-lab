"""роутер MoE — код из «Роутер MoE» с одной ошибкой. Найди и исправь её."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def route(logits: torch.Tensor, top_k: int, scoring_func: str = "softmax", norm_topk_prob: bool = True,
          routed_scaling_factor: float = 1.0):
    """logits [N, E] → (weights [N, k] float32, indices [N, k])."""
    logits = logits.float()                                     # роутинг считают в fp32
    if scoring_func == "softmax":
        scores = logits.softmax(dim=0)                          # распределение вероятностей экспертов
    else:
        scores = logits.sigmoid()                               # DeepSeek-V3: независимая оценка эксперта
    weights, indices = scores.topk(top_k, dim=-1)               # [N, k]
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)   # веса выбранных → сумма 1
    return weights * routed_scaling_factor, indices


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
        h = x.reshape(-1, x.shape[-1])                          # [N, C]
        logits = F.linear(h.float(), self.weight.float())       # [N, E]
        weights, indices = route(logits, self.top_k, self.scoring_func, self.norm_topk_prob,
                                 self.routed_scaling_factor)
        return logits, weights, indices
