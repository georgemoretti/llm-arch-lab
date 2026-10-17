"""MoE-слой — код из «MoE-слой» с одной ошибкой. Найди и исправь её."""

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
        shape = x.shape
        h = x.reshape(-1, shape[-1])                                           # [N, C]
        logits, weights, indices = self.gate(h)
        dense = torch.zeros(h.shape[0], self.n_experts, dtype=weights.dtype, device=h.device)
        dense.scatter_(1, indices, weights)                                    # [N, E], нули вне top-k
        out = sum(dense[:, e:e + 1].to(h.dtype) * expert(h) for e, expert in enumerate(self.experts))
        return out.reshape(shape), logits

    def forward(self, x: torch.Tensor):
        """x [B, T, C] → (out [B, T, C], router_logits [N, E]). Эксперт e видит только свои токены."""
        shape = x.shape
        h = x.reshape(-1, shape[-1])                                          # [N, C]
        N, k = h.shape[0], self.top_k
        logits, weights, indices = self.gate(h)                               # weights, indices [N, k]
        flat_expert = indices.reshape(-1)                                     # [N·k] — эксперт каждого слота
        flat_token = torch.arange(N, device=h.device).repeat_interleave(k)    # [N·k] — чей это слот
        flat_weight = weights.reshape(-1)                                     # [N·k]
        order = flat_expert.argsort(stable=True)                              # слоты, сгруппированные по эксперту
        counts = torch.bincount(flat_expert, minlength=self.n_experts).tolist()
        out = torch.zeros_like(h)
        start = 0
        for e, n_e in enumerate(counts):
            if n_e == 0:
                continue                                                      # эксперта никто не выбрал
            slots = order[start:start + n_e]
            tok = flat_token[slots]
            y = self.experts[e](h[tok]) * flat_weight[slots, None].to(h.dtype)   # [n_e, C]
            out[tok] = y                                                      # вернуть результат на места токенов
            start += n_e
        return out.reshape(shape), logits
