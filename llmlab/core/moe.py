"""Готовые компоненты MoE (модуль 3 блока C) — эталоны задач C12–C15.

C12 роутер (scores → top-k → перенормировка), C13 MoE-слой (наивный и через сортировку токенов),
C14 балансировка нагрузки (aux loss Switch-style и bias-based из DeepSeek-V3), C15 shared expert и
первые dense-слои. Поздние задачи и финальная модель (Фаза 8) импортируют их отсюда.

⚠️ Это готовые ответы на задачи C12–C15. Если ещё не решил их — не подглядывай.

Код намеренно совпадает с ``content/blocks/C/module-3-moe/C1*/solution.py``; расхождение ловит
``tests/test_core_moe.py``. Имена параметров — как в Hugging Face ``transformers``
(``modeling_deepseek_v3.py``: ``gate.weight``, ``experts.{i}.gate_proj``, ``shared_experts``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import SwiGLUFFN

# ---------------------------------------------------------------------------- C12 роутер


def route(logits: torch.Tensor, top_k: int, scoring_func: str = "softmax", norm_topk_prob: bool = True,
          routed_scaling_factor: float = 1.0):
    """logits [N, E] → (weights [N, k] fp32, indices [N, k]).

    scoring_func: "softmax" (Mixtral, Qwen) — распределение по экспертам; "sigmoid" (DeepSeek-V3) —
    независимая оценка каждого эксперта. top-k выбирается по оценкам, веса — оценки выбранных;
    при norm_topk_prob веса делятся на свою сумму, затем умножаются на routed_scaling_factor.
    """
    logits = logits.float()
    scores = logits.softmax(dim=-1) if scoring_func == "softmax" else logits.sigmoid()
    weights, indices = scores.topk(top_k, dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * routed_scaling_factor, indices


class TopKRouter(nn.Module):
    """Роутер MoE: линейный слой без bias [E, C] + route()."""

    def __init__(self, d_model: int, n_experts: int, top_k: int, scoring_func: str = "softmax",
                 norm_topk_prob: bool = True, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter(torch.randn(n_experts, d_model) * d_model ** -0.5)

    def forward(self, x: torch.Tensor):
        """x [..., C] → (logits [N, E], weights [N, k], indices [N, k]), N — все токены."""
        h = x.reshape(-1, x.shape[-1])
        logits = F.linear(h.float(), self.weight.float())
        weights, indices = route(logits, self.top_k, self.scoring_func, self.norm_topk_prob,
                                 self.routed_scaling_factor)
        return logits, weights, indices


# ---------------------------------------------------------------------------- C13 MoE-слой


class MoELayer(nn.Module):
    """Sparse MoE: роутер + E экспертов SwiGLU; каждый токен проходит ровно через top-k экспертов."""

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
        h = x.reshape(-1, shape[-1])
        logits, weights, indices = self.gate(h)
        dense = torch.zeros(h.shape[0], self.n_experts, dtype=weights.dtype, device=h.device)
        dense.scatter_(1, indices, weights)                                    # [N, E], нули вне top-k
        out = sum(dense[:, e:e + 1].to(h.dtype) * expert(h) for e, expert in enumerate(self.experts))
        return out.reshape(shape), logits

    def forward(self, x: torch.Tensor):
        """x [B, T, C] → (out [B, T, C], router_logits [N, E]). Эксперт e видит только свои токены."""
        shape = x.shape
        h = x.reshape(-1, shape[-1])
        N, k = h.shape[0], self.top_k
        logits, weights, indices = self.gate(h)
        flat_expert = indices.reshape(-1)                                     # [N·k]
        flat_token = torch.arange(N, device=h.device).repeat_interleave(k)    # [N·k]: чей это слот
        flat_weight = weights.reshape(-1)
        order = flat_expert.argsort(stable=True)                              # слоты, сгруппированные по эксперту
        counts = torch.bincount(flat_expert, minlength=self.n_experts).tolist()
        out = torch.zeros_like(h)
        start = 0
        for e, n_e in enumerate(counts):
            if n_e == 0:
                continue                                                      # эксперта никто не выбрал
            slots = order[start:start + n_e]
            tok = flat_token[slots]
            y = self.experts[e](h[tok]) * flat_weight[slots, None].to(h.dtype)
            out.index_add_(0, tok, y)                                         # сумма вкладов k экспертов
            start += n_e
        return out.reshape(shape), logits


# ---------------------------------------------------------------------------- C14 балансировка


def load_balancing_loss(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Switch-style aux loss: E · Σ_i f_i · P_i, где f_i — доля назначений (top-k) на эксперта i,
    P_i — средняя вероятность роутера для i. При идеальном балансе = 1, при коллапсе → E."""
    N, E = router_logits.shape
    probs = router_logits.float().softmax(dim=-1)                              # [N, E]
    indices = probs.topk(top_k, dim=-1).indices                               # [N, k]
    f = torch.bincount(indices.reshape(-1), minlength=E).float() / (N * top_k)  # [E], Σ = 1, без градиента
    P = probs.mean(dim=0)                                                     # [E], Σ = 1, с градиентом
    return E * (f * P).sum()


def expert_load(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Доля назначений на каждого эксперта [E] (сумма 1)."""
    counts = torch.bincount(indices.reshape(-1), minlength=n_experts).float()
    return counts / counts.sum()


def route_with_bias(scores: torch.Tensor, bias: torch.Tensor, top_k: int, norm_topk_prob: bool = True):
    """DeepSeek-V3: bias влияет только на ВЫБОР экспертов; веса — исходные scores выбранных."""
    indices = (scores + bias).topk(top_k, dim=-1).indices
    weights = scores.gather(-1, indices)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, indices


def update_bias(bias: torch.Tensor, load: torch.Tensor, gamma: float) -> torch.Tensor:
    """Шаг aux-loss-free балансировки: перегруженным экспертам bias −γ, недогруженным +γ."""
    return bias - gamma * torch.sign(load - load.mean())


# ---------------------------------------------------------------------------- C15 DeepSeekMoE


def mlp_layer_types(n_layers: int, first_k_dense_replace: int, moe_layer_freq: int = 1) -> list[str]:
    """DeepSeek: первые k слоёв — dense FFN, дальше MoE в каждом moe_layer_freq-м слое."""
    return ["moe" if i >= first_k_dense_replace and i % moe_layer_freq == 0 else "dense"
            for i in range(n_layers)]


class DeepSeekMoE(nn.Module):
    """Shared experts (видят каждый токен) + routed experts (top-k из E)."""

    def __init__(self, d_model: int, d_expert: int, n_routed: int, n_shared: int, top_k: int,
                 scoring_func: str = "softmax", norm_topk_prob: bool = True, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.routed = MoELayer(d_model, d_expert, n_routed, top_k, scoring_func, norm_topk_prob,
                               routed_scaling_factor)
        # как в HF DeepSeek: n_shared общих экспертов слиты в один SwiGLU шириной d_expert · n_shared
        self.shared_experts = SwiGLUFFN(d_model, d_expert * n_shared) if n_shared > 0 else None

    def forward(self, x: torch.Tensor):
        out, logits = self.routed(x)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out, logits


def ffn_params(d_model: int, d_hidden: int) -> int:
    """Параметры одного SwiGLU-FFN без bias: gate, up, down."""
    return 3 * d_model * d_hidden


def moe_param_counts(n_layers: int, d_model: int, d_dense: int, d_expert: int, n_routed: int, n_shared: int,
                     top_k: int, first_k_dense_replace: int) -> tuple[int, int]:
    """(всего, активно на токен) параметров FFN-части модели: dense-слои + MoE-слои (роутер тоже)."""
    n_moe = sum(t == "moe" for t in mlp_layer_types(n_layers, first_k_dense_replace))
    n_dense = n_layers - n_moe
    dense = n_dense * ffn_params(d_model, d_dense)
    router = d_model * n_routed
    total = dense + n_moe * ((n_routed + n_shared) * ffn_params(d_model, d_expert) + router)
    active = dense + n_moe * ((top_k + n_shared) * ffn_params(d_model, d_expert) + router)
    return total, active
