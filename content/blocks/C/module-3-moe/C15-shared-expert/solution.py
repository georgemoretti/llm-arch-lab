"""Эталон: DeepSeekMoE — shared experts + routed experts, первые слои dense."""

import torch
import torch.nn as nn

from llmlab.core.llama import SwiGLUFFN       # эксперт и dense-FFN — SwiGLU из «SwiGLU FFN»
from llmlab.core.moe import MoELayer          # routed experts — эталон «MoE-слой»


def mlp_layer_types(n_layers: int, first_k_dense_replace: int, moe_layer_freq: int = 1) -> list[str]:
    """DeepSeek: первые k слоёв — "dense", дальше "moe" в каждом moe_layer_freq-м слое."""
    return ["moe" if i >= first_k_dense_replace and i % moe_layer_freq == 0 else "dense"
            for i in range(n_layers)]


class DeepSeekMoE(nn.Module):
    def __init__(self, d_model: int, d_expert: int, n_routed: int, n_shared: int, top_k: int,
                 scoring_func: str = "softmax", norm_topk_prob: bool = True, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.routed = MoELayer(d_model, d_expert, n_routed, top_k, scoring_func, norm_topk_prob,
                               routed_scaling_factor)
        # как в HF DeepSeek: n_shared общих экспертов слиты в один SwiGLU шириной d_expert · n_shared
        self.shared_experts = SwiGLUFFN(d_model, d_expert * n_shared) if n_shared > 0 else None

    def forward(self, x: torch.Tensor):
        """x [B, T, C] → (out [B, T, C], router_logits [N, E])."""
        out, logits = self.routed(x)                       # top-k routed experts
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)             # shared — каждый токен, без роутера и весов
        return out, logits


def build_ffn_layers(n_layers: int, d_model: int, d_dense: int, d_expert: int, n_routed: int, n_shared: int,
                     top_k: int, first_k_dense_replace: int, moe_layer_freq: int = 1) -> nn.ModuleList:
    """FFN-часть всех слоёв: SwiGLUFFN(d_dense) для "dense", DeepSeekMoE для "moe"."""
    return nn.ModuleList(
        DeepSeekMoE(d_model, d_expert, n_routed, n_shared, top_k) if t == "moe" else SwiGLUFFN(d_model, d_dense)
        for t in mlp_layer_types(n_layers, first_k_dense_replace, moe_layer_freq))


def moe_param_counts(n_layers: int, d_model: int, d_dense: int, d_expert: int, n_routed: int, n_shared: int,
                     top_k: int, first_k_dense_replace: int) -> tuple[int, int]:
    """(всего, активно на токен) параметров FFN-части: dense-слои + MoE-слои (с роутером), без bias."""
    ffn = lambda d_hidden: 3 * d_model * d_hidden                       # gate, up, down
    n_moe = mlp_layer_types(n_layers, first_k_dense_replace).count("moe")
    dense = (n_layers - n_moe) * ffn(d_dense)
    router = d_model * n_routed
    total = dense + n_moe * ((n_routed + n_shared) * ffn(d_expert) + router)
    active = dense + n_moe * ((top_k + n_shared) * ffn(d_expert) + router)
    return total, active
