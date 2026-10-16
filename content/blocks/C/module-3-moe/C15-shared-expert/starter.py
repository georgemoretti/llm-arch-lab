"""C15: DeepSeekMoE — shared experts + routed experts, первые слои dense.

Реализуй:
1. mlp_layer_types — какие слои dense, какие MoE;
2. DeepSeekMoE.forward — routed (MoE из C13) + shared experts;
3. build_ffn_layers — FFN-часть модели по расписанию;
4. moe_param_counts — сколько параметров FFN всего и сколько активно на токен.
"""

import torch
import torch.nn as nn

from llmlab.core.llama import SwiGLUFFN       # эксперт и dense-FFN — SwiGLU из C02
from llmlab.core.moe import MoELayer          # routed experts — эталон C13


def mlp_layer_types(n_layers: int, first_k_dense_replace: int, moe_layer_freq: int = 1) -> list[str]:
    """DeepSeek: первые k слоёв — "dense", дальше "moe" в каждом moe_layer_freq-м слое."""
    # TODO: слой i — "moe", если он не среди первых first_k_dense_replace и i делится на moe_layer_freq.
    raise NotImplementedError("mlp_layer_types")


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
        # TODO 1: out, logits = self.routed(x).
        # TODO 2: если shared experts есть — прибавить их выход. Нужны ли им роутер и веса?
        raise NotImplementedError("DeepSeekMoE.forward")


def build_ffn_layers(n_layers: int, d_model: int, d_dense: int, d_expert: int, n_routed: int, n_shared: int,
                     top_k: int, first_k_dense_replace: int, moe_layer_freq: int = 1) -> nn.ModuleList:
    """FFN-часть всех слоёв: SwiGLUFFN(d_dense) для "dense", DeepSeekMoE для "moe"."""
    # TODO: nn.ModuleList по mlp_layer_types: SwiGLUFFN(d_model, d_dense) или
    #       DeepSeekMoE(d_model, d_expert, n_routed, n_shared, top_k).
    raise NotImplementedError("build_ffn_layers")


def moe_param_counts(n_layers: int, d_model: int, d_dense: int, d_expert: int, n_routed: int, n_shared: int,
                     top_k: int, first_k_dense_replace: int) -> tuple[int, int]:
    """(всего, активно на токен) параметров FFN-части: dense-слои + MoE-слои (с роутером), без bias."""
    # TODO 1: параметры одного SwiGLU шириной F: три матрицы C×F без bias.
    # TODO 2: сколько слоёв dense и сколько MoE (mlp_layer_types).
    # TODO 3: всего: dense-слои + MoE-слои × (все routed + shared эксперты + роутер E×C).
    # TODO 4: активно на токен: то же, но из routed — только top_k. Роутер считается для каждого токена.
    raise NotImplementedError("moe_param_counts")
