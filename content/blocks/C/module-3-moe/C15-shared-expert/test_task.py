"""Тесты C15: DeepSeekMoE — shared expert и первые dense-слои.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: расписание dense/MoE совпадает с config.json DeepSeek-V3 и V2-Lite; shared expert видит
каждый токен без роутера (обнулённые routed-эксперты → выход = shared(x)); без shared experts слой —
это MoE из C13; число параметров «всего / активно» для DeepSeek-V3 и Qwen3-30B-A3B.
"""

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from llmlab.core.llama import SwiGLUFFN
from llmlab.core.moe import MoELayer

B, T, C, F_EXP, E, K = 2, 8, 16, 12, 6, 2


def _x(seed: int = 1):
    return torch.randn(B, T, C, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# расписание слоёв
# ---------------------------------------------------------------------------

def test_layer_types_deepseek(impl):
    """DeepSeek-V3: 61 слой, первые 3 — dense, остальные 58 — MoE; V2-Lite: первый dense"""
    # Источник: deepseek-ai/DeepSeek-V3 config.json: num_hidden_layers = 61, first_k_dense_replace = 3,
    # moe_layer_freq = 1; deepseek-ai/DeepSeek-V2-Lite: 27 слоёв, first_k_dense_replace = 1.
    types = impl.mlp_layer_types(61, 3)
    assert isinstance(types, list) and len(types) == 61, "mlp_layer_types должна вернуть список длины n_layers."
    assert set(types) <= {"dense", "moe"}, f"Значения — 'dense' / 'moe', получено {set(types)}."
    assert types[:3] == ["dense"] * 3 and types[3:] == ["moe"] * 58, (
        f"DeepSeek-V3: первые 3 слоя dense, затем 58 MoE; получено dense на индексах "
        f"{[i for i, t in enumerate(types) if t == 'dense']}.")
    assert impl.mlp_layer_types(27, 1) == ["dense"] + ["moe"] * 26, "DeepSeek-V2-Lite: только слой 0 — dense."
    assert impl.mlp_layer_types(6, 0) == ["moe"] * 6, "first_k_dense_replace = 0 (как у Qwen3-MoE): все слои MoE."
    assert impl.mlp_layer_types(6, 1, 2) == ["dense", "dense", "moe", "dense", "moe", "dense"], (
        "moe_layer_freq = 2: после первых dense-слоёв MoE только в слоях с чётным номером (i % 2 == 0).")


# ---------------------------------------------------------------------------
# DeepSeekMoE
# ---------------------------------------------------------------------------

def _make(impl, n_shared=1, seed=0, **kw):
    torch.manual_seed(seed)
    return impl.DeepSeekMoE(C, F_EXP, E, n_shared, K, **kw).eval()


def test_forward_is_routed_plus_shared(impl):
    """forward = routed MoE (C13) + shared expert; возвращает (out, router_logits)"""
    m = _make(impl, n_shared=2)
    assert isinstance(m.shared_experts, SwiGLUFFN) and m.shared_experts.up_proj.out_features == 2 * F_EXP, (
        "Конструктор дан: 2 shared-эксперта слиты в один SwiGLU шириной 2·d_expert.")
    x = _x()
    with torch.no_grad():
        res = m(x)
        assert isinstance(res, tuple) and len(res) == 2, "forward должен вернуть (out, router_logits), как MoE из C13."
        out, logits = res
        routed, logits_ref = m.routed(x)
        shared = m.shared_experts(x)
    assert tuple(out.shape) == (B, T, C), f"out формы {tuple(out.shape)}, ожидалось {(B, T, C)}."
    assert_close(logits, logits_ref, msg=lambda _: "router_logits — те, что вернул self.routed.")
    if not torch.allclose(out, routed + shared, atol=1e-5, rtol=1e-4):
        hint = ""
        if torch.allclose(out, routed, atol=1e-5):
            hint = " Выход совпал с одними routed-экспертами — shared expert не прибавлен."
        elif torch.allclose(out, shared, atol=1e-5):
            hint = " Выход совпал с одним shared expert — routed-эксперты потерялись."
        pytest.fail(f"Выход DeepSeekMoE ≠ routed(x) + shared(x).{hint}", pytrace=False)


def test_shared_expert_sees_every_token(impl):
    """Если routed-эксперты выдают 0, выход = shared(x) для КАЖДОГО токена — без весов роутера"""
    m = _make(impl)
    with torch.no_grad():
        for e in m.routed.experts:
            e.down_proj.weight.zero_()
        x = _x(seed=3)
        out, _ = m(x)
        shared = m.shared_experts(x)
    assert_close(out, shared, atol=1e-6, rtol=1e-5,
                 msg=lambda _: "С нулевыми routed-экспертами выход должен быть ровно shared(x): shared expert не "
                               "проходит через роутер и не умножается на его веса — он обрабатывает каждый токен.")


def test_no_shared_is_plain_moe(impl):
    """n_shared = 0: DeepSeekMoE — это просто MoE из C13"""
    m = _make(impl, n_shared=0)
    assert m.shared_experts is None, "Конструктор дан: при n_shared = 0 shared_experts = None."
    plain = MoELayer(C, F_EXP, E, K).eval()
    plain.load_state_dict(m.routed.state_dict())
    x = _x()
    with torch.no_grad():
        out, _ = m(x)
        ref, _ = plain(x)
    assert_close(out, ref, atol=1e-6, rtol=1e-5, msg=lambda _: "Без shared experts выход = routed MoE.")


def test_build_ffn_layers(impl):
    """build_ffn_layers: SwiGLUFFN(d_dense) в dense-слоях, DeepSeekMoE в MoE-слоях"""
    torch.manual_seed(0)
    layers = impl.build_ffn_layers(5, C, 40, F_EXP, E, 1, K, first_k_dense_replace=2)
    assert isinstance(layers, nn.ModuleList) and len(layers) == 5, (
        "build_ffn_layers должна вернуть nn.ModuleList длины n_layers (ModuleList — чтобы параметры были видны).")
    kinds = ["dense" if isinstance(layer, SwiGLUFFN) else "moe" if isinstance(layer, impl.DeepSeekMoE) else "?"
             for layer in layers]
    assert kinds == ["dense", "dense", "moe", "moe", "moe"], f"Типы слоёв {kinds}, ожидалось 2 dense + 3 moe."
    assert layers[0].up_proj.out_features == 40, "Dense-слой — SwiGLUFFN шириной d_dense (у DeepSeek-V3 это 18 432)."
    assert layers[2].routed.experts[0].up_proj.out_features == F_EXP, "Эксперты MoE-слоя — шириной d_expert."


# ---------------------------------------------------------------------------
# параметры
# ---------------------------------------------------------------------------

def test_param_counts_toy_model(impl):
    """moe_param_counts совпадает с числом параметров, которое строит build_ffn_layers"""
    torch.manual_seed(0)
    layers = impl.build_ffn_layers(4, C, 40, F_EXP, E, 1, K, first_k_dense_replace=1)
    real = sum(p.numel() for p in layers.parameters())
    total, active = impl.moe_param_counts(4, C, 40, F_EXP, E, 1, K, 1)
    assert total == real, (
        f"Всего параметров {total}, а в построенных слоях {real}. Учитываются: dense-слои (3·C·d_dense), а в каждом "
        "MoE-слое — routed и shared эксперты (3·C·d_expert каждый) и роутер E·C.")
    per_moe = (K + 1) * 3 * C * F_EXP + E * C
    assert active == 3 * C * 40 + 3 * per_moe, (
        f"Активно {active}, ожидалось {3 * C * 40 + 3 * per_moe}: dense-слой целиком + в каждом MoE-слое top_k routed "
        "+ shared эксперты + роутер.")


def test_param_counts_real_models(impl):
    """DeepSeek-V3: FFN ≈ 658B всего / 24B активно; Qwen3-30B-A3B: ≈ 29B / 1.8B"""
    # Источник: deepseek-ai/DeepSeek-V3 config.json: hidden_size 7168, intermediate_size 18432,
    # moe_intermediate_size 2048, n_routed_experts 256, n_shared_experts 1, num_experts_per_tok 8,
    # first_k_dense_replace 3, num_hidden_layers 61.
    total, active = impl.moe_param_counts(61, 7168, 18432, 2048, 256, 1, 8, 3)
    ffn = 3 * 7168 * 2048
    exp_total = 3 * 3 * 7168 * 18432 + 58 * (257 * ffn + 7168 * 256)
    exp_active = 3 * 3 * 7168 * 18432 + 58 * (9 * ffn + 7168 * 256)
    assert (total, active) == (exp_total, exp_active), (
        f"DeepSeek-V3: получено {total / 1e9:.1f}B / {active / 1e9:.2f}B, ожидалось {exp_total / 1e9:.1f}B всего / "
        f"{exp_active / 1e9:.2f}B активно (FFN-часть; вместе с attention (MLA) и эмбеддингами в отчёте "
        "DeepSeek — 671B / 37B).")
    # Источник: Qwen/Qwen3-30B-A3B config.json: 48 слоёв, hidden 2048, moe_intermediate_size 768,
    # num_experts 128, num_experts_per_tok 8, без shared experts, все слои MoE (mlp_only_layers = []).
    total, active = impl.moe_param_counts(48, 2048, 6144, 768, 128, 0, 8, 0)
    assert abs(total / 1e9 - 29.0) < 0.1 and abs(active / 1e9 - 1.82) < 0.02, (
        f"Qwen3-30B-A3B: получено {total / 1e9:.2f}B / {active / 1e9:.2f}B, ожидалось ≈ 29.0B / 1.82B для FFN-части "
        "(в названии 30B всего и 3B активных — вместе с attention и эмбеддингами).")
