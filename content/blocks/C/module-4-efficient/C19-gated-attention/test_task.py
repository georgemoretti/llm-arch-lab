"""Тесты «Gated attention»: gated attention и partial RoPE (Qwen3-Next).

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: partial RoPE вращает только первые R каналов (остальные не трогает), сохраняет норму и
делает score зависящим только от m − n; при R = D это обычный RoPE. Gated attention совпадает с
оракулом, гейт берётся из своей половины q_proj КАЖДОЙ головы, при нулевых весах гейта выход
attention ровно вдвое меньше (σ(0) = ½); будущее не протекает.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from llmlab.core.llama import apply_rope, rms_norm, rope_cos_sin

B, T, C, H, HKV, D, FACTOR = 2, 10, 32, 4, 2, 16, 0.25
R = int(D * FACTOR)
_SDPA = F.scaled_dot_product_attention


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _x(t: int = T, seed: int = 1):
    return torch.randn(B, t, C, generator=_gen(seed))


# ---------------------------------------------------------------------------
# apply_partial_rope
# ---------------------------------------------------------------------------

def test_partial_rope_passthrough(impl):
    """apply_partial_rope: каналы R…D−1 не меняются, форма сохраняется"""
    x = torch.randn(B, H, T, D, generator=_gen(2))
    cos, sin = rope_cos_sin(torch.arange(T), R)
    out = impl.apply_partial_rope(x, cos, sin)
    assert tuple(out.shape) == (B, H, T, D), (
        f"Форма {tuple(out.shape)}, ожидалось {(B, H, T, D)}: невращаемые каналы надо приклеить обратно (torch.cat).")
    assert_close(out[..., R:], x[..., R:], atol=0, rtol=0, msg=lambda _: (
        f"Каналы {R}…{D - 1} изменились. Partial RoPE вращает только ПЕРВЫЕ R = {R} каналов головы "
        "(как в HF: q[..., :rotary_dim]), остальные несут чистое содержание без позиции."))


def test_partial_rope_matches_rope(impl):
    """apply_partial_rope: первые R каналов — ровно RoPE с cos/sin размера R; при R = D — обычный RoPE"""
    x = torch.randn(B, H, T, D, generator=_gen(3))
    cos, sin = rope_cos_sin(torch.arange(T), R)
    out = impl.apply_partial_rope(x, cos, sin)
    expected = apply_rope(x[..., :R], cos, sin)
    if not torch.allclose(out[..., :R], expected, atol=1e-6):
        tail = apply_rope(x[..., D - R:], cos, sin)
        if torch.allclose(out[..., D - R:], tail, atol=1e-6):
            pytest.fail("Вращаются ПОСЛЕДНИЕ R каналов, а надо первые: x[..., :R] — вращаемая часть.", pytrace=False)
        pytest.fail("Первые R каналов не совпадают с apply_rope(x[..., :R], cos, sin). Вращаемая часть — это "
                    "отдельный вектор размера R со своими R/2 частотами (rotate_half внутри него).", pytrace=False)
    cos_d, sin_d = rope_cos_sin(torch.arange(T), D)
    assert_close(impl.apply_partial_rope(x, cos_d, sin_d), apply_rope(x, cos_d, sin_d), atol=1e-6, rtol=1e-5, msg=lambda _: (
        "При cos/sin размера D (factor = 1) partial RoPE должен совпасть с обычным RoPE. Бери R из cos.shape[-1]."))


def test_partial_rope_relative(impl):
    """apply_partial_rope: норма сохраняется, q·k зависит только от сдвига m − n"""
    g = _gen(4)
    q, k = torch.randn(1, 1, 1, D, generator=g), torch.randn(1, 1, 1, D, generator=g)
    pos = torch.arange(40)
    cos, sin = rope_cos_sin(pos, R)

    def at(v, p):
        return impl.apply_partial_rope(v, cos[p:p + 1], sin[p:p + 1])

    assert_close(at(q, 17).norm(), q.norm(), atol=1e-6, rtol=1e-5, msg=lambda _: "Поворот должен сохранять норму вектора.")
    s1 = (at(q, 9) * at(k, 4)).sum()
    s2 = (at(q, 30) * at(k, 25)).sum()
    assert_close(s1, s2, atol=1e-4, rtol=1e-4, msg=lambda _: (
        "score(m = 9, n = 4) ≠ score(m = 30, n = 25): при одинаковом m − n скалярное произведение после RoPE "
        "должно совпадать. Проверь, что q и k вращаются одинаковыми cos/sin своих позиций."))


# ---------------------------------------------------------------------------
# GatedAttention
# ---------------------------------------------------------------------------

def _make(impl, seed=0):
    torch.manual_seed(seed)
    m = impl.GatedAttention(C, H, HKV, D, partial_rotary_factor=FACTOR).eval()
    with torch.no_grad():
        m.q_norm.weight.copy_(0.5 + torch.rand(D))
        m.k_norm.weight.copy_(0.5 + torch.rand(D))
    return m


def _oracle(m, x, gate="sigmoid", split="head", rope="first", ungated=False):
    """Независимый расчёт; аргументы задают типичные ошибки для диагностики."""
    t = x.shape[1]
    with torch.no_grad():
        qg = m.q_proj(x)
        if split == "head":
            q, g = qg.view(B, t, H, 2 * D).chunk(2, dim=-1)
        else:                                     # ошибка: гейт — вторая половина ВСЕЙ проекции
            q, g = (z.reshape(B, t, H, D) for z in qg.chunk(2, dim=-1))
        g = g.reshape(B, t, H * D)
        q = rms_norm(q, m.q_norm.weight, m.q_norm.eps).transpose(1, 2)
        k = rms_norm(m.k_proj(x).view(B, t, HKV, D), m.k_norm.weight, m.k_norm.eps).transpose(1, 2)
        v = m.v_proj(x).view(B, t, HKV, D).transpose(1, 2)
        pos = torch.arange(t)
        if rope == "first":
            cos, sin = rope_cos_sin(pos, R)
            q = torch.cat((apply_rope(q[..., :R], cos, sin), q[..., R:]), -1)
            k = torch.cat((apply_rope(k[..., :R], cos, sin), k[..., R:]), -1)
        elif rope == "full":
            cos, sin = rope_cos_sin(pos, D)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        elif rope == "sliced":                    # ошибка: частоты от D, потом отрезаны первые R
            cos, sin = (z[:, :R] for z in rope_cos_sin(pos, D))
            q = torch.cat((apply_rope(q[..., :R], cos, sin), q[..., R:]), -1)
            k = torch.cat((apply_rope(k[..., :R], cos, sin), k[..., R:]), -1)
        k, v = k.repeat_interleave(H // HKV, 1), v.repeat_interleave(H // HKV, 1)
        out = _SDPA(q, k, v, is_causal=True).transpose(1, 2).reshape(B, t, H * D)
        if not ungated:
            out = out * (torch.sigmoid(g) if gate == "sigmoid" else F.silu(g))
        return m.o_proj(out)


def test_gated_attention_matches_oracle(impl):
    """GatedAttention: GQA + QK-Norm + partial RoPE + σ-гейт по каналам каждой головы"""
    m = _make(impl)
    x = _x()
    with torch.no_grad():
        out = m(x)
    assert tuple(out.shape) == (B, T, C), f"Выход формы {tuple(out.shape)}, ожидалось {(B, T, C)}."
    if torch.allclose(out, _oracle(m, x), atol=1e-5, rtol=1e-4):
        return
    variants = {
        "гейт — вторая половина ВСЕЙ проекции q_proj. В Qwen3-Next раскладка по головам: "
        "view(B, T, H, 2·D).chunk(2, dim=-1) — у каждой головы [запрос | гейт]": dict(split="all"),
        "нет гейта: выход attention надо умножить на σ(gate) до o_proj": dict(ungated=True),
        "SiLU вместо сигмоиды: гейт — σ(g) ∈ (0, 1), он только приглушает каналы": dict(gate="silu"),
        "RoPE на всех D каналах: вращаются только первые R = D · partial_rotary_factor": dict(rope="full"),
        "cos/sin посчитаны для размера D и обрезаны до R: у вращаемой части свои R/2 частот — "
        "rope_cos_sin(pos, R)": dict(rope="sliced"),
    }
    for name, kw in variants.items():
        if torch.allclose(out, _oracle(m, x, **kw), atol=1e-5, rtol=1e-4):
            pytest.fail(f"Выход совпал с вариантом «{name}».", pytrace=False)
    pytest.fail("GatedAttention не совпадает с оракулом. Порядок: q_proj → (запрос, гейт) по головам → QK-Norm "
                "по голове → partial RoPE → GQA attention (causal) → ⊙ σ(гейт) → o_proj.", pytrace=False)


def test_gate_layout_zero_gate_halves(impl):
    """GatedAttention: при нулевых весах гейта (строки гейта в q_proj) выход attention ровно вдвое меньше"""
    m = _make(impl)
    with torch.no_grad():
        w = m.q_proj.weight.view(H, 2 * D, C)
        w[:, D:] = 0                               # вторая половина строк каждой головы — гейт
    x = _x()
    with torch.no_grad():
        out = m(x)
    half = _oracle(m, x, ungated=True) * 0.5      # o_proj линейный и без bias: o_proj(½·a) = ½·o_proj(a)
    assert_close(out, half, atol=1e-5, rtol=1e-4, msg=lambda _: (
        "Обнулил строки гейта (q_proj.weight.view(H, 2·D, C)[:, D:]) — ожидалось σ(0) = ½ на всех каналах, "
        "выход вдвое меньше негейтированного. Не совпало → гейт берётся не из той половины проекции головы "
        "или применяется не как сигмоида."))


def test_gate_is_query_dependent(impl):
    """GatedAttention: гейт зависит от текущего токена — может «закрыть» голову для одного запроса"""
    m = _make(impl)
    with torch.no_grad():
        w = m.q_proj.weight.view(H, 2 * D, C)
        w[0, D:] = 0
        w[0, D:, 0] = 50.0                         # гейт головы 0 = σ(50·x₀): ≈ 1 при x₀ > 0, ≈ 0 при x₀ < 0
        x = _x()
        x[:, :, 0] = 1.0
        x[:, 3, 0] = -1.0                          # у запроса 3 голова 0 закрыта
        out = m(x)
        full = m.o_proj.weight.clone()
        m.o_proj.weight[:, :D] = 0                 # выключаем вклад головы 0 в o_proj
        no_head0 = m(x)
        m.o_proj.weight.copy_(full)
    assert_close(out[:, 3], no_head0[:, 3], atol=1e-4, rtol=1e-4, msg=lambda _: (
        "Запрос 3 закрыл гейт головы 0 (σ(−50) ≈ 0), но её вклад остался. Гейт умножает выход головы "
        "поэлементно, отдельно для каждого запроса — так голова может «промолчать» без attention sink."))
    assert not torch.allclose(out[:, 2], no_head0[:, 2], atol=1e-3), (
        "У запроса 2 гейт головы 0 открыт (σ(50) ≈ 1), её вклад должен быть.")


def test_gated_attention_causal(impl):
    """GatedAttention: выход в позиции t не зависит от токенов t + 1, …"""
    m = _make(impl)
    x = _x()
    with torch.no_grad():
        out = m(x)
        x2 = x.clone()
        x2[:, 6:] = torch.randn(B, T - 6, C)
        out2 = m(x2)
    assert_close(out[:, :6], out2[:, :6], atol=1e-6, rtol=1e-5, msg=lambda _: (
        "Выход позиций < 6 изменился после замены токенов 6… → утечка будущего: нужна causal-маска."))
