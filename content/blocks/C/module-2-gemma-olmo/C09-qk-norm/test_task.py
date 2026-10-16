"""Тесты C09: QK-Norm.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: совпадение с оракулом в обоих режимах (у норм случайные веса — тогда порядок «норма →
RoPE» и ось нормы различимы); инвариантность к масштабу W_Q и W_K; ограниченные логиты
|score| ≤ √D при единичных весах нормы; causal.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from llmlab.core.llama import apply_rope, rms_norm, rope_cos_sin

B, T, C, H, H_KV = 2, 12, 32, 4, 2
D = C // H
THETA = 10000.0
_SDPA = F.scaled_dot_product_attention


def _x(seed: int = 1):
    return torch.randn(B, T, C, generator=torch.Generator().manual_seed(seed))


def _make(impl, mode, seed=0, random_norms=True):
    torch.manual_seed(seed)
    m = impl.QKNormAttention(C, H, H_KV, qk_norm=mode, rope_theta=THETA).eval()
    if random_norms:
        with torch.no_grad():
            m.q_norm.weight.copy_(0.3 + 1.5 * torch.rand_like(m.q_norm.weight))
            m.k_norm.weight.copy_(0.3 + 1.5 * torch.rand_like(m.k_norm.weight))
    return m


def _sim(m, x, mode, order="norm_rope", eps=1e-6):
    """Оракул с переключателями типичных ошибок. mode: head | full | none."""
    t = x.shape[1]
    with torch.no_grad():
        q, k = m.q_proj(x), m.k_proj(x)
        wq, wk = m.q_norm.weight, m.k_norm.weight
        if mode == "full":
            q, k = rms_norm(q, wq, eps), rms_norm(k, wk, eps)
        q = q.view(B, t, H, D).transpose(1, 2)
        k = k.view(B, t, H_KV, D).transpose(1, 2)
        cos, sin = rope_cos_sin(torch.arange(t), D, THETA)
        if order == "rope_norm":
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if mode == "head":
            q, k = rms_norm(q, wq, eps), rms_norm(k, wk, eps)
        if order == "norm_rope":
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        v = m.v_proj(x).view(B, t, H_KV, D).transpose(1, 2)
        k, v = k.repeat_interleave(H // H_KV, 1), v.repeat_interleave(H // H_KV, 1)
        out = _SDPA(q, k, v, is_causal=True)
        return m.o_proj(out.transpose(1, 2).reshape(B, t, C))


def _diagnose(m, x, out, mode) -> str:
    variants = {
        "нормы применены ПОСЛЕ RoPE. RMSNorm с весами по каналам не коммутирует с поворотом: "
        "сначала норма, потом RoPE (как в Gemma 3 и OLMo 2)": _sim(m, x, mode, order="rope_norm"),
        "QK-Norm не применяется вовсе": _sim(m, x, "none"),
    }
    for hint, cand in variants.items():
        if cand.shape == out.shape and torch.allclose(out, cand, atol=1e-4, rtol=1e-3):
            return f" Похоже, {hint}."
    return ""


def _check(impl, mode):
    m = _make(impl, mode)
    x = _x()
    with torch.no_grad():
        out = m(x)
    assert tuple(out.shape) == (B, T, C), f"Выход формы {tuple(out.shape)}, ожидалось {(B, T, C)}."
    expected = _sim(m, x, mode)
    if not torch.allclose(out, expected, atol=1e-5, rtol=1e-4):
        pytest.fail(f"QKNormAttention(qk_norm='{mode}') не совпадает с оракулом.{_diagnose(m, x, out, mode)}",
                    pytrace=False)


def test_head_mode_matches_oracle(impl):
    """qk_norm='head' (Gemma 3): RMSNorm по каждой голове [.., D], затем RoPE"""
    _check(impl, "head")


def test_full_mode_matches_oracle(impl):
    """qk_norm='full' (OLMo 2): RMSNorm по всей проекции [.., H·D] до разбиения на головы, затем RoPE"""
    _check(impl, "full")


def test_norm_before_rope(impl):
    """Порядок важен: норма → RoPE. С обученными (неединичными) весами нормы перестановка меняет выход"""
    m = _make(impl, "head", seed=5)
    x = _x(seed=7)
    with torch.no_grad():
        out = m(x)
    wrong = _sim(m, x, "head", order="rope_norm")
    assert not torch.allclose(out, wrong, atol=1e-4), (
        "Выход совпал с вариантом «RoPE → норма». Так делать нельзя: RMSNorm умножает каналы на разные веса, "
        "а RoPE смешивает каналы i и i + D/2 поворотом — эти операции не коммутируют. Веса нормы обучены на "
        "векторах ДО поворота, поэтому порядок как в Gemma 3/OLMo 2: q_norm → apply_rope.")
    assert_close(out, _sim(m, x, "head"), atol=1e-5, rtol=1e-4,
                 msg=lambda _: "Режим 'head' не совпадает с «норма по голове → RoPE».")


def test_scores_shape_and_bound(impl):
    """scores: [B, H, T, T]; при единичных весах нормы |score| ≤ √D — даже если W_Q, W_K огромные"""
    m = _make(impl, "head", random_norms=False)
    with torch.no_grad():
        m.q_proj.weight.mul_(1000)
        m.k_proj.weight.mul_(1000)
        scores, v = m.scores(_x())
    assert tuple(scores.shape) == (B, H, T, T), (
        f"scores формы {tuple(scores.shape)}, ожидалось {(B, H, T, T)} — до маски и softmax, после repeat_kv.")
    assert tuple(v.shape) == (B, H, T, D), f"v формы {tuple(v.shape)}, ожидалось {(B, H, T, D)} (после repeat_kv)."
    bound = math.sqrt(D)
    mx = scores.abs().max().item()
    assert mx <= bound * (1 + 1e-4), (
        f"max |score| = {mx:.3g} при √D = {bound:.3g}. После RMSNorm по голове ‖q‖ = ‖k‖ = √D, RoPE норму "
        "не меняет, значит |q·k|/√D ≤ D/√D = √D при любом масштабе W_Q, W_K. Большие логиты — норма не "
        "по голове или не применяется.")


def test_scale_invariance(impl):
    """С QK-Norm выход не зависит от масштаба W_Q и W_K (в обоих режимах) — логиты не «взрываются»"""
    x = _x()
    for mode in ("head", "full"):
        m = _make(impl, mode)
        with torch.no_grad():
            base = m(x)
            m.q_proj.weight.mul_(30)
            m.k_proj.weight.mul_(4)
            scaled = m(x)
        assert_close(scaled, base, atol=1e-4, rtol=1e-3,
                     msg=lambda _: f"qk_norm='{mode}': после умножения W_Q на 30 и W_K на 4 выход изменился. "
                                   "RMSNorm делит q и k на их RMS — масштаб проекций должен сокращаться. Именно "
                                   "это не даёт логитам attention расти во время обучения (рост логитов → softmax "
                                   "вырождается в one-hot → нестабильность).")


def test_no_future_leak(impl):
    """Causal: изменение токенов после t не меняет выход в позициях ≤ t"""
    m = _make(impl, "head")
    x = _x()
    x2 = x.clone()
    x2[:, 6:] += 3 * torch.randn_like(x2[:, 6:])
    with torch.no_grad():
        delta = (m(x)[:, :6] - m(x2)[:, :6]).abs().max().item()
    assert delta < 1e-5, f"Выход в позициях 0..5 зависит от токенов 6+ (|Δ| = {delta:.3g}) — нет causal-маски."
