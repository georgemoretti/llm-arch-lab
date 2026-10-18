"""Тесты «Attention: одна голова с causal-маской».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул — F.scaled_dot_product_attention (сохранён до запрета).
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, D = 2, 8, 16
C, BLOCK = 32, 16
_SDPA = F.scaled_dot_product_attention


@pytest.fixture(autouse=True)
def _forbid_builtin_attention(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Использована готовая реализация attention (F.scaled_dot_product_attention / "
                             "nn.MultiheadAttention). Здесь механизм пишется руками: q·kᵀ/√D → маска → softmax → @v.")

    monkeypatch.setattr(F, "scaled_dot_product_attention", forbidden)
    monkeypatch.setattr(nn.MultiheadAttention, "forward", forbidden)


def _qkv(t=T, d=D, dv=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, t, d, generator=g), torch.randn(B, t, d, generator=g), torch.randn(B, t, dv or d, generator=g))


def _diagnose(out, q, k, v, causal: bool) -> str:
    mask = torch.tril(torch.ones(q.shape[1], q.shape[1], dtype=torch.bool)) if causal else None

    def run(scale, before=True, dim=-1):
        s = q @ k.transpose(-2, -1) * scale
        if mask is not None and before:
            s = s.masked_fill(~mask, float("-inf"))
        w = s.softmax(dim)
        if mask is not None and not before:
            w = w.masked_fill(~mask, 0.0)
        return w @ v

    d = q.shape[-1]
    if torch.allclose(out, run(1.0), atol=1e-5):
        return "оценки не поделены на √D: при D = 16 логиты в 4 раза больше, и softmax становится почти one-hot."
    if torch.allclose(out, run(1.0 / d), atol=1e-5):
        return "оценки поделены на D, а нужно на √D: дисперсия q·k равна D, корень возвращает её к 1."
    if causal and torch.allclose(out, run(1.0 / math.sqrt(d), before=False), atol=1e-5):
        return ("маска применена после softmax: веса будущих позиций обнулены, но строка весов больше не "
                "суммируется в 1. Маску ставят до softmax значением −inf, тогда softmax сам даст им вес 0.")
    if torch.allclose(out, run(1.0 / math.sqrt(d), dim=-2), atol=1e-5):
        return "softmax взят по оси запросов (dim = −2): распределением должна быть каждая строка — по ключам."
    return "выход не совпадает с эталоном: q @ kᵀ / √D → маска (−inf) → softmax по ключам → @ v."


def test_causal_mask(impl):
    """causal_mask: bool [T, T], True на диагонали и ниже"""
    m = impl.causal_mask(4)
    assert m.shape == (4, 4) and m.dtype == torch.bool, f"маска — bool [4, 4], получено {m.dtype} {tuple(m.shape)}"
    want = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    if torch.equal(m, ~want) or torch.equal(m, want.T):
        pytest.fail("маска перевёрнута: строка i — запрос позиции i, и ей можно смотреть на столбцы j ≤ i "
                    "(нижний треугольник, torch.tril).", pytrace=False)
    assert torch.equal(m, want), f"ждём нижний треугольник с диагональю, получено\n{m.int()}"


def test_attention_without_mask(impl):
    """attention без маски: совпадает с оракулом, каждая строка весов — распределение"""
    q, k, v = _qkv()
    out, w = impl.attention(q, k, v)
    assert out.shape == (B, T, D) and w.shape == (B, T, T), (
        f"out [B, T, D] и weights [B, T, T], получено {tuple(out.shape)} и {tuple(w.shape)}")
    if not torch.allclose(out, _SDPA(q, k, v), atol=1e-5, rtol=1e-4):
        pytest.fail(_diagnose(out, q, k, v, causal=False), pytrace=False)
    torch.testing.assert_close(w.sum(-1), torch.ones(B, T), msg="веса каждого запроса суммируются в 1")
    torch.testing.assert_close(w @ v, out, msg="out = weights @ v")


def test_attention_with_causal_mask(impl):
    """attention с causal-маской: совпадает с оракулом, веса будущих ключей ровно 0, строки суммируются в 1"""
    q, k, v = _qkv()
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    out, w = impl.attention(q, k, v, mask)
    if not torch.allclose(out, _SDPA(q, k, v, is_causal=True), atol=1e-5, rtol=1e-4):
        pytest.fail(_diagnose(out, q, k, v, causal=True), pytrace=False)
    assert torch.all(w.masked_select(~mask.expand(B, T, T)) == 0), "веса будущих ключей должны быть ровно 0"
    torch.testing.assert_close(w.sum(-1), torch.ones(B, T), msg="каждая строка весов суммируется в 1")
    torch.testing.assert_close(out[:, 0], v[:, 0], msg="позиция 0 видит только себя: её выход — ровно v[0]")


def test_value_dim_and_large_scores(impl):
    """v может иметь другой размер; большие оценки не дают nan (softmax с −inf устойчив)"""
    q, k, v = _qkv(dv=5)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    out, _ = impl.attention(q * 30, k * 30, v, mask)
    assert out.shape == (B, T, 5), f"размер выхода берётся из v: [B, T, 5], получено {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "на больших оценках получились nan/inf"
    torch.testing.assert_close(out, _SDPA(q * 30, k * 30, v, is_causal=True), atol=1e-4, rtol=1e-4)


def _heads(impl, ref):
    torch.manual_seed(0)
    a = ref.AttentionHead(C, D, BLOCK)
    b = impl.AttentionHead(C, D, BLOCK)
    b.load_state_dict(a.state_dict())
    return a, b


def test_head_matches_reference(impl, ref):
    """AttentionHead: [B, T, C] → [B, T, head_size], совпадает с эталоном при T < block_size"""
    a, b = _heads(impl, ref)
    x = torch.randn(B, T, C, generator=torch.Generator().manual_seed(2))
    out = b(x)
    assert out.shape == (B, T, D), f"выход [B, T, head_size] = [{B}, {T}, {D}], получено {tuple(out.shape)}"
    want = _SDPA(a.query(x), a.key(x), a.value(x), is_causal=True)
    torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-4,
                               msg="выход головы: q, k, v — свои проекции x, маска [:T, :T], attention")


def test_head_no_future_leak(impl, ref):
    """Изменение токенов после позиции t не меняет выход на позициях ≤ t"""
    _, b = _heads(impl, ref)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(B, T, C, generator=g)
    for t in (0, 3, T - 2):
        y = x.clone()
        y[:, t + 1:] = torch.randn(B, T - t - 1, C, generator=g)
        if not torch.allclose(b(x)[:, : t + 1], b(y)[:, : t + 1], atol=1e-6):
            pytest.fail(f"выход на позиции ≤ {t} зависит от входа в позиции {t + 1} и дальше → утечка будущего "
                        "через маску. При обучении модель подсмотрит правильный ответ.", pytrace=False)
    other = b(torch.cat([x[:1], torch.randn(1, T, C, generator=g)]))
    torch.testing.assert_close(other[0], b(x)[0], msg="примеры батча не должны влиять друг на друга")
