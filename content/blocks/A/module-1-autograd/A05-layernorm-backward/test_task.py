"""Тесты «Backward LayerNorm».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул — F.layer_norm и torch.autograd в float64 (сохранены до запрета в autouse-фикстуре).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

B, T, C = 2, 5, 16
EPS = 1e-5
ATOL, RTOL = 1e-10, 1e-8

_LAYER_NORM = F.layer_norm
_AUTOGRAD_BACKWARD = torch.autograd.backward


@pytest.fixture(autouse=True)
def _forbid(monkeypatch):
    def forbid(name: str):
        def f(*args, **kwargs):
            raise AssertionError(f"Использован {name}. Здесь LayerNorm и её градиент пишутся руками.")
        return f
    monkeypatch.setattr(F, "layer_norm", forbid("F.layer_norm"))
    monkeypatch.setattr(nn.LayerNorm, "forward", forbid("nn.LayerNorm"))
    monkeypatch.setattr(torch.Tensor, "backward", forbid("tensor.backward()"))
    monkeypatch.setattr(torch.autograd, "backward", forbid("torch.autograd.backward"))
    monkeypatch.setattr(torch.autograd, "grad", forbid("torch.autograd.grad"))


def _t(*shape, seed=0, scale=1.0, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64) * scale + shift


def _params(c=C):
    return _t(c, seed=1, scale=0.5, shift=1.0), _t(c, seed=2, scale=0.3)


def _oracle(x, gamma, beta, dy, eps=EPS):
    x, gamma, beta = (t.clone().requires_grad_() for t in (x, gamma, beta))
    y = _LAYER_NORM(x, (x.shape[-1],), gamma, beta, eps)
    _AUTOGRAD_BACKWARD([(y * dy).sum()])
    return y.detach(), x.grad, gamma.grad, beta.grad


def _diagnose_dx(dx, x, gamma, dy, eps=EPS) -> str:
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    rstd = (var + eps).rsqrt()
    xhat = (x - mu) * rstd
    g = dy * gamma
    variants = {
        "μ и σ считаются константами: dx = g/σ. Забыты оба пути через статистики — вычти mean(g) и x̂·mean(g·x̂)":
            g * rstd,
        "учтён путь через μ, но забыт путь через σ: не хватает слагаемого −x̂·mean(g·x̂)":
            rstd * (g - g.mean(-1, keepdim=True)),
        "учтён путь через σ, но забыт путь через μ: не хватает слагаемого −mean(g)":
            rstd * (g - xhat * (g * xhat).mean(-1, keepdim=True)),
        "забыто умножение на γ: g = dy·γ, а не dy":
            rstd * (dy - dy.mean(-1, keepdim=True) - xhat * (dy * xhat).mean(-1, keepdim=True)),
    }
    for hint, cand in variants.items():
        if cand.shape == dx.shape and torch.allclose(dx, cand, atol=1e-8, rtol=1e-6):
            return f" Похоже, {hint}."
    return ""


def test_forward_matches_layer_norm(impl):
    """layernorm_forward == F.layer_norm на [B, T, C], форма та же"""
    x = _t(B, T, C, scale=2.0, shift=0.5)
    gamma, beta = _params()
    y, _ = impl.layernorm_forward(x, gamma, beta, EPS)
    assert y.shape == x.shape, f"y должен иметь форму входа {tuple(x.shape)}, а получено {tuple(y.shape)}"
    want = _LAYER_NORM(x, (C,), gamma, beta, EPS)
    if not torch.allclose(y, want, atol=1e-10, rtol=1e-8):
        var_u = x.var(-1, unbiased=True, keepdim=True)
        alt = (x - x.mean(-1, keepdim=True)) / torch.sqrt(var_u + EPS) * gamma + beta
        if torch.allclose(y, alt, atol=1e-8):
            pytest.fail("дисперсия посчитана с делением на C − 1 (unbiased). LayerNorm делит на C: "
                        "x.var(-1, unbiased=False) или mean((x − μ)²).", pytrace=False)
    assert_close(y, want, atol=ATOL, rtol=RTOL, msg="выход не совпадает с LayerNorm: (x − μ)/sqrt(σ² + eps)·γ + β по оси C")


def test_backward_matches_autograd(impl):
    """dx, dgamma, dbeta == autograd на [B, T, C]"""
    x, dy = _t(B, T, C, scale=2.0, shift=0.5), _t(B, T, C, seed=3)
    gamma, beta = _params()
    _, dx_w, dg_w, db_w = _oracle(x, gamma, beta, dy)
    _, cache = impl.layernorm_forward(x, gamma, beta, EPS)
    dx, dg, db = impl.layernorm_backward(dy, cache)
    assert dg.shape == (C,) and db.shape == (C,), (
        f"dgamma и dbeta — формы [C] = ({C},), как γ и β; получено {tuple(dg.shape)} и {tuple(db.shape)}. "
        "Суммируй по всем осям, кроме последней.")
    assert_close(db, db_w, atol=ATOL, rtol=RTOL, msg="dbeta = Σ dy по всем осям, кроме C")
    assert_close(dg, dg_w, atol=ATOL, rtol=RTOL, msg="dgamma = Σ dy·x̂ по всем осям, кроме C")
    assert dx.shape == x.shape, f"dx — форма x {tuple(x.shape)}, а получено {tuple(dx.shape)}"
    if not torch.allclose(dx, dx_w, atol=1e-9, rtol=1e-7):
        pytest.fail("dx расходится с autograd." + _diagnose_dx(dx, x, gamma, dy), pytrace=False)


def test_dx_sums_to_zero_per_row(impl):
    """Свойство: Σ dx по оси C = 0 для каждой строки (сдвиг x на константу не меняет выход)"""
    x, dy = _t(B, T, C, seed=4, scale=3.0), _t(B, T, C, seed=5)
    gamma, beta = _params()
    _, cache = impl.layernorm_forward(x, gamma, beta, EPS)
    dx, _, _ = impl.layernorm_backward(dy, cache)
    s = dx.sum(-1).abs().max().item()
    assert s < 1e-10, (
        f"max |Σ_C dx| = {s:.2e}, а должно быть ~0. LayerNorm не замечает сдвиг x + c·1, значит, градиент "
        "ортогонален вектору из единиц. Ненулевая сумма значит, что не учтён путь через среднее μ.")


def test_backward_scale_invariance(impl):
    """Свойство: для x·k градиент dx ровно в k раз меньше (при eps = 0)"""
    x, dy = _t(4, C, seed=6), _t(4, C, seed=7)
    gamma, beta = _params()
    _, c1 = impl.layernorm_forward(x, gamma, beta, 0.0)
    _, c2 = impl.layernorm_forward(x * 10.0, gamma, beta, 0.0)
    dx1, dg1, _ = impl.layernorm_backward(dy, c1)
    dx2, dg2, _ = impl.layernorm_backward(dy, c2)
    assert_close(dx2 * 10.0, dx1, atol=1e-10, rtol=1e-8,
                 msg="выход LayerNorm не зависит от масштаба x, поэтому dx(10·x) = dx(x)/10. Проверь множитель 1/σ.")
    assert_close(dg2, dg1, atol=1e-10, rtol=1e-8, msg="dgamma не должен зависеть от масштаба входа (x̂ тот же)")


@pytest.mark.parametrize("c, eps", [(2, 1e-5), (7, 0.5)])
def test_small_c_and_large_eps(impl, c, eps):
    """Крайние случаи: C = 2 и большой eps = 0.5 — градиенты по-прежнему точные"""
    x, dy = _t(3, c, seed=8, scale=0.3), _t(3, c, seed=9)
    gamma, beta = _params(c)
    y_w, dx_w, dg_w, db_w = _oracle(x, gamma, beta, dy, eps)
    y, cache = impl.layernorm_forward(x, gamma, beta, eps)
    assert_close(y, y_w, atol=ATOL, rtol=RTOL, msg=f"forward при C = {c}, eps = {eps}: eps стоит под корнем, рядом с σ²")
    dx, dg, db = impl.layernorm_backward(dy, cache)
    if not torch.allclose(dx, dx_w, atol=1e-9, rtol=1e-7):
        pytest.fail(f"dx при C = {c}, eps = {eps} расходится с autograd. Используй в backward тот же 1/sqrt(σ² + eps), "
                    "что и в forward (сохрани его в cache)." + _diagnose_dx(dx, x, gamma, dy, eps), pytrace=False)
    assert_close(dg, dg_w, atol=ATOL, rtol=RTOL, msg="dgamma")
    assert_close(db, db_w, atol=ATOL, rtol=RTOL, msg="dbeta")
