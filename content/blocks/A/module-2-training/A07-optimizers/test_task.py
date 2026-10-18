"""Тесты «Оптимизаторы руками».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул — torch.optim в float64 на той же функции потерь. Пока работает код пользователя, step() у torch.optim
подменён на ошибку: свой оптимизатор не должен вызывать готовый.
"""

import pytest
import torch
from torch.testing import assert_close

ATOL, RTOL = 1e-12, 1e-9


def _init():
    g = torch.Generator().manual_seed(0)
    return [torch.randn(5, generator=g, dtype=torch.float64), torch.randn(3, 4, generator=g, dtype=torch.float64)]


def _loss(ps):
    """Функция с сильно разным масштабом градиентов по параметрам — на ней видна разница SGD и Adam."""
    a, B = ps
    return 50 * (a[:2] ** 2).sum() + 0.1 * (a[2:] ** 2).sum() + (B.tanh() * a[:3, None]).sum() + 0.01 * (B**4).sum()


def _run(make_opt, steps: int, monkeypatch=None):
    ps = [x.clone().requires_grad_() for x in _init()]
    opt = make_opt(ps)
    for _ in range(steps):
        opt.zero_grad()
        _loss(ps).backward()
        opt.step()
    return ps


def _run_impl(make_opt, steps, monkeypatch):
    def forbid(*a, **k):
        raise AssertionError("Вызван torch.optim.*.step: здесь шаг оптимизатора пишется руками.")
    with monkeypatch.context() as m:
        for cls in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
            m.setattr(cls, "step", forbid)
        return _run(make_opt, steps)


def _compare(got, want, what: str, hint: str = ""):
    for i, (g, w) in enumerate(zip(got, want)):
        if not torch.allclose(g.detach(), w.detach(), atol=1e-10, rtol=1e-8):
            d = (g.detach() - w.detach()).abs().max().item()
            pytest.fail(f"{what}: параметр #{i} разошёлся с torch.optim (max |Δ| = {d:.2e}). {hint}", pytrace=False)


def test_sgd_plain(impl, monkeypatch):
    """SGD без momentum == torch.optim.SGD (10 шагов)"""
    got = _run_impl(lambda ps: impl.SGD(ps, lr=0.01), 10, monkeypatch)
    want = _run(lambda ps: torch.optim.SGD(ps, lr=0.01), 10)
    _compare(got, want, "SGD", "θ ← θ − lr·g для каждого параметра.")


def test_sgd_momentum(impl, monkeypatch):
    """SGD с momentum 0.9 == torch.optim.SGD (10 шагов)"""
    got = _run_impl(lambda ps: impl.SGD(ps, lr=0.01, momentum=0.9), 10, monkeypatch)
    want = _run(lambda ps: torch.optim.SGD(ps, lr=0.01, momentum=0.9), 10)
    _compare(got, want, "SGD momentum",
             "buf ← momentum·buf + g (без множителя 1 − momentum, как в PyTorch), θ ← θ − lr·buf; buf — своё у "
             "каждого параметра и живёт между шагами.")


def test_adam(impl, monkeypatch):
    """Adam == torch.optim.Adam (10 шагов)"""
    got = _run_impl(lambda ps: impl.Adam(ps, lr=0.05), 10, monkeypatch)
    want = _run(lambda ps: torch.optim.Adam(ps, lr=0.05), 10)
    _compare(got, want, "Adam",
             "m ← β1·m + (1−β1)·g, v ← β2·v + (1−β2)·g², m̂ = m/(1−β1^t), v̂ = v/(1−β2^t), θ ← θ − lr·m̂/(√v̂ + eps); "
             "t увеличивается один раз за step().")


def test_adam_first_step_is_lr_sized(impl):
    """Bias correction: первый шаг Adam сдвигает каждую координату почти ровно на lr"""
    ps = [x.clone().requires_grad_() for x in _init()]
    before = [p.detach().clone() for p in ps]
    opt = impl.Adam(ps, lr=0.01)
    _loss(ps).backward()
    opt.step()
    deltas = torch.cat([(p.detach() - b).abs().reshape(-1) for p, b in zip(ps, before)])
    if torch.allclose(deltas, torch.full_like(deltas, 0.01 * 0.1 / 0.001**0.5), rtol=1e-3):
        pytest.fail("первый шаг ≈ 3.16·lr: m и v поделены не на свои поправки. m̂ = m/(1 − β1^t), v̂ = v/(1 − β2^t).",
                    pytrace=False)
    if deltas.max() < 0.005:
        pytest.fail(f"первый шаг всего {deltas.max():.2e} при lr = 0.01: нет поправки m̂ = m/(1 − β1^t). Без неё m "
                    "стартует с нуля и первые шаги в разы меньше lr.", pytrace=False)
    assert_close(deltas, torch.full_like(deltas, 0.01), atol=1e-6, rtol=1e-4,
                 msg="на первом шаге m̂ = g и v̂ = g², поэтому |Δθ| = lr·|g|/(|g| + eps) ≈ lr для любой координаты")


def test_adamw(impl, monkeypatch):
    """AdamW (weight_decay = 0.1) == torch.optim.AdamW (10 шагов)"""
    got = _run_impl(lambda ps: impl.AdamW(ps, lr=0.05, weight_decay=0.1), 10, monkeypatch)
    want = _run(lambda ps: torch.optim.AdamW(ps, lr=0.05, weight_decay=0.1), 10)
    l2 = _run(lambda ps: torch.optim.Adam(ps, lr=0.05, weight_decay=0.1), 10)
    if all(torch.allclose(g.detach(), w.detach(), atol=1e-10) for g, w in zip(got, l2)):
        pytest.fail("получился Adam с L2-регуляризацией: wd·θ прибавлен к градиенту и прошёл через деление на √v̂. "
                    "В AdamW decay отдельный: θ ← θ·(1 − lr·wd), потом обычный шаг Adam.", pytrace=False)
    _compare(got, want, "AdamW", "Сначала θ ← θ·(1 − lr·wd), затем шаг Adam по исходному градиенту.")


def test_adamw_decays_params_without_gradient_signal(impl):
    """AdamW: параметр с нулевым градиентом всё равно уменьшается ровно в (1 − lr·wd) раз за шаг"""
    p = torch.tensor([2.0, -4.0, 0.5], dtype=torch.float64, requires_grad=True)
    opt = impl.AdamW([p], lr=0.1, weight_decay=0.5)
    for _ in range(3):
        p.grad = torch.zeros_like(p)
        opt.step()
    want = torch.tensor([2.0, -4.0, 0.5], dtype=torch.float64) * (1 - 0.1 * 0.5) ** 3
    if torch.allclose(p.detach().abs(), (torch.tensor([2.0, 4.0, 0.5]) - 0.3).double(), atol=1e-3):
        pytest.fail("каждый шаг сдвигает параметр на lr независимо от его величины — decay прошёл через "
                    "нормировку Adam (g/√v̂ ≈ ±1). Decoupled decay умножает θ на (1 − lr·wd).", pytrace=False)
    assert_close(p.detach(), want, atol=1e-12, rtol=1e-10,
                 msg="при g = 0 AdamW делает только decay: θ·(1 − lr·wd) за каждый шаг")


def test_zero_grad_and_skip_params_without_grad(impl):
    """zero_grad сбрасывает градиенты; параметр без .grad step() не трогает"""
    a = torch.ones(3, dtype=torch.float64, requires_grad=True)
    b = torch.ones(2, dtype=torch.float64, requires_grad=True)
    for make in (lambda: impl.SGD([a, b], lr=0.1, momentum=0.9), lambda: impl.Adam([a, b], lr=0.1),
                 lambda: impl.AdamW([a, b], lr=0.1, weight_decay=0.1)):
        opt = make()
        a.grad, b.grad = torch.ones_like(a), torch.ones_like(b)
        opt.step()                       # шаг 1: у b есть градиент — накапливаются momentum / m, v
        a.grad, b.grad = torch.ones_like(a), None
        before_b = b.detach().clone()
        opt.step()                       # шаг 2: у b градиента нет — b не трогаем вообще
        assert torch.equal(b.detach(), before_b), (
            f"{type(opt).__name__}: параметр без градиента (p.grad is None) изменился на втором шаге. Такие параметры "
            "пропускай целиком (как torch.optim) — не подставляй нулевой градиент: накопленные momentum/m сдвинут их.")
        opt.zero_grad()
        assert a.grad is None or not a.grad.any(), f"{type(opt).__name__}.zero_grad не сбросил градиент"
        a.grad = None


def test_step_updates_in_place(impl):
    """step() меняет те же тензоры на месте и не строит граф"""
    ps = [x.clone().requires_grad_() for x in _init()]
    ids = [id(p) for p in ps]
    opt = impl.AdamW(ps, lr=0.01)
    _loss(ps).backward()
    opt.step()
    assert [id(p) for p in opt.params] == ids, "оптимизатор должен обновлять те же объекты-параметры, что получил"
    assert all(p.is_leaf and p.grad_fn is None for p in ps), (
        "после step() параметр перестал быть листом: обновляй под torch.no_grad() и на месте (sub_, mul_)")
