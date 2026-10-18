"""Тесты «Расписание LR и gradient clipping».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.
"""

import math

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

_CLIP = nn.utils.clip_grad_norm_


@pytest.fixture(autouse=True)
def _forbid(monkeypatch):
    def forbid(*a, **k):
        raise AssertionError("Использован готовый clip_grad_norm_ / get_total_norm: здесь обрезка пишется руками.")
    monkeypatch.setattr(nn.utils, "clip_grad_norm_", forbid)
    monkeypatch.setattr(nn.utils.clip_grad, "clip_grad_norm_", forbid)
    if hasattr(nn.utils, "get_total_norm"):
        monkeypatch.setattr(nn.utils, "get_total_norm", forbid)


def test_warmup_is_linear_from_first_step(impl):
    """Warmup: lr растёт линейно, шаг 0 уже ненулевой, на шаге warmup − 1 ровно max_lr"""
    lrs = [impl.lr_at(s, 1e-3, 10, 100) for s in range(10)]
    if lrs[0] == 0.0:
        pytest.fail("lr на шаге 0 равен 0: первый шаг оптимизатора пропадёт впустую. Считай (step + 1) / warmup.",
                    pytrace=False)
    for s, v in enumerate(lrs):
        assert math.isclose(v, 1e-3 * (s + 1) / 10, rel_tol=1e-12), (
            f"шаг {s}: lr = {v:.3e}, а при линейном warmup нужно max_lr·(step + 1)/warmup = {1e-3 * (s + 1) / 10:.3e}")


def test_cosine_decay_points(impl):
    """Cosine: max_lr в начале, середина = (max + min)/2, min_lr после конца"""
    f = lambda s: impl.lr_at(s, 3e-4, 100, 1100, min_lr=3e-5)  # noqa: E731
    assert math.isclose(f(100), 3e-4, rel_tol=1e-12), f"сразу после warmup lr = max_lr, а получено {f(100):.3e}"
    mid = f(600)
    if math.isclose(mid, 1.5e-4, rel_tol=1e-6):
        pytest.fail("в середине lr = max/2: косинус идёт к нулю, а должен к min_lr. Формула: min + (max − min)·(1 + cos)/2.",
                    pytrace=False)
    assert math.isclose(mid, (3e-4 + 3e-5) / 2, rel_tol=1e-9), f"в середине decay lr = (max + min)/2, а получено {mid:.3e}"
    assert math.isclose(f(350), 3e-5 + 0.5 * 2.7e-4 * (1 + math.cos(math.pi * 0.25)), rel_tol=1e-9), (
        "на четверти decay значение не совпало с min + (max − min)·(1 + cos(π·0.25))/2")
    for s in (1100, 1101, 5000):
        assert math.isclose(f(s), 3e-5, rel_tol=1e-12), f"после total_steps lr = min_lr, а на шаге {s} получено {f(s):.3e}"


def test_schedule_shape(impl):
    """Расписание без скачков: растёт в warmup, монотонно падает после, без разрывов на стыке"""
    lrs = [impl.lr_at(s, 1.0, 20, 200, 0.1) for s in range(220)]
    assert all(a <= b + 1e-15 for a, b in zip(lrs[:20], lrs[1:21])), "в warmup lr не должен убывать"
    assert all(a >= b - 1e-15 for a, b in zip(lrs[20:], lrs[21:])), "после warmup lr не должен расти"
    jumps = max(abs(a - b) for a, b in zip(lrs, lrs[1:]))
    assert jumps <= 0.05 + 1e-12, f"соседние шаги отличаются на {jumps:.3f}: где-то разрыв (проверь стык warmup и decay)"


def test_no_warmup(impl):
    """warmup_steps = 0: без деления на ноль, сразу max_lr"""
    try:
        v = impl.lr_at(0, 1e-3, 0, 100)
    except ZeroDivisionError:
        pytest.fail("ZeroDivisionError при warmup_steps = 0: без warmup расписание сразу начинает decay с max_lr.",
                    pytrace=False)
    assert math.isclose(v, 1e-3, rel_tol=1e-12), f"без warmup lr на шаге 0 = max_lr, а получено {v}"


def _params(scale: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ps = [torch.zeros(3, 4, requires_grad=True), torch.zeros(5, requires_grad=True), torch.zeros(2, requires_grad=True)]
    for p in ps[:2]:
        p.grad = torch.randn(p.shape, generator=g) * scale
    return ps                                         # у третьего параметра .grad = None


def test_clip_matches_torch(impl):
    """Обрезка: та же глобальная норма и те же градиенты, что у nn.utils.clip_grad_norm_"""
    a, b = _params(10.0), _params(10.0)
    want = _CLIP(b, 1.0).item()
    got = impl.clip_grad_norm_(a, 1.0)
    assert isinstance(got, float), f"верни норму числом float (.item()), а не {type(got).__name__}"
    per_tensor = [p.grad.norm().item() for p in _params(10.0)[:2]]
    if any(math.isclose(got, n, rel_tol=1e-5) for n in per_tensor) or math.isclose(got, sum(per_tensor), rel_tol=1e-5):
        pytest.fail("норма посчитана не глобально: нужен корень из суммы квадратов всех элементов всех градиентов "
                    "(как если бы их склеили в один вектор).", pytrace=False)
    assert math.isclose(got, want, rel_tol=1e-5), f"норма до обрезки {got:.5f}, а должна быть {want:.5f}"
    for i, (p, q) in enumerate(zip(a[:2], b[:2])):
        assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5,
                     msg=f"градиент #{i} после обрезки не совпал: все градиенты умножаются на одно число max_norm/(norm + eps)")
    assert a[2].grad is None, "параметр без градиента должен остаться без градиента"
    total = math.sqrt(sum((p.grad**2).sum().item() for p in a[:2]))
    assert total <= 1.0 + 1e-5, f"после обрезки глобальная норма {total:.4f} > max_norm = 1"


def test_clip_keeps_small_gradients(impl):
    """Если норма меньше max_norm, градиенты не меняются"""
    ps = _params(0.01)
    before = [p.grad.clone() for p in ps[:2]]
    n = impl.clip_grad_norm_(ps, 1.0)
    assert n < 1.0
    for p, b0 in zip(ps[:2], before):
        assert torch.equal(p.grad, b0), (
            "градиенты с нормой меньше max_norm изменились: обрезка только уменьшает, а не нормирует к max_norm")


def test_clip_preserves_direction(impl):
    """Обрезка сохраняет направление: отношение компонент не меняется"""
    ps = _params(50.0, seed=3)
    ratio_before = (ps[0].grad.flatten()[:5] / ps[1].grad[0]).clone()
    impl.clip_grad_norm_(ps, 0.5)
    assert_close(ps[0].grad.flatten()[:5] / ps[1].grad[0], ratio_before, atol=1e-6, rtol=1e-5,
                 msg="после обрезки градиенты разных параметров изменились в разное число раз: клиппинг по каждому "
                     "тензору отдельно меняет направление шага. Масштаб один на все параметры.")
