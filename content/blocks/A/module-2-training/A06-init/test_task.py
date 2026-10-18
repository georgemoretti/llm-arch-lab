"""Тесты «Инициализация весов».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.
"""

import math

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

_GAIN = {"linear": 1.0, "tanh": 5 / 3, "relu": math.sqrt(2)}


def _xavier(shape, gain, seed):
    """Независимый оракул: то же, что nn.init.xavier_uniform_."""
    a = gain * math.sqrt(6 / (shape[0] + shape[1]))
    return torch.empty(*shape).uniform_(-a, a, generator=torch.Generator().manual_seed(seed))


def _kaiming(shape, nl, seed):
    """Независимый оракул: то же, что nn.init.kaiming_normal_ (fan_in = shape[1])."""
    return torch.empty(*shape).normal_(0, _GAIN[nl] / math.sqrt(shape[1]), generator=torch.Generator().manual_seed(seed))


@pytest.fixture(autouse=True)
def _forbid_nn_init(monkeypatch):
    def forbid(name):
        def f(*a, **k):
            raise AssertionError(f"Использован nn.init.{name}. Здесь формулы инициализации пишутся руками.")
        return f
    for name in ["xavier_uniform_", "xavier_normal_", "kaiming_normal_", "kaiming_uniform_", "calculate_gain"]:
        monkeypatch.setattr(nn.init, name, forbid(name))


def test_gain(impl):
    """gain: linear → 1, relu → √2, tanh → 5/3"""
    assert impl.gain("linear") == 1.0
    g = impl.gain("relu")
    if math.isclose(g, 2.0):
        pytest.fail("gain(relu) = 2 — это множитель для ДИСПЕРСИИ. gain умножает std, поэтому √2.", pytrace=False)
    assert math.isclose(g, math.sqrt(2)), (
        f"gain(relu) = {g}, а нужно √2: ReLU обнуляет половину значений и вдвое уменьшает E[x²], "
        "std компенсируется множителем √2")
    assert math.isclose(impl.gain("tanh"), 5 / 3), "gain(tanh) = 5/3 (как в PyTorch)"


def test_xavier_matches_reference_init(impl):
    """xavier_uniform_: то же распределение и те же числа, что nn.init (при том же генераторе)"""
    for shape, gn in [((64, 32), 1.0), ((10, 200), 5 / 3)]:
        W = torch.empty(*shape)
        out = impl.xavier_uniform_(W, gn, generator=torch.Generator().manual_seed(1))
        assert out is W, "функция с «_» на конце меняет W на месте и возвращает его же"
        want = _xavier(shape, gn, 1)
        a = gn * math.sqrt(6 / sum(shape))
        assert W.abs().max() <= a + 1e-6, f"значения выходят за ±{a:.4f} = ±gain·sqrt(6 / (fan_in + fan_out))"
        assert_close(W, want, atol=1e-7, rtol=1e-6,
                     msg=f"форма {shape}: числа не совпадают с эталоном. Нужно ровно W.uniform_(−a, a, generator=…)")


def test_kaiming_matches_reference_init(impl):
    """kaiming_normal_: std = gain/√fan_in, совпадает с nn.init при том же генераторе"""
    for shape, nl in [((64, 32), "relu"), ((48, 300), "tanh"), ((16, 16), "linear")]:
        W = torch.empty(*shape)
        out = impl.kaiming_normal_(W, nl, generator=torch.Generator().manual_seed(2))
        assert out is W, "kaiming_normal_ меняет W на месте и возвращает его"
        want = _kaiming(shape, nl, 2)
        if not torch.allclose(W, want, atol=1e-7, rtol=1e-6):
            ratio = (W.std() / want.std()).item()
            hint = ""
            if math.isclose(ratio, math.sqrt(shape[1] / shape[0]), rel_tol=0.05):
                hint = " Похоже, взят fan_out вместо fan_in: у W [fan_out, fan_in] fan_in — вторая ось."
            pytest.fail(f"{shape}, {nl}: числа не совпадают с эталоном (отношение std {ratio:.3f}).{hint}", pytrace=False)


def test_kaiming_requires_no_grad_tracking(impl):
    """Инициализация параметра с requires_grad не строит граф (внутри torch.no_grad)"""
    W = torch.empty(8, 4, requires_grad=True)
    impl.kaiming_normal_(W, "relu", generator=torch.Generator().manual_seed(0))
    assert W.grad_fn is None and W.is_leaf, (
        "после инициализации параметр перестал быть листом графа: заполняй его внутри torch.no_grad()")
    impl.xavier_uniform_(W, 1.0, generator=torch.Generator().manual_seed(0))
    assert W.grad_fn is None and W.is_leaf, "xavier_uniform_ тоже должна работать внутри torch.no_grad()"


def test_activation_stds_match_reference(impl, ref):
    """activation_stds: те же числа, что у эталона (порядок вызовов генератора)"""
    for init, act in [("kaiming", "relu"), ("xavier", "tanh"), ("normal-0.01", "linear")]:
        got = impl.activation_stds(init, act, depth=6, width=64, n=128, seed=3)
        want = ref.activation_stds(init, act, depth=6, width=64, n=128, seed=3)
        assert len(got) == 6, f"нужно по одному числу на слой (depth = 6), получено {len(got)}"
        assert_close(torch.tensor(got), torch.tensor(want), atol=1e-6, rtol=1e-4,
                     msg=f"{init} + {act}: std по слоям не совпали с эталоном. Проверь: x = act(x @ W.T), "
                         "одна и та же g для x и всех W, std после активации")


def test_depth_behaviour(impl):
    """Глубина 30: Kaiming держит масштаб, N(0,1) взрывается, N(0, 0.01) гаснет, Xavier гаснет с ReLU"""
    k = impl.activation_stds("kaiming", "relu")
    assert 0.3 < k[-1] < 3, f"Kaiming + ReLU: std на 30-м слое {k[-1]:.3g} — должен остаться порядка 1"
    big = impl.activation_stds("normal-1", "relu")
    assert big[-1] > 1e20, f"N(0, 1) + ReLU должен взорваться (каждый слой ×√(width/2) ≈ 11), а std = {big[-1]:.3g}"
    small = impl.activation_stds("normal-0.01", "linear")
    assert small[-1] < 1e-20, f"N(0, 0.01) должен погаснуть (каждый слой ×0.16), а std = {small[-1]:.3g}"
    x = impl.activation_stds("xavier", "relu")
    assert x[-1] < 1e-3 < x[0], (
        f"Xavier (gain = 1) + ReLU: std {x[0]:.3g} → {x[-1]:.3g}. Должен гаснуть: ReLU вдвое режет дисперсию, "
        "а Xavier этого не компенсирует — поэтому для ReLU нужен множитель √2 (Kaiming)")
