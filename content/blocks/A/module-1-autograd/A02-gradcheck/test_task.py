"""Тесты «Проверка градиентов».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.
"""

import math

import pytest
import torch
from torch.testing import assert_close

from llmlab.core.autograd import Value

ATOL, RTOL = 1e-8, 1e-6


def _x(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_numerical_grad_shape_and_value(impl):
    """numerical_grad: та же форма, что у x, и верные значения на f = Σ x³"""
    x = _x(3, 4)
    g = impl.numerical_grad(lambda t: (t**3).sum(), x)
    assert isinstance(g, torch.Tensor) and g.shape == x.shape, (
        f"градиент должен быть тензором формы {tuple(x.shape)}, как x, а получено "
        f"{tuple(g.shape) if isinstance(g, torch.Tensor) else type(g).__name__}")
    assert_close(g, 3 * x**2, atol=ATOL, rtol=RTOL,
                 msg="для f = Σ x³ градиент равен 3x², а численный результат с ним не совпал")


def test_numerical_grad_does_not_change_x(impl):
    """numerical_grad не меняет x (ни значения, ни форму)"""
    x = _x(5)
    before = x.clone()
    impl.numerical_grad(lambda t: (t.sin() * t).sum(), x)
    assert torch.equal(x, before), (
        "после numerical_grad тензор x изменился: сдвиги ±h остались в исходных данных. Работай с копией "
        "(x.detach().clone()) или возвращай каждое значение на место.")


def test_central_not_forward_difference(impl):
    """Разность центральная: при h = 1e-3 ошибка ~h², а не ~h"""
    x = _x(6, seed=1)
    exact = x.exp()
    g = impl.numerical_grad(lambda t: t.exp().sum(), x, h=1e-3)
    err = ((g - exact).abs() / exact).max().item()
    if err > 1e-5:
        hint = ""
        if 1e-4 < err < 1e-3:
            hint = (" Ошибка ~h/2: похоже на одностороннюю разность (f(x + h) − f(x)) / h. Центральная "
                    "(f(x + h) − f(x − h)) / 2h сокращает член второго порядка, и ошибка падает до ~h².")
        pytest.fail(f"при h = 1e-3 относительная ошибка {err:.1e}, для центральной разности ожидается < 1e-5.{hint}",
                    pytrace=False)


def test_default_h_is_accurate(impl):
    """Значение h по умолчанию даёт точность лучше 1e-8 (не слишком большое и не слишком маленькое)"""
    x = _x(4, seed=2)
    g = impl.numerical_grad(lambda t: (t.tanh() * t).sum(), x)
    exact = x.tanh() + x * (1 - x.tanh() ** 2)
    err = (g - exact).abs().max().item()
    assert err < 1e-8, (
        f"ошибка {err:.1e} при h по умолчанию. Если h слишком маленькое (1e-10 и меньше), разность f(x+h) − f(x−h) "
        "теряет значащие цифры (ошибка округления ~ eps_машины / h). Если слишком большое, растёт ошибка "
        "формулы ~h². В float64 хороший компромисс — h ≈ 1e-5.")


def test_numerical_grad_works_without_autograd(impl):
    """numerical_grad работает и для функций на чистом Python (без torch.autograd)"""
    x = _x(4, seed=3)

    def f(t: torch.Tensor) -> float:             # .tolist() отрывает граф: autograd здесь не поможет
        return math.fsum(math.sin(v) * v for v in t.tolist())
    g = impl.numerical_grad(f, x)
    assert_close(g, x.sin() + x * x.cos(), atol=1e-8, rtol=1e-6,
                 msg="на функции без autograd численный градиент неверен: считай только через вызовы f")


def test_numerical_grad_matrix_function(impl):
    """Функция с взаимодействием элементов: f(W) = Σ tanh(W @ v) — совпадает с torch.autograd"""
    W, v = _x(3, 4, seed=4), _x(4, seed=5)
    Wg = W.clone().requires_grad_(True)
    torch.tanh(Wg @ v).sum().backward()
    g = impl.numerical_grad(lambda t: torch.tanh(t @ v).sum(), W)
    assert_close(g, Wg.grad, atol=1e-8, rtol=1e-6, msg="численный градиент по матрице расходится с torch.autograd")


def test_rel_error(impl):
    """rel_error: относительная, максимум по элементам, без NaN на нулях"""
    t = lambda *v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    e0 = impl.rel_error(t(0.0, 0.0), t(0.0, 0.0))
    assert isinstance(e0, float), f"rel_error должна возвращать число float, а не {type(e0).__name__}"
    assert e0 == 0.0, (
        f"rel_error(0, 0) = {e0}: для двух нулей ошибка 0. Если вышел NaN — нужен eps в знаменателе.")
    assert math.isclose(impl.rel_error(t(1e-8), t(2e-8)), 0.5, rel_tol=1e-9), (
        "rel_error(1e-8, 2e-8) должно быть 0.5: ошибка относительная, маленькие числа сравниваются по их масштабу")
    assert math.isclose(impl.rel_error(t(1.0, 100.0), t(1.0, 101.0)), 1 / 101, rel_tol=1e-9), (
        "rel_error([1, 100], [1, 101]) = 1/101: делим на max(|a|, |b|) и берём максимум по элементам")
    assert math.isclose(impl.rel_error(t(2.0, -4.0), t(1.0, 4.0)), 2.0, rel_tol=1e-9), (
        "rel_error([2, −4], [1, 4]): второй элемент даёт |−4 − 4| / 4 = 2 — знак важен, используй модули")


def test_check_value_grads_accepts_correct(impl):
    """check_value_grads подтверждает верные градиенты Value"""
    ok, err = impl.check_value_grads(lambda w, x, b: (w * x + b).tanh() * x, [0.6, -1.1, 0.2])
    assert ok and err < 1e-6, f"для верного autograd проверка должна пройти, а получено ok={ok}, err={err:.2e}"
    ok, err = impl.check_value_grads(lambda a, b: (a * b).exp() / (a**2 + 1) - b.log(), [0.4, 1.7])
    assert ok and err < 1e-6, f"для верного autograd проверка должна пройти, а получено ok={ok}, err={err:.2e}"


def test_check_value_grads_catches_wrong_derivative(impl):
    """check_value_grads ловит операцию с неверной локальной производной"""

    def bad_square(x: Value) -> Value:            # значение x², а производная записана как x вместо 2x
        return Value(x.data**2, (x,), (x.data,), "bad²")
    ok, err = impl.check_value_grads(lambda a, b: bad_square(a) * b + b, [1.5, 0.7])
    assert not ok, (
        f"операция с неверной производной прошла проверку (err={err:.2e}). Аналитика должна идти из backward, "
        "численный градиент — только из значений .data функции.")
    assert err > 0.1, f"ошибка должна быть заметной (∂/∂a отличается вдвое), а err = {err:.2e}"
    ok, _ = impl.check_value_grads(lambda a: bad_square(a), [0.0])
    assert ok, "в точке 0 обе производные равны 0 — проверка должна пройти (относительная ошибка 0, без NaN)"
