"""Тесты «Найди баг: autograd» — те же, что у задачи «Value: autograd на скалярах», без подсказок про баг.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул — torch.autograd в float64: одни и те же функции считаются на Value и на torch.Tensor
(у тензора тоже есть .exp(), .log(), .tanh(), .relu() и операторы).
"""

import math
import sys
import time

import pytest
import torch

ATOL, RTOL = 1e-9, 1e-7


def _torch_grads(fn, *xs: float) -> tuple[float, list[float]]:
    ts = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in xs]
    out = fn(*ts)
    out.backward()
    return out.item(), [t.grad.item() if t.grad is not None else 0.0 for t in ts]


def _value_grads(impl, fn, *xs: float) -> tuple[float, list[float]]:
    vs = [impl.Value(x) for x in xs]
    out = fn(*vs)
    assert isinstance(out, impl.Value), f"результат должен быть Value, а получен {type(out).__name__}"
    out.backward()
    return out.data, [v.grad for v in vs]


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def _check(impl, fn, xs, what: str) -> None:
    want_val, want_grads = _torch_grads(fn, *xs)
    got_val, got_grads = _value_grads(impl, fn, *xs)
    if not _close(got_val, want_val):
        pytest.fail(f"{what}: значение {got_val:.6g}, ожидалось {want_val:.6g} — ошибка в прямом проходе.", pytrace=False)
    for i, (g, w) in enumerate(zip(got_grads, want_grads)):
        if not _close(g, w):
            pytest.fail(f"{what}: градиент по входу #{i} = {g:.6g}, а torch.autograd даёт {w:.6g}.", pytrace=False)


UNARY = {
    "exp": (lambda x: x.exp(), 0.7, "производная exp(x) — сам exp(x)"),
    "log": (lambda x: x.log(), 2.5, "производная log(x) — 1/x"),
    "tanh": (lambda x: x.tanh(), 0.4, "производная tanh(x) — 1 − tanh²(x)"),
    "relu (x > 0)": (lambda x: x.relu(), 1.3, "при x > 0 relu пропускает градиент как есть (производная 1)"),
    "relu (x < 0)": (lambda x: x.relu(), -1.3, "при x < 0 relu обнуляет и значение, и градиент"),
    "x ** 3": (lambda x: x**3, 1.5, "производная x^k — k·x^(k−1)"),
    "x ** -1": (lambda x: x**-1, 2.0, "x^(−1) = 1/x, производная −1/x²; формула k·x^(k−1) работает и для k < 0"),
    "x ** 0.5": (lambda x: x**0.5, 4.0, "x^0.5 = √x, производная 1/(2√x)"),
}


def test_add_and_mul(impl):
    """a·b + a: значение и градиенты по a и b"""
    a, b = impl.Value(2.0), impl.Value(-3.0)
    out = a * b + a
    assert isinstance(out, impl.Value), "a * b + a должно вернуть Value"
    assert math.isclose(out.data, -4.0), f"2·(−3) + 2 = −4, а получилось {out.data}: ошибка в прямом проходе + или *"
    out.backward()
    if math.isclose(a.grad, 3.0) and math.isclose(b.grad, -3.0):
        pytest.fail("Градиенты по a и b перепутаны: ∂(a·b)/∂a = b, а ∂(a·b)/∂b = a — каждый вход получает "
                    "значение ДРУГОГО входа.", pytrace=False)
    assert math.isclose(b.grad, 2.0), f"∂(a·b + a)/∂b = a = 2, а получилось {b.grad:g}"
    assert math.isclose(a.grad, -2.0), (
        f"∂(a·b + a)/∂a = b + 1 = −2, а получилось {a.grad:g}.")


@pytest.mark.parametrize("name", list(UNARY))
def test_unary_ops(impl, name):
    """Унарные операции: exp, log, tanh, relu, степень — значение и производная"""
    fn, x, hint = UNARY[name]
    want_val, (want,) = _torch_grads(fn, x)
    got_val, (got,) = _value_grads(impl, fn, x)
    if not _close(got_val, want_val):
        pytest.fail(f"{name} при x = {x}: значение {got_val:.6g}, ожидалось {want_val:.6g}.", pytrace=False)
    if not _close(got, want):
        pytest.fail(f"{name} при x = {x}: производная {got:.6g}, ожидалось {want:.6g}. Подсказка: {hint}.",
                    pytrace=False)


def test_python_numbers_on_both_sides(impl):
    """Числа слева и справа: 2·x, x + 1, 1 − x, 3 / x, x / 4, −x"""
    cases = {
        "2 * x": lambda x: 2 * x, "x + 1": lambda x: x + 1, "1 - x": lambda x: 1 - x,
        "3 / x": lambda x: 3 / x, "x / 4": lambda x: x / 4, "-x": lambda x: -x,
    }
    for name, fn in cases.items():
        _check(impl, fn, [1.7], name)


def test_reused_node_accumulates_gradient(impl):
    """Узел, который используется дважды (x·x + x), получает сумму градиентов"""
    x = impl.Value(3.0)
    y = x * x + x
    y.backward()
    if math.isclose(x.grad, 7.0):
        return
    pytest.fail(f"d(x·x + x)/dx при x = 3 должно быть 7, а получилось {x.grad:g}.", pytrace=False)


def test_backward_seeds_output_with_one(impl):
    """После backward у выхода grad = 1, у узлов вне графа — 0"""
    a, b, unused = impl.Value(2.0), impl.Value(-1.0), impl.Value(5.0)
    out = a * b
    out.backward()
    assert out.grad == 1.0, (
        f"out.grad = {out.grad}: обратный проход начинается с ∂out/∂out = 1. Без этого все градиенты — нули.")
    assert unused.grad == 0.0, "узел, который не участвует в графе, не должен получать градиент"


def test_order_matters_shared_subexpressions(impl):
    """Граф с общими подвыражениями: градиент узла используется только когда он полный"""

    def fn(x):
        a = x * 2
        b = a * a          # a → b
        c = b + a          # a используется ещё раз, после b
        return c * b       # b используется дважды
    want_val, (want,) = _torch_grads(fn, 0.9)
    got_val, (got,) = _value_grads(impl, fn, 0.9)
    assert _close(got_val, want_val), f"значение {got_val:.6g} ≠ {want_val:.6g}"
    if not _close(got, want):
        pytest.fail(
            f"dL/dx = {got:.6g}, а должно быть {want:.6g}. Значения считаются верно, значит, ошибка в порядке обхода: "
            "градиент узла можно передавать дальше только когда пришли ВСЕ вклады от его потребителей. "
            "Это гарантирует обход в обратном топологическом порядке (каждый узел после всех, кто его использует).",
            pytrace=False)


@pytest.mark.parametrize("name", ["neuron", "softplus-ish", "rational", "deep mix"])
def test_composite_functions_match_torch(impl, name):
    """Составные выражения на нескольких входах == torch.autograd"""
    fns = {
        "neuron": (lambda w1, x1, w2, x2, b: (w1 * x1 + w2 * x2 + b).tanh(), [0.5, -1.2, -0.7, 2.0, 0.3]),
        "softplus-ish": (lambda x, y: (1 + (x * y).exp()).log() - y.relu(), [0.8, -1.5]),
        "rational": (lambda x, y, z: (x * y + z) / (x**2 + 1) - z / y, [1.3, -0.6, 2.2]),
        "deep mix": (lambda x, y: ((x * y).tanh() * x + (y - x) ** 2).exp() / (y * y + x.exp()), [0.3, 0.9]),
    }
    fn, xs = fns[name]
    _check(impl, fn, xs, name)


def test_deep_graph_without_recursion_error(impl):
    """Цепочка из 5000 операций: обратный проход без RecursionError и быстро"""
    x = impl.Value(0.5)
    y = x
    for _ in range(5000):
        y = y * 1.0001 + 0.0001
    t0 = time.perf_counter()
    try:
        y.backward()
    except RecursionError:
        pytest.fail(
            f"RecursionError: граф глубиной 10 000 узлов, а лимит рекурсии Python — {sys.getrecursionlimit()}. "
            "Сделай топологическую сортировку на явном стеке (список, append/pop) вместо рекурсивной функции.",
            pytrace=False)
    elapsed = time.perf_counter() - t0
    assert math.isclose(x.grad, 1.0001**5000, rel_tol=1e-9), (
        f"dy/dx = {x.grad:.6g}, а должно быть 1.0001^5000 ≈ {1.0001**5000:.6g}")
    assert elapsed < 1.0, (
        f"backward на 10 000 узлах занял {elapsed:.2f} с. Обычно это значит, что узлы обходятся много раз: "
        "отмечай посещённые узлы (множество id) и добавляй каждый в порядок один раз.")
