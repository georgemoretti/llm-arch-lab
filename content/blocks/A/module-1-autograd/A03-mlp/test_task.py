"""Тесты «MLP на скалярах».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул — та же сеть на torch (float64) с весами, переписанными из parameters() в документированном порядке:
слой за слоем, нейрон за нейроном, сначала веса, потом bias.
"""

import math
import random

import pytest
import torch
from torch.testing import assert_close

ATOL, RTOL = 1e-9, 1e-7


def _torch_net(net):
    """Список (W [nout, nin], b [nout], nonlin) из слоёв impl-сети, как тензоры float64 c requires_grad."""
    layers = []
    for layer in net.layers:
        W = torch.tensor([[w.data for w in n.w] for n in layer.neurons], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([n.b.data for n in layer.neurons], dtype=torch.float64, requires_grad=True)
        layers.append((W, b, layer.neurons[0].nonlin))
    return layers


def _torch_forward(layers, x):
    h = torch.as_tensor(x, dtype=torch.float64)
    for W, b, nonlin in layers:
        h = W @ h + b
        if nonlin:
            h = torch.tanh(h)
    return h


def _data_circles(n=24, seed=7):
    r = random.Random(seed)
    xs, ys = [], []
    for i in range(n):
        a = r.uniform(0, 2 * math.pi)
        rad = (0.4 if i % 2 == 0 else 1.0) + r.uniform(-0.1, 0.1)
        xs.append([rad * math.cos(a), rad * math.sin(a)])
        ys.append(1.0 if i % 2 == 0 else -1.0)
    return xs, ys


def test_parameters_count_and_order(impl):
    """parameters(): число параметров и порядок — слой, нейрон, веса, затем bias"""
    net = impl.MLP(3, [4, 2], seed=0)
    ps = net.parameters()
    assert len(ps) == 3 * 4 + 4 + 4 * 2 + 2, (
        f"у MLP(3, [4, 2]) должно быть 3·4 + 4 + 4·2 + 2 = 26 параметров, а parameters() вернул {len(ps)}. "
        "Не забыт ли bias у каждого нейрона?")
    n0 = net.layers[0].neurons[0]
    assert ps[0] is n0.w[0] and ps[2] is n0.w[2], "первыми идут веса первого нейрона первого слоя, по порядку входов"
    assert ps[3] is n0.b, "после весов нейрона — его bias (тот же объект Value, не копия)"
    assert ps[4] is net.layers[0].neurons[1].w[0], "затем следующий нейрон того же слоя"
    assert ps[-1] is net.layers[-1].neurons[-1].b, "последний параметр — bias последнего нейрона последнего слоя"


def test_forward_matches_torch(impl):
    """Прямой проход MLP(3, [5, 4, 2]) совпадает с той же сетью на torch"""
    net = impl.MLP(3, [5, 4, 2], seed=1)
    layers = _torch_net(net)
    for x in ([0.5, -1.0, 2.0], [0.0, 0.0, 0.0], [-0.3, 0.8, 0.1]):
        out = net(x)
        assert isinstance(out, list) and len(out) == 2, (
            f"MLP возвращает список выходов последнего слоя (здесь 2), а получено {out!r}")
        got = torch.tensor([o.data for o in out], dtype=torch.float64)
        want = _torch_forward(layers, x).detach()
        if not torch.allclose(got, want, atol=1e-9, rtol=1e-7):
            if torch.allclose(got, torch.tanh(want), atol=1e-9, rtol=1e-7):
                pytest.fail("выход прошёл через tanh в последнем слое. Выходной слой линейный (nonlin=False): "
                            "иначе предсказания зажаты в (−1, 1).", pytrace=False)
            pytest.fail(f"на x = {x} выход {got.tolist()} ≠ {want.tolist()}. Нейрон: b + Σ wᵢ·xᵢ, затем tanh "
                        "(кроме выходного слоя); слой — все нейроны на одном и том же входе.", pytrace=False)


def test_mse_loss(impl):
    """mse_loss: среднее квадратов разностей, градиент по предсказанию = 2(p − y)/N"""
    V = type(impl.MLP(1, [1]).parameters()[0])
    preds = [V(0.5), V(-1.0), V(2.0)]
    ys = [1.0, -1.0, 0.0]
    loss = impl.mse_loss(preds, ys)
    assert isinstance(loss, V), "mse_loss должна возвращать Value, чтобы от неё можно было вызвать backward()"
    want = (0.25 + 0.0 + 4.0) / 3
    if math.isclose(loss.data, 0.25 + 4.0):
        pytest.fail("mse_loss вернула сумму, а нужно среднее: раздели на число примеров.", pytrace=False)
    assert math.isclose(loss.data, want, rel_tol=1e-12), f"MSE = {want:.6g}, а получено {loss.data:.6g}"
    loss.backward()
    for p, y in zip(preds, ys):
        assert math.isclose(p.grad, 2 * (p.data - y) / 3, rel_tol=1e-12, abs_tol=1e-15), (
            f"∂MSE/∂pred = 2(pred − y)/N = {2 * (p.data - y) / 3:.6g}, а получено {p.grad:.6g}")


def test_backward_through_network_matches_torch(impl):
    """Градиенты всех весов после loss.backward() == torch.autograd"""
    net = impl.MLP(2, [3, 3, 1], seed=2)
    xs, ys = [[0.5, -0.2], [-1.0, 0.7], [0.3, 0.3]], [1.0, -1.0, 0.5]
    layers = _torch_net(net)
    loss = impl.mse_loss([net(x)[0] for x in xs], ys)
    loss.backward()
    preds = torch.stack([_torch_forward(layers, x)[0] for x in xs])
    torch_loss = ((preds - torch.tensor(ys, dtype=torch.float64)) ** 2).mean()
    torch_loss.backward()
    assert math.isclose(loss.data, torch_loss.item(), rel_tol=1e-9), "лосс сети расходится с torch"
    want = torch.cat([torch.cat([W.grad, b.grad[:, None]], dim=1).reshape(-1) for W, b, _ in layers])
    got = torch.tensor([p.grad for p in net.parameters()], dtype=torch.float64)
    assert_close(got, want, atol=ATOL, rtol=RTOL,
                 msg="градиенты весов не совпадают с torch.autograd: проверь прямой проход и порядок parameters()")


def test_zero_grad_and_sgd_step(impl):
    """zero_grad обнуляет .grad, sgd_step сдвигает .data на −lr·grad (те же объекты)"""
    net = impl.MLP(2, [2, 1], seed=3)
    ps = net.parameters()
    for i, p in enumerate(ps):
        p.grad = 0.1 * (i + 1)
    before = [p.data for p in ps]
    impl.sgd_step(ps, lr=0.5)
    for i, (p, b) in enumerate(zip(ps, before)):
        assert math.isclose(p.data, b - 0.5 * 0.1 * (i + 1), rel_tol=1e-12, abs_tol=1e-15), (
            "sgd_step: θ ← θ − lr·grad. Знак минус: идём ПРОТИВ градиента, туда, где лосс уменьшается.")
    assert all(p is q for p, q in zip(ps, net.parameters())), (
        "после sgd_step модель должна держать те же объекты Value: меняй p.data, а не заменяй элементы списка")
    impl.zero_grad(ps)
    assert all(p.grad == 0.0 for p in ps), "zero_grad должен обнулить .grad у каждого параметра"


def test_train_xor(impl):
    """train на XOR: лосс падает с ~1 почти до нуля за 200 шагов"""
    xs, ys = [[0, 0], [0, 1], [1, 0], [1, 1]], [-1.0, 1.0, 1.0, -1.0]
    net = impl.MLP(2, [4, 1], seed=0)
    losses = impl.train(net, xs, ys, steps=200, lr=0.1)
    assert len(losses) == 200 and all(isinstance(v, float) for v in losses), (
        "train возвращает список из steps чисел (loss.data на каждом шаге)")
    assert losses[-1] < 0.05, (
        f"после 200 шагов лосс {losses[-1]:.3f} (в начале {losses[0]:.3f}), а эталон доходит до < 0.05. "
        "Если лосс растёт или скачет — проверь знак шага и обнуление градиентов перед каждым backward.")
    preds = [net(x)[0].data for x in xs]
    assert all((p > 0) == (y > 0) for p, y in zip(preds, ys)), f"XOR не выучен: предсказания {preds}"


def test_train_circles(impl):
    """Два концентрических кольца: MLP(2, [8, 1]) разделяет их на 100% за 150 шагов"""
    xs, ys = _data_circles()
    net = impl.MLP(2, [8, 1], seed=0)
    losses = impl.train(net, xs, ys, steps=150, lr=0.1)
    acc = sum((net(x)[0].data > 0) == (y > 0) for x, y in zip(xs, ys)) / len(xs)
    assert acc == 1.0, (
        f"точность {acc:.0%} (лосс {losses[0]:.3f} → {losses[-1]:.3f}). Кольца линейно не разделимы — "
        "нужен нелинейный скрытый слой (tanh) и правильный цикл обучения.")
