"""Тесты «Найди баг: softmax + cross-entropy» — те же, что у задачи «Backprop руками на тензорах», без подсказок про баг.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул — torch.autograd в float64. В коде пользователя autograd и готовые softmax/CE запрещены (autouse-фикстура),
поэтому оракул вызывает сохранённые до запрета функции.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

N, D, H, V = 6, 5, 7, 4
ATOL, RTOL = 1e-10, 1e-8

# Оригиналы — для оракула (фикстура ниже подменяет их на время теста).
_AUTOGRAD_BACKWARD = torch.autograd.backward
_CROSS_ENTROPY = F.cross_entropy


@pytest.fixture(autouse=True)
def _forbid_autograd_and_builtins(monkeypatch):
    def forbid(name: str):
        def f(*args, **kwargs):
            raise AssertionError(
                f"Использован {name}. В этой задаче градиенты считаются руками по формулам, а softmax и "
                "cross-entropy пишутся из exp / sum / log.")
        return f
    monkeypatch.setattr(torch.Tensor, "backward", forbid("tensor.backward()"))
    monkeypatch.setattr(torch.autograd, "backward", forbid("torch.autograd.backward"))
    monkeypatch.setattr(torch.autograd, "grad", forbid("torch.autograd.grad"))
    for mod, names in [(F, ["cross_entropy", "log_softmax", "softmax", "nll_loss"]),
                       (torch, ["softmax", "log_softmax", "logsumexp"]),
                       (torch.Tensor, ["softmax", "log_softmax", "logsumexp"])]:
        for n in names:
            monkeypatch.setattr(mod, n, forbid(f"{getattr(mod, '__name__', 'Tensor')}.{n}"))


def _t(*shape, seed=0, scale=1.0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) * scale


def _targets(n=N, v=V, seed=9):
    return torch.randint(0, v, (n,), generator=torch.Generator().manual_seed(seed))


def _grads(loss, *ts):
    _AUTOGRAD_BACKWARD([loss])
    return [t.grad for t in ts]


def test_linear_forward(impl):
    """linear_forward: y = x Wᵀ + b, форма [N, D_out]"""
    x, W, b = _t(N, D), _t(H, D, seed=1), _t(H, seed=2)
    y, _ = impl.linear_forward(x, W, b)
    assert y.shape == (N, H), f"y должен иметь форму [N, D_out] = {(N, H)}, а получено {tuple(y.shape)}"
    assert_close(y, x @ W.T + b, atol=ATOL, rtol=RTOL, msg="y ≠ x Wᵀ + b")


def test_linear_backward(impl):
    """linear_backward: dx, dW, db совпадают с autograd и имеют формы x, W, b"""
    x, W, b = (_t(N, D).requires_grad_(), _t(H, D, seed=1).requires_grad_(), _t(H, seed=2).requires_grad_())
    dy = _t(N, H, seed=3)
    want = _grads(((x @ W.T + b) * dy).sum(), x, W, b)
    _, cache = impl.linear_forward(x.detach(), W.detach(), b.detach())
    got = impl.linear_backward(dy, cache)
    for name, g, w in zip(["dx", "dW", "db"], got, want):
        if g.shape != w.shape:
            hint = " Похоже, dW транспонирован: dyᵀ @ x даёт [D_out, D_in], как у W." if name == "dW" and g.shape == w.T.shape else ""
            pytest.fail(f"{name}: форма {tuple(g.shape)}, а должна совпадать с формой параметра {tuple(w.shape)}.{hint}",
                        pytrace=False)
        if name == "db" and torch.allclose(g * N, w, atol=1e-9) and N > 1:
            pytest.fail("db в N раз меньше нужного: градиенты строк батча нужно СКЛАДЫВАТЬ (sum), а не усреднять. "
                        "Усреднение по батчу уже учтено в dy.", pytrace=False)
        assert_close(g, w, atol=ATOL, rtol=RTOL, msg=f"{name} расходится с torch.autograd")


def test_relu_backward(impl):
    """relu_backward: градиент проходит только там, где вход > 0"""
    x = _t(N, H, seed=4)
    y, cache = impl.relu_forward(x)
    assert_close(y, x.clamp(min=0), atol=0, rtol=0, msg="relu_forward: max(0, x)")
    dy = _t(N, H, seed=5)
    dx = impl.relu_backward(dy, cache)
    want = dy * (x > 0)
    assert_close(dx, want, atol=0, rtol=0, msg="dx = dy там, где x > 0, и 0 в остальных местах")


def test_softmax_cross_entropy_loss(impl):
    """softmax_cross_entropy: loss == F.cross_entropy (среднее по батчу), скаляр-тензор"""
    logits, y = _t(N, V, seed=6, scale=2.0), _targets()
    loss, _ = impl.softmax_cross_entropy(logits, y)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0, (
        f"loss — тензор-скаляр (форма []), а получено {loss!r}")
    want = _CROSS_ENTROPY(logits, y)
    if torch.allclose(loss, want * N):
        pytest.fail("loss в N раз больше: нужен средний по батчу лосс, а не сумма.", pytrace=False)
    assert_close(loss, want, atol=ATOL, rtol=RTOL, msg="loss ≠ cross-entropy: −mean(log softmax(logits)[i, y_i])")


def test_softmax_cross_entropy_grad(impl):
    """dlogits == autograd: (softmax − onehot) / N; сумма каждой строки = 0"""
    logits, y = _t(N, V, seed=6, scale=2.0).requires_grad_(), _targets()
    (want,) = _grads(_CROSS_ENTROPY(logits, y), logits)
    _, d = impl.softmax_cross_entropy(logits.detach(), y)
    assert d.shape == (N, V), f"dlogits — форма логитов {(N, V)}, а получено {tuple(d.shape)}"
    if torch.allclose(d, want * N, atol=1e-9):
        pytest.fail("dlogits в N раз больше: лосс — среднее по батчу, поэтому градиент делится на N.", pytrace=False)
    if torch.allclose(d, -want, atol=1e-9):
        pytest.fail("dlogits с обратным знаком: градиент −log p по логитам — это p − onehot, а не onehot − p.",
                    pytrace=False)
    assert_close(d, want, atol=ATOL, rtol=RTOL, msg="dlogits ≠ (softmax(logits) − onehot(y)) / N")
    assert_close(d.sum(dim=1), torch.zeros(N, dtype=torch.float64), atol=1e-12, rtol=0,
                 msg="сумма каждой строки dlogits должна быть 0: Σ p = 1 и onehot тоже даёт 1")


def test_softmax_cross_entropy_large_logits(impl):
    """Большие логиты (~1000): loss и градиент конечные и точные"""
    logits = _t(N, V, seed=7, scale=1.0) * 300 + 1000.0
    logits[0, 0] = 2000.0
    y = _targets()
    lg = logits.clone().requires_grad_()
    want_loss = _CROSS_ENTROPY(lg, y)
    (want,) = _grads(want_loss, lg)
    loss, d = impl.softmax_cross_entropy(logits, y)
    if not torch.isfinite(loss) or not torch.isfinite(d).all():
        pytest.fail(
            "на логитах ~1000 loss или dlogits содержат inf/nan.",
            pytrace=False)
    assert_close(loss, want_loss, atol=1e-9, rtol=1e-9, msg="на больших логитах loss неточен")
    assert_close(d, want, atol=1e-12, rtol=1e-8, msg="на больших логитах dlogits неточен")


def test_mlp_loss_and_grads(impl):
    """Двухслойный MLP: loss и градиенты всех параметров == autograd"""
    x, y = _t(N, D, seed=10), _targets()
    params = {"W1": _t(H, D, seed=11, scale=0.5), "b1": _t(H, seed=12, scale=0.1),
              "W2": _t(V, H, seed=13, scale=0.5), "b2": _t(V, seed=14, scale=0.1)}
    leaf = {k: v.clone().requires_grad_() for k, v in params.items()}
    logits = (x @ leaf["W1"].T + leaf["b1"]).clamp(min=0) @ leaf["W2"].T + leaf["b2"]
    want_loss = _CROSS_ENTROPY(logits, y)
    _grads(want_loss, *leaf.values())
    loss, grads = impl.mlp_loss_and_grads(x, y, params)
    assert_close(loss, want_loss.detach(), atol=ATOL, rtol=RTOL, msg="loss MLP расходится с эталоном")
    assert set(grads) == set(params), f"grads должен иметь ключи {sorted(params)}, а получено {sorted(grads)}"
    for k in params:
        assert_close(grads[k], leaf[k].grad, atol=ATOL, rtol=RTOL,
                     msg=f"градиент {k} расходится с autograd. Обратный проход идёт в обратном порядке: "
                         "CE → Linear2 → ReLU → Linear1, и каждый backward получает cache своего forward.")
