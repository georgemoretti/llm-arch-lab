"""Тесты «Блок трансформера: attention, MLP, остатки, pre-LN и post-LN».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.
"""

import pytest
import torch
import torch.nn.functional as F

B, T, C, BLOCK = 2, 8, 32, 16


def _x(seed=0, scale=1.0):
    return scale * torch.randn(B, T, C, generator=torch.Generator().manual_seed(seed))


def _randomize_norms(block, seed=5):
    """Неединичные веса LayerNorm: иначе ln_1 и ln_2 неотличимы."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for ln in (block.ln_1, block.ln_2):
            ln.weight.copy_(1 + 0.3 * torch.randn(C, generator=g))
            ln.bias.copy_(0.1 * torch.randn(C, generator=g))


def _blocks(impl, ref, pre_norm):
    torch.manual_seed(0)
    a = ref.Block(C, BLOCK, pre_norm=pre_norm)
    _randomize_norms(a)
    b = impl.Block(C, BLOCK, pre_norm=pre_norm)
    b.load_state_dict(a.state_dict())
    return a, b


def test_mlp(impl, ref):
    """MLP: C → 4C → GELU(tanh) → C, совпадает с эталоном"""
    torch.manual_seed(0)
    a, b = ref.MLP(C), impl.MLP(C)
    b.load_state_dict(a.state_dict())
    x = _x()
    out = b(x)
    assert out.shape == (B, T, C), f"MLP сохраняет форму [B, T, C], получено {tuple(out.shape)}"
    if torch.allclose(out, a.c_proj(F.gelu(a.c_fc(x))), atol=1e-7) and not torch.allclose(out, a(x), atol=1e-7):
        pytest.fail("использован точный GELU, а в GPT-2 — tanh-аппроксимация: F.gelu(..., approximate='tanh'). "
                    "Разница ~1e-3, но с ней не совпадут веса настоящей GPT-2 (C24, C25).", pytrace=False)
    if torch.allclose(out, a.c_proj(F.relu(a.c_fc(x))), atol=1e-6):
        pytest.fail("вместо GELU использован ReLU: в GPT-2 нелинейность — GELU (tanh-аппроксимация).", pytrace=False)
    torch.testing.assert_close(out, a(x), atol=1e-6, rtol=1e-5, msg="MLP: c_proj(gelu(c_fc(x), approximate='tanh'))")


def _diagnose_pre(b, x, out) -> str:
    def h(f):
        with torch.no_grad():
            return f()
    a, m, n1, n2 = b.attn, b.mlp, b.ln_1, b.ln_2
    variants = {
        "остаток взят от нормализованного входа (n = ln_1(x); n + attn(n)): по остаточной связи должен идти сам x, "
        "иначе сигнал нормализуется в каждом блоке и residual-поток теряет смысл.":
            lambda: (lambda y: (lambda n: n + m(n))(n2(y)))((lambda n: n + a(n))(n1(x))),
        "выполнен post-LN (нормализация суммы) при pre_norm=True: в pre-LN нормализуется только вход подслоя.":
            lambda: n2((lambda y: y + m(y))(n1(x + a(x)))),
        "ln_1 и ln_2 перепутаны: ln_1 — перед attention, ln_2 — перед MLP.":
            lambda: (lambda y: y + m(n1(y)))(x + a(n2(x))),
        "MLP получил тот же нормализованный вход, что attention (параллельный блок): MLP должен видеть x уже с "
        "добавкой attention.":
            lambda: x + a(n1(x)) + m(n2(x)),
        "нет второй остаточной связи: x = mlp(ln_2(x)) вместо x = x + mlp(ln_2(x)).":
            lambda: m(n2(x + a(n1(x)))),
    }
    for msg, f in variants.items():
        if torch.allclose(out, h(f), atol=1e-5):
            return msg
    return "pre-LN: x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))."


def test_pre_norm_block(impl, ref):
    """pre-LN: x + attn(ln_1(x)), затем + mlp(ln_2(·)); совпадает с эталоном"""
    a, b = _blocks(impl, ref, pre_norm=True)
    x = _x(1)
    out = b(x)
    assert out.shape == (B, T, C)
    if not torch.allclose(out, a(x), atol=1e-5, rtol=1e-4):
        pytest.fail(_diagnose_pre(a, x, out), pytrace=False)


def test_post_norm_block(impl, ref):
    """post-LN: ln_1(x + attn(x)), затем ln_2(· + mlp(·)); совпадает с эталоном"""
    a, b = _blocks(impl, ref, pre_norm=False)
    x = _x(2)
    out = b(x)
    with torch.no_grad():
        h = x + a.attn(a.ln_1(x))
        as_pre = h + a.mlp(a.ln_2(h))
    if torch.allclose(out, as_pre, atol=1e-5):
        pytest.fail("при pre_norm=False выполнен pre-LN: режим нужно выбирать по self.pre_norm.", pytrace=False)
    torch.testing.assert_close(out, a(x), atol=1e-5, rtol=1e-4,
                               msg="post-LN: x = ln_1(x + attn(x)); x = ln_2(x + mlp(x))")


def test_zero_sublayers(impl, ref):
    """Нулевые подслои: pre-LN-блок — тождество (x проходит по остаткам как есть), post-LN — нет"""
    for pre in (True, False):
        _, b = _blocks(impl, ref, pre_norm=pre)
        with torch.no_grad():
            b.attn.value.weight.zero_()
            b.mlp.c_proj.weight.zero_()
            b.mlp.c_proj.bias.zero_()
        x = _x(3, scale=5.0)
        out = b(x)
        if pre:
            torch.testing.assert_close(out, x, msg="pre-LN при нулевых attention и MLP должен вернуть x без изменений: "
                                                   "остаточная связь передаёт вход напрямую", atol=1e-6, rtol=1e-5)
        else:
            assert not torch.allclose(out, x, atol=1e-2), "post-LN нормализует сумму, даже при нулевых подслоях выход ≠ x"
            torch.testing.assert_close(out, b.ln_2(b.ln_1(x)), atol=1e-6, rtol=1e-5,
                                       msg="post-LN при нулевых подслоях = ln_2(ln_1(x))")


def test_post_norm_output_is_normalized(impl):
    """post-LN: выход каждой позиции нормализован (среднее 0, std 1); pre-LN сохраняет масштаб residual-потока"""
    torch.manual_seed(0)
    post, pre = impl.Block(C, BLOCK, pre_norm=False), impl.Block(C, BLOCK, pre_norm=True)
    x = _x(4, scale=10.0)
    y = post(x)
    torch.testing.assert_close(y.mean(-1), torch.zeros(B, T), atol=1e-4, rtol=0,
                               msg="у post-LN последняя операция — LayerNorm: среднее по C равно 0")
    torch.testing.assert_close(y.std(-1, unbiased=False), torch.ones(B, T), atol=1e-3, rtol=0)
    assert pre(x).std() > 5, "pre-LN не нормализует residual-поток: вход масштаба 10 остаётся масштаба ~10"


def test_block_is_causal(impl, ref):
    """Выход блока на позиции t не зависит от входов после t"""
    for pre in (True, False):
        _, b = _blocks(impl, ref, pre_norm=pre)
        x = _x(5)
        y = x.clone()
        y[:, 5:] = _x(6)[:, 5:]
        assert torch.allclose(b(x)[:, :5], b(y)[:, :5], atol=1e-5), (
            f"{'pre' if pre else 'post'}-LN блок: выход на позициях 0–4 зависит от входа на позициях 5+ → утечка будущего")


def test_gradient_reaches_input(impl, ref):
    """Остаточная связь: в pre-LN-стеке из 8 блоков градиент по входу не меньше, чем при одном слое"""
    torch.manual_seed(0)
    blocks = [impl.Block(C, BLOCK, pre_norm=True) for _ in range(8)]
    x = _x(7).requires_grad_()
    h = x
    for blk in blocks:
        h = blk(h)
    h.sum().backward()
    g = x.grad
    # у суммы выходов по x вклад остаточного пути — ровно 1 на каждый элемент
    assert g.abs().mean() > 0.5, (
        f"средний |∂L/∂x| = {g.abs().mean():.3f}: по остаточным связям градиент должен доходить до входа почти "
        "без затухания (вклад пути «в обход» равен 1)")
