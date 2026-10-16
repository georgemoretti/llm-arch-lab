"""Тесты C13d (= тесты C13, но без подсказок, называющих баг).

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: forward_naive и forward совпадают с оракулом (взвешенная сумма top-k экспертов); каждый
токен обработан ровно своими top-k экспертами, эксперт видит только свои токены (Σ строк = N·k, а не
N·E); невыбранные эксперты не вызываются и не получают градиент; выход токена не зависит от батча.
"""

import pytest
import torch
from torch.testing import assert_close

B, T, C, F_EXP, E, K = 2, 10, 16, 24, 6, 2


def _x(seed: int = 1, t: int = T):
    return torch.randn(B, t, C, generator=torch.Generator().manual_seed(seed))


def _make(impl, seed=0, k=K, **kw):
    torch.manual_seed(seed)
    m = impl.MoELayer(C, F_EXP, E, k, **kw).eval()
    with torch.no_grad():
        m.gate.weight.mul_(3)              # более «решительный» роутер — веса заметно разные
    return m


def _oracle(m, x, keep="all"):
    """Взвешенная сумма выходов top-k экспертов. keep='last' — только эксперт с наибольшим номером."""
    h = x.reshape(-1, C)
    with torch.no_grad():
        _, w, idx = m.gate(h)
        out = torch.zeros_like(h)
        for n in range(h.shape[0]):
            slots = range(idx.shape[1])
            if keep == "last":
                slots = [int(idx[n].argmax())]
            elif keep == "unweighted":
                pass
            for j in slots:
                e = int(idx[n, j])
                y = m.experts[e](h[n:n + 1])[0]
                out[n] += y if keep == "unweighted" else w[n, j] * y
    return out.reshape(x.shape)


def _diagnose(m, x, out) -> str:
    variants = {
        "выходы экспертов не умножены на веса роутера": _oracle(m, x, keep="unweighted"),
    }
    for hint, cand in variants.items():
        if torch.allclose(out, cand, atol=1e-5, rtol=1e-4):
            return f" Похоже, {hint}."
    return ""


def _check(impl, method):
    m = _make(impl)
    x = _x()
    with torch.no_grad():
        res = getattr(m, method)(x)
    assert isinstance(res, tuple) and len(res) == 2, f"{method} должен вернуть пару (out, router_logits)."
    out, logits = res
    assert tuple(out.shape) == (B, T, C), f"{method}: out формы {tuple(out.shape)}, ожидалось {(B, T, C)}."
    assert tuple(logits.shape) == (B * T, E), (
        f"{method}: router_logits формы {tuple(logits.shape)}, ожидалось {(B * T, E)} — они понадобятся для aux loss (C14).")
    expected = _oracle(m, x)
    if not torch.allclose(out, expected, atol=1e-5, rtol=1e-4):
        pytest.fail(f"{method}: выход не совпадает с взвешенной суммой top-{K} экспертов.{_diagnose(m, x, out)}",
                    pytrace=False)


def test_naive_matches_oracle(impl):
    """forward_naive: все эксперты на всех токенах, смешивание плотной матрицей весов [N, E]"""
    _check(impl, "forward_naive")


def test_forward_matches_oracle(impl):
    """forward (через сортировку): тот же результат, что взвешенная сумма top-k экспертов"""
    _check(impl, "forward")


def test_forward_equals_naive_variants(impl):
    """forward == forward_naive при top-1, top-3, sigmoid-роутере с ×2.5 и без перенормировки"""
    x = _x(seed=2)
    for kw in (dict(k=1), dict(k=3), dict(k=2, scoring_func="sigmoid", routed_scaling_factor=2.5),
               dict(k=4, norm_topk_prob=False)):
        m = _make(impl, seed=1, **kw)
        with torch.no_grad():
            a, _ = m(x)
            b, _ = m.forward_naive(x)
        assert_close(a, b, atol=1e-5, rtol=1e-4,
                     msg=lambda _: f"{kw}: forward и forward_naive расходятся.{_diagnose(m, x, a)}")


def _record_inputs(m):
    seen = {e: [] for e in range(E)}
    hooks = [m.experts[e].register_forward_hook(lambda mod, inp, out, e=e: seen[e].append(inp[0].detach()))
             for e in range(E)]
    return seen, hooks


def test_each_expert_sees_only_its_tokens(impl):
    """Эксперт e получает ровно токены, выбравшие e; всего N·k строк, а не N·E"""
    m = _make(impl)
    x = _x()
    h = x.reshape(-1, C)
    seen, hooks = _record_inputs(m)
    with torch.no_grad():
        m(x)
        _, _, idx = m.gate(h)
    for hk in hooks:
        hk.remove()
    total = sum(sum(t.shape[0] for t in v) for v in seen.values())
    assert total == B * T * K, (
        f"Эксперты в сумме обработали {total} строк, а должно быть N·k = {B * T}·{K} = {B * T * K}. "
        + ("Похоже, каждый эксперт считает все токены — это forward_naive, а forward должен отдавать эксперту "
           "только его токены." if total == B * T * E else ""))
    for e in range(E):
        expected = sorted((idx == e).any(-1).nonzero().flatten().tolist())
        rows = torch.cat(seen[e]) if seen[e] else torch.empty(0, C)
        got = sorted(int((h == r).all(-1).nonzero()[0]) for r in rows)
        assert got == expected, (
            f"Эксперт {e} получил токены {got}, а выбрали его токены {expected}. Сгруппируй пары (токен, эксперт) "
            "по эксперту и отдавай эксперту строки h[токены этой группы].")


def test_unused_experts_untouched(impl):
    """Эксперт, которого не выбрал ни один токен, не вызывается и не получает градиент"""
    m = _make(impl, k=1)
    with torch.no_grad():
        m.gate.weight[3] = -50 * m.gate.weight[3].abs().sum() * torch.ones(C) / C   # эксперт 3 никому не нужен
    x = _x(seed=4).abs()                                                          # положительный вход
    seen, hooks = _record_inputs(m)
    _, _, idx = m.gate(x.reshape(-1, C))
    assert not (idx == 3).any(), "внутренняя ошибка теста: эксперт 3 оказался выбран"
    out, _ = m(x)
    for hk in hooks:
        hk.remove()
    assert not seen[3], (
        "Эксперт 3 не выбран ни одним токеном, но был вызван. Пропускай экспертов с нулём токенов — на больших "
        "моделях это экономит вычисления (и не вызывает эксперта на пустом тензоре).")
    out.sum().backward()
    g = m.experts[3].gate_proj.weight.grad
    assert g is None or g.abs().max() == 0, "Невыбранный эксперт получил градиент."
    used = int(idx[0, 0])
    assert m.experts[used].gate_proj.weight.grad is not None, f"Выбранный эксперт {used} не получил градиент."
    assert m.gate.weight.grad is not None and m.gate.weight.grad.abs().max() > 0, (
        "Градиент не дошёл до роутера: веса роутера должны умножать выходы экспертов (роутер учится только "
        "через них).")


def test_batch_independence(impl):
    """Выход токена не зависит от того, с какими токенами он в одном батче"""
    m = _make(impl)
    x = _x()
    with torch.no_grad():
        full, _ = m(x)
        part, _ = m(x[:1, 3:7])
    assert_close(part, full[:1, 3:7], atol=1e-5, rtol=1e-4,
                 msg=lambda _: "Выход токенов 3..6 первого примера изменился, когда их подали отдельно от остальных. "
                               "Проверь, что номера токенов слотов (repeat_interleave) и вклады возвращаются на свои места.")
