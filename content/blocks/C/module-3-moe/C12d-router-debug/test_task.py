"""Тесты C12d (= тесты C12, но без подсказок, называющих баг).

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: top-k по оценкам; после перенормировки веса суммируются в 1 (Mixtral, Qwen3, DeepSeek-V3),
без неё — это исходные вероятности (OLMoE, Qwen1.5-MoE); sigmoid-роутер DeepSeek-V3 с множителем 2.5;
маршрут токена не зависит от других токенов в батче; softmax(top-k логитов) == перенормированный
top-k softmax; градиент в роутер при top-1.
"""

import pytest
import torch
from torch.testing import assert_close

N, E, K, C = 24, 8, 2, 16


def _logits(n=N, e=E, seed=0):
    return 2 * torch.randn(n, e, generator=torch.Generator().manual_seed(seed))


def _oracle(logits, k, scoring="softmax", norm=True, scale=1.0):
    s = logits.float().softmax(-1) if scoring == "softmax" else logits.float().sigmoid()
    w, i = s.topk(k, dim=-1)
    if norm:
        w = w / w.sum(-1, keepdim=True)
    return w * scale, i


def _same_routing(w, i, w_ref, i_ref):
    """Сравнение без учёта порядка внутри top-k: сортируем по номеру эксперта."""
    if w.shape != w_ref.shape or i.shape != i_ref.shape:
        return False
    a, b = i.sort(-1), i_ref.sort(-1)
    return torch.equal(a.values, b.values) and torch.allclose(
        w.float().gather(-1, a.indices), w_ref.gather(-1, b.indices), atol=1e-6)


def _diagnose(logits, w, i, k, norm=True) -> str:
    variants = {
        "веса не перенормированы: после top-k их нужно поделить на их сумму (norm_topk_prob=True)":
            _oracle(logits, k, norm=False),
        "веса перенормированы, хотя norm_topk_prob=False: тогда это исходные вероятности softmax":
            _oracle(logits, k, norm=True),
        "выбраны эксперты с НАИМЕНЬШИМИ оценками (topk(..., largest=False) или сортировка по возрастанию)":
            _oracle_smallest(logits, k, norm),
    }
    for hint, (w_c, i_c) in variants.items():
        if _same_routing(w, i, w_c, i_c):
            return f" Похоже, {hint}."
    return ""


def _oracle_axis0(logits, k, norm):
    s = logits.float().softmax(0)
    w, i = s.topk(k, dim=-1)
    return (w / w.sum(-1, keepdim=True) if norm else w), i


def _oracle_smallest(logits, k, norm):
    s = logits.float().softmax(-1)
    w, i = s.topk(k, dim=-1, largest=False)
    return (w / w.sum(-1, keepdim=True) if norm else w), i


def _check(impl, logits, k, scoring="softmax", norm=True, scale=1.0, label=""):
    w, i = impl.route(logits, k, scoring, norm, scale)
    assert tuple(w.shape) == (logits.shape[0], k) and tuple(i.shape) == (logits.shape[0], k), (
        f"{label}: weights {tuple(w.shape)}, indices {tuple(i.shape)}; ожидалось ({logits.shape[0]}, {k}) — "
        "по k экспертов на каждый токен.")
    assert w.dtype == torch.float32, f"{label}: веса должны быть float32 (роутинг считают в fp32), получено {w.dtype}."
    w_ref, i_ref = _oracle(logits, k, scoring, norm, scale)
    if not _same_routing(w, i, w_ref, i_ref):
        hint = _diagnose(logits, w, i, k, norm) if scoring == "softmax" else ""
        pytest.fail(f"{label}: маршрут не совпадает с эталоном.{hint}", pytrace=False)
    return w, i


def test_softmax_topk_renorm(impl):
    """softmax по экспертам → top-2 → перенормировка (Mixtral): веса каждого токена суммируются в 1"""
    w, _ = _check(impl, _logits(), K, label="Mixtral-роутер (softmax, top-2, norm)")
    assert_close(w.sum(-1), torch.ones(N), msg=lambda _: "Сумма весов выбранных экспертов должна быть 1.")


def test_no_renorm(impl):
    """norm_topk_prob=False (OLMoE, Qwen1.5-MoE): веса — исходные вероятности, сумма < 1"""
    logits = _logits(seed=1)
    w, _ = _check(impl, logits, 4, norm=False, label="softmax, top-4, без перенормировки")
    assert (w.sum(-1) < 1).all(), "Без перенормировки сумма top-4 вероятностей из 8 меньше 1."


def test_indices_are_topk_of_logits(impl):
    """Выбранные эксперты — k лучших по логитам (softmax и sigmoid монотонны — порядок тот же)"""
    logits = _logits(seed=2)
    for scoring in ("softmax", "sigmoid"):
        _, i = impl.route(logits, 3, scoring, True, 1.0)
        expected = logits.topk(3, dim=-1).indices
        assert torch.equal(i.sort(-1).values, expected.sort(-1).values), (
            f"scoring={scoring}: выбраны не 3 эксперта с наибольшими логитами.{_diagnose(logits, *impl.route(logits, 3), 3)}")


def test_sigmoid_deepseek_v3(impl):
    """DeepSeek-V3: sigmoid-оценки, top-8 из 256, перенормировка, × routed_scaling_factor = 2.5"""
    # Источник: deepseek-ai/DeepSeek-V3 config.json: n_routed_experts = 256, num_experts_per_tok = 8,
    # scoring_func = "sigmoid", norm_topk_prob = true, routed_scaling_factor = 2.5.
    logits = _logits(n=6, e=256, seed=3)
    w, _ = _check(impl, logits, 8, "sigmoid", True, 2.5, label="DeepSeek-V3 (sigmoid, top-8, norm, ×2.5)")
    assert_close(w.sum(-1), torch.full((6,), 2.5),
                 msg=lambda _: "После перенормировки и умножения на routed_scaling_factor сумма весов = 2.5.")
    w2, _ = _check(impl, logits, 8, "sigmoid", False, 1.0, label="sigmoid без перенормировки")
    assert ((w2 > 0) & (w2 < 1)).all(), "sigmoid-оценки лежат в (0, 1) и не обязаны суммироваться в 1."


def test_routing_is_per_token(impl):
    """Маршрут токена не зависит от остальных токенов в батче"""
    logits = _logits()
    w_all, i_all = impl.route(logits, K)
    w_one, i_one = impl.route(logits[5:6], K)
    other = logits.clone()
    other[6:] = 5 * torch.randn(N - 6, E, generator=torch.Generator().manual_seed(9))
    w_mix, i_mix = impl.route(other, K)
    ok = _same_routing(w_one, i_one, w_all[5:6].float(), i_all[5:6]) and _same_routing(w_mix[5:6], i_mix[5:6],
                                                                                        w_all[5:6].float(), i_all[5:6])
    if not ok:
        pytest.fail("Маршрут токена 5 изменился, когда поменялись ДРУГИЕ токены батча (или когда токен подали один). "
                    f"Роутинг каждого токена — функция только его логитов.{_diagnose(logits, w_all, i_all, K)}",
                    pytrace=False)


def test_softmax_of_topk_equivalence(impl):
    """softmax(top-k логитов) == перенормированный top-k от softmax по всем экспертам"""
    logits = _logits(seed=4)
    w, i = impl.route(logits, K)
    top = logits.topk(K, dim=-1)
    alt = top.values.softmax(-1)
    assert _same_routing(w, i, alt, top.indices), (
        "Перенормированные веса top-k должны совпасть с softmax, взятым только по k выбранным логитам: "
        "общий знаменатель softmax сокращается при перенормировке. (Так и было в исходном коде Mixtral: "
        "сначала top-k по логитам, потом softmax.)")


def test_router_module(impl):
    """TopKRouter: x [B, T, C] → logits [B·T, E] = x·Wᵀ в fp32, и маршрут по ним"""
    torch.manual_seed(0)
    r = impl.TopKRouter(C, E, K)
    assert tuple(r.weight.shape) == (E, C), "Конструктор дан: weight [E, C] (как gate.weight в HF)."
    x = torch.randn(2, 5, C)
    logits, w, i = r(x)
    assert tuple(logits.shape) == (10, E), (
        f"logits формы {tuple(logits.shape)}, ожидалось (10, {E}): токены батча склеиваются в одну ось N = B·T.")
    assert_close(logits, x.reshape(-1, C) @ r.weight.T, atol=1e-5, rtol=1e-5,
                 msg=lambda _: "logits = x · Wᵀ (линейный слой без bias).")
    w_ref, i_ref = _oracle(logits.detach(), K)
    assert _same_routing(w.detach(), i, w_ref, i_ref), "Веса и индексы роутера должны быть route(logits, …)."
    logits16, _, _ = r(x.to(torch.bfloat16))
    assert logits16.dtype == torch.float32, "Логиты роутера считаются в float32 даже при bf16-входе."


def test_gradient_reaches_router(impl):
    """Градиент к весам роутера идёт через веса top-k; при top-1 с перенормировкой он равен нулю"""
    torch.manual_seed(0)
    x = torch.randn(16, C)
    target = torch.randn(16, 1)
    grads = {}
    for norm in (True, False):
        r = impl.TopKRouter(C, E, 1, norm_topk_prob=norm)
        _, w, _ = r(x)
        if not w.requires_grad:
            pytest.fail("Веса роутера не связаны с графом вычислений (requires_grad = False): градиент до weight "
                        "роутера не дойдёт и он не будет учиться. Нет ли .detach(), torch.no_grad() или "
                        "пересоздания тензора (torch.tensor(...)) на пути logits → weights?", pytrace=False)
        (w * target).sum().backward()
        grads[norm] = r.weight.grad.abs().max().item() if r.weight.grad is not None else 0.0
    assert grads[False] > 1e-6, (
        "top-1 без перенормировки: градиент до weight роутера не дошёл. Веса должны вычисляться из логитов "
        "дифференцируемо (softmax → gather/topk), без .detach() и без пересоздания тензоров.")
    assert grads[True] < 1e-6, (
        "top-1 с перенормировкой: вес всегда ровно 1, поэтому градиент к роутеру — ноль. Если он не ноль, "
        "перенормировка делается не так. Именно поэтому Switch Transformer (top-1) не перенормирует вес.")
