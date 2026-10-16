"""Тесты C08: размещение нормализаций.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства:
- каждый блок совпадает со своей формулой (нормы с неединичными весами — чтобы было видно, какая где);
- блок с «выключенными» подслоями (нулевые o_proj и down_proj) — тождество у Pre / OLMo 2 / sandwich
  (остаток идёт напрямую), но не у Post-Norm;
- Post-Norm держит RMS остаточного потока = 1; у OLMo 2 и sandwich каждый подслой добавляет
  вектор с RMS ≈ 1, поэтому после L блоков RMS ≈ √(1 + 2L);
- градиент при инициализации (Xiong et al., 2020): у Post-Norm градиент весов последнего блока не
  зависит от глубины, у Pre-Norm убывает примерно как 1/√L.
"""

import math

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from llmlab.core.llama import RMSNorm

B, T, C, H, H_KV, F_HID = 2, 12, 32, 4, 2, 64
CLASSES = ["PostNormBlock", "PreNormBlock", "OlmoPostNormBlock", "SandwichNormBlock"]


def _x(seed: int = 1, b: int = B):
    return torch.randn(b, T, C, generator=torch.Generator().manual_seed(seed))


def _make(impl, name, seed=0, random_norms=True):
    torch.manual_seed(seed)
    m = getattr(impl, name)(C, H, H_KV, F_HID).eval()
    if random_norms:
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, RMSNorm):
                    mod.weight.copy_(0.5 + torch.rand(C))      # разные нормы → разные веса
    return m


def _candidates(m, x, name):
    """Варианты формулы для блока `name` — правильный и типичные ошибки — на весах модуля m."""
    A, Fn = m.self_attn, m.mlp
    if name == "SandwichNormBlock":
        n1, n2, n3, n4 = (m.input_layernorm, m.post_attention_layernorm,
                          m.pre_feedforward_layernorm, m.post_feedforward_layernorm)
        h = x + n2(A(n1(x)))
        good = h + n4(Fn(n3(h)))
        h1 = x + A(n1(x))
        h2 = x + n2(A(x))
        h3 = n2(x + A(n1(x)))
        return good, {
            "нет норм на выходе подслоёв (post_attention_layernorm / post_feedforward_layernorm) — "
            "получился обычный Pre-Norm": h1 + Fn(n3(h1)),
            "нет норм на входе подслоёв (input_layernorm / pre_feedforward_layernorm) — получился OLMo 2":
                h2 + n4(Fn(h2)),
            "выходная норма применена к сумме с остатком (n(x + подслой)), а не к выходу подслоя — "
            "остаток должен идти мимо обеих норм": n4(h3 + Fn(n3(h3))),
        }
    na = m.post_attention_layernorm if name != "PreNormBlock" else m.input_layernorm
    nf = m.post_feedforward_layernorm if name != "PreNormBlock" else m.post_attention_layernorm
    forms = {
        "post": lambda: (lambda h: nf(h + Fn(h)))(na(x + A(x))),
        "pre": lambda: (lambda h: h + Fn(nf(h)))(x + A(na(x))),
        "olmo": lambda: (lambda h: h + nf(Fn(h)))(x + na(A(x))),
        "pre_res_from_norm": lambda: (lambda h: nf(h) + Fn(nf(h)))(na(x) + A(na(x))),
        "no_residual": lambda: nf(Fn(na(A(x)))),
        "one_norm": lambda: (lambda h: h + Fn(na(h)))(x + A(na(x))),
    }
    good_key = {"PostNormBlock": "post", "PreNormBlock": "pre", "OlmoPostNormBlock": "olmo"}[name]
    msgs = {
        "post": "норма стоит на сумме с остатком: n(x + подслой(x)) — это Post-Norm из оригинального Transformer",
        "pre": "норма стоит на ВХОДЕ подслоя: x + подслой(n(x)) — это Pre-Norm (GPT-2, Llama)",
        "olmo": "норма стоит на ВЫХОДЕ подслоя внутри остатка: x + n(подслой(x)) — это вариант OLMo 2",
        "pre_res_from_norm": "остаток берётся от нормированного входа: n(x) + подслой(n(x)) вместо x + подслой(n(x))",
        "no_residual": "нет остаточных соединений",
        "one_norm": "одна и та же норма использована перед обоими подслоями",
    }
    return forms[good_key](), {msgs[k]: f() for k, f in forms.items() if k != good_key}


def _check_formula(impl, name):
    m = _make(impl, name)
    x = _x()
    with torch.no_grad():
        out = m(x)
        good, wrong = _candidates(m, x, name)
    assert tuple(out.shape) == (B, T, C), f"{name}: выход формы {tuple(out.shape)}, ожидалось {(B, T, C)}."
    if not torch.allclose(out, good, atol=1e-5, rtol=1e-4):
        hint = ""
        for msg, cand in wrong.items():
            if torch.allclose(out, cand, atol=1e-5, rtol=1e-4):
                hint = f" Похоже, {msg}."
                break
        pytest.fail(f"{name} не совпадает со своей формулой (см. таблицу в README).{hint}", pytrace=False)


def test_post_norm_formula(impl):
    """PostNormBlock: x = n(x + attn(x)); x = n(x + mlp(x))"""
    _check_formula(impl, "PostNormBlock")


def test_pre_norm_formula(impl):
    """PreNormBlock: x = x + attn(n(x)); x = x + mlp(n(x))"""
    _check_formula(impl, "PreNormBlock")


def test_olmo_formula(impl):
    """OlmoPostNormBlock: x = x + n(attn(x)); x = x + n(mlp(x))"""
    _check_formula(impl, "OlmoPostNormBlock")


def test_sandwich_formula(impl):
    """SandwichNormBlock: x = x + n(attn(n(x))); x = x + n(mlp(n(x)))"""
    _check_formula(impl, "SandwichNormBlock")


def _zero_sublayers(m):
    with torch.no_grad():
        m.self_attn.o_proj.weight.zero_()
        m.mlp.down_proj.weight.zero_()
    return m


def test_identity_path(impl):
    """Подслои выдают 0 → Pre / OLMo 2 / sandwich — тождество; Post-Norm — нет (нормирует остаток)"""
    x = 3 * _x()
    for name in CLASSES[1:]:
        m = _zero_sublayers(_make(impl, name))
        with torch.no_grad():
            out = m(x)
        assert_close(out, x, atol=1e-5, rtol=1e-5,
                     msg=lambda _: f"{name}: если оба подслоя выдают ноль, блок должен вернуть вход как есть — "
                                   "остаточный поток идёт мимо всех норм. Здесь он изменился: норма стоит на "
                                   "пути остатка (или остатка нет).")
    m = _zero_sublayers(_make(impl, "PostNormBlock", random_norms=False))
    with torch.no_grad():
        out = m(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4,
                 msg=lambda _: "PostNormBlock: даже с нулевыми подслоями выход — нормированный вход (RMS = 1 по "
                               "каждому токену), потому что норма стоит на сумме с остатком. Здесь RMS другой.")


def _stack_rms(impl, name, L, seed=0):
    torch.manual_seed(seed)
    blocks = [getattr(impl, name)(C, H, H_KV, F_HID).eval() for _ in range(L)]
    h = _x(seed=seed + 10, b=4)
    with torch.no_grad():
        for blk in blocks:
            h = blk(h)
    return h.pow(2).mean().sqrt().item()


def test_residual_stream_growth(impl):
    """RMS остатка после 16 блоков: Post-Norm = 1; OLMo 2 и sandwich ≈ √(1 + 2L); Pre-Norm — растёт"""
    L = 16
    post = _stack_rms(impl, "PostNormBlock", L)
    assert abs(post - 1) < 1e-3, (
        f"Post-Norm: после {L} блоков RMS остаточного потока = {post:.3f}, а последняя операция блока — "
        "норма, значит RMS = 1. Норма не на сумме с остатком?")
    target = math.sqrt(1 + 2 * L)
    for name in ("OlmoPostNormBlock", "SandwichNormBlock"):
        r = _stack_rms(impl, name, L)
        assert 0.85 * target < r < 1.15 * target, (
            f"{name}: после {L} блоков RMS остатка = {r:.2f}, ожидалось ≈ √(1 + 2·{L}) = {target:.2f}. Каждый "
            "подслой добавляет к остатку нормированный выход (RMS ≈ 1, почти ортогональный остатку), так что "
            "квадрат RMS растёт на 1 за подслой. Если RMS ≈ 1 — норма стоит на сумме; если сильно больше — "
            "выходной нормы нет.")
    pre = _stack_rms(impl, "PreNormBlock", L)
    assert 1.05 < pre < target, (
        f"Pre-Norm: RMS остатка после {L} блоков = {pre:.2f}. Остаток не нормируется, так что он растёт "
        f"(> 1), но медленнее, чем у OLMo 2 (≈ {target:.1f}): выходы подслоёв не нормированы и при "
        "инициализации малы.")


def _last_block_grad(impl, name, L, seed):
    torch.manual_seed(seed)
    blocks = nn.ModuleList(getattr(impl, name)(C, H, H_KV, F_HID) for _ in range(L))
    for mod in blocks.modules():
        if isinstance(mod, nn.Linear):
            nn.init.xavier_normal_(mod.weight)          # инициализация из анализа Xiong et al.
    h = _x(seed=seed + 10, b=4)
    for blk in blocks:
        h = blk(h)
    if name == "PreNormBlock":
        h = RMSNorm(C)(h)                                # у Pre-Norm-моделей есть финальная норма
    target = torch.randn(h.shape, generator=torch.Generator().manual_seed(100))
    (h * target).sum().backward()
    return math.sqrt(sum(p.grad.pow(2).sum().item() for p in blocks[-1].parameters()
                         if p.grad is not None and p.dim() == 2))


def test_gradient_at_init(impl):
    """Xiong et al.: градиент последнего блока у Post-Norm не зависит от глубины, у Pre-Norm падает ~ 1/√L"""
    seeds = range(3)
    g = {(n, L): sum(_last_block_grad(impl, n, L, s) for s in seeds) / len(seeds)
         for n in ("PostNormBlock", "PreNormBlock") for L in (1, 16)}
    post_ratio = g["PostNormBlock", 1] / g["PostNormBlock", 16]
    pre_ratio = g["PreNormBlock", 1] / g["PreNormBlock", 16]
    assert 0.7 < post_ratio < 1.4, (
        f"Post-Norm: ‖∇W‖ последнего блока при L=1 и L=16 отличаются в {post_ratio:.2f} раза, ожидалось ≈ 1. "
        "Норма на выходе каждого блока «отрезает» вклад глубины: градиент у выхода остаётся большим при любой "
        "глубине — поэтому Post-Norm нужен warmup learning rate.")
    assert pre_ratio > 2, (
        f"Pre-Norm: ‖∇W‖ последнего блока при L=1 и L=16 отличаются лишь в {pre_ratio:.2f} раза, ожидалось > 2 "
        "(теория: ~√16 = 4). Остаток не нормируется и растёт с глубиной, а финальная норма делит на его RMS — "
        "градиенты верхних слоёв уменьшаются, и обучение стабильнее без warmup.")
