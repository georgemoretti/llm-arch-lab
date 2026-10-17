"""Тесты C16: Multi-head Latent Attention.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: forward совпадает с оракулом (K и V всех голов восстановлены из латента); в кеше лежит
именно латент [B, S, r], а не K/V (память в 2·H·d/r раз меньше); генерация с кешем == полный прогон;
forward_absorbed даёт тот же выход, не вызывая kv_b_proj (поглощение W_UK и W_UV).
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from llmlab.core.llama import rms_norm

B, T, C, H, R, D, DV = 2, 10, 32, 4, 8, 8, 6
_SDPA = F.scaled_dot_product_attention


def _x(t: int = T, seed: int = 1):
    return torch.randn(B, t, C, generator=torch.Generator().manual_seed(seed))


def _make(impl, seed=0):
    torch.manual_seed(seed)
    m = impl.MLAttention(C, H, R, D, DV).eval()
    with torch.no_grad():
        m.kv_a_layernorm.weight.copy_(0.5 + torch.rand(R))
    return m


def _latent(m, x):
    return rms_norm(m.kv_a_proj(x), m.kv_a_layernorm.weight, m.kv_a_layernorm.eps)


def _oracle(m, x):
    t = x.shape[1]
    with torch.no_grad():
        q = m.q_proj(x).view(B, t, H, D).transpose(1, 2)
        kv = m.kv_b_proj(_latent(m, x)).view(B, t, H, D + DV).transpose(1, 2)
        k, v = kv.split([D, DV], dim=-1)
        out = _SDPA(q, k, v, is_causal=True)
        return m.o_proj(out.transpose(1, 2).reshape(B, t, H * DV))


def _run_chunks(impl, m, x, chunks, method="forward"):
    cache = impl.LatentCache()
    outs, s = [], 0
    with torch.no_grad():
        for c in chunks:
            outs.append(getattr(m, method)(x[:, s:s + c], cache))
            s += c
    return torch.cat(outs, 1), cache


def _first_bad(out, expected):
    bad = ((out - expected).abs() > 1e-4).any(-1).any(0)
    return int(bad.nonzero()[0]) if bad.any() else None


# ---------------------------------------------------------------------------
# LatentCache
# ---------------------------------------------------------------------------

def test_latent_cache_update(impl):
    """LatentCache.update: латенты [B, S, r] дописываются по оси времени (dim=1)"""
    cache = impl.LatentCache()
    assert cache.seq_len == 0
    a, b = torch.randn(B, 3, R), torch.randn(B, 1, R)
    c1, _ = cache.update(a)
    c2, kr = cache.update(b)
    assert tuple(c2.shape) == (B, 4, R), (
        f"После 3 + 1 токенов кеш формы {tuple(c2.shape)}, ожидалось {(B, 4, R)}: латенты лежат как [B, S, r], "
        "время — ось 1 (не 2, как у K/V в C05).")
    assert_close(c2, torch.cat((a, b), 1), atol=0, rtol=0, msg=lambda _: "Порядок: сначала старые токены, потом новые.")
    assert kr is None and cache.seq_len == 4, "Без k_rope второе значение — None; seq_len = 4."
    other = impl.LatentCache()
    assert other.seq_len == 0, "Два кеша не должны делить состояние (храни данные в атрибутах экземпляра)."


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

def test_forward_matches_oracle(impl):
    """forward без кеша: q из x, K и V всех голов — из латента c = norm(W_dkv x), causal"""
    m = _make(impl)
    x = _x()
    with torch.no_grad():
        out = m(x)
    assert tuple(out.shape) == (B, T, C), f"Выход формы {tuple(out.shape)}, ожидалось {(B, T, C)}."
    expected = _oracle(m, x)
    if not torch.allclose(out, expected, atol=1e-5, rtol=1e-4):
        hint = ""
        with torch.no_grad():
            q = m.q_proj(x).view(B, T, H, D).transpose(1, 2)
            kv = m.kv_b_proj(m.kv_a_proj(x)).view(B, T, H, D + DV).transpose(1, 2)
            k, v = kv.split([D, DV], -1)
            no_norm = m.o_proj(_SDPA(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, H * DV))
            kv2 = m.kv_b_proj(_latent(m, x)).view(B, T, H, DV + D).transpose(1, 2)
            v2, k2 = kv2.split([DV, D], -1)
        if torch.allclose(out, no_norm, atol=1e-5):
            hint = " Похоже, латент не нормирован: c = kv_a_layernorm(kv_a_proj(x))."
        elif out.shape == expected.shape:
            hint = (" Проверь разбиение kv_b_proj(c): [B, S, H·(d + dv)] → view(B, S, H, d + dv) → transpose(1, 2) → "
                    "split([d, dv]) — сначала d каналов ключа, потом dv каналов значения, внутри КАЖДОЙ головы.")
        pytest.fail(f"MLA без кеша не совпадает с оракулом.{hint}", pytrace=False)


def test_cache_stores_latent(impl):
    """Prefill: в кеше латент c [B, T, r] — не K и не V; это и есть экономия памяти MLA"""
    m = _make(impl)
    x = _x()
    _, cache = _run_chunks(impl, m, x, [T])
    assert cache.c is not None and tuple(cache.c.shape) == (B, T, R), (
        f"В кеше cache.c формы {None if cache.c is None else tuple(cache.c.shape)}, ожидалось {(B, T, R)}. "
        f"В MLA кешируется латент (r = {R} чисел на токен), а не восстановленные K и V "
        f"(H·(d + dv) = {H * (D + DV)} чисел) — иначе MLA теряет весь смысл.")
    with torch.no_grad():
        assert_close(cache.c, _latent(m, x), atol=1e-5, rtol=1e-4,
                     msg=lambda _: "В кеше должен лежать c = kv_a_layernorm(kv_a_proj(x)) — нормированный латент.")


def _check_chunks(impl, chunks, method="forward"):
    m = _make(impl)
    x = _x()
    out, _ = _run_chunks(impl, m, x, chunks, method)
    expected = _oracle(m, x)
    if not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        pytest.fail(f"{method} с кешем, куски {chunks}: расхождение с полным прогоном с позиции "
                    f"{_first_bad(out, expected)}. Проверь, что новые k, v восстанавливаются из ВСЕХ латентов кеша "
                    "и маска [T, S] сдвинута на S − T (как в C05).", pytrace=False)


def test_token_by_token(impl):
    """Генерация по одному токену с латентным кешем == полный прогон"""
    _check_chunks(impl, [1] * T)


def test_chunked(impl):
    """Prefill 4 токена, потом куски 1 + 5 == полный прогон"""
    _check_chunks(impl, [4, 1, 5])


# ---------------------------------------------------------------------------
# forward_absorbed
# ---------------------------------------------------------------------------

def test_absorbed_equals_forward(impl):
    """forward_absorbed == forward — и без кеша, и при генерации по токену"""
    m = _make(impl)
    x = _x()
    with torch.no_grad():
        a, b = m(x), m.forward_absorbed(x)
    assert_close(b, a, atol=1e-5, rtol=1e-4,
                 msg=lambda _: "forward_absorbed не совпал с forward. q·k = q·(W_uk c) = (W_ukᵀ q)·c — проверь, что "
                               "q умножается на W_uk своей головы (einsum 'bhtd,hdr->bhtr'), масштаб тот же 1/√d, а "
                               "выход o_lat [B, H, T, r] переводится в dv через W_uv своей головы.")
    _check_chunks(impl, [3, 1, 1, 5], method="forward_absorbed")


def test_absorbed_does_not_expand_kv(impl):
    """forward_absorbed не вызывает kv_b_proj: K и V всех голов не материализуются"""
    m = _make(impl)
    calls = []
    hook = m.kv_b_proj.register_forward_hook(lambda *a: calls.append(1))
    with torch.no_grad():
        m.forward_absorbed(_x())
    hook.remove()
    assert not calls, (
        "forward_absorbed вызвал kv_b_proj(...) — то есть восстановил K и V всех голов. Смысл поглощения — "
        "использовать только веса kv_b_proj.weight: свернуть W_uk в запрос и W_uv в выход, а attention считать "
        "по латентам c [B, S, r].")


def test_no_future_leak(impl):
    """Causal: изменение токенов после t не меняет выход в позициях ≤ t"""
    m = _make(impl)
    x = _x()
    x2 = x.clone()
    x2[:, 6:] += 3 * torch.randn_like(x2[:, 6:])
    with torch.no_grad():
        delta = (m(x)[:, :6] - m(x2)[:, :6]).abs().max().item()
    assert delta < 1e-5, f"Выход в позициях 0..5 зависит от токенов 6+ (|Δ| = {delta:.3g}) — нет causal-маски."
