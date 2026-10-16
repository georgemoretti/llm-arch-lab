"""Тесты C16b: MLA с decoupled RoPE.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: совпадение с оракулом (ключ головы = [k_nope из латента ; общий k_rope], масштаб
1/√(dn + dr)); в кеше латент [B, S, r] и уже повёрнутые общие RoPE-ключи [B, S, dr] без оси голов;
генерация с кешем == полный прогон; память кеша DeepSeek-V3 в ~71 раз меньше, чем у MHA.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from llmlab.core.llama import apply_rope, rms_norm, rope_cos_sin
from llmlab.core.mla import LatentCache

B, T, C, H, R, DN, DR, DV = 2, 10, 32, 4, 8, 6, 4, 6
THETA = 10000.0
_SDPA = F.scaled_dot_product_attention


def _x(t: int = T, seed: int = 1):
    return torch.randn(B, t, C, generator=torch.Generator().manual_seed(seed))


def _make(impl, seed=0):
    torch.manual_seed(seed)
    return impl.MLARopeAttention(C, H, R, DN, DR, DV, rope_theta=THETA).eval()


def _sim(m, x, chunks, pos_offset=True, rope_k=True, scale="full"):
    """Оракул с кешем и переключателями типичных ошибок."""
    cs, krs, outs, start = [], [], [], 0
    with torch.no_grad():
        for t in chunks:
            xs = x[:, start:start + t]
            pos = torch.arange(start, start + t) if pos_offset else torch.arange(t)
            cos, sin = rope_cos_sin(pos, DR, THETA)
            q = m.q_proj(xs).view(B, t, H, DN + DR).transpose(1, 2)
            q_nope, q_rope = q.split([DN, DR], -1)
            q_rope = apply_rope(q_rope, cos, sin)
            c, kr = m.kv_a_proj_with_mqa(xs).split([R, DR], -1)
            c = rms_norm(c, m.kv_a_layernorm.weight, m.kv_a_layernorm.eps)
            if rope_k:
                kr = apply_rope(kr.unsqueeze(1), cos, sin).squeeze(1)
            cs.append(c)
            krs.append(kr)
            C_all, K_all = torch.cat(cs, 1), torch.cat(krs, 1)
            S = C_all.shape[1]
            kv = m.kv_b_proj(C_all).view(B, S, H, DN + DV).transpose(1, 2)
            k_nope, v = kv.split([DN, DV], -1)
            k = torch.cat((k_nope, K_all.unsqueeze(1).expand(B, H, S, DR)), -1)
            qq = torch.cat((q_nope, q_rope), -1)
            sc = 1 / math.sqrt(DN + DR) if scale == "full" else 1 / math.sqrt(DN)
            mask = torch.ones(t, S, dtype=torch.bool).tril(S - t)
            out = _SDPA(qq, k, v, attn_mask=mask, scale=sc)
            outs.append(m.o_proj(out.transpose(1, 2).reshape(B, t, H * DV)))
            start += t
    return torch.cat(outs, 1)


def _diagnose(m, x, chunks, out) -> str:
    variants = {
        "позиции RoPE на каждом шаге начинаются с 0 — новые токены стоят после кеша: start = cache.seq_len до update":
            dict(pos_offset=False),
        "RoPE не применён к общему ключу k_rope (повернуть нужно и q_rope, и k_rope)": dict(rope_k=False),
        "масштаб 1/√dn вместо 1/√(dn + dr): размерность головы в скалярном произведении — dn + dr":
            dict(scale="nope"),
    }
    for hint, kw in variants.items():
        cand = _sim(m, x, chunks, **kw)
        if cand.shape == out.shape and torch.allclose(out, cand, atol=1e-4, rtol=1e-3):
            return f" Похоже, {hint}."
    return ""


def _run(impl, m, x, chunks):
    cache = LatentCache()
    outs, s = [], 0
    with torch.no_grad():
        for t in chunks:
            outs.append(m(x[:, s:s + t], cache))
            s += t
    return torch.cat(outs, 1), cache


def test_no_cache_matches_oracle(impl):
    """Без кеша: голова = [часть без позиций из латента ; RoPE-часть с общим ключом], масштаб 1/√(dn+dr)"""
    m = _make(impl)
    x = _x()
    with torch.no_grad():
        out = m(x)
    assert tuple(out.shape) == (B, T, C), f"Выход формы {tuple(out.shape)}, ожидалось {(B, T, C)}."
    expected = _sim(m, x, [T])
    if not torch.allclose(out, expected, atol=1e-5, rtol=1e-4):
        pytest.fail(f"MLA с decoupled RoPE не совпадает с оракулом.{_diagnose(m, x, [T], out)}", pytrace=False)


def test_cache_contents(impl):
    """В кеше: латент c [B, S, r] и УЖЕ повёрнутый общий RoPE-ключ [B, S, dr] — без оси голов"""
    m = _make(impl)
    x = _x()
    _, cache = _run(impl, m, x, [T])
    assert cache.c is not None and tuple(cache.c.shape) == (B, T, R), (
        f"cache.c формы {None if cache.c is None else tuple(cache.c.shape)}, ожидалось {(B, T, R)} — латент.")
    assert cache.k_rope is not None and tuple(cache.k_rope.shape) == (B, T, DR), (
        f"cache.k_rope формы {None if cache.k_rope is None else tuple(cache.k_rope.shape)}, ожидалось {(B, T, DR)}: "
        "RoPE-ключ ОДИН на все головы (поэтому проекция и называется kv_a_proj_with_mqa) — в кеше нет оси голов. "
        "Передай его вторым аргументом cache.update(c, k_rope).")
    with torch.no_grad():
        cos, sin = rope_cos_sin(torch.arange(T), DR, THETA)
        kr = apply_rope(m.kv_a_proj_with_mqa(x)[..., R:].unsqueeze(1), cos, sin).squeeze(1)
    if not torch.allclose(cache.k_rope, kr, atol=1e-5):
        pytest.fail("В кеше лежит не повёрнутый RoPE-ключ. Как и в C05, ключ кешируют ПОСЛЕ RoPE: позиция токена не "
                    "меняется, и поворачивать весь кеш на каждом шаге незачем.", pytrace=False)


def _check(impl, chunks, label):
    m = _make(impl)
    x = _x()
    out, _ = _run(impl, m, x, chunks)
    expected = _sim(m, x, [T])
    if not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        bad = ((out - expected).abs() > 1e-4).any(-1).any(0)
        t = int(bad.nonzero()[0]) if bad.any() else None
        pytest.fail(f"{label}: расхождение с полным прогоном с позиции {t}.{_diagnose(m, x, chunks, out)}",
                    pytrace=False)


def test_token_by_token(impl):
    """По одному токену с кешем == полный прогон"""
    _check(impl, [1] * T, "По одному токену")


def test_chunked(impl):
    """Куски 4 + 1 + 5 == полный прогон"""
    _check(impl, [4, 1, 5], "Куски 4 + 1 + 5")


def test_no_future_leak(impl):
    """Causal: изменение токенов после t не меняет выход в позициях ≤ t"""
    m = _make(impl)
    x = _x()
    x2 = x.clone()
    x2[:, 6:] += 3 * torch.randn_like(x2[:, 6:])
    with torch.no_grad():
        delta = (m(x)[:, :6] - m(x2)[:, :6]).abs().max().item()
    assert delta < 1e-5, f"Выход в позициях 0..5 зависит от токенов 6+ (|Δ| = {delta:.3g}) — нет causal-маски."


def test_kv_cache_bytes(impl):
    """DeepSeek-V3: 576 чисел на токен и слой (512 латент + 64 RoPE) — в ~71 раз меньше MHA"""
    # Источник: deepseek-ai/DeepSeek-V3 config.json: num_hidden_layers 61, num_attention_heads 128,
    # kv_lora_rank 512, qk_rope_head_dim 64, qk_nope_head_dim 128, v_head_dim 128.
    got = impl.mla_kv_cache_bytes(61, 512, 64, 1)
    expected = 61 * 576 * 2
    hint = " Похоже, добавлен множитель 2 «за K и V» — но в MLA K и V восстанавливаются из одного латента." \
        if got == 2 * expected else ""
    assert got == expected, (
        f"Кеш DeepSeek-V3 на 1 токен (bf16): {got} байт, ожидалось {expected} = 61 слой · (512 + 64) · 2 байта.{hint}")
    mha = 61 * 128 * (192 + 128) * 2                     # тот же MHA: ключ 128 + 64, значение 128 на голову
    assert round(mha / got) == 71, "внутренняя ошибка теста"
    assert impl.mla_kv_cache_bytes(61, 512, 64, 131072, batch=2, bytes_per_elem=1) == 61 * 576 * 131072 * 2, (
        "seq_len, batch и bytes_per_elem — множители, как в kv_cache_bytes из C04.")
