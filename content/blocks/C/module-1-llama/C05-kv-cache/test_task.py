"""Тесты C05: KV-cache.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Главное свойство: генерация по кускам с кешем даёт ровно то же, что полный прогон всей
последовательности без кеша. Оракул собран из эталонов C03/C04 (llmlab.core.llama) и
F.scaled_dot_product_attention.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from llmlab.core.llama import apply_rope, rope_cos_sin

B, T, C, H, H_KV = 2, 12, 32, 4, 2
D = C // H
THETA = 10000.0           # в тестах θ поменьше, чтобы позиции сильнее влияли на результат
ATOL, RTOL = 1e-5, 1e-4
_SDPA = F.scaled_dot_product_attention


def _x(t: int = T, seed: int = 1):
    return torch.randn(B, t, C, generator=torch.Generator().manual_seed(seed))


def _make(impl, seed=0):
    torch.manual_seed(seed)
    return impl.LlamaAttention(C, H, H_KV, rope_theta=THETA).eval()


def _run_chunks(impl, m, x, chunks):
    """Прогон по кускам с одним кешем → склеенный выход [B, sum(chunks), C]."""
    cache = impl.KVCache()
    outs, start = [], 0
    with torch.no_grad():
        for c in chunks:
            outs.append(m(x[:, start:start + c], cache))
            start += c
    return torch.cat(outs, dim=1), cache


def _sim(m, x, chunks, pos_offset=True, mask="offset", history=True):
    """Независимая модель того, что делает attention с кешем, — с переключателями типичных ошибок."""
    ks, vs, outs, start = [], [], [], 0
    with torch.no_grad():
        for c in chunks:
            xs = x[:, start:start + c]
            pos = torch.arange(start, start + c) if pos_offset else torch.arange(c)
            cos, sin = rope_cos_sin(pos, D, THETA)
            q = apply_rope(m.q_proj(xs).view(B, c, H, D).transpose(1, 2), cos, sin)
            k = apply_rope(m.k_proj(xs).view(B, c, H_KV, D).transpose(1, 2), cos, sin)
            v = m.v_proj(xs).view(B, c, H_KV, D).transpose(1, 2)
            if history:
                ks.append(k)
                vs.append(v)
                K, V = torch.cat(ks, 2), torch.cat(vs, 2)
            else:
                K, V = k, v
            S = K.shape[2]
            K, V = K.repeat_interleave(H // H_KV, 1), V.repeat_interleave(H // H_KV, 1)
            if mask == "offset":
                allowed = torch.ones(c, S, dtype=torch.bool).tril(diagonal=S - c)
            elif mask == "topleft":
                allowed = torch.ones(c, S, dtype=torch.bool).tril()
            else:
                allowed = torch.ones(c, S, dtype=torch.bool)
            out = _SDPA(q, K, V, attn_mask=allowed)
            outs.append(m.o_proj(out.transpose(1, 2).reshape(B, c, H * D)))
            start += c
    return torch.cat(outs, dim=1)


def _diagnose(m, x, chunks, out) -> str:
    variants = {
        "позиции RoPE на каждом шаге начинаются с 0. Новые токены стоят ПОСЛЕ тех, что в кеше: "
        "positions = arange(start, start + T), где start = cache.seq_len до обновления":
            _sim(m, x, chunks, pos_offset=False),
        "маска выровнена по левому верхнему углу (tril без сдвига или is_causal=True в sdpa при T < S). "
        "Запрос i стоит на позиции start + i и должен видеть ключи 0..start + i: tril(diagonal=S − T)":
            _sim(m, x, chunks, mask="topleft"),
        "новым токенам не разрешено смотреть на кеш / друг на друга так, как нужно — маски нет вовсе, "
        "и в куске из нескольких токенов ранние видят поздние":
            _sim(m, x, chunks, mask="none"),
        "attention смотрит только на новые токены: результат cache.update (весь кеш) не используется":
            _sim(m, x, chunks, history=False),
    }
    for hint, cand in variants.items():
        if cand.shape == out.shape and torch.allclose(out, cand, atol=1e-4, rtol=1e-3):
            return f"Похоже, {hint}."
    return ("Типичную ошибку распознать не удалось. Сравни выходы по шагам: с какого шага начинается "
            "расхождение? Проверь позиции RoPE, что кладётся в кеш, и маску [T, S].")


def _first_bad(out, expected):
    bad = ((out - expected).abs() > 1e-4).any(dim=-1).any(dim=0)   # [T]
    return int(bad.nonzero()[0]) if bad.any() else None


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------

def test_cache_update(impl):
    """KVCache.update: дописывает новые токены по оси времени и возвращает весь кеш"""
    cache = impl.KVCache()
    assert cache.seq_len == 0, "Пустой кеш: seq_len должен быть 0."
    k1, v1 = torch.randn(B, H_KV, 3, D), torch.randn(B, H_KV, 3, D)
    k2, v2 = torch.randn(B, H_KV, 1, D), torch.randn(B, H_KV, 1, D)
    K, V = cache.update(k1, v1)
    assert_close(K, k1, atol=0, rtol=0, msg=lambda _: "После первого update кеш должен совпадать с первыми k.")
    K, V = cache.update(k2, v2)
    assert tuple(K.shape) == (B, H_KV, 4, D), (
        f"После 3 + 1 токенов кеш K имеет форму {tuple(K.shape)} вместо {(B, H_KV, 4, D)}. "
        "Склеивать нужно по оси времени (dim=2), а не по батчу или головам.")
    assert_close(K, torch.cat((k1, k2), 2), atol=0, rtol=0, msg=lambda _: "Порядок в кеше: сначала старые токены, потом новые.")
    assert_close(V, torch.cat((v1, v2), 2), atol=0, rtol=0, msg=lambda _: "V кешируется так же, как K.")
    assert cache.seq_len == 4, f"seq_len = {cache.seq_len}, а в кеше 4 токена."


def test_caches_are_independent(impl):
    """KVCache: два кеша не делят состояние (у каждого слоя/запроса свой)"""
    a, b = impl.KVCache(), impl.KVCache()
    a.update(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
    assert b.seq_len == 0, (
        "Запись в один KVCache изменила другой. Похоже, хранилище — атрибут класса (например, "
        "`keys = []` в теле класса), а не экземпляра: заводи его в __init__ через self.")


# ---------------------------------------------------------------------------
# LlamaAttention
# ---------------------------------------------------------------------------

def test_no_cache_matches_oracle(impl):
    """forward без кеша: GQA + RoPE на q и k, causal-маска"""
    m = _make(impl)
    x = _x()
    with torch.no_grad():
        out = m(x)
    expected = _sim(m, x, [T])
    if out.shape != expected.shape or not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        hint = ""
        if out.shape == expected.shape:
            with torch.no_grad():
                # без RoPE: GQA из C04
                q = m.q_proj(x).view(B, T, H, D).transpose(1, 2)
                k = m.k_proj(x).view(B, T, H_KV, D).transpose(1, 2).repeat_interleave(2, 1)
                v = m.v_proj(x).view(B, T, H_KV, D).transpose(1, 2).repeat_interleave(2, 1)
                no_rope = m.o_proj(_SDPA(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C))
            if torch.allclose(out, no_rope, atol=1e-4):
                hint = " Похоже, RoPE не применён к q и k."
        pytest.fail(f"Прогон без кеша не совпадает с оракулом (GQA из C04 + RoPE из C03 на q и k)."
                    f"{hint}", pytrace=False)


def test_prefill_fills_cache(impl):
    """forward с пустым кешем: тот же выход, в кеше T токенов, H_kv голов, ключи уже повёрнуты"""
    m = _make(impl)
    x = _x()
    out, cache = _run_chunks(impl, m, x, [T])
    assert_close(out, _sim(m, x, [T]), atol=1e-4, rtol=1e-4,
                 msg=lambda _: "С пустым кешем выход должен быть таким же, как без кеша.")
    assert cache.seq_len == T, f"После прогона {T} токенов в кеше {cache.seq_len} — кеш не обновляется."
    assert tuple(cache.k.shape) == (B, H_KV, T, D), (
        f"cache.k имеет форму {tuple(cache.k.shape)}, ожидалось {(B, H_KV, T, D)}. В кеш кладутся K/V до "
        "repeat_kv — иначе экономия памяти от GQA пропадает (кеш раздувается в n_rep раз).")
    with torch.no_grad():
        cos, sin = rope_cos_sin(torch.arange(T), D, THETA)
        k_rot = apply_rope(m.k_proj(x).view(B, T, H_KV, D).transpose(1, 2), cos, sin)
    if not torch.allclose(cache.k, k_rot, atol=1e-5):
        pytest.fail("В кеше лежат не повёрнутые RoPE ключи. Принято кешировать k ПОСЛЕ RoPE: позиция "
                    "токена не меняется, так что поворот делается один раз, а не на каждом шаге для "
                    "всего кеша.", pytrace=False)


def test_token_by_token_matches_full(impl):
    """Генерация по одному токену с кешем == полный прогон без кеша"""
    m = _make(impl)
    x = _x()
    chunks = [1] * T
    out, _ = _run_chunks(impl, m, x, chunks)
    expected = _sim(m, x, [T])
    if not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        t = _first_bad(out, expected)
        pytest.fail(f"Пошаговый прогон расходится с полным начиная с шага {t}. {_diagnose(m, x, chunks, out)}",
                    pytrace=False)


def test_prefill_then_decode(impl):
    """Промпт целиком (prefill), затем по одному токену == полный прогон"""
    m = _make(impl)
    x = _x()
    chunks = [7] + [1] * (T - 7)
    out, cache = _run_chunks(impl, m, x, chunks)
    expected = _sim(m, x, [T])
    if not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        t = _first_bad(out, expected)
        pytest.fail(f"Prefill 7 токенов + генерация по одному: расхождение с полным прогоном с позиции {t}. "
                    f"{_diagnose(m, x, chunks, out)}", pytrace=False)
    assert cache.seq_len == T, f"В конце в кеше {cache.seq_len} токенов вместо {T}."


def test_chunked_prefill(impl):
    """Куски по несколько токенов поверх кеша (маска [T, S] со сдвигом) == полный прогон"""
    m = _make(impl)
    x = _x()
    chunks = [4, 3, 5]
    out, _ = _run_chunks(impl, m, x, chunks)
    expected = _sim(m, x, [T])
    if not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        t = _first_bad(out, expected)
        pytest.fail(f"Прогон кусками 4 + 3 + 5 расходится с полным с позиции {t}. "
                    f"{_diagnose(m, x, chunks, out)}", pytrace=False)


def test_no_future_leak(impl):
    """Без кеша: изменение токена t+1 не меняет выход в позициях ≤ t"""
    m = _make(impl)
    x = _x()
    x2 = x.clone()
    x2[:, 6:] += 3 * torch.randn_like(x2[:, 6:])
    with torch.no_grad():
        a, b = m(x), m(x2)
    delta = (a[:, :6] - b[:, :6]).abs().max().item()
    assert delta < 1e-5, f"Выход в позициях 0..5 зависит от токенов 6+ (max |Δ| = {delta:.3g}) — нет causal-маски."


def test_default_theta_is_llama3(impl):
    """LlamaAttention: θ по умолчанию — 500 000, как в Llama 3, и он реально используется"""
    torch.manual_seed(0)
    m = impl.LlamaAttention(C, H, H_KV)
    assert m.rope_theta == 500000.0, "Конструктор дан готовым: rope_theta по умолчанию = 500 000 (Llama 3)."
    x = _x()
    with torch.no_grad():
        a = m(x)
        m.rope_theta = THETA
        b = m(x)
    assert not torch.allclose(a, b, atol=1e-5), (
        "Смена self.rope_theta не изменила выход: θ не передаётся в rope_cos_sin.")
