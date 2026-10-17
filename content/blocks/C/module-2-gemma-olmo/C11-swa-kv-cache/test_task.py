"""Тесты «KV-cache для окна»: KV-cache для SWA с кольцевым буфером.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Главное свойство: генерация с кольцевым кешем (по одному токену, prefill + decode, кусками) даёт
ровно то же, что полный прогон SWA из «Sliding window attention» без кеша, — а буфер при этом никогда не больше W токенов.
Оракул — SlidingWindowAttention из llmlab.core.gemma_olmo (эталон «Sliding window attention»).
"""

import pytest
import torch
from torch.testing import assert_close

from llmlab.core.gemma_olmo import SlidingWindowAttention, layer_types
from llmlab.core.llama import kv_cache_bytes

B, T, C, H, H_KV = 2, 16, 32, 4, 2
D = C // H
W = 4
THETA = 10000.0


def _x(t: int = T, seed: int = 1):
    return torch.randn(B, t, C, generator=torch.Generator().manual_seed(seed))


def _make(impl, window=W, seed=0):
    torch.manual_seed(seed)
    m = impl.SlidingWindowCachedAttention(C, H, H_KV, window, rope_theta=THETA).eval()
    full = SlidingWindowAttention(C, H, H_KV, window, rope_theta=THETA).eval()
    full.load_state_dict(m.state_dict())
    return m, full


def _run_chunks(impl, m, x, chunks, window=W):
    cache = impl.RingKVCache(window)
    outs, start = [], 0
    with torch.no_grad():
        for c in chunks:
            outs.append(m(x[:, start:start + c], cache))
            start += c
    return torch.cat(outs, dim=1), cache


def _kv(t, start=0):
    """k, v, где каждый токен «подписан» своей позицией: k[..., j, :] = start + j."""
    vals = torch.arange(start, start + t, dtype=torch.float32)
    k = vals.view(1, 1, t, 1).expand(B, H_KV, t, D).clone()
    return k, -k


def _first_bad(out, expected):
    bad = ((out - expected).abs() > 1e-4).any(dim=-1).any(dim=0)
    return int(bad.nonzero()[0]) if bad.any() else None


# ---------------------------------------------------------------------------
# RingKVCache
# ---------------------------------------------------------------------------

def test_ring_slots(impl):
    """Токен на позиции p лежит в ячейке p % W; буфер — ровно W ячеек"""
    cache = impl.RingKVCache(W)
    for p in range(10):
        cache.update(*_kv(1, p))
    assert cache.seq_len == 10, f"seq_len = {cache.seq_len}, а через кеш прошло 10 токенов (это счётчик, а не размер буфера)."
    assert tuple(cache.k.shape) == (B, H_KV, W, D), (
        f"Буфер cache.k формы {tuple(cache.k.shape)}, ожидалось {(B, H_KV, W, D)}: кольцевой буфер не растёт, "
        "в нём всегда W ячеек.")
    expected_pos = [8, 9, 6, 7]
    assert cache.pos.tolist() == expected_pos, (
        f"cache.pos = {cache.pos.tolist()}, ожидалось {expected_pos}: после 10 токенов в буфере последние 4 "
        "(позиции 6..9), и токен p лежит в ячейке p % 4.")
    stored = cache.k[0, 0, :, 0].tolist()
    assert stored == [8.0, 9.0, 6.0, 7.0], (
        f"В ячейках буфера лежат ключи токенов {stored}, а cache.pos говорит {expected_pos}: содержимое и "
        "позиции разъехались.")
    assert_close(cache.v, -cache.k, atol=0, rtol=0, msg=lambda _: "V хранится в тех же ячейках, что и K.")


def test_long_chunk_keeps_last_window(impl):
    """Кусок длиннее окна (prefill T=11, W=4): в буфере остаются только позиции 7..10"""
    cache = impl.RingKVCache(W)
    cache.update(*_kv(11))
    assert cache.seq_len == 11, f"seq_len = {cache.seq_len} после куска из 11 токенов."
    got = sorted(cache.pos.tolist())
    assert got == [7, 8, 9, 10], (
        f"После prefill из 11 токенов в буфере позиции {got}, ожидалось [7, 8, 9, 10]: дальше нужны только "
        "последние W токенов. Если T > W, пиши в буфер k[:, :, -W:].")
    for slot, p in enumerate(cache.pos.tolist()):
        assert cache.k[0, 0, slot, 0].item() == p, f"В ячейке {slot} позиция {p}, а ключ — от токена {cache.k[0, 0, slot, 0].item():g}."


def test_update_returns_needed_keys(impl):
    """update возвращает все ключи, нужные новым запросам, с правильными позициями"""
    cache = impl.RingKVCache(W)
    cache.update(*_kv(6))                         # позиции 0..5, в буфере 2..5
    K, V, k_pos = cache.update(*_kv(3, 6))        # новые 6, 7, 8
    assert K.shape[:2] == (B, H_KV) and K.shape[3] == D and K.shape == V.shape, (
        f"K, V форм {tuple(K.shape)}, {tuple(V.shape)}: ожидалось [B, H_kv, S, D] с одинаковым S.")
    assert k_pos.shape == (K.shape[2],), f"k_pos формы {tuple(k_pos.shape)}, а ключей {K.shape[2]}: по позиции на ключ."
    valid = k_pos >= 0
    labels = K[0, 0, :, 0]
    assert torch.equal(labels[valid], k_pos[valid].float()), (
        f"Позиции не соответствуют ключам: k_pos = {k_pos.tolist()}, а ключи от токенов {labels.tolist()}. "
        "Возвращай позиции в том же порядке, что и ключи.")
    needed = set(range(6 - W + 1, 9))             # запрос 6 видит 3..6, запрос 8 — до 8
    have = set(k_pos[valid].tolist())
    assert needed <= have, (
        f"Новым запросам (позиции 6..8, окно {W}) нужны ключи {sorted(needed)}, а update вернул {sorted(have)}. "
        "Верни содержимое буфера ДО записи новых токенов плюс все новые: иначе запрос 6 потеряет токены, "
        "которые уже вытеснены записью 7 и 8.")
    assert cache.seq_len == 9


# ---------------------------------------------------------------------------
# SlidingWindowCachedAttention
# ---------------------------------------------------------------------------

def test_no_cache_is_swa(impl):
    """Без кеша — ровно SlidingWindowAttention из «Sliding window attention»"""
    m, full = _make(impl)
    x = _x()
    with torch.no_grad():
        assert_close(m(x), full(x), atol=1e-5, rtol=1e-4,
                     msg=lambda _: "cache=None: выход должен совпасть с SWA из «Sliding window attention» (те же веса, окно W). Проверь "
                                   "маску: ключ в окне, если 0 ≤ q_pos − k_pos < W.")


def _sim(m, x, chunks, window, empty_mask=True, pos_offset=True, by_index=False):
    """Независимая модель кольцевого кеша — с переключателями типичных ошибок."""
    from llmlab.core.llama import apply_rope, rope_cos_sin
    buf_k = buf_v = None
    buf_pos = torch.full((window,), -1)
    outs, start = [], 0
    with torch.no_grad():
        for c in chunks:
            xs = x[:, start:start + c]
            q_pos = torch.arange(start, start + c)
            cos, sin = rope_cos_sin(q_pos if pos_offset else torch.arange(c), D, THETA)
            q = apply_rope(m.q_proj(xs).view(B, c, H, D).transpose(1, 2), cos, sin)
            k = apply_rope(m.k_proj(xs).view(B, c, H_KV, D).transpose(1, 2), cos, sin)
            v = m.v_proj(xs).view(B, c, H_KV, D).transpose(1, 2)
            if buf_k is None:
                buf_k, buf_v = k.new_zeros(B, H_KV, window, D), v.new_zeros(B, H_KV, window, D)
            K, V, k_pos = torch.cat((buf_k, k), 2), torch.cat((buf_v, v), 2), torch.cat((buf_pos, q_pos))
            keep = min(c, window)
            slots = q_pos[-keep:] % window
            buf_k, buf_v, buf_pos = buf_k.clone(), buf_v.clone(), buf_pos.clone()
            buf_k[:, :, slots], buf_v[:, :, slots], buf_pos[slots] = k[:, :, -keep:], v[:, :, -keep:], q_pos[-keep:]
            if by_index:
                k_pos = torch.arange(K.shape[2]) + (start + c - K.shape[2])
            dist = q_pos[:, None] - k_pos[None, :]
            mask = (dist >= 0) & (dist < window)
            if empty_mask:
                mask = mask & (k_pos[None, :] >= 0)
            K, V = K.repeat_interleave(H // H_KV, 1), V.repeat_interleave(H // H_KV, 1)
            scores = (q @ K.transpose(-2, -1) / D ** 0.5).masked_fill(~mask, float("-inf"))
            out = torch.softmax(scores, -1) @ V
            outs.append(m.o_proj(out.transpose(1, 2).reshape(B, c, C)))
            start += c
    return torch.cat(outs, 1)


def _diagnose(m, x, chunks, window, out) -> str:
    variants = {
        "позиции RoPE на каждом шаге начинаются с 0. Новые токены стоят после всех, что прошли через кеш: "
        "start = cache.seq_len ДО update (как в «KV-cache»)": dict(pos_offset=False),
        "пустые ячейки буфера (позиция −1) не исключены из маски — запросы смотрят на нулевые ключи. "
        "Добавь в маску условие k_pos ≥ 0": dict(empty_mask=False),
        "маска строится по индексам ключей в тензоре, а не по их абсолютным позициям. После того как буфер "
        "заполнился, запись идёт по кругу и ключи лежат не по порядку — используй k_pos из update":
            dict(by_index=True),
    }
    for hint, kw in variants.items():
        cand = _sim(m, x, chunks, window, **kw)
        if cand.shape == out.shape and torch.allclose(out, cand, atol=1e-4, rtol=1e-3):
            return f" Похоже, {hint}."
    return (" Проверь по шагам: позиции запросов (start = cache.seq_len до update), что возвращает update "
            "(буфер ДО записи + новые токены), и маску по позициям: k_pos ≥ 0, 0 ≤ q_pos − k_pos < W.")


def _check_chunks(impl, chunks, window=W, label=""):
    m, full = _make(impl, window)
    x = _x(sum(chunks))
    out, cache = _run_chunks(impl, m, x, chunks, window)
    with torch.no_grad():
        expected = full(x)
    assert torch.allclose(_sim(m, x, chunks, window), expected, atol=1e-5), "внутренняя ошибка теста: _sim ≠ SWA"
    if not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        t = _first_bad(out, expected)
        where = f"начиная с позиции {t}" if t is not None else "(небольшое численное расхождение)"
        pytest.fail(f"{label}: расхождение с полным прогоном SWA {where}.{_diagnose(m, x, chunks, window, out)}",
                    pytrace=False)
    return cache


def test_token_by_token(impl):
    """Генерация по одному токену с кольцевым кешем == полный прогон SWA"""
    _check_chunks(impl, [1] * T, label="По одному токену")


def test_prefill_then_decode(impl):
    """Prefill длиннее окна (11 токенов), затем по одному == полный прогон"""
    _check_chunks(impl, [11] + [1] * (T - 11), label="Prefill 11 + decode")


def test_chunked(impl):
    """Куски 5 + 3 + 6 + 2 (и длиннее, и короче окна) == полный прогон"""
    _check_chunks(impl, [5, 3, 6, 2], label="Куски 5 + 3 + 6 + 2")


def test_window_one_and_large(impl):
    """W = 1 (только сам токен) и W > T (кеш не переполняется) — тоже совпадают"""
    _check_chunks(impl, [1] * 8, window=1, label="W = 1")
    _check_chunks(impl, [3, 1, 1, 5], window=20, label="W = 20")


def test_memory_is_bounded(impl):
    """60 шагов генерации: буфер всё ещё W ячеек"""
    m, _ = _make(impl)
    cache = impl.RingKVCache(W)
    x = _x(60, seed=2)
    with torch.no_grad():
        for t in range(60):
            m(x[:, t:t + 1], cache)
    assert cache.seq_len == 60
    assert cache.k.shape[2] == W and cache.v.shape[2] == W, (
        f"После 60 токенов в буфере {cache.k.shape[2]} позиций, ожидалось W = {W}. Весь смысл SWA-кеша — "
        "память O(W), а не O(T).")


# ---------------------------------------------------------------------------
# swa_kv_cache_bytes
# ---------------------------------------------------------------------------

def test_swa_kv_cache_bytes(impl):
    """Gemma 3 27B на 128K: 10.4 ГиБ вместо 62 ГиБ; Mistral 7B: кеш перестаёт расти после 4096"""
    # Источник: google/gemma-3-27b-pt config.json (text_config): 62 слоя, sliding_window_pattern = 6,
    # sliding_window = 1024, num_key_value_heads = 16, head_dim = 128.
    gemma = layer_types(62, 6)
    got = impl.swa_kv_cache_bytes(gemma, 1024, 16, 128, 131072)
    expected = 2 * (10 * 131072 + 52 * 1024) * 16 * 128 * 2
    assert got == expected, (
        f"Gemma 3 27B, 128K токенов, bf16: получено {got / 2**30:.3f} ГиБ, ожидалось {expected / 2**30:.3f} ГиБ. "
        "10 full-слоёв хранят все 131 072 токена, 52 sliding — только 1024.")
    all_full = kv_cache_bytes(62, 16, 128, 131072)
    assert impl.swa_kv_cache_bytes(["full_attention"] * 62, 1024, 16, 128, 131072) == all_full, (
        "Если все слои full, результат должен совпасть с kv_cache_bytes из «GQA и MQA» (62 ГиБ для этой конфигурации).")
    # Источник: mistralai/Mistral-7B-v0.1 config.json: 32 слоя, sliding_window = 4096, 8 KV-голов, head_dim 128.
    mistral = ["sliding_attention"] * 32
    a = impl.swa_kv_cache_bytes(mistral, 4096, 8, 128, 4096)
    b = impl.swa_kv_cache_bytes(mistral, 4096, 8, 128, 32768, batch=1)
    assert a == b == 2 * 32 * 4096 * 8 * 128 * 2, (
        f"Mistral 7B (все слои SWA, окно 4096): кеш на 4K = {a / 2**20:.0f} МиБ и на 32K = {b / 2**20:.0f} МиБ, "
        "ожидалось 512 МиБ в обоих случаях — после заполнения окна кеш не растёт.")
    assert impl.swa_kv_cache_bytes(mistral, 4096, 8, 128, 1000, batch=4, bytes_per_elem=4) == \
        2 * 32 * 1000 * 8 * 128 * 4 * 4, "Короче окна sliding-слой хранит все seq_len токенов; batch и bytes_per_elem — множители."
