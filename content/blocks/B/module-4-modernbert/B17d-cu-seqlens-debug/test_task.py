"""Тесты «Найди баг: cu_seqlens» — те же, что у «Unpadding».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter).

Оракул — обычный батч с паддингом: attention с маской PAD-ключей и слой ModernBERT из core (эталон B16).
"""

import pytest
import torch
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, rope_cos_sin
from llmlab.core.modernbert import ModernBertAttention, ModernBertConfig, local_window_mask

H, D = 2, 8
C = H * D
MASK = torch.tensor([[1, 1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 0, 0]])                 # длины 3, 2, 6, 4 → N = 15


def _blocks(lengths):
    m = torch.zeros(sum(lengths), sum(lengths), dtype=torch.bool)
    s = 0
    for L in lengths:
        m[s:s + L, s:s + L] = True
        s += L
    return m


def _cu(impl, mask=MASK):
    res = impl.unpad_input(mask)
    assert isinstance(res, tuple) and len(res) == 3, "unpad_input возвращает (indices, cu_seqlens, max_seqlen)"
    return res


def test_unpad_input(impl):
    """indices настоящих токенов, cu_seqlens = [0, L₀, L₀ + L₁, …, N] (int32), max_seqlen"""
    indices, cu, mx = _cu(impl)
    assert indices.tolist() == [0, 1, 2, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], (
        f"indices — номера единиц в mask.flatten() по порядку, получено {indices.tolist()}")
    assert cu.dtype == torch.int32, f"cu_seqlens — int32 (как ждёт flash_attn_varlen_func), получено {cu.dtype}"
    assert cu.tolist() == [0, 3, 5, 11, 15], (
        f"длины 3, 2, 6, 4 → cu_seqlens [0, 3, 5, 11, 15] (B + 1 чисел, последнее = N), получено {cu.tolist()}")
    assert mx == 6 and isinstance(mx, int), f"max_seqlen = 6 (int), получено {mx!r}"
    left = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])  # паддинг слева и пустой текст
    i2, cu2, mx2 = impl.unpad_input(left)
    assert i2.tolist() == [2, 3, 4, 5, 6, 7] and cu2.tolist() == [0, 2, 6, 6] and mx2 == 4, (
        "паддинг слева и пустой текст: indices [2, 3, 4, 5, 6, 7], cu_seqlens [0, 2, 6, 6] (у пустого текста "
        f"начало = конец), получено {i2.tolist()}, {cu2.tolist()}, {mx2}")


def test_pack_unpack_roundtrip(impl):
    """pack → [N, …]; unpack возвращает настоящие токены на место, PAD — нули; работают и лишние оси"""
    indices, _, _ = _cu(impl)
    x = torch.randn(4, 6, H, D, generator=torch.Generator().manual_seed(0))
    p = impl.pack(x, indices)
    assert p.shape == (15, H, D), f"[B, T, H, D] → [N, H, D] = [15, {H}, {D}], получено {tuple(p.shape)}"
    torch.testing.assert_close(p[:3], x[0, :3], atol=0, rtol=0, msg="первые 3 строки упаковки — текст 0")
    torch.testing.assert_close(p[3:5], x[1, :2], atol=0, rtol=0, msg="строки 3–4 — текст 1")
    back = impl.unpack(p, indices, 4, 6)
    assert back.shape == x.shape, f"unpack → [B, T, …], получено {tuple(back.shape)}"
    m = MASK.bool()
    torch.testing.assert_close(back[m], x[m], atol=0, rtol=0, msg="настоящие токены вернулись на свои места")
    assert (back[~m] == 0).all(), "на месте PAD после unpack — нули"


def test_packed_position_ids(impl):
    """Позиции в упаковке начинаются с 0 у каждого текста"""
    _, cu, _ = _cu(impl)
    pos = impl.packed_position_ids(cu)
    if pos.tolist() == list(range(15)):
        pytest.fail("позиции 0 … N − 1 подряд: второй текст начинается с позиции 3, как будто он продолжение первого. "
                    "RoPE тогда кодирует чужое смещение. У каждого текста позиции свои: 0, 1, 2, …", pytrace=False)
    assert pos.tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3], (
        f"длины 3, 2, 6, 4 → позиции 0 1 2 | 0 1 | 0 … 5 | 0 … 3, получено {pos.tolist()}")


def test_varlen_mask(impl):
    """Блочно-диагональная маска: токен видит только свой текст; с окном — ещё и |i − j| ≤ W/2"""
    _, cu, _ = _cu(impl)
    m = impl.varlen_mask(cu)
    assert m.shape == (15, 15) and m.dtype == torch.bool, f"маска [N, N] bool, получено {tuple(m.shape)}, {m.dtype}"
    want = _blocks([3, 2, 6, 4])
    assert torch.equal(m, want), "mask[i, j] = True, только если i и j из одного текста (блоки 3, 2, 6, 4 на диагонали)"
    mw = impl.varlen_mask(cu, window=2)
    i, j = torch.arange(15)[:, None], torch.arange(15)[None, :]
    assert torch.equal(mw, want & ((i - j).abs() <= 1)), "с окном W = 2: свой текст И |i − j| ≤ 1"


def _padded_attention(q, k, v, mask, window=None):
    """q, k, v [B, T, H, D] обычного батча → [B, T, H, D] (маска PAD-ключей, при window — ещё окно)."""
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    s = qh @ kh.transpose(-2, -1) / D ** 0.5
    allow = mask.bool()[:, None, None, :]
    if window is not None:
        allow = allow & local_window_mask(q.shape[1], window)
    s = s.masked_fill(~allow, torch.finfo(s.dtype).min)
    return (s.softmax(-1) @ vh).transpose(1, 2)


def test_varlen_attention_equals_padded(impl):
    """Attention на упаковке == attention батча с паддингом на настоящих токенах (полное и с окном)"""
    indices, cu, _ = _cu(impl)
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(4, 6, H, D, generator=g) for _ in range(3))
    pq, pk, pv = (impl.pack(t, indices) for t in (q, k, v))
    out = impl.varlen_attention(pq, pk, pv, cu)
    assert out.shape == (15, H, D), f"выход [N, H, D] = [15, {H}, {D}], получено {tuple(out.shape)}"
    m = MASK.bool()
    want = _padded_attention(q, k, v, MASK)[m]
    full = (pq.transpose(0, 1) @ pk.transpose(0, 1).transpose(-2, -1) / D ** 0.5).softmax(-1) @ pv.transpose(0, 1)
    if torch.allclose(out, full.transpose(0, 1), atol=1e-5):
        pytest.fail("attention по всей упаковке без маски: каждый токен смотрит на все N токенов, в том числе на "
                    "чужие тексты. Нужна varlen_mask(cu_seqlens).", pytrace=False)
    torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-4,
                               msg="выход на упаковке = выход батча с паддингом на тех же (настоящих) токенах")
    outw = impl.varlen_attention(pq, pk, pv, cu, window=2)
    torch.testing.assert_close(outw, _padded_attention(q, k, v, MASK, window=2)[m], atol=1e-5, rtol=1e-4,
                               msg="с окном: как локальный слой ModernBERT на батче с паддингом")


def test_neighbours_do_not_leak(impl):
    """Изменение текста 1 не меняет выход на текстах 0, 2, 3 — даже на границах"""
    indices, cu, _ = _cu(impl)
    g = torch.Generator().manual_seed(2)
    q, k, v = (torch.randn(15, H, D, generator=g) for _ in range(3))
    base = impl.varlen_attention(q, k, v, cu)
    k2, v2 = k.clone(), v.clone()
    k2[3:5] += 5.0
    v2[3:5] += 5.0                                                     # текст 1 — строки 3 и 4
    out = impl.varlen_attention(q, k2, v2, cu)
    keep = torch.ones(15, dtype=torch.bool)
    keep[3:5] = False
    changed = [int(i) for i in torch.nonzero(~torch.isclose(out, base, atol=1e-5).all(-1).all(-1)).flatten()
               if keep[i]]
    assert not changed, (
        f"после правки текста 1 (токены 3–4 упаковки) изменились токены {changed} других текстов: тексты в упаковке "
        "видят друг друга. Проверь границы cu_seqlens и блочную маску.")


def test_modernbert_layer_unpadded(impl):
    """Слой attention ModernBERT на упаковке (RoPE по packed_position_ids) == тот же слой на батче с паддингом"""
    torch.manual_seed(0)
    cfg = ModernBertConfig(vocab_size=10, hidden_size=C, num_attention_heads=H, num_hidden_layers=3, local_attention=4)
    for idx in (0, 1):                                                # глобальный и локальный слой
        att = ModernBertAttention(cfg, idx)
        x = torch.randn(4, 6, C, generator=torch.Generator().manual_seed(3))
        window = None if att.attention_type == "full_attention" else cfg.local_attention
        pad_mask = MASK.bool()[:, None, None, :]
        if window is not None:
            pad_mask = pad_mask & local_window_mask(6, window)
        want = att(x, pad_mask)[MASK.bool()]
        indices, cu, _ = impl.unpad_input(MASK)
        px = impl.pack(x, indices)                                    # [N, C]
        q, k, v = att.Wqkv(px).view(-1, 3, H, D).unbind(1)            # [N, H, D]
        cos, sin = rope_cos_sin(impl.packed_position_ids(cu), D, att.rope_theta)       # [N, D]
        q = apply_rope(q.transpose(0, 1), cos, sin).transpose(0, 1)
        k = apply_rope(k.transpose(0, 1), cos, sin).transpose(0, 1)
        got = att.Wo(impl.varlen_attention(q, k, v, cu, window).reshape(-1, C))
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4,
                                   msg=f"слой {idx} ({att.attention_type}): упаковка + позиции с нуля у каждого текста "
                                       "+ блочная маска дают ровно тот же выход, что батч с паддингом")


def test_padding_savings(impl):
    """Доля сэкономленной работы: линейные слои 1 − ΣL/(B·T), attention 1 − ΣL²/(B·T²)"""
    s = impl.padding_savings(MASK)
    assert set(s) == {"linear", "attention"}, f"словарь с ключами linear и attention, получено {sorted(s)}"
    assert abs(s["linear"] - (1 - 15 / 24)) < 1e-9, f"линейные: 1 − 15/24 = 0.375, получено {s['linear']}"
    assert abs(s["attention"] - (1 - (9 + 4 + 36 + 16) / (4 * 36))) < 1e-9, (
        f"attention: 1 − (3² + 2² + 6² + 4²)/(4 · 6²) = 0.549, получено {s['attention']}")
    lengths = torch.tensor([512, 60, 80, 100, 120, 90, 70, 50])         # один длинный текст и короткие
    m = (torch.arange(512)[None, :] < lengths[:, None]).long()
    s2 = impl.padding_savings(m)
    assert s2["linear"] > 0.7 and s2["attention"] > 0.8, (
        f"батч, где один текст 512, а остальные ~80: паддинг съедает 74 % токенов и 85 % работы attention, "
        f"получено {s2}")
