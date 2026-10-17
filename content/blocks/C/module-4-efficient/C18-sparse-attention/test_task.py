"""Тесты C18: sparse attention в стиле DSA (DeepSeek-V3.2).

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: индексатор считает Σ_h w·ReLU(q·k) и не видит будущего; topk_mask выбирает в каждой строке
ровно min(k, t + 1) лучших допустимых ключей и никогда — будущие; sparse attention не зависит от
невыбранных ключей; при k ≥ T DSA совпадает с обычным causal attention.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

B, T, C, H, HI, DI = 2, 12, 32, 4, 3, 8
_SDPA = F.scaled_dot_product_attention


def _x(t: int = T, seed: int = 1):
    return torch.randn(B, t, C, generator=torch.Generator().manual_seed(seed))


def _indexer(impl, seed=0):
    torch.manual_seed(seed)
    return impl.LightningIndexer(C, HI, DI).eval()


def _oracle_index(m, x, relu=True, weights=True, causal=True):
    t = x.shape[1]
    with torch.no_grad():
        q = m.q_proj(x).view(B, t, HI, DI)
        k = m.k_proj(x)
        logits = torch.einsum("bthd,bsd->bths", q, k) / math.sqrt(DI)
        if relu:
            logits = F.relu(logits)
        w = m.weights_proj(x) / math.sqrt(HI) if weights else torch.full((B, t, HI), 1 / math.sqrt(HI))
        s = torch.einsum("bth,bths->bts", w, logits)
        if causal:
            s = s.masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf"))
        return s


def _causal(t, s=None):
    s = t if s is None else s
    return torch.ones(t, s, dtype=torch.bool).tril(diagonal=s - t)


def _qkv(t=T, s=None, seed=2):
    g = torch.Generator().manual_seed(seed)
    s = t if s is None else s
    return (torch.randn(B, H, t, 8, generator=g), torch.randn(B, H, s, 8, generator=g),
            torch.randn(B, H, s, 8, generator=g))


# ---------------------------------------------------------------------------
# LightningIndexer
# ---------------------------------------------------------------------------

def test_indexer_matches_formula(impl):
    """LightningIndexer: I(t, s) = Σ_h w_{t,h}·ReLU(q_{t,h}·k_s) / √(H_I·d_I), форма [B, T, T]"""
    m = _indexer(impl)
    x = _x()
    with torch.no_grad():
        out = m(x)
    assert tuple(out.shape) == (B, T, T), (
        f"Оценки индексатора формы {tuple(out.shape)}, ожидалось {(B, T, T)}: одна оценка на пару (запрос t, ключ s), "
        "головы индексатора суммируются с весами w_{t,h}.")
    expected = _oracle_index(m, x)
    lower = _causal(T).expand(B, T, T)
    if torch.allclose(out[lower], expected[lower], atol=1e-5, rtol=1e-4):
        return
    variants = {
        "без ReLU: ReLU(q·k) отбрасывает отрицательные совпадения — без него головы гасят друг друга":
            _oracle_index(m, x, relu=False),
        "без весов голов w = weights_proj(x): у каждого запроса свои веса голов индексатора":
            _oracle_index(m, x, weights=False),
    }
    for name, v in variants.items():
        if torch.allclose(out[lower], v[lower], atol=1e-5, rtol=1e-4):
            pytest.fail(f"Оценки индексатора совпали с вариантом «{name}».", pytrace=False)
    ratio = (out[lower] / expected[lower]).nanmedian().item()
    hint = (f" Отношение к эталону ≈ {ratio:.3f} — проверь масштаб: 1/√d_I для q·k и 1/√H_I для весов голов."
            if torch.isfinite(torch.tensor(ratio)) and abs(ratio - 1) > 1e-3 else "")
    pytest.fail("Оценки индексатора не совпадают с формулой Σ_h w·ReLU(q·k)." + hint, pytrace=False)


def test_indexer_is_causal(impl):
    """LightningIndexer: оценки будущих ключей = −inf, прошлые строки не зависят от будущих токенов"""
    m = _indexer(impl)
    x = _x()
    with torch.no_grad():
        out = m(x)
        x2 = x.clone()
        x2[:, 7:] = torch.randn(B, T - 7, C)
        out2 = m(x2)
    future = ~_causal(T)
    assert torch.isneginf(out[:, future]).all(), (
        "Оценки будущих ключей (s > t) должны быть −inf: иначе top-k может выбрать токен из будущего, "
        "и запрос «подсмотрит» ответ. Замаскируй верхний треугольник после суммы по головам.")
    assert torch.isfinite(out[:, ~future]).all(), "Оценки допустимых ключей (s ≤ t) должны быть конечными."
    assert_close(out[:, :7, :7], out2[:, :7, :7], atol=1e-6, rtol=1e-5, msg=lambda _: (
        "Оценки строк t < 7 изменились после замены токенов 7…: индексатор утекает в будущее."))


# ---------------------------------------------------------------------------
# topk_mask
# ---------------------------------------------------------------------------

def test_topk_mask_counts_and_causality(impl):
    """topk_mask: в строке t ровно min(k, t + 1) ключей, ни одного из будущего"""
    m = _indexer(impl)
    with torch.no_grad():
        scores = _oracle_index(m, _x())
    for k in (1, 4):
        keep = impl.topk_mask(scores, k)
        assert keep.dtype == torch.bool and tuple(keep.shape) == (B, T, T), (
            f"topk_mask должна вернуть bool [B, T, S], а вернула {keep.dtype} {tuple(keep.shape)}.")
        assert not keep[:, ~_causal(T)].any(), (
            f"k = {k}: выбран ключ из будущего. В первых строках допустимых ключей меньше k, и topk добирает "
            "позиции с −inf — их надо отбросить (keep & isfinite(scores)).")
        counts = keep.sum(-1)
        want = torch.clamp(torch.arange(T) + 1, max=k).expand(B, T)
        if not torch.equal(counts, want):
            t = int((counts != want).any(0).nonzero()[0])
            pytest.fail(f"k = {k}: в строке t = {t} выбрано {counts[0, t].item()} ключей, ожидалось "
                        f"{want[0, t].item()} = min(k, t + 1).", pytrace=False)


def test_topk_mask_picks_best(impl):
    """topk_mask: выбранные ключи — с наибольшими оценками индексатора"""
    m = _indexer(impl)
    with torch.no_grad():
        scores = _oracle_index(m, _x())
    keep = impl.topk_mask(scores, 3)
    valid = _causal(T).expand(B, T, T)
    for b in range(B):
        for t in range(3, T):
            chosen = scores[b, t][keep[b, t]]
            rest = scores[b, t][valid[b, t] & ~keep[b, t]]
            if chosen.numel() and rest.numel() and chosen.min() < rest.max():
                pytest.fail(f"Строка t = {t}: выбран ключ с оценкой {chosen.min():.3f}, хотя есть невыбранный с "
                            f"{rest.max():.3f}. Бери topk по последней оси (по ключам), а не по запросам.",
                            pytrace=False)


def test_topk_mask_large_k_is_causal(impl):
    """topk_mask: при k ≥ S маска = обычная causal-маска"""
    m = _indexer(impl)
    with torch.no_grad():
        scores = _oracle_index(m, _x())
    keep = impl.topk_mask(scores, 100)
    assert torch.equal(keep, _causal(T).expand(B, T, T)), (
        "При k ≥ S выбираются все допустимые ключи → маска совпадает с causal. topk(k) с k > S упадёт — "
        "ограничь k = min(top_k, S).")


# ---------------------------------------------------------------------------
# sparse_attention
# ---------------------------------------------------------------------------

def test_sparse_attention_matches_oracle(impl):
    """sparse_attention: обычный softmax-attention, но только по ключам из маски (общей для голов)"""
    q, k, v = _qkv()
    keep = torch.rand(B, T, T, generator=torch.Generator().manual_seed(3)) < 0.4
    keep |= torch.eye(T, dtype=torch.bool)
    out = impl.sparse_attention(q, k, v, keep)
    assert tuple(out.shape) == (B, H, T, 8), f"Выход формы {tuple(out.shape)}, ожидалось {(B, H, T, 8)}."
    expected = _SDPA(q, k, v, attn_mask=keep.unsqueeze(1))
    if torch.allclose(out, expected, atol=1e-5, rtol=1e-4):
        return
    if torch.allclose(out, _SDPA(q, k, v, attn_mask=~keep.unsqueeze(1)), atol=1e-5):
        pytest.fail("Маска применена наоборот: True в keep = ключ ОСТАВИТЬ, заполнять −inf надо ~keep.", pytrace=False)
    pytest.fail("sparse_attention не совпадает с softmax(q·kᵀ/√d + маска)·v. Маска [B, T, S] одна на все головы — "
                "добавь ось голов: keep.unsqueeze(1).", pytrace=False)


def test_sparse_attention_ignores_unselected(impl):
    """sparse_attention: невыбранные ключи и значения не влияют на выход"""
    q, k, v = _qkv()
    keep = torch.zeros(B, T, T, dtype=torch.bool)
    keep[..., :3] = True                     # каждый запрос видит только ключи 0, 1, 2
    out = impl.sparse_attention(q, k, v, keep)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 3:] += 10 * torch.randn_like(k2[:, :, 3:])
    v2[:, :, 3:] += 10 * torch.randn_like(v2[:, :, 3:])
    assert_close(impl.sparse_attention(q, k2, v2, keep), out, atol=1e-6, rtol=1e-5, msg=lambda _: (
        "Выход изменился после замены НЕвыбранных ключей/значений → маска не применяется (или применяется "
        "после softmax). Невыбранным позициям ставь −inf ДО softmax."))


# ---------------------------------------------------------------------------
# DSAttention
# ---------------------------------------------------------------------------

def _dsa(impl, top_k, seed=0):
    torch.manual_seed(seed)
    return impl.DSAttention(C, H, HI, DI, top_k).eval()


def _dense(m, x, mask=None):
    t = x.shape[1]
    with torch.no_grad():
        q, k, v = (p(x).view(B, t, H, C // H).transpose(1, 2) for p in (m.q_proj, m.k_proj, m.v_proj))
        out = _SDPA(q, k, v, is_causal=mask is None, attn_mask=None if mask is None else mask.unsqueeze(1))
        return m.o_proj(out.transpose(1, 2).reshape(B, t, C))


def test_dsa_large_k_equals_dense(impl):
    """DSAttention: при top_k ≥ T — ровно обычное causal multi-head attention"""
    m = _dsa(impl, top_k=64)
    x = _x()
    with torch.no_grad():
        out = m(x)
    assert tuple(out.shape) == (B, T, C), f"Выход формы {tuple(out.shape)}, ожидалось {(B, T, C)}."
    assert_close(out, _dense(m, x), atol=1e-5, rtol=1e-4, msg=lambda _: (
        "При top_k ≥ T индексатор выбирает все прошлые ключи — DSA должна совпасть с обычным causal attention. "
        "Проверь q/k/v → [B, H, T, d], маску из topk_mask и o_proj."))


def test_dsa_small_k_matches_oracle(impl):
    """DSAttention: при top_k = 3 каждый запрос смотрит на 3 ключа, выбранных индексатором"""
    m = _dsa(impl, top_k=3)
    x = _x()
    with torch.no_grad():
        out, keep = m(x, return_mask=True)
        idx_scores = _oracle_index(m.indexer, x)
    assert tuple(keep.shape) == (B, T, T), "return_mask=True → (выход, маска выбранных ключей [B, T, T])."
    want = torch.zeros(B, T, T, dtype=torch.bool).scatter_(-1, idx_scores.topk(3, -1).indices, True) & _causal(T)
    assert torch.equal(keep, want), "Маска выбранных ключей не совпадает с top-3 по оценкам индексатора."
    assert_close(out, _dense(m, x, want), atol=1e-5, rtol=1e-4, msg=lambda _: (
        "Выход не совпадает с attention по выбранным ключам: маска одна на все головы, softmax только по ним."))
    assert not torch.allclose(out, _dense(m, x), atol=1e-3), (
        "При top_k = 3 < T выход совпал с плотным attention — маска индексатора не применяется.")


def test_dsa_causal(impl):
    """DSAttention: выход в позиции t не зависит от токенов t + 1, …"""
    m = _dsa(impl, top_k=4)
    x = _x()
    with torch.no_grad():
        out = m(x)
        x2 = x.clone()
        x2[:, 8:] = torch.randn(B, T - 8, C)
        out2 = m(x2)
    assert_close(out[:, :8], out2[:, :8], atol=1e-6, rtol=1e-5, msg=lambda _: (
        "Выход позиций < 8 изменился после замены токенов 8… → утечка будущего (в индексаторе или в маске)."))
