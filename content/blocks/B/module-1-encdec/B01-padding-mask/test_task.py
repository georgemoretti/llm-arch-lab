"""Тесты «Padding-маска и MHA».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракулы — F.scaled_dot_product_attention и nn.MultiheadAttention (сохранены до запрета).
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

B, H, T, S, D = 2, 4, 5, 7, 8
C = H * D
PAD = 0
_SDPA = F.scaled_dot_product_attention
_MHA_FORWARD = nn.MultiheadAttention.forward


@pytest.fixture(autouse=True)
def _forbid_builtin_attention(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Использована готовая реализация attention (F.scaled_dot_product_attention / "
                             "nn.MultiheadAttention). Здесь механизм пишется руками: q·kᵀ/√D → маска → softmax → @v.")

    monkeypatch.setattr(F, "scaled_dot_product_attention", forbidden)
    monkeypatch.setattr(nn.MultiheadAttention, "forward", forbidden)


def _ids():
    """Два текста: длины 4 и 7, PAD справа."""
    return torch.tensor([[5, 6, 7, 8, PAD, PAD, PAD], [3, 4, 5, 6, 7, 8, 9]])


def _qkv(seed=1, t=T, s=S, dv=D):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, H, t, D, generator=g), torch.randn(B, H, s, D, generator=g),
            torch.randn(B, H, s, dv, generator=g))


def _oracle(q, k, v, mask):
    # monkeypatch подменил F.scaled_dot_product_attention, а _SDPA — исходная функция
    return _SDPA(q, k, v, attn_mask=mask)


def _diagnose(out, q, k, v, mask) -> str:
    def run(scale=None, after=False):
        s = q @ k.transpose(-2, -1) * (scale if scale is not None else 1 / math.sqrt(q.shape[-1]))
        m = mask
        if m is not None and not after:
            s = s.masked_fill(~m, float("-inf"))
        w = s.softmax(-1)
        if m is not None and after:
            w = w.masked_fill(~m, 0.0)
        return w @ v

    if torch.isnan(out).any():
        return ("в выходе nan: какая-то строка оценок целиком −inf. Маска запрещает запросы (строки), а должна "
                "запрещать PAD-ключи (столбцы): форма маски [B, 1, 1, S], последняя ось — ключи.")
    if torch.allclose(out, run(scale=1.0), atol=1e-5):
        return "оценки не поделены на √D: логиты слишком большие, softmax почти one-hot."
    if mask is not None and torch.allclose(out, run(after=True), atol=1e-5):
        return ("маска применена после softmax: веса PAD-ключей обнулены, но строки весов больше не суммируются "
                "в 1, а PAD всё равно влияет через знаменатель softmax.")
    return "выход не совпадает с эталоном: q @ kᵀ / √D → маска (−inf, до softmax) → softmax по ключам → @ v."


def test_padding_mask(impl):
    """padding_mask: bool [B, 1, 1, S], True у настоящих токенов"""
    ids = _ids()
    m = impl.padding_mask(ids, PAD)
    assert m.dtype == torch.bool, f"маска — bool (True = можно смотреть), получено {m.dtype}"
    if torch.equal(m.squeeze(), ids == PAD) or torch.equal(m.reshape(-1), (ids == PAD).reshape(-1)):
        pytest.fail("маска перевёрнута: True должен стоять у настоящих токенов, False — у PAD. (В PyTorch у "
                    "key_padding_mask смысл обратный: True = игнорировать. В курсе True = можно смотреть.)",
                    pytrace=False)
    assert m.shape == (B, 1, 1, S), (
        f"форма [B, 1, 1, S] = [{B}, 1, 1, {S}], получено {tuple(m.shape)}. Оси 1 и 2 (головы и запросы) — "
        "размера 1, чтобы маска растягивалась по ним, а различалась только по ключам.")
    torch.testing.assert_close(m[:, 0, 0], ids != PAD, msg="True там, где токен не PAD")


def test_attention_cross_shapes_no_mask(impl):
    """attention без маски при T ≠ S и Dv ≠ D: формы и совпадение с оракулом"""
    q, k, v = _qkv(dv=3)
    out, w = impl.attention(q, k, v)
    assert out.shape == (B, H, T, 3) and w.shape == (B, H, T, S), (
        f"out [B, H, T, Dv] = [{B}, {H}, {T}, 3] и weights [B, H, T, S] = [{B}, {H}, {T}, {S}], получено "
        f"{tuple(out.shape)} и {tuple(w.shape)}. Запросов T, ключей S — матрица оценок прямоугольная.")
    want = _oracle(q, k, v, None)
    if not torch.allclose(out, want, atol=1e-5, rtol=1e-4):
        pytest.fail(_diagnose(out, q, k, v, None), pytrace=False)
    torch.testing.assert_close(w.sum(-1), torch.ones(B, H, T), msg="веса каждого запроса суммируются в 1")


def test_attention_padding_mask(impl):
    """attention с маской [B, 1, 1, S]: веса PAD-ключей ровно 0, строки суммируются в 1"""
    q, k, v = _qkv()
    mask = (_ids() != PAD)[:, None, None, :]
    out, w = impl.attention(q, k, v, mask)
    if not torch.allclose(out, _oracle(q, k, v, mask), atol=1e-5, rtol=1e-4):
        pytest.fail(_diagnose(out, q, k, v, mask), pytrace=False)
    assert torch.all(w[0, :, :, 4:] == 0), "веса PAD-ключей должны быть ровно 0"
    torch.testing.assert_close(w.sum(-1), torch.ones(B, H, T), msg="каждая строка весов суммируется в 1")


def test_attention_general_mask(impl):
    """attention с маской полной формы [B, 1, T, S] (например, causal + padding)"""
    q, k, v = _qkv(seed=2, t=S)
    mask = torch.tril(torch.ones(S, S, dtype=torch.bool)) & (_ids() != PAD)[:, None, None, :]
    out, _ = impl.attention(q, k, v, mask)
    if not torch.allclose(out, _oracle(q, k, v, mask), atol=1e-5, rtol=1e-4):
        pytest.fail(_diagnose(out, q, k, v, mask), pytrace=False)


def _mha_pair(impl, n_heads=H):
    torch.manual_seed(0)
    ours = impl.MultiHeadAttention(C, n_heads)
    torch_mha = nn.MultiheadAttention(C, n_heads, batch_first=True)
    with torch.no_grad():
        torch_mha.in_proj_weight.copy_(torch.cat([ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight]))
        torch_mha.in_proj_bias.copy_(torch.cat([ours.q_proj.bias, ours.k_proj.bias, ours.v_proj.bias]))
        torch_mha.out_proj.weight.copy_(ours.out_proj.weight)
        torch_mha.out_proj.bias.copy_(ours.out_proj.bias)
    return ours, torch_mha


def _torch_mha(mod, x, context, ids):
    # need_weights=True — путь без F.scaled_dot_product_attention (она подменена). key_padding_mask: True = PAD.
    out, _ = _MHA_FORWARD(mod, x, context, context, key_padding_mask=ids == PAD, need_weights=True)
    return out


def test_mha_matches_torch(impl):
    """MultiHeadAttention (self-attention с паддингом) == nn.MultiheadAttention с теми же весами"""
    ours, torch_mha = _mha_pair(impl)
    ids = _ids()
    x = torch.randn(B, S, C, generator=torch.Generator().manual_seed(3))
    out = ours(x, mask=(ids != PAD)[:, None, None, :])
    assert out.shape == (B, S, C), f"выход [B, T, C] = [{B}, {S}, {C}], получено {tuple(out.shape)}"
    want = _torch_mha(torch_mha, x, x, ids)
    if not torch.allclose(out, want, atol=1e-5, rtol=1e-4):
        one_ours, one_torch = _mha_pair(impl, n_heads=1)
        same_one = torch.allclose(one_ours(x), _torch_mha(one_torch, x, x, torch.ones_like(ids)), atol=1e-5)
        if same_one:
            pytest.fail("с одной головой выход верный, с четырьмя — нет. Значит, неверно режутся головы: "
                        "вектор каждой позиции [C] нужно разрезать на H кусков по D (view(B, L, H, D)), и только "
                        "потом переставить оси. Склейка обратно — в том же порядке.", pytrace=False)
        pytest.fail("выход не совпадает с nn.MultiheadAttention: q/k/v-проекции → головы → attention с маской → "
                    "склейка голов → out_proj.", pytrace=False)


def test_mha_cross_attention(impl):
    """context ≠ x: запросы из x [B, T, C], ключи и значения из context [B, S, C]; need_weights"""
    ours, torch_mha = _mha_pair(impl)
    g = torch.Generator().manual_seed(4)
    x, ctx, ids = torch.randn(B, T, C, generator=g), torch.randn(B, S, C, generator=g), _ids()
    try:
        res = ours(x, ctx, mask=(ids != PAD)[:, None, None, :], need_weights=True)
    except RuntimeError as e:
        pytest.fail(f"cross-attention при T = {T}, S = {S} упал: {e}. Запросы берутся из x, а ключи и значения — "
                    "из context; маска [B, 1, 1, S] относится к ключам.", pytrace=False)
    assert isinstance(res, tuple) and len(res) == 2, "при need_weights=True вернуть (out, weights)"
    out, w = res
    assert out.shape == (B, T, C), (f"выход [B, T, C] = [{B}, {T}, {C}]: длина — от запросов (x), а не от "
                                    f"context; получено {tuple(out.shape)}")
    assert w.shape == (B, H, T, S), f"веса [B, H, T, S] = [{B}, {H}, {T}, {S}], получено {tuple(w.shape)}"
    torch.testing.assert_close(out, _torch_mha(torch_mha, x, ctx, ids), atol=1e-5, rtol=1e-4,
                               msg="q — из x, k и v — из context")


def test_padding_invariance(impl):
    """Выход на настоящих токенах не зависит ни от числа PAD, ни от того, что лежит на PAD-позициях"""
    ours, _ = _mha_pair(impl)
    g = torch.Generator().manual_seed(5)
    real = torch.randn(1, 4, C, generator=g)
    alone = ours(real, mask=torch.ones(1, 1, 1, 4, dtype=torch.bool))
    for n_pad in (1, 3):
        x = torch.cat([real, 10 * torch.randn(1, n_pad, C, generator=g)], dim=1)
        mask = torch.tensor([[True] * 4 + [False] * n_pad])[:, None, None, :]
        out = ours(x, mask=mask)
        assert torch.isfinite(out).all(), ("в выходе nan/inf. Маска должна запрещать PAD-ключи (столбцы), а не "
                                           "запросы: тогда в каждой строке остаётся хотя бы один разрешённый ключ.")
        if not torch.allclose(out[:, :4], alone, atol=1e-5):
            pytest.fail(f"при {n_pad} PAD справа выход на настоящих токенах изменился → PAD протекает в attention. "
                        "Маска [B, 1, 1, S] должна обнулять веса PAD-ключей для всех запросов.", pytrace=False)


def test_bidirectional_and_permutation(impl):
    """Энкодер двунаправленный: позиция 0 видит последний токен; без позиций перестановка входа переставляет выход"""
    ours, _ = _mha_pair(impl)
    g = torch.Generator().manual_seed(6)
    x = torch.randn(1, 6, C, generator=g)
    y = x.clone()
    y[:, -1] = torch.randn(C, generator=g)
    if torch.allclose(ours(x)[:, 0], ours(y)[:, 0], atol=1e-6):
        pytest.fail("выход позиции 0 не зависит от последнего токена. В энкодере нет causal-маски: каждый токен "
                    "видит весь текст.", pytrace=False)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    torch.testing.assert_close(ours(x[:, perm]), ours(x)[:, perm], atol=1e-5, rtol=1e-4,
                               msg="attention без позиций переставочно-эквивариантен: позиции токенам даёт B02")
