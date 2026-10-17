"""Тесты «Causal Multi-Head Attention»: causal multi-head self-attention.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

B, H, T, D = 2, 4, 8, 16
D_MODEL, N_HEADS = 32, 4
ATOL, RTOL = 1e-5, 1e-4

# Независимый «оракул» для сравнения — встроенная функция PyTorch.
# Сохраняем её до того, как запретим пользователю её вызывать.
_SDPA = F.scaled_dot_product_attention


@pytest.fixture(autouse=True)
def _forbid_builtin_attention(monkeypatch):
    """Готовые реализации attention в этой задаче запрещены."""

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Использована готовая реализация attention (F.scaled_dot_product_attention / "
            "nn.MultiheadAttention). В этой задаче механизм нужно написать руками: "
            "QKᵀ/√D → маска → softmax → @V."
        )

    monkeypatch.setattr(F, "scaled_dot_product_attention", forbidden)
    monkeypatch.setattr(nn.MultiheadAttention, "forward", forbidden)


def _qkv(t: int = T, d: int = D):
    g = torch.Generator().manual_seed(1)
    return tuple(torch.randn(B, H, t, d, generator=g) for _ in range(3))


def _diagnose_attention(out: torch.Tensor, q, k, v) -> str:
    """Пытается узнать типичную ошибку по тому, с каким «неправильным» вариантом совпал выход."""
    if torch.isnan(out).any():
        return ("В выходе есть NaN. Обычно это значит, что в какой-то строке ВСЕ позиции "
                "замаскированы −∞ (softmax от одних −∞ = NaN). Проверь, что маска разрешает "
                "диагональ: токен должен видеть сам себя (tril, а не triu / diagonal=-1).")
    d = q.shape[-1]
    t = q.shape[-2]
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
    raw = q @ k.transpose(-2, -1)

    def attn(scores, mask=causal, dim=-1):
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return F.softmax(scores, dim=dim) @ v

    variants = {
        "забыто деление на √D: без него логиты растут с размером головы и softmax насыщается":
            lambda: attn(raw),
        "делишь на D, а не на √D": lambda: attn(raw / d),
        "маска не применена — позиции видят будущее": lambda: attn(raw / math.sqrt(d), mask=None),
        "маска перевёрнута: позиция видит будущее вместо прошлого (triu вместо tril?)":
            lambda: attn(raw / math.sqrt(d), mask=torch.triu(torch.ones(t, t, dtype=torch.bool))),
    }
    for hint, fn in variants.items():
        with torch.no_grad():
            candidate = fn()
        if not torch.isnan(candidate).any() and torch.allclose(out, candidate, atol=1e-4, rtol=1e-3):
            return f"Похоже, {hint}."
    try:
        wrong_dim = F.softmax(
            (raw / math.sqrt(d)).masked_fill(~causal, float("-inf")), dim=-2).nan_to_num() @ v
        if torch.allclose(out, wrong_dim, atol=1e-4, rtol=1e-3):
            return "Похоже, softmax взят по оси запросов (dim=-2). Нормировать нужно по ключам: dim=-1."
    except RuntimeError:
        pass
    return "Типичную ошибку распознать не удалось: проверь масштаб, маску и ось softmax по шагам."


# ---------------------------------------------------------------------------
# causal_attention
# ---------------------------------------------------------------------------

def test_attention_output_shape(impl):
    """causal_attention: форма выхода [B, H, T, D]"""
    q, k, v = _qkv()
    out = impl.causal_attention(q, k, v)
    assert out.shape == (B, H, T, D), (
        f"Ожидалась форма {(B, H, T, D)}, получено {tuple(out.shape)}. "
        "Веса внимания [B, H, T, T] нужно умножить на v [B, H, T, D]."
    )


def test_attention_weights_sum_to_one(impl):
    """causal_attention: веса строки суммируются в 1 (если v ≡ 1, выход ≡ 1)"""
    q, k, _ = _qkv()
    ones = torch.ones(B, H, T, D)
    out = impl.causal_attention(q, k, ones)
    assert not torch.isnan(out).any(), _diagnose_attention(out, q, k, ones)
    assert_close(out, ones, atol=ATOL, rtol=RTOL, msg=lambda m: (
        "Когда все значения v равны 1, выход тоже должен быть равен 1: выход — это взвешенное "
        "среднее v, а веса в каждой строке после softmax суммируются в 1. Если это не так — "
        "softmax взят не по той оси, или маска применена после softmax (обнулением весов)."))


def test_first_position_sees_only_itself(impl):
    """causal_attention: первая позиция видит только себя → выход[0] == v[0]"""
    q, k, v = _qkv()
    out = impl.causal_attention(q, k, v)
    assert not torch.isnan(out).any(), _diagnose_attention(out, q, k, v)
    assert_close(out[..., 0, :], v[..., 0, :], atol=ATOL, rtol=RTOL, msg=lambda m: (
        "Токену в позиции 0 доступен только он сам, поэтому весь его вес внимания (=1) "
        "приходится на v[0], и выход должен совпасть с v[0]. Не совпал → позиция 0 видит "
        "будущие токены: маска отсутствует или перевёрнута."))


def test_attention_no_future_leak(impl):
    """causal_attention: изменение будущих k, v не меняет выход в прошлых позициях"""
    q, k, v = _qkv()
    base = impl.causal_attention(q, k, v)
    for t in range(T - 1):
        k2, v2 = k.clone(), v.clone()
        k2[..., t + 1:, :] = torch.randn_like(k2[..., t + 1:, :]) * 5
        v2[..., t + 1:, :] = torch.randn_like(v2[..., t + 1:, :]) * 5
        out = impl.causal_attention(q, k2, v2)
        if not torch.allclose(out[..., : t + 1, :], base[..., : t + 1, :], atol=ATOL, rtol=RTOL):
            pytest.fail(
                f"Выход в позициях 0..{t} изменился, когда поменяли только k, v в позициях "
                f"{t + 1}..{T - 1} → утечка будущего через маску. Позиция t должна видеть "
                "только позиции ≤ t.", pytrace=False)


def test_attention_matches_oracle(impl):
    """causal_attention: численно совпадает с эталонной формулой softmax(QKᵀ/√D + mask)·V"""
    q, k, v = _qkv()
    expected = _SDPA(q, k, v, is_causal=True)
    out = impl.causal_attention(q, k, v)
    if out.shape != expected.shape or not torch.allclose(out, expected, atol=ATOL, rtol=RTOL):
        hint = _diagnose_attention(out, q, k, v) if out.shape == expected.shape else "Неверная форма выхода."
        max_err = (out - expected).abs().max().item() if out.shape == expected.shape else float("nan")
        pytest.fail(f"Выход не совпадает с эталоном (max |Δ| = {max_err:.3g}). {hint}", pytrace=False)


def test_attention_large_head_dim(impl):
    """causal_attention: при D=64 масштабирование 1/√D критично"""
    q, k, v = _qkv(d=64)
    expected = _SDPA(q, k, v, is_causal=True)
    out = impl.causal_attention(q, k, v)
    if not torch.allclose(out, expected, atol=ATOL, rtol=RTOL):
        pytest.fail(f"При D=64 выход расходится с эталоном. {_diagnose_attention(out, q, k, v)}", pytrace=False)


# ---------------------------------------------------------------------------
# CausalSelfAttention (модуль)
# ---------------------------------------------------------------------------

def _make_pair(impl, ref, bias=True):
    """Модуль пользователя с теми же весами, что у эталона."""
    torch.manual_seed(0)
    reference = ref.CausalSelfAttention(D_MODEL, N_HEADS, bias=bias)
    mine = impl.CausalSelfAttention(D_MODEL, N_HEADS, bias=bias)
    try:
        mine.load_state_dict(reference.state_dict())
    except RuntimeError as exc:
        pytest.fail(
            "Не удалось загрузить веса эталона в твой модуль: имена или формы параметров "
            f"отличаются. Не меняй конструктор (c_attn: [C → 3C], c_proj: [C → C]).\n{exc}",
            pytrace=False)
    return mine.eval(), reference.eval()


def _tokens(t: int = T, b: int = B):
    return torch.randn(b, t, D_MODEL, generator=torch.Generator().manual_seed(2))


def test_module_output_shape(impl, ref):
    """CausalSelfAttention: [B, T, C] → [B, T, C]"""
    mine, _ = _make_pair(impl, ref)
    out = mine(_tokens())
    assert out.shape == (B, T, D_MODEL), (
        f"Ожидалась форма {(B, T, D_MODEL)}, получено {tuple(out.shape)}. После attention "
        "головы [B, H, T, D] нужно вернуть в [B, T, H, D] и склеить в [B, T, C].")


def test_module_matches_reference(impl, ref):
    """CausalSelfAttention: численно совпадает с эталоном при одинаковых весах"""
    mine, reference = _make_pair(impl, ref)
    x = _tokens()
    with torch.no_grad():
        out, expected = mine(x), reference(x)
    if out.shape != expected.shape:
        pytest.fail(f"Форма {tuple(out.shape)} вместо {tuple(expected.shape)}.", pytrace=False)
    if not torch.allclose(out, expected, atol=ATOL, rtol=RTOL):
        with torch.no_grad():
            first_ok = torch.allclose(out[:, 0], expected[:, 0], atol=ATOL, rtol=RTOL)
        hint = (
            "Позиция 0 совпадает, а дальше нет — проверь маску/масштаб в causal_attention."
            if first_ok else
            "Не совпадает даже позиция 0. Частые причины: q, k, v взяты из c_attn не в порядке "
            "[q | k | v]; разбиение на головы сделано как view(B, T, D, H) или view(B, H, T, D) "
            "вместо view(B, T, H, D).transpose(1, 2); головы склеены без transpose обратно; "
            "забыт c_proj."
        )
        pytest.fail(f"Выход модуля расходится с эталоном (max |Δ| = "
                    f"{(out - expected).abs().max().item():.3g}). {hint}", pytrace=False)


def test_module_no_future_leak(impl, ref):
    """CausalSelfAttention: токен t+1 не влияет на выход в позициях ≤ t"""
    mine, _ = _make_pair(impl, ref)
    x = _tokens()
    with torch.no_grad():
        base = mine(x)
        for t in range(T - 1):
            x2 = x.clone()
            x2[:, t + 1] += 3.0
            out = mine(x2)
            leaked = ~torch.isclose(out[:, : t + 1], base[:, : t + 1], atol=ATOL, rtol=RTOL)
            if leaked.any():
                pos = int(leaked.any(dim=(0, 2)).nonzero()[0])
                pytest.fail(
                    f"Изменили только токен в позиции {t + 1}, а выход изменился в позиции {pos} ≤ {t} "
                    "→ утечка будущего через маску.", pytrace=False)


def test_module_batch_independence(impl, ref):
    """CausalSelfAttention: примеры в батче обрабатываются независимо"""
    mine, _ = _make_pair(impl, ref)
    x = _tokens(b=3)
    with torch.no_grad():
        together = mine(x)
        alone = torch.cat([mine(x[i : i + 1]) for i in range(3)])
    assert_close(together, alone, atol=ATOL, rtol=RTOL, msg=lambda m: (
        "Выход для примера зависит от других примеров батча. Обычно это view/reshape, "
        "который перемешивает ось B с T или H (например, view(T, B, ...) или transpose(0, 1))."))


def test_module_single_token(impl, ref):
    """CausalSelfAttention: работает при T=1 и без bias"""
    mine, reference = _make_pair(impl, ref, bias=False)
    x = _tokens(t=1)
    with torch.no_grad():
        out, expected = mine(x), reference(x)
    assert_close(out, expected, atol=ATOL, rtol=RTOL, msg=lambda m: (
        "При T=1 и bias=False выход не совпал с эталоном. Проверь, что размеры берутся из "
        "x.shape, а не зашиты константами, и что маска строится под текущую длину T."))


def test_module_gradients_flow(impl, ref):
    """CausalSelfAttention: градиент доходит до всех параметров и до входа"""
    mine, _ = _make_pair(impl, ref)
    x = _tokens().requires_grad_(True)
    mine(x).pow(2).sum().backward()
    for name, p in mine.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (
            f"Параметр {name} не получил (конечного ненулевого) градиента: он не участвует в "
            "вычислении или где-то есть .detach()/torch.no_grad()/NaN.")
    assert x.grad is not None and torch.isfinite(x.grad).all(), "Градиент по входу не дошёл или содержит NaN."
