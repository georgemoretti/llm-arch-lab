"""Тесты C20d (= тесты C20 без подсказок, называющих баг): линейное внимание.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: φ(x) = elu(x) + 1 > 0; некаузальная форма через φ(K)ᵀV совпадает с явной формулой;
causal-форма — взвешенное среднее только по прошлым значениям (константное v → тот же выход,
будущее не влияет); рекуррентная форма == параллельной, по кускам с переносом состояния == целиком;
состояние (S, z) имеет фиксированный размер [d, dv] + [d] независимо от длины. Softmax запрещён.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

B, H, T, D, DV = 2, 2, 12, 8, 6
EPS = 1e-6


@pytest.fixture(autouse=True)
def _no_softmax(monkeypatch):
    def banned(*a, **kw):
        raise AssertionError("Линейное внимание обходится без softmax: exp(q·k) заменён на φ(q)·φ(k), "
                             "нормировка — делением на сумму весов.")
    monkeypatch.setattr(torch, "softmax", banned)
    monkeypatch.setattr(F, "softmax", banned)
    monkeypatch.setattr(torch.Tensor, "softmax", banned)


def _qkv(t=T, seed=2):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, H, t, D, generator=g), torch.randn(B, H, t, D, generator=g),
            torch.randn(B, H, t, DV, generator=g))


def _phi(x):
    return F.elu(x) + 1


def _weights(q, k, causal=True, phi=_phi):
    A = phi(q) @ phi(k).transpose(-2, -1)
    return A.tril() if causal else A


def _oracle(q, k, v, causal=True):
    A = _weights(q, k, causal)
    return (A @ v) / (A.sum(-1, keepdim=True) + EPS)


# ---------------------------------------------------------------------------
# feature_map и некаузальная форма
# ---------------------------------------------------------------------------

def test_feature_map(impl):
    """feature_map: φ(x) = elu(x) + 1 — строго положительна"""
    x = torch.linspace(-6, 6, 101)
    out = impl.feature_map(x)
    assert (out > 0).all(), (
        "φ(x) должна быть строго положительной: тогда все веса φ(q)·φ(k) > 0 и знаменатель не обращается в 0 "
        "(ReLU даёт нули при x < 0 — у такого запроса может не оказаться ни одного ненулевого веса).")
    assert_close(out, F.elu(x) + 1, msg=lambda _: "Ожидалась φ(x) = elu(x) + 1 (как в статье Katharopoulos et al.).")


def test_noncausal_matches_oracle(impl):
    """linear_attention_noncausal: φ(q)(φ(K)ᵀV) / φ(q)·Σφ(k) — то же, что явная формула с матрицей [T, T]"""
    q, k, v = _qkv()
    out = impl.linear_attention_noncausal(q, k, v)
    assert tuple(out.shape) == (B, H, T, DV), f"Выход формы {tuple(out.shape)}, ожидалось {(B, H, T, DV)}."
    assert_close(out, _oracle(q, k, v, causal=False), atol=1e-5, rtol=1e-4, msg=lambda _: (
        "Некаузальное линейное внимание не совпадает с Σ_s φ(q)·φ(k_s) v_s / Σ_s φ(q)·φ(k_s). Порядок умножения: "
        "сначала φ(K)ᵀV [d, dv] и Σφ(k) [d], потом запросы."))


# ---------------------------------------------------------------------------
# Параллельная causal-форма
# ---------------------------------------------------------------------------

def test_causal_matches_oracle(impl):
    """linear_attention (causal): o_t = Σ_{s≤t} φ(q_t)·φ(k_s) v_s / Σ_{s≤t} φ(q_t)·φ(k_s)"""
    q, k, v = _qkv()
    out = impl.linear_attention(q, k, v)
    assert tuple(out.shape) == (B, H, T, DV), f"Выход формы {tuple(out.shape)}, ожидалось {(B, H, T, DV)}."
    if not torch.allclose(out, _oracle(q, k, v), atol=1e-5, rtol=1e-4):
        pytest.fail("Causal линейное внимание не совпадает с эталонной формулой.",
                    pytrace=False)


def test_causal_constant_values(impl):
    """linear_attention: при v_s = const выход = const (веса каждого запроса суммируются в 1)"""
    q, k, _ = _qkv()
    v = torch.full((B, H, T, DV), 3.0)
    out = impl.linear_attention(q, k, v)
    err = (out - 3.0).abs().amax(dim=(0, 1, 3))
    if (err > 1e-4).any():
        t = int((err > 1e-4).nonzero()[0])
        pytest.fail(f"Все значения = 3, а выход в позиции {t} = {out[0, 0, t, 0]:.4f}. Выход — взвешенное СРЕДНЕЕ "
                    "прошлых значений: веса каждого запроса должны суммироваться в 1.", pytrace=False)


def test_causal_no_future_leak(impl):
    """linear_attention: выход в позиции t не зависит от токенов t + 1, …"""
    q, k, v = _qkv()
    out = impl.linear_attention(q, k, v)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    for z in (q2, k2, v2):
        z[:, :, 7:] = torch.randn_like(z[:, :, 7:])
    out2 = impl.linear_attention(q2, k2, v2)
    assert_close(out[:, :, :7], out2[:, :, :7], msg=lambda _: (
        "Выход позиций < 7 изменился после замены токенов 7… → будущее протекает (в числитель или знаменатель)."))


# ---------------------------------------------------------------------------
# Рекуррентная форма
# ---------------------------------------------------------------------------

def test_recurrent_matches_parallel(impl):
    """linear_attention_recurrent: тот же выход, что у параллельной формы; состояние S = Σφ(k)vᵀ, z = Σφ(k)"""
    q, k, v = _qkv()
    out, state = impl.linear_attention_recurrent(q, k, v)
    assert tuple(out.shape) == (B, H, T, DV), f"Выход формы {tuple(out.shape)}, ожидалось {(B, H, T, DV)}."
    assert isinstance(state, tuple) and len(state) == 2, "Вторым значением верни состояние (S, z)."
    S, z = state
    assert tuple(S.shape) == (B, H, D, DV) and tuple(z.shape) == (B, H, D), (
        f"Состояние: S {tuple(S.shape)}, z {tuple(z.shape)}; ожидалось S {(B, H, D, DV)} = Σ φ(k) vᵀ "
        f"(внешнее произведение) и z {(B, H, D)} = Σ φ(k).")
    assert_close(S, _phi(k).transpose(-2, -1) @ v, atol=1e-5, rtol=1e-4, msg=lambda _: (
        "После всей последовательности S должно равняться Σ_t φ(k_t) v_tᵀ = φ(K)ᵀV."))
    assert_close(z, _phi(k).sum(-2), atol=1e-5, rtol=1e-4, msg=lambda _: "z должно равняться Σ_t φ(k_t).")
    if not torch.allclose(out, _oracle(q, k, v), atol=1e-5, rtol=1e-4):
        bad = ((out - _oracle(q, k, v)).abs() > 1e-5).any(-1).any(0).any(0).nonzero()
        t = int(bad[0]) if len(bad) else -1
        pytest.fail(f"Рекуррентная форма расходится с параллельной начиная с t = {t}. Порядок шага: СНАЧАЛА "
                    "обновить S и z токеном t, ПОТОМ читать φ(q_t)ᵀS_t / φ(q_t)ᵀz_t — токен видит сам себя.",
                    pytrace=False)


def test_recurrent_chunks_with_state(impl):
    """linear_attention_recurrent: кусками 3 + 1 + 5 + 3 с переносом состояния == целиком"""
    q, k, v = _qkv()
    full, (S_full, z_full) = impl.linear_attention_recurrent(q, k, v)
    outs, state, s = [], None, 0
    for c in (3, 1, 5, 3):
        o, state = impl.linear_attention_recurrent(q[:, :, s:s + c], k[:, :, s:s + c], v[:, :, s:s + c], state)
        outs.append(o)
        s += c
    assert_close(torch.cat(outs, 2), full, atol=1e-5, rtol=1e-4, msg=lambda _: (
        "Генерация кусками с переносом state не совпала с прогоном целиком: если state передан, начинать надо "
        "с него, а не с нулей."))
    assert_close(state[0], S_full, atol=1e-5, rtol=1e-4)
    assert_close(state[1], z_full, atol=1e-5, rtol=1e-4)


def test_state_size_constant(impl):
    """linear_attention_recurrent: размер состояния не растёт с длиной — в отличие от KV-cache"""
    sizes = []
    for t in (2, 24):
        q, k, v = _qkv(t)
        _, (S, z) = impl.linear_attention_recurrent(q, k, v)
        sizes.append(S.numel() + z.numel())
    assert sizes[0] == sizes[1] == B * H * (D * DV + D), (
        f"Состояние после 2 и 24 токенов: {sizes[0]} и {sizes[1]} чисел; ожидалось {B * H * (D * DV + D)} в обоих "
        "случаях. Рекуррентная форма хранит сжатую сумму, а не историю токенов.")
