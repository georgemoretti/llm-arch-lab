"""Тесты «Mamba-2 (SSD)»: упрощённый Mamba-2 (SSD — state space duality).

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: segsum даёт суммы a_{s+1..t} под диагональю и −inf над ней; рекуррентный скан совпадает с
эталонным; квадратичная форма (L ⊙ CBᵀ)(Δx) + Dx == скану (двойственность SSM ↔ attention); кусками с
переносом состояния == целиком; будущее не протекает; при A = 0 и D = 0 Mamba-2 — это causal линейное
внимание без нормировки (q = C, k = B, v = Δ·x).
"""

import pytest
import torch
from torch.testing import assert_close

Bt, T, H, P, N = 2, 12, 3, 4, 5


def _inputs(t=T, seed=2):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(Bt, t, H, P, generator=g)
    dt = torch.nn.functional.softplus(torch.randn(Bt, t, H, generator=g) - 1)   # Δ > 0
    A = -torch.rand(H, generator=g) * 2 - 0.1                                 # A < 0
    B = torch.randn(Bt, t, N, generator=g)
    C = torch.randn(Bt, t, N, generator=g)
    D = torch.randn(H, generator=g)
    return x, dt, A, B, C, D


def _oracle(x, dt, A, B, C, D, state=None, *, decay_after=False, no_dt=False, no_skip=False):
    """Независимый скан по формуле; флаги — типичные ошибки."""
    b, t, h, p = x.shape
    S = torch.zeros(b, h, B.shape[-1], p) if state is None else state
    ys = []
    for i in range(t):
        a = torch.exp(dt[:, i] * A)[..., None, None]
        w = torch.einsum("bn,bhp->bhnp", B[:, i], x[:, i] * (1 if no_dt else dt[:, i, :, None]))
        S = a * (S + w) if decay_after else a * S + w
        y = torch.einsum("bn,bhnp->bhp", C[:, i], S)
        ys.append(y if no_skip else y + D[:, None] * x[:, i])
    return torch.stack(ys, 1), S


# ---------------------------------------------------------------------------
# segsum
# ---------------------------------------------------------------------------

def test_segsum(impl):
    """segsum: out[t, s] = a_{s+1} + … + a_t при s ≤ t (на диагонали 0), −inf при s > t"""
    a = torch.randn(2, 7, generator=torch.Generator().manual_seed(1))
    out = impl.segsum(a)
    assert tuple(out.shape) == (2, 7, 7), f"Форма {tuple(out.shape)}, ожидалось (2, 7, 7)."
    assert torch.isneginf(out[:, ~torch.ones(7, 7, dtype=torch.bool).tril()]).all(), (
        "Над диагональю (s > t) должно быть −inf: после exp там 0 — будущее не влияет на прошлое.")
    expected = torch.zeros(2, 7, 7)
    for t in range(7):
        for s in range(t + 1):
            expected[:, t, s] = a[:, s + 1:t + 1].sum(-1)
    lower = torch.ones(7, 7, dtype=torch.bool).tril()
    if not torch.allclose(out[:, lower], expected[:, lower], atol=1e-5):
        if torch.allclose(out[:, 3, 3], a[:, 3], atol=1e-5):
            pytest.fail("На диагонали стоит a_t, а должен быть 0: сумма идёт по r = s+1 … t, при s = t она пустая "
                        "(токен t не затухает сам относительно себя).", pytrace=False)
        pytest.fail("Под диагональю ожидалась сумма a_{s+1} + … + a_t. Подсказка: cumsum_t − cumsum_s.", pytrace=False)


# ---------------------------------------------------------------------------
# Рекуррентный скан
# ---------------------------------------------------------------------------

def test_recurrent_matches_oracle(impl):
    """ssd_recurrent: h_t = e^{Δ_t A} h_{t−1} + Δ_t B_t x_tᵀ, y_t = C_tᵀ h_t + D x_t"""
    inp = _inputs()
    y, S = impl.ssd_recurrent(*inp)
    assert tuple(y.shape) == (Bt, T, H, P) and tuple(S.shape) == (Bt, H, N, P), (
        f"Формы y {tuple(y.shape)}, state {tuple(S.shape)}; ожидалось {(Bt, T, H, P)} и {(Bt, H, N, P)}.")
    if torch.allclose(y, _oracle(*inp)[0], atol=1e-5, rtol=1e-4):
        assert_close(S, _oracle(*inp)[1], atol=1e-5, rtol=1e-4)
        return
    variants = {
        "затухание применено и к новой записи: e^{Δ_t A} умножает только h_{t−1}, а Δ_t B_t x_tᵀ добавляется после":
            dict(decay_after=True),
        "в записи нет Δ_t: вход дискретизуется как Δ_t · B_t x_tᵀ (шаг Δ — «насколько сильно запомнить токен»)":
            dict(no_dt=True),
        "нет skip-связи D · x_t": dict(no_skip=True),
    }
    for name, kw in variants.items():
        if torch.allclose(y, _oracle(*inp, **kw)[0], atol=1e-5, rtol=1e-4):
            pytest.fail(f"Скан совпал с вариантом «{name}».", pytrace=False)
    pytest.fail("Рекуррентный скан не совпадает с формулой. Состояние h [b, H, N, P]: B_t и C_t общие для голов "
                "(размер N), x_t — по головам.", pytrace=False)


def test_recurrent_chunks(impl):
    """ssd_recurrent: кусками 5 + 1 + 6 с переносом состояния == целиком; размер состояния не растёт"""
    x, dt, A, B, C, D = _inputs()
    full, S_full = impl.ssd_recurrent(x, dt, A, B, C, D)
    outs, S, s = [], None, 0
    for c in (5, 1, 6):
        sl = slice(s, s + c)
        o, S = impl.ssd_recurrent(x[:, sl], dt[:, sl], A, B[:, sl], C[:, sl], D, S)
        outs.append(o)
        s += c
        assert tuple(S.shape) == (Bt, H, N, P), "Состояние всегда [b, H, N, P], независимо от длины куска."
    assert_close(torch.cat(outs, 1), full, atol=1e-5, rtol=1e-4, msg=lambda _: (
        "Кусками с переносом state результат другой: если state передан, скан продолжается с него."))
    assert_close(S, S_full, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Квадратичная форма (двойственность)
# ---------------------------------------------------------------------------

def test_quadratic_equals_recurrent(impl):
    """ssd_quadratic: (L ⊙ CBᵀ)(Δ·x) + D·x с L = exp(segsum(ΔA)) — ровно то же, что скан"""
    inp = _inputs()
    y = impl.ssd_quadratic(*inp)
    assert tuple(y.shape) == (Bt, T, H, P), f"Форма {tuple(y.shape)}, ожидалось {(Bt, T, H, P)}."
    expected = _oracle(*inp)[0]
    if torch.allclose(y, expected, atol=1e-4, rtol=1e-4):
        return
    x, dt, A, B, C, D = inp
    M = torch.exp(_segsum_ref((dt * A).transpose(1, 2))) * (C @ B.transpose(1, 2))[:, None]
    hint = ""
    if torch.allclose(y, torch.einsum("bhts,bshp->bthp", M, x) + D[:, None] * x, atol=1e-4, rtol=1e-4):
        hint = " Значения надо брать как Δ_s · x_s (та же дискретизация, что в скане)."
    elif torch.allclose(y, torch.einsum("bhts,bshp->bthp", M.transpose(-2, -1), dt[..., None] * x) + D[:, None] * x,
                        atol=1e-4, rtol=1e-4):
        hint = " Матрица транспонирована: строка t — запрос (C_t), столбец s — ключ (B_s)."
    pytest.fail("Квадратичная форма не совпала со сканом: M[t, s] = L[t, s] · (C_t · B_s), y_t = Σ_s M[t, s] Δ_s x_s + "
                "D x_t; L = exp(segsum(Δ·A)) по каждой голове (ось времени — последняя)." + hint, pytrace=False)


def _segsum_ref(a):
    t = a.shape[-1]
    cs = a.cumsum(-1)
    return (cs[..., :, None] - cs[..., None, :]).masked_fill(~torch.ones(t, t, dtype=torch.bool).tril(), float("-inf"))


def test_quadratic_causal(impl):
    """ssd_quadratic: выход в позиции t не зависит от токенов t + 1, …"""
    x, dt, A, B, C, D = _inputs()
    y = impl.ssd_quadratic(x, dt, A, B, C, D)
    x2, B2 = x.clone(), B.clone()
    x2[:, 8:] += 3
    B2[:, 8:] += 3
    y2 = impl.ssd_quadratic(x2, dt, A, B2, C, D)
    assert_close(y[:, :8], y2[:, :8], atol=1e-6, rtol=1e-5, msg=lambda _: (
        "Выход позиций < 8 изменился после изменения токенов 8… → маска L пропускает будущее (exp(−inf) = 0 над "
        "диагональю)."))


def test_zero_decay_is_linear_attention(impl):
    """При A = 0 и D = 0 Mamba-2 = causal линейное внимание без нормировки: q = C, k = B, v = Δ·x"""
    x, dt, _, B, C, _ = _inputs()
    A0, D0 = torch.zeros(H), torch.zeros(H)
    y = impl.ssd_quadratic(x, dt, A0, B, C, D0)
    scores = (C @ B.transpose(1, 2)).tril()                                  # [b, T, T]
    expected = torch.einsum("bts,bshp->bthp", scores, dt[..., None] * x)
    assert_close(y, expected, atol=1e-4, rtol=1e-4, msg=lambda _: (
        "При A = 0 затухания нет (L — просто causal-маска из единиц), и SSD превращается в Σ_{s≤t} (C_t·B_s) Δ_s x_s — "
        "линейное внимание из «Линейное внимание» без φ и без знаменателя."))
    y_rec, _ = impl.ssd_recurrent(x, dt, A0, B, C, D0)
    assert_close(y_rec, expected, atol=1e-4, rtol=1e-4)
