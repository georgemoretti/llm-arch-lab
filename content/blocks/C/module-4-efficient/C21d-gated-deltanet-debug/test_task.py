"""Тесты C21d (= тесты C21 без подсказок, называющих баг): Gated DeltaNet.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: правило дельты ПЕРЕЗАПИСЫВАЕТ значение по ключу (линейное внимание из C20 только накапливает);
ортонормированные ключи при β = 1 извлекаются точно; β интерполирует между старым и новым значением;
гейт g → −∞ стирает память, g = 0 — обычный DeltaNet; кусками с переносом состояния == целиком; будущее
не протекает; слой Gated DeltaNet: β ∈ (0, 1), g ≤ 0, состояние фиксированного размера.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

B, H, T, DK, DV, C = 2, 2, 10, 8, 6, 32


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def _inputs(t=T, seed=2, gate=True):
    g_ = _gen(seed)
    q = _norm(torch.randn(B, H, t, DK, generator=g_))
    k = _norm(torch.randn(B, H, t, DK, generator=g_))
    v = torch.randn(B, H, t, DV, generator=g_)
    beta = torch.rand(B, H, t, generator=g_) * 0.8 + 0.1
    g = -torch.rand(B, H, t, generator=g_) * 0.5 if gate else None
    return q, k, v, beta, g


def _oracle(q, k, v, beta, g=None, state=None, *, delta=True, gate_after=False, read_before=False, scale=True):
    """Независимая реализация (как torch_recurrent_gated_delta_rule в HF); флаги — типичные ошибки."""
    Bq, Hq, t, dk = k.shape
    S = torch.zeros(Bq, Hq, dk, v.shape[-1]) if state is None else state
    q = q / math.sqrt(dk) if scale else q
    outs = []
    for i in range(t):
        if g is not None and not gate_after:
            S = S * g[:, :, i].exp()[..., None, None]
        if read_before:
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, i], S))
        kv_mem = torch.einsum("bhk,bhkv->bhv", k[:, :, i], S)
        upd = beta[:, :, i, None] * ((v[:, :, i] - kv_mem) if delta else v[:, :, i])
        S = S + k[:, :, i, :, None] * upd[:, :, None, :]
        if g is not None and gate_after:
            S = S * g[:, :, i].exp()[..., None, None]
        if not read_before:
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, i], S))
    return torch.stack(outs, 2), S


def _run(impl, *a, **kw):
    out = impl.gated_delta_rule(*a, **kw)
    assert isinstance(out, tuple) and len(out) == 2, "gated_delta_rule возвращает (out [B, H, T, dv], state [B, H, dk, dv])."
    return out


# ---------------------------------------------------------------------------
# l2norm и правило дельты
# ---------------------------------------------------------------------------

def test_l2norm(impl):
    """l2norm: единичная норма по последней оси, направление сохраняется"""
    x = torch.randn(3, 5, 8, generator=_gen(1)) * 7
    y = impl.l2norm(x)
    assert_close(y.norm(dim=-1), torch.ones(3, 5), atol=1e-5, rtol=0, msg=lambda _: (
        "После l2norm у каждого вектора (по последней оси) норма 1: x · rsqrt(Σx² + eps)."))
    assert_close(y, F.normalize(x, dim=-1), atol=1e-5, rtol=1e-4)


def test_delta_rule_matches_oracle(impl):
    """gated_delta_rule: совпадает с эталонной рекуррентной формой (без гейта и с гейтом)"""
    for gate in (False, True):
        q, k, v, beta, g = _inputs(gate=gate)
        out, S = _run(impl, q, k, v, beta, g)
        assert tuple(out.shape) == (B, H, T, DV) and tuple(S.shape) == (B, H, DK, DV), (
            f"Формы: out {tuple(out.shape)}, state {tuple(S.shape)}; ожидалось {(B, H, T, DV)} и {(B, H, DK, DV)}.")
        exp_out, exp_S = _oracle(q, k, v, beta, g)
        if torch.allclose(out, exp_out, atol=1e-5, rtol=1e-4):
            assert_close(S, exp_S, atol=1e-5, rtol=1e-4, msg=lambda _: "Выход верный, а итоговое состояние — нет.")
            continue
        pytest.fail(f"{'С гейтом' if gate else 'Без гейта (g=None)'}: выход не совпадает с эталонной рекуррентной формой.",
                    pytrace=False)


def _one_head(keys, values, beta=1.0):
    """[n, dk], [n, dv] → входы для B = H = 1."""
    n = keys.shape[0]
    return keys[None, None], values[None, None], torch.full((1, 1, n), beta)


def test_overwrite_by_key(impl):
    """Правило дельты: запись (k, v₁), затем (k, v₂) при β = 1 → по k читается v₂, а не v₁ + v₂"""
    k1 = _norm(torch.randn(DK, generator=_gen(5)))
    keys = torch.stack([k1, k1])
    v1, v2 = torch.randn(DV, generator=_gen(6)), torch.randn(DV, generator=_gen(7))
    k, v, beta = _one_head(keys, torch.stack([v1, v2]))
    out, S = _run(impl, k, k, v, beta)
    read = (S[0, 0].T @ k1)
    if torch.allclose(read, v1 + v2, atol=1e-4):
        pytest.fail("Память вернула v₁ + v₂ — это линейное внимание: старое значение по тому же ключу не стёрто. "
                    "Правило дельты сначала читает старое v_old = Sᵀk и пишет β·(v − v_old).", pytrace=False)
    assert_close(read, v2, atol=1e-4, rtol=1e-4, msg=lambda _: "Sᵀk после двух записей по одному ключу должно быть v₂.")
    assert_close(out[0, 0, 1] * math.sqrt(DK), v2, atol=1e-4, rtol=1e-4, msg=lambda _: (
        "Выход шага 1 при q = k: S₁ᵀk / √dk = v₂ / √dk."))


def test_orthogonal_keys_exact_recall(impl):
    """При ортонормированных ключах и β = 1 память хранит все пары точно (ассоциативная память)"""
    Q, _ = torch.linalg.qr(torch.randn(DK, DK, generator=_gen(8)))
    keys = Q[:5]                                            # 5 ортонормированных ключей
    values = torch.randn(5, DV, generator=_gen(9))
    k, v, beta = _one_head(keys, values)
    _, S = _run(impl, k, k, v, beta)
    assert_close(keys @ S[0, 0], values, atol=1e-4, rtol=1e-4, msg=lambda _: (
        "По каждому из ортонормированных ключей память должна вернуть ровно записанное значение: запись по k_i "
        "не трогает направления k_j ⟂ k_i."))


def test_beta_interpolates(impl):
    """β — сила записи: из (k, v₁) и записи (k, v₂) с β = 0.25 читается v₁ + 0.25·(v₂ − v₁)"""
    k1 = _norm(torch.randn(DK, generator=_gen(10)))
    v1, v2 = torch.randn(DV, generator=_gen(11)), torch.randn(DV, generator=_gen(12))
    k, v, _ = _one_head(torch.stack([k1, k1]), torch.stack([v1, v2]))
    beta = torch.tensor([[[1.0, 0.25]]])
    _, S = _run(impl, k, k, v, beta)
    assert_close(S[0, 0].T @ k1, v1 + 0.25 * (v2 - v1), atol=1e-4, rtol=1e-4, msg=lambda _: (
        "β умножает разность (v − v_old): при β = 0.25 значение сдвигается к новому на четверть."))


def test_gate_forgets(impl):
    """Гейт: g = 0 — обычный DeltaNet; g → −∞ на шаге t стирает всё, что было до t"""
    q, k, v, beta, _ = _inputs(gate=False)
    zero = torch.zeros(B, H, T)
    assert_close(_run(impl, q, k, v, beta, zero)[0], _run(impl, q, k, v, beta, None)[0], atol=1e-5, rtol=1e-4,
                 msg=lambda _: "При g = 0 (α = e⁰ = 1) Gated DeltaNet должен совпасть с DeltaNet (g=None).")
    g = zero.clone()
    g[:, :, 6] = -60.0                                      # α₆ = e⁻⁶⁰ ≈ 0
    out_a = _run(impl, q, k, v, beta, g)[0]
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    k2[:, :, :6] = _norm(torch.randn(B, H, 6, DK))
    v2[:, :, :6] = torch.randn(B, H, 6, DV)
    out_b = _run(impl, q2, k2, v2, beta, g)[0]
    assert_close(out_a[:, :, 6:], out_b[:, :, 6:], atol=1e-5, rtol=1e-4, msg=lambda _: (
        "После шага с g = −60 (α ≈ 0) выходы не должны зависеть от токенов до него: память стёрта. Гейт умножает "
        "состояние S ← exp(g_t)·S до записи токена t."))


def test_chunks_and_causality(impl):
    """gated_delta_rule: кусками 4 + 1 + 5 с переносом состояния == целиком; будущее не влияет на прошлое"""
    q, k, v, beta, g = _inputs()
    full, S_full = _run(impl, q, k, v, beta, g)
    outs, S, s = [], None, 0
    for c in (4, 1, 5):
        o, S = _run(impl, q[:, :, s:s + c], k[:, :, s:s + c], v[:, :, s:s + c], beta[:, :, s:s + c], g[:, :, s:s + c], S)
        outs.append(o)
        s += c
    assert_close(torch.cat(outs, 2), full, atol=1e-5, rtol=1e-4, msg=lambda _: (
        "По кускам с переносом state результат другой: если state передан, начинать надо с него."))
    assert_close(S, S_full, atol=1e-5, rtol=1e-4)
    v2 = v.clone()
    v2[:, :, 7:] += 5
    assert_close(_run(impl, q, k, v2, beta, g)[0][:, :, :7], full[:, :, :7], msg=lambda _: (
        "Выход позиций < 7 изменился после изменения значений 7… → будущее протекает."))


# ---------------------------------------------------------------------------
# Слой GatedDeltaNet
# ---------------------------------------------------------------------------

def _layer(impl, seed=0):
    torch.manual_seed(seed)
    return impl.GatedDeltaNet(C, H, DK, DV).eval()


def test_layer_gates(impl):
    """GatedDeltaNet.gates: β = σ(b) ∈ (0, 1), g = −exp(A_log)·softplus(a + dt_bias) ≤ 0, формы [B, H, T]"""
    m = _layer(impl)
    x = torch.randn(B, T, C, generator=_gen(3))
    with torch.no_grad():
        beta, g = m.gates(x)
        exp_beta = torch.sigmoid(m.b_proj(x)).transpose(1, 2)
        exp_g = (-m.A_log.exp() * F.softplus(m.a_proj(x) + m.dt_bias)).transpose(1, 2)
    assert tuple(beta.shape) == (B, H, T) and tuple(g.shape) == (B, H, T), (
        f"Формы β {tuple(beta.shape)}, g {tuple(g.shape)}; ожидалось {(B, H, T)} — головы на оси 1, как у q/k/v.")
    assert_close(beta, exp_beta, msg=lambda _: "β = σ(b_proj(x)) — сила записи в (0, 1).")
    assert (g <= 0).all(), "g — логарифм коэффициента забывания α = exp(g) ≤ 1, поэтому g ≤ 0."
    assert_close(g, exp_g, msg=lambda _: (
        "g = −exp(A_log) · softplus(a_proj(x) + dt_bias) — как Δ·A в Mamba-2: A < 0 по голове, Δ > 0 по токену."))


def test_layer_forward(impl):
    """GatedDeltaNet.forward: проекции → L2-норма q, k по голове → правило дельты → o_proj; state [B, H, dk, dv]"""
    m = _layer(impl)
    x = torch.randn(B, T, C, generator=_gen(4))
    with torch.no_grad():
        y, S = m(x)
        q = _norm(m.q_proj(x).view(B, T, H, DK)).transpose(1, 2)
        k = _norm(m.k_proj(x).view(B, T, H, DK)).transpose(1, 2)
        v = m.v_proj(x).view(B, T, H, DV).transpose(1, 2)
        beta = torch.sigmoid(m.b_proj(x)).transpose(1, 2)
        g = (-m.A_log.exp() * F.softplus(m.a_proj(x) + m.dt_bias)).transpose(1, 2)
        o, exp_S = _oracle(q, k, v, beta, g)
        exp_y = m.o_proj(o.transpose(1, 2).reshape(B, T, H * DV))
    assert tuple(y.shape) == (B, T, C), f"Выход формы {tuple(y.shape)}, ожидалось {(B, T, C)}."
    assert tuple(S.shape) == (B, H, DK, DV), f"Состояние формы {tuple(S.shape)}, ожидалось {(B, H, DK, DV)}."
    assert_close(y, exp_y, atol=1e-5, rtol=1e-4, msg=lambda _: (
        "Слой не совпадает с оракулом. Проверь: L2-норма q и k по последней оси ПОСЛЕ разбиения на головы; "
        "v без нормы; β и g из gates(x); выход [B, H, T, dv] → [B, T, H·dv] → o_proj."))
    with torch.no_grad():
        y1, S1 = m(x[:, :6])
        y2, _ = m(x[:, 6:], S1)
    assert_close(torch.cat([y1, y2], 1), y, atol=1e-5, rtol=1e-4, msg=lambda _: (
        "Генерация кусками с переносом состояния слоя не совпала с прогоном целиком."))
