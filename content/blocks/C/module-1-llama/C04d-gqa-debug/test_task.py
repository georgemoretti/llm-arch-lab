"""Тесты «Найди баг: GQA и MQA» (= тесты «GQA и MQA», но без подсказок, называющих баг).

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Независимый оракул — torch.nn.functional.scaled_dot_product_attention на явно размноженных
KV-головах (сохранён до того, как тесты запрещают его в коде пользователя).
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

B, T, C = 2, 10, 32
ATOL, RTOL = 1e-5, 1e-4
_SDPA = F.scaled_dot_product_attention


@pytest.fixture(autouse=True)
def _forbid_sdpa(monkeypatch):
    """Готовый scaled_dot_product_attention запрещён: attention с группами пишем сами."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Использован F.scaled_dot_product_attention. В этой задаче attention "
                             "пишется вручную (как в «Causal Multi-Head Attention»): scores → маска → softmax → @ v.")

    monkeypatch.setattr(F, "scaled_dot_product_attention", forbidden)


def _x(t: int = T, seed: int = 1):
    return torch.randn(B, t, C, generator=torch.Generator().manual_seed(seed))


def _make(impl, n_heads, n_kv_heads, seed=0):
    torch.manual_seed(seed)
    m = impl.GroupedQueryAttention(C, n_heads, n_kv_heads)
    return m.eval()


def _heads(m, x):
    Bx, Tx, _ = x.shape
    H, H_kv = m.n_heads, m.n_kv_heads
    D = C // H
    q = m.q_proj(x).view(Bx, Tx, H, D).transpose(1, 2)
    k = m.k_proj(x).view(Bx, Tx, H_kv, D).transpose(1, 2)
    v = m.v_proj(x).view(Bx, Tx, H_kv, D).transpose(1, 2)
    return q, k, v


def _oracle(m, x, scale=None, causal=True):
    """GQA через sdpa: голова запроса h использует KV-голову h // n_rep."""
    with torch.no_grad():
        q, k, v = _heads(m, x)
        n_rep = m.n_heads // m.n_kv_heads
        k, v = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)
        out = _SDPA(q, k, v, is_causal=causal, scale=scale)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return m.o_proj(out)


def _diagnose(m, x, out) -> str:
    variants = {
        "забыто деление scores на √D": _oracle(m, x, scale=1.0),
        "нет causal-маски — токены видят будущее": _oracle(m, x, causal=False),
    }
    for hint, cand in variants.items():
        if cand.shape == out.shape and torch.allclose(out, cand, atol=1e-4, rtol=1e-3):
            return f"Похоже, {hint}."
    return ("Сравни, при каких (H, H_kv) тест проходит, а при каких нет, и проверь, какую KV-голову "
            "получает каждая голова запроса.")


# ---------------------------------------------------------------------------
# repeat_kv
# ---------------------------------------------------------------------------

def test_repeat_kv_shape(impl):
    """repeat_kv: [B, H_kv, T, D] → [B, H_kv·n_rep, T, D]"""
    x = torch.randn(2, 3, 5, 4)
    for n_rep in (1, 2, 4):
        out = impl.repeat_kv(x, n_rep)
        assert tuple(out.shape) == (2, 3 * n_rep, 5, 4), (
            f"n_rep={n_rep}: форма {tuple(out.shape)} вместо {(2, 3 * n_rep, 5, 4)}. Размножать нужно "
            "ось голов (dim=1), остальные оси не трогать.")


def test_repeat_kv_layout(impl):
    """repeat_kv: голова запроса h получает KV-голову h // n_rep"""
    x = torch.arange(3.0).view(1, 3, 1, 1).expand(1, 3, 2, 2).contiguous()   # голова i заполнена числом i
    out = impl.repeat_kv(x, 2)
    got = out[0, :, 0, 0].tolist()
    expected = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    if got != expected:
        hint = ""
        pytest.fail(f"Номера KV-голов по головам запросов: {got}, ожидалось {expected}.{hint}", pytrace=False)
    assert_close(impl.repeat_kv(x, 1), x, atol=0, rtol=0, msg=lambda _: "При n_rep = 1 (обычный MHA) тензор не должен меняться.")


# ---------------------------------------------------------------------------
# kv_cache_bytes
# ---------------------------------------------------------------------------

def test_kv_cache_bytes(impl):
    """kv_cache_bytes: Llama 2 7B (MHA) и Llama 3 8B (GQA) — числа из их config.json"""
    # Llama 2 7B: 32 слоя, 32 головы, num_key_value_heads = 32, head_dim = 4096 / 32 = 128; fp16
    got = impl.kv_cache_bytes(32, 32, 128, 4096, batch=1, bytes_per_elem=2)
    expected = 2 * 1024 ** 3
    if got != expected:
        hint = ""
        if got == expected // 2:
            hint = " Ровно в 2 раза меньше: забыт множитель 2 — хранятся и K, и V."
        elif got == expected // 32:
            hint = " В 32 раза меньше: забыто умножение на число слоёв (у каждого слоя свой кеш)."
        pytest.fail(f"Llama 2 7B, контекст 4096, fp16: {got} байт, ожидалось {expected} (= 2 ГиБ).{hint}",
                    pytrace=False)
    # Llama 3 8B: 32 слоя, 8 KV-голов, head_dim 128; bf16; контекст 8192
    got = impl.kv_cache_bytes(32, 8, 128, 8192, batch=1, bytes_per_elem=2)
    assert got == 1024 ** 3, (
        f"Llama 3 8B, 8 KV-голов, контекст 8192, bf16: {got} байт, ожидалось 1 ГиБ. Считать нужно "
        "по KV-головам (n_kv_heads), а не по головам запросов.")
    assert impl.kv_cache_bytes(2, 1, 4, 10, batch=3, bytes_per_elem=4) == 2 * 2 * 1 * 4 * 10 * 3 * 4, (
        "Размер должен расти линейно по batch и по bytes_per_elem.")


# ---------------------------------------------------------------------------
# GroupedQueryAttention
# ---------------------------------------------------------------------------

def test_output_shape(impl):
    """forward: [B, T, C] → [B, T, C] для MHA, GQA и MQA; k_proj/v_proj в n_rep раз уже q_proj"""
    x = _x()
    m = _make(impl, 8, 2)
    D = C // 8
    assert tuple(m.k_proj.weight.shape) == (2 * D, C) and tuple(m.v_proj.weight.shape) == (2 * D, C), (
        f"k_proj.weight {tuple(m.k_proj.weight.shape)}: у GQA K/V-проекции дают только n_kv_heads·D = {2 * D} "
        "выходов — в этом и экономия. Конструктор дан готовым, верни его к исходному виду.")
    for H, H_kv in [(4, 4), (4, 2), (4, 1), (8, 2)]:
        out = _make(impl, H, H_kv)(x)
        assert tuple(out.shape) == (B, T, C), (
            f"H={H}, H_kv={H_kv}: форма выхода {tuple(out.shape)} вместо {(B, T, C)}. Головы собираются "
            "обратно в [B, T, H·D] перед o_proj.")


@pytest.mark.parametrize("H,H_kv", [(4, 2), (8, 2), (8, 4)])
def test_gqa_matches_oracle(impl, H, H_kv):
    """forward (GQA): совпадает с attention, где каждая группа голов делит одну KV-голову"""
    m = _make(impl, H, H_kv)
    x = _x()
    with torch.no_grad():
        out = m(x)
    expected = _oracle(m, x)
    if out.shape != expected.shape or not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        pytest.fail(f"H={H}, H_kv={H_kv}: выход не совпадает с оракулом "
                    f"(max |Δ| = {(out - expected).abs().max():.3g}). {_diagnose(m, x, out)}", pytrace=False)


def test_mha_special_case(impl):
    """H_kv = H: GQA превращается в обычный multi-head attention"""
    m = _make(impl, 4, 4)
    x = _x()
    with torch.no_grad():
        out = m(x)
    expected = _oracle(m, x)
    if not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        pytest.fail(f"При H_kv = H (n_rep = 1) должен получаться обычный MHA, но выход другой. "
                    f"{_diagnose(m, x, out)}", pytrace=False)


def test_mqa_special_case(impl):
    """H_kv = 1 (MQA): все головы делят одни K и V"""
    m = _make(impl, 4, 1)
    x = _x()
    with torch.no_grad():
        out = m(x)
    expected = _oracle(m, x)
    if not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        pytest.fail(f"При H_kv = 1 (multi-query attention) выход не совпадает с оракулом. "
                    f"{_diagnose(m, x, out)}", pytrace=False)


def test_equals_mha_with_shared_weights(impl):
    """GQA = MHA, у которого веса K/V внутри группы одинаковые"""
    gqa = _make(impl, 4, 2)
    mha = _make(impl, 4, 4)
    D = C // 4
    with torch.no_grad():
        mha.q_proj.weight.copy_(gqa.q_proj.weight)
        mha.o_proj.weight.copy_(gqa.o_proj.weight)
        # KV-голова g превращается в головы 2g и 2g+1 у MHA
        for name in ("k_proj", "v_proj"):
            w = getattr(gqa, name).weight.view(2, D, C)
            getattr(mha, name).weight.copy_(w.repeat_interleave(2, dim=0).reshape(4 * D, C))
        x = _x()
        a, b = gqa(x), mha(x)
    if not torch.allclose(a, b, atol=1e-4, rtol=1e-4):
        pytest.fail("GQA с 2 KV-головами должен давать то же, что MHA, где головы 0–1 используют копию "
                    "KV-головы 0, а головы 2–3 — копию KV-головы 1. Это не так — проверь, какая KV-голова "
                    f"достаётся каждой голове запроса. {_diagnose(gqa, x, a)}", pytrace=False)


def test_causal_no_future_leak(impl):
    """forward: изменение токена t+1 не меняет выход в позициях ≤ t"""
    m = _make(impl, 4, 2)
    x = _x()
    t = 4
    x2 = x.clone()
    x2[:, t + 1:] += torch.randn_like(x2[:, t + 1:]) * 3
    with torch.no_grad():
        a, b = m(x), m(x2)
    delta = (a[:, : t + 1] - b[:, : t + 1]).abs().max().item()
    assert delta < 1e-5, (
        f"Выход в позициях ≤ {t} изменился (max |Δ| = {delta:.3g}), когда поменяли только токены после "
        f"{t} — утечка будущего. Нужна causal-маска (как в «Causal Multi-Head Attention»).")


def test_batch_independence(impl):
    """forward: примеры в батче не влияют друг на друга"""
    m = _make(impl, 8, 2)
    x = _x()
    with torch.no_grad():
        full = m(x)
        single = m(x[1:2])
    assert_close(full[1:2], single, atol=1e-5, rtol=1e-4, msg=lambda _: (
        "Выход для примера 1 зависит от примера 0. Похоже, при разбиении на головы перепутаны оси "
        "(view без transpose или transpose не тех осей)."))
