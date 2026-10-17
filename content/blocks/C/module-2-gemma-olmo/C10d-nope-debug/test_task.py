"""Тесты «Найди баг: NoPE» (= тесты «NoPE», но без подсказок, называющих баг).

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: расписание совпадает с config.json SmolLM3; RoPE-слой — это LlamaAttention из «KV-cache»;
NoPE-слой для последнего токена — «мешок слов» (перестановка предыдущих токенов ничего не меняет),
но благодаря causal-маске всё равно различает позиции одинаковых токенов.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from llmlab.core.llama import LlamaAttention

B, T, C, H, H_KV = 2, 12, 32, 4, 2
D = C // H
THETA = 10000.0
_SDPA = F.scaled_dot_product_attention

# Источник: HuggingFaceTB/SmolLM3-3B config.json — num_hidden_layers = 36, no_rope_layer_interval = 4,
# no_rope_layers (1 = слой С RoPE, 0 = NoPE; так в modeling_smollm3.py: use_rope = no_rope_layers[i]):
SMOLLM3_NO_ROPE_LAYERS = [1, 1, 1, 0] * 9


def _x(t: int = T, seed: int = 1):
    return torch.randn(B, t, C, generator=torch.Generator().manual_seed(seed))


def _make(impl, use_rope, seed=0):
    torch.manual_seed(seed)
    return impl.NoPEAttention(C, H, H_KV, use_rope=use_rope, rope_theta=THETA).eval()


def _diagnose_schedule(got, n, k) -> str:
    variants = {
        "значения инвертированы: True должно означать «RoPE есть»":
            [(i + 1) % k == 0 for i in range(n)],
    }
    for hint, cand in variants.items():
        if got == cand:
            return f" Похоже, {hint}."
    return ""


def test_schedule_smollm3(impl):
    """rope_layers(36, 4) совпадает с no_rope_layers из config.json SmolLM3-3B"""
    got = impl.rope_layers(36, 4)
    expected = [bool(v) for v in SMOLLM3_NO_ROPE_LAYERS]
    assert isinstance(got, list) and len(got) == 36, "rope_layers должна вернуть список длины n_layers."
    got = [bool(v) for v in got]
    if got != expected:
        nope = [i for i, v in enumerate(got) if not v]
        pytest.fail(f"NoPE-слои на индексах {nope}, а у SmolLM3 — 3, 7, 11, …, 35 (каждый 4-й, считая с 1)."
                    f"{_diagnose_schedule(got, 36, 4)}", pytrace=False)


def test_schedule_edge_cases(impl):
    """interval > n_layers — RoPE во всех слоях; interval = 1 — все слои NoPE"""
    assert [bool(v) for v in impl.rope_layers(6, 100)] == [True] * 6, (
        "interval = 100 при 6 слоях: ни один слой не «каждый сотый» — RoPE должен быть везде.")
    assert [bool(v) for v in impl.rope_layers(5, 1)] == [False] * 5, "interval = 1: каждый слой — NoPE."


def test_rope_layer_is_llama_attention(impl):
    """use_rope=True — ровно LlamaAttention из «KV-cache» (без кеша) с теми же весами"""
    m = _make(impl, True)
    llama = LlamaAttention(C, H, H_KV, rope_theta=THETA).eval()
    llama.load_state_dict(m.state_dict())
    x = _x()
    with torch.no_grad():
        assert_close(m(x), llama(x), atol=1e-5, rtol=1e-4,
                     msg=lambda _: "use_rope=True: слой должен совпасть с LlamaAttention (GQA + RoPE + causal). "
                                   "Проверь RoPE на q и k с позициями 0..T−1 и θ = self.rope_theta.")


def _oracle_nope(m, x, causal=True):
    t = x.shape[1]
    with torch.no_grad():
        q = m.q_proj(x).view(B, t, H, D).transpose(1, 2)
        k = m.k_proj(x).view(B, t, H_KV, D).transpose(1, 2).repeat_interleave(H // H_KV, 1)
        v = m.v_proj(x).view(B, t, H_KV, D).transpose(1, 2).repeat_interleave(H // H_KV, 1)
        out = _SDPA(q, k, v, is_causal=causal)
        return m.o_proj(out.transpose(1, 2).reshape(B, t, C))


def test_nope_layer_matches_oracle(impl):
    """use_rope=False: GQA + causal-маска, никакого поворота q и k"""
    m = _make(impl, False)
    x = _x()
    with torch.no_grad():
        out = m(x)
    if not torch.allclose(out, _oracle_nope(m, x), atol=1e-5, rtol=1e-4):
        hint = ""
        llama = LlamaAttention(C, H, H_KV, rope_theta=THETA).eval()
        llama.load_state_dict(m.state_dict())
        with torch.no_grad():
            if torch.allclose(out, llama(x), atol=1e-5):
                hint = " Выход совпал со слоем С RoPE — флаг use_rope не учитывается."
            elif torch.allclose(out, _oracle_nope(m, x, causal=False), atol=1e-5):
                hint = (" Выход совпал с attention БЕЗ causal-маски. NoPE убирает позиционное кодирование, "
                        "но не маску: декодер по-прежнему не должен видеть будущее.")
        pytest.fail(f"NoPE-слой не совпадает с оракулом (GQA + causal, без RoPE).{hint}", pytrace=False)


def test_nope_last_token_is_bag_of_words(impl):
    """NoPE: перестановка токенов 0..T−2 не меняет выход последнего токена; с RoPE — меняет"""
    x = _x()
    perm = torch.randperm(T - 1, generator=torch.Generator().manual_seed(3))
    xp = x.clone()
    xp[:, : T - 1] = x[:, perm]
    nope, rope = _make(impl, False), _make(impl, True)
    with torch.no_grad():
        d_nope = (nope(x)[:, -1] - nope(xp)[:, -1]).abs().max().item()
        d_rope = (rope(x)[:, -1] - rope(xp)[:, -1]).abs().max().item()
    assert d_nope < 1e-5, (
        f"NoPE-слой: после перестановки предыдущих токенов выход последнего изменился (|Δ| = {d_nope:.3g}). "
        "Без позиционного кодирования последний запрос видит ВСЕ предыдущие ключи, и softmax-сумма от их "
        "порядка не зависит — это «мешок слов». Похоже, позиции где-то всё же используются.")
    assert d_rope > 1e-4, (
        "RoPE-слой: перестановка предыдущих токенов не изменила выход последнего — значит, RoPE не применён, "
        "хотя use_rope=True.")


def test_nope_knows_positions_through_mask(impl):
    """NoPE + causal-маска различают позиции: одинаковые токены после первого дают разные выходы"""
    m = _make(impl, False)
    x = _x()
    same = x.clone()
    same[:, 1:] = x[:, 1:2]                       # токены 1..T−1 одинаковые, токен 0 — другой
    with torch.no_grad():
        out = m(same)[0, 1:]                      # [T−1, C]
    gaps = (out[1:] - out[:-1]).abs().max(dim=-1).values
    assert gaps.min().item() > 1e-5, (
        "Одинаковые токены на позициях 1, 2, …, T−1 дали одинаковые выходы. С causal-маской так быть не должно: "
        "запрос на позиции t видит t одинаковых ключей и один другой (токен 0), и вес токена 0 равен "
        "e^s₀ / (e^s₀ + t·e^s) — он зависит от t. Так NoPE-слой получает информацию о позиции без всякого "
        "позиционного кодирования. Проверь, что causal-маска на месте.")


def test_no_future_leak(impl):
    """NoPE-слой: изменение токенов после t не меняет выход в позициях ≤ t"""
    m = _make(impl, False)
    x = _x()
    x2 = x.clone()
    x2[:, 6:] += 3 * torch.randn_like(x2[:, 6:])
    with torch.no_grad():
        delta = (m(x)[:, :6] - m(x2)[:, :6]).abs().max().item()
    assert delta < 1e-5, (
        f"Выход в позициях 0..5 зависит от токенов 6+ (|Δ| = {delta:.3g}). NoPE убирает RoPE, но не маску.")


def test_build_attention_layers(impl):
    """build_attention_layers(8, …, interval=4): 8 слоёв NoPEAttention, без RoPE — слои 3 и 7"""
    torch.manual_seed(0)
    layers = impl.build_attention_layers(8, C, H, H_KV, 4, rope_theta=THETA)
    assert isinstance(layers, torch.nn.ModuleList) and len(layers) == 8, (
        "build_attention_layers должна вернуть nn.ModuleList из n_layers слоёв (ModuleList — чтобы параметры "
        "слоёв были видны оптимизатору и state_dict).")
    assert all(isinstance(layer, impl.NoPEAttention) for layer in layers), "Каждый слой — NoPEAttention."
    flags = [bool(layer.use_rope) for layer in layers]
    assert flags == [True, True, True, False] * 2, (
        f"use_rope по слоям: {flags}, ожидалось [T, T, T, F, T, T, T, F] — по расписанию rope_layers(8, 4)."
        f"{_diagnose_schedule(flags, 8, 4)}")
    assert all(layer.rope_theta == THETA for layer in layers), "rope_theta нужно передать в каждый слой."
