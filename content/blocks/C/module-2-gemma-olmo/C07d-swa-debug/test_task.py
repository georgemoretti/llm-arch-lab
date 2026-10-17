"""Тесты «Найди баг: Sliding window attention» (= тесты «Sliding window attention», но без подсказок, называющих баг).

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: окно из W позиций (включая свою), causal сохраняется; при W ≥ T и W=None — обычный
causal attention (как LlamaAttention из «KV-cache»); выход в позиции t не зависит от токенов старше t−W+1;
L слоёв SWA видят на L·(W−1) позиций назад; расписание слоёв Gemma 3 (5 local : 1 global).
"""

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from llmlab.core.llama import LlamaAttention

B, T, C, H, H_KV = 2, 16, 32, 4, 2
D = C // H
THETA = 10000.0
_SDPA = F.scaled_dot_product_attention


def _x(t: int = T, seed: int = 1):
    return torch.randn(B, t, C, generator=torch.Generator().manual_seed(seed))


def _make(impl, window, seed=0):
    torch.manual_seed(seed)
    return impl.SlidingWindowAttention(C, H, H_KV, window, rope_theta=THETA).eval()


def _oracle_mask(T, S, W):
    q = torch.arange(S - T, S)[:, None]
    k = torch.arange(S)[None, :]
    m = k <= q
    return m if W is None else m & (q - k < W)


def _show(m: torch.Tensor) -> str:
    return "\n".join("    " + " ".join("1" if v else "·" for v in row) for row in m.tolist())


def _diagnose_mask(m, T, S, W) -> str:
    q = torch.arange(S - T, S)[:, None]
    k = torch.arange(S)[None, :]
    variants = {
        "окно на одну позицию уже: `p − j < W − 1`. В окно W=3 входят сам токен и два предыдущих":
            (k <= q) & (q - k < W - 1),
        "нет causal-условия: окно симметричное |p − j| < W, запрос видит будущее":
            (q - k).abs() < W,
        "окна нет — это обычная causal-маска (window проигнорирован)": k <= q,
        "окно считается от начала последовательности, а не от запроса (j < W)": (k <= q) & (k < W),
        "запросы считаются стоящими на позициях 0..T−1, а не на последних T из S "
        "(при S > T маску надо сдвигать, как в «KV-cache»)": (k <= q - (S - T)) & (q - (S - T) - k < W),
    }
    for hint, cand in variants.items():
        if cand.shape == m.shape and torch.equal(cand, m):
            return f" Похоже, {hint}."
    return ""


def _check_mask(impl, T, S, W):
    m = impl.sliding_window_mask(T, S, W)
    expected = _oracle_mask(T, S, W)
    assert isinstance(m, torch.Tensor) and m.dtype == torch.bool, (
        "sliding_window_mask должна вернуть bool-тензор (True = можно смотреть).")
    assert tuple(m.shape) == (T, S), f"Маска формы {tuple(m.shape)}, ожидалось ({T}, {S}): строки — запросы, столбцы — ключи."
    if not torch.equal(m, expected):
        pytest.fail(f"Маска для T={T}, S={S}, window={W} неверна.{_diagnose_mask(m, T, S, W)}\n"
                    f"  Получено:\n{_show(m)}\n  Ожидалось:\n{_show(expected)}", pytrace=False)


# ---------------------------------------------------------------------------
# sliding_window_mask
# ---------------------------------------------------------------------------

def test_mask_small_example(impl):
    """sliding_window_mask(6, window=3): каждый запрос видит себя и два предыдущих токена"""
    _check_mask(impl, 6, 6, 3)


def test_mask_rows_count(impl):
    """Строка p разрешает ровно min(p + 1, W) ключей — ни больше, ни меньше"""
    for W in (1, 2, 5):
        m = impl.sliding_window_mask(12, None, W)
        counts = m.sum(-1).tolist()
        expected = [min(p + 1, W) for p in range(12)]
        assert counts == expected, (
            f"window={W}: число разрешённых ключей по строкам {counts}, ожидалось {expected}. "
            f"Окно из W позиций включает сам токен.{_diagnose_mask(m, 12, 12, W)}")


def test_mask_degenerate_windows(impl):
    """window=None и window ≥ T — обычная causal-маска; window=1 — только диагональ"""
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    for W in (None, T, 10 * T):
        m = impl.sliding_window_mask(T, None, W)
        assert torch.equal(m, causal), (
            f"window={W}: окно не меньше длины последовательности, маска должна совпасть с causal (tril)."
            f"{_diagnose_mask(m, T, T, W) if W is not None else ''}")
    m = impl.sliding_window_mask(T, None, 1)
    assert torch.equal(m, torch.eye(T, dtype=torch.bool)), (
        f"window=1: токен видит только себя — маска должна быть диагональной.{_diagnose_mask(m, T, T, 1)}")


def test_mask_with_offset(impl):
    """Запросы — последние T из S позиций (пригодится для KV-cache в «KV-cache для окна»): маска сдвинута"""
    _check_mask(impl, 3, 10, 4)
    _check_mask(impl, 1, 10, 4)


# ---------------------------------------------------------------------------
# layer_types
# ---------------------------------------------------------------------------

def _diagnose_types(types, n, p) -> str:
    variants = {
        "full-слой — первый в каждой группе (`i % pattern == 0`). У Gemma слой 0 — sliding, а "
        "global — последний в группе: (i + 1) % pattern == 0":
            ["full_attention" if i % p == 0 else "sliding_attention" for i in range(n)],
        "перепутаны местами sliding и full":
            ["full_attention" if (i + 1) % p else "sliding_attention" for i in range(n)],
    }
    for hint, cand in variants.items():
        if types == cand:
            return f" Похоже, {hint}."
    return ""


def test_layer_types_gemma3(impl):
    """layer_types(62, 6): Gemma 3 27B — 52 sliding, 10 full, full на слоях 5, 11, …, 59"""
    # Источник: google/gemma-3-27b-pt config.json (text_config): num_hidden_layers = 62,
    # sliding_window_pattern = 6; правило — configuration_gemma3.py в transformers.
    types = impl.layer_types(62, 6)
    assert isinstance(types, list) and len(types) == 62, "layer_types должна вернуть список длины n_layers."
    assert set(types) <= {"sliding_attention", "full_attention"}, (
        f"Значения — строки 'sliding_attention' / 'full_attention' (как layer_types в HF), получено {set(types)}.")
    full = [i for i, t in enumerate(types) if t == "full_attention"]
    expected = list(range(5, 62, 6))
    assert full == expected, (
        f"full-слои на индексах {full}, ожидалось {expected}.{_diagnose_types(types, 62, 6)}")


def test_layer_types_other_patterns(impl):
    """pattern=2 — Gemma 2 (чередование, начиная со sliding); pattern=1 — все слои full"""
    types = impl.layer_types(42, 2)
    assert types[:4] == ["sliding_attention", "full_attention"] * 2 and types.count("full_attention") == 21, (
        f"Gemma 2 9B (42 слоя, pattern 2) чередует sliding/full, начиная со sliding: получено {types[:4]}…"
        f"{_diagnose_types(types, 42, 2)}")
    assert impl.layer_types(5, 1) == ["full_attention"] * 5, "pattern=1: каждый слой — full."
    one_b = impl.layer_types(26, 6)
    assert one_b.count("full_attention") == 4, (
        f"Gemma 3 1B: 26 слоёв и pattern 6 → 4 full-слоя (5, 11, 17, 23), получено {one_b.count('full_attention')}.")


# ---------------------------------------------------------------------------
# SlidingWindowAttention
# ---------------------------------------------------------------------------

def _oracle(m, x, W):
    """GQA + RoPE из модуля 1 и F.sdpa с маской-окном."""
    from llmlab.core.llama import apply_rope, rope_cos_sin
    t = x.shape[1]
    with torch.no_grad():
        cos, sin = rope_cos_sin(torch.arange(t), D, THETA)
        q = apply_rope(m.q_proj(x).view(B, t, H, D).transpose(1, 2), cos, sin)
        k = apply_rope(m.k_proj(x).view(B, t, H_KV, D).transpose(1, 2), cos, sin).repeat_interleave(H // H_KV, 1)
        v = m.v_proj(x).view(B, t, H_KV, D).transpose(1, 2).repeat_interleave(H // H_KV, 1)
        out = _SDPA(q, k, v, attn_mask=_oracle_mask(t, t, W))
        return m.o_proj(out.transpose(1, 2).reshape(B, t, C))


def test_attention_matches_oracle(impl):
    """SlidingWindowAttention(window=4) совпадает с оракулом: GQA + RoPE + маска-окно"""
    m = _make(impl, 4)
    x = _x()
    with torch.no_grad():
        out = m(x)
    assert tuple(out.shape) == (B, T, C), f"Выход формы {tuple(out.shape)}, ожидалось {(B, T, C)}."
    expected = _oracle(m, x, 4)
    if not torch.allclose(out, expected, atol=1e-5, rtol=1e-4):
        hint = ""
        if torch.allclose(out, _oracle(m, x, None), atol=1e-5):
            hint = " Выход совпал с ПОЛНЫМ causal attention — маска-окно не применяется."
        elif torch.allclose(out, _oracle(m, x, 5), atol=1e-5):
            hint = " Выход совпал с attention, у которого окно 5, а не 4."
        pytest.fail(f"SlidingWindowAttention(window=4) не совпадает с оракулом.{hint}", pytrace=False)


def test_full_window_is_llama_attention(impl):
    """window=None и window ≥ T — ровно LlamaAttention из «KV-cache» (без кеша): SWA — это только маска"""
    for W in (None, T):
        m = _make(impl, W)
        llama = LlamaAttention(C, H, H_KV, rope_theta=THETA).eval()
        llama.load_state_dict(m.state_dict())
        x = _x()
        with torch.no_grad():
            assert_close(m(x), llama(x), atol=1e-5, rtol=1e-4,
                         msg=lambda _: f"window={W}: окно не ограничивает ничего, выход должен совпасть с "
                                       "LlamaAttention (те же веса). Проверь RoPE и causal-часть маски.")


def test_locality(impl):
    """Выход в позиции t не зависит от токенов старше t − W + 1, но зависит от токена t − W + 1"""
    W, t = 4, 12
    m = _make(impl, W)
    x = _x()
    far = x.clone()
    far[:, : t - W + 1] += 3 * torch.randn_like(far[:, : t - W + 1])
    edge = x.clone()
    edge[:, t - W + 1] += 3 * torch.randn_like(edge[:, t - W + 1])
    with torch.no_grad():
        a, b, c = m(x)[:, t], m(far)[:, t], m(edge)[:, t]
    d_far = (a - b).abs().max().item()
    assert d_far < 1e-5, (
        f"Выход в позиции {t} изменился (|Δ| = {d_far:.3g}) от токенов 0..{t - W}, которые дальше окна W={W}. "
        "Окно пропускает лишнее.")
    d_edge = (a - c).abs().max().item()
    assert d_edge > 1e-4, (
        f"Выход в позиции {t} не зависит от токена {t - W + 1}, а он в окне (W={W}: позиции {t - W + 1}..{t}). "
        "Окно на одну позицию уже, чем нужно.")


def test_no_future_leak(impl):
    """Causal сохраняется: изменение токенов после t не меняет выход в позициях ≤ t"""
    m = _make(impl, 4)
    x = _x()
    x2 = x.clone()
    x2[:, 9:] += 3 * torch.randn_like(x2[:, 9:])
    with torch.no_grad():
        delta = (m(x)[:, :9] - m(x2)[:, :9]).abs().max().item()
    assert delta < 1e-5, (
        f"Выход в позициях 0..8 зависит от токенов 9+ (|Δ| = {delta:.3g}). Окно должно быть "
        "ограничено и слева (W), и справа (causal): j ≤ p.")


def test_stacked_receptive_field(impl):
    """L слоёв SWA с остатком: позиция t «видит» на L·(W−1) токенов назад — и не дальше"""
    W, L, t = 3, 3, 15
    torch.manual_seed(3)
    layers = [impl.SlidingWindowAttention(C, H, H_KV, W, rope_theta=THETA).eval() for _ in range(L)]
    x = _x().requires_grad_()
    h = x
    for layer in layers:
        h = h + layer(h)
    h[:, t].sum().backward()
    influence = x.grad.abs().sum(dim=(0, 2))            # [T]: влияние каждого входа на выход в t
    reach = L * (W - 1)
    seen = [s for s in range(T) if influence[s] > 1e-7]
    expected = list(range(t - reach, t + 1))
    assert seen == expected, (
        f"{L} слоя SWA с окном {W}: выход в позиции {t} зависит от входов {seen}, ожидалось "
        f"{t - reach}..{t}. Каждый слой расширяет поле зрения на W − 1 = {W - 1} позиции, поэтому "
        f"{L} слоя видят на {reach} назад — так SWA-модели и обрабатывают длинный контекст.")
