"""Тесты «SwiGLU FFN»: SwiGLU FFN.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

B, T, C, HID = 2, 8, 32, 96
ATOL, RTOL = 1e-5, 1e-4


@pytest.fixture(autouse=True)
def _forbid_builtin_silu(monkeypatch):
    """Готовая SiLU в этой задаче запрещена — её нужно написать самому (это одна строка)."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Использована готовая F.silu / nn.SiLU. Напиши SiLU сам: x · sigmoid(x).")

    monkeypatch.setattr(F, "silu", forbidden)
    monkeypatch.setattr(nn.SiLU, "forward", forbidden)


def _x(shape=(B, T, C), scale=1.0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(1)) * scale


def _silu(x):
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# silu
# ---------------------------------------------------------------------------

def test_silu_values(impl):
    """silu: совпадает с x · σ(x), включая отрицательные x"""
    x = torch.linspace(-6, 6, 101)
    out = impl.silu(x)
    expected = _silu(x)
    if out.shape != x.shape or not torch.allclose(out, expected, atol=ATOL, rtol=RTOL):
        hint = "Проверь формулу: x · sigmoid(x)."
        if out.shape == x.shape:
            if torch.allclose(out, torch.sigmoid(x), atol=1e-4):
                hint = "Это просто sigmoid(x) — забыто умножение на x."
            elif torch.allclose(out, torch.relu(x), atol=1e-4):
                hint = "Это ReLU. SiLU гладкая и слегка отрицательна при x < 0."
            elif torch.allclose(out, F.gelu(x), atol=1e-4) or torch.allclose(out, F.gelu(x, approximate="tanh"), atol=1e-4):
                hint = "Это GELU (как в GPT-2). Похожая форма, но Llama использует SiLU = x · σ(x)."
        pytest.fail(f"silu не совпадает с x · σ(x). {hint}", pytrace=False)


def test_silu_shape_properties(impl):
    """silu: SiLU(0) = 0, минимум ≈ −0.278 при x ≈ −1.28, форма сохраняется"""
    x = _x()
    assert impl.silu(x).shape == x.shape, "silu должна работать поэлементно и не менять форму."
    z = impl.silu(torch.zeros(3))
    assert_close(z, torch.zeros(3), atol=ATOL, rtol=0, msg=lambda _: "SiLU(0) должно быть 0.")
    grid = torch.linspace(-4, 0, 4001)
    mn = impl.silu(grid).min().item()
    assert abs(mn + 0.2785) < 1e-3, (
        f"Минимум SiLU должен быть ≈ −0.2785, получено {mn:.4f}. В отличие от ReLU, SiLU "
        "пропускает немного отрицательных значений — это часть её «гладкости».")


# ---------------------------------------------------------------------------
# llama_ffn_hidden_dim
# ---------------------------------------------------------------------------

def test_hidden_dim_llama2(impl):
    """llama_ffn_hidden_dim: Llama 2 7B (d=4096, multiple_of=256) → 11 008"""
    got = impl.llama_ffn_hidden_dim(4096, multiple_of=256)
    if got != 11008:
        hint = ""
        if got == 16384:
            hint = " Это 4·d, как в GPT-2: забыт множитель 2/3."
        elif got == 10752:
            hint = " Округлено вниз; нужно вверх до кратного multiple_of."
        elif got in (10922, 10923):
            hint = " Не сделано округление до кратного multiple_of."
        pytest.fail(f"Для d=4096 получено {got}, а в config.json Llama 2 7B intermediate_size = "
                    f"11008 = 2/3·4·4096 = 10922 → вверх до кратного 256.{hint}", pytrace=False)


def test_hidden_dim_llama3(impl):
    """llama_ffn_hidden_dim: Llama 3 8B (d=4096, multiple_of=1024, ×1.3) → 14 336"""
    got = impl.llama_ffn_hidden_dim(4096, multiple_of=1024, ffn_dim_multiplier=1.3)
    if got != 14336:
        hint = ""
        if got == 15360:
            hint = (" Похоже, множитель 1.3 применён после округления (или к 4·d) — "
                    "сначала 2/3·4d, затем ×1.3, затем округление.")
        elif got == 13312:
            hint = " Округлено вниз; нужно вверх."
        elif got == 11264:
            hint = " Множитель ffn_dim_multiplier не применён."
        pytest.fail(f"Для Llama 3 8B получено {got}, а в config.json intermediate_size = 14336 "
                    f"(10922 × 1.3 = 14198 → вверх до кратного 1024).{hint}", pytrace=False)


def test_hidden_dim_small(impl):
    """llama_ffn_hidden_dim: маленькие размеры, результат кратен multiple_of и ≥ 8d/3"""
    for d, m in [(64, 32), (48, 16), (32, 1), (96, 64)]:
        got = impl.llama_ffn_hidden_dim(d, multiple_of=m)
        expected = m * -(-int(8 * d / 3) // m)
        assert got == expected, (
            f"d={d}, multiple_of={m}: получено {got}, ожидалось {expected} "
            "(int(8d/3), округлённое вверх до кратного multiple_of).")


# ---------------------------------------------------------------------------
# SwiGLUFFN (модуль)
# ---------------------------------------------------------------------------

def _make_pair(impl, ref):
    torch.manual_seed(0)
    reference = ref.SwiGLUFFN(C, HID)
    mine = impl.SwiGLUFFN(C, HID)
    try:
        mine.load_state_dict(reference.state_dict())
    except RuntimeError as exc:
        pytest.fail("Не удалось загрузить веса эталона: имена или формы параметров отличаются. "
                    f"Не меняй конструктор (gate_proj, up_proj: C → F; down_proj: F → C; без bias).\n{exc}",
                    pytrace=False)
    return mine.eval(), reference.eval()


def _diagnose(out, x, m) -> str:
    g, u, d = m.gate_proj, m.up_proj, m.down_proj
    variants = {
        "SiLU применена к ветке up, а не gate (gate и up перепутаны)": lambda: d(_silu(u(x)) * g(x)),
        "SiLU применена после умножения веток, а не к ветке gate": lambda: d(_silu(g(x) * u(x))),
        "нет SiLU: ветки просто перемножаются (это bilinear-GLU)": lambda: d(g(x) * u(x)),
        "вместо SiLU стоит GELU — это GeGLU, а Llama использует SwiGLU": lambda: d(F.gelu(g(x)) * u(x)),
        "вместо SiLU стоит sigmoid — это классический GLU": lambda: d(torch.sigmoid(g(x)) * u(x)),
        "ветки складываются, а не перемножаются": lambda: d(_silu(g(x)) + u(x)),
        "ветка up не используется: это обычный MLP с SiLU": lambda: d(_silu(g(x))),
    }
    with torch.no_grad():
        for hint, fn in variants.items():
            if torch.allclose(out, fn(), atol=1e-4, rtol=1e-3):
                return f"Похоже, {hint}."
    return "Типичную ошибку распознать не удалось: down(silu(gate(x)) · up(x)) — проверь по шагам."


def test_ffn_shape(impl, ref):
    """SwiGLUFFN: [B, T, C] → [B, T, C]"""
    mine, _ = _make_pair(impl, ref)
    out = mine(_x())
    assert out.shape == (B, T, C), (
        f"Форма {tuple(out.shape)} вместо {(B, T, C)}: после поэлементного произведения веток "
        "[B, T, F] нужен down_proj обратно в C.")


def test_ffn_matches_reference(impl, ref):
    """SwiGLUFFN: численно совпадает с down(silu(gate(x)) · up(x))"""
    mine, reference = _make_pair(impl, ref)
    x = _x()
    with torch.no_grad():
        out, expected = mine(x), reference(x)
    if out.shape != expected.shape:
        pytest.fail(f"Форма {tuple(out.shape)} вместо {tuple(expected.shape)}.", pytrace=False)
    if not torch.allclose(out, expected, atol=ATOL, rtol=RTOL):
        pytest.fail(f"Выход не совпадает с эталоном (max |Δ| = {(out - expected).abs().max():.3g}). "
                    f"{_diagnose(out, x, reference)}", pytrace=False)


def test_up_branch_is_linear(impl, ref):
    """SwiGLUFFN: ветка up линейна — up_proj × 3 даёт выход × 3"""
    mine, _ = _make_pair(impl, ref)
    x = _x()
    with torch.no_grad():
        base = mine(x)
        mine.up_proj.weight.mul_(3.0)
        out = mine(x)
    assert_close(out, 3.0 * base, atol=1e-4, rtol=1e-4, msg=lambda _: (
        "Умножили веса up_proj на 3 — выход должен ровно утроиться, потому что ветка up входит "
        "линейно (нелинейность только у gate). Не утроился → SiLU стоит не на той ветке или "
        "после произведения."))


def test_gate_closed_gives_zero(impl, ref):
    """SwiGLUFFN: если gate ≡ 0, то SiLU(0) = 0 и выход ≡ 0"""
    mine, _ = _make_pair(impl, ref)
    with torch.no_grad():
        mine.gate_proj.weight.zero_()
        out = mine(_x())
    assert_close(out, torch.zeros_like(out), atol=ATOL, rtol=0, msg=lambda _: (
        "При нулевых весах gate_proj выход должен быть 0: SiLU(0) = 0 «закрывает» все каналы. "
        "Ненулевой выход → ветки складываются или гейт применён не к gate_proj."))


def test_no_bias_and_param_count(impl, ref):
    """SwiGLUFFN: три матрицы без bias, 3·C·F параметров"""
    mine, _ = _make_pair(impl, ref)
    n = sum(p.numel() for p in mine.parameters())
    assert n == 3 * C * HID, f"Параметров {n}, ожидалось 3·C·F = {3 * C * HID}. Не меняй конструктор."
    with torch.no_grad():
        out = mine(torch.zeros(1, 2, C))
    assert_close(out, torch.zeros_like(out), atol=ATOL, rtol=0, msg=lambda _: (
        "На нулевом входе выход должен быть 0 (bias нет). Ненулевой → где-то добавляется константа."))


def test_tokens_independent(impl, ref):
    """SwiGLUFFN: FFN применяется к каждому токену отдельно"""
    mine, _ = _make_pair(impl, ref)
    x = _x()
    with torch.no_grad():
        base = mine(x)
        x2 = x.clone()
        x2[:, 5] += 4.0
        out = mine(x2)
    keep = torch.ones(T, dtype=torch.bool)
    keep[5] = False
    assert_close(out[:, keep], base[:, keep], atol=ATOL, rtol=RTOL, msg=lambda _: (
        "Изменили только токен 5, а изменились и другие позиции: FFN должен работать "
        "поточечно, без смешивания по T."))


def test_gradients_flow(impl, ref):
    """SwiGLUFFN: градиент доходит до gate_proj, up_proj, down_proj и входа"""
    mine, _ = _make_pair(impl, ref)
    x = _x().requires_grad_(True)
    mine(x).pow(2).sum().backward()
    for name, p in mine.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (
            f"Параметр {name} не получил градиента: он не участвует в вычислении.")
    assert x.grad is not None and torch.isfinite(x.grad).all(), "Градиент по входу не дошёл."


def test_matches_reference_llama_sized(impl, ref):
    """SwiGLUFFN: с F = llama_ffn_hidden_dim(64, 32) = 192 совпадает с эталоном"""
    hid = impl.llama_ffn_hidden_dim(64, multiple_of=32)
    assert hid == 192, f"llama_ffn_hidden_dim(64, 32) = {hid}, ожидалось 192."
    torch.manual_seed(3)
    reference = ref.SwiGLUFFN(64, hid)
    mine = impl.SwiGLUFFN(64, hid)
    mine.load_state_dict(reference.state_dict())
    x = _x(shape=(1, 4, 64))
    with torch.no_grad():
        out, expected = mine(x), reference(x)
    if not torch.allclose(out, expected, atol=ATOL, rtol=RTOL):
        pytest.fail(f"Выход не совпадает с эталоном. {_diagnose(out, x, reference)}", pytrace=False)
