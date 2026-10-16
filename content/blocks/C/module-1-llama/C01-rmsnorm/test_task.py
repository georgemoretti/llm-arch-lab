"""Тесты C01: RMSNorm.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

B, T, C = 2, 8, 32
EPS = 1e-5
ATOL, RTOL = 1e-5, 1e-4


@pytest.fixture(autouse=True)
def _forbid_builtin_rmsnorm(monkeypatch):
    """Готовые RMSNorm из PyTorch в этой задаче запрещены."""

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Использована готовая RMSNorm (F.rms_norm / nn.RMSNorm). В этой задаче формулу "
            "нужно написать руками: x / sqrt(mean(x²) + eps) · weight.")

    monkeypatch.setattr(F, "rms_norm", forbidden)
    monkeypatch.setattr(nn.RMSNorm, "forward", forbidden)


def _x(scale: float = 1.0, shift: float = 0.0, shape=(B, T, C)) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.randn(*shape, generator=g) * scale + shift


def _weight() -> torch.Tensor:
    return torch.rand(C, generator=torch.Generator().manual_seed(2)) + 0.5


def _oracle(x, w, eps=EPS):
    """Независимая запись формулы (в float64, чтобы не зависеть от точности)."""
    h = x.double()
    return (h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w.double()).to(x.dtype)


def _diagnose(out: torch.Tensor, x: torch.Tensor, w: torch.Tensor, eps=EPS) -> str:
    """Сравнивает выход с типичными неправильными вариантами и называет совпавший."""
    if not torch.isfinite(out).all():
        return ("В выходе есть NaN/inf. Если вход нулевой — значит eps не добавлен под корень "
                "(0 / sqrt(0) = NaN). Если вход в fp16 — x² переполнился: считай в float32.")
    h = x.double()
    ms = h.pow(2).mean(-1, keepdim=True)
    variants = {
        "вычитается среднее — это LayerNorm, а RMSNorm не центрирует":
            lambda: (h - h.mean(-1, keepdim=True)) / torch.sqrt(h.var(-1, unbiased=False, keepdim=True) + eps) * w,
        "используется x.std() — он вычитает среднее и делит на (C−1), RMSNorm так не делает":
            lambda: h / (h.std(-1, keepdim=True) + eps) * w,
        "eps прибавлен ПОСЛЕ корня: x / (sqrt(mean(x²)) + eps). В Llama eps стоит под корнем":
            lambda: h / (torch.sqrt(ms) + eps) * w,
        "делишь на L2-норму, а не на RMS: забыто усреднение (mean) по C":
            lambda: h / torch.sqrt(h.pow(2).sum(-1, keepdim=True) + eps) * w,
        "забыт корень: делишь на mean(x²), а не на sqrt(mean(x²))":
            lambda: h / (ms + eps) * w,
        "не умножаешь на weight (γ)":
            lambda: h / torch.sqrt(ms + eps),
        "нормируешь не по той оси (среднее берётся не по последней оси C)":
            lambda: h / torch.sqrt(h.pow(2).mean(0, keepdim=True) + eps) * w,
        "нормируешь по оси T вместо C":
            lambda: h / torch.sqrt(h.pow(2).mean(-2, keepdim=True) + eps) * w,
    }
    for hint, fn in variants.items():
        try:
            candidate = fn().to(out.dtype)
        except RuntimeError:
            continue
        if candidate.shape == out.shape and torch.allclose(out, candidate, atol=1e-4, rtol=1e-3):
            return f"Похоже, {hint}."
    return "Типичную ошибку распознать не удалось: проверь по шагам x², mean по C, +eps, sqrt, ·weight."


def _check(out, x, w, what: str, eps=EPS, atol=ATOL, rtol=RTOL):
    expected = _oracle(x, w, eps)
    if out.shape != expected.shape:
        pytest.fail(f"{what}: форма {tuple(out.shape)} вместо {tuple(expected.shape)}. "
                    "RMSNorm не меняет форму — mean по C берётся с keepdim=True.", pytrace=False)
    if not torch.allclose(out.double(), expected.double(), atol=atol, rtol=rtol):
        err = (out.double() - expected.double()).abs().max().item()
        pytest.fail(f"{what}: выход не совпадает с формулой (max |Δ| = {err:.3g}). "
                    f"{_diagnose(out, x, w, eps)}", pytrace=False)


# ---------------------------------------------------------------------------
# rms_norm (функция)
# ---------------------------------------------------------------------------

def test_shape_and_dtype(impl):
    """rms_norm: форма и dtype выхода совпадают со входом"""
    x = _x()
    out = impl.rms_norm(x, torch.ones(C), EPS)
    assert out.shape == x.shape, (
        f"Ожидалась форма {tuple(x.shape)}, получено {tuple(out.shape)}. Среднее по C нужно "
        "брать с keepdim=True, чтобы делитель [..., 1] транслировался на [..., C].")
    assert out.dtype == x.dtype, f"dtype выхода {out.dtype}, а вход {x.dtype}: верни результат в dtype входа."


def test_matches_formula(impl):
    """rms_norm: численно совпадает с x / sqrt(mean(x²) + eps) · weight"""
    x, w = _x(), _weight()
    _check(impl.rms_norm(x, w, EPS), x, w, "Обычный вход")


def test_unit_rms(impl):
    """rms_norm: при weight = 1 RMS каждого вектора на выходе ≈ 1"""
    x = _x(scale=7.0)
    out = impl.rms_norm(x, torch.ones(C), EPS)
    rms = out.pow(2).mean(-1).sqrt()
    if not torch.allclose(rms, torch.ones_like(rms), atol=1e-3):
        pytest.fail(
            f"RMS выхода должен быть ≈ 1 у каждого токена, а он от {rms.min():.3f} до {rms.max():.3f}. "
            f"{_diagnose(out, x, torch.ones(C))}", pytrace=False)


def test_scale_invariance(impl):
    """rms_norm: RMSNorm(c·x) == RMSNorm(x) для c > 0"""
    x, w = _x(), _weight()
    base = impl.rms_norm(x, w, EPS)
    for c in (0.5, 4.0, 30.0):
        out = impl.rms_norm(c * x, w, EPS)
        if not torch.allclose(out, base, atol=1e-4, rtol=1e-4):
            pytest.fail(
                f"Умножили вход на {c} — выход изменился. RMSNorm делит вектор на его собственный "
                "RMS, поэтому масштаб входа должен сокращаться полностью. "
                f"{_diagnose(base, x, w)}", pytrace=False)


def test_no_centering(impl):
    """rms_norm: RMSNorm не вычитает среднее (в отличие от LayerNorm)"""
    x, w = _x(scale=0.5, shift=2.0), torch.ones(C)
    out = impl.rms_norm(x, w, EPS)
    _check(out, x, w, "Вход со сдвигом +2")
    assert out.mean(-1).abs().min() > 0.5, "Среднее выхода ≈ 0 — похоже, ты центрируешь вход, как LayerNorm."


def test_zero_input_is_finite(impl):
    """rms_norm: нулевой вектор → нули, без NaN (eps под корнем)"""
    x = torch.zeros(B, T, C)
    out = impl.rms_norm(x, _weight(), EPS)
    if not torch.isfinite(out).all():
        pytest.fail("На нулевом входе получились NaN/inf: 0 / sqrt(0) не определено. Именно для "
                    "этого eps прибавляют к mean(x²) под корнем.", pytrace=False)
    assert_close(out, torch.zeros_like(out), atol=ATOL, rtol=0,
                 msg=lambda m: "Нулевой вход должен давать нулевой выход.")


def test_small_activations(impl):
    """rms_norm: на маленьких активациях (|x| ~ 1e-3) eps влияет — и стоит под корнем"""
    x, w = _x(scale=1e-3), _weight()
    _check(impl.rms_norm(x, w, EPS), x, w,
           "Вход масштаба 1e-3 (mean(x²) ≈ 1e-6 — того же порядка, что eps = 1e-5)")


def test_custom_eps(impl):
    """rms_norm: eps берётся из аргумента, а не зашит константой"""
    x, w = _x(scale=0.05), _weight()
    _check(impl.rms_norm(x, w, 1e-2), x, w, "eps = 1e-2", eps=1e-2)


def test_fp16_large_values(impl):
    """rms_norm: fp16 с большими значениями — вычисление в float32, без переполнения"""
    x = _x(scale=300.0).half()  # 300² = 90 000 > 65 504 — максимума fp16
    w = _weight().half()
    out = impl.rms_norm(x, w, EPS)
    assert out.dtype == torch.float16, f"Для fp16-входа выход должен быть fp16, получено {out.dtype}."
    if not torch.isfinite(out).all() or out.abs().max() < 1e-2:
        pytest.fail("В fp16 x² переполнился (300² > 65 504 = максимум fp16) и выход испорчен. "
                    "Llama считает норму в float32: x.float() → нормировка → .to(x.dtype).",
                    pytrace=False)
    _check(out, x, w, "fp16-вход", atol=2e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# RMSNorm (модуль)
# ---------------------------------------------------------------------------

def test_module_fresh(impl):
    """RMSNorm: только параметр weight [C] = 1, и свежий модуль просто нормирует вход"""
    m = impl.RMSNorm(C, eps=EPS)
    params = dict(m.named_parameters())
    assert set(params) == {"weight"}, (
        f"Параметры модуля: {sorted(params)}. У RMSNorm только weight (γ) — сдвига β нет, "
        "это одно из отличий от LayerNorm. Не меняй конструктор.")
    x = _x(scale=2.0)
    with torch.no_grad():
        out = m(x)
    _check(out, x, torch.ones(C), "Свежий модуль (weight = 1)")


def test_module_matches_reference(impl, ref):
    """RMSNorm: модуль с обученным weight совпадает с эталоном"""
    reference = ref.RMSNorm(C, eps=EPS)
    with torch.no_grad():
        reference.weight.copy_(_weight())
    mine = impl.RMSNorm(C, eps=EPS)
    mine.load_state_dict(reference.state_dict())
    x = _x(scale=3.0)
    with torch.no_grad():
        out = mine(x)
    _check(out, x, reference.weight.detach(), "Модуль RMSNorm")


def test_tokens_independent(impl):
    """RMSNorm: каждый токен нормируется сам по себе (другие токены и батч не влияют)"""
    m = impl.RMSNorm(C, eps=EPS)
    x = _x()
    with torch.no_grad():
        base = m(x)
        x2 = x.clone()
        x2[:, 3] *= 10.0
        x2[1] += 5.0
        out = m(x2)
    keep = torch.ones(B, T, dtype=torch.bool)
    keep[:, 3] = False
    keep[1] = False
    assert_close(out[keep], base[keep], atol=ATOL, rtol=RTOL, msg=lambda _: (
        "Поменяли токен 3 и пример 1 батча — изменились выходы других токенов. Среднее x² "
        "нужно брать только по оси C (dim=-1), не по T и не по батчу."))


def test_gradients_flow(impl):
    """RMSNorm: градиент доходит до weight и до входа"""
    m = impl.RMSNorm(C, eps=EPS)
    x = _x().requires_grad_(True)
    (m(x) * torch.arange(C)).sum().backward()
    g = m.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, (
        "weight не получил градиента: он не участвует в вычислении (или стоит .detach()).")
    assert x.grad is not None and torch.isfinite(x.grad).all(), "Градиент по входу не дошёл или содержит NaN."
