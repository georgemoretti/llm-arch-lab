"""Тесты «Найди баг: RoPE»: те же проверки, что в «RoPE» (RoPE); подсказка про конкретный баг убрана.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Независимый оракул — поворот через комплексные числа: пара (x_j, x_{j+D/2}) — это число
x_j + i·x_{j+D/2}, и RoPE умножает его на e^{i·m·θ_j}.
"""

import math

import pytest
import torch
from torch.testing import assert_close

B, H, T, D = 2, 3, 12, 16
ATOL, RTOL = 1e-5, 1e-4


def _x(t: int = T, d: int = D, seed: int = 1):
    return torch.randn(B, H, t, d, generator=torch.Generator().manual_seed(seed))


def _oracle_inv_freq(d, theta):
    return torch.tensor([theta ** (-2 * i / d) for i in range(d // 2)], dtype=torch.float32)


def _oracle_cos_sin(positions, d, theta):
    angles = positions.double()[:, None] * _oracle_inv_freq(d, theta).double()[None]
    emb = torch.cat((angles, angles), -1)
    return emb.cos().float(), emb.sin().float()


def _oracle_rope(x, positions, theta):
    d = x.shape[-1]
    z = torch.complex(x[..., : d // 2].double(), x[..., d // 2:].double())
    angles = positions.double()[:, None] * _oracle_inv_freq(d, theta).double()[None]
    z = z * torch.polar(torch.ones_like(angles), angles)
    return torch.cat((z.real, z.imag), -1).float()


def _rotate_variants(x):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    ev, od = x[..., 0::2], x[..., 1::2]
    return {
        "знак перепутан: вернул [x2 | −x1] вместо [−x2 | x1] (это поворот в обратную сторону)":
            torch.cat((x2, -x1), -1),
        "половины переставлены без смены знака: [x2 | x1]": torch.cat((x2, x1), -1),
        "использована «интерливная» раскладка (пары 0-1, 2-3, …), а в задаче пары — (j, j + D/2)":
            torch.stack((-od, ev), -1).flatten(-2),
    }


def _diagnose_apply(out, x, cos, sin) -> str:
    if out.shape != x.shape:
        return f"Форма {tuple(out.shape)} вместо {tuple(x.shape)}: RoPE не меняет форму."
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    rh = torch.cat((-x2, x1), -1)
    variants = {
        "поворот в обратную сторону: x·cos − rotate_half(x)·sin (или неверный знак в rotate_half)":
            x * cos - rh * sin,
        "cos и sin перепутаны местами": x * sin + rh * cos,
        "забыт x·cos — остался только rotate_half(x)·sin": rh * sin,
        "вместо rotate_half(x) умножаешь на sin сам x": x * cos + x * sin,
    }
    for name, cand in _rotate_variants(x).items():
        variants[f"rotate_half: {name}"] = x * cos + cand * sin
    for hint, cand in variants.items():
        if torch.allclose(out, cand, atol=1e-4, rtol=1e-3):
            return f"Похоже, {hint}."
    return "Типичную ошибку распознать не удалось: x·cos + rotate_half(x)·sin — проверь по шагам."


# ---------------------------------------------------------------------------
# rope_inv_freq, rope_cos_sin
# ---------------------------------------------------------------------------

def test_inv_freq(impl):
    """rope_inv_freq: [D/2] частот θ^(−2i/D), от 1 до самой медленной"""
    for d, theta in [(16, 10000.0), (8, 500000.0)]:
        got = impl.rope_inv_freq(d, theta)
        expected = _oracle_inv_freq(d, theta)
        if got.shape != expected.shape:
            hint = " Похоже, взят arange(D) с шагом 1: частот должно быть D/2 — по одной на пару." \
                if got.numel() == d else ""
            pytest.fail(f"D={d}: форма {tuple(got.shape)} вместо {tuple(expected.shape)}.{hint}", pytrace=False)
        if not torch.allclose(got.float(), expected, rtol=1e-5, atol=0):
            hint = "Проверь формулу θ^(−2i/D)."
            if torch.allclose(got.float(), expected.flip(0), rtol=1e-5):
                hint = "Порядок перевёрнут: пара 0 должна вращаться быстрее всех (частота 1)."
            elif torch.allclose(got.float(), 1 / expected, rtol=1e-5):
                hint = "Пропущен минус в показателе: частоты должны убывать от 1, а не расти."
            elif torch.allclose(got.float(), torch.tensor([theta ** (-i / d) for i in range(d // 2)]), rtol=1e-5):
                hint = "Показатель i/D вместо 2i/D."
            pytest.fail(f"D={d}, θ={theta:g}: частоты не совпадают: {got[:4].tolist()}… "
                        f"вместо {expected[:4].tolist()}…. {hint}", pytrace=False)


def test_cos_sin_shape_and_layout(impl):
    """rope_cos_sin: формы [T, D], позиция 0 → cos = 1, sin = 0, половины одинаковые"""
    pos = torch.arange(T)
    cos, sin = impl.rope_cos_sin(pos, D, 10000.0)
    assert cos.shape == (T, D) and sin.shape == (T, D), (
        f"Формы cos/sin: {tuple(cos.shape)}, {tuple(sin.shape)}; нужны [T, D] = {(T, D)}. "
        "Углы [T, D/2] нужно продублировать до D.")
    assert_close(cos[0], torch.ones(D), atol=ATOL, rtol=0, msg=lambda _: "На позиции 0 угол 0: cos должен быть 1.")
    assert_close(sin[0], torch.zeros(D), atol=ATOL, rtol=0, msg=lambda _: "На позиции 0 угол 0: sin должен быть 0.")
    if not torch.allclose(cos[:, : D // 2], cos[:, D // 2:], atol=ATOL):
        pytest.fail("Координаты j и j + D/2 — это одна пара, им нужен один и тот же угол: "
                    "cos[:, :D/2] должен совпадать с cos[:, D/2:]. Напечатай cos[1] и посмотри, "
                    "какие координаты получили одинаковые значения.", pytrace=False)


def test_cos_sin_values(impl):
    """rope_cos_sin: угол позиции m для пары i равен m·θ^(−2i/D), позиции — произвольные"""
    pos = torch.tensor([0, 1, 2, 7, 100, 4095])
    cos, sin = impl.rope_cos_sin(pos, D, 500000.0)
    ecos, esin = _oracle_cos_sin(pos, D, 500000.0)
    assert_close(cos, ecos, atol=1e-4, rtol=1e-4, msg=lambda _: (
        "cos не совпадает с cos(m·θ_i). Проверь, что углы — внешнее произведение позиций [T] на "
        "частоты [D/2] (positions[:, None] * inv_freq[None, :]), и что берутся именно переданные "
        "positions, а не arange(T)."))
    assert_close(sin, esin, atol=1e-4, rtol=1e-4, msg=lambda _: "sin не совпадает с sin(m·θ_i).")


# ---------------------------------------------------------------------------
# rotate_half, apply_rope (на эталонных cos/sin — чтобы ошибки не смешивались)
# ---------------------------------------------------------------------------

def test_rotate_half(impl):
    """rotate_half: [x1 | x2] → [−x2 | x1]"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = impl.rotate_half(x)
    expected = torch.tensor([-4.0, -5.0, -6.0, 1.0, 2.0, 3.0])
    if out.shape != x.shape or not torch.allclose(out, expected):
        hint = ""
        for name, cand in _rotate_variants(x).items():
            if out.shape == cand.shape and torch.allclose(out, cand):
                hint = f" Похоже, {name}."
        pytest.fail(f"rotate_half([1, 2, 3, 4, 5, 6]) = {out.tolist()}, ожидалось {expected.tolist()}.{hint}",
                    pytrace=False)
    batched = _x()
    assert_close(impl.rotate_half(batched)[0, 1, 2], impl.rotate_half(batched[0, 1, 2]),
                 atol=0, rtol=0, msg=lambda _: "rotate_half должна работать по последней оси при любом числе ведущих осей.")


def test_apply_position_zero_is_identity(impl):
    """apply_rope: на позиции 0 вектор не меняется"""
    x = _x(t=1)
    cos, sin = _oracle_cos_sin(torch.zeros(1, dtype=torch.long), D, 10000.0)
    out = impl.apply_rope(x, cos, sin)
    assert_close(out, x, atol=ATOL, rtol=RTOL, msg=lambda _: (
        f"На позиции 0 угол поворота 0 — вектор должен остаться прежним. {_diagnose_apply(out, x, cos, sin)}"))


def test_apply_2d_rotation(impl):
    """apply_rope: при D = 2 это поворот плоскости на угол m (радиан)"""
    pos = torch.arange(6)
    x = torch.tensor([1.0, 0.0]).expand(1, 1, 6, 2).clone()
    cos, sin = _oracle_cos_sin(pos, 2, 10000.0)  # при D = 2 единственная частота = 1
    out = impl.apply_rope(x, cos, sin)[0, 0]
    expected = torch.stack((pos.float().cos(), pos.float().sin()), -1)
    assert_close(out, expected, atol=1e-5, rtol=1e-4, msg=lambda _: (
        "Вектор (1, 0) на позиции m должен повернуться против часовой стрелки на m радиан: "
        f"(cos m, sin m). Получено для m = 1: {out[1].tolist()}. {_diagnose_apply(out[None, None], x, cos, sin)}"))


def test_apply_preserves_norm(impl):
    """apply_rope: поворот сохраняет норму каждого вектора q/k"""
    x = _x()
    cos, sin = _oracle_cos_sin(torch.arange(T), D, 10000.0)
    out = impl.apply_rope(x, cos, sin)
    if not torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=1e-4):
        pytest.fail("Норма векторов изменилась после RoPE, а поворот её сохраняет. Значит, это не "
                    f"поворот пар координат. {_diagnose_apply(out, x, cos, sin)}", pytrace=False)


def test_apply_matches_oracle(impl):
    """apply_rope: совпадает с поворотом пар (x_j, x_{j+D/2}) на угол m·θ_j"""
    pos = torch.arange(3, 3 + T)
    x = _x()
    cos, sin = _oracle_cos_sin(pos, D, 10000.0)
    out = impl.apply_rope(x, cos, sin)
    expected = _oracle_rope(x, pos, 10000.0)
    if out.shape != expected.shape or not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        pytest.fail(f"Выход не совпадает с оракулом. {_diagnose_apply(out, x, cos, sin)}", pytrace=False)


# ---------------------------------------------------------------------------
# Всё вместе: свойства RoPE
# ---------------------------------------------------------------------------

def _rope(impl, x, positions, theta=10000.0):
    cos, sin = impl.rope_cos_sin(positions, x.shape[-1], theta)
    return impl.apply_rope(x, cos, sin)


def test_relative_position(impl):
    """RoPE: score ⟨q_m, k_n⟩ зависит только от m − n"""
    g = torch.Generator().manual_seed(5)
    q = torch.randn(1, 1, 1, D, generator=g).expand(1, 1, 32, D).contiguous()
    k = torch.randn(1, 1, 1, D, generator=g).expand(1, 1, 32, D).contiguous()
    pos = torch.arange(32)
    rq, rk = _rope(impl, q, pos)[0, 0], _rope(impl, k, pos)[0, 0]
    scores = rq @ rk.T                                  # scores[m, n] = ⟨R_m q, R_n k⟩
    for delta in (0, 1, 3, 10):
        diag = torch.diagonal(scores, offset=-delta)    # все пары с m − n = delta
        if (diag.max() - diag.min()).abs() > 1e-3 * (1 + diag.abs().max()):
            pytest.fail(
                f"Пары позиций с одинаковым m − n = {delta} дали разные scores (от {diag.min():.4f} до "
                f"{diag.max():.4f}). У RoPE абсолютные позиции сокращаются: ⟨R_m q, R_n k⟩ = "
                "⟨q, R_{n−m} k⟩. Если это не так — повороты для q и k не согласованы или это не "
                "поворот (сломана пара координат / разные углы у половин).", pytrace=False)
    assert abs(scores[0, 0].item() - scores[5, 0].item()) > 1e-4, (
        "Scores для разных m − n совпали — похоже, позиции вообще не влияют на результат.")


def test_full_pipeline_llama3_theta(impl):
    """RoPE: rope_cos_sin + apply_rope с θ = 500 000 и смещёнными позициями совпадают с оракулом"""
    pos = torch.arange(20, 20 + T)  # как при генерации с KV-cache: новые токены не с нуля
    x = _x(d=32, seed=3)
    out = _rope(impl, x, pos, theta=500000.0)
    expected = _oracle_rope(x, pos, 500000.0)
    if out.shape != expected.shape or not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        pytest.fail("С θ = 500 000 и позициями 20..31 результат не совпал с оракулом. Если отдельные "
                    "тесты rope_cos_sin и apply_rope проходят, проверь, что theta и positions "
                    "действительно передаются внутрь, а не заменяются значениями по умолчанию.",
                    pytrace=False)


def test_theta_matters(impl):
    """RoPE: больший θ — медленнее вращение дальних пар (разные θ дают разный результат)"""
    pos = torch.arange(T)
    x = _x()
    a, b = _rope(impl, x, pos, 10000.0), _rope(impl, x, pos, 500000.0)
    assert not torch.allclose(a, b, atol=1e-4), "θ = 10 000 и θ = 500 000 дали одинаковый результат — theta игнорируется."
    # первая пара вращается с частотой 1 при любом θ
    assert_close(a[..., 0], b[..., 0], atol=1e-4, rtol=1e-4, msg=lambda _: (
        "Пара 0 должна вращаться с частотой θ⁰ = 1 независимо от θ."))
