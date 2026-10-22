"""Тесты «Синусоидальные позиции».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул таблицы — формула §3.5 статьи, посчитанная в float64.
"""

import math

import pytest
import torch

N, DM = 50, 16


def _oracle(n_pos=N, d=DM, layout="interleave", expo="2i", base=10000.0):
    p = torch.arange(n_pos, dtype=torch.float64)[:, None]
    i2 = torch.arange(0, d, 2, dtype=torch.float64) if expo == "2i" else torch.arange(d // 2, dtype=torch.float64)
    ang = p / base ** (i2 / d)
    if layout == "halves":
        return torch.cat([ang.sin(), ang.cos()], dim=1)
    pe = torch.zeros(n_pos, d, dtype=torch.float64)
    if layout == "swap":
        pe[:, 0::2], pe[:, 1::2] = ang.cos(), ang.sin()
    else:
        pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
    return pe


def _diagnose_table(pe) -> str:
    pe = pe.double()
    close = lambda o: pe.shape == o.shape and torch.allclose(pe, o, atol=1e-4)
    if close(_oracle(layout="halves")):
        return ("sin и cos лежат двумя половинами [sin … sin, cos … cos]. В статье они чередуются: столбец 2i — sin, "
                "2i + 1 — cos той же частоты (pe[:, 0::2] и pe[:, 1::2]). Обе раскладки работают, но тесты и B03–B06 "
                "ждут раскладку статьи.")
    if close(_oracle(layout="swap")):
        return "sin и cos перепутаны: в чётных столбцах (2i) — sin, в нечётных (2i + 1) — cos."
    if close(_oracle(expo="i")):
        return ("показатель частоты i/d вместо 2i/d: частоты убывают слишком медленно, самая низкая — 10000^(−1/2) "
                "вместо ≈ 10000^(−1). Пары нумеруются через 2i = 0, 2, 4, … (arange(0, d_model, 2)).")
    if pe.shape == (N, DM) and torch.allclose(pe[0], _oracle()[0].double(), atol=1e-6):
        return "позиция 0 верная, дальше нет: проверь частоты ω_i = 10000^(−2i/d_model) и углы p·ω_i."
    return "таблица не совпадает с формулой: PE[p, 2i] = sin(p / 10000^(2i/d)), PE[p, 2i+1] = cos(того же угла)."


def test_table_values(impl):
    """sinusoidal_positions: форма, float32, совпадение с формулой статьи"""
    pe = impl.sinusoidal_positions(N, DM)
    assert pe.shape == (N, DM), f"таблица [n_pos, d_model] = [{N}, {DM}], получено {tuple(pe.shape)}"
    assert pe.dtype == torch.float32, f"float32, получено {pe.dtype}"
    if not torch.allclose(pe.double(), _oracle(), atol=1e-5):
        pytest.fail(_diagnose_table(pe), pytrace=False)
    torch.testing.assert_close(pe[0, 0::2], torch.zeros(DM // 2), msg="позиция 0: все sin = 0", atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(pe[0, 1::2], torch.ones(DM // 2), msg="позиция 0: все cos = 1", atol=1e-6, rtol=1e-5)


def test_table_properties(impl):
    """Свойства: |PE| ≤ 1, ‖PE[p]‖² = d/2, скалярное произведение PE[p]·PE[p + k] зависит только от k"""
    pe = impl.sinusoidal_positions(64, 32).double()
    assert pe.abs().max() <= 1 + 1e-6, "значения синусов и косинусов лежат в [−1, 1]"
    torch.testing.assert_close(pe.pow(2).sum(-1), torch.full((64,), 16.0, dtype=torch.float64), atol=1e-4, rtol=0,
                               msg="у каждой позиции ‖PE‖² = d/2: в каждой паре sin² + cos² = 1")
    for k in (1, 4, 9):
        dots = (pe[:-k] * pe[k:]).sum(-1)
        if dots.max() - dots.min() > 1e-4:
            pytest.fail(f"PE[p]·PE[p + {k}] меняется с p (от {dots.min():.3f} до {dots.max():.3f}), а должно зависеть "
                        "только от сдвига k: Σ_i cos(k·ω_i). Проверь, что sin и cos одной частоты стоят в паре.",
                        pytrace=False)


def test_shift_matrix(impl):
    """shift_matrix: PE[p + k] = M @ PE[p] при любом p; M ортогональна; M(a)·M(b) = M(a + b)"""
    pe = _oracle(60, DM).float()   # таблица из оракула: тест проверяет только M
    for k in (1, 5, 12):
        m = impl.shift_matrix(k, DM)
        assert m.shape == (DM, DM), f"M — [d_model, d_model] = [{DM}, {DM}], получено {tuple(m.shape)}"
        got = pe[:-k] @ m.T                         # строки: M @ PE[p]
        if torch.allclose(pe[:-k] @ m, pe[k:], atol=1e-4) and not torch.allclose(got, pe[k:], atol=1e-4):
            pytest.fail("M транспонирована: она сдвигает на −k. Ждём PE[p + k] = M @ PE[p] (вектор-столбец справа).",
                        pytrace=False)
        if not torch.allclose(got, pe[k:], atol=1e-4):
            pytest.fail(f"M @ PE[p] ≠ PE[p + {k}]. Для пары (sin a, cos a) и b = k·ω_i: sin(a + b) = sin a·cos b + "
                        "cos a·sin b, cos(a + b) = cos a·cos b − sin a·sin b — это блок 2×2 на диагонали.",
                        pytrace=False)
        torch.testing.assert_close(m @ m.T, torch.eye(DM), atol=1e-5, rtol=0,
                                   msg="M — поворот, значит ортогональна: M Mᵀ = I")
    torch.testing.assert_close(impl.shift_matrix(0, DM), torch.eye(DM), atol=1e-6, rtol=0, msg="сдвиг на 0 — единичная")
    torch.testing.assert_close(impl.shift_matrix(3, DM) @ impl.shift_matrix(4, DM), impl.shift_matrix(7, DM),
                               atol=1e-5, rtol=0, msg="сдвиг на 3, затем на 4 = сдвиг на 7")


def _emb(impl):
    torch.manual_seed(0)
    return impl.TransformerEmbedding(20, DM, 12, pad_id=0)


def test_embedding_forward(impl):
    """TransformerEmbedding: tok(ids)·√d_model + PE[:T]; ValueError при T > max_len"""
    emb = _emb(impl)
    ids = torch.tensor([[5, 6, 7, 0, 0], [1, 2, 3, 4, 5]])
    out = emb(ids)
    assert out.shape == (2, 5, DM), f"выход [B, T, d_model] = [2, 5, {DM}], получено {tuple(out.shape)}"
    pe = _oracle(12, DM).float()
    want = emb.tok(ids) * math.sqrt(DM) + pe[:5]
    if torch.allclose(out, emb.tok(ids) + pe[:5], atol=1e-5):
        pytest.fail("эмбеддинги токенов не умножены на √d_model. Матрица инициализирована с std = 1/√d (она общая "
                    "с выходным слоем), и без множителя позиции заглушили бы токены.", pytrace=False)
    torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-5, msg="tok(ids) · √d_model + PE[0 … T−1]")
    with pytest.raises(ValueError):
        emb(torch.ones(1, 13, dtype=torch.long))
    assert not any("pe" in n for n, _ in emb.named_parameters()), (
        "позиции — не параметр: синусоиды не обучаются (buffer, а не nn.Parameter)")


def test_embedding_positions(impl):
    """Позиции считаются с 0 в каждом тексте; один токен в разных позициях — разные векторы; PAD = только позиция"""
    emb = _emb(impl)
    ids = torch.tensor([[7, 3, 3, 7], [7, 3, 0, 0]])
    out = emb(ids)
    torch.testing.assert_close(out[0, :2], out[1, :2], msg="одинаковое начало текстов → одинаковые векторы: позиции "
                                                           "у каждого текста с нуля", atol=1e-6, rtol=1e-5)
    assert not torch.allclose(out[0, 0], out[0, 3], atol=1e-3), "токен 7 в позициях 0 и 3 должен давать разные векторы"
    torch.testing.assert_close(out[1, 2:], _oracle(12, DM).float()[2:4], atol=1e-5, rtol=1e-5,
                               msg="эмбеддинг PAD нулевой (padding_idx) → на PAD-позиции остаётся только PE")
