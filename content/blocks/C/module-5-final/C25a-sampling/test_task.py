"""Тесты C25a: сэмплирование следующего токена.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter).

Вероятностные свойства проверяются частотами на 4000 выборках с фиксированным torch.Generator: temperature
обостряет / сглаживает, top-k и top-p оставляют нужные токены и перенормируют вероятности внутри набора.
"""

import pytest
import torch


def _draws(impl, logits, n=4000, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    out = impl.sample_next(logits.expand(n, -1).clone(), generator=g, **kw)
    assert out.shape == (n, 1), f"sample_next возвращает [B, 1], получено {tuple(out.shape)}."
    assert out.dtype == torch.long, f"Токены — int64 (long), получено {out.dtype}."
    return torch.bincount(out[:, 0], minlength=logits.shape[-1]).float() / n


PROBS = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
LOGITS = PROBS.log() + 3.0                                  # логиты определены с точностью до сдвига


def test_sample_greedy_and_top_k(impl):
    """sample_next: temperature = 0 → argmax (без случайности); top_k = 1 → тоже argmax при любой температуре"""
    logits = torch.randn(5, 64, generator=torch.Generator().manual_seed(3))
    got = impl.sample_next(logits, temperature=0.0)
    assert got.shape == (5, 1), f"sample_next возвращает [B, 1], получено {tuple(got.shape)}."
    assert torch.equal(got[:, 0], logits.argmax(-1)), "temperature = 0 — жадный выбор: argmax по последней оси."
    freq = _draws(impl, LOGITS, temperature=1.5, top_k=1)
    assert freq[0] == 1.0, f"top_k = 1 оставляет только самый вероятный токен, а выпадали и другие: {freq.tolist()}."
    freq = _draws(impl, LOGITS, temperature=1.0, top_k=2)
    assert freq[2:].sum() == 0, f"top_k = 2: токены 2 и 3 вне двух лучших, но выпадали: {freq.tolist()}."
    assert abs(freq[0] - 0.625) < 0.03, (
        f"top_k = 2: после отсечения вероятности перенормируются: 0.5 / 0.8 = 0.625, получено {freq[0]:.3f}.")


def test_sample_temperature(impl):
    """sample_next: логиты делятся на T до softmax — T = 0.5 делает распределение острее, T = 2 — площе"""
    logits = torch.tensor([[2.0, 0.0]])
    sharp = _draws(impl, logits, temperature=0.5)[0].item()     # e⁴ / (e⁴ + 1) ≈ 0.982
    flat = _draws(impl, logits, temperature=2.0)[0].item()      # e / (e + 1) ≈ 0.731
    assert abs(sharp - 0.982) < 0.015 and abs(flat - 0.731) < 0.03, (
        f"P(токен 0) при T = 0.5: {sharp:.3f} (нужно ≈ 0.982), при T = 2: {flat:.3f} (нужно ≈ 0.731). "
        "softmax(logits / T): T < 1 обостряет, T > 1 сглаживает. Умножение вместо деления даёт обратный эффект.")


def test_sample_top_p(impl):
    """sample_next: top-p — наименьший набор самых вероятных токенов с суммой ≥ p; токен, перешедший порог, остаётся"""
    freq = _draws(impl, LOGITS, temperature=1.0, top_p=0.7)     # 0.5 < 0.7 ≤ 0.5 + 0.3 → ядро {0, 1}
    if freq[1] == 0:
        pytest.fail("top_p = 0.7: выпадал только токен 0. Ядро — наименьший набор с суммой ≥ p: 0.5 < 0.7, поэтому "
                    "токен 1 (накопленная сумма 0.8) тоже входит. Типичная ошибка — маска cumsum > p выкидывает "
                    "токен, который пересёк порог; сдвинь cumsum на один (масса токенов ДО текущего).", pytrace=False)
    if freq[2:].sum() > 0:
        pytest.fail(f"top_p = 0.7: выпадали токены 2–3 (частоты {freq.tolist()}), а ядро — {{0, 1}}: суммы 0.5 и 0.8.",
                    pytrace=False)
    assert abs(freq[0] - 0.625) < 0.03, f"Внутри ядра вероятности перенормируются: 0.5 / 0.8 = 0.625, получено {freq[0]:.3f}."
    freq = _draws(impl, LOGITS, temperature=1.0, top_p=0.4)
    assert freq[0] == 1.0, "top_p = 0.4 < 0.5: в ядре один самый вероятный токен (он остаётся всегда)."
    # порядок токенов в словаре не важен: ядро — по вероятностям, а не по индексам
    perm = torch.tensor([2, 0, 3, 1])
    freq = _draws(impl, LOGITS[:, perm], temperature=1.0, top_p=0.7)
    assert freq[0] == 0 and freq[2] == 0 and freq[1] > 0 and freq[3] > 0, (
        f"Логиты переставлены (вероятные токены — 1 и 3), частоты {freq.tolist()}. Сортируй по вероятности и "
        "возвращай маску на исходные места (scatter по индексам сортировки).")


def test_sample_reproducible(impl):
    """sample_next: один и тот же torch.Generator с тем же сидом → те же токены"""
    logits = torch.randn(8, 64, generator=torch.Generator().manual_seed(5))
    a = impl.sample_next(logits, temperature=1.0, top_p=0.9, generator=torch.Generator().manual_seed(7))
    b = impl.sample_next(logits, temperature=1.0, top_p=0.9, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b), "Передай generator в torch.multinomial — иначе генерацию не воспроизвести."


def test_sample_top_p_boundary(impl):
    """sample_next: если сумма ровно равна p, ядро на этом заканчивается — следующий токен уже лишний"""
    logits = torch.zeros(1, 4)                                  # 4 равновероятных токена, суммы 0.25, 0.5, … точные
    freq = _draws(impl, logits, temperature=1.0, top_p=0.5)
    kept = int((freq > 0).sum())
    assert kept == 2, (
        f"top_p = 0.5 при четырёх токенах по 0.25: выпадало {kept} разных токенов, а ядро — 2 (0.25 + 0.25 = 0.5 ≥ p). "
        "Третий токен выбрасывается: масса токенов выше него уже 0.5 ≥ p. Сравнивай «до текущего» с p через ≥, а не >.")


def test_sample_top_k_larger_than_vocab(impl):
    """sample_next: top_k больше словаря — ничего не отсекается и не падает"""
    freq = _draws(impl, LOGITS, temperature=1.0, top_k=10)
    err = (freq - PROBS[0]).abs().max().item()
    assert err < 0.03, (
        f"top_k = 10 при словаре из 4 токенов не должен ничего менять: частоты {freq.tolist()}, "
        f"вероятности {PROBS[0].tolist()}. Ограничь k размером словаря: min(top_k, V).")
