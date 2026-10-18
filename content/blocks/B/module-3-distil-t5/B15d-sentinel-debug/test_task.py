"""Тесты «Отладка: sentinel-токены T5» — те же, что в B15.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул — разбор спанов циклом на Python и пример из рисунка 2 статьи T5.
"""

import itertools

import pytest
import torch

SENT = [99 - i for i in range(30)]          # как в T5: <extra_id_0> — самый большой id, дальше по убыванию
EOS = 1


def _spans(mask: list[bool]) -> list[tuple[bool, int, int]]:
    """[(шум?, начало, конец)] — подряд идущие куски маски."""
    out, start = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            out.append((mask[start], start, i))
            start = i
    return out


def test_random_segmentation(impl):
    """random_segmentation: num_segments непустых кусков, сумма = num_items; все разбиения равновероятны"""
    g = torch.Generator().manual_seed(0)
    seg = impl.random_segmentation(10, 3, generator=g)
    assert seg.shape == (3,), f"длины кусков [num_segments] = [3], получено {tuple(seg.shape)}"
    assert int(seg.sum()) == 10 and bool((seg >= 1).all()), f"куски непустые и в сумме 10, получено {seg.tolist()}"
    assert impl.random_segmentation(7, 1).tolist() == [7], "один кусок — весь отрезок"
    assert impl.random_segmentation(5, 5).tolist() == [1] * 5, "5 кусков из 5 элементов — по одному"
    a = impl.random_segmentation(20, 4, generator=torch.Generator().manual_seed(1))
    b = impl.random_segmentation(20, 4, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b), "с одинаковым generator — одинаковый результат (randperm(…, generator=generator))"
    seen, firsts = set(), []
    g = torch.Generator().manual_seed(2)
    for _ in range(1500):
        s = impl.random_segmentation(10, 3, generator=g).tolist()
        seen.add(tuple(s))
        firsts.append(s[0])
    assert len(seen) == 36, (
        f"разбиений 10 на 3 непустых куска — C(9, 2) = 36, за 1500 попыток встретилось {len(seen)}: разрезы нужно "
        "выбирать случайно и без повторов среди 9 щелей")
    mean_first = sum(firsts) / len(firsts)
    assert abs(mean_first - 10 / 3) < 0.25, (
        f"средняя длина первого куска ≈ 10/3 = 3.33 (все разбиения равновероятны), получено {mean_first:.2f}")


@pytest.mark.parametrize("length", [10, 20, 100, 512, 568])
def test_noise_mask_counts(impl, length):
    """Маска шума: ровно round(L·0.15) токенов шума в round(·/3) спанах; текст, шум, текст, …, шум"""
    mask = impl.random_spans_noise_mask(length, generator=torch.Generator().manual_seed(length))
    assert mask.shape == (length,) and mask.dtype == torch.bool, (
        f"маска bool [length] = [{length}], получено {tuple(mask.shape)}, {mask.dtype}")
    n_noise = min(max(round(length * 0.15), 1), length - 1)
    n_spans = max(round(n_noise / 3), 1)
    assert int(mask.sum()) == n_noise, f"L = {length}: шума round({length}·0.15) = {n_noise}, получено {int(mask.sum())}"
    spans = _spans(mask.tolist())
    noise_spans = [s for s in spans if s[0]]
    assert len(noise_spans) == n_spans, (
        f"L = {length}: спанов шума round({n_noise}/3) = {n_spans}, получено {len(noise_spans)}")
    assert not spans[0][0] and spans[-1][0], (
        "первым идёт спан текста, последним — спан шума: чередование текст₀, шум₀, …, текстₙ, шумₙ, как в T5")
    assert len(spans) == 2 * n_spans, "спаны текста и шума чередуются, их поровну"


def test_noise_mask_random(impl):
    """Маска случайна и воспроизводима; длины спанов разные, шум равномерно по тексту"""
    m1 = impl.random_spans_noise_mask(100, generator=torch.Generator().manual_seed(5))
    m2 = impl.random_spans_noise_mask(100, generator=torch.Generator().manual_seed(5))
    m3 = impl.random_spans_noise_mask(100, generator=torch.Generator().manual_seed(6))
    assert torch.equal(m1, m2), "одинаковый generator → одинаковая маска"
    assert not torch.equal(m1, m3), "разный generator → разные маски"
    g = torch.Generator().manual_seed(7)
    lens, freq = [], torch.zeros(100)
    for _ in range(300):
        m = impl.random_spans_noise_mask(100, generator=g)
        freq += m.float()
        lens += [e - s for noise, s, e in _spans(m.tolist()) if noise]
    assert len(set(lens)) >= 4, (
        f"длины спанов шума за 300 масок: {sorted(set(lens))} — должны быть разными (в среднем 3), а не все равными "
        "mean_noise_span_length")
    assert abs(sum(lens) / len(lens) - 3.0) < 0.01, f"средняя длина спана = 15 / 5 = 3, получено {sum(lens) / len(lens):.2f}"
    middle = freq[10:90] / 300
    assert float(middle.min()) > 0.05 and float(middle.max()) < 0.3, (
        f"доля шума на позициях 10 … 89 должна быть около 0.15 везде, получено от {float(middle.min()):.2f} до "
        f"{float(middle.max()):.2f}: шум должен быть разбросан по всему тексту")


WORDS = "Thank you for inviting me to your party last week .".split()


def test_paper_example(impl):
    """Пример из статьи T5: Thank you <X> me to your party <Y> week . / <X> for inviting <Y> last <Z>"""
    vocab = {w: i + 2 for i, w in enumerate(WORDS)}
    ids = torch.tensor([vocab[w] for w in WORDS])
    mask = torch.tensor([w in ("for", "inviting", "last") for w in WORDS])
    inputs, targets = impl.span_corruption(ids, mask, SENT, EOS)
    name = {v: k for k, v in vocab.items()} | {SENT[0]: "<X>", SENT[1]: "<Y>", SENT[2]: "<Z>", EOS: "</s>"}
    show = lambda t: " ".join(name.get(int(i), f"#{int(i)}") for i in t)
    want_in = "Thank you <X> me to your party <Y> week . </s>"
    want_tg = "<X> for inviting <Y> last <Z> </s>"
    assert show(inputs) == want_in, f"вход: ждём «{want_in}», получено «{show(inputs)}»"
    assert show(targets) == want_tg, (
        f"цель: ждём «{want_tg}», получено «{show(targets)}»")
    assert ids.tolist() == [vocab[w] for w in WORDS], "исходный тензор ids не меняется на месте"


def _reconstruct(inputs: list[int], targets: list[int]) -> list[int]:
    """Обратная операция: вместо каждого sentinel во входе — кусок цели после того же sentinel."""
    pieces, cur = {}, None
    for t in targets[:-1]:
        if t in SENT:
            cur = t
            pieces[cur] = []
        elif cur is not None:
            pieces[cur].append(t)
    out = []
    for t in inputs[:-1]:
        out += pieces.get(t, [t]) if t in SENT else [t]
    return out


def test_roundtrip_and_lengths(impl):
    """На случайных масках: из (вход, цель) восстанавливается исходный текст; длины = формула T5"""
    g = torch.Generator().manual_seed(11)
    for length in [12, 30, 64, 101]:
        ids = torch.randint(2, 60, (length,), generator=g)          # не пересекается с sentinel 70 … 99
        mask = impl.random_spans_noise_mask(length, generator=g)
        inputs, targets = impl.span_corruption(ids, mask, SENT, EOS)
        n_noise = int(mask.sum())
        spans = _spans(mask.tolist())
        n_noise_spans = sum(1 for s in spans if s[0])
        n_text_spans = len(spans) - n_noise_spans
        assert inputs.dtype == ids.dtype and targets.dtype == ids.dtype, "тип id сохраняется (long)"
        assert int(inputs[-1]) == EOS and int(targets[-1]) == EOS, "вход и цель заканчиваются EOS"
        assert inputs.numel() == length - n_noise + n_noise_spans + 1, (
            f"L = {length}: длина входа = L − шум + спаны шума + EOS = {length - n_noise + n_noise_spans + 1}, "
            f"получено {inputs.numel()}")
        assert targets.numel() == n_noise + n_text_spans + 1, (
            f"L = {length}: длина цели = шум + спаны текста + EOS = {n_noise + n_text_spans + 1}, получено "
            f"{targets.numel()}")
        used_in = [int(t) for t in inputs if int(t) in SENT]
        assert used_in == SENT[:n_noise_spans], (
            f"во входе sentinel-ы идут по порядку: {SENT[:n_noise_spans]}, получено {used_in}")
        rec = _reconstruct(inputs.tolist(), targets.tolist())
        assert rec == ids.tolist(), (
            f"L = {length}: подстановка кусков цели вместо sentinel-ов входа не вернула исходный текст")


def test_sentinel_limits(impl):
    """Арбитрарная маска: каждый спан шума — свой sentinel; если sentinel-ов не хватает — ValueError"""
    ids = torch.arange(10, 30)
    mask = torch.zeros(20, dtype=torch.bool)
    mask[[2, 3, 7, 11, 12, 13, 19]] = True
    out = impl.noise_span_to_unique_sentinel(ids, mask, [90, 91, 92, 93])
    assert out.tolist() == [10, 11, 90, 14, 15, 16, 91, 18, 19, 20, 92, 24, 25, 26, 27, 28, 93], (
        f"4 спана шума → 4 sentinel-а по порядку, остальные токены на месте; получено {out.tolist()}")
    with pytest.raises(ValueError):
        impl.noise_span_to_unique_sentinel(ids, mask, [90, 91, 92])
    combos = list(itertools.product([False, True], repeat=6))
    for c in combos:
        m = torch.tensor((False,) + c)
        o = impl.noise_span_to_unique_sentinel(torch.arange(7), m, SENT)
        assert o.numel() == 7 - int(m.sum()) + sum(1 for s in _spans(m.tolist()) if s[0]), (
            f"маска {m.int().tolist()}: каждый спан шума → ровно один токен")
