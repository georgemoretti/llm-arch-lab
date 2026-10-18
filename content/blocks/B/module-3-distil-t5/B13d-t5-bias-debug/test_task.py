"""Тесты «Отладка: относительный bias T5» — те же, что в B13.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул бакетов — поэлементная формула на Python-числах; если установлен transformers, ещё и
T5Attention._relative_position_bucket / compute_bias.
"""

import math

import pytest
import torch
import torch.nn as nn

REL = torch.arange(-300, 301)


def _bucket_py(r: int, bidirectional: bool, num_buckets: int, max_distance: int) -> int:
    """Та же формула по одному числу: половина бакетов — точные расстояния, половина — логарифмические."""
    base = 0
    if bidirectional:
        num_buckets //= 2
        base = num_buckets if r > 0 else 0
        n = abs(r)
    else:
        n = max(-r, 0)
    exact = num_buckets // 2
    if n < exact:
        return base + n
    large = exact + int(math.log(n / exact) / math.log(max_distance / exact) * (num_buckets - exact))
    return base + min(large, num_buckets - 1)


def _diagnose_bucket(got: torch.Tensor, bidirectional: bool, nb: int, md: int) -> str:
    want = torch.tensor([_bucket_py(r, bidirectional, nb, md) for r in REL.tolist()])
    flipped = torch.tensor([_bucket_py(-r, bidirectional, nb, md) for r in REL.tolist()])
    if torch.equal(got, flipped):
        return ("бакеты отражены: ключ СЛЕВА (relative_position < 0) должен попадать в первую половину, справа "
                "(> 0) — во вторую; в causal-режиме расстояние назад n = max(−relative_position, 0).")
    bad = (got != want).nonzero().flatten()
    i = int(bad[0])
    r = int(REL[i])
    hint = ""
    if bidirectional and r > 0 and int(got[i]) == int(want[i]) - nb // 2:
        hint = " Ключам справа нужно прибавить смещение второй половины (num_buckets // 2)."
    elif abs(r) >= (nb // (2 if bidirectional else 1)) // 2 and abs(int(got[i]) - int(want[i])) == 1:
        hint = " Похоже на округление: логарифмическая часть округляется ВНИЗ (.long()), а не round."
    elif int(got[i]) > int(want[i]):
        hint = " Номер бакета не должен превышать num_buckets − 1 своей половины (clamp)."
    return (f"{int(bad.numel())} из {REL.numel()} относительных позиций в неверном бакете; первая: "
            f"relative_position = {r} → {int(got[i])}, ожидался {int(want[i])} "
            f"(bidirectional={bidirectional}, num_buckets={nb}, max_distance={md}).{hint}")


@pytest.mark.parametrize("bidirectional", [True, False], ids=["bidirectional", "causal"])
def test_bucket_formula(impl, bidirectional):
    """Бакеты = формула T5 для расстояний −300 … 300 (32 бакета, max_distance = 128, как в t5-small)"""
    got = impl.relative_position_bucket(REL, bidirectional=bidirectional, num_buckets=32, max_distance=128)
    assert got.shape == REL.shape, f"форма выхода = форма входа {tuple(REL.shape)}, получено {tuple(got.shape)}"
    assert got.dtype == torch.long, f"номера бакетов — long (индексы для nn.Embedding), получено {got.dtype}"
    want = torch.tensor([_bucket_py(r, bidirectional, 32, 128) for r in REL.tolist()])
    if not torch.equal(got, want):
        pytest.fail(_diagnose_bucket(got, bidirectional, 32, 128), pytrace=False)


def test_bucket_examples(impl):
    """Ручные примеры: 0 → 0, слева −1 … −7 точно, справа +1 → 17; −20 → 10; дальше 90 — последний бакет"""
    f = lambda r, bi=True: int(impl.relative_position_bucket(torch.tensor([r]), bi, 32, 128)[0])
    assert f(0) == 0, "relative_position = 0 (ключ = запрос) → бакет 0"
    assert [f(-r) for r in range(1, 8)] == list(range(1, 8)), "ключи слева на 1 … 7 — каждый в своём бакете 1 … 7"
    assert f(1) == 17 and f(7) == 23, (
        f"ключи справа — во второй половине: +1 → 16 + 1 = 17, +7 → 23; получено {f(1)}, {f(7)}")
    assert f(-8) == f(-11) == 8, "расстояния 8 … 11 — уже логарифмический бакет 8 (бакеты становятся шире)"
    assert f(-20) == 10, f"−20 → 8 + ⌊log(20/8)/log(128/8)·8⌋ = 10, получено {f(-20)}"
    assert f(-91) == f(-1000) == 15 and f(1000) == 31, "дальние расстояния — в последнем бакете половины (15 и 31)"
    assert f(5, False) == 0 and f(-20, False) == 17, (
        "causal: будущие ключи → 0 (их всё равно закроет маска), назад на 20 → 16 + ⌊log(20/16)/log(8)·16⌋ = 17")


def test_bucket_properties(impl):
    """Свойства: бакет не убывает с расстоянием, близкие различаются точно, дальние сливаются"""
    for bi, half in [(True, 16), (False, 32)]:
        left = impl.relative_position_bucket(-torch.arange(0, 400), bi, 32, 128)
        assert bool((left.diff() >= 0).all()), "чем дальше ключ, тем больше (или тот же) номер бакета"
        assert int(left.max()) == half - 1, f"самый дальний бакет половины — {half - 1}, получено {int(left.max())}"
        exact = half // 2
        assert torch.equal(left[:exact], torch.arange(exact)), f"первые {exact} расстояний — каждое в своём бакете"
        assert int(left[200]) == int(left[399]) == half - 1, "за max_distance всё в одном бакете"
    right = impl.relative_position_bucket(torch.arange(1, 400), True, 32, 128)
    assert int(right.min()) >= 16, "двунаправленный режим: ключи справа только во второй половине (16 … 31)"


@pytest.mark.parametrize("cfg", [(32, 128), (16, 64), (8, 20)], ids=lambda c: f"nb{c[0]}-md{c[1]}")
def test_bucket_matches_transformers(impl, cfg):
    """Совпадение с transformers T5Attention._relative_position_bucket при разных num_buckets / max_distance"""
    t5 = pytest.importorskip("transformers.models.t5.modeling_t5", reason="нужен transformers: uv sync --extra weights")
    nb, md = cfg
    for bi in (True, False):
        want = t5.T5Attention._relative_position_bucket(REL[None, :], bidirectional=bi, num_buckets=nb, max_distance=md)
        got = impl.relative_position_bucket(REL[None, :], bidirectional=bi, num_buckets=nb, max_distance=md)
        if not torch.equal(got, want):
            pytest.fail(_diagnose_bucket(got[0], bi, nb, md), pytrace=False)


def _table(H=3, nb=8, seed=0):
    torch.manual_seed(seed)
    return nn.Embedding(nb, H)


def test_bias_shape_and_lookup(impl):
    """relative_position_bias: [1, H, T, S], в ячейке (i, j) — строка таблицы для бакета позиции j − i"""
    emb = _table()
    out = impl.relative_position_bias(emb, 5, 7, bidirectional=True, num_buckets=8, max_distance=16)
    assert out.shape == (1, 3, 5, 7), f"форма [1, H, T, S] = [1, 3, 5, 7], получено {tuple(out.shape)}"
    for i, j in [(0, 0), (4, 0), (0, 6), (2, 5), (3, 1)]:
        b = _bucket_py(j - i, True, 8, 16)
        torch.testing.assert_close(out[0, :, i, j], emb.weight[b], atol=0, rtol=0,
                                   msg=f"запрос {i}, ключ {j}: в out[0, :, {i}, {j}] ожидалась строка таблицы бакета {b}")


def test_bias_is_relative(impl):
    """Bias зависит только от j − i (матрица Тёплица) и различает «слева» и «справа»"""
    emb = _table(nb=32)
    out = impl.relative_position_bias(emb, 12, 12, bidirectional=True, num_buckets=32, max_distance=128)[0]
    for d in range(-11, 12):
        diag = torch.diagonal(out, offset=d, dim1=-2, dim2=-1)          # ячейки с j − i = d
        assert bool((diag == diag[:, :1]).all()), f"на диагонали j − i = {d} bias одинаков: важен сдвиг, а не позиция"
    left, right = out[:, 6, 3], out[:, 3, 6]                          # ключ на 3 левее и на 3 правее запроса
    assert not torch.equal(left, right), "ключ на 3 левее и на 3 правее — разные бакеты, разный bias"
    torch.testing.assert_close(left, emb.weight[3], atol=0, rtol=0,
                               msg="запрос 6, ключ 3 (на 3 левее) → ожидался бакет 3")


def test_bias_causal_and_offset(impl):
    """Causal: прошлые ключи различаются, будущие — бакет 0; offset = позиция запроса при генерации с кешем"""
    emb = _table(nb=32)
    full = impl.relative_position_bias(emb, 10, 10, bidirectional=False, num_buckets=32, max_distance=128)[0]
    row = full[:, 9]                                                   # последний запрос смотрит на ключи 0 … 9
    assert len({tuple(row[:, j].tolist()) for j in range(10)}) == 10, (
        "causal: для последнего запроса прошлые ключи 0 … 9 — на разных расстояниях, bias должен быть разный")
    torch.testing.assert_close(full[:, 2, 7], emb.weight[0], atol=0, rtol=0,
                               msg="causal: ключ в будущем (j > i) → бакет 0 (маска его всё равно закроет)")
    step = impl.relative_position_bias(emb, 1, 10, bidirectional=False, num_buckets=32, max_distance=128, offset=9)
    assert step.shape == (1, 3, 1, 10), f"один запрос и 10 ключей → [1, H, 1, 10], получено {tuple(step.shape)}"
    torch.testing.assert_close(step[0, :, 0], row, atol=0, rtol=0,
                               msg="offset = 9: единственный запрос стоит на позиции 9 — та же строка, что в полном bias")


def test_bias_matches_transformers(impl):
    """Совпадение с transformers T5Attention.compute_bias (энкодер — двунаправленный, декодер — causal)"""
    tr = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    from transformers.models.t5.modeling_t5 import T5Attention
    for is_decoder in (False, True):
        cfg = tr.T5Config(d_model=16, d_kv=4, num_heads=4, relative_attention_num_buckets=16,
                          relative_attention_max_distance=40, is_decoder=is_decoder)
        torch.manual_seed(0)
        att = T5Attention(cfg, has_relative_attention_bias=True, layer_idx=0)
        want = att.compute_bias(9, 13)
        got = impl.relative_position_bias(att.relative_attention_bias, 9, 13, bidirectional=not is_decoder,
                                          num_buckets=16, max_distance=40)
        torch.testing.assert_close(got, want, atol=0, rtol=0,
                                   msg=f"{'декодер (causal)' if is_decoder else 'энкодер'}: bias [1, H, T, S] = transformers")
