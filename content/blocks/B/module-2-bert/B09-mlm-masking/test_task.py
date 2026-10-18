"""Тесты «MLM-маскирование».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Маскирование случайное, поэтому тесты проверяют инварианты и доли на большом батче (≈ 60 тысяч токенов).
"""

import pytest
import torch

PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4
SPECIAL = [PAD, UNK, CLS, SEP, MASK]
V = 1000


def _batch(B=256, T=256, seed=0):
    """Тексты разной длины: [CLS] токены [SEP] PAD…"""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, V, (B, T), generator=g)
    lengths = torch.randint(T // 2, T, (B,), generator=g)
    for i, n in enumerate(lengths.tolist()):
        ids[i, 0], ids[i, n - 1], ids[i, n:] = CLS, SEP, PAD
    return ids


def _run(impl, ids, prob=0.15, word_ids=None, seed=1):
    before = ids.clone()
    out = impl.mlm_mask(ids, SPECIAL, MASK, V, prob=prob, word_ids=word_ids,
                        generator=torch.Generator().manual_seed(seed))
    assert isinstance(out, tuple) and len(out) == 2, "mlm_mask возвращает (inputs, labels)"
    assert torch.equal(ids, before), "mlm_mask испортил входной тензор: работай с копией (masked_fill, where)"
    return out


def test_labels_and_untouched(impl):
    """labels: исходный токен у выбранных, −100 у остальных; невыбранные токены входа не меняются"""
    ids = _batch()
    inputs, labels = _run(impl, ids)
    assert inputs.shape == ids.shape and labels.shape == ids.shape, "формы inputs и labels — как у входа"
    assert labels.dtype == torch.long and inputs.dtype == torch.long, "inputs и labels — long"
    chosen = labels != -100
    assert chosen.any(), "ни один токен не выбран"
    assert torch.equal(labels[chosen], ids[chosen]), (
        "в labels у выбранных позиций должен стоять ИСХОДНЫЙ токен (его модель и восстанавливает), а не [MASK]")
    assert torch.equal(inputs[~chosen], ids[~chosen]), (
        "невыбранные токены входа меняться не должны: лосс по ним не считается, и модель должна видеть их как есть")


def test_specials_never_chosen(impl):
    """[CLS], [SEP] и PAD никогда не выбираются и не меняются"""
    ids = _batch()
    inputs, labels = _run(impl, ids)
    special = torch.isin(ids, torch.tensor(SPECIAL))
    n_bad = int((labels[special] != -100).sum())
    if n_bad:
        kinds = sorted({ {PAD: "[PAD]", CLS: "[CLS]", SEP: "[SEP]"}.get(int(t), str(int(t)))
                        for t in ids[special & (labels != -100)][:50]})
        pytest.fail(f"выбрано {n_bad} спецтокенов ({', '.join(kinds)}). Их не нужно предсказывать: [CLS] и [SEP] стоят "
                    "на известных местах, а PAD — вообще не текст. Лосс по ним учит модель ерунде, а PAD-позиции ещё и "
                    "закрыты маской внимания.", pytrace=False)
    assert torch.equal(inputs[special], ids[special]), "спецтокены во входе не меняются"


def test_rates_80_10_10(impl):
    """≈ 15 % обычных токенов выбрано; из них ≈ 80 % [MASK], ≈ 10 % случайный токен, ≈ 10 % без изменений"""
    ids = _batch()
    inputs, labels = _run(impl, ids)
    normal = ~torch.isin(ids, torch.tensor(SPECIAL))
    chosen = labels != -100
    rate = chosen[normal].float().mean().item()
    assert abs(rate - 0.15) < 0.01, f"выбрано {rate:.1%} обычных токенов, ждём 15 % ± 1 %"
    n = chosen.sum().item()
    masked = (inputs[chosen] == MASK).sum().item() / n
    kept = (inputs[chosen] == ids[chosen]).sum().item() / n
    rand = 1 - masked - kept
    if masked > 0.97:
        pytest.fail(f"все выбранные токены заменены на [MASK] ({masked:.0%}). Схема 80 / 10 / 10 нужна потому, что при "
                    "дообучении [MASK] не встречается: модель должна строить полезные представления и для обычных токенов.",
                    pytrace=False)
    assert abs(masked - 0.8) < 0.02, f"[MASK] у {masked:.1%} выбранных, ждём 80 %"
    assert abs(rand - 0.1) < 0.015, f"случайный токен у {rand:.1%} выбранных, ждём 10 %"
    assert abs(kept - 0.1) < 0.015, f"без изменений {kept:.1%} выбранных, ждём 10 %"
    replaced = inputs[chosen & (inputs != MASK) & (inputs != ids)]
    assert replaced.min() >= 0 and replaced.max() < V, "случайные токены — id из [0, vocab_size)"
    assert replaced.unique().numel() > 200, "случайные токены должны быть разными (равномерно по словарю)"


def test_prob_and_determinism(impl):
    """prob задаёт долю выбора; один и тот же генератор даёт один и тот же результат"""
    ids = _batch(B=128)
    normal = ~torch.isin(ids, torch.tensor(SPECIAL))
    _, labels = _run(impl, ids, prob=0.4)
    rate = (labels != -100)[normal].float().mean().item()
    assert abs(rate - 0.4) < 0.015, f"при prob = 0.4 выбрано {rate:.1%}"
    a = _run(impl, ids, seed=7)
    b = _run(impl, ids, seed=7)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), (
        "с одинаковым генератором результат должен совпадать: передавай generator во все torch.rand / torch.randint")
    c = _run(impl, ids, seed=8)
    assert not torch.equal(a[1], c[1]), "с другим сидом выбор должен быть другим"


def _words(B=64, n_words=60, seed=3):
    """Тексты из слов по 1–3 подтокена; word_ids: номер слова, −1 у [CLS] / [SEP] / PAD."""
    g = torch.Generator().manual_seed(seed)
    rows_ids, rows_w = [], []
    for _ in range(B):
        ids, wid = [CLS], [-1]
        for w in range(n_words):
            for _ in range(int(torch.randint(1, 4, (1,), generator=g))):
                ids.append(int(torch.randint(5, V, (1,), generator=g)))
                wid.append(w)
        ids.append(SEP)
        wid.append(-1)
        rows_ids.append(ids)
        rows_w.append(wid)
    T = max(map(len, rows_ids))
    ids = torch.tensor([r + [PAD] * (T - len(r)) for r in rows_ids])
    wids = torch.tensor([r + [-1] * (T - len(r)) for r in rows_w])
    return ids, wids


def test_whole_word_masking(impl):
    """word_ids: слово выбирается целиком — все его подтокены или ни одного; ≈ 15 % слов"""
    ids, wids = _words()
    inputs, labels = _run(impl, ids, word_ids=wids)
    chosen = labels != -100
    assert not chosen[wids < 0].any(), "позиции с word_ids = −1 (спецтокены, PAD) не выбираются"
    split, picked, total = 0, 0, 0
    for b in range(ids.shape[0]):
        for w in range(int(wids[b].max()) + 1):
            c = chosen[b][wids[b] == w]
            total += 1
            picked += bool(c.all())
            split += bool(c.any() and not c.all())
    if split:
        pytest.fail(f"у {split} слов выбрана только часть подтокенов. При whole word masking решение принимается для "
                    "слова и раздаётся всем его подтокенам: иначе «tok [MASK] ##ization» угадывается по соседним "
                    "кускам того же слова.", pytrace=False)
    assert abs(picked / total - 0.15) < 0.02, f"выбрано {picked / total:.1%} слов, ждём 15 %"
    masked = (inputs[chosen] == MASK).float().mean().item()
    assert abs(masked - 0.8) < 0.04, f"80 / 10 / 10 применяется к каждому выбранному подтокену: [MASK] у {masked:.0%}"
