"""Тесты «Проект: число → слова».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Полное обучение (~25 с) в тесты не входит: тесты проверяют части и запоминание маленькой выборки.
"""

import pytest
import torch

PAD, BOS, EOS = 0, 1, 2


@pytest.fixture(scope="module")
def vocab(ref):
    return ref.build_vocab()


def test_encode_pair(impl, vocab):
    """encode_pair: id цифр → id слов + EOS, без BOS и PAD"""
    itos, stoi = vocab
    src, tgt = impl.encode_pair(2022, stoi)
    assert [itos[i] for i in src] == list("2022"), f"источник — цифры числа по одной, получено {[itos[i] for i in src]}"
    words = [itos[i] for i in tgt]
    if words and words[0] == "<bos>":
        pytest.fail("в цели не нужен BOS: его ставит shift_right на вход декодера, а цель начинается с первого слова.",
                    pytrace=False)
    assert words == ["две", "тысячи", "двадцать", "два", "<eos>"], (
        f"цель — слова number_to_words(n) и EOS в конце, получено {words}")
    src0, tgt0 = impl.encode_pair(0, stoi)
    assert [itos[i] for i in src0] == ["0"] and [itos[i] for i in tgt0] == ["ноль", "<eos>"], "0 → «0» и «ноль <eos>»"


def test_collate(impl, vocab):
    """collate: PAD справа до максимальной длины, тензоры long"""
    _, stoi = vocab
    pairs = [impl.encode_pair(n, stoi) for n in (7, 123456, 40)]
    src, tgt = impl.collate(pairs)
    assert src.dtype == torch.long and tgt.dtype == torch.long, "id токенов — torch.long (их ждёт nn.Embedding)"
    assert src.shape == (3, 6) and tgt.shape == (3, 8), (
        f"src [3, 6] (самое длинное число — 6 цифр), tgt [3, 8] (7 слов + EOS), получено {tuple(src.shape)} и "
        f"{tuple(tgt.shape)}")
    for i, (s, t) in enumerate(pairs):
        if src[i, 0] == PAD and len(s) < 6:
            pytest.fail("PAD стоит слева. Дополняй справа: позиции считаются с 0 от начала текста (B02), а маски "
                        "считают PAD по id, где бы он ни стоял.", pytrace=False)
        assert src[i, : len(s)].tolist() == s and torch.all(src[i, len(s):] == PAD), f"строка {i} src: токены, затем PAD"
        assert tgt[i, : len(t)].tolist() == t and torch.all(tgt[i, len(t):] == PAD), f"строка {i} tgt: токены, затем PAD"


def _two_models(ref, vocab, d=32):
    itos, _ = vocab

    def make():
        torch.manual_seed(0)
        return ref.Seq2Seq(ref.Seq2SeqConfig(vocab_size=len(itos), d_model=d, n_heads=4, d_ff=2 * d, n_layers=1,
                                             max_len=16))

    return make(), make()


def test_train_step_matches_reference(impl, ref, vocab):
    """train_step: те же лоссы и веса, что у эталона, за 3 шага (shift right, label smoothing, clip, AdamW)"""
    _, stoi = vocab
    a, b = _two_models(ref, vocab)
    opt_a = torch.optim.AdamW(a.parameters(), lr=1e-2, weight_decay=0.01)
    opt_b = torch.optim.AdamW(b.parameters(), lr=1e-2, weight_decay=0.01)
    batch = ref.collate([ref.encode_pair(n, stoi) for n in (5, 2022, 123456, 999999, 70)])
    b.eval()
    for step in range(3):
        want = ref.train_step(a, opt_a, *batch)
        got = impl.train_step(b, opt_b, *batch)
        assert isinstance(got, float), f"train_step возвращает float (loss.item()), получено {type(got).__name__}"
        if step == 0 and abs(got - want) > 1e-4:
            pytest.fail(f"лосс первого шага {got:.4f}, ждём {want:.4f}: вход декодера — shift_right(tgt, BOS), цель — "
                        "tgt, label_smoothed_ce(logits, tgt, eps, ignore_index=PAD).", pytrace=False)
        assert b.training, "train_step переводит модель в режим обучения (model.train())"
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        if not torch.allclose(pa, pb, atol=1e-5):
            pytest.fail(f"после 3 шагов веса {name} расходятся с эталоном, хотя лосс первого шага верный. Проверь "
                        "порядок zero_grad → backward → clip_grad_norm_(…, clip) → step: градиенты обрезаются по "
                        "общей норме до шага оптимизатора.", pytrace=False)


def test_exact_match_rules(impl, ref, vocab, monkeypatch):
    """exact_match: совпадение до первого EOS включительно; без EOS или с лишним словом — промах; режим восстановлен"""
    _, stoi = vocab
    numbers = [5, 21, 300, 1000]
    tgts = [ref.encode_pair(n, stoi)[1] for n in numbers]
    rows = [
        tgts[0] + [PAD] * 3,                         # верно, хвост PAD
        tgts[1][:-1] + [stoi["один"], EOS, PAD],      # лишнее слово
        tgts[2][:-1] + [stoi["три"]] * 3,             # без EOS
        tgts[3] + [stoi["два"], stoi["два"]],          # после EOS мусор — не считается
    ]
    width = max(len(r) for r in rows)
    fake = torch.tensor([r + [PAD] * (width - len(r)) for r in rows])
    calls = {}

    def fake_greedy(model, src, max_len):
        calls["training"] = model.training
        calls["max_len"] = max_len
        return fake

    monkeypatch.setattr(impl, "greedy_decode", fake_greedy)
    model, _ = _two_models(ref, vocab)
    model.train()
    em = impl.exact_match(model, numbers, stoi)
    assert calls, "exact_match должен генерировать ответы через greedy_decode"
    assert calls["training"] is False, "генерировать нужно в eval-режиме (model.eval())"
    assert model.training, "после exact_match модель должна вернуться в прежний режим (была в train)"
    assert calls["max_len"] >= 8, f"max_len генерации — хотя бы 7 слов + EOS = 8, передано {calls['max_len']}"
    if em == pytest.approx(0.75):
        pytest.fail("ответ без EOS засчитан как верный: модель, которая не умеет остановиться, ошибается. "
                    "Сравнивай ответ до EOS включительно с целью без PAD.", pytrace=False)
    assert em == pytest.approx(0.5), (
        f"exact match = {em}, ждём 0.5: верны строки 1 и 4 (хвост после EOS не считается), строка 2 — лишнее "
        "слово, строка 3 — нет EOS")


def test_memorizes_small_set(impl, ref, vocab):
    """Smoke test: маленькая модель запоминает 12 чисел — exact match 1.0 на них"""
    itos, stoi = vocab
    torch.manual_seed(0)
    model = ref.Seq2Seq(ref.Seq2SeqConfig(vocab_size=len(itos), d_model=32, n_heads=4, d_ff=64, n_layers=1,
                                          max_len=16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    numbers = [3, 17, 40, 512, 2001, 11000, 21021, 99999, 123456, 500300, 700007, 999999]
    batch = impl.collate([impl.encode_pair(n, stoi) for n in numbers])
    for _ in range(250):
        loss = impl.train_step(model, opt, *batch, eps=0.0)
    em = impl.exact_match(model, numbers, stoi)
    assert loss < 0.05, f"на 12 числах лосс должен упасть почти до 0, получено {loss:.3f}"
    assert em == 1.0, f"модель должна запомнить все 12 чисел, exact match {em:.2f}"
