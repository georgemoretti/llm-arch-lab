"""Тесты «Byte-level BPE: обучение словаря слияний».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Корпус — content/data/corpus-ru.txt (текст написан авторами курса, ~23 тыс. символов, ~42 тыс. байтов UTF-8).
"""

from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[4] / "data" / "corpus-ru.txt"
N_MERGES = 120


@pytest.fixture(scope="module")
def corpus_bytes() -> list[int]:
    return list(CORPUS.read_text(encoding="utf-8").encode("utf-8"))


@pytest.fixture(scope="module")
def corpus_merges(impl, corpus_bytes):
    return impl.train_bpe([corpus_bytes], 256 + N_MERGES)


def _show(merges: dict, k: int = 6) -> str:
    items = list(merges.items())[:k]
    return ", ".join(f"{p}→{i}" for p, i in items) + (" …" if len(merges) > k else "")


def test_count_pairs(impl):
    """count_pairs: частоты соседних пар, перекрывающиеся пары считаются, через границу последовательностей — нет"""
    got = impl.count_pairs([[1, 2, 1, 2, 3]])
    assert dict(got) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}, (
        f"в [1, 2, 1, 2, 3] пары (1, 2) ×2, (2, 1) ×1, (2, 3) ×1, а получено {dict(got)}")
    got = impl.count_pairs([[5, 5, 5]])
    assert dict(got) == {(5, 5): 2}, (
        f"в [5, 5, 5] пара (5, 5) стоит в позициях 0–1 и 1–2: при подсчёте это 2 вхождения, получено {dict(got)}")
    got = dict(impl.count_pairs([[1, 2], [3, 4], [2, 3]]))
    if (2, 3) in got and got[(2, 3)] == 2:
        pytest.fail("пара (2, 3) посчитана через границу последовательностей [1, 2] и [3, 4]. Последовательности — "
                    "независимые куски текста (в A13 — отдельные слова), пары между ними не существуют.", pytrace=False)
    assert got == {(1, 2): 1, (3, 4): 1, (2, 3): 1}, f"для [[1, 2], [3, 4], [2, 3]] получено {got}"
    assert dict(impl.count_pairs([[7], []])) == {}, "в последовательностях длины 0 и 1 пар нет"


def test_merge_pair(impl):
    """merge_pair: замена слева направо без перекрытий, исходный список не меняется"""
    seq = [1, 2, 3, 1, 2]
    got = impl.merge_pair(seq, (1, 2), 9)
    assert got == [9, 3, 9], f"[1, 2, 3, 1, 2] с парой (1, 2) → 9 должно стать [9, 3, 9], а получено {got}"
    assert seq == [1, 2, 3, 1, 2], "merge_pair изменил входной список: верни новый, исходный ещё нужен"
    got = impl.merge_pair([5, 5, 5], (5, 5), 7)
    if got == [5, 7]:
        pytest.fail("[5, 5, 5] → [5, 7]: слияние шло справа налево. Слева направо получается [7, 5].", pytrace=False)
    if got == [7, 7]:
        pytest.fail("[5, 5, 5] → [7, 7]: средняя пятёрка использована дважды. После замены перепрыгни через оба "
                    "элемента пары: правильно [7, 5].", pytrace=False)
    assert got == [7, 5], f"[5, 5, 5] с парой (5, 5) → 7 должно стать [7, 5], а получено {got}"
    assert impl.merge_pair([5, 5, 5, 5], (5, 5), 7) == [7, 7], (
        "[5, 5, 5, 5] с парой (5, 5) → [7, 7]: две непересекающиеся пары, обе заменяются")
    assert impl.merge_pair([1, 3, 1], (1, 2), 9) == [1, 3, 1], "пары нет — список не меняется"
    assert impl.merge_pair([4, 1, 2], (1, 2), 9) == [4, 9], "пара в самом конце тоже сливается"
    assert impl.merge_pair([], (1, 2), 9) == [], "пустая последовательность остаётся пустой"


def test_train_toy(impl, ref):
    """train_bpe на «aaabdaaabac»: те же слияния и в том же порядке, что у эталона"""
    seqs = [list(b"aaabdaaabac")]
    got = impl.train_bpe(seqs, 259)
    want = ref.train_bpe(seqs, 259)
    assert seqs == [list(b"aaabdaaabac")], "train_bpe изменил входные последовательности: работай с копиями"
    assert isinstance(got, dict), f"train_bpe возвращает dict {{пара: id}}, а не {type(got).__name__}"
    if list(got.values()) and min(got.values()) < 256:
        pytest.fail(f"id новых токенов начинаются с {min(got.values())}: id 0..255 уже заняты байтами, первое "
                    "слияние создаёт токен 256.", pytrace=False)
    assert list(got.items()) == list(want.items()), (
        f"слияния не совпадают: ждём {_show(want)}, получено {_show(got)}. a = 97, b = 98: самая частая пара "
        "сначала (97, 97) — «aa», затем в новом тексте ищется следующая самая частая.")


def test_tie_break_and_ranks(impl):
    """Равные частоты → меньшая пара; id новых токенов идут подряд с 256; vocab_size < 256 → ValueError"""
    got = impl.train_bpe([[3, 4, 1, 2]], 257)
    assert list(got) == [(1, 2)], (
        f"в [3, 4, 1, 2] все пары встречаются по одному разу, и при равенстве берётся меньшая пара (1, 2), а "
        f"выбрана {list(got)}. Правило «меньшая пара» делает обучение воспроизводимым: результат не зависит от "
        "порядка обхода словаря частот.")
    got = impl.train_bpe([[3, 4, 1, 2]], 300)
    assert list(got.values()) == [256, 257, 258], (
        f"после 3 слияний последовательность [3, 4, 1, 2] стала одним токеном, и обучение должно остановиться: "
        f"ждём id 256, 257, 258, получено {list(got.values())}")
    assert impl.train_bpe([[1, 2]], 256) == {}, "vocab_size = 256 — ни одного слияния"
    with pytest.raises(ValueError):
        impl.train_bpe([[1, 2]], 200)


def test_merges_use_only_earlier_tokens(impl, corpus_merges):
    """На корпусе: каждое слияние склеивает байты или уже выученные токены"""
    assert len(corpus_merges) == N_MERGES, f"vocab_size = 256 + {N_MERGES} → {N_MERGES} слияний, получено {len(corpus_merges)}"
    assert list(corpus_merges.values()) == list(range(256, 256 + N_MERGES)), (
        "id новых токенов идут подряд в порядке обучения: 256, 257, …")
    for (a, b), new_id in corpus_merges.items():
        assert a < new_id and b < new_id, (
            f"слияние ({a}, {b}) → {new_id} ссылается на токен, который ещё не выучен: в паре могут быть только "
            "байты и токены с меньшим id")


def test_corpus_matches_reference(impl, ref, corpus_bytes, corpus_merges):
    """На корпусе: первые 120 слияний совпадают с эталоном"""
    want = ref.train_bpe([corpus_bytes], 256 + N_MERGES)
    got_items, want_items = list(corpus_merges.items()), list(want.items())
    for k, (g, w) in enumerate(zip(got_items, want_items)):
        if g != w:
            pytest.fail(f"слияние №{k} отличается: ждём {w}, получено {g}. Самая частая пара считается заново "
                        "после каждого слияния, по текущей (уже слитой) последовательности.", pytrace=False)
    assert got_items == want_items, (
        f"слияний {len(got_items)}, а у эталона {len(want_items)}: проверь условие остановки (vocab_size и «пар больше нет»)")


def test_build_vocab(impl, ref, corpus_bytes, corpus_merges):
    """build_vocab: 256 байтов + склейки; склейка токенов обученной последовательности = исходные байты"""
    vocab = impl.build_vocab(corpus_merges)
    assert len(vocab) == 256 + N_MERGES, f"в словаре 256 байтов + {N_MERGES} токенов, получено {len(vocab)}"
    assert all(vocab[i] == bytes([i]) for i in range(256)), "vocab[i] для i < 256 — один байт bytes([i])"
    assert all(isinstance(v, bytes) for v in vocab.values()), (
        "значения словаря — bytes, а не str: токен может оказаться половиной символа UTF-8")
    for (a, b), new_id in corpus_merges.items():
        assert vocab[new_id] == vocab[a] + vocab[b], (
            f"vocab[{new_id}] должен быть vocab[{a}] + vocab[{b}] = {vocab[a] + vocab[b]!r}, а получено {vocab[new_id]!r}")
    seq = list(corpus_bytes)
    lengths = [len(seq)]
    for pair, new_id in corpus_merges.items():
        seq = impl.merge_pair(seq, pair, new_id)
        lengths.append(len(seq))
    assert b"".join(vocab[i] for i in seq) == bytes(corpus_bytes), (
        "склейка байтов всех токенов обученной последовательности должна дать исходный текст байт в байт")
    assert all(x > y for x, y in zip(lengths, lengths[1:])), "каждое слияние самой частой пары укорачивает текст"
    assert lengths[-1] < 0.45 * lengths[0], (
        f"после {N_MERGES} слияний {lengths[0]} байтов должны сжаться хотя бы вдвое с запасом, получено "
        f"{lengths[-1]} токенов")
