"""Тесты «BPE: encode по рангам и decode через байты».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Слияния обучаются эталоном (llmlab.core.bpe) на content/data/corpus-ru.txt.
"""

import random
from pathlib import Path

import pytest

from llmlab.core.bpe import build_vocab, merge_pair, train_bpe

CORPUS = Path(__file__).resolve().parents[4] / "data" / "corpus-ru.txt"
N_MERGES = 200

EXTRA = [
    "Токенизатор не видел этой строки при обучении.",
    "ёлка, Ёж и «кавычки» — тире…",
    "English words, digits 12345 and symbols #$%&",
    "汉字 и эмодзи 🙂👍🏽, флаг 🇷🇺",
    "  два пробела\tтаб\nперевод строки\r\n",
    "\x00\x7f служебные байты",
]


@pytest.fixture(scope="module")
def text() -> str:
    return CORPUS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def merges(text):
    return train_bpe([list(text.encode("utf-8"))], 256 + N_MERGES)


@pytest.fixture(scope="module")
def vocab(merges):
    return build_vocab(merges)


def test_decode_joins_bytes(impl, vocab, merges):
    """decode: байты токенов склеиваются, и только потом декодируется UTF-8"""
    assert impl.decode([], vocab) == "", "пустая последовательность → пустая строка"
    assert impl.decode([72, 105], vocab) == "Hi", "id < 256 — отдельные байты: [72, 105] → «Hi»"
    # первый токен корпуса — «пробел + первый байт кириллической буквы»: половина символа
    half = next(i for (a, b), i in merges.items() if (a, b) == (32, 208))
    got = impl.decode([half, 190], vocab)
    if "�" in got:
        pytest.fail(f"decode([{half}, 190]) = {got!r}: токен {half} = b' \\xd0' — это пробел и ПЕРВАЯ половина буквы "
                    "«о», вторая половина (0xBE) лежит в следующем токене. Если декодировать каждый токен отдельно, "
                    "обе половины превращаются в «�». Склей байты всех токенов, потом декодируй один раз.",
                    pytrace=False)
    assert got == " о", f"b' \\xd0' + b'\\xbe' = « о», а получено {got!r}"


def test_decode_invalid_utf8_does_not_raise(impl, vocab):
    """decode не падает на байтах, которые не складываются в UTF-8 (модель может сгенерировать любой id)"""
    try:
        got = impl.decode([208], vocab)
    except UnicodeDecodeError:
        pytest.fail("decode([208]) бросил UnicodeDecodeError: 0xD0 — начало двухбайтовой буквы без продолжения. "
                    "Языковая модель может сгенерировать такой id, и decode не должен падать: "
                    "bytes.decode('utf-8', errors='replace') заменит обрывок на «�».", pytrace=False)
    assert got == "�", f"обрывок символа заменяется на U+FFFD «�», получено {got!r}"


def test_encode_basics(impl, merges):
    """encode: список int, пустая строка → [], без подходящих слияний остаются байты"""
    assert impl.encode("", merges) == [], "пустая строка → []"
    assert impl.encode("Z", merges) == [90], "один байт сливать не с чем"
    got = impl.encode("~~~", merges)
    assert got == [126, 126, 126], f"пары «~~» нет в merges — остаются байты, получено {got}"
    got = impl.encode("о", merges)
    assert isinstance(got, list) and all(isinstance(i, int) for i in got), "encode возвращает list[int]"
    assert got == [merges[(208, 190)]], f"«о» = D0 BE, и эта пара есть в merges → один токен, получено {got}"


def test_ranks_not_frequency(impl):
    """Порядок слияний — по рангу обучения, а не по частоте пар в кодируемом тексте"""
    merges = {(97, 98): 256, (98, 99): 257}            # сначала выучено «ab», потом «bc»
    got = impl.encode("bcbcabc", merges)
    if got == [257, 257, 97, 257]:
        pytest.fail("«bcbcabc» закодирован как [bc, bc, a, bc]: первой слита пара «bc», потому что в этом тексте "
                    "она встречается чаще. Но при обучении раньше выучено «ab» (id 256), и его нужно применить "
                    "первым: правильно [bc, bc, ab, c] = [257, 257, 256, 99]. Encode повторяет порядок обучения, "
                    "иначе один и тот же кусок текста кодируется по-разному в разных контекстах.", pytrace=False)
    assert got == [257, 257, 256, 99], f"ждём [257, 257, 256, 99], получено {got}"


def test_encode_repeats_training(impl, text, merges):
    """encode(обучающий текст) == последовательность, полученная при обучении"""
    seq = list(text.encode("utf-8"))
    for pair, new_id in merges.items():
        seq = merge_pair(seq, pair, new_id)
    got = impl.encode(text, merges)
    assert len(got) == len(seq), (
        f"на обучающем тексте encode должен повторить обучение: {len(seq)} токенов, а получено {len(got)}")
    k = next((i for i, (a, b) in enumerate(zip(got, seq)) if a != b), None)
    assert k is None, f"первое расхождение с обучением в позиции {k}: ждём {seq[k]}, получено {got[k]}"


def test_encode_matches_reference(impl, ref, merges):
    """encode на новых строках совпадает с эталоном"""
    for s in EXTRA:
        assert impl.encode(s, merges) == ref.encode(s, merges), f"encode({s!r}) не совпадает с эталоном"


def test_roundtrip_any_utf8(impl, text, merges, vocab):
    """decode(encode(s)) == s для любой строки: корпус, эмодзи, иероглифы, служебные символы, случайный Unicode"""
    rnd = random.Random(0)
    samples = EXTRA + [text[:2000]]
    for _ in range(30):
        n = rnd.randint(1, 12)
        samples.append("".join(chr(rnd.choice([rnd.randint(32, 0x24F), rnd.randint(0x400, 0x4FF),
                                                rnd.randint(0x4E00, 0x9FFF), rnd.randint(0x1F300, 0x1FAFF)]))
                               for _ in range(n)))
    for s in samples:
        got = impl.decode(impl.encode(s, merges), vocab)
        assert got == s, f"decode(encode(s)) != s для s = {s[:40]!r}: получено {got[:40]!r}"


def test_compression(impl, merges):
    """На новом русском тексте токенов заметно меньше, чем байтов"""
    s = ("Модель читает текст по токенам и каждый раз предсказывает следующий. Чем короче последовательность, "
         "тем больше текста помещается в контекст и тем меньше шагов генерации.")
    n_bytes, n_tok = len(s.encode("utf-8")), len(impl.encode(s, merges))
    assert n_tok < 0.5 * n_bytes, (
        f"{n_bytes} байтов → {n_tok} токенов: после {N_MERGES} слияний на русском тексте ждём сжатие больше чем вдвое")
