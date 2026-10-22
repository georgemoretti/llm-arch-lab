"""Тесты «Найди баг: decode» — те же, что у задачи «Пре-токенизация GPT-2, спецтокены, byte ↔ unicode», без подсказок про баг.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Корпус — content/data/corpus-ru.txt.
"""

from pathlib import Path

import pytest

from llmlab.core.bpe import build_vocab, encode, pretokenize

CORPUS = Path(__file__).resolve().parents[4] / "data" / "corpus-ru.txt"
N_MERGES = 150
EOT = "<|endoftext|>"

CASES = [
    ("Hello world", ["Hello", " world"]),
    ("I'm here, it's 2024!", ["I", "'m", " here", ",", " it", "'s", " 2024", "!"]),
    ("Привет, мир", ["Привет", ",", " мир"]),
    ("x = 3.14", ["x", " =", " 3", ".", "14"]),
    ("a  b", ["a", " ", " b"]),
    ("end   ", ["end", "   "]),
    ("a\n\nb", ["a", "\n", "\n", "b"]),
    ("ёжик-ёж", ["ёжик", "-", "ёж"]),
    ("GPT2 и 3dnow", ["GPT", "2", " и", " 3", "dnow"]),
]


@pytest.fixture(scope="module")
def text() -> str:
    return CORPUS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def merges(ref, text):
    return ref.train_regex_bpe(text, 256 + N_MERGES)


@pytest.fixture(scope="module")
def vocab(merges):
    return build_vocab(merges)


@pytest.fixture(scope="module")
def special(merges):
    return {EOT: 256 + len(merges)}


def test_pretokenize_examples(impl):
    """pretokenize: пробел прилипает к началу слова, буквы, цифры и пунктуация — отдельные куски"""
    for s, want in CASES:
        got = impl.pretokenize(s)
        assert "".join(got) == s, f"куски pretokenize({s!r}) должны склеиваться обратно в исходную строку, получено {got}"
        if s == "Hello world" and got == ["Hello ", "world"]:
            pytest.fail("пробел приклеен к концу слова: в GPT-2 он идёт в НАЧАЛО следующего (« world»). Тогда "
                        "слово в середине предложения всегда выглядит одинаково: пробел + буквы.", pytrace=False)
        if s == "x = 3.14" and " 3.14" in got:
            pytest.fail("« 3.14» оказался одним куском: цифры и пунктуация — разные классы (\\p{N} и "
                        "[^\\s\\p{L}\\p{N}]), правильно « 3», «.», «14».", pytrace=False)
        if s == "a  b" and got == ["a", "  b"]:
            pytest.fail("«a  b» → ['a', '  b']: необязательный пробел перед словом — ровно один (` ?`). Лишние "
                        "пробелы забирает альтернатива \\s+(?!\\S): она не берёт последний пробел перед словом.",
                        pytrace=False)
        if s == "end   " and got[-1] != "   ":
            pytest.fail(f"«end   » → {got}: пробелы в конце строки — один кусок «   » (\\s+(?!\\S) или \\s+).",
                        pytrace=False)
        assert got == want, f"pretokenize({s!r}) = {got}, ждём {want}"


def test_pretokenize_matches_reference(impl, ref, text):
    """pretokenize корпуса совпадает с эталоном"""
    got, want = impl.pretokenize(text), ref.pretokenize(text)
    k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    assert k is None and len(got) == len(want), (
        f"куски корпуса расходятся с эталоном{'' if k is None else f' в позиции {k}: ждём {want[k]!r}, получено {got[k]!r}'}"
        f" (кусков: {len(got)} против {len(want)})")


def test_bytes_to_unicode(impl, ref):
    """bytes_to_unicode: 256 различных печатных символов, видимые байты — сами собой, пробел → «Ġ»"""
    table = impl.bytes_to_unicode()
    assert sorted(table) == list(range(256)), "ключи — все байты 0..255"
    chars = list(table.values())
    assert len(set(chars)) == 256, "отображение обратимо: 256 разных символов"
    bad = [b for b, c in table.items() if not (len(c) == 1 and c.isprintable() and not c.isspace())]
    assert not bad, (f"байты {bad[:6]} отображены в непечатные или пробельные символы: смысл отображения в том, "
                     "чтобы любой токен записывался видимыми символами в vocab.json и merges.txt")
    assert all(table[b] == chr(b) for b in range(ord("!"), ord("~") + 1)), "видимые ASCII-байты отображаются в себя"
    assert table[32] == "Ġ", f"пробел (байт 32) — 33-й «невидимый» байт → chr(256 + 32) = «Ġ», получено {table[32]!r}"
    assert table[10] == "Ċ" and table[0] == "Ā", "байт 0 → «Ā» (chr 256), перевод строки (10) → «Ċ» (chr 266)"
    assert table == ref.bytes_to_unicode(), (
        "отображение не совпадает с эталоном: видимые диапазоны «!»..«~», «¡»..«¬», «®»..«ÿ» — включительно, "
        "остальные байты по возрастанию → chr(256), chr(257), …")
    as_str = "".join(table[b] for b in " мир".encode("utf-8"))
    assert as_str == "ĠÐ¼Ð¸ÑĢ", f"токен « мир» в записи GPT-2 выглядит как «ĠÐ¼Ð¸ÑĢ», получено {as_str!r}"


def test_train_regex_bpe(impl, ref, text, merges):
    """train_regex_bpe: слияния = эталон; ни один токен не склеивает слово с пробелом после него"""
    got = impl.train_regex_bpe(text, 256 + N_MERGES)
    plain = ref.train_bpe([list(text.encode("utf-8"))], 256 + N_MERGES)
    if list(got.items()) == list(plain.items()):
        pytest.fail("слияния совпали с обычным BPE по всему тексту: пары нужно считать внутри кусков "
                    "pretokenize(text), каждый кусок — отдельная последовательность байтов.", pytrace=False)
    assert list(got.items()) == list(merges.items()), "слияния не совпадают с эталоном"
    vocab = build_vocab(got)
    for i in range(256, 256 + N_MERGES):
        tok = vocab[i]
        assert b" " not in tok[1:] or not tok.strip(), (
            f"токен {tok!r} содержит пробел не в начале: слияния не должны выходить за границы кусков")


def test_split_special(impl):
    """split_special: спецтокены — отдельные элементы, пустые куски выброшены, | в спецтокене экранирован"""
    sp = {EOT: 400}
    got = impl.split_special(f"a{EOT}b", sp)
    if got != ["a", EOT, "b"] and any(p in ("<", ">", "endoftext") for p in got):
        pytest.fail(f"split_special('a{EOT}b') = {got}: символ | в «<|endoftext|>» сработал как «или» регулярки. "
                    "Экранируй спецтокены: regex.escape(s).", pytrace=False)
    if got == ["a", "b"]:
        pytest.fail("спецтокен пропал из результата: regex.split сохраняет разделители, только если они в "
                    "скобках (группе).", pytrace=False)
    assert got == ["a", EOT, "b"], f"ждём ['a', '{EOT}', 'b'], получено {got}"
    got = impl.split_special(f"{EOT}{EOT}x{EOT}", sp)
    assert got == [EOT, EOT, "x", EOT], f"спецтокены в начале, подряд и в конце: пустых кусков нет, получено {got}"
    assert impl.split_special(f"a{EOT}b", {}) == [f"a{EOT}b"], "без спецтокенов весь текст — один кусок"
    assert impl.split_special("", sp) == [] and impl.split_special("", {}) == [], "пустая строка → []"
    two = {EOT: 400, "<|pad|>": 401}
    assert impl.split_special(f"<|pad|>hi{EOT}", two) == ["<|pad|>", "hi", EOT], (
        "несколько спецтокенов: режем по каждому из них, сами спецтокены — отдельные куски")


def test_encode_with_special(impl, merges, special):
    """encode_with_special: спецтокен — один id; обычный текст кодируется по кускам pretokenize"""
    s = f"Первый документ.{EOT}Второй документ"
    got = impl.encode_with_special(s, merges, special)
    eot = special[EOT]
    assert got.count(eot) == 1, f"«{EOT}» должен стать одним токеном {eot}, а он встречается {got.count(eot)} раз"
    left, right = s.split(EOT)

    def by_chunks(t):
        return [i for chunk in pretokenize(t) for i in encode(chunk, merges)]

    assert got == by_chunks(left) + [eot] + by_chunks(right), "ждём: куски левой части, спецтокен, куски правой"
    plain = impl.encode_with_special(s, merges, {})
    assert eot not in plain, "без словаря спецтокенов «<|endoftext|>» — обычный текст, его id не появляется"
    assert impl.encode_with_special("", merges, special) == [], "пустая строка → пустой список id"
    # слияние «a + пробел» обучение с пре-токенизацией не выучит, но encode обязан уважать куски при любых merges
    got = impl.encode_with_special("a b", {(97, 32): 256}, {})
    if got == [256, 98]:
        pytest.fail("«a b» с merges {(a, пробел): 256} → [256, 98]: обычный текст закодирован целиком, и слияние "
                    "прошло через границу кусков «a» | « b». Сначала pretokenize, потом encode каждого куска.",
                    pytrace=False)
    assert got == [97, 32, 98], f"куски «a», « b» кодируются отдельно → [97, 32, 98], получено {got}"


def test_decode_with_special(impl, vocab, special):
    """decode_with_special: спецтокены и эмодзи из четырёх байтовых токенов"""
    eot = special[EOT]
    assert impl.decode_with_special([72, 105, eot, 33], vocab, special) == f"Hi{EOT}!", (
        "id спецтокена декодируется в его строку «<|endoftext|>», остальные id — через vocab: ждём «Hi<|endoftext|>!»")
    emoji = list("🙂".encode("utf-8"))                     # 4 байта, на корпусе не сливаются
    got = impl.decode_with_special(emoji + [eot], vocab, special)
    assert got == "🙂" + EOT, f"4 байта эмодзи 🙂 и спецтокен должны дать «🙂{EOT}», получено {got!r}"


def test_roundtrip(impl, text, merges, vocab, special):
    """decode(encode(s)) == s: корпус, спецтокены между документами, эмодзи, иероглифы, пробелы"""
    samples = [text[:3000], f"{EOT}Документ 1{EOT}Документ 2 🙂{EOT}", "汉字 ёлка  \n\t x=1.5", f"<|endof{EOT}text|>"]
    for s in samples:
        ids = impl.encode_with_special(s, merges, special)
        got = impl.decode_with_special(ids, vocab, special)
        assert got == s, f"decode(encode(s)) != s для {s[:40]!r}: получено {got[:40]!r}"
