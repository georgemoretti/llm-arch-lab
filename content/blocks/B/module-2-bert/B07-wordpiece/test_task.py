"""Тесты «WordPiece».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул — библиотека tokenizers (BertNormalizer + BertPreTokenizer + WordPiece), если установлена.
"""

import collections
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[4] / "data" / "corpus-ru.txt"
TOY = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "un", "una", "##aff", "##a", "##ff", "##able", "##ab", "##le",
       "play", "##ing", "##s", "при", "##вет", "##ве", "##т", "мир", "аз", "##бук", "##а"]
VOCAB = {t: i for i, t in enumerate(TOY)}


def test_strip_accents(impl):
    """strip_accents: NFD + удаление знаков Mn — «café» → «cafe», «й» → «и», «ё» → «е»"""
    assert impl.strip_accents("café naïve Ångström") == "cafe naive Angstrom", (
        "диакритика латиницы должна исчезнуть: NFD раскладывает «é» на «e» + знак U+0301 (категория Mn)")
    got = impl.strip_accents("йод ёлка")
    assert got == "иод елка", (
        f"получено {got!r}. У BERT uncased «й» тоже теряет кратку: в NFD это «и» + U+0306. Так устроен словарь "
        "bert-base-uncased — модель никогда не видит букву «й».")
    assert impl.strip_accents("привет, мир 42") == "привет, мир 42", "буквы без диакритики не меняются"


def test_basic_tokenize(impl):
    """basic_tokenize: нижний регистр, без диакритики, разрез по пробелам и по каждому знаку пунктуации"""
    got = impl.basic_tokenize("Hello,  World!")
    if got == ["hello,", "world!"]:
        pytest.fail("пунктуация не отделена: каждый знак пунктуации — отдельное слово ('hello', ',', …).",
                    pytrace=False)
    if got == ["Hello", ",", "World", "!"]:
        pytest.fail("регистр не понижен: при lower=True текст переводится в нижний регистр.", pytrace=False)
    assert got == ["hello", ",", "world", "!"], f"получено {got}"
    cases = {
        "don't stop...": ["don", "'", "t", "stop", ".", ".", "."],
        "π = 3.14": ["π", "=", "3", ".", "14"],
        "«Ёлка» — это\tдерево\n": ["«", "елка", "»", "—", "это", "дерево"],
        "a​b\x00c  ": ["abc"],
        "": [],
        "price: $5+3=8 <ok> a^b `x` |y| ~z": ["price", ":", "$", "5", "+", "3", "=", "8", "<", "ok", ">", "a", "^",
                                             "b", "`", "x", "`", "|", "y", "|", "~", "z"],
        "a!b/c:d@e[f`g{h~i0": ["a", "!", "b", "/", "c", ":", "d", "@", "e", "[", "f", "`", "g", "{", "h", "~", "i0"],
    }
    for text, want in cases.items():
        got = impl.basic_tokenize(text)
        assert got == want, f"basic_tokenize({text!r}) = {got}, ждём {want}"
    assert impl.basic_tokenize("Café, Ёж!", lower=False) == ["Café", ",", "Ёж", "!"], (
        "при lower=False регистр и диакритика сохраняются (так работает bert-base-cased)")
    assert impl.basic_tokenize("Zoo AZ", lower=False) == ["Zoo", "AZ"], "заглавные A … Z — буквы, а не пунктуация"


def test_wordpiece_greedy(impl):
    """wordpiece: самый длинный префикс из словаря, продолжения с «##»"""
    got = impl.wordpiece("unaffable", VOCAB)
    if got == ["un", "##a", "##ff", "##ab", "##le"]:
        pytest.fail("берутся самые короткие куски. WordPiece жадный: с каждой позиции — САМЫЙ ДЛИННЫЙ кусок из "
                    "словаря.", pytrace=False)
    assert got == ["una", "##ff", "##able"], (
        f"получено {got}. С позиции 0 самый длинный кусок — «una» (а не «un»), затем «##ff», «##able». Жадность не "
        "ищет лучшего разбиения: «un ##aff ##able» тоже возможно, но алгоритм его не найдёт.")
    assert impl.wordpiece("playing", VOCAB) == ["play", "##ing"], (
        f"«playing» → play ##ing, получено {impl.wordpiece('playing', VOCAB)}")
    assert impl.wordpiece("привет", VOCAB) == ["при", "##вет"], "у кириллицы так же: при + ##вет"
    assert impl.wordpiece("мир", VOCAB) == ["мир"], "слово целиком в словаре — один токен без ##"


def test_wordpiece_unknown(impl):
    """Неразбираемое слово целиком → [UNK]; продолжения ищутся только с «##»; длинные слова → [UNK]"""
    got = impl.wordpiece("playx", VOCAB)
    if got[:1] == ["play"]:
        pytest.fail(f"получено {got}: начало слова разобралось, а хвост — нет. В BERT тогда всё слово заменяется одним "
                    "[UNK], частичных разборов не бывает.", pytrace=False)
    assert got == ["[UNK]"], f"получено {got}"
    got = impl.wordpiece("unпри", VOCAB)
    assert got == ["[UNK]"], (f"получено {got}: «при» есть в словаре, но как начало слова. Продолжение ищется только в "
                              "виде «##при», которого нет.")
    assert impl.wordpiece("азбука", VOCAB) == ["аз", "##бук", "##а"], (
        f"кириллица режется так же: аз ##бук ##а, получено {impl.wordpiece('азбука', VOCAB)}")
    assert impl.wordpiece("un" * 60, {**VOCAB, "##un": 99}) == ["[UNK]"], "слово длиннее max_chars = 100 → [UNK]"
    assert impl.wordpiece("x", VOCAB, unk="<unk>") == ["<unk>"], "токен неизвестного слова — параметр unk"


def test_tokenize(impl):
    """tokenize: basic_tokenize, затем wordpiece каждого слова"""
    got = impl.tokenize("Unaffable playing, мир!", {**VOCAB, ",": 50, "!": 51})
    assert got == ["una", "##ff", "##able", "play", "##ing", ",", "мир", "!"], f"получено {got}"


def _corpus_vocab(ref, text):
    words = collections.Counter(ref.basic_tokenize(text))
    toks = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted({w[:3] for w in words})
    toks += sorted({"##" + w[i:i + 2] for w in words for i in range(1, len(w), 2)})[:400]
    return {t: i for i, t in enumerate(dict.fromkeys(toks))}


def test_matches_tokenizers_on_corpus(impl, ref):
    """На корпусе курса (плюс пунктуация, табы, невидимые символы) — те же токены, что у библиотеки tokenizers"""
    tok = pytest.importorskip("tokenizers", reason="нужен tokenizers: uv sync --extra weights")
    text = CORPUS.read_text(encoding="utf-8") + " Ça va? Naïve café — «ёлка» и йод!\tTab​zw  ... (1,5%)"
    vocab = _corpus_vocab(ref, text)
    oracle = tok.Tokenizer(tok.models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    oracle.normalizer = tok.normalizers.BertNormalizer(lowercase=True)
    oracle.pre_tokenizer = tok.pre_tokenizers.BertPreTokenizer()
    want = oracle.encode(text, add_special_tokens=False).tokens
    got = impl.tokenize(text, vocab)
    if got != want:
        i = next((k for k, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
        pytest.fail(f"расхождение с tokenizers на токене {i}: у тебя {got[max(0, i - 3):i + 3]}, у библиотеки "
                    f"{want[max(0, i - 3):i + 3]} (всего {len(got)} против {len(want)})", pytrace=False)


@pytest.mark.weights
def test_bert_base_uncased_vocab(impl):
    """Словарь bert-base-uncased (30 522 токена): разбор совпадает с официальным токенизатором"""
    hub = pytest.importorskip("huggingface_hub", reason="нужен huggingface_hub: uv sync --extra weights")
    tok = pytest.importorskip("tokenizers", reason="нужен tokenizers: uv sync --extra weights")
    path = hub.hf_hub_download("google-bert/bert-base-uncased", "vocab.txt")
    vocab = {line.rstrip("\n"): i for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines())}
    assert len(vocab) == 30522, f"vocab.txt bert-base-uncased — 30 522 строки, прочитано {len(vocab)}"
    oracle = tok.Tokenizer.from_pretrained("google-bert/bert-base-uncased")
    for text in ["The unaffable tokenizer isn't playing!", "Transformers: attention is all you need (2017).",
                 "Привет, мир! Йогурт и ёжик.", CORPUS.read_text(encoding="utf-8")[:3000]]:
        want = oracle.encode(text, add_special_tokens=False).tokens
        assert impl.tokenize(text, vocab) == want, f"расхождение с bert-base-uncased на тексте {text[:40]!r}"
