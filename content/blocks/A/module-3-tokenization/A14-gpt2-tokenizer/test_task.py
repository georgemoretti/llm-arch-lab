"""Тесты «Свой токенизатор с merges GPT-2: сверка с эталонным».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Без сети: тесты пишут «игрушечные» vocab.json и merges.txt в формате GPT-2 (из BPE, обученного эталоном на
корпусе курса, id в vocab.json перемешаны). С --weights (`llmlab test A14-gpt2-tokenizer --weights`, нужен
`uv sync --extra weights`, ~2 МБ): настоящие файлы openai-community/gpt2 и сверка с библиотекой tokenizers.
"""

import json
import random
from pathlib import Path

import pytest

from llmlab.core.bpe import bytes_to_unicode, encode_with_special, train_regex_bpe

CORPUS = Path(__file__).resolve().parents[4] / "data" / "corpus-ru.txt"
N_MERGES = 150
EOT = "<|endoftext|>"
SAMPLES = [
    "Hello world",
    "  два пробела, хвост   ",
    "I'M fine, it's 3.14159!!",
    "🙂 汉字 ёлка",
    "def f(x):\n    return x**2\n\n",
    f"документ 1{EOT}документ 2",
]


@pytest.fixture(scope="module")
def text() -> str:
    return CORPUS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def toy(tmp_path_factory, text):
    """Файлы в формате GPT-2 из своего BPE: строки через bytes_to_unicode, id перемешаны, спецтокен в конце."""
    merges = train_regex_bpe(text, 256 + N_MERGES)
    enc = bytes_to_unicode()
    strings = {b: enc[b] for b in range(256)}
    for (a, b), new_id in merges.items():
        strings[new_id] = strings[a] + strings[b]
    perm = list(range(256 + N_MERGES))
    random.Random(0).shuffle(perm)                          # id GPT-2 ≠ своим id: так и есть в настоящем vocab.json
    to_gpt2 = {ours: perm[ours] for ours in strings}
    vocab_json = {strings[ours]: to_gpt2[ours] for ours in strings}
    vocab_json[EOT] = 256 + N_MERGES
    d = tmp_path_factory.mktemp("gpt2-toy")
    (d / "vocab.json").write_text(json.dumps(vocab_json, ensure_ascii=False), encoding="utf-8")
    lines = ["#version: 0.2"] + [f"{strings[a]} {strings[b]}" for (a, b) in merges]
    (d / "merges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"dir": d, "merges": merges, "to_gpt2": to_gpt2, "special": {EOT: 256 + N_MERGES}}


def _paths(d: Path):
    return d / "vocab.json", d / "merges.txt"


def _oracle(toy, s, allowed_special=True):
    special = {EOT: -1} if allowed_special else {}
    ours = encode_with_special(s, toy["merges"], special)
    return [toy["special"][EOT] if i == -1 else toy["to_gpt2"][i] for i in ours]


def test_load_merges(impl, toy):
    """load_gpt2_files: слияния в своих id с рангами 256 + k, заголовок #version пропущен"""
    merges, to_gpt2, special = impl.load_gpt2_files(*_paths(toy["dir"]))
    want = toy["merges"]
    if len(merges) == len(want) + 1:
        pytest.fail("слияний на одно больше: первая строка merges.txt («#version: 0.2») — заголовок, а не "
                    "правило слияния.", pytrace=False)
    assert len(merges) == len(want), f"в merges.txt {len(want)} правил, загружено {len(merges)}"
    first = next(iter(merges.items()))
    if first[1] == 0:
        pytest.fail("ранги начинаются с 0: свои id 0..255 заняты байтами, слияние k создаёт токен 256 + k.",
                    pytrace=False)
    k = next((i for i, (g, w) in enumerate(zip(merges.items(), want.items())) if g != w), None)
    assert k is None, (
        f"правило №{k} не совпадает: ждём {list(want.items())[k]}, получено {list(merges.items())[k]}. Части "
        "правила — строки из bytes_to_unicode: одиночный символ — байт, склейка — уже выученный токен.")


def test_load_id_mapping_and_special(impl, toy):
    """load_gpt2_files: to_gpt2 переводит свои id в id из vocab.json; спецтокены — то, что не байт и не слияние"""
    merges, to_gpt2, special = impl.load_gpt2_files(*_paths(toy["dir"]))
    assert special == toy["special"], f"спецтокены: ждём {toy['special']}, получено {special}"
    assert len(to_gpt2) == 256 + N_MERGES, f"to_gpt2 — для всех {256 + N_MERGES} своих токенов, получено {len(to_gpt2)}"
    bad = [o for o, g in toy["to_gpt2"].items() if to_gpt2.get(o) != g]
    if bad and all(to_gpt2.get(o) == o for o in to_gpt2):
        pytest.fail("to_gpt2 — тождественное отображение: id в vocab.json не совпадают со своими (у настоящего "
                    "GPT-2 байт «!» имеет id 0, а пробел «Ġ» — 220). Бери id из vocab.json по строке токена.",
                    pytrace=False)
    assert not bad, f"to_gpt2[{bad[0]}] = {to_gpt2.get(bad[0])}, а в vocab.json у этого токена id {toy['to_gpt2'][bad[0]]}"


def test_encode_matches_oracle(impl, toy, text):
    """encode: id из vocab.json, по кускам пре-токенизации, спецтокен — одним id"""
    tok = impl.GPT2Tokenizer(*_paths(toy["dir"]))
    for s in SAMPLES + [text[:2000]]:
        got, want = tok.encode(s), _oracle(toy, s)
        if got == encode_with_special(s, toy["merges"], {EOT: -1}):
            pytest.fail("encode вернул свои id, а не id GPT-2: переведи каждый через self.to_gpt2.", pytrace=False)
        assert got == want, f"encode({s[:30]!r}) не совпадает с ожидаемым: {got[:10]} против {want[:10]}"
    s = f"a{EOT}b"
    assert tok.encode(s, allowed_special=False) == _oracle(toy, s, allowed_special=False), (
        "allowed_special=False: «<|endoftext|>» кодируется как обычный текст (так безопаснее для текста от "
        "пользователя: он не сможет вставить служебный токен)")
    assert toy["special"][EOT] not in tok.encode(s, allowed_special=False), (
        "при allowed_special=False id спецтокена не должен появиться в выходе")


def test_encode_respects_chunks(impl, tmp_path):
    """encode: слияния не выходят за границы кусков, даже если правило в merges.txt это позволяет"""
    enc = bytes_to_unicode()
    vocab_json = {enc[b]: b for b in range(256)} | {"aĠ": 256, EOT: 257}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab_json, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "merges.txt").write_text("#version: 0.2\na Ġ\n", encoding="utf-8")
    tok = impl.GPT2Tokenizer(*_paths(tmp_path))
    got = tok.encode("a b")
    if got == [256, 98]:
        pytest.fail("«a b» → [256, 98]: правило «a Ġ» склеило «a» с пробелом из следующего куска. Текст сначала "
                    "режется pretokenize («a», « b»), и BPE работает внутри каждого куска.", pytrace=False)
    assert got == [97, 32, 98], f"куски «a», « b» → [97, 32, 98], получено {got}"


def test_decode_roundtrip(impl, toy, text):
    """decode(encode(s)) == s, спецтокены и эмодзи (символы, разрезанные между токенами) — целыми"""
    tok = impl.GPT2Tokenizer(*_paths(toy["dir"]))
    for s in SAMPLES + [text[:2000]]:
        got = tok.decode(tok.encode(s))
        assert got == s, f"decode(encode(s)) != s для {s[:30]!r}: получено {got[:30]!r}"


# ---------------------------------------------------------------------------
# Настоящие файлы GPT-2 (--weights)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_files():
    hub = pytest.importorskip("huggingface_hub", reason="нужен huggingface_hub: uv sync --extra weights")
    get = lambda name: Path(hub.hf_hub_download("openai-community/gpt2", name))  # noqa: E731
    return get("vocab.json"), get("merges.txt"), get("tokenizer.json")


@pytest.mark.weights
def test_gpt2_sizes(impl, gpt2_files):
    """GPT-2: 50 000 слияний + 256 байтов + <|endoftext|> (id 50 256) = 50 257 токенов; «Hello world» → [15496, 995]"""
    tok = impl.GPT2Tokenizer(gpt2_files[0], gpt2_files[1])
    assert len(tok.merges) == 50_000 and tok.special == {EOT: 50256} and len(tok.to_gpt2) == 50_256, (
        f"ждём 50 000 слияний, special = {{EOT: 50256}} и 50 256 обычных токенов; получено {len(tok.merges)}, "
        f"{tok.special}, {len(tok.to_gpt2)}: проверь чтение merges.txt (первая строка — версия) и vocab.json")
    assert tok.encode("Hello world") == [15496, 995], (
        f"«Hello world» в GPT-2 — [15496, 995] («Hello», « world»), получено {tok.encode('Hello world')}")


@pytest.mark.weights
def test_gpt2_matches_tokenizers(impl, gpt2_files, text):
    """Свой токенизатор == библиотека tokenizers на корпусе, коде, пробелах, эмодзи и спецтокене"""
    tokenizers = pytest.importorskip("tokenizers", reason="нужен tokenizers: uv sync --extra weights")
    hf = tokenizers.Tokenizer.from_file(str(gpt2_files[2]))
    tok = impl.GPT2Tokenizer(gpt2_files[0], gpt2_files[1])
    for s in SAMPLES + [text, "a\n\nb", "\t\t\n \n", "x" * 50]:
        got, want = tok.encode(s), hf.encode(s).ids
        k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
        assert got == want, (f"на {s[:30]!r} расхождение с tokenizers"
                             + (f" в позиции {k}: {got[k]} против {want[k]}" if k is not None else
                                f": длина {len(got)} против {len(want)}"))
        assert tok.decode(got) == s, f"decode(encode(s)) должен вернуть исходную строку {s[:30]!r}"
