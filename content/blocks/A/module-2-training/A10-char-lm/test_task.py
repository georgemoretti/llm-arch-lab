"""Тесты «Языковая модель на символах».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Корпус — content/data/corpus-ru.txt (текст написан авторами курса, ~23 тыс. символов).
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

CORPUS = Path(__file__).resolve().parents[4] / "data" / "corpus-ru.txt"
CONTEXT = 4


@pytest.fixture(scope="module")
def text() -> str:
    return CORPUS.read_text(encoding="utf-8")


def test_vocab(impl, text):
    """build_vocab: уникальные символы по возрастанию, stoi и itos — взаимно обратные"""
    stoi, itos = impl.build_vocab(text)
    assert itos == sorted(set(text)), (
        "itos должен быть списком уникальных символов текста, отсортированным по возрастанию (sorted(set(text)))")
    assert len(itos) == 67, f"в корпусе 67 уникальный символ, а в словаре {len(itos)}"
    assert all(itos[stoi[c]] == c for c in itos) and len(stoi) == len(itos), "stoi[c] — позиция символа c в itos"


def test_encode_decode_roundtrip(impl, text):
    """encode → тензор long, decode(encode(s)) == s"""
    stoi, itos = impl.build_vocab(text)
    ids = impl.encode(text[:300], stoi)
    assert isinstance(ids, torch.Tensor) and ids.dtype == torch.long, (
        f"encode возвращает torch.long тензор, а получено {type(ids).__name__} {getattr(ids, 'dtype', '')}: "
        "nn.Embedding и cross_entropy принимают только целые индексы")
    assert ids.shape == (300,), f"300 символов → 300 id, форма [300]; получено {tuple(ids.shape)}"
    assert impl.decode(ids, itos) == text[:300], "decode(encode(s)) должен вернуть исходную строку"
    assert impl.decode(ids.tolist(), itos) == text[:300], "decode должен принимать и список индексов"


def test_make_dataset(impl):
    """make_dataset: X[i] = ids[i : i + k], Y[i] = ids[i + k], ровно N − k примеров"""
    ids = torch.arange(10)
    X, Y = impl.make_dataset(ids, 3)
    if X.shape[0] == 8:
        pytest.fail("получилось 8 окон из 10 символов при context = 3, а нужно 7: у последнего окна нет "
                    "следующего символа (ids[7:10] — последнее окно, за ним ничего нет).", pytrace=False)
    assert X.shape == (7, 3) and Y.shape == (7,), f"X [N − k, k] = [7, 3], Y [7]; получено {tuple(X.shape)}, {tuple(Y.shape)}"
    assert X[0].tolist() == [0, 1, 2] and Y[0].item() == 3, f"первый пример: окно [0, 1, 2] → 3, а получено {X[0].tolist()} → {Y[0].item()}"
    assert X[-1].tolist() == [6, 7, 8] and Y[-1].item() == 9, "последний пример: окно [6, 7, 8] → 9"
    assert torch.equal(X[1:, :-1], X[:-1, 1:]), "соседние окна сдвинуты ровно на один символ"


def _models(impl, ref, V=67):
    torch.manual_seed(0)
    a = ref.MLPLM(V, CONTEXT)
    b = impl.MLPLM(V, CONTEXT)
    b.load_state_dict(a.state_dict())
    return a, b


def test_forward_matches_reference(impl, ref):
    """MLPLM.forward: логиты [B, V], совпадают с эталоном при тех же весах"""
    a, b = _models(impl, ref)
    X = torch.randint(0, 67, (5, CONTEXT), generator=torch.Generator().manual_seed(1))
    out = b(X)
    assert out.shape == (5, 67), f"логиты [B, V] = [5, 67], а получено {tuple(out.shape)}"
    want = a(X)
    if not torch.allclose(out, want, atol=1e-6, rtol=1e-5):
        if torch.allclose(out.softmax(-1), want, atol=1e-6) or torch.allclose(out, want.softmax(-1), atol=1e-6):
            pytest.fail("в forward применён softmax: модель возвращает логиты, а softmax делает cross_entropy.", pytrace=False)
        e = a.emb(X)
        wrong = a.out(torch.tanh(a.hidden(e.transpose(1, 2).flatten(1))))
        if torch.allclose(out, wrong, atol=1e-6):
            pytest.fail("эмбеддинги склеены не в том порядке: [B, context, E] нужно сплющить как есть (flatten(1) или "
                        "view(B, −1)), без транспонирования — сначала все E чисел первого символа окна.", pytrace=False)
        pytest.fail("логиты не совпадают с эталоном: emb → склейка [B, context·E] → tanh(hidden) → out.", pytrace=False)


def test_perplexity_of_uniform_model_is_vocab_size(impl, ref):
    """Перплексия: у модели с одинаковыми логитами она ровно V; режим модели восстанавливается"""
    _, b = _models(impl, ref)
    with torch.no_grad():
        b.out.weight.zero_()
        b.out.bias.zero_()
    X = torch.randint(0, 67, (20, CONTEXT), generator=torch.Generator().manual_seed(2))
    Y = torch.randint(0, 67, (20,), generator=torch.Generator().manual_seed(3))
    b.train()
    ppl = impl.perplexity(b, X, Y)
    assert isinstance(ppl, float), f"perplexity возвращает float, а не {type(ppl).__name__}"
    if math.isclose(ppl, math.log(67), rel_tol=1e-5):
        pytest.fail("вернулась cross-entropy (log V), а перплексия — это exp от неё.", pytrace=False)
    assert math.isclose(ppl, 67.0, rel_tol=1e-5), (
        f"модель, которая всем символам даёт равные логиты, «колеблется» между всеми V = 67 вариантами — "
        f"перплексия должна быть 67, а получено {ppl:.3f}")
    assert b.training, "perplexity должна вернуть модель в прежний режим (train)"


def _bigram_ppl(train_ids, val_ids, V):
    """Независимый ориентир: биграммная модель по частотам (+1 сглаживание)."""
    counts = torch.ones(V, V)
    counts.index_put_((train_ids[:-1], train_ids[1:]), torch.ones(len(train_ids) - 1), accumulate=True)
    P = counts / counts.sum(1, keepdim=True)
    return math.exp(-P[val_ids[:-1], val_ids[1:]].log().mean().item())


def test_mlp_lm_beats_bigram(impl, text):
    """После 400 шагов обучения MLP-модель с окном 4 предсказывает отложенный текст лучше биграммной"""
    stoi, itos = impl.build_vocab(text)
    ids = impl.encode(text, stoi)
    n = int(0.9 * len(ids))
    Xtr, Ytr = impl.make_dataset(ids[:n], CONTEXT)
    Xva, Yva = impl.make_dataset(ids[n:], CONTEXT)
    torch.manual_seed(0)
    model = impl.MLPLM(len(itos), CONTEXT)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    for _ in range(400):
        idx = torch.randint(0, len(Xtr), (128,), generator=g)
        loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    ppl = impl.perplexity(model, Xva, Yva)
    bigram = _bigram_ppl(ids[:n], ids[n:], len(itos))
    assert ppl < bigram * 0.9, (
        f"перплексия MLP на отложенном тексте {ppl:.2f}, у биграммной модели {bigram:.2f}: модель, видящая 4 символа, "
        "должна быть заметно лучше той, что видит один. Проверь make_dataset (X и Y не сдвинуты?) и forward.")
    sample = impl.decode(impl.generate(model, ids[:CONTEXT], 60, torch.Generator().manual_seed(4)), itos)
    assert len(sample) == CONTEXT + 60 and set(sample) <= set(itos), "generate + decode должны дать строку из символов словаря"
