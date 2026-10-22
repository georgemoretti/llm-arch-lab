"""Тесты «Вход BERT».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракулы — tokenizers (обрезка пары) и transformers BertEmbeddings (если установлены).
"""

import pytest
import torch

CLS, SEP, PAD = 2, 3, 0


def _loop_truncate(a, b, n):
    a, b = list(a), list(b)
    while len(a) + len(b) > n:
        (a if len(a) > len(b) else b).pop()
    return a, b


def test_truncate_pair_examples(impl):
    """truncate_pair: короткий сегмент цел, если ≤ половины бюджета; иначе бюджет пополам"""
    a, b = list(range(10, 20)), list(range(30, 33))
    got = impl.truncate_pair(a, b, 8)
    assert got == (a[:5], b), f"a из 10, b из 3, бюджет 8: b цел (3 ≤ 4), a обрезан до 5; получено {got}"
    assert a == list(range(10, 20)), "truncate_pair не должен менять входные списки"
    assert impl.truncate_pair(a[:3], b, 8) == (a[:3], b), "влезает — ничего не режем"
    got = impl.truncate_pair(a[:5], b, 8)
    assert got == (a[:5], b), f"5 + 3 = 8 ровно по бюджету — тоже влезает, ничего не режем (≤, а не <); получено {got}"
    got = impl.truncate_pair(a[:7], list(range(30, 36)), 9)
    assert got == (a[:5], list(range(30, 34))), f"a 7, b 6, бюджет 9: половина 4 → b 4, a 5; получено {got}"
    got = impl.truncate_pair(a[:4], list(range(30, 34)), 5)
    if got == _loop_truncate(a[:4], list(range(30, 34)), 5):
        pytest.fail("обрезка «по одному токену у более длинного» — так делал исходный код BERT, но быстрые токенизаторы "
                    "Hugging Face делят бюджет иначе. Здесь ждём их правило: при равных длинах длинным считается b; "
                    "оба сегмента длиннее половины бюджета → короткому max_tokens // 2, длинному — остаток.",
                    pytrace=False)
    assert got == (a[:2], list(range(30, 33))), (
        f"равные длины 4 и 4, бюджет 5: длинным считается b, он получает 3, a — 2; получено {got}")

def test_truncate_pair_matches_tokenizers(impl):
    """Для всех длин 0–13 и бюджетов 3–10 обрезка совпадает с tokenizers (strategy = longest_first)"""
    tok = pytest.importorskip("tokenizers", reason="нужен tokenizers: uv sync --extra weights")
    vocab = {"[UNK]": 0, "[CLS]": CLS, "[SEP]": SEP, **{f"w{i}": 10 + i for i in range(40)}}
    t = tok.Tokenizer(tok.models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = tok.pre_tokenizers.WhitespaceSplit()
    for n in range(3, 11):
        t.enable_truncation(max_length=n, strategy="longest_first")
        for la in range(14):
            for lb in range(14):
                a, b = [f"w{i}" for i in range(la)], [f"w{20 + i}" for i in range(lb)]
                enc = t.encode(" ".join(a), " ".join(b), add_special_tokens=False)
                want_a = [x for x, s in zip(enc.ids, enc.sequence_ids) if s == 0]
                want_b = [x for x, s in zip(enc.ids, enc.sequence_ids) if s == 1]
                got = impl.truncate_pair([vocab[x] for x in a], [vocab[x] for x in b], n)
                assert got == (want_a, want_b), f"len(a) = {la}, len(b) = {lb}, бюджет {n}: ждём длины " \
                                                f"{len(want_a)} и {len(want_b)}, получено {len(got[0])} и {len(got[1])}"


def test_build_inputs(impl):
    """build_inputs: [CLS] a [SEP] (b [SEP]), типы 0/1, длина ≤ max_len"""
    ids, types = impl.build_inputs([10, 11, 12], None, CLS, SEP, 16)
    assert ids == [CLS, 10, 11, 12, SEP] and types == [0] * 5, f"один сегмент: [CLS] a [SEP], типы нули; получено {ids}, {types}"
    ids, types = impl.build_inputs([10, 11], [20, 21, 22], CLS, SEP, 16)
    assert ids == [CLS, 10, 11, SEP, 20, 21, 22, SEP], f"пара: [CLS] a [SEP] b [SEP]; получено {ids}"
    if types == [0, 0, 0, 0, 1, 1, 1, 0]:
        pytest.fail("последний [SEP] относится ко второму сегменту: его token_type = 1.", pytrace=False)
    if types == [0, 0, 0, 1, 1, 1, 1, 1]:
        pytest.fail("первый [SEP] закрывает первый сегмент: его token_type = 0.", pytrace=False)
    assert types == [0, 0, 0, 0, 1, 1, 1, 1], f"типы: 0 у [CLS] a [SEP], 1 у b [SEP]; получено {types}"
    ids, types = impl.build_inputs(list(range(10, 30)), list(range(40, 50)), CLS, SEP, 12)
    assert len(ids) == 12 and len(types) == 12, f"длина ровно max_len = 12 (три служебных токена + 9), получено {len(ids)}"
    assert ids.count(SEP) == 2 and ids[0] == CLS and ids[-1] == SEP, "служебные токены не обрезаются"
    ids, _ = impl.build_inputs(list(range(10, 30)), None, CLS, SEP, 8)
    assert ids == [CLS] + list(range(10, 16)) + [SEP], f"один сегмент обрезается до max_len − 2; получено {ids}"


def test_pad_batch(impl):
    """pad_batch: input_ids, token_type_ids, attention_mask [B, T_max], long, PAD справа"""
    ex = [([CLS, 10, SEP, 20, SEP], [0, 0, 0, 1, 1]), ([CLS, 11, 12, SEP], [0, 0, 0, 0]), ([CLS, SEP], [0, 0])]
    out = impl.pad_batch(ex, PAD)
    assert set(out) >= {"input_ids", "token_type_ids", "attention_mask"}, f"ключи: input_ids, token_type_ids, attention_mask; есть {sorted(out)}"
    for k in ("input_ids", "token_type_ids", "attention_mask"):
        assert out[k].shape == (3, 5) and out[k].dtype == torch.long, f"{k}: long [3, 5], получено {out[k].dtype} {tuple(out[k].shape)}"
    assert out["input_ids"].tolist() == [[CLS, 10, SEP, 20, SEP], [CLS, 11, 12, SEP, PAD], [CLS, SEP, PAD, PAD, PAD]], (
        f"PAD добавляется справа до самого длинного примера; получено {out['input_ids'].tolist()}")
    assert out["token_type_ids"].tolist() == [[0, 0, 0, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], "на PAD тип 0"
    assert out["attention_mask"].tolist() == [[1] * 5, [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], "1 у настоящих токенов, 0 у PAD"


def _cfg(ref):
    return ref.BertConfig(vocab_size=40, hidden_size=16, max_position_embeddings=12, type_vocab_size=2)


def _emb_pair(impl, ref):
    torch.manual_seed(0)
    a = ref.BertEmbeddings(_cfg(ref))
    with torch.no_grad():
        a.LayerNorm.weight.uniform_(0.5, 1.5)
        a.LayerNorm.bias.uniform_(-0.3, 0.3)
    b = impl.BertEmbeddings(impl.BertConfig(vocab_size=40, hidden_size=16, max_position_embeddings=12, type_vocab_size=2))
    b.load_state_dict(a.state_dict())
    return a, b


def test_embeddings_match_reference(impl, ref):
    """BertEmbeddings: LayerNorm(word + position + token_type)"""
    a, b = _emb_pair(impl, ref)
    ids = torch.tensor([[CLS, 10, SEP, 20, SEP], [CLS, 11, 12, SEP, PAD]])
    types = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 0, 0]])
    out = b(ids, types)
    assert out.shape == (2, 5, 16), f"выход [B, T, H] = [2, 5, 16], получено {tuple(out.shape)}"
    no_ln = b.word_embeddings(ids) + b.position_embeddings(torch.arange(5)) + b.token_type_embeddings(types)
    if torch.allclose(out, no_ln, atol=1e-5):
        pytest.fail("нет LayerNorm после суммы эмбеддингов: в BERT сумма сразу нормируется.", pytrace=False)
    if torch.allclose(out, b.LayerNorm(b.word_embeddings(ids) + b.position_embeddings(torch.arange(5))), atol=1e-5):
        pytest.fail("не прибавлен эмбеддинг сегмента (token_type_embeddings).", pytrace=False)
    torch.testing.assert_close(out, a(ids, types), atol=1e-5, rtol=1e-4,
                               msg="LayerNorm(word_embeddings + position_embeddings(0 … T−1) + token_type_embeddings)")
    torch.testing.assert_close(b(ids), a(ids, torch.zeros_like(ids)), atol=1e-5, rtol=1e-4,
                               msg="без token_type_ids — все сегменты 0")


def test_embeddings_match_transformers(impl, ref):
    """Веса из transformers.BertEmbeddings загружаются без переименований, выход совпадает"""
    transformers = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    from transformers.models.bert.modeling_bert import BertEmbeddings as HFEmbeddings
    hc = transformers.BertConfig(vocab_size=40, hidden_size=16, max_position_embeddings=12, type_vocab_size=2,
                                 hidden_dropout_prob=0.0)
    torch.manual_seed(1)
    hf = HFEmbeddings(hc).eval()
    ours = impl.BertEmbeddings(impl.BertConfig(vocab_size=40, hidden_size=16, max_position_embeddings=12))
    state = {k: v for k, v in hf.state_dict().items() if k in dict(ours.state_dict())}
    ours.load_state_dict(state, strict=True)
    ids = torch.randint(4, 40, (3, 7))
    types = (torch.arange(7) >= 4).long().expand(3, 7)
    torch.testing.assert_close(ours(ids, types), hf(input_ids=ids, token_type_ids=types), atol=1e-5, rtol=1e-4,
                               msg="выход должен совпасть с transformers BertEmbeddings")
