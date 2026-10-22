"""Тесты «Найди баг: QA» — те же, что у задачи «Головы дообучения», без подсказок про баг.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракулы: transformers BertForSequenceClassification (если установлен), F.cross_entropy, перебор всех пар (i, j).
"""

import pytest
import torch
import torch.nn.functional as F

V, H, PAD, CLS, SEP = 60, 32, 0, 2, 3


def _cfg(mod):
    return mod.BertConfig(vocab_size=V, hidden_size=H, num_hidden_layers=2, num_attention_heads=4,
                          intermediate_size=64, max_position_embeddings=16)


def _batch():
    ids = torch.tensor([[CLS, 10, 11, 12, SEP, PAD, PAD], [CLS, 13, 14, SEP, 20, 21, SEP]])
    return ids, torch.tensor([[0] * 7, [0, 0, 0, 0, 1, 1, 1]]), (ids != PAD).long()


def test_classifier(impl, ref):
    """SequenceClassifier: pooled → dropout → classifier, логиты [B, num_labels]"""
    torch.manual_seed(0)
    a = ref.SequenceClassifier(_cfg(ref), 3)
    b = impl.SequenceClassifier(_cfg(impl), 3)
    b.load_state_dict(a.state_dict())
    ids, types, am = _batch()
    out = b(ids, types, am)
    assert out.shape == (2, 3), f"логиты [B, num_labels] = [2, 3], получено {tuple(out.shape)}"
    seq, pooled = b.bert(ids, types, am)
    if torch.allclose(out, b.classifier(seq[:, 0]), atol=1e-5):
        pytest.fail("классификатор взят от сырого h[CLS]. BertForSequenceClassification использует pooled-выход: "
                    "tanh(pooler.dense(h[CLS])).", pytrace=False)
    if torch.allclose(out, b.classifier(seq.mean(1)), atol=1e-5):
        pytest.fail("классификатор взят от среднего по токенам. Здесь — как в HF: pooled-выход [CLS].", pytrace=False)
    torch.testing.assert_close(out, a(ids, types, am), atol=1e-5, rtol=1e-4, msg="classifier(dropout(pooled))")


def test_classifier_matches_transformers(impl):
    """state_dict transformers.BertForSequenceClassification загружается без переименований, логиты совпадают"""
    transformers = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    hc = transformers.BertConfig(vocab_size=V, hidden_size=H, num_hidden_layers=2, num_attention_heads=4,
                                 intermediate_size=64, max_position_embeddings=16, hidden_dropout_prob=0.0,
                                 attention_probs_dropout_prob=0.0, num_labels=3,
                                 id2label={0: "a", 1: "b", 2: "c"}, label2id={"a": 0, "b": 1, "c": 2})
    torch.manual_seed(1)
    hf = transformers.BertForSequenceClassification(hc).eval()
    ours = impl.SequenceClassifier(_cfg(impl), 3)
    ours.load_state_dict(hf.state_dict(), strict=True)
    ids, types, am = _batch()
    with torch.no_grad():
        want = hf(input_ids=ids, token_type_ids=types, attention_mask=am).logits
    torch.testing.assert_close(ours(ids, types, am), want, atol=1e-5, rtol=1e-4,
                               msg="логиты должны совпасть с transformers BertForSequenceClassification")


def test_classifier_learns(impl):
    """Smoke test: маленький классификатор за 60 шагов отличает тексты с токеном 7 от текстов без него"""
    torch.manual_seed(0)
    model = impl.SequenceClassifier(_cfg(impl), 2)
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(10, V, (16, 8), generator=g)
    y = torch.arange(16) % 2
    ids[y == 1, torch.randint(1, 7, (8,), generator=g)] = 7
    ids[:, 0], ids[:, -1] = CLS, SEP
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(60):
        loss = F.cross_entropy(model(ids), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = (model(ids).argmax(-1) == y).float().mean().item()
    assert acc == 1.0, f"после обучения точность {acc:.0%} (лосс {loss.item():.3f}): градиент должен доходить до BERT"


def test_align_labels(impl):
    """align_labels: метка слова — на первом подтокене, −100 на остальных и на спецтокенах"""
    #            [CLS] ни   ##жний  новгород  —  столица    ##и   [SEP]
    word_ids = [None, 0,    0,      1,        2, 3,         3,    None]
    got = impl.align_labels(word_ids, [5, 6, 0, 1])
    want = [-100, 5, -100, 6, 0, 1, -100, -100]
    if got == [-100, 5, 5, 6, 0, 1, 1, -100]:
        pytest.fail("метка слова стоит на каждом подтокене. Обычно метку ставят только на первый: иначе длинные слова "
                    "весят в лоссе больше, а предсказание слова всё равно берут с первого подтокена.", pytrace=False)
    if len(got) == len(want) and got[0] != -100:
        pytest.fail("у спецтокенов ([CLS], [SEP]) word_id = None — им метка −100.", pytrace=False)
    assert got == want, f"получено {got}, ждём {want}"
    assert impl.align_labels([None, 0, 1, 1, 1, None, None], [3, 4]) == [-100, 3, 4, -100, -100, -100, -100], (
        "метка — только у первого подтокена слова; продолжения слова и спецтокены — −100")
    assert impl.align_labels([None, 0, None, 1, None], [7, 8]) == [-100, 7, -100, 8, -100], (
        "пара сегментов: слово 0 и слово 1 разделены спецтокеном — у каждого своя метка")


def test_qa_loss(impl):
    """qa_loss: (CE по началу + CE по концу) / 2"""
    g = torch.Generator().manual_seed(4)
    s, e = torch.randn(3, 9, generator=g, requires_grad=True), torch.randn(3, 9, generator=g)
    sp, ep = torch.tensor([2, 5, 0]), torch.tensor([4, 5, 0])
    got = impl.qa_loss(s, e, sp, ep)
    want = (F.cross_entropy(s, sp) + F.cross_entropy(e, ep)) / 2
    assert got.dim() == 0, "лосс — скаляр"
    if torch.allclose(got, 2 * want):
        pytest.fail("сумма двух cross-entropy без деления на 2: в BertForQuestionAnswering — среднее.", pytrace=False)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-5,
                               msg="(cross_entropy(start_logits, start) + cross_entropy(end_logits, end)) / 2")


def _brute(s, e, mask, L):
    best, arg = float("-inf"), (0, 0)
    for i in range(len(s)):
        for j in range(i, min(len(s), i + L)):
            if mask[i] and mask[j] and s[i] + e[j] > best:
                best, arg = s[i] + e[j], (i, j)
    return arg


def test_best_span_constraints(impl):
    """best_span: конец не раньше начала, длина ≤ max_answer_len, ответ только в контексте"""
    #                    [CLS] q   q  [SEP] c0   c1   c2   c3   [SEP]
    s = torch.tensor([[9.0, 8.0, 0, 0, 1.0, 0.5, 3.0, 0.2, 0], [0, 0, 0, 0, 2.0, 0, 0, 0, 0]])
    e = torch.tensor([[9.0, 0, 7.0, 0, 0.1, 2.5, 0.3, 0.4, 0], [0, 0, 0, 0, 0, 0, 0, 3.0, 0]])
    ctx = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 0]] * 2).bool()
    start, end = impl.best_span(s, e, ctx, max_answer_len=3)
    assert start.shape == (2,) and end.shape == (2,), f"(start [B], end [B]), получено {tuple(start.shape)}, {tuple(end.shape)}"
    got = list(zip(start.tolist(), end.tolist()))
    if got[0][1] < got[0][0]:
        pytest.fail(f"ответ {got[0]}: конец раньше начала.", pytrace=False)
    if got[0][0] < 4 or got[0][1] < 4:
        pytest.fail(f"ответ {got[0]} лежит вне контекста (в [CLS] или вопросе): начало и конец — только там, где "
                    "context_mask = True.", pytrace=False)
    assert got[0] == (4, 5), f"пример 0: лучшая допустимая пара (4, 5) с суммой 3.5; получено {got[0]}"
    if got[1] == (4, 7):
        pytest.fail("пример 1: ответ (4, 7) длиной 4 токена длиннее max_answer_len = 3.", pytrace=False)
    assert got[1] == (5, 7), f"пример 1: лучшая пара длиной ≤ 3 — (5, 7) с суммой 3.0; получено {got[1]}"

def test_best_span_random(impl):
    """На случайных логитах best_span совпадает с перебором всех пар"""
    g = torch.Generator().manual_seed(5)
    s, e = torch.randn(16, 20, generator=g), torch.randn(16, 20, generator=g)
    ctx = torch.zeros(16, 20, dtype=torch.bool)
    for b in range(16):
        q = int(torch.randint(2, 8, (1,), generator=g))
        ctx[b, q:19] = True
    start, end = impl.best_span(s, e, ctx, max_answer_len=5)
    for b in range(16):
        want = _brute(s[b].tolist(), e[b].tolist(), ctx[b].tolist(), 5)
        assert (start[b].item(), end[b].item()) == want, f"пример {b}: перебор даёт {want}, получено {(start[b].item(), end[b].item())}"
