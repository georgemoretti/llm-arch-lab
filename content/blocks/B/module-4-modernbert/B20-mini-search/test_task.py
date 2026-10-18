"""Тесты проекта «Мини-поиск по курсу».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Полное обучение (~1.5 мин) — вне тестов: python solution.py. Здесь — части по отдельности, шаг обучения == эталон и
короткое обучение на 16 абзацах снимка корпуса.
"""

import math
import random

import pytest
import torch

WORDS = "a b c d e f g h i j k l m n o p q r s t".split()


def test_ict_pair(impl):
    """Запрос — 6 … 14 слов подряд; контекст — абзац без запроса в ~90 % случаев; окно может стоять где угодно"""
    rng = random.Random(0)
    lengths, removed, starts = set(), 0, set()
    for _ in range(2000):
        q, ctx = impl.ict_pair(WORDS, rng)
        n = len(q)
        assert 6 <= n <= 14, f"длина запроса {n} вне [6, 14]"
        lengths.add(n)
        s = next((i for i in range(len(WORDS) - n + 1) if WORDS[i:i + n] == q), None)
        assert s is not None, f"запрос — слова абзаца подряд, а получено {q}"
        starts.add((n, s))
        if ctx == WORDS[:s] + WORDS[s + n:]:
            removed += 1
        else:
            assert ctx == WORDS, ("контекст — либо абзац без запроса (words[:s] + words[s + n:]), либо весь абзац; "
                                  f"получено {ctx}")
    assert lengths == set(range(6, 15)), f"встретились не все длины 6 … 14: {sorted(lengths)}"
    assert (6, 0) in starts and (6, 14) in starts, (
        "окно из 6 слов в абзаце из 20 должно начинаться и на позиции 0, и на последней возможной, 14: "
        "start = rng.randrange(len − n + 1) (randrange не включает правую границу).")
    assert 0.86 < removed / 2000 < 0.94, (
        f"кусок вырезан в {removed / 20:.1f} % случаев, ждём ~90 % (remove_prob = 0.9). Если всегда вырезать, модель "
        "учится только на контексте; если никогда — учится искать точное совпадение слов.")
    q, ctx = impl.ict_pair(WORDS[:4], random.Random(1))
    assert q == WORDS[:4], f"абзац короче min_len: запрос — весь абзац, получено {q}"
    a = impl.ict_pair(WORDS, random.Random(5))
    assert a == impl.ict_pair(WORDS, random.Random(5)), "одинаковый rng → одинаковая пара (вся случайность — из rng)"


def test_idf_weights(impl):
    """idf = log((1 + N)/(1 + df)) + 1; повтор термина в документе не увеличивает df; термины вне словаря пропускаются"""
    docs = [["rope", "rope", "attention"], ["attention", "mask"], ["attention", "zzz"]]
    vocab = {"attention": 0, "mask": 1, "rope": 2, "unused": 3}
    idf = impl.idf_weights(docs, vocab)
    want = torch.tensor([math.log(4 / 4) + 1, math.log(4 / 2) + 1, math.log(4 / 2) + 1, math.log(4 / 1) + 1])
    if torch.allclose(idf, torch.tensor([1.0, math.log(2) + 1, math.log(4 / 3) + 1, math.log(4) + 1]), atol=1e-5):
        pytest.fail("df термина «rope» = 2: повторы внутри документа посчитаны. df — число ДОКУМЕНТОВ с термином "
                    "(set по документу).", pytrace=False)
    torch.testing.assert_close(idf, want, atol=1e-5, rtol=1e-5,
                               msg="attention во всех 3 → 1.0; mask и rope в одном → log 2 + 1; unused нигде → log 4 + 1")


def test_tfidf_vectors(impl):
    """tf × idf, L2 по строке; пустой текст — нулевая строка; запрос ближе всего к своему документу"""
    vocab = {"attention": 0, "mask": 1, "rope": 2}
    idf = torch.tensor([1.0, 2.0, 3.0])
    M = impl.tfidf_vectors([["rope", "rope", "attention", "zzz"], [], ["mask"]], vocab, idf)
    assert M.shape == (3, 3), f"[N, V] = [3, 3], получено {tuple(M.shape)}"
    row = torch.tensor([1.0, 0.0, 6.0])
    if torch.allclose(M[0], torch.tensor([1.0, 0.0, 3.0]) / math.sqrt(10), atol=1e-5):
        pytest.fail("повтор термина не учтён: tf — число вхождений («rope» дважды → 2 · idf).", pytrace=False)
    torch.testing.assert_close(M[0], row / row.norm(), atol=1e-5, rtol=1e-5, msg="(tf · idf) / ‖tf · idf‖")
    assert torch.isfinite(M).all() and (M[1] == 0).all(), "пустой текст — нулевая строка без nan"
    torch.testing.assert_close(M[2], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0, msg="один термин → орт")


def test_hit_at_k(impl):
    """hit@k с несколькими правильными документами: попадание, если хоть один из gold в top-k"""
    scores = torch.tensor([[0.9, 0.8, 0.1, 0.0],
                           [0.1, 0.2, 0.3, 0.9],
                           [0.5, 0.4, 0.3, 0.2]])
    gold = [{1, 2}, {0}, {3, 0}]
    assert abs(impl.hit_at_k(scores, gold, 1) - 1 / 3) < 1e-9, "k = 1: попал только запрос 2 (лучший — документ 0)"
    assert abs(impl.hit_at_k(scores, gold, 2) - 2 / 3) < 1e-9, "k = 2: запрос 0 (документ 1 второй) и запрос 2"
    assert impl.hit_at_k(scores, gold, 99) == 1.0, "k больше числа документов → все попали"


def _batch(stoi, impl, texts, seed=0):
    rng = random.Random(seed)
    pairs = [impl.ict_pair(t.split(), rng) for t in texts]
    q = impl.collate([impl.encode(" ".join(a), stoi, 32) for a, _ in pairs])
    d = impl.collate([impl.encode(" ".join(b), stoi, 64) for _, b in pairs])
    return q, d


def _paragraphs(ref, n=16):
    paras = [p["text"] for p in ref.load_corpus()["paragraphs"]]
    return random.Random(3).sample(paras, n)


def test_train_step_matches_reference(impl, ref):
    """3 шага train_step == эталон: тот же лосс и те же веса (InfoNCE, порядок zero_grad → backward → step)"""
    paras = _paragraphs(ref)
    stoi = ref.build_vocab(paras, min_count=1)
    a, b = ref.make_model(len(stoi)), impl.make_model(len(stoi))
    b.load_state_dict(a.state_dict())
    oa, ob = (torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01) for m in (a, b))
    for step in range(3):
        q, d = _batch(stoi, ref, paras, seed=step)
        la, lb = ref.train_step(a, oa, *q, *d), impl.train_step(b, ob, *q, *d)
        assert isinstance(lb, float), f"train_step возвращает лосс как float, получено {type(lb).__name__}"
        assert abs(la - lb) < 1e-5, (
            f"шаг {step}: лосс {lb:.5f}, у эталона {la:.5f}. Лосс — info_nce(эмбеддинги запросов, эмбеддинги "
            "документов, temperature): запросы — строки, документы — столбцы.")
    for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-5), (
            f"после 3 шагов веса {n} разошлись с эталоном: zero_grad → backward → step, ровно один шаг на вызов.")


def test_short_training_finds_fragments(impl, ref):
    """40 шагов на 16 абзацах: фрагмент из 8 слов находит свой абзац (hit@1 ≥ 0.75)"""
    paras = _paragraphs(ref)
    stoi = ref.build_vocab(paras, min_count=1)
    model = impl.make_model(len(stoi))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    for step in range(40):
        q, d = _batch(stoi, impl, paras, seed=100 + step)
        impl.train_step(model, opt, *q, *d)
    rng = random.Random(7)
    frags = []
    for t in paras:
        w = t.split()
        s = rng.randrange(max(1, len(w) - 8))
        frags.append(" ".join(w[s:s + 8]))
    D = impl.embed(model, paras, stoi, 64)
    Q = impl.embed(model, frags, stoi, 32)
    hit = impl.hit_at_k(Q @ D.T, [{i} for i in range(len(paras))], 1)
    assert hit >= 0.75, (
        f"после 40 шагов фрагмент находит свой абзац среди 16 в {hit:.0%} случаев, ждём ≥ 75 %. Проверь ict_pair "
        "(запрос — кусок СВОЕГО абзаца) и train_step.")
