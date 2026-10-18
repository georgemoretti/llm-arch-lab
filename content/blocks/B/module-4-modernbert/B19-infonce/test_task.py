"""Тесты «InfoNCE».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter).

Оракулы — ручная формула −log softmax, градиент (p − onehot)·d/(τN) и игрушечный bi-encoder, который учится находить
документ по запросу из «синонимов» его слов (общих токенов у запроса и документа нет).
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _unit(*shape, seed=0):
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(seed))
    return x / x.norm(dim=-1, keepdim=True)


def _manual(q, d, tau):
    s = q @ d.T / tau
    return float(sum(-(s[i, i] - torch.logsumexp(s[i], 0)) for i in range(q.shape[0])) / q.shape[0])


def test_info_nce_formula(impl):
    """L = −1/N · Σ log softmax(q_i · dᵀ / τ)_i — CE по строкам, правильный документ i"""
    q, d = _unit(6, 8, seed=0), _unit(6, 8, seed=1)
    loss = impl.info_nce(q, d, temperature=0.1)
    assert loss.dim() == 0, f"лосс — скаляр, получена форма {tuple(loss.shape)}"
    want = _manual(q, d, 0.1)
    s = q @ d.T
    variants = {
        "температура не применена: logits = q · dᵀ без деления на τ. Косинусы лежат в [−1, 1], softmax по ним почти "
        "равномерный, и сигнал обучения слабый.": F.cross_entropy(s, torch.arange(6)),
        "logits умножены на τ, а не поделены: τ = 0.05 должна РАСТЯГИВАТЬ косинусы в 20 раз.":
            F.cross_entropy(s * 0.1, torch.arange(6)),
        "softmax по столбцам (logits.T): это направление «документ → запросы». Обычный InfoNCE для поиска — по "
        "строкам, запрос выбирает свой документ.": F.cross_entropy(s.T / 0.1, torch.arange(6)),
        "сумма вместо среднего по запросам: лосс растёт с размером батча.":
            F.cross_entropy(s / 0.1, torch.arange(6), reduction="sum"),
    }
    for msg, alt in variants.items():
        if abs(float(loss) - float(alt)) < 1e-5 and abs(float(alt) - want) > 1e-4:
            pytest.fail(msg, pytrace=False)
    assert abs(float(loss) - want) < 1e-5, f"−mean_i log softmax(q_i dᵀ/τ)_i = {want:.5f}, получено {float(loss):.5f}"


def test_info_nce_gradient(impl):
    """∂L/∂q_i = (p_i − onehot_i) · d / (τ N): позитив тянет к себе, негативы — отталкивают по своей вероятности"""
    q, d = _unit(5, 8, seed=2).requires_grad_(), _unit(5, 8, seed=3)
    impl.info_nce(q, d, temperature=0.2).backward()
    p = (q.detach() @ d.T / 0.2).softmax(-1)
    want = (p - torch.eye(5)) @ d / (0.2 * 5)
    torch.testing.assert_close(q.grad, want, atol=1e-5, rtol=1e-4, msg="градиент InfoNCE по запросам")


def test_info_nce_bounds_and_temperature(impl):
    """Все косинусы равны → L = log M; идеальная пара и малая τ → L → 0; меньше τ — сильнее штраф за трудный негатив"""
    q = torch.ones(4, 3) / math.sqrt(3)
    loss = float(impl.info_nce(q, q.clone(), temperature=0.05))
    assert abs(loss - math.log(4)) < 1e-5, f"все косинусы одинаковы → каждый документ равновероятен, L = log 4, получено {loss}"
    e = torch.eye(4)
    assert float(impl.info_nce(e, e, temperature=0.01)) < 1e-6, "q_i = d_i, остальные ортогональны, τ = 0.01 → L ≈ 0"
    q2 = torch.tensor([[1.0, 0.0]])
    d2 = torch.tensor([[0.9, math.sqrt(1 - 0.81)], [0.8, -0.6]])     # позитив 0.9, негатив 0.8
    hot, cold = float(impl.info_nce(q2, d2, 1.0)), float(impl.info_nce(q2, d2, 0.05))
    assert cold < hot, (
        f"при τ = 0.05 разница 0.9 − 0.8 растягивается до 2.0 в logits, и позитив выигрывает увереннее: лосс должен "
        f"быть меньше, чем при τ = 1 ({cold:.3f} vs {hot:.3f})")


def test_hard_negatives(impl):
    """M > N: документы d[N:] — лишние негативы (hard negatives) для всех запросов; позитив запроса i — d[i]"""
    q, d = _unit(3, 8, seed=4), _unit(7, 8, seed=5)
    loss = impl.info_nce(q, d, temperature=0.1)
    assert abs(float(loss) - _manual(q, d, 0.1)) < 1e-5, (
        "q [3, C], d [7, C]: logits [3, 7], правильные классы 0, 1, 2 — первые N документов. Используй N = q.shape[0], "
        "а не d.shape[0].")
    more = impl.info_nce(q, torch.cat([d[:3], q + 0.0]), temperature=0.1)
    assert float(more) > float(impl.info_nce(q, d[:3], temperature=0.1)), (
        "добавили негативы, совпадающие с запросами (самые трудные): лосс должен вырасти")


def test_symmetric(impl):
    """Симметричный InfoNCE = (CE по строкам + CE по столбцам) / 2"""
    q, d = _unit(5, 8, seed=6), _unit(5, 8, seed=7)
    s = q @ d.T / 0.1
    rows = float(F.cross_entropy(s, torch.arange(5)))
    cols = float(F.cross_entropy(s.T, torch.arange(5)))
    got = float(impl.symmetric_info_nce(q, d, temperature=0.1))
    if abs(got - rows) < 1e-5:
        pytest.fail("получился обычный InfoNCE (только строки). Симметричный добавляет направление «документ → "
                    "запросы»: CE по logits.T, и берёт среднее.", pytrace=False)
    assert abs(got - (rows + cols) / 2) < 1e-5, f"(строки {rows:.4f} + столбцы {cols:.4f}) / 2, получено {got:.4f}"


def test_recall_at_k(impl):
    """recall@k — доля запросов, у которых правильный документ среди k лучших"""
    scores = torch.tensor([[0.9, 0.1, 0.5, 0.0],
                           [0.2, 0.3, 0.8, 0.1],
                           [0.1, 0.2, 0.3, 0.4]])
    pos = torch.tensor([0, 1, 3])
    r1, r2 = impl.recall_at_k(scores, pos, 1), impl.recall_at_k(scores, pos, 2)
    assert isinstance(r1, float), f"recall — float, получено {type(r1).__name__}"
    assert abs(r1 - 2 / 3) < 1e-6, (
        f"k = 1: запрос 0 — документ 0 лучший ✓, запрос 1 — лучший 2 ✗, запрос 2 — лучший 3 ✓ → 2/3, получено {r1}")
    assert abs(r2 - 1.0) < 1e-6, f"k = 2: у запроса 1 документ 1 второй → все три попали, 1.0, получено {r2}"
    assert impl.recall_at_k(scores, pos, 10) == 1.0, "k больше числа документов → 1.0 (не ошибка topk)"
    assert abs(impl.recall_at_k(-scores, pos, 1) - 0.0) < 1e-6, "чем больше score, тем лучше: topk по убыванию"


def test_training_toy_bi_encoder(impl):
    """60 шагов InfoNCE: мешок-эмбеддинг учится находить документ по «синонимам» половины его слов"""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    n_docs, vocab = 64, 200
    docs = torch.stack([torch.randperm(vocab, generator=g)[:12] for _ in range(n_docs)])     # 12 «слов» в документе
    syn = torch.randperm(vocab, generator=g) + vocab                 # у каждого слова свой «синоним» из другого словаря
    enc = nn.EmbeddingBag(2 * vocab, 32, mode="mean")
    embed = lambda ids: (lambda e: e / e.norm(dim=-1, keepdim=True))(enc(ids))
    opt = torch.optim.Adam(enc.parameters(), lr=0.05)

    def queries():
        return torch.stack([syn[docs[i, torch.randperm(12, generator=g)[:6]]] for i in range(n_docs)])

    with torch.no_grad():
        before = impl.recall_at_k(embed(queries()) @ embed(docs).T, torch.arange(n_docs), 1)
    for _ in range(60):
        idx = torch.randperm(n_docs, generator=g)[:16]
        loss = impl.info_nce(embed(queries()[idx]), embed(docs[idx]), temperature=0.05)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = impl.recall_at_k(embed(queries()) @ embed(docs).T, torch.arange(n_docs), 1)
    assert after >= 0.9 and after > before + 0.3, (
        f"recall@1 по 64 документам: до обучения {before:.2f}, после 60 шагов {after:.2f}; ждём ≥ 0.9. Лосс должен "
        "тянуть запрос к своему документу и отталкивать от остальных документов батча.")
