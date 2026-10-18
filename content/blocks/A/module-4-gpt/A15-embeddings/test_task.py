"""Тесты «Эмбеддинги токенов и позиций, weight tying».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = Path(__file__).resolve().parents[4] / "data" / "corpus-ru.txt"
V, BLOCK, C = 67, 16, 32


def _idx(B=3, T=10, seed=0):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed))


def _emb(impl, ref):
    torch.manual_seed(0)
    a = ref.TokenPositionEmbedding(V, BLOCK, C)
    b = impl.TokenPositionEmbedding(V, BLOCK, C)
    b.load_state_dict(a.state_dict())
    return a, b


def test_embedding_sum(impl, ref):
    """forward: wte(idx) + wpe(0 … T − 1), форма [B, T, C]"""
    a, b = _emb(impl, ref)
    idx = _idx()
    out = b(idx)
    assert out.shape == (3, 10, C), f"[B, T] = [3, 10] → [B, T, C] = [3, 10, {C}], получено {tuple(out.shape)}"
    want = a.wte.weight[idx] + a.wpe.weight[:10]
    if torch.allclose(out, a.wte.weight[idx]):
        pytest.fail("позиции не добавлены: выход равен одному wte(idx). Без позиций модель не отличит «кот съел "
                    "мышь» от «мышь съела кот» — attention не знает порядка.", pytrace=False)
    if torch.allclose(out, a.wte.weight[idx] + a.wpe.weight[1:11]):
        pytest.fail("позиции начинаются с 1: первая позиция последовательности — номер 0 (строка wpe[0]).",
                    pytrace=False)
    torch.testing.assert_close(out, want, atol=1e-6, rtol=1e-6,
                               msg="выход должен быть wte(idx) + wpe(arange(T)): позиции t = 0 … T − 1")


def test_positions_shared_across_batch_and_shorter_T(impl, ref):
    """Один и тот же токен на разных позициях → разные векторы; позиции общие для всего батча; T < block_size"""
    _, b = _emb(impl, ref)
    idx = torch.full((2, 5), 7)
    out = b(idx)
    assert not torch.allclose(out[0, 0], out[0, 1]), "одинаковый токен на позициях 0 и 1 должен давать разные векторы"
    torch.testing.assert_close(out[0], out[1], msg="позиции одинаковы для каждого примера батча")
    torch.testing.assert_close(out[0, 3] - out[0, 0], b.wpe.weight[3] - b.wpe.weight[0])


def test_too_long_raises(impl, ref):
    """T > block_size → ValueError с понятным сообщением, T = block_size — можно"""
    _, b = _emb(impl, ref)
    assert b(_idx(T=BLOCK)).shape == (3, BLOCK, C), "T = block_size допустимо"
    with pytest.raises(ValueError):
        b(_idx(T=BLOCK + 1))


def test_tie_weights(impl):
    """tie_weights: один параметр в двух модулях, градиент приходит по обоим путям"""
    torch.manual_seed(0)
    wte, head = nn.Embedding(V, C), nn.Linear(C, V, bias=False)
    impl.tie_weights(head, wte)
    if head.weight is not wte.weight:
        if torch.equal(head.weight, wte.weight):
            pytest.fail("значения скопированы, но это два разных тензора: после первого шага обучения они "
                        "разойдутся. Нужно присвоить сам параметр: lm_head.weight = wte.weight.", pytrace=False)
        pytest.fail("lm_head.weight и wte.weight должны быть одним объектом nn.Parameter", pytrace=False)
    idx = torch.tensor([[1, 2, 3]])
    head(wte(idx)).sum().backward()
    grad = wte.weight.grad.clone()
    # тот же расчёт на двух независимых копиях: общий градиент = сумма градиентов двух путей
    w1 = wte.weight.detach().clone().requires_grad_()
    w2 = wte.weight.detach().clone().requires_grad_()
    (F.embedding(idx, w1) @ w2.T).sum().backward()
    torch.testing.assert_close(grad, w1.grad + w2.grad, msg="у связанного веса градиент — сумма вкладов входа и выхода")


def test_count_params(impl, ref):
    """count_params: связанный вес [V, C] считается один раз"""
    torch.manual_seed(0)
    model = ref.EmbeddingLM(V, BLOCK, C)
    got = impl.count_params(model)
    want = V * C + BLOCK * C
    if got == want + V * C:
        pytest.fail(f"получено {got}: связанный вес [V, C] посчитан дважды (как wte и как lm_head). В state_dict он "
                    "лежит под двумя ключами, а model.parameters() отдаёт каждый параметр один раз.", pytrace=False)
    assert got == want, f"V·C + block_size·C = {want}, получено {got}"
    assert impl.count_params(nn.Linear(4, 3)) == 15, "Linear(4, 3): 12 весов + 3 bias"


def _lm(impl, ref):
    torch.manual_seed(0)
    a = ref.EmbeddingLM(V, BLOCK, C)
    b = impl.EmbeddingLM(V, BLOCK, C)
    b.load_state_dict(a.state_dict())
    return a, b


def test_lm_forward_and_loss(impl, ref):
    """EmbeddingLM: логиты [B, T, V] = (wte + wpe) @ wteᵀ, loss — средняя CE по B·T позициям, без targets — None"""
    a, b = _lm(impl, ref)
    assert b.lm_head.weight is b.emb.wte.weight, "веса выходного слоя связаны с эмбеддингами (tie_weights)"
    idx, tgt = _idx(seed=1), _idx(seed=2)
    logits, loss = b(idx, tgt)
    assert logits.shape == (3, 10, V), f"логиты [B, T, V] = [3, 10, {V}], получено {tuple(logits.shape)}"
    want_logits, want_loss = a(idx, tgt)
    if torch.allclose(logits, want_logits.softmax(-1), atol=1e-6):
        pytest.fail("из forward вернулись вероятности: модель отдаёт логиты, softmax делает cross_entropy.",
                    pytrace=False)
    torch.testing.assert_close(logits, want_logits, atol=1e-6, rtol=1e-5, msg="логиты не совпадают с эталоном")
    assert loss is not None and loss.dim() == 0, "loss — скаляр"
    torch.testing.assert_close(loss, want_loss, atol=1e-6, rtol=1e-5,
                               msg="loss = F.cross_entropy(logits [B·T, V], targets [B·T]) — среднее по всем позициям")
    logits2, loss2 = b(idx)
    assert loss2 is None and torch.allclose(logits2, logits), "без targets loss = None, логиты те же"


def test_initial_loss_is_log_vocab(impl, ref):
    """При инициализации N(0, 0.02²) логиты почти нулевые → loss ≈ ln V: полезная проверка перед обучением"""
    _, b = _lm(impl, ref)
    _, loss = b(_idx(B=8, T=16, seed=3), _idx(B=8, T=16, seed=4))
    assert abs(loss.item() - math.log(V)) < 0.05, (
        f"стартовый loss {loss.item():.3f}, а у модели с почти равными логитами он ≈ ln V = {math.log(V):.3f}")


def test_learns_bigrams_on_corpus(impl):
    """300 шагов AdamW на корпусе: loss падает с ln V заметно ниже (модель выучила частоты пар символов)"""
    text = CORPUS.read_text(encoding="utf-8")
    itos = sorted(set(text))
    data = torch.tensor([itos.index(c) for c in text[:6000]])
    torch.manual_seed(0)
    model = impl.EmbeddingLM(len(itos), BLOCK, C)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-2)
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        i = torch.randint(0, len(data) - BLOCK - 1, (32,), generator=g)
        x = torch.stack([data[j:j + BLOCK] for j in i])
        y = torch.stack([data[j + 1:j + BLOCK + 1] for j in i])
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < math.log(len(itos)) - 1.0, (
        f"после 300 шагов loss {loss.item():.2f}, ждём меньше ln V − 1 = {math.log(len(itos)) - 1:.2f}")
