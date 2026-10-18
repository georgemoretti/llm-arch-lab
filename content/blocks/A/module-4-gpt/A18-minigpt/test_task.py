"""Тесты проекта «Мини-GPT на символах».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Тесты — на маленькой конфигурации (2 слоя, n_embd = 32) и коротком обучении. Полное обучение (~30 с на CPU)
запускается вручную: python workspace/A18-minigpt/solution.py
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

CORPUS = Path(__file__).resolve().parents[4] / "data" / "corpus-ru.txt"
V, BLOCK = 67, 16


def _cfg(mod, **kw):
    return mod.MiniGPTConfig(**{"vocab_size": V, "block_size": BLOCK, "n_layer": 2, "n_embd": 32, **kw})


def _models(impl, ref, **kw):
    torch.manual_seed(0)
    a = ref.MiniGPT(_cfg(ref, **kw))
    b = impl.MiniGPT(_cfg(impl, **kw))
    b.load_state_dict(a.state_dict())
    return a, b


def _idx(B=3, T=12, seed=0):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed))


@pytest.fixture(scope="module")
def data() -> torch.Tensor:
    text = CORPUS.read_text(encoding="utf-8")
    itos = sorted(set(text))
    stoi = {c: i for i, c in enumerate(itos)}
    return torch.tensor([stoi[c] for c in text])


def test_forward_matches_reference(impl, ref):
    """forward: логиты [B, T, V] и loss совпадают с эталоном (pre-LN и post-LN)"""
    for pre in (True, False):
        a, b = _models(impl, ref, pre_norm=pre)
        idx, tgt = _idx(), _idx(seed=1)
        logits, loss = b(idx, tgt)
        assert logits.shape == (3, 12, V), f"логиты [B, T, V] = [3, 12, {V}], получено {tuple(logits.shape)}"
        want_logits, want_loss = a(idx, tgt)
        if not torch.allclose(logits, want_logits, atol=1e-5, rtol=1e-4):
            with torch.no_grad():
                x = a.emb(idx)
                for blk in a.blocks:
                    x = blk(x)
                no_ln = a.lm_head(x)
            if torch.allclose(logits, no_ln, atol=1e-5):
                pytest.fail("пропущен финальный LayerNorm (ln_f) перед lm_head: в pre-LN residual-поток после "
                            "последнего блока не нормализован.", pytrace=False)
            pytest.fail("логиты не совпадают с эталоном: emb → блоки по порядку → ln_f → lm_head.", pytrace=False)
        torch.testing.assert_close(loss, want_loss, atol=1e-5, rtol=1e-4,
                                   msg="loss — cross_entropy по всем B·T позициям")
        assert b(idx)[1] is None, "без targets loss = None"


def test_initial_loss_is_log_vocab(impl, ref):
    """Стартовый loss ≈ ln V: при инициализации N(0, 0.02²) модель ещё ничего не предпочитает"""
    _, b = _models(impl, ref)
    _, loss = b(_idx(B=8, T=BLOCK, seed=2), _idx(B=8, T=BLOCK, seed=3))
    assert abs(loss.item() - math.log(V)) < 0.1, f"стартовый loss {loss.item():.3f}, ждём ≈ ln {V} = {math.log(V):.3f}"


def test_causal(impl, ref):
    """Логиты позиции t не зависят от токенов после t"""
    _, b = _models(impl, ref)
    idx = _idx()
    other = idx.clone()
    other[:, 7:] = _idx(seed=9)[:, 7:]
    assert torch.allclose(b(idx)[0][:, :7], b(other)[0][:, :7], atol=1e-5), (
        "логиты позиций 0–6 зависят от токенов 7+ → модель подсматривает ответ")


def test_get_batch(impl, data):
    """get_batch: x, y [B, T], y — x со сдвигом на 1, куски подряд из data, крайние начала достижимы"""
    g = torch.Generator().manual_seed(0)
    x, y = impl.get_batch(data, BLOCK, 8, g)
    assert x.shape == (8, BLOCK) and y.shape == (8, BLOCK), f"x, y — [8, {BLOCK}], получено {tuple(x.shape)}, {tuple(y.shape)}"
    if torch.equal(x, y):
        pytest.fail("y совпадает с x: цель позиции t — СЛЕДУЮЩИЙ токен, y = data[s + 1 : s + 1 + T].", pytrace=False)
    assert torch.equal(x[:, 1:], y[:, :-1]), "y — тот же кусок, сдвинутый на один токен вперёд"
    windows = data.unfold(0, BLOCK + 1, 1)
    for xi, yi in zip(x, y):
        full = torch.cat([xi, yi[-1:]])
        assert (windows == full).all(1).any(), "x и y должны быть одним непрерывным куском data"
    small = torch.arange(BLOCK + 2)                      # ровно два возможных начала: 0 и 1
    starts = set()
    for _ in range(64):
        xs, ys = impl.get_batch(small, BLOCK, 4, g)
        assert int(ys.max()) <= BLOCK + 1, "y выходит за конец данных"
        starts.update(xs[:, 0].tolist())
    assert starts == {0, 1}, (f"из данных длины block_size + 2 начала могут быть только 0 и 1, а встретились "
                              f"{sorted(starts)}: правая граница randint — len(data) − block_size (не включительно)")
    x2, _ = impl.get_batch(data, BLOCK, 8, torch.Generator().manual_seed(0))
    assert torch.equal(x2, x), "get_batch должен брать случайность только из переданного generator"


def test_train_matches_reference(impl, ref, data):
    """train: та же история лоссов, что у эталона (AdamW, zero_grad, clip 1.0), лосс падает"""
    a, b = _models(impl, ref)
    want = ref.train(a, data, 40, batch_size=8, lr=3e-3, generator=torch.Generator().manual_seed(1))
    got = impl.train(b, data, 40, batch_size=8, lr=3e-3, generator=torch.Generator().manual_seed(1))
    assert isinstance(got, list) and len(got) == 40 and all(isinstance(v, float) for v in got), (
        "train возвращает список из steps чисел (loss.item() каждого шага)")
    if abs(got[0] - want[0]) > 1e-4:
        pytest.fail(f"первый лосс {got[0]:.4f}, у эталона {want[0]:.4f}: батч берётся через get_batch(data, "
                    "model.config.block_size, batch_size, generator) до шага оптимизатора.", pytrace=False)
    k = next((i for i, (u, w) in enumerate(zip(got, want)) if abs(u - w) > 1e-6), None)
    assert k is None, (f"история лоссов разошлась с эталоном на шаге {k}: {got[k]:.4f} против {want[k]:.4f}. "
                       "Шаг: zero_grad → backward → clip_grad_norm_(…, 1.0) → step, AdamW(lr, weight_decay=0.01).")
    assert sum(got[-5:]) / 5 < got[0] - 0.3, "за 40 шагов лосс должен заметно упасть"


def test_overfit_one_batch(impl, data):
    """Smoke-тест: на одном батче из 4 кусков модель доходит почти до нулевого лосса"""
    torch.manual_seed(0)
    model = impl.MiniGPT(_cfg(impl))
    x, y = impl.get_batch(data, BLOCK, 4, torch.Generator().manual_seed(5))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.15, (
        f"после 150 шагов на одном батче loss {loss.item():.3f}: модель обязана запомнить 64 символа. Если нет — "
        "ошибка в forward (маска, остатки, связка весов), а не в данных.")


def test_generate(impl, ref):
    """generate: [B, T + n], жадный выбор и сэмплирование как у эталона, контекст обрезается до block_size"""
    a, b = _models(impl, ref)
    b.train()
    idx = _idx(B=2, T=5)
    out = b.generate(idx, 7, temperature=0)
    assert out.shape == (2, 12), f"[B, T + max_new_tokens] = [2, 12], получено {tuple(out.shape)}"
    assert torch.equal(out[:, :5], idx), "начало — исходный контекст без изменений"
    assert torch.equal(out, a.generate(idx, 7, temperature=0)), "temperature = 0 — argmax, как у эталона"
    assert b.training, "после generate модель возвращается в прежний режим (train)"
    s1 = b.generate(idx, 7, 0.5, torch.Generator().manual_seed(3))
    s2 = a.generate(idx, 7, 0.5, torch.Generator().manual_seed(3))
    assert torch.equal(s1, s2), "сэмплирование: torch.multinomial(softmax(логиты / temperature), 1, generator=generator)"
    long = b.generate(_idx(B=1, T=BLOCK), 2 * BLOCK, 0)
    assert long.shape == (1, 3 * BLOCK), "генерация длиннее block_size: в модель идут только последние block_size токенов"
    assert torch.equal(long, a.generate(_idx(B=1, T=BLOCK), 2 * BLOCK, 0))
