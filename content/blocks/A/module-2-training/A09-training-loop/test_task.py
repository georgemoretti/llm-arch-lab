"""Тесты «Тренировочный цикл».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Данные — три спирали на плоскости (классификация на 3 класса), модель — маленький MLP с dropout.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _spirals(n_per=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in range(3):
        t = torch.linspace(0.1, 1.0, n_per)
        a = t * 4.0 + c * 2 * math.pi / 3 + 0.2 * torch.randn(n_per, generator=g)
        xs.append(torch.stack([t * a.cos(), t * a.sin()], dim=1))
        ys.append(torch.full((n_per,), c))
    X, y = torch.cat(xs), torch.cat(ys)
    perm = torch.randperm(len(X), generator=g)
    return X[perm], y[perm]


def _model(seed=0, p=0.1):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 3))


def _cos(step, max_lr=0.02, warm=10, total=120):
    if step < warm:
        return max_lr * (step + 1) / warm
    return 0.5 * max_lr * (1 + math.cos(math.pi * (step - warm) / (total - warm)))


class _Spy(torch.optim.AdamW):
    """AdamW, который запоминает lr и глобальную норму градиента в момент каждого step()."""

    def __init__(self, params, **kw):
        super().__init__(params, **kw)
        self.seen = []

    @torch.no_grad()
    def step(self, closure=None):
        norm = torch.sqrt(sum((p.grad**2).sum() for g in self.param_groups for p in g["params"] if p.grad is not None))
        self.seen.append((self.param_groups[0]["lr"], norm.item()))
        return super().step(closure)


def _run(mod, steps=120, clip=0.5, eval_every=40):
    X, y = _spirals()
    Xtr, ytr, Xva, yva = X[:140], y[:140], X[140:], y[140:]
    model = _model()
    opt = _Spy(model.parameters(), lr=0.0, weight_decay=0.01)
    torch.manual_seed(1)                      # dropout — от глобального генератора
    hist = mod.train_loop(model, F.cross_entropy, opt, Xtr, ytr, Xva, yva, steps=steps, batch_size=32, lr_fn=_cos,
                          max_grad_norm=clip, eval_every=eval_every, generator=torch.Generator().manual_seed(7))
    return hist, opt.seen, model


def test_get_batch(impl):
    """get_batch: формы [B, …], пары (x, y) сохраняются, те же индексы, что randint с генератором"""
    X = torch.arange(20.0).reshape(10, 2)
    y = torch.arange(10)
    xb, yb = impl.get_batch(X, y, 6, torch.Generator().manual_seed(3))
    assert xb.shape == (6, 2) and yb.shape == (6,), f"батч: x [6, 2], y [6]; получено {tuple(xb.shape)}, {tuple(yb.shape)}"
    assert torch.equal(xb[:, 0] / 2, yb.float()), "x и y в батче перепутаны между собой: бери одни и те же индексы"
    idx = torch.randint(0, 10, (6,), generator=torch.Generator().manual_seed(3))
    assert torch.equal(yb, idx), (
        "батч не совпал с torch.randint(0, N, (batch_size,), generator=g): делай ровно один такой вызов — тогда "
        "обучение воспроизводимо по seed генератора")


def test_evaluate_uses_eval_mode_and_no_grad(impl):
    """evaluate: dropout выключен, граф не строится, режим модели восстанавливается"""
    X, y = _spirals()
    model = _model(p=0.5)
    model.train()
    a = impl.evaluate(model, F.cross_entropy, X, y)
    b = impl.evaluate(model, F.cross_entropy, X, y)
    assert isinstance(a, float), f"evaluate возвращает число float, а не {type(a).__name__}"
    model.eval()
    with torch.no_grad():
        want = F.cross_entropy(model(X), y).item()
    assert math.isclose(a, b) and math.isclose(a, want, rel_tol=1e-6), (
        "лосс на валидации меняется от вызова к вызову или не равен лоссу в eval-режиме: включи model.eval() — "
        "иначе dropout случайно выключает нейроны и при оценке")
    model.train()
    impl.evaluate(model, F.cross_entropy, X, y)
    assert model.training, "после evaluate модель должна вернуться в режим, в котором была (train)"
    assert all(p.grad is None for p in model.parameters()), "evaluate не должен вызывать backward"


def test_train_loop_matches_reference(impl, ref):
    """train_loop: та же история лоссов, что у эталона (порядок шагов, батчи, валидация)"""
    got, _, _ = _run(impl)
    want, _, _ = _run(ref)
    assert set(got) == {"train", "val"}, f"history — словарь с ключами 'train' и 'val', а получено {sorted(got)}"
    assert len(got["train"]) == 120, f"в history['train'] по одному лоссу на шаг (120), а получено {len(got['train'])}"
    assert [s for s, _ in got["val"]] == [39, 79, 119], (
        f"валидация после шагов {[s for s, _ in got['val']]}, а нужно [39, 79, 119]: (step + 1) % eval_every == 0 "
        "и после последнего шага")
    diff = max(abs(a - b) for a, b in zip(got["train"], want["train"]))
    if diff > 1e-5:
        first = next(i for i, (a, b) in enumerate(zip(got["train"], want["train"])) if abs(a - b) > 1e-5)
        pytest.fail(f"история лоссов разошлась с эталоном начиная с шага {first} (max |Δ| = {diff:.2e}). Проверь порядок: "
                    "lr → батч → лосс → zero_grad → backward → clip → step.", pytrace=False)
    for (s, a), (_, b) in zip(got["val"], want["val"]):
        assert math.isclose(a, b, rel_tol=1e-5), f"валидационный лосс на шаге {s}: {a:.5f} ≠ {b:.5f}"


def test_eval_after_last_step(impl):
    """Валидация и после последнего шага, даже если он не кратен eval_every"""
    hist, _, _ = _run(impl, steps=50)
    steps = [s for s, _ in hist["val"]]
    assert steps == [39, 49], (
        f"при steps = 50 и eval_every = 40 валидация должна быть после шагов [39, 49], а была после {steps}. "
        "Последний шаг (steps − 1) валидируется всегда — иначе итоговое качество модели неизвестно.")

def test_lr_and_clip_applied_before_step(impl):
    """В момент step(): lr уже равен lr_fn(step), глобальная норма градиента ≤ max_grad_norm"""
    _, seen, _ = _run(impl)
    assert len(seen) == 120, f"optimizer.step() должен вызываться ровно раз за шаг, а вызван {len(seen)} раз"
    for s, (lr, _) in enumerate(seen):
        if not math.isclose(lr, _cos(s), rel_tol=1e-9, abs_tol=1e-12):
            hint = " Похоже, lr выставляется после step() (на шаг позже)." if s and math.isclose(lr, _cos(s - 1), rel_tol=1e-9) else ""
            pytest.fail(f"на шаге {s} step() увидел lr = {lr:.3e}, а lr_fn({s}) = {_cos(s):.3e}.{hint}", pytrace=False)
    worst = max(n for _, n in seen)
    assert worst <= 0.5 + 1e-4, (
        f"в момент step() глобальная норма градиента {worst:.3f} > max_grad_norm = 0.5: clip_grad_norm_ нужно "
        "вызывать после backward() и до step()")


def test_training_learns(impl):
    """Цикл реально обучает: валидационный лосс падает, точность на train > 80%"""
    hist, _, model = _run(impl, steps=120)
    assert hist["val"][-1][1] < hist["val"][0][1], (
        f"валидационный лосс не падает: {[round(v, 3) for _, v in hist['val']]}")
    X, y = _spirals()
    model.eval()
    with torch.no_grad():
        acc = (model(X[:140]).argmax(-1) == y[:140]).float().mean().item()
    assert acc > 0.8, f"точность на train {acc:.0%} после 120 шагов — эталон даёт > 80%"


def test_overfit_one_batch(impl):
    """Smoke-тест: на одном батче из 16 примеров лосс падает почти до нуля за 300 шагов"""
    X, y = _spirals()
    model = _model(p=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    losses = impl.overfit_one_batch(model, F.cross_entropy, opt, X[:16], y[:16], 300)
    assert len(losses) == 300, f"верни по лоссу на шаг (300), получено {len(losses)}"
    assert losses[-1] < 0.05, (
        f"на одном батче лосс дошёл только до {losses[-1]:.3f} (с {losses[0]:.3f}). Модель с тысячей параметров "
        "обязана запомнить 16 примеров: если не выходит, сломан сам шаг обучения (zero_grad? step?).")
