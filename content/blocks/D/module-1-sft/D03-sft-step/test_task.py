"""Тесты «Шаг SFT».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter).

Модель — игрушечная Llama из llmlab.core.model (конфиг llama-like, 2 слоя, d_model 32) под посимвольный токенизатор
курса; данные — синтетические инструкции из llmlab.core.sft, метки — эталоном D02. Оракул накопления — один большой
батч из тех же примеров с лоссом «сумма CE / число токенов ответа».
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from llmlab.core import sft as core
from llmlab.core.model import Model

IGNORE = -100
TOK = core.CharTokenizer()
HEADER = core.assistant_header(TOK)


def _example(ex):
    ids = core.encode_chat(core.render_chat(core.to_messages(ex)), TOK)
    return ids, core.sft_targets(ids, core.assistant_mask(ids, HEADER, TOK.eos_id))


def _model(seed=0):
    torch.manual_seed(seed)
    return Model(core.toy_config(TOK.vocab_size, d_model=32, n_layers=2))


def _uneven_micro_batches():
    """Три микро-батча с сильно разным числом токенов ответа: JSON (~30 токенов) против чисел (~8)."""
    exs = core.make_instructions(30, seed=3)
    by_task = {t: [_example(e) for e in exs if e["task"] == t] for t in core.TASKS}
    groups = [by_task["json"][:4], by_task["arith"][:1], by_task["arith"][1:3] + by_task["upper"][:1]]
    return [core.collate(g, TOK.pad_id) for g in groups], [ex for g in groups for ex in g]


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def test_masked_ce_sum(impl):
    """Сумма CE по позициям с меткой ≠ −100 и число таких позиций"""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 7, generator=g)
    targets = torch.tensor([[IGNORE, 3, 1, IGNORE, IGNORE], [2, IGNORE, 6, 0, IGNORE]])
    out = impl.masked_ce_sum(logits, targets)
    assert isinstance(out, tuple) and len(out) == 2, "masked_ce_sum возвращает (сумма лосса, число меченых позиций)"
    loss, n = out
    want = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1), ignore_index=IGNORE, reduction="sum")
    assert n == 5 and isinstance(n, int), f"меченых позиций 5 (int), получено {n!r}"
    if abs(float(loss) - float(want) / 5) < 1e-5:
        pytest.fail("вернулось среднее, а нужна СУММА: делить будем на число токенов во всём шаге, а не в микро-батче",
                    pytrace=False)
    torch.testing.assert_close(loss, want, atol=1e-5, rtol=1e-5,
                               msg="сумма cross-entropy по позициям с меткой (−100 пропускаются)")


def test_accumulation_equals_big_batch(impl):
    """Накопление по микро-батчам разной длины == один большой батч с лоссом «сумма / все токены ответа»"""
    micro, flat = _uneven_micro_batches()
    big = _model()
    ids, tgt = core.collate(flat, TOK.pad_id)
    s, n = core.masked_ce_sum(big(ids), tgt)
    (s / n).backward()
    want, s = _grads(big), float(s.detach())

    model = _model()
    out = impl.accumulate_gradients(model, micro)
    assert isinstance(out, tuple) and len(out) == 2, "accumulate_gradients возвращает (средний лосс на токен, N)"
    loss, n_tok = out
    assert n_tok == n, f"токенов ответа во всех микро-батчах {n}, получено {n_tok}"
    assert abs(loss - s / n) < 1e-4, f"средний лосс на токен {s / n:.4f}, получено {loss:.4f}"
    got = _grads(model)
    assert got, "после accumulate_gradients у параметров нет .grad: вызови backward"
    name = "layers.0.self_attn.q_proj.weight"
    ratio = float(got[name].norm() / want[name].norm())

    # «Среднее средних»: каждый микро-батч — свой mean, затем / число микро-батчей
    mm = _model()
    for x, t in micro:
        ls, k = core.masked_ce_sum(mm(x), t)
        (ls / k / len(micro)).backward()
    if all(torch.allclose(got[k], v, atol=1e-6, rtol=1e-4) for k, v in _grads(mm).items()):
        pytest.fail("среднее средних: каждый микро-батч усреднён по СВОИМ токенам и поделён на число микро-батчей. "
                    "Тогда токен из короткого микро-батча весит больше токена из длинного, и градиент != большому "
                    "батчу. Дели сумму каждого микро-батча на число токенов во ВСЁМ шаге.", pytrace=False)
    if abs(ratio - n) / n < 0.01:
        pytest.fail(f"градиент в {n} раз больше: сумма не поделена на число токенов ответа", pytrace=False)
    for k, v in want.items():
        torch.testing.assert_close(got[k], v, atol=1e-6, rtol=1e-4, msg=f"градиент {k} != градиенту большого батча")


def test_accumulation_ignores_padding_rows(impl):
    """Лишний микро-батч из одних PAD (все метки −100) не меняет градиент"""
    micro, _ = _uneven_micro_batches()
    a, b = _model(), _model()
    impl.accumulate_gradients(a, micro)
    pad = (torch.full((2, 6), TOK.pad_id), torch.full((2, 6), IGNORE))
    impl.accumulate_gradients(b, micro + [pad])
    for k, v in _grads(a).items():
        torch.testing.assert_close(_grads(b)[k], v, atol=1e-6, rtol=1e-4,
                                   msg="микро-батч без единой метки изменил градиент: считай N только по меткам ≠ −100")
    with pytest.raises(ValueError):
        impl.accumulate_gradients(_model(), [pad])


def test_sft_step_matches_reference(impl):
    """sft_step: lr по расписанию A08 → zero_grad → накопление → clip → step; 4 шага == эталон"""
    micro, _ = _uneven_micro_batches()
    kw = dict(max_lr=3e-3, warmup_steps=2, total_steps=4, min_lr=3e-4, max_norm=0.5)
    ours, theirs = _model(), _model()
    o1 = torch.optim.AdamW(ours.parameters(), lr=1.0, weight_decay=0.0)
    o2 = torch.optim.AdamW(theirs.parameters(), lr=1.0, weight_decay=0.0)
    for p in ours.parameters():
        p.grad = torch.full_like(p, 7.0)                  # мусор от прошлого шага: должен быть стёрт
    for step in range(4):
        st = impl.sft_step(ours, o1, micro, step, **kw)
        want = core.sft_step(theirs, o2, micro, step, **kw)
        assert isinstance(st, dict) and {"loss", "tokens", "lr", "grad_norm"} <= set(st), (
            "sft_step возвращает dict с ключами loss, tokens, lr, grad_norm")
        assert abs(st["lr"] - want["lr"]) < 1e-12, (
            f"шаг {step}: lr {st['lr']} вместо {want['lr']} — возьми lr_at(step, …) из A08 и запиши в param_groups")
        if step == 0 and abs(st["grad_norm"] - want["grad_norm"]) > 1e-3 * want["grad_norm"]:
            pytest.fail(f"норма градиента на шаге 0 = {st['grad_norm']:.3f}, ждём {want['grad_norm']:.3f}. Градиенты "
                        "прошлого шага не обнулены (zero_grad до накопления) или норма считается после clip.",
                        pytrace=False)
        assert abs(st["grad_norm"] - want["grad_norm"]) < 1e-3 * max(1.0, want["grad_norm"]), (
            f"шаг {step}: grad_norm {st['grad_norm']:.4f} vs {want['grad_norm']:.4f} (норма ДО обрезки)")
    for (k, p), q in zip(ours.named_parameters(), theirs.parameters()):
        torch.testing.assert_close(p, q, atol=1e-5, rtol=1e-4,
                                   msg=f"{k} после 4 шагов разошёлся с эталоном: порядок zero_grad → backward → clip → step")


def test_overfit_one_batch(impl):
    """60 шагов на одном батче из 8 инструкций: лосс по ответу падает ниже 0.1"""
    torch.manual_seed(0)
    model = Model(core.toy_config(TOK.vocab_size, d_model=64, n_layers=2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    batch = core.collate([_example(e) for e in core.make_instructions(8, seed=5)], TOK.pad_id)
    first = last = None
    for step in range(60):
        st = impl.sft_step(model, opt, [batch], step, max_lr=1e-2, warmup_steps=5, total_steps=60, min_lr=1e-3)
        first = st["loss"] if first is None else first
        last = st["loss"]
    assert last < 0.1 and last < first / 10, (
        f"лосс по ответу {first:.3f} → {last:.3f} за 60 шагов. Модель должна выучить 8 ответов почти наизусть; если "
        "лосс стоит — проверь, что optimizer.step() вызывается и lr не нулевой.")
