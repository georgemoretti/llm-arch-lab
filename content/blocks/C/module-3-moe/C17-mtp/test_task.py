"""Тесты «Multi-Token Prediction»: Multi-Token Prediction.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства: модуль = eh_proj([enorm(emb); hnorm(h)]) → блок; логиты глубины k имеют длину T − k и в
позиции i видят токены ≤ i + k (и не дальше); глубины связаны цепочкой; эмбеддинг и голова общие с
основной моделью; loss: основной + λ/D·Σ L_k с целями i + 1 и i + k + 1; градиент MTP доходит до
скрытых состояний основной модели.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

B, T, C, H, H_KV, F_HID, V = 2, 12, 32, 4, 2, 48, 40


def _setup(impl, D=2, seed=0):
    torch.manual_seed(seed)
    embed = nn.Embedding(V, C)
    head = nn.Linear(C, V, bias=False)
    mods = nn.ModuleList(impl.MTPModule(C, H, H_KV, F_HID) for _ in range(D)).eval()
    with torch.no_grad():
        for m in mods:
            m.enorm.weight.uniform_(0.5, 1.5)
            m.hnorm.weight.uniform_(0.5, 1.5)
    tokens = torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed + 1))
    h_main = torch.randn(B, T, C, generator=torch.Generator().manual_seed(seed + 2))
    return embed, head, mods, tokens, h_main


def test_module_forward(impl):
    """MTPModule: eh_proj([enorm(emb) ; hnorm(h)]) → блок трансформера"""
    embed, head, mods, tokens, h = _setup(impl, D=1)
    m = mods[0]
    e = embed(tokens)
    with torch.no_grad():
        out = m(h, e)
        good = m.block(m.eh_proj(torch.cat((m.enorm(e), m.hnorm(h)), -1)))
        swapped = m.block(m.eh_proj(torch.cat((m.hnorm(h), m.enorm(e)), -1)))
        no_norm = m.block(m.eh_proj(torch.cat((e, h), -1)))
    assert tuple(out.shape) == (B, T, C), f"Выход формы {tuple(out.shape)}, ожидалось {(B, T, C)}."
    if not torch.allclose(out, good, atol=1e-5, rtol=1e-4):
        hint = ""
        if torch.allclose(out, swapped, atol=1e-5):
            hint = (" Похоже, порядок конкатенации [h ; эмбеддинг]. В статье формула записана так, но в коде инференса "
                    "(vLLM, deepseek_mtp.py) — [enorm(эмбеддинг) ; hnorm(h)], и веса eh_proj обучены под этот порядок.")
        elif torch.allclose(out, no_norm, atol=1e-5):
            hint = " Похоже, входы не нормированы: enorm для эмбеддинга, hnorm для h."
        pytest.fail(f"MTPModule.forward не совпадает с eh_proj([enorm(emb) ; hnorm(h)]) → block.{hint}", pytrace=False)


def test_logits_shapes(impl):
    """mtp_logits: D тензоров, глубина k — [B, T − k, V]"""
    embed, head, mods, tokens, h = _setup(impl, D=3)
    with torch.no_grad():
        out = impl.mtp_logits(h, tokens, embed, head, mods)
    assert isinstance(out, list) and len(out) == 3, "mtp_logits возвращает список логитов по глубинам 1..D."
    shapes = [tuple(o.shape) for o in out]
    expected = [(B, T - k, V) for k in (1, 2, 3)]
    assert shapes == expected, (
        f"Формы {shapes}, ожидалось {expected}: на глубине k позиция i получает эмбеддинг токена i + k, поэтому "
        "позиций на k меньше, чем токенов.")


def _logits_k(impl, setup, tokens, k):
    embed, head, mods, _, h = setup
    with torch.no_grad():
        return impl.mtp_logits(h, tokens, embed, head, mods)[k - 1]


def test_causality(impl):
    """Глубина k, позиция i видит токены ≤ i + k: токен j влияет только на позиции ≥ j − k"""
    setup = _setup(impl, D=2)
    tokens = setup[3]
    j = 8
    t2 = tokens.clone()
    t2[:, j] = (t2[:, j] + 7) % V
    for k in (1, 2):
        a, b = _logits_k(impl, setup, tokens, k), _logits_k(impl, setup, t2, k)
        changed = ((a - b).abs() > 1e-6).any(-1).any(0)
        first = int(changed.nonzero()[0]) if changed.any() else None
        assert first == j - k, (
            f"Глубина {k}: изменение токена {j} впервые меняет логиты в позиции {first}, ожидалось {j - k}. Позиция i "
            f"глубины k получает эмбеддинг токена i + k (то есть tokens[:, k:]) и предсказывает токен i + k + 1; "
            "видеть более поздние токены она не должна, а токен i + k — должна.")


def test_depths_are_chained(impl):
    """Глубина 2 строится на выходе глубины 1, а не на h основной модели"""
    setup = _setup(impl, D=2)
    embed, head, mods, tokens, h = setup
    with torch.no_grad():
        before = impl.mtp_logits(h, tokens, embed, head, mods)[1]
        for p in mods[0].parameters():
            p.add_(0.1 * torch.randn_like(p))
        after = impl.mtp_logits(h, tokens, embed, head, mods)[1]
    assert not torch.allclose(before, after, atol=1e-6), (
        "Логиты глубины 2 не изменились при изменении модуля глубины 1. В DeepSeek-V3 модули идут цепочкой: глубина k "
        "получает h глубины k − 1 (для k = 1 — h основной модели), так сохраняется причинная цепочка предсказаний.")


def test_shared_embedding_and_head(impl):
    """Эмбеддинг и выходная голова — общие с основной моделью (вызываются D раз), у модулей их нет"""
    embed, head, mods, tokens, h = _setup(impl, D=2)
    calls = {"e": 0, "h": 0}
    he = embed.register_forward_hook(lambda *a: calls.__setitem__("e", calls["e"] + 1))
    hh = head.register_forward_hook(lambda *a: calls.__setitem__("h", calls["h"] + 1))
    with torch.no_grad():
        impl.mtp_logits(h, tokens, embed, head, mods)
    he.remove()
    hh.remove()
    assert calls == {"e": 2, "h": 2}, (
        f"embed вызван {calls['e']} раз, lm_head — {calls['h']}; ожидалось по 2 (по разу на глубину). MTP-модули не "
        "имеют своих эмбеддингов и головы — используют общие с основной моделью.")


def _manual_loss(main_logits, mtp, tokens, lam):
    ce = lambda lg, tg: F.cross_entropy(lg.reshape(-1, V), tg.reshape(-1))
    l_main = ce(main_logits[:, :-1], tokens[:, 1:])
    ls = [ce(lg[:, :-1], tokens[:, k + 1:]) for k, lg in enumerate(mtp, 1)]
    return l_main + lam / len(ls) * sum(ls), l_main, ls


def test_loss(impl):
    """mtp_loss: L_main (цель i+1) + λ/D · Σ L_k (цель i+k+1); последняя позиция каждой глубины без цели"""
    embed, head, mods, tokens, h = _setup(impl, D=2)
    with torch.no_grad():
        main = head(h)
        mtp = impl.mtp_logits(h, tokens, embed, head, mods)
    try:
        total, l_main, ls = impl.mtp_loss(main, mtp, tokens, 0.3)
    except RuntimeError as e:
        pytest.fail(f"mtp_loss упал: {str(e)[:120]}. Скорее всего, длины логитов и целей не совпали: у глубины k логиты "
                    f"[B, T − k, V], цели — tokens[:, k + 1:] длины T − k − 1, значит, последняя позиция логитов "
                    "отбрасывается (её цель за пределами последовательности).", pytrace=False)
    e_total, e_main, e_ls = _manual_loss(main, mtp, tokens, 0.3)
    assert_close(l_main, e_main, atol=1e-6, rtol=1e-5, msg=lambda _: "L_main: логиты позиции i предсказывают tokens[:, i + 1].")
    assert len(ls) == 2, "Вторым элементом — список L_k по глубинам."
    for k, (a, b) in enumerate(zip(ls, e_ls), 1):
        assert_close(a, b, atol=1e-6, rtol=1e-5, msg=lambda _: f"L_{k}: позиция i глубины {k} предсказывает tokens[:, i + {k} + 1]; "
                                         f"логиты [:, :-1], цели tokens[:, {k + 1}:].")
    assert_close(total, e_total, atol=1e-6, rtol=1e-5, msg=lambda _: "total = L_main + λ/D · Σ L_k (усреднение по глубинам, затем λ).")


def test_mtp_gradient_reaches_main(impl):
    """Градиент MTP-loss доходит до скрытых состояний основной модели — сигнал обучения плотнее"""
    embed, head, mods, tokens, h = _setup(impl, D=1)
    h = h.clone().requires_grad_()
    mtp = impl.mtp_logits(h, tokens, embed, head, mods)
    _, _, ls = impl.mtp_loss(head(h), mtp, tokens, 0.3)
    ls[0].backward()
    g = h.grad
    assert g is not None and g[:, : T - 1].abs().sum() > 0, (
        "Градиент loss глубины 1 не дошёл до h основной модели. h_main должен входить в MTP-модуль без .detach(): "
        "MTP учит основную модель «планировать» следующие токены.")
    assert g[:, T - 1].abs().max() == 0, (
        "Последняя позиция h_main не должна участвовать в глубине 1: её эмбеддинга t_{T} нет — модуль берёт h[:, :T−1].")
