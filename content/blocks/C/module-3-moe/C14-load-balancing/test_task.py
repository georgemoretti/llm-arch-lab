"""Тесты «Балансировка экспертов»: балансировка нагрузки экспертов.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Свойства aux loss: идеальный баланс → 1, коллапс в одного эксперта → ≈ E; совпадает с
load_balancing_loss_func из HF Mixtral, делённым на top_k; градиент идёт только через P (f — счётчик);
несколько шагов SGD по одному aux loss разгоняют «схлопнувшийся» роутер. Свойства bias-балансировки:
bias меняет выбор, но не веса; правило шага; 300 шагов выравнивают перекошенную нагрузку.
"""

import pytest
import torch
from torch.testing import assert_close

E, K, C = 8, 2, 16


def _hf_mixtral_loss(logits, top_k):
    """load_balancing_loss_func из transformers (modeling_mixtral.py) для одного слоя без маски."""
    probs = logits.float().softmax(-1)
    sel = probs.topk(top_k, dim=-1).indices
    tokens_per_expert = torch.bincount(sel.reshape(-1), minlength=logits.shape[1]).float() / logits.shape[0]
    return logits.shape[1] * (tokens_per_expert * probs.mean(0)).sum()


# ---------------------------------------------------------------------------
# expert_load
# ---------------------------------------------------------------------------

def test_expert_load(impl):
    """expert_load: доли назначений на экспертов, сумма 1, неиспользуемые эксперты — 0"""
    idx = torch.tensor([[0, 1], [0, 2], [0, 1]])
    load = impl.expert_load(idx, 5)
    assert tuple(load.shape) == (5,), (
        f"expert_load вернула форму {tuple(load.shape)}, ожидалось (5,): по доле на КАЖДОГО эксперта, включая "
        "тех, кого никто не выбрал (bincount(..., minlength=n_experts)).")
    assert_close(load, torch.tensor([3, 2, 1, 0, 0]) / 6.0,
                 atol=1e-6, rtol=1e-5, msg=lambda _: "Доли: эксперт 0 выбран 3 раза из 6 назначений, 1 — 2 раза, 2 — 1 раз.")


# ---------------------------------------------------------------------------
# load_balancing_loss
# ---------------------------------------------------------------------------

def _balanced_logits():
    """N = E токенов, токен n сильно хочет экспертов n и n+1 — нагрузка и вероятности равномерные."""
    logits = torch.zeros(E, E)
    for n in range(E):
        logits[n, n] = logits[n, (n + 1) % E] = 4.0
    return logits


def test_loss_balanced_is_one(impl):
    """Идеальный баланс (равные доли назначений и средние вероятности) → loss = 1"""
    loss = impl.load_balancing_loss(_balanced_logits(), K)
    assert loss.dim() == 0, f"loss должен быть скаляром, получена форма {tuple(loss.shape)}."
    hf = _hf_mixtral_loss(_balanced_logits(), K)
    hint = ""
    if abs(loss.item() - hf.item()) < 1e-4:
        hint = (f" Получилось {hf.item():.3g} = top_k — так считает HF Mixtral (доли нормированы на N, а не на N·k). "
                "В этой задаче f_i нормированы к сумме 1, тогда минимум loss = 1 при любом k.")
    assert abs(loss.item() - 1.0) < 1e-4, f"При идеальном балансе loss = {loss.item():.4g}, ожидалось 1.{hint}"


def test_loss_collapse_is_E(impl):
    """Все токены уходят к одному эксперту с вероятностью ≈ 1 → loss ≈ E"""
    logits = torch.zeros(32, E)
    logits[:, 3] = 20.0
    loss = impl.load_balancing_loss(logits, 1).item()
    assert abs(loss - E) < 1e-2, (
        f"Коллапс в эксперта 3 (top-1): loss = {loss:.3g}, ожидалось ≈ E = {E}: f_3 = 1, P_3 ≈ 1, E·1·1.")


def test_loss_matches_hf(impl):
    """На случайных логитах: loss == HF load_balancing_loss_func / top_k"""
    for k in (1, 2, 4):
        logits = torch.randn(40, E, generator=torch.Generator().manual_seed(k))
        loss = impl.load_balancing_loss(logits, k)
        expected = _hf_mixtral_loss(logits, k) / k
        assert_close(loss, expected, atol=1e-5, rtol=1e-5,
                     msg=lambda _: f"top_k={k}: loss = {loss.item():.5g}, ожидалось {expected.item():.5g}. "
                                   "f_i — доля назначений top-k (сумма по экспертам = 1), P_i — средняя вероятность "
                                   "softmax по всем экспертам, loss = E · Σ f_i·P_i.")


def test_loss_gradient_only_through_P(impl):
    """Градиент по логитам — только через P (средние вероятности); f — недифференцируемый счётчик"""
    logits = torch.randn(30, E, generator=torch.Generator().manual_seed(7)).requires_grad_()
    loss = impl.load_balancing_loss(logits, K)
    if not loss.requires_grad:
        pytest.fail("loss не зависит от логитов дифференцируемо: P_i = mean(softmax(logits)) должна быть в графе — "
                    "иначе aux loss ничего не учит.", pytrace=False)
    loss.backward()
    ref_logits = logits.detach().clone().requires_grad_()
    probs = ref_logits.softmax(-1)
    f = torch.bincount(probs.topk(K, -1).indices.reshape(-1), minlength=E).float() / (30 * K)
    (E * (f * probs.mean(0)).sum()).backward()
    assert_close(logits.grad, ref_logits.grad, atol=1e-6, rtol=1e-5,
                 msg=lambda _: "Градиент aux loss по логитам не совпал с ожидаемым. f_i (доля назначений) — это "
                               "счётчик, он не дифференцируем; если ты заменил его «мягкой» версией (например, "
                               "суммой вероятностей), то f и P совпадают и градиент другой.")


def test_aux_loss_rebalances_router(impl):
    """30 шагов SGD только по aux loss разгоняют роутер, у которого почти все токены шли к эксперту 0"""
    torch.manual_seed(1)
    W = (torch.randn(E, C) * C ** -0.5)
    W[0] += 0.5
    W.requires_grad_()
    X = torch.randn(512, C) + 0.5
    opt = torch.optim.SGD([W], lr=0.5)

    def max_load():
        with torch.no_grad():
            idx = (X @ W.T).topk(K, -1).indices
        return (torch.bincount(idx.reshape(-1), minlength=E).float() / idx.numel()).max().item()

    before = max_load()
    for _ in range(30):
        loss = impl.load_balancing_loss(X @ W.T, K)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = max_load()
    assert before > 0.3, "внутренняя ошибка теста: начальный роутер недостаточно перекошен"
    assert after < 0.2, (
        f"После 30 шагов по aux loss самый загруженный эксперт получает {after:.2f} назначений (было {before:.2f}, "
        f"идеал {1 / E:.3f}). Градиент через P_i должен уменьшать вероятность перегруженных экспертов: проверь "
        "знак и что P_i = среднее softmax по токенам.")


# ---------------------------------------------------------------------------
# bias-based (DeepSeek-V3)
# ---------------------------------------------------------------------------

def test_route_with_bias(impl):
    """Bias меняет только выбор экспертов; веса — исходные scores выбранных (перенормированные)"""
    scores = torch.tensor([[0.9, 0.6, 0.3, 0.2]])
    bias = torch.tensor([-1.0, 0.0, 0.0, 0.5])
    w, idx = impl.route_with_bias(scores, bias, 2)
    assert sorted(idx[0].tolist()) == [1, 3], (
        f"Выбраны эксперты {sorted(idx[0].tolist())}, ожидалось [1, 3]: top-2 по scores + bias = "
        "[−0.1, 0.6, 0.3, 0.7].")
    got = dict(zip(idx[0].tolist(), w[0].tolist()))
    expected = {1: 0.6 / 0.8, 3: 0.2 / 0.8}                       # сумма выбранных scores = 0.8 ≠ 1
    if not all(abs(got[e] - v) < 1e-6 for e, v in expected.items()):
        hint = ""
        if abs(got[1] - 0.6 / 1.3) < 1e-6:
            hint = " Похоже, веса взяты из scores + bias. Bias нужен только для выбора — веса из исходных scores."
        elif abs(got[1] - 0.6) < 1e-6:
            hint = " Похоже, веса не перенормированы: при norm_topk_prob = True их делят на сумму выбранных (0.8)."
        pytest.fail(f"Веса выбранных экспертов {got}, ожидалось {expected} (0.6 и 0.2, нормированные к сумме 1).{hint}",
                    pytrace=False)
    w2, _ = impl.route_with_bias(scores, bias, 2, norm_topk_prob=False)
    assert sorted(w2[0].tolist()) == pytest.approx([0.2, 0.6]), "Без перенормировки веса — сами scores выбранных."


def test_update_bias_rule(impl):
    """Перегруженным −γ, недогруженным +γ, средним — 0; шаг не зависит от величины перекоса"""
    bias = torch.zeros(4)
    load = torch.tensor([0.55, 0.25, 0.2, 0.0])          # среднее 0.25
    new = impl.update_bias(bias, load, 0.1)
    assert_close(new, torch.tensor([-0.1, 0.0, 0.1, 0.1]),
                 atol=1e-6, rtol=1e-5, msg=lambda _: f"update_bias дал {[round(v, 3) for v in new.tolist()]}, ожидалось [−0.1, 0, +0.1, +0.1]: эксперт 0 "
                               "перегружен (bias уменьшается на γ), 1 ровно средний, 2 и 3 недогружены (+γ) — "
                               "на одинаковый шаг γ, независимо от того, насколько.")


def test_bias_balancing_simulation(impl):
    """300 шагов bias-балансировки выравнивают нагрузку перекошенного sigmoid-роутера"""
    torch.manual_seed(2)
    Wr = torch.randn(E, C)
    skew = torch.linspace(-1, 1, E)                       # эксперт 7 изначально «любимый», 0 — заброшенный
    bias = torch.zeros(E)
    for step in range(301):
        X = torch.randn(256, C)
        scores = (X @ Wr.T / 4 + skew).sigmoid()
        _, idx = impl.route_with_bias(scores, bias, K)
        load = torch.bincount(idx.reshape(-1), minlength=E).float() / idx.numel()
        if step == 0:
            start = (load.max().item(), load.min().item())
        bias = impl.update_bias(bias, load, 0.01)
    end = (load.max().item(), load.min().item())
    assert end[0] < 0.18 and end[1] > 0.08, (
        f"Нагрузка экспертов (max, min): в начале {start[0]:.2f}, {start[1]:.3f}; через 300 шагов {end[0]:.2f}, "
        f"{end[1]:.3f}; идеал {1 / E:.3f}. Bias перегруженных экспертов должен уменьшаться, чтобы их реже "
        "выбирали. Если перекос растёт — перепутан знак шага.")
