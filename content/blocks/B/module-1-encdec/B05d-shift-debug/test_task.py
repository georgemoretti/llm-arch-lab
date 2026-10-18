"""Тесты «Найди баг: seq2seq» — те же, что у задачи «Seq2seq целиком», без подсказок про баг.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул лосса — F.cross_entropy(label_smoothing=…, ignore_index=…) (сохранена до запрета).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, V = 0, 1, 2, 16
_CE = F.cross_entropy


@pytest.fixture(autouse=True)
def _forbid_builtin_ce(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Использована готовая cross-entropy (F.cross_entropy / F.nll_loss / nn.CrossEntropyLoss). "
                             "Здесь лосс пишется руками: log_softmax → −log p(y) и −mean log p → смесь с весами.")

    monkeypatch.setattr(F, "cross_entropy", forbidden)
    monkeypatch.setattr(F, "nll_loss", forbidden)
    monkeypatch.setattr(nn.CrossEntropyLoss, "forward", forbidden)


def _oracle_loss(logits, target, eps, ignore):
    return _CE(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), label_smoothing=eps, ignore_index=ignore)


def test_shift_right(impl):
    """shift_right: [y0 … y_{T−1}] → [BOS, y0 … y_{T−2}], вход не меняется"""
    tgt = torch.tensor([[5, 6, 7, EOS], [8, EOS, PAD, PAD]])
    before = tgt.clone()
    out = impl.shift_right(tgt, BOS)
    assert torch.equal(tgt, before), "shift_right испортил входной тензор: работай с копией (torch.cat создаёт новый)"
    want = torch.tensor([[BOS, 5, 6, 7], [BOS, 8, EOS, PAD]])
    if out.shape == want.shape and torch.equal(out[:, 1:], want[:, 1:]) and not torch.equal(out[:, 0], want[:, 0]):
        pytest.fail(f"на позиции 0 стоит {out[:, 0].tolist()}, а должен быть BOS = {BOS}.", pytrace=False)
    assert out.shape == tgt.shape and out.dtype == tgt.dtype, (
        f"длина и тип не меняются: {tuple(tgt.shape)} {tgt.dtype}, получено {tuple(out.shape)} {out.dtype}")
    assert torch.equal(out, want), f"ждём\n{want}\nполучено\n{out}"


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.3])
def test_label_smoothing_matches_torch(impl, eps):
    """label_smoothed_ce == F.cross_entropy(label_smoothing=ε, ignore_index=PAD)"""
    g = torch.Generator().manual_seed(1)
    logits = 3 * torch.randn(2, 6, V, generator=g)
    target = torch.randint(3, V, (2, 6), generator=g)
    target[0, 4:] = PAD
    got = impl.label_smoothed_ce(logits, target, eps, PAD)
    want = _oracle_loss(logits, target, eps, PAD)
    assert got.dim() == 0, f"лосс — скаляр, получена форма {tuple(got.shape)}"
    if not torch.allclose(got, want, atol=1e-5):
        keep_all = _oracle_loss(logits, target, eps, -100)
        if torch.allclose(got, keep_all, atol=1e-5):
            pytest.fail("PAD-позиции вошли в среднее: лосс считается только там, где target ≠ ignore_index.",
                        pytrace=False)
        if eps > 0:
            logp = logits.log_softmax(-1)
            nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            excl = (1 - eps) * nll + eps / (V - 1) * (-logp.sum(-1) - nll)
            if torch.allclose(got, excl[target != PAD].mean(), atol=1e-5):
                pytest.fail("ε распределено по V − 1 неверным классам. Вариант существует (fairseq), но здесь — как в "
                            "PyTorch и статье Szegedy et al.: цель (1 − ε)·one-hot + ε/V по всем V классам.",
                            pytrace=False)
        pytest.fail(f"лосс {got.item():.5f}, ждём {want.item():.5f}: (1 − ε)·(−log p(y)) + ε·(−mean_v log p(v)), "
                    "среднее по не-PAD позициям.", pytrace=False)


def test_label_smoothing_gradient(impl):
    """Градиент по логитам = (softmax − q)/N, где q = (1 − ε)·one-hot + ε/V; на PAD-позициях — ноль"""
    eps = 0.2
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(1, 5, V, generator=g, requires_grad=True)
    target = torch.tensor([[4, 5, 6, PAD, PAD]])
    impl.label_smoothed_ce(logits, target, eps, PAD).backward()
    q = torch.full((1, 5, V), eps / V)
    q.scatter_add_(-1, target.unsqueeze(-1), torch.full((1, 5, 1), 1 - eps))
    want = (logits.detach().softmax(-1) - q) / 3
    want[0, 3:] = 0
    torch.testing.assert_close(logits.grad, want, atol=1e-6, rtol=1e-4,
                               msg="∂loss/∂logits = (p − q)/N: модель тянут к сглаженной цели, а не к one-hot")


def _models(impl, ref, **kw):
    torch.manual_seed(0)
    cfg = ref.Seq2SeqConfig(vocab_size=V, max_len=16, **kw)
    a = ref.Seq2Seq(cfg)
    b = impl.Seq2Seq(impl.Seq2SeqConfig(vocab_size=V, max_len=16, **kw))
    b.load_state_dict(a.state_dict())
    return a, b


def _batch():
    src = torch.tensor([[5, 6, 7, 8, 9, PAD], [10, 11, 12, 13, 14, 15]])
    tgt = torch.tensor([[9, 8, 7, EOS, PAD], [15, 14, 13, 12, EOS]])
    return src, tgt


def test_model_forward(impl, ref):
    """encode → (memory, src_mask); decode → логиты [B, T, V]; forward = decode(encode)"""
    a, b = _models(impl, ref)
    src, tgt = _batch()
    tgt_in = torch.cat([torch.full((2, 1), BOS), tgt[:, :-1]], 1)
    res = b.encode(src)
    assert isinstance(res, tuple) and len(res) == 2, "encode возвращает (memory, src_mask)"
    memory, mask = res
    assert memory.shape == (2, 6, 32) and mask.shape == (2, 1, 1, 6), (
        f"memory [B, S, C] = [2, 6, 32] и src_mask [B, 1, 1, S] = [2, 1, 1, 6], получено {tuple(memory.shape)} и "
        f"{tuple(mask.shape)}")
    logits = b(src, tgt_in)
    assert logits.shape == (2, 5, V), f"логиты [B, T, V] = [2, 5, {V}], получено {tuple(logits.shape)}"
    torch.testing.assert_close(logits, a(src, tgt_in), atol=1e-5, rtol=1e-4,
                               msg="forward = lm_head(decoder(tgt_in, encoder(src), src_mask))")
    torch.testing.assert_close(b.decode(tgt_in, memory, mask), logits, atol=1e-6, rtol=1e-5,
                               msg="decode(tgt_in, *encode(src)) должен давать те же логиты, что forward")


def test_teacher_forcing_is_causal(impl, ref):
    """Логиты позиции t не зависят от tgt_in после t, но зависят от всего источника"""
    _, b = _models(impl, ref)
    src, tgt = _batch()
    tgt_in = torch.cat([torch.full((2, 1), BOS), tgt[:, :-1]], 1)
    base = b(src, tgt_in)
    other = tgt_in.clone()
    other[:, 2:] = 3
    torch.testing.assert_close(b(src, other)[:, :2], base[:, :2], atol=1e-5, rtol=1e-4,
                               msg="логиты позиций 0–1 зависят от входа декодера на позициях ≥ 2 → утечка ответа")
    src2 = src.clone()
    src2[1, -1] = 3
    assert not torch.allclose(b(src2, tgt_in)[1, 0], base[1, 0], atol=1e-5), (
        "первая позиция цели не видит последний токен источника")


class _Scripted(nn.Module):
    """Фальшивая модель: на шаге t строка b выдаёт script[b][t]. Считает вызовы encode и decode."""

    def __init__(self, ref, script):
        super().__init__()
        self.cfg = ref.Seq2SeqConfig(vocab_size=V)
        self.script = script
        self.n_encode = self.n_decode = 0
        self.prefixes = []

    def encode(self, src):
        self.n_encode += 1
        return torch.zeros(src.shape[0], src.shape[1], 4), (src != PAD)[:, None, None, :]

    def decode(self, tgt_in, memory, src_mask):
        self.n_decode += 1
        self.prefixes.append(tgt_in.clone())
        B, T = tgt_in.shape
        logits = torch.zeros(B, T, V)
        for b in range(B):
            logits[b, -1, self.script[b][min(T - 1, len(self.script[b]) - 1)]] = 5.0
        return logits


def test_greedy_stops_and_pads(impl, ref):
    """greedy_decode: энкодер один раз, BOS в начале, PAD после EOS, остановка когда закончили все"""
    m = _Scripted(ref, [[5, EOS, 7, 7, 7, 7], [6, 6, 6, EOS, 9, 9]])
    out = impl.greedy_decode(m, torch.tensor([[5, 6, 7], [5, PAD, PAD]]), max_len=6)
    assert m.n_encode == 1, f"энкодер вызван {m.n_encode} раз: память источника не меняется, считать её нужно один раз"
    assert m.prefixes and torch.equal(m.prefixes[0], torch.full((2, 1), BOS)), (
        "первый вызов decode — на префиксе из одного BOS в каждой строке")
    want = torch.tensor([[5, EOS, PAD, PAD], [6, 6, 6, EOS]])
    if out.numel() and torch.all(out == PAD):
        pytest.fail("все сгенерированные токены — PAD (argmax нулевых логитов): следующий токен берётся из логитов "
                    "ПОСЛЕДНЕЙ позиции префикса, logits[:, -1].", pytrace=False)
    if out.shape[1] == 7 or (out.shape[1] == 5 and torch.all(out[:, 0] == BOS)):
        pytest.fail("в выходе остался BOS: верни сгенерированные токены без первого столбца.", pytrace=False)
    if out.shape == want.shape and torch.equal(out[1], want[1]) and out[0, 2] == 7:
        pytest.fail("после EOS строка продолжила генерацию: закончившим строкам дописывай PAD.", pytrace=False)
    assert out.shape[1] == 4, (f"оба текста закончились на шаге 4 → длина выхода 4, получено {out.shape[1]}: "
                               "остановись, когда EOS есть в каждой строке")
    assert torch.equal(out, want), f"ждём\n{want}\nполучено\n{out}"
    assert m.n_decode == 4, f"decode вызван {m.n_decode} раз, ждём 4 (по разу на шаг)"


def test_greedy_max_len_and_reference(impl, ref):
    """Без EOS — ровно max_len шагов; на случайной модели совпадает с эталоном шаг за шагом"""
    m = _Scripted(ref, [[5] * 10, [6] * 10])
    out = impl.greedy_decode(m, torch.tensor([[5, 6], [7, 8]]), max_len=3)
    assert out.shape == (2, 3), f"без EOS — ровно max_len = 3 токена, получено {tuple(out.shape)}"
    a, b = _models(impl, ref)
    src, _ = _batch()
    got = impl.greedy_decode(b.eval(), src, 6)
    torch.testing.assert_close(got, ref.greedy_decode(a.eval(), src, 6),
                               msg="на каждом шаге — argmax логитов ПОСЛЕДНЕЙ позиции")
    assert not got.requires_grad, "генерация без графа градиентов (@torch.no_grad)"


def test_overfit_reverse(impl):
    """Smoke test: свои shift_right + лосс + модель запоминают 6 пар «развернуть последовательность»"""
    torch.manual_seed(0)
    cfg = impl.Seq2SeqConfig(vocab_size=V, d_model=32, n_heads=4, d_ff=64, n_layers=1, max_len=8)
    model = impl.Seq2Seq(cfg)
    g = torch.Generator().manual_seed(3)
    src = torch.randint(3, V, (6, 4), generator=g)
    tgt = torch.cat([src.flip(1), torch.full((6, 1), EOS)], 1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = impl.label_smoothed_ce(model(src, impl.shift_right(tgt, BOS)), tgt, 0.0, PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    out = impl.greedy_decode(model, src, 6)
    assert loss.item() < 0.05, f"на 6 примерах лосс должен упасть почти до 0, получено {loss.item():.3f}"
    assert out.shape[1] == 5 and torch.equal(out, tgt), (
        "после обучения жадная генерация должна воспроизвести развёрнутые последовательности с EOS.")
