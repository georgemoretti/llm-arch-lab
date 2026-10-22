"""Тесты «Найди баг: cross-attention» — те же, что у задачи «Декодер и cross-attention», без подсказок про баг.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул слоя — nn.TransformerDecoderLayer (dropout = 0) с теми же весами.
"""

import pytest
import torch
import torch.nn as nn

C, H, FF, PAD, V = 32, 4, 64, 0, 30


def _src():
    return torch.tensor([[5, 6, 7, PAD, PAD, PAD, PAD], [3, 4, 5, 6, 7, 8, 9]])   # S = 7


def _tgt():
    return torch.tensor([[1, 9, 8, PAD, PAD], [1, 4, 4, 4, 2]])                    # T = 5


def _src_mask(src):
    return (src != PAD)[:, None, None, :]


def _no_builtin(mod):
    banned = (nn.TransformerDecoderLayer, nn.TransformerDecoder, nn.MultiheadAttention)
    used = sorted({type(m).__name__ for m in mod.modules() if isinstance(m, banned)})
    assert not used, f"использованы готовые слои PyTorch ({', '.join(used)}): здесь слой собирается руками"


def test_causal_mask(impl):
    """causal_mask: bool [T, T], нижний треугольник с диагональю"""
    m = impl.causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4), f"bool [4, 4], получено {m.dtype} {tuple(m.shape)}"
    torch.testing.assert_close(m, torch.tril(torch.ones(4, 4, dtype=torch.bool)),
                               msg="строка — запрос, столбец — ключ; разрешено ключ ≤ запрос", atol=0, rtol=0)


def test_target_mask(impl):
    """target_mask: [B, 1, T, T] = causal И не PAD-ключ"""
    tgt = _tgt()
    m = impl.target_mask(tgt, PAD)
    causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    if m.shape[-2:] == (5, 5) and m.shape[0] == 1 and torch.equal(m.reshape(5, 5), causal):
        pytest.fail("в маске цели только causal-часть: PAD-ключи цели тоже нужно запретить (& padding_mask), "
                    "тогда маска станет [B, 1, T, T].", pytrace=False)
    assert m.dtype == torch.bool and m.shape == (2, 1, 5, 5), (
        f"bool [B, 1, T, T] = [2, 1, 5, 5], получено {m.dtype} {tuple(m.shape)}")
    want = causal & (tgt != PAD)[:, None, None, :]
    if torch.equal(m, causal.expand(2, 1, 5, 5)):
        pytest.fail("в маске цели только causal-часть: PAD-ключи цели тоже нужно запретить (& padding_mask).",
                    pytrace=False)
    if torch.equal(m, causal & (tgt != PAD)[:, None, :, None]):
        pytest.fail("паддинг запрещает запросы (строки), а должен — ключи (столбцы): padding_mask даёт [B, 1, 1, T].",
                    pytrace=False)
    assert torch.equal(m, want), f"ждём causal & padding по ключам, получено\n{m[0, 0].int()}"


def _layer_pair(impl, norm_first):
    torch.manual_seed(0)
    ours = impl.DecoderLayer(C, H, FF, norm_first)
    with torch.no_grad():
        for n in (ours.norm1, ours.norm2, ours.norm3):
            n.weight.uniform_(0.5, 1.5)
            n.bias.uniform_(-0.3, 0.3)
    th = nn.TransformerDecoderLayer(C, H, FF, dropout=0.0, batch_first=True, norm_first=norm_first)
    with torch.no_grad():
        for mine, theirs in ((ours.self_attn, th.self_attn), (ours.cross_attn, th.multihead_attn)):
            theirs.in_proj_weight.copy_(torch.cat([mine.q_proj.weight, mine.k_proj.weight, mine.v_proj.weight]))
            theirs.in_proj_bias.copy_(torch.cat([mine.q_proj.bias, mine.k_proj.bias, mine.v_proj.bias]))
            theirs.out_proj.load_state_dict(mine.out_proj.state_dict())
    th.linear1.load_state_dict(ours.ffn.linear1.state_dict())
    th.linear2.load_state_dict(ours.ffn.linear2.state_dict())
    for i in (1, 2, 3):
        getattr(th, f"norm{i}").load_state_dict(getattr(ours, f"norm{i}").state_dict())
    return ours, th.train()


def _inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 5, C, generator=g), torch.randn(2, 7, C, generator=g)


def _tmask(tgt):
    return torch.tril(torch.ones(tgt.shape[1], tgt.shape[1], dtype=torch.bool)) & (tgt != PAD)[:, None, None, :]


@pytest.mark.parametrize("norm_first", [False, True], ids=["post-ln", "pre-ln"])
def test_layer_matches_torch(impl, norm_first):
    """DecoderLayer == nn.TransformerDecoderLayer с теми же весами (T ≠ S, паддинг в источнике и цели)"""
    ours, th = _layer_pair(impl, norm_first)
    _no_builtin(ours)
    x, memory = _inputs()
    src, tgt = _src(), _tgt()
    try:
        out = ours(x, memory, _tmask(tgt), _src_mask(src))
    except RuntimeError as e:
        pytest.fail(f"слой упал при T = 5, S = 7: {e}. В cross-attention запросы — из x [B, T, C], ключи и значения "
                    "— из memory [B, S, C], маска — src_mask [B, 1, 1, S].", pytrace=False)
    assert out.shape == x.shape, f"выход [B, T, C] = {tuple(x.shape)}: длина — от цели, получено {tuple(out.shape)}"
    want = th(x, memory, tgt_mask=~torch.tril(torch.ones(5, 5, dtype=torch.bool)), tgt_key_padding_mask=tgt == PAD,
              memory_key_padding_mask=src == PAD)
    real = tgt != PAD
    if not torch.allclose(out[real], want[real], atol=1e-5, rtol=1e-4):
        no_mem = ours(x, torch.randn_like(memory), _tmask(tgt), _src_mask(src))
        if torch.allclose(no_mem, out, atol=1e-6):
            pytest.fail("выход не зависит от memory: в cross-attention ключи и значения должны браться из памяти "
                        "энкодера (cross_attn(x, memory, mask=src_mask)).", pytrace=False)
        pytest.fail("выход слоя не совпадает с nn.TransformerDecoderLayer: self_attn (tgt_mask) → остаток → norm1, "
                    "cross_attn к memory (src_mask) → остаток → norm2, FFN → остаток → norm3.", pytrace=False)


def test_cross_attention_sees_whole_source(impl):
    """Каждая позиция цели (и первая) видит весь источник: cross-attention не causal"""
    ours, _ = _layer_pair(impl, False)
    x, memory = _inputs(2)
    src, tgt = _src(), _tgt()
    mem2 = memory.clone()
    mem2[1, -1] += 3.0   # последний токен второго источника
    a = ours(x, memory, _tmask(tgt), _src_mask(src))
    b = ours(x, mem2, _tmask(tgt), _src_mask(src))
    changed = (a[1] - b[1]).abs().amax(-1) > 1e-6
    if not changed[0]:
        pytest.fail("позиция 0 цели не зависит от последнего токена источника.", pytrace=False)
    assert changed.all(), "каждая позиция цели должна видеть весь источник"
    mem3 = memory.clone()
    mem3[0, 3:] = 100 * torch.randn(4, C)   # PAD-позиции первого источника
    torch.testing.assert_close(ours(x, mem3, _tmask(tgt), _src_mask(src))[0], a[0], atol=1e-5, rtol=1e-4,
                               msg="память на PAD-позициях источника не должна влиять: src_mask в cross-attention")


def _decoders(impl, ref, norm_first=False):
    torch.manual_seed(0)
    a = ref.Decoder(V, C, H, FF, 2, 16, PAD, norm_first)
    with torch.no_grad():
        for n, p in a.named_parameters():
            if "norm" in n:
                p.add_(0.3 * torch.randn_like(p))
    b = impl.Decoder(V, C, H, FF, 2, 16, PAD, norm_first)
    b.load_state_dict(a.state_dict())
    return a, b


@pytest.mark.parametrize("norm_first", [False, True], ids=["post-ln", "pre-ln"])
def test_decoder_matches_reference(impl, ref, norm_first):
    """Decoder: маска цели, эмбеддинги, слои с общей памятью, финальная норма у Pre-LN"""
    a, b = _decoders(impl, ref, norm_first)
    _no_builtin(b)
    _, memory = _inputs(3)
    src, tgt = _src(), _tgt()
    out = b(tgt, memory, _src_mask(src))
    assert out.shape == (2, 5, C), f"выход [B, T, C] = [2, 5, {C}], получено {tuple(out.shape)}"
    real = tgt != PAD
    torch.testing.assert_close(out[real], a(tgt, memory, _src_mask(src))[real], atol=1e-5, rtol=1e-4,
                               msg="target_mask(tgt) → embed → слои (memory, src_mask) → self.norm")


def test_decoder_is_causal(impl, ref):
    """Изменение токена цели в позиции t + 1 не меняет выход на позициях ≤ t"""
    _, dec = _decoders(impl, ref)
    _, memory = _inputs(4)
    src = _src()
    tgt = torch.tensor([[1, 5, 6, 7, 8], [1, 9, 9, 9, 9]])
    base = dec(tgt, memory, _src_mask(src))
    for t in (0, 2):
        other = tgt.clone()
        other[:, t + 1:] = 3
        if not torch.allclose(dec(other, memory, _src_mask(src))[:, : t + 1], base[:, : t + 1], atol=1e-5):
            pytest.fail(f"выход декодера на позиции ≤ {t} зависит от токенов цели после {t} → утечка будущего. "
                        "При обучении (teacher forcing) декодер подсмотрит ответ.", pytrace=False)
