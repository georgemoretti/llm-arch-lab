"""Тесты «Энкодер».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул слоя — nn.TransformerEncoderLayer (dropout = 0) с теми же весами.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

C, H, FF, PAD, V = 32, 4, 64, 0, 30


def _ids():
    return torch.tensor([[5, 6, 7, 8, PAD, PAD], [3, 4, 5, 6, 7, 8]])


def _mask(ids):
    return (ids != PAD)[:, None, None, :]


def _no_builtin(mod):
    banned = (nn.TransformerEncoderLayer, nn.TransformerEncoder, nn.MultiheadAttention)
    used = sorted({type(m).__name__ for m in mod.modules() if isinstance(m, banned)})
    assert not used, f"использованы готовые слои PyTorch ({', '.join(used)}): здесь слой собирается руками"


def _layer_pair(impl, norm_first):
    torch.manual_seed(0)
    ours = impl.EncoderLayer(C, H, FF, norm_first)
    with torch.no_grad():   # неединичные LayerNorm: иначе часть ошибок с нормами не видна
        for n in (ours.norm1, ours.norm2):
            n.weight.uniform_(0.5, 1.5)
            n.bias.uniform_(-0.3, 0.3)
    th = nn.TransformerEncoderLayer(C, H, FF, dropout=0.0, batch_first=True, norm_first=norm_first)
    a = ours.self_attn
    with torch.no_grad():
        th.self_attn.in_proj_weight.copy_(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]))
        th.self_attn.in_proj_bias.copy_(torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]))
        th.self_attn.out_proj.load_state_dict(a.out_proj.state_dict())
    th.linear1.load_state_dict(ours.ffn.linear1.state_dict())
    th.linear2.load_state_dict(ours.ffn.linear2.state_dict())
    th.norm1.load_state_dict(ours.norm1.state_dict())
    th.norm2.load_state_dict(ours.norm2.state_dict())
    return ours, th.train()   # train(): без «быстрого пути» nested tensors; dropout = 0, так что детерминировано


def _diagnose_layer(out, layer, x, mask) -> str:
    a = lambda h: layer.self_attn(h, mask=mask)
    f, n1, n2 = layer.ffn, layer.norm1, layer.norm2
    variants = {
        "pre": lambda: (lambda h: h + f(n2(h)))(x + a(n1(x))),
        "post": lambda: (lambda h: n2(h + f(h)))(n1(x + a(x))),
        "no_res": lambda: n2(f(n1(a(x)))),
        "res_from_norm": lambda: (lambda h: h + f(n2(h)))(n1(x) + a(n1(x))),
    }
    want = "pre" if layer.norm_first else "post"
    got = next((k for k, fn in variants.items() if torch.allclose(out, fn(), atol=1e-5)), None)
    if got == "pre" and want == "post":
        return ("получился Pre-LN: x + f(LN(x)). При norm_first=False (как в статье) нормализуется сумма: "
                "LN(x + f(x)).")
    if got == "post" and want == "pre":
        return "получился Post-LN: LN(x + f(x)). При norm_first=True нужно x + f(LN(x))."
    if got == "no_res":
        return "нет остаточных связей: к выходу подслоя прибавляется его вход (x + f(…))."
    if got == "res_from_norm":
        return "остаток взят от LN(x), а не от x: в Pre-LN «магистраль» x не нормируется."
    return ("выход слоя не совпадает с nn.TransformerEncoderLayer: attention → остаток → norm1, FFN → остаток → "
            "norm2 (Post-LN); при norm_first=True — норма перед подслоем.")


def test_feedforward(impl):
    """FeedForward: linear2(ReLU(linear1(x)))"""
    torch.manual_seed(0)
    ffn = impl.FeedForward(C, FF)
    x = torch.randn(2, 5, C)
    out = ffn(x)
    assert out.shape == x.shape, f"форма не меняется: [2, 5, {C}], получено {tuple(out.shape)}"
    if torch.allclose(out, ffn.linear2(F.gelu(ffn.linear1(x))), atol=1e-6):
        pytest.fail("активация GELU, а в статье (и в nn.TransformerEncoderLayer по умолчанию) — ReLU. GELU придёт с "
                    "BERT (B10).", pytrace=False)
    torch.testing.assert_close(out, ffn.linear2(F.relu(ffn.linear1(x))), msg="linear1 → ReLU → linear2")


@pytest.mark.parametrize("norm_first", [False, True], ids=["post-ln", "pre-ln"])
def test_layer_matches_torch(impl, norm_first):
    """EncoderLayer == nn.TransformerEncoderLayer с теми же весами (Post-LN и Pre-LN, с паддингом)"""
    ours, th = _layer_pair(impl, norm_first)
    _no_builtin(ours)
    ids = _ids()
    x = torch.randn(2, 6, C, generator=torch.Generator().manual_seed(1))
    out = ours(x, _mask(ids))
    assert out.shape == x.shape, f"слой не меняет форму [B, S, C], получено {tuple(out.shape)}"
    want = th(x, src_key_padding_mask=ids == PAD)
    real = ids != PAD
    if not torch.allclose(out[real], want[real], atol=1e-5, rtol=1e-4):
        pytest.fail(_diagnose_layer(out, ours, x, _mask(ids)), pytrace=False)


def test_post_ln_output_normalized(impl):
    """Post-LN: выход слоя — после LayerNorm, у каждой позиции среднее 0 и дисперсия 1 (при единичных весах LN)"""
    torch.manual_seed(0)
    layer = impl.EncoderLayer(C, H, FF, norm_first=False)
    out = layer(5 * torch.randn(2, 6, C) + 3)
    torch.testing.assert_close(out.mean(-1), torch.zeros(2, 6), atol=1e-5, rtol=0,
                               msg="у Post-LN последняя операция слоя — LayerNorm: среднее по C равно 0")
    torch.testing.assert_close(out.var(-1, unbiased=False), torch.ones(2, 6), atol=1e-3, rtol=0,
                               msg="у Post-LN последняя операция слоя — LayerNorm: дисперсия по C равна 1")


def _encoders(impl, ref, norm_first):
    torch.manual_seed(0)
    a = ref.Encoder(V, C, H, FF, 2, 16, PAD, norm_first)
    with torch.no_grad():
        for n, p in a.named_parameters():
            if "norm" in n:
                p.add_(0.3 * torch.randn_like(p))
    b = impl.Encoder(V, C, H, FF, 2, 16, PAD, norm_first)
    b.load_state_dict(a.state_dict())
    return a, b


@pytest.mark.parametrize("norm_first", [False, True], ids=["post-ln", "pre-ln"])
def test_encoder_matches_reference(impl, ref, norm_first):
    """Encoder: эмбеддинги → слои с одной padding-маской → финальная норма (только Pre-LN)"""
    a, b = _encoders(impl, ref, norm_first)
    _no_builtin(b)
    ids = _ids()
    out = b(ids)
    assert out.shape == (2, 6, C), f"память [B, S, C] = [2, 6, {C}], получено {tuple(out.shape)}"
    want = a(ids)
    real = ids != PAD
    if norm_first and torch.allclose(out[real], b.layers[1](b.layers[0](b.embed(ids), _mask(ids)), _mask(ids))[real],
                                     atol=1e-5):
        pytest.fail("у Pre-LN нет финального LayerNorm (self.norm): выход последнего слоя не нормирован.",
                    pytrace=False)
    if torch.allclose(out[real], a.layers[1](a.layers[0](a.embed(ids)))[real], atol=1e-5):
        pytest.fail("маска не передана в слои: PAD-токены участвуют в attention.", pytrace=False)
    torch.testing.assert_close(out[real], want[real], atol=1e-5, rtol=1e-4,
                               msg="эмбеддинги → слои с padding-маской → self.norm")


def test_padding_invariance(impl, ref):
    """Лишние PAD справа не меняют память на настоящих токенах"""
    _, enc = _encoders(impl, ref, False)
    ids = torch.tensor([[5, 6, 7, 8]])
    base = enc(ids)
    for n_pad in (1, 4):
        padded = torch.cat([ids, torch.full((1, n_pad), PAD)], dim=1)
        if not torch.allclose(enc(padded)[:, :4], base, atol=1e-5):
            pytest.fail(f"с {n_pad} PAD справа выход на настоящих токенах изменился: в Encoder.forward нужна "
                        "padding-маска по src, и её нужно передать в каждый слой.", pytrace=False)


def test_bidirectional(impl, ref):
    """Энкодер видит весь текст: изменение последнего токена меняет память в позиции 0"""
    _, enc = _encoders(impl, ref, False)
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    other = ids.clone()
    other[0, -1] = 11
    assert not torch.allclose(enc(ids)[0, 0], enc(other)[0, 0], atol=1e-5), (
        "память в позиции 0 не зависит от последнего токена: в энкодере не должно быть causal-маски")
