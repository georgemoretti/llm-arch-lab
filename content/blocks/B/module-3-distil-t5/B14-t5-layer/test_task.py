"""Тесты «Слой T5».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул — transformers T5Model на случайных весах (если установлен) и ручные формулы.
С --weights: настоящий t5-small (~240 МБ) в твоей модели заполняет пропуски <extra_id_0>.
"""

import pytest
import torch
import torch.nn.functional as F

from llmlab.core.t5 import T5Config, relative_position_bias

V, C, H, D = 50, 32, 4, 8
PAD = 0


def _cfg(**kw):
    return T5Config(vocab_size=V, d_model=C, d_kv=D, d_ff=64, num_layers=2, num_heads=H,
                    relative_attention_num_buckets=8, relative_attention_max_distance=16, **kw)


def _randomize_norms(model):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "layer_norm" in n:
                p.add_(0.3 * torch.randn_like(p))


def test_layer_norm(impl):
    """T5LayerNorm: x / √(mean(x²) + eps) · weight — без вычитания среднего; mean(x²) в fp32"""
    torch.manual_seed(0)
    norm = impl.T5LayerNorm(C, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, C))
    x = torch.randn(2, 5, C) * 3 + 1.0                              # среднее заметно ≠ 0
    out = norm(x)
    assert out.shape == x.shape, f"форма не меняется, получено {tuple(out.shape)}"
    want = norm.weight * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    if torch.allclose(out, norm.weight * F.layer_norm(x, (C,), eps=1e-6), atol=1e-4):
        pytest.fail("получился LayerNorm: среднее вычтено. В T5 норма без центрирования: делим на RMS = √(mean(x²)), "
                    "среднее не трогаем (это RMSNorm, C01).", pytrace=False)
    if torch.allclose(out, norm.weight * x / x.std(-1, keepdim=True), atol=1e-4):
        pytest.fail("деление на std: в std вычитается среднее. Нужен RMS: √(mean(x²) + eps).", pytrace=False)
    torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-5, msg="weight · x / √(mean(x²) + eps)")
    tiny = torch.randn(1, C, generator=torch.Generator().manual_seed(9)) * 1e-3   # mean(x²) ~ eps
    torch.testing.assert_close(impl.T5LayerNorm(C, eps=1e-6)(tiny), tiny / torch.sqrt(tiny.pow(2).mean(-1, keepdim=True) + 1e-6),
                               atol=1e-5, rtol=1e-4, msg="|x| ~ 1e-3: eps стоит ПОД корнем, √(mean(x²) + eps), а не RMS + eps")
    half = torch.full((1, C), 300.0, dtype=torch.float16)
    out16 = impl.T5LayerNorm(C)(half)
    assert torch.isfinite(out16).all() and torch.allclose(out16.float(), torch.ones(1, C), atol=1e-2), (
        "fp16, x = 300: x² = 90 000 > 65 504 переполняется в inf. Считай mean(x²) в fp32 (x.float()).")


def _att(impl, has_bias=True, is_decoder=False, seed=0):
    torch.manual_seed(seed)
    a = impl.T5Attention(_cfg(), has_relative_attention_bias=has_bias, is_decoder=is_decoder)
    with torch.no_grad():
        for p in a.parameters():
            p.normal_(0, 0.5)
    return a


def _manual(att, x, kv=None, bias=None, mask=None):
    src = x if kv is None else kv
    sp = lambda t: t.view(t.shape[0], t.shape[1], H, D).transpose(1, 2)
    s = sp(att.q(x)) @ sp(att.k(src)).transpose(-2, -1)
    if bias is not None:
        s = s + bias
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    out = (s.softmax(-1) @ sp(att.v(src))).transpose(1, 2).reshape(x.shape[0], x.shape[1], H * D)
    return att.o(out)


def test_attention_scores_and_bias(impl):
    """T5Attention: scores = q·kᵀ БЕЗ 1/√D + относительный bias из таблицы; возвращает (выход, bias [1, H, T, S])"""
    att = _att(impl)
    x = torch.randn(2, 6, C, generator=torch.Generator().manual_seed(1))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1] * 6]).bool()[:, None, None, :]
    res = att(x, mask)
    assert isinstance(res, tuple) and len(res) == 2, "forward возвращает (выход, position_bias)"
    out, bias = res
    assert out.shape == (2, 6, C), f"выход [B, T, C] = [2, 6, {C}], получено {tuple(out.shape)}"
    want_bias = relative_position_bias(att.relative_attention_bias, 6, 6, True, 8, 16)
    assert bias is not None and bias.shape == (1, H, 6, 6), (
        f"второй выход — position_bias [1, H, T, S] = [1, {H}, 6, 6], получено {None if bias is None else tuple(bias.shape)}")
    torch.testing.assert_close(bias, want_bias, atol=1e-6, rtol=1e-6,
                               msg="энкодер: bias = relative_position_bias(таблица, T, S, bidirectional=True, …) из B13")
    real = mask[:, 0, 0]
    want = _manual(att, x, bias=want_bias, mask=mask)
    sp = lambda t: t.view(2, 6, H, D).transpose(1, 2)
    scaled = (sp(att.q(x)) @ sp(att.k(x)).transpose(-2, -1)) / D ** 0.5 + want_bias
    alt = att.o((scaled.masked_fill(~mask, float("-inf")).softmax(-1) @ sp(att.v(x))).transpose(1, 2).reshape(2, 6, H * D))
    if torch.allclose(out[real], alt[real], atol=1e-5):
        pytest.fail("scores поделены на √D. В T5 деления нет: q·kᵀ + bias. Масштаб заложен в инициализацию "
                    "(веса q ~ N(0, 1/(d_model·d_kv))), и обученные веса t5-small рассчитаны на scores без 1/√D.",
                    pytrace=False)
    if torch.allclose(out[real], _manual(att, x, mask=mask)[real], atol=1e-5):
        pytest.fail("относительный bias не прибавлен к scores: у слоя с таблицей (has_relative_attention_bias) "
                    "position_bias = relative_position_bias(...), и scores = q·kᵀ + position_bias.", pytrace=False)
    torch.testing.assert_close(out[real], want[real], atol=1e-5, rtol=1e-4,
                               msg="softmax(q·kᵀ + bias, маска) @ v → склейка голов → o")


def test_attention_bias_reuse_and_cross(impl):
    """Переданный position_bias используется как есть; без таблицы — нули; cross-attention: k, v из kv, T ≠ S"""
    att = _att(impl, has_bias=False)
    x = torch.randn(1, 4, C, generator=torch.Generator().manual_seed(2))
    out, bias = att(x)
    torch.testing.assert_close(bias, torch.zeros(1, H, 4, 4), atol=0, rtol=0,
                               msg="у слоя без таблицы (не слой 0) и без переданного bias — нули [1, H, T, S]")
    torch.testing.assert_close(out, _manual(att, x), atol=1e-5, rtol=1e-4, msg="без bias: softmax(q·kᵀ) @ v")
    given = torch.randn(1, H, 4, 4, generator=torch.Generator().manual_seed(3))
    out2, bias2 = att(x, position_bias=given)
    assert bias2 is given or torch.equal(bias2, given), "переданный position_bias возвращается без изменений"
    torch.testing.assert_close(out2, _manual(att, x, bias=given), atol=1e-5, rtol=1e-4,
                               msg="переданный position_bias прибавляется к scores (так слой 1 использует bias слоя 0)")
    mem = torch.randn(1, 7, C, generator=torch.Generator().manual_seed(4))
    out3, bias3 = att(x, kv=mem)
    assert out3.shape == (1, 4, C) and bias3.shape == (1, H, 4, 7), (
        f"cross-attention: 4 запроса, 7 ключей → выход [1, 4, {C}], bias [1, {H}, 4, 7]; "
        f"получено {tuple(out3.shape)}, {tuple(bias3.shape)}")
    torch.testing.assert_close(out3, _manual(att, x, kv=mem), atol=1e-5, rtol=1e-4,
                               msg="cross-attention: q из x, k и v из kv (памяти энкодера)")


def _blocks(impl, ref, is_decoder):
    torch.manual_seed(0)
    a = ref.T5Block(_cfg(), True, is_decoder)
    _randomize_norms(a)
    b = impl.T5Block(_cfg(), True, is_decoder)
    b.load_state_dict(a.state_dict())
    return a, b


def _diagnose_block(out, blk, x, mask):
    sa, ff = blk.layer[0], blk.layer[-1]
    att = lambda h: sa.SelfAttention(h, mask)[0]
    ffn = lambda h, act=F.relu: ff.DenseReluDense.wo(act(ff.DenseReluDense.wi(h)))
    h = sa.layer_norm(x + att(x))
    post = ff.layer_norm(h + ffn(h))
    if torch.allclose(out, post, atol=1e-5):
        return ("получился Post-LN (норма после сложения, как в BERT). T5 — Pre-LN: x + подслой(norm(x)), "
                "остаток идёт мимо нормы.")
    h = x + att(sa.layer_norm(x))
    if torch.allclose(out, h + ffn(ff.layer_norm(h), F.gelu), atol=1e-5):
        return "в FFN стоит GELU. У исходного T5 (и t5-small) — ReLU: wo(ReLU(wi(x))). GeGLU появился в T5 v1.1."
    if torch.allclose(out, ffn(ff.layer_norm(h)), atol=1e-5) or torch.allclose(out, att(sa.layer_norm(x)), atol=1e-5):
        return "потеряна остаточная связь: каждый подслой прибавляется к x, а не заменяет его."
    return "выход блока не совпадает: x + SelfAttention(LN(x)) → x + FFN(LN(x)), нормы — свои у каждого подслоя."


def test_encoder_block(impl, ref):
    """T5Block энкодера: Pre-LN, ReLU-FFN без bias, остатки; совпадение с эталоном при неединичных нормах"""
    a, b = _blocks(impl, ref, is_decoder=False)
    x = torch.randn(2, 6, C, generator=torch.Generator().manual_seed(5))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1] * 6]).bool()[:, None, None, :]
    res = b(x, mask)
    assert isinstance(res, tuple) and len(res) == 2, "T5Block.forward возвращает (x, position_bias)"
    out, bias = res
    want, want_bias = a(x, mask)
    assert bias is not None and bias.shape == (1, H, 6, 6), "блок 0 возвращает посчитанный bias — его возьмут следующие"
    if not torch.allclose(out, want, atol=1e-5, rtol=1e-4):
        pytest.fail(_diagnose_block(out, b, x, mask), pytrace=False)


def test_decoder_block(impl, ref):
    """T5Block декодера: self-attention (causal bias) → cross-attention к памяти → FFN"""
    a, b = _blocks(impl, ref, is_decoder=True)
    assert len(b.layer) == 3, "у блока декодера три подслоя: SelfAttention, EncDecAttention, FFN"
    x = torch.randn(1, 5, C, generator=torch.Generator().manual_seed(6))
    mem = torch.randn(1, 7, C, generator=torch.Generator().manual_seed(7))
    causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))[None, None]
    enc_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]]).bool()[:, None, None, :]
    out, bias = b(x, causal, None, mem, enc_mask)
    want, want_bias = a(x, causal, None, mem, enc_mask)
    torch.testing.assert_close(bias, want_bias, atol=1e-6, rtol=1e-6,
                               msg="у декодера bias causal (bidirectional=False): self.is_decoder решает режим")
    no_cross, _ = a(x, causal)
    if torch.allclose(out, no_cross, atol=1e-5):
        pytest.fail("cross-attention не применён: у декодера после self-attention идёт layer[1].EncDecAttention "
                    "(q из LN(x), k и v из памяти энкодера).", pytrace=False)
    torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-4,
                               msg="x + SA(LN(x)) → x + EncDecAttention(LN(x), память) → x + FFN(LN(x))")


def _models(impl, ref):
    torch.manual_seed(0)
    a = ref.T5Model(_cfg())
    _randomize_norms(a)
    b = impl.T5Model(_cfg())
    b.load_state_dict(a.state_dict())
    return a, b


def _batch():
    g = torch.Generator().manual_seed(8)
    src = torch.randint(1, V, (2, 9), generator=g)
    src[0, 6:] = PAD
    tgt = torch.randint(1, V, (2, 6), generator=g)
    tgt[1, 4:] = PAD
    return src, (src != PAD).long(), tgt, (tgt != PAD).long()


def test_stack_shares_bias(impl, ref):
    """T5Stack: таблица bias только у блока 0, остальные блоки получают его bias; финальная норма"""
    a, b = _models(impl, ref)
    assert hasattr(b.encoder.block[0].layer[0].SelfAttention, "relative_attention_bias"), (
        "таблица relative_attention_bias есть у блока 0 (так устроен T5)")
    assert not hasattr(b.encoder.block[1].layer[0].SelfAttention, "relative_attention_bias"), (
        "у блоков 1 и дальше своей таблицы нет: они получают bias блока 0")
    src, am, _, _ = _batch()
    seen = []
    orig = b.encoder.block[1].forward

    def spy(x, mask=None, position_bias=None, *args, **kw):
        seen.append(position_bias)
        return orig(x, mask, position_bias, *args, **kw)

    b.encoder.block[1].forward = spy
    out = b.encoder(src, am)
    b.encoder.block[1].forward = orig
    assert seen and seen[0] is not None, (
        "блок 1 получил position_bias = None: у него нет таблицы, и без bias слоя 0 он не видит позиций вовсе. "
        "Передавай bias, который вернул предыдущий блок, в следующий.")
    real = am.bool()
    want = a.encoder(src, am)
    x = b.shared(src)
    bias = None
    for blk in b.encoder.block:
        x, bias = blk(x, am.bool()[:, None, None, :], bias)
    if torch.allclose(out[real], x[real], atol=1e-5):
        pytest.fail("пропущена final_layer_norm: в Pre-LN выход последнего блока не нормирован, нужна финальная норма.",
                    pytrace=False)
    torch.testing.assert_close(out[real], want[real], atol=1e-5, rtol=1e-4,
                               msg="эмбеддинги (без позиций и масштаба) → блоки с общим bias → final_layer_norm")


def test_encoder_padding_and_decoder_causality(impl, ref):
    """Энкодер: PAD не влияет на настоящие токены. Декодер: позиция t не видит t + 1 …"""
    _, b = _models(impl, ref)
    ids = torch.tensor([[5, 6, 7, 8]])
    alone = b.encoder(ids)
    padded = torch.tensor([[5, 6, 7, 8, PAD, PAD, PAD]])
    out = b.encoder(padded, (padded != PAD).long())
    if not torch.allclose(out[:, :4], alone, atol=1e-5):
        pytest.fail("с паддингом выход энкодера изменился: attention_mask → bool-маска ключей [B, 1, 1, T] во все "
                    "блоки.", pytrace=False)
    mem = b.encoder(ids)
    tgt = torch.tensor([[1, 9, 10, 11, 12]])
    full = b.decoder(tgt, None, mem, None)
    changed = tgt.clone()
    changed[0, 3] = 20
    out2 = b.decoder(changed, None, mem, None)
    if not torch.allclose(full[:, :3], out2[:, :3], atol=1e-5):
        pytest.fail("выход декодера на позициях 0 … 2 зависит от токена на позиции 3: у декодера маска — padding "
                    "И causal [T, T] (torch.tril).", pytrace=False)


def test_matches_transformers(impl):
    """state_dict transformers.T5Model загружается без переименований; выходы энкодера и декодера совпадают"""
    tr = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    hc = tr.T5Config(vocab_size=V, d_model=C, d_kv=D, d_ff=64, num_layers=2, num_heads=H, dropout_rate=0.0,
                     relative_attention_num_buckets=8, relative_attention_max_distance=16, feed_forward_proj="relu")
    torch.manual_seed(1)
    hf = tr.T5Model(hc).eval()
    ours = impl.T5Model(_cfg())
    ours.load_state_dict(hf.state_dict(), strict=True)
    src, am, tgt, tm = _batch()
    with torch.no_grad():
        enc = hf.encoder(input_ids=src, attention_mask=am).last_hidden_state
        dec = hf(input_ids=src, attention_mask=am, decoder_input_ids=tgt, decoder_attention_mask=tm).last_hidden_state
    torch.testing.assert_close(ours.encoder(src, am)[am.bool()], enc[am.bool()], atol=1e-5, rtol=1e-4,
                               msg="выход энкодера должен совпасть с transformers T5Model.encoder")
    torch.testing.assert_close(ours(src, tgt, am, tm)[tm.bool()], dec[tm.bool()], atol=1e-5, rtol=1e-4,
                               msg="выход декодера должен совпасть с transformers T5Model")


@pytest.mark.weights
def test_t5_small_fills_sentinels(impl):
    """t5-small (веса HF, ~240 МБ) в твоей модели: логиты = transformers, пропуски <extra_id_0> заполняются"""
    tr = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    hf = tr.T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").eval()
    tok = tr.AutoTokenizer.from_pretrained("google-t5/t5-small")
    c = hf.config
    cfg = T5Config(vocab_size=c.vocab_size, d_model=c.d_model, d_kv=c.d_kv, d_ff=c.d_ff, num_layers=c.num_layers,
                   num_decoder_layers=c.num_decoder_layers, num_heads=c.num_heads,
                   relative_attention_num_buckets=c.relative_attention_num_buckets,
                   relative_attention_max_distance=c.relative_attention_max_distance,
                   layer_norm_epsilon=c.layer_norm_epsilon)
    ours = impl.T5Model(cfg).eval()
    sd = {k: v for k, v in hf.state_dict().items() if k != "lm_head.weight"}
    ours.load_state_dict(sd, strict=True)
    src = tok("The <extra_id_0> walks in <extra_id_1> park.", return_tensors="pt").input_ids
    dec = torch.tensor([[c.decoder_start_token_id]])
    with torch.no_grad():
        for _ in range(8):                                    # жадная генерация своей моделью
            h = ours(src, dec)[:, -1]
            logits = (h * c.d_model ** -0.5) @ ours.shared.weight.T    # связанная голова: масштаб d_model^−½
            dec = torch.cat([dec, logits.argmax(-1, keepdim=True)], 1)
        want = hf(input_ids=src, decoder_input_ids=dec).logits
        got = (ours(src, dec) * c.d_model ** -0.5) @ ours.shared.weight.T
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3, msg="логиты своей модели = transformers")
    text = tok.decode(dec[0], skip_special_tokens=False)
    assert "<extra_id_0>" in text and "<extra_id_1>" in text, f"ответ T5 — sentinel и спан за ним, получено: {text}"
