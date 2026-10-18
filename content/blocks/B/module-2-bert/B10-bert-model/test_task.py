"""Тесты «Модель BERT».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракул — transformers BertForMaskedLM / BertModel на случайных весах (если установлен).
С --weights: настоящий bert-base-uncased (~440 МБ) заполняет [MASK].
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

V, H, PAD, CLS, SEP, MASK = 60, 32, 0, 2, 3, 4


def _cfg(mod, **kw):
    return mod.BertConfig(vocab_size=V, hidden_size=H, num_hidden_layers=2, num_attention_heads=4,
                          intermediate_size=64, max_position_embeddings=16, **kw)


def _randomize_norms(model):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "LayerNorm" in n:
                p.add_(0.3 * torch.randn_like(p))


def _batch():
    ids = torch.tensor([[CLS, 10, 11, 12, SEP, 20, 21, SEP, PAD, PAD], [CLS, 13, 14, 15, 16, 17, SEP, 22, 23, SEP]])
    types = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 0, 0], [0] * 7 + [1] * 3])
    return ids, types, (ids != PAD).long()


def test_self_attention(impl):
    """BertSelfAttention: проекции, головы, attention с маской, склейка — без выходной проекции"""
    torch.manual_seed(0)
    att = impl.BertSelfAttention(_cfg(impl))
    x = torch.randn(2, 6, H)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1] * 6]).bool()[:, None, None, :]
    out = att(x, mask)
    assert out.shape == (2, 6, H), f"выход [B, T, H] = [2, 6, {H}], получено {tuple(out.shape)}"
    sp = lambda t: t.view(2, 6, 4, 8).transpose(1, 2)
    s = sp(att.query(x)) @ sp(att.key(x)).transpose(-2, -1) / math.sqrt(8)
    want = (s.masked_fill(~mask, float("-inf")).softmax(-1) @ sp(att.value(x))).transpose(1, 2).reshape(2, 6, H)
    torch.testing.assert_close(out, want, atol=1e-5, rtol=1e-4,
                               msg="query/key/value → головы [B, 4, T, 8] → softmax(q·kᵀ/√8 + маска) @ v → [B, T, H]")


def _layers(impl, ref):
    torch.manual_seed(0)
    a = ref.BertLayer(_cfg(ref))
    _randomize_norms(a)
    b = impl.BertLayer(_cfg(impl))
    b.load_state_dict(a.state_dict())
    return a, b


def _diagnose_layer(out, layer, x, mask) -> str:
    sa, ao, it, o = layer.attention.self, layer.attention.output, layer.intermediate, layer.output

    def post(act):
        h = ao.LayerNorm(x + ao.dense(sa(x, mask)))
        return o.LayerNorm(h + o.dense(act(it.dense(h))))

    def pre():
        h = x + ao.dense(sa(ao.LayerNorm(x), mask))
        return h + o.dense(F.gelu(it.dense(o.LayerNorm(h))))

    if torch.allclose(out, post(lambda t: F.gelu(t, approximate="tanh")), atol=1e-6):
        return ("GELU в tanh-аппроксимации (как в GPT-2), а у BERT точный: 0.5·x·(1 + erf(x/√2)) — F.gelu без "
                "approximate. Разница ~1e-3 — мала, но веса обучались с точным.")
    if torch.allclose(out, post(F.relu), atol=1e-5):
        return "активация ReLU (как в исходном Transformer), а у BERT — GELU."
    if torch.allclose(out, pre(), atol=1e-5):
        return "получился Pre-LN. BERT — Post-LN: LayerNorm(x + подслой(x)) после каждого подслоя."
    return ("выход слоя не совпадает с эталоном: self → attention.output.dense → LN(x + ·) → intermediate.dense → GELU "
            "→ output.dense → LN(h + ·).")


def test_layer(impl, ref):
    """BertLayer: Post-LN, точный GELU, совпадение с эталоном при неединичных LayerNorm"""
    a, b = _layers(impl, ref)
    x = torch.randn(2, 6, H, generator=torch.Generator().manual_seed(1))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1] * 6]).bool()[:, None, None, :]
    out = b(x, mask)
    assert out.shape == x.shape, f"слой не меняет форму, получено {tuple(out.shape)}"
    if not torch.allclose(out, a(x, mask), atol=1e-5, rtol=1e-4):
        pytest.fail(_diagnose_layer(out, b, x, mask), pytrace=False)


def _models(impl, ref, cls="BertModel", **kw):
    torch.manual_seed(0)
    a = getattr(ref, cls)(_cfg(ref), **kw)
    _randomize_norms(a)
    b = getattr(impl, cls)(_cfg(impl), **kw)
    b.load_state_dict(a.state_dict())
    return a, b


def test_model_and_pooler(impl, ref):
    """BertModel: (последовательность [B, T, H], pooled = tanh(dense(h[CLS]))), без pooler — None"""
    a, b = _models(impl, ref)
    ids, types, am = _batch()
    res = b(ids, types, am)
    assert isinstance(res, tuple) and len(res) == 2, "BertModel.forward возвращает (sequence_output, pooled_output)"
    seq, pooled = res
    assert seq.shape == (2, 10, H) and pooled.shape == (2, H), (
        f"sequence [B, T, H] и pooled [B, H], получено {tuple(seq.shape)} и {tuple(pooled.shape)}")
    want_seq, want_pooled = a(ids, types, am)
    real = am.bool()
    torch.testing.assert_close(seq[real], want_seq[real], atol=1e-5, rtol=1e-4, msg="эмбеддинги → слои с маской")
    if torch.allclose(pooled, seq[:, 0], atol=1e-6):
        pytest.fail("pooled — это не сам вектор [CLS], а tanh(pooler.dense(h[CLS])).", pytrace=False)
    torch.testing.assert_close(pooled, torch.tanh(b.pooler.dense(seq[:, 0])), atol=1e-6, rtol=1e-5,
                               msg="pooled = tanh(pooler.dense(h[:, 0]))")
    _, b2 = _models(impl, ref, add_pooling_layer=False)
    assert b2(ids, types, am)[1] is None, "без pooler (add_pooling_layer=False) pooled = None"
    torch.testing.assert_close(b(ids)[0], a(ids)[0], atol=1e-5, rtol=1e-4,
                               msg="без token_type_ids и attention_mask: сегмент 0, маска из единиц")


def test_padding_invariance(impl, ref):
    """attention_mask работает: выход настоящих токенов не зависит от паддинга"""
    _, b = _models(impl, ref)
    ids = torch.tensor([[CLS, 10, 11, 12, SEP]])
    alone = b(ids)[0]
    padded = torch.cat([ids, torch.full((1, 4), PAD)], 1)
    out = b(padded, None, (padded != PAD).long())[0]
    if not torch.allclose(out[:, :5], alone, atol=1e-5):
        pytest.fail("с паддингом выход настоящих токенов изменился: attention_mask (1 — токен, 0 — PAD) нужно "
                    "превратить в bool-маску ключей [B, 1, 1, T] и передать во все слои.", pytrace=False)


def test_mlm_head_and_tying(impl, ref):
    """BertForMaskedLM: логиты [B, T, V], decoder.weight — та же матрица, что word_embeddings"""
    a, b = _models(impl, ref, cls="BertForMaskedLM")
    ids, types, am = _batch()
    logits = b(ids, types, am)
    assert logits.shape == (2, 10, V), f"логиты [B, T, V] = [2, 10, {V}], получено {tuple(logits.shape)}"
    assert b.cls.predictions.decoder.weight is b.bert.embeddings.word_embeddings.weight, "матрица декодера связана с эмбеддингами"
    real = am.bool()
    x, _ = b.bert(ids, types, am)
    t = b.cls.predictions.transform
    if torch.allclose(logits[real], (x @ b.bert.embeddings.word_embeddings.weight.T)[real], atol=1e-4):
        pytest.fail("логиты посчитаны прямо из выхода энкодера. У BERT перед декодером ещё transform: dense → GELU → "
                    "LayerNorm, а у декодера есть bias.", pytrace=False)
    if torch.allclose(logits[real], b.cls.predictions.decoder(t.dense(x))[real], atol=1e-4):
        pytest.fail("в transform пропущены GELU и LayerNorm: dense → GELU → LayerNorm → decoder.", pytrace=False)
    torch.testing.assert_close(logits[real], a(ids, types, am)[real], atol=1e-4, rtol=1e-4,
                               msg="bert → transform (dense → GELU → LayerNorm) → decoder")


def test_matches_transformers(impl):
    """state_dict transformers.BertForMaskedLM загружается без переименований, логиты совпадают"""
    transformers = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    hc = transformers.BertConfig(vocab_size=V, hidden_size=H, num_hidden_layers=2, num_attention_heads=4,
                                 intermediate_size=64, max_position_embeddings=16, hidden_dropout_prob=0.0,
                                 attention_probs_dropout_prob=0.0)
    torch.manual_seed(2)
    hf = transformers.BertForMaskedLM(hc).eval()
    ours = impl.BertForMaskedLM(_cfg(impl))
    ours.load_state_dict(hf.state_dict(), strict=True)
    ids, types, am = _batch()
    with torch.no_grad():
        want = hf(input_ids=ids, token_type_ids=types, attention_mask=am).logits
    real = am.bool()
    torch.testing.assert_close(ours(ids, types, am)[real], want[real], atol=1e-4, rtol=1e-4,
                               msg="логиты должны совпасть с transformers BertForMaskedLM")


def test_learns_to_fill_mask(impl):
    """Smoke test: маленький BERT за 150 шагов учится восстанавливать закрытый токен по контексту с обеих сторон"""
    torch.manual_seed(0)
    model = impl.BertForMaskedLM(_cfg(impl))
    g = torch.Generator().manual_seed(3)
    # 8 «предложений»: середина однозначно определяется соседями слева и справа
    left, right = torch.randint(10, 30, (8,), generator=g), torch.randint(30, 50, (8,), generator=g)
    middle = torch.randint(50, 60, (8,), generator=g)
    ids = torch.stack([torch.full((8,), CLS), left, middle, right, torch.full((8,), SEP)], 1)
    masked = ids.clone()
    masked[:, 2] = MASK
    labels = torch.full_like(ids, -100)
    labels[:, 2] = ids[:, 2]
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = F.cross_entropy(model(masked).reshape(-1, V), labels.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = (model(masked)[:, 2].argmax(-1) == ids[:, 2]).float().mean().item()
    assert acc == 1.0, f"после обучения [MASK] восстановлен в {acc:.0%} примеров (лосс {loss.item():.3f})"


@pytest.mark.weights
def test_bert_base_uncased_fills_mask(impl, ref):
    """bert-base-uncased (веса HF, ~440 МБ) в твоей модели: «the capital of france is [MASK].» → paris"""
    transformers = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    hub = pytest.importorskip("huggingface_hub", reason="нужен huggingface_hub: uv sync --extra weights")
    from pathlib import Path
    from llmlab.core.bert import tokenize

    hf = transformers.BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased").eval()
    c = hf.config
    cfg = impl.BertConfig(vocab_size=c.vocab_size, hidden_size=c.hidden_size, num_hidden_layers=c.num_hidden_layers,
                          num_attention_heads=c.num_attention_heads, intermediate_size=c.intermediate_size,
                          max_position_embeddings=c.max_position_embeddings, type_vocab_size=c.type_vocab_size,
                          layer_norm_eps=c.layer_norm_eps)
    ours = impl.BertForMaskedLM(cfg).eval()
    ours.load_state_dict(hf.state_dict(), strict=True)
    vocab_file = hub.hf_hub_download("google-bert/bert-base-uncased", "vocab.txt")
    itos = Path(vocab_file).read_text(encoding="utf-8").splitlines()
    stoi = {t: i for i, t in enumerate(itos)}
    for text, want in [("the capital of france is [MASK] .", "paris"), ("i drank a cup of hot [MASK] this morning .", None)]:
        toks = ["[CLS]"] + [t for w in text.split() for t in (["[MASK]"] if w == "[MASK]" else tokenize(w, stoi))] + ["[SEP]"]
        ids = torch.tensor([[stoi[t] for t in toks]])
        with torch.no_grad():
            logits = ours(ids)
            torch.testing.assert_close(logits, hf(input_ids=ids).logits, atol=1e-3, rtol=1e-3)
        pos = toks.index("[MASK]")
        top = [itos[i] for i in logits[0, pos].topk(5).indices]
        if want:
            assert top[0] == want, f"ждём «{want}» на месте [MASK], топ-5: {top}"
