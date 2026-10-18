"""Тесты «Слой ModernBERT».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракулы — ручные формулы, RoPE из C03 и transformers ModernBertModel на случайных весах (если установлен).
С --weights: настоящий ModernBERT-base (~600 МБ) в твоей модели заполняет [MASK].
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llmlab.core.llama import apply_rope, rope_cos_sin
from llmlab.core.modernbert import ModernBertConfig

V, C, H = 60, 32, 4
D = C // H
PAD = 0


@pytest.fixture(autouse=True)
def _forbid_builtin_attention(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Использована готовая реализация attention (F.scaled_dot_product_attention / "
                             "nn.MultiheadAttention). Здесь механизм пишется руками: q·kᵀ/√D → маска → softmax → @v.")

    monkeypatch.setattr(F, "scaled_dot_product_attention", forbidden)
    monkeypatch.setattr(nn.MultiheadAttention, "forward", forbidden)


def _cfg(**kw):
    base = dict(vocab_size=V, hidden_size=C, intermediate_size=48, num_hidden_layers=4, num_attention_heads=H,
                local_attention=8)
    base.update(kw)
    return ModernBertConfig(**base)


def _randomize_norms(model):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.add_(0.3 * torch.randn_like(p))


def test_layer_types(impl):
    """Слой 0 и каждый 3-й — глобальные, остальные — локальные (ModernBERT-base: 22 слоя, 8 глобальных)"""
    got = impl.layer_types(22, 3)
    assert isinstance(got, list) and len(got) == 22, f"нужен список из 22 строк, получено {got!r}"
    assert set(got) <= {"full_attention", "sliding_attention"}, (
        f"типы называются как в HF: 'full_attention' и 'sliding_attention', получено {sorted(set(got))}")
    want = ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(22)]
    if got == ["full_attention" if i % 3 == 2 else "sliding_attention" for i in range(22)]:
        pytest.fail("глобальные слои 2, 5, 8, …: сдвиг на единицу. У ModernBERT глобальный — слой 0 и дальше каждый "
                    "третий (0, 3, 6, …, 21): первый слой видит весь текст.", pytrace=False)
    assert got == want, ("ModernBERT-base: global_attn_every_n_layers = 3 → глобальные слои 0, 3, 6, …, 21 "
                         f"(8 из 22). Получено: глобальные {[i for i, t in enumerate(got) if t == 'full_attention']}")
    assert impl.layer_types(4, 2) == ["full_attention", "sliding_attention"] * 2, "шаг 2: глобальные 0 и 2"
    assert impl.layer_types(3, 1) == ["full_attention"] * 3, "шаг 1: все слои глобальные"


def test_local_window_mask(impl):
    """Окно local_attention = W — это ±W/2 токенов: |i − j| ≤ W // 2"""
    m = impl.local_window_mask(10, 4)
    assert m.shape == (10, 10) and m.dtype == torch.bool, (
        f"маска [T, T] типа bool, получено {tuple(m.shape)}, {m.dtype}")
    i, j = torch.arange(10)[:, None], torch.arange(10)[None, :]
    want = (i - j).abs() <= 2
    if torch.equal(m, (i - j).abs() <= 4):
        pytest.fail("окно ±W вместо ±W/2: local_attention = 128 у ModernBERT — это ширина всего окна, по 64 токена "
                    "в каждую сторону (|i − j| ≤ W // 2).", pytrace=False)
    if torch.equal(m, (i - j).abs() < 2):
        pytest.fail("граница не включена: ключ на расстоянии ровно W/2 виден (|i − j| ≤ W // 2, как в HF).",
                    pytrace=False)
    if torch.equal(m, (j <= i) & (i - j <= 2)):
        pytest.fail("окно только назад (causal): энкодер двунаправленный, окно симметричное — и влево, и вправо.",
                    pytrace=False)
    assert torch.equal(m, want), "запрос i видит ключи i − W/2 … i + W/2"
    big = impl.local_window_mask(300, 128)
    assert int(big[150].sum()) == 129 and int(big[0].sum()) == 65, (
        "ModernBERT-base, W = 128: в середине текста запрос видит 129 ключей (64 + сам + 64), у края — 65")


def test_mlp_geglu(impl):
    """GeGLU: Wi(x) = [вход | gate], Wo(GELU(вход) · gate); точный GELU, без bias"""
    torch.manual_seed(0)
    mlp = impl.ModernBertMLP(_cfg())
    x = torch.randn(2, 5, C)
    out = mlp(x)
    assert out.shape == x.shape, f"выход той же формы [B, T, C], получено {tuple(out.shape)}"
    a, g = mlp.Wi(x).chunk(2, -1)
    want = mlp.Wo(F.gelu(a) * g)
    variants = {
        "активация досталась gate, а не «входу»: в HF первая половина Wi — вход (с GELU), вторая — gate (линейно).":
            mlp.Wo(a * F.gelu(g)),
        "SiLU вместо GELU: это SwiGLU из Llama (C02). У ModernBERT — GeGLU, hidden_activation = \"gelu\".":
            mlp.Wo(F.silu(a) * g),
        "GELU в tanh-приближении: у ModernBERT точный GELU через erf (F.gelu без approximate).":
            mlp.Wo(F.gelu(a, approximate="tanh") * g),
        "нет умножения на gate: Wo(GELU(вход)) — обычный FFN, половина Wi пропадает.":
            mlp.Wo(F.gelu(a)),
    }
    for msg, alt in variants.items():
        if torch.allclose(out, alt, atol=1e-6):
            pytest.fail(msg, pytrace=False)
    torch.testing.assert_close(out, want, atol=1e-6, rtol=1e-5, msg="Wo(GELU(Wi(x)[:, :, :I]) · Wi(x)[:, :, I:])")


def _att(impl, layer_idx=0, **kw):
    torch.manual_seed(layer_idx)
    att = impl.ModernBertAttention(_cfg(**kw), layer_idx)
    with torch.no_grad():
        for p in att.parameters():
            p.normal_(0, 0.3)
    return att


def _manual(att, x, mask=None, theta=None, scale=True, rope=True):
    B, T, _ = x.shape
    q, k, v = att.Wqkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
    if rope:
        cos, sin = rope_cos_sin(torch.arange(T), D, theta or att.rope_theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    s = q @ k.transpose(-2, -1) / (D ** 0.5 if scale else 1.0)
    if mask is not None:
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
    return att.Wo((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, C))


def test_attention(impl):
    """Wqkv → q, k, v; RoPE с θ своего типа слоя; q·kᵀ/√D; маска; Wo"""
    g, loc = _att(impl, 0), _att(impl, 1)
    assert g.attention_type == "full_attention" and loc.attention_type == "sliding_attention"
    x = torch.randn(2, 7, C, generator=torch.Generator().manual_seed(1))
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1] * 7]).bool()[:, None, None, :]
    out = g(x, mask)
    assert out.shape == (2, 7, C), f"выход [B, T, C] = [2, 7, {C}], получено {tuple(out.shape)}"
    real = mask[:, 0, 0]
    if torch.allclose(out[real], _manual(g, x, mask, rope=False)[real], atol=1e-5):
        pytest.fail("RoPE не применён: в ModernBERT нет позиционных эмбеддингов на входе, позиции — только поворот "
                    "q и k (rope_cos_sin + apply_rope из C03).", pytrace=False)
    if torch.allclose(out[real], _manual(g, x, mask, scale=False)[real], atol=1e-5):
        pytest.fail("scores не поделены на √D. В отличие от T5 (B14), у ModernBERT обычное масштабирование q·kᵀ/√D.",
                    pytrace=False)
    if torch.allclose(out[real], _alt_split(g, x, mask)[real], atol=1e-5):
        pytest.fail("q, k, v нарезаны не так: выход Wqkv — [q всех голов | k всех голов | v всех голов], то есть "
                    "view(B, T, 3, H, D), а не view(B, T, H, 3, D).", pytrace=False)
    torch.testing.assert_close(out[real], _manual(g, x, mask)[real], atol=1e-5, rtol=1e-4,
                               msg="глобальный слой: RoPE с θ = global_rope_theta (160 000), q·kᵀ/√D, маска, Wo")
    out_l = loc(x, mask)
    if torch.allclose(out_l[real], _manual(loc, x, mask, theta=160_000.0)[real], atol=1e-5):
        pytest.fail("локальный слой повернул q и k с θ = 160 000: у локальных слоёв своя θ = local_rope_theta "
                    "(10 000) — она уже в self.rope_theta.", pytrace=False)
    torch.testing.assert_close(out_l[real], _manual(loc, x, mask)[real], atol=1e-5, rtol=1e-4,
                               msg="локальный слой: RoPE с θ = local_rope_theta (10 000)")


def _alt_split(att, x, mask):
    B, T, _ = x.shape
    q, k, v = att.Wqkv(x).view(B, T, H, 3, D).permute(3, 0, 2, 1, 4)
    cos, sin = rope_cos_sin(torch.arange(T), D, att.rope_theta)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    s = (q @ k.transpose(-2, -1) / D ** 0.5).masked_fill(~mask, torch.finfo(x.dtype).min)
    return att.Wo((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, C))


def test_attention_fully_masked_row_is_finite(impl):
    """Запрос, которому не виден ни один ключ (PAD в локальном окне из PAD), не даёт nan"""
    att = _att(impl, 1)
    x = torch.randn(1, 6, C, generator=torch.Generator().manual_seed(2))
    mask = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    mask[..., 5, :] = False                                             # строке 5 не виден ни один ключ
    out = att(x, mask)
    assert torch.isfinite(out).all(), (
        "в выходе nan: строка scores целиком −inf → softmax делит 0 на 0. Так бывает у PAD-запроса в локальном слое, "
        "когда всё окно — PAD, и nan потом расползается по слоям. Закрывай ключи значением torch.finfo(dtype).min.")
    torch.testing.assert_close(out[:, :5], _manual(att, x, mask)[:, :5], atol=1e-5, rtol=1e-4,
                               msg="остальные строки считаются как обычно")


def _layers(impl, ref, idx):
    torch.manual_seed(0)
    a = ref.ModernBertEncoderLayer(_cfg(), idx)
    _randomize_norms(a)
    b = impl.ModernBertEncoderLayer(_cfg(), idx)
    b.load_state_dict(a.state_dict())
    return a, b


def test_encoder_layer(impl, ref):
    """Pre-LN: x + attn(attn_norm(x)) → x + mlp(mlp_norm(x)); у слоя 0 attn_norm = Identity"""
    for idx in (0, 1):
        a, b = _layers(impl, ref, idx)
        x = torch.randn(2, 6, C, generator=torch.Generator().manual_seed(3)) * 2 + 0.5
        mask = torch.ones(2, 1, 1, 6, dtype=torch.bool)
        out = b(x, mask)
        assert out.shape == x.shape, f"слой {idx}: форма не меняется, получено {tuple(out.shape)}"
        att = lambda h: b.attn(h, mask)
        h = b.attn_norm(x + att(x))
        if torch.allclose(out, b.mlp_norm(h + b.mlp(h)), atol=1e-5):
            pytest.fail("получился Post-LN (норма после сложения, как в BERT). ModernBERT — Pre-LN: x + подслой(norm(x)).",
                        pytrace=False)
        hn = b.attn_norm(x)
        h = hn + att(hn)
        if idx == 1 and torch.allclose(out, h + b.mlp(b.mlp_norm(h)), atol=1e-5):
            pytest.fail("остаток прибавлен к norm(x), а не к x: x = x + attn(attn_norm(x)) — мимо нормы.",
                        pytrace=False)
        torch.testing.assert_close(out, a(x, mask), atol=1e-5, rtol=1e-4,
                                   msg=f"слой {idx}: x + attn(attn_norm(x), mask), затем x + mlp(mlp_norm(x))")


def _models(impl, ref, **kw):
    torch.manual_seed(0)
    a = ref.ModernBertModel(_cfg(**kw))
    _randomize_norms(a)
    b = impl.ModernBertModel(_cfg(**kw))
    b.load_state_dict(a.state_dict())
    return a, b


def test_local_layers_see_only_window(impl, ref):
    """Если глобальный слой 0 «выключить», выход в позиции i зависит только от токенов в пределах окон локальных слоёв"""
    _, b = _models(impl, ref, num_hidden_layers=3, local_attention=4)    # слой 0 глобальный, 1 и 2 — окно ±2
    with torch.no_grad():
        b.layers[0].attn.Wo.weight.zero_()                             # глобальный слой ничего не смешивает
    ids = torch.randint(1, V, (1, 16), generator=torch.Generator().manual_seed(4))
    base = b(ids)
    far, near = ids.clone(), ids.clone()
    far[0, 13] = (ids[0, 13] % (V - 1)) + 1                            # 13 − 8 = 5 > 2 · 2
    near[0, 11] = (ids[0, 11] % (V - 1)) + 1                           # 11 − 8 = 3 ≤ 4
    if not torch.allclose(b(far)[0, 8], base[0, 8], atol=1e-5):
        pytest.fail("позиция 8 зависит от токена 13, хотя два локальных слоя с окном ±2 видят не дальше ±4: "
                    "в слои \"sliding_attention\" не попала маска окна (PAD И local_window_mask).", pytrace=False)
    assert not torch.allclose(b(near)[0, 8], base[0, 8], atol=1e-5), (
        "позиция 8 не зависит от токена 11 (расстояние 3): два слоя с окном ±2 дотягиваются до ±4. Окно слишком узкое?")


def test_padding_and_long_pad(impl, ref):
    """PAD не влияет на настоящие токены; длинный хвост PAD длиннее окна не даёт nan"""
    a, b = _models(impl, ref)
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    alone = b(ids)
    padded = torch.tensor([[5, 6, 7, 8, 9, 10] + [PAD] * 14])
    out = b(padded, (padded != PAD).long())
    assert out.shape == (1, 20, C), f"выход [B, T, C], получено {tuple(out.shape)}"
    assert torch.isfinite(out).all(), (
        "nan в выходе: у PAD-запросов в конце хвоста всё локальное окно — PAD. Маскируй значением finfo.min, не −inf.")
    if not torch.allclose(out[:, :6], alone, atol=1e-5):
        pytest.fail("с паддингом выход на настоящих токенах изменился: attention_mask → bool [B, 1, 1, T] — маска "
                    "ключей в КАЖДОМ слое, и в локальных тоже (PAD И окно).", pytrace=False)
    torch.testing.assert_close(out[:, :6], a(padded, (padded != PAD).long())[:, :6], atol=1e-5, rtol=1e-4,
                               msg="эмбеддинги → слои со своими масками → final_norm")


def test_model_matches_reference(impl, ref):
    """Модель целиком: 4 слоя (глобальный, локальный, локальный, глобальный), T = 20 > окна"""
    a, b = _models(impl, ref)
    ids = torch.randint(1, V, (2, 20), generator=torch.Generator().manual_seed(5))
    ids[0, 12:] = PAD
    am = (ids != PAD).long()
    out = b(ids, am)
    real = am.bool()
    x = b.embeddings(ids)
    for layer in b.layers:
        x = layer(x, am.bool()[:, None, None, :] if layer.attention_type == "full_attention"
                  else am.bool()[:, None, None, :] & impl.local_window_mask(20, 8))
    if torch.allclose(out[real], x[real], atol=1e-5):
        pytest.fail("пропущена final_norm: в Pre-LN выход последнего слоя не нормирован.", pytrace=False)
    torch.testing.assert_close(out[real], a(ids, am)[real], atol=1e-5, rtol=1e-4,
                               msg="выход ModernBertModel совпадает с эталоном на настоящих токенах")


def _hf_config(tr, **kw):
    return tr.ModernBertConfig(vocab_size=V, hidden_size=C, intermediate_size=48, num_hidden_layers=4,
                               num_attention_heads=H, local_attention=8, global_attn_every_n_layers=3,
                               pad_token_id=PAD, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2,
                               global_rope_theta=160_000.0, local_rope_theta=10_000.0, attn_implementation="eager", **kw)


def test_matches_transformers(impl):
    """state_dict transformers.ModernBertModel загружается без переименований; выходы совпадают (T = 20 > окна)"""
    tr = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    if not hasattr(tr, "ModernBertModel"):
        pytest.skip("в установленной версии transformers нет ModernBertModel (нужна ≥ 4.48)")
    torch.manual_seed(1)
    hf = tr.ModernBertModel(_hf_config(tr)).eval()
    _randomize_norms(hf)
    ours = impl.ModernBertModel(_cfg())
    ours.load_state_dict(hf.state_dict(), strict=True)
    ids = torch.randint(1, V, (2, 20), generator=torch.Generator().manual_seed(6))
    ids[0, 13:] = PAD
    am = (ids != PAD).long()
    with torch.no_grad():
        want = hf(input_ids=ids, attention_mask=am).last_hidden_state
    real = am.bool()
    torch.testing.assert_close(ours(ids, am)[real], want[real], atol=1e-5, rtol=1e-4,
                               msg="выход должен совпасть с transformers ModernBertModel (eager attention)")


@pytest.mark.weights
def test_modernbert_base_fill_mask(impl):
    """ModernBERT-base (веса HF, ~600 МБ) в твоей модели: логиты = transformers, «столица Франции» → Paris"""
    tr = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    name = "answerdotai/ModernBERT-base"
    hf = tr.ModernBertForMaskedLM.from_pretrained(name, attn_implementation="eager").eval()
    tok = tr.AutoTokenizer.from_pretrained(name)
    c = hf.config
    cfg = ModernBertConfig(vocab_size=c.vocab_size, hidden_size=c.hidden_size, intermediate_size=c.intermediate_size,
                           num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads,
                           global_attn_every_n_layers=3, local_attention=c.local_attention,
                           norm_eps=c.norm_eps, pad_token_id=c.pad_token_id)
    ours = impl.ModernBertForMaskedLM(cfg).eval()
    ours.load_state_dict(hf.state_dict(), strict=True)
    enc = tok("The capital of France is [MASK].", return_tensors="pt")
    with torch.no_grad():
        got = ours(enc.input_ids, enc.attention_mask)
        want = hf(**enc).logits
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3, msg="логиты своей модели = transformers")
    pos = (enc.input_ids[0] == tok.mask_token_id).nonzero()[0, 0]
    word = tok.decode(got[0, pos].argmax()).strip()
    assert word == "Paris", f"ModernBERT-base должен заполнить [MASK] словом Paris, получено {word!r}"
