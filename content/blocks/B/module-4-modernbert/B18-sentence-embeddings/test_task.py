"""Тесты «Sentence-эмбеддинги».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter).

Энкодеры — эталоны курса на случайных весах: ModernBertModel (B16) и BertModel (B10).
С --weights: all-MiniLM-L6-v2 (mean pooling) и bge-small-en-v1.5 (CLS pooling) в BertModel курса.
"""

import pytest
import torch
import torch.nn.functional as F

from llmlab.core.bert import BertConfig, BertModel, build_inputs, pad_batch
from llmlab.core.modernbert import ModernBertConfig, ModernBertModel

V, C, PAD = 40, 32, 0


@pytest.fixture(autouse=True)
def _forbid_builtin_normalize(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Использована готовая нормировка (F.normalize / F.cosine_similarity). Здесь она пишется "
                             "руками: x / ‖x‖.")

    monkeypatch.setattr(F, "normalize", forbidden)
    monkeypatch.setattr(F, "cosine_similarity", forbidden)


def _modernbert(seed=0):
    torch.manual_seed(seed)
    return ModernBertModel(ModernBertConfig(vocab_size=V, num_hidden_layers=2, local_attention=8)).eval()


def _bert(seed=0):
    torch.manual_seed(seed)
    return BertModel(BertConfig(vocab_size=V, hidden_size=C)).eval()


def test_mean_pool(impl):
    """Среднее только по настоящим токенам; PAD (с любыми значениями) не влияет; пустой текст → нули, не nan"""
    g = torch.Generator().manual_seed(0)
    h = torch.randn(3, 5, 4, generator=g)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    out = impl.mean_pool(h, mask)
    assert out.shape == (3, 4), f"[B, T, C] → [B, C] = [3, 4], получено {tuple(out.shape)}"
    if torch.allclose(out[:2], h[:2].mean(1), atol=1e-6):
        pytest.fail("в среднее вошли PAD-токены: h.mean(1) усредняет все T позиций.", pytrace=False)
    if torch.allclose(out[0], (h[0] * mask[0, :, None]).sum(0) / 5, atol=1e-6):
        pytest.fail("PAD обнулены, но сумма поделена на T = 5, а не на число настоящих токенов (3): эмбеддинг "
                    "короткого текста съёживается тем сильнее, чем больше паддинга.", pytrace=False)
    torch.testing.assert_close(out[0], h[0, :3].mean(0), atol=1e-6, rtol=1e-5, msg="текст 0: среднее трёх токенов")
    torch.testing.assert_close(out[1], h[1].mean(0), atol=1e-6, rtol=1e-5, msg="текст 1 без PAD: среднее всех")
    assert torch.isfinite(out[2]).all() and (out[2] == 0).all(), (
        "пустой текст (маска из нулей): деление 0/0 даёт nan; ограничь знаменатель снизу (clamp(min=1e-9)).")
    h2 = h.clone()
    h2[0, 3:] = 1e4                                                    # мусор на месте PAD
    torch.testing.assert_close(impl.mean_pool(h2, mask)[0], out[0], atol=1e-6, rtol=1e-5,
                               msg="значения на PAD-позициях не должны влиять на среднее")


def test_cls_pool(impl):
    """CLS pooling — вектор позиции 0"""
    h = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(1))
    out = impl.cls_pool(h)
    assert out.shape == (2, 4), f"[B, T, C] → [B, C], получено {tuple(out.shape)}"
    torch.testing.assert_close(out, h[:, 0], atol=0, rtol=0, msg="CLS pooling = h[:, 0]")


def test_l2_and_cosine(impl):
    """L2: норма 1, направление то же, ноль → ноль; косинусы всех пар [N, M]"""
    g = torch.Generator().manual_seed(2)
    x = torch.randn(4, 6, generator=g) * 5
    n = impl.l2_normalize(x)
    torch.testing.assert_close(n.norm(dim=-1), torch.ones(4), atol=1e-6, rtol=1e-6, msg="после L2 норма каждой строки 1")
    torch.testing.assert_close(n * x.norm(dim=-1, keepdim=True), x, atol=1e-5, rtol=1e-5,
                               msg="L2 меняет длину, но не направление: x / ‖x‖")
    z = impl.l2_normalize(torch.zeros(1, 6))
    assert torch.isfinite(z).all() and (z == 0).all(), "нулевой вектор: ‖x‖ = 0 → clamp(min=eps), без nan"
    a, b = torch.randn(3, 6, generator=g), torch.randn(5, 6, generator=g) * 3
    cos = impl.cosine_matrix(a, b)
    assert cos.shape == (3, 5), f"[N, C], [M, C] → [N, M] = [3, 5], получено {tuple(cos.shape)}"
    want = torch.tensor([[float(a[i] @ b[j] / (a[i].norm() * b[j].norm())) for j in range(5)] for i in range(3)])
    if torch.allclose(cos, a @ b.T, atol=1e-5):
        pytest.fail("получились скалярные произведения, а не косинусы: перед умножением нормируй оба набора.",
                    pytrace=False)
    torch.testing.assert_close(cos, want, atol=1e-5, rtol=1e-5, msg="cos(a_i, b_j) = a_i · b_j / (‖a_i‖ ‖b_j‖)")


def _ids(rows, T):
    ids = torch.full((len(rows), T), PAD)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
    return ids, (ids != PAD).long()


def test_bi_encoder_padding_invariance(impl):
    """Эмбеддинг текста не зависит от длины паддинга и от соседей по батчу (ModernBERT и BERT)"""
    for enc in (_modernbert(), _bert()):
        model = impl.BiEncoder(enc)
        text = [5, 6, 7, 8]
        ids1, m1 = _ids([text], 4)
        ids2, m2 = _ids([text, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], 12)
        e1, e2 = model(ids1, m1), model(ids2, m2)
        assert e1.shape == (1, C) and e2.shape == (2, C), f"[B, C], получено {tuple(e1.shape)}, {tuple(e2.shape)}"
        assert torch.allclose(e1[0], e2[0], atol=1e-5), (
            f"{type(enc).__name__}: эмбеддинг текста из 4 токенов изменился, когда в батч добавили длинный текст и "
            "дополнили его PAD до 12. Mean pooling должен усреднять только настоящие токены (маска, деление на их "
            "число), и маска должна попасть в энкодер.")
        torch.testing.assert_close(e1.norm(dim=-1), torch.ones(1), atol=1e-5, rtol=1e-5,
                                   msg="normalize=True: эмбеддинги единичной длины")


def test_bi_encoder_pooling_modes(impl):
    """pooling='mean' → mean_pool, 'cls' → h[:, 0]; normalize=False оставляет длину"""
    enc = _modernbert(1)
    ids, m = _ids([[1, 5, 6, 7, 2], [1, 8, 2]], 5)
    h = enc(ids, m)
    mean = impl.BiEncoder(enc, "mean", normalize=False)(ids, m)
    cls = impl.BiEncoder(enc, "cls", normalize=False)(ids, m)
    want_mean = torch.stack([h[0].mean(0), h[1, :3].mean(0)])
    torch.testing.assert_close(mean, want_mean, atol=1e-5, rtol=1e-4, msg="mean: среднее скрытых состояний по маске")
    torch.testing.assert_close(cls, h[:, 0], atol=1e-6, rtol=1e-5, msg="cls: скрытое состояние позиции 0")
    normed = impl.BiEncoder(enc, "cls")(ids, m)
    torch.testing.assert_close(normed, h[:, 0] / h[:, 0].norm(dim=-1, keepdim=True), atol=1e-5, rtol=1e-4,
                               msg="normalize=True (по умолчанию): после pooling — L2")


def test_cross_encoder(impl):
    """Cross-encoder: одна оценка на пару [CLS] q [SEP] d [SEP] — Linear по [CLS], форма [B]; token type доходит до BERT"""
    enc = _bert(2)
    torch.manual_seed(3)
    model = impl.CrossEncoder(enc, C)
    pairs = [build_inputs([5, 6], [7, 8, 9], cls_id=1, sep_id=2, max_len=32), build_inputs([5], [10, 11], cls_id=1, sep_id=2, max_len=32)]
    batch = pad_batch(pairs, PAD)
    s = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
    assert s.shape == (2,), f"одна оценка на пару: форма [B] = [2], получено {tuple(s.shape)}"
    h, _ = enc(batch["input_ids"], batch["token_type_ids"], batch["attention_mask"])
    want = model.classifier(h[:, 0]).squeeze(-1)
    no_tt, _ = enc(batch["input_ids"], None, batch["attention_mask"])
    if torch.allclose(s, model.classifier(no_tt[:, 0]).squeeze(-1), atol=1e-6) and not torch.allclose(want, s):
        pytest.fail("token_type_ids не дошли до энкодера: у BERT сегменты запроса и документа различаются.",
                    pytrace=False)
    torch.testing.assert_close(s, want, atol=1e-5, rtol=1e-4, msg="оценка = classifier(h[CLS]) пары")
    pad_more = pad_batch(pairs + [build_inputs([5] * 8, [6] * 8, cls_id=1, sep_id=2, max_len=32)], PAD)
    s2 = model(pad_more["input_ids"][:2], pad_more["attention_mask"][:2], pad_more["token_type_ids"][:2])
    torch.testing.assert_close(s2, s, atol=1e-5, rtol=1e-4, msg="оценка пары не зависит от паддинга")


def _hf_bi_encoder(name):
    tr = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    import json

    from huggingface_hub import hf_hub_download
    hf = tr.AutoModel.from_pretrained(name).eval()
    tok = tr.AutoTokenizer.from_pretrained(name)
    pool = json.load(open(hf_hub_download(name, "1_Pooling/config.json")))
    c = hf.config
    ours = BertModel(BertConfig(vocab_size=c.vocab_size, hidden_size=c.hidden_size,
                                num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads,
                                intermediate_size=c.intermediate_size, max_position_embeddings=c.max_position_embeddings,
                                type_vocab_size=c.type_vocab_size, layer_norm_eps=c.layer_norm_eps)).eval()
    ours.load_state_dict(hf.state_dict(), strict=True)
    return hf, tok, ours, ("cls" if pool["pooling_mode_cls_token"] else "mean")


@pytest.mark.weights
@pytest.mark.parametrize("name", ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"])
def test_real_embedding_models(impl, name):
    """Настоящие модели эмбеддингов в BertModel курса: pooling из 1_Pooling/config.json, == transformers, смысл ловится"""
    hf, tok, ours, pooling = _hf_bi_encoder(name)
    texts = ["A man is playing a guitar.", "Someone plays an acoustic guitar on stage.",
             "The stock market fell sharply today.", "Shares dropped as investors panicked."]
    enc = tok(texts, padding=True, return_tensors="pt")
    model = impl.BiEncoder(ours, pooling)
    with torch.no_grad():
        got = model(enc.input_ids, enc.attention_mask)
        h = hf(**enc).last_hidden_state
    m = enc.attention_mask[..., None].float()
    want = h[:, 0] if pooling == "cls" else (h * m).sum(1) / m.sum(1)
    want = want / want.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4, msg=f"{name}: эмбеддинги = transformers + {pooling} + L2")
    cos = got @ got.T
    assert cos[0, 1] > cos[0, 2] + 0.2 and cos[2, 3] > cos[2, 1] + 0.2, (
        f"{name}: парафразы должны быть ближе друг к другу, чем к чужой теме; косинусы:\n{cos}")
