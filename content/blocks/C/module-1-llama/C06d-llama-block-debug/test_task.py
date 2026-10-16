"""Тесты C06d (= тесты C06, но без подсказок, называющих баг).

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Схема: строим эталонную модель, делаем веса RMSNorm неединичными (иначе ошибки в размещении норм
не видны), загружаем те же веса в реализацию пользователя и сравниваем выходы.
"""

import pytest
import torch
from torch.testing import assert_close

B, T = 2, 10
ATOL, RTOL = 1e-5, 1e-4


def _cfg(mod, **kw):
    base = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_hidden=64,
                rope_theta=10000.0)
    base.update(kw)
    return mod.LlamaConfig(**base)


def _perturb_norms(model):
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name == "norm.weight":
                p.copy_(1 + 0.5 * torch.randn(p.shape, generator=g))


def _ref_model(ref, **kw):
    torch.manual_seed(0)
    m = ref.LlamaModel(_cfg(ref, **kw))
    _perturb_norms(m)
    return m.eval()


def _load(target, source, what="модель"):
    try:
        target.load_state_dict(source.state_dict(), strict=True)
    except RuntimeError as e:
        pytest.fail(f"Не удалось загрузить веса эталона в твою {what}: имена или формы параметров не "
                    f"совпадают. Нужны имена как в HF (input_layernorm, self_attn, post_attention_layernorm, "
                    f"mlp, embed_tokens, layers, norm, lm_head).\n{str(e)[:600]}", pytrace=False)
    return target.eval()


def _impl_model(impl, ref, **kw):
    r = _ref_model(ref, **kw)
    torch.manual_seed(1)
    return _load(impl.LlamaModel(_cfg(impl, **kw)), r), r


def _idx(t=T, seed=3, vocab=64):
    return torch.randint(0, vocab, (B, t), generator=torch.Generator().manual_seed(seed))


def _x(seed=2):
    return torch.randn(B, T, 32, generator=torch.Generator().manual_seed(seed))


def _diagnose_block(rb, x, out) -> str:
    """rb — эталонный блок с теми же весами; сравниваем с типичными неверными сборками."""
    n1, attn, n2, mlp = rb.input_layernorm, rb.self_attn, rb.post_attention_layernorm, rb.mlp
    with torch.no_grad():
        h = n1(x)
        post = n1(x + attn(x))
        variants = {
            "Post-Norm (норма после сложения с остатком, как в оригинальном Transformer), а в Llama — "
            "Pre-Norm: x + f(norm(x))": n2(post + mlp(post)),
            "нет остаточных соединений (x = f(norm(x)))": mlp(n2(attn(h))),
            "обе нормы — input_layernorm (post_attention_layernorm не используется)":
                (x + attn(h)) + mlp(n1(x + attn(h))),
            "attention и MLP считаются параллельно от одного входа (как в GPT-J), а в Llama — "
            "последовательно": x + attn(h) + mlp(n2(x)),
        }
    for hint, cand in variants.items():
        if cand.shape == out.shape and torch.allclose(out, cand, atol=1e-4, rtol=1e-3):
            return f"Похоже, {hint}."
    return "Сравни свой блок с формулой из README построчно."


# ---------------------------------------------------------------------------
# Структура
# ---------------------------------------------------------------------------

def test_parameter_names_and_shapes(impl, ref):
    """LlamaModel: параметры с именами и формами как в HF (без позиционных эмбеддингов и bias)"""
    torch.manual_seed(0)
    mine = impl.LlamaModel(_cfg(impl))
    theirs = ref.LlamaModel(_cfg(ref))
    a = {k: tuple(v.shape) for k, v in mine.state_dict().items()}
    b = {k: tuple(v.shape) for k, v in theirs.state_dict().items()}
    missing, extra = sorted(set(b) - set(a)), sorted(set(a) - set(b))
    if missing or extra:
        hint = ""
        if any("wpe" in k or "pos" in k for k in extra):
            hint = " Позиционные эмбеддинги не нужны — позиции кодирует RoPE внутри attention."
        if any(k.endswith(".bias") for k in extra):
            hint += " В Llama нет bias."
        pytest.fail(f"Не хватает параметров: {missing[:6]}; лишние: {extra[:6]}.{hint}", pytrace=False)
    wrong = {k: (a[k], b[k]) for k in b if a[k] != b[k]}
    assert not wrong, f"Формы не совпадают (твоя, нужная): {wrong}"


def test_block_matches_reference(impl, ref):
    """LlamaBlock: Pre-Norm, attention → MLP, два остаточных соединения"""
    torch.manual_seed(0)
    cfg = _cfg(ref)
    rb = ref.LlamaBlock(cfg)
    _perturb_norms(rb)
    torch.manual_seed(1)
    mb = _load(impl.LlamaBlock(_cfg(impl)), rb.eval(), "блок")
    x = _x()
    with torch.no_grad():
        out, expected = mb(x), rb(x)
    if out.shape != expected.shape or not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        pytest.fail(f"Выход блока не совпадает с эталоном. {_diagnose_block(rb, x, out)}", pytrace=False)


def test_block_is_identity_with_zero_sublayers(impl, ref):
    """LlamaBlock: если o_proj и down_proj нулевые, блок пропускает вход без изменений"""
    torch.manual_seed(0)
    mb = impl.LlamaBlock(_cfg(impl)).eval()
    _perturb_norms(mb)
    with torch.no_grad():
        mb.self_attn.o_proj.weight.zero_()
        mb.mlp.down_proj.weight.zero_()
        x = _x()
        out = mb(x)
    if not torch.allclose(out, x, atol=1e-6):
        pytest.fail("Подслои выдают нули, значит блок должен вернуть x как есть: x + 0 + 0. Он вернул "
                    f"другое (max |Δ| = {(out - x).abs().max():.3g}). Проследи, что на самом деле "
                    "прибавляется к выходу подслоя.", pytrace=False)


def test_tied_embeddings(impl):
    """tie_embeddings=True: lm_head и embed_tokens — один и тот же тензор (как в Llama 3.2 1B)"""
    torch.manual_seed(0)
    untied = impl.LlamaModel(_cfg(impl))
    tied = impl.LlamaModel(_cfg(impl, tie_embeddings=True))
    assert tied.lm_head.weight is tied.embed_tokens.weight, (
        "При tie_embeddings=True веса должны быть общими (один объект Parameter), а не копией: "
        "self.lm_head.weight = self.embed_tokens.weight.")
    assert untied.lm_head.weight is not untied.embed_tokens.weight, (
        "По умолчанию (как у Llama 3 8B) выходной слой отдельный — tie_embeddings=False.")
    n_tied = sum(p.numel() for p in tied.parameters())
    n_untied = sum(p.numel() for p in untied.parameters())
    assert n_untied - n_tied == 64 * 32, "Связывание должно экономить ровно vocab × d_model параметров."


# ---------------------------------------------------------------------------
# Модель целиком
# ---------------------------------------------------------------------------

def test_model_logits_match_reference(impl, ref):
    """LlamaModel: idx [B, T] → логиты [B, T, vocab], совпадают с эталоном"""
    mine, theirs = _impl_model(impl, ref)
    idx = _idx()
    with torch.no_grad():
        out, expected = mine(idx), theirs(idx)
    assert tuple(out.shape) == (B, T, 64), f"Форма логитов {tuple(out.shape)} вместо {(B, T, 64)}."
    if not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        hint = "Проверь порядок: embed_tokens → слои → norm → lm_head."
        with torch.no_grad():
            x = theirs.embed_tokens(idx)
            for layer in theirs.layers:
                x = layer(x)
            if torch.allclose(out, theirs.lm_head(x), atol=1e-4, rtol=1e-4):
                hint = "Похоже, пропущена финальная норма (self.norm) перед lm_head."
            else:
                x = theirs.embed_tokens(idx)
                x = theirs.layers[0](x)
                if torch.allclose(out, theirs.lm_head(theirs.norm(x)), atol=1e-4, rtol=1e-4):
                    hint = "Похоже, работает только первый слой — проверь цикл по self.layers."
        pytest.fail(f"Логиты не совпадают с эталоном (max |Δ| = {(out - expected).abs().max():.3g}). {hint}",
                    pytrace=False)


def test_model_causal(impl, ref):
    """LlamaModel: логиты в позициях ≤ t не зависят от токенов после t"""
    mine, _ = _impl_model(impl, ref)
    idx = _idx()
    idx2 = idx.clone()
    idx2[:, 6:] = (idx2[:, 6:] + 5) % 64
    with torch.no_grad():
        a, b = mine(idx), mine(idx2)
    delta = (a[:, :6] - b[:, :6]).abs().max().item()
    assert delta < 1e-5, f"Логиты в позициях 0..5 изменились (max |Δ| = {delta:.3g}) от смены токенов 6+ — утечка будущего."


def test_model_cache_matches_full(impl, ref):
    """LlamaModel с кешами: prefill + по одному токену == полный прогон"""
    mine, theirs = _impl_model(impl, ref)
    idx = _idx()
    caches = mine.new_caches()
    with torch.no_grad():
        parts = [mine(idx[:, :6], caches)] + [mine(idx[:, t:t + 1], caches) for t in range(6, T)]
        out = torch.cat(parts, dim=1)
        expected = theirs(idx)
    if not torch.allclose(out, expected, atol=1e-4, rtol=1e-4):
        bad = ((out - expected).abs() > 1e-4).any(-1).any(0).nonzero()
        first = int(bad[0]) if len(bad) else -1
        hint = ""
        lens = [c.seq_len for c in caches]
        if any(n != lens[0] for n in lens) or lens[0] != T:
            hint = (f" Длины кешей по слоям: {lens} (должно быть по {T}). Каждый слой должен писать в СВОЙ "
                    "кеш caches[i] — у слоёв разные K и V.")
        elif first >= 6:
            hint = (" Prefill верен, а дальше нет: проверь, что каждый слой получает СВОЙ кеш (caches[i]), "
                    "а не один общий и не None.")
        pytest.fail(f"С кешами логиты расходятся с полным прогоном начиная с позиции {first}.{hint}",
                    pytrace=False)
    assert all(c.seq_len == T for c in caches), (
        f"После генерации длины кешей {[c.seq_len for c in caches]}, а должны быть по {T} в каждом слое.")


def test_generate_with_and_without_cache(impl, ref):
    """generate: жадная генерация с кешем даёт те же токены, что без кеша"""
    mine, theirs = _impl_model(impl, ref)
    prompt = _idx(t=4)
    a = impl.generate(mine, prompt, max_new_tokens=8, use_cache=True)
    b = impl.generate(mine, prompt, max_new_tokens=8, use_cache=False)
    expected = ref.generate(theirs, prompt, max_new_tokens=8, use_cache=False)
    assert tuple(a.shape) == (B, 12), f"generate вернул форму {tuple(a.shape)} вместо {(B, 12)}."
    assert torch.equal(b, expected), "Без кеша сгенерированы не те токены, что у эталона."
    assert torch.equal(a, b), f"С кешем и без кеша сгенерированы разные токены:\n{a.tolist()}\n{b.tolist()}"
