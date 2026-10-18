"""Тесты «Дистилляция».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter);
  ref  — эталон solution.py.

Оракулы — F.kl_div / F.cross_entropy / F.cosine_similarity и аналитический градиент T·(q − p)/N.
Учитель — маленький BertForMaskedLM из llmlab.core.bert (эталон B10) на случайных весах.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from llmlab.core.bert import BertConfig, BertForMaskedLM

V = 40


def _logits(n=6, seed=0, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, V, generator=g) * scale, torch.randn(n, V, generator=g) * scale


def test_soft_targets(impl):
    """softmax(z / T): T = 1 — обычный softmax, с ростом T распределение площе, порядок классов тот же"""
    z, _ = _logits()
    p1 = impl.soft_targets(z, 1.0)
    torch.testing.assert_close(p1, z.softmax(-1), atol=1e-6, rtol=1e-5, msg="T = 1 — обычный softmax по последней оси")
    ent = lambda p: -(p * p.clamp_min(1e-30).log()).sum(-1).mean().item()
    es = [ent(impl.soft_targets(z, T)) for T in (1.0, 2.0, 4.0, 8.0)]
    assert all(a < b for a, b in zip(es, es[1:])), (
        f"энтропия мягких меток должна расти с T (1, 2, 4, 8): {[round(e, 2) for e in es]} — делим логиты на T")
    assert torch.equal(impl.soft_targets(z, 4.0).argsort(-1), z.argsort(-1)), "температура не меняет порядок классов"
    assert es[-1] < math.log(V), "даже при T = 8 это ещё не равномерное распределение"


def _diagnose_kl(got, s, t, T):
    lq, lp = F.log_softmax(s / T, -1), F.log_softmax(t / T, -1)
    kl = F.kl_div(lq, lp, log_target=True, reduction="batchmean")
    if math.isclose(got, kl.item(), rel_tol=1e-4):
        return f"не хватает множителя T² = {T * T:g}: без него градиент мягкой части падает как 1/T²."
    rev = F.kl_div(lp, lq, log_target=True, reduction="batchmean") * T * T
    if math.isclose(got, rev.item(), rel_tol=1e-4):
        return "KL в обратную сторону: нужно KL(p_учителя ‖ p_ученика) = Σ p·(log p − log q), p — учитель."
    if math.isclose(got, (kl * T * T / V).item(), rel_tol=1e-4):
        return ("усреднение по всем элементам [N, V] (reduction='mean'): KL — сумма по классам, среднее только по "
                "строкам (batchmean).")
    ce = -(lp.exp() * lq).sum(-1).mean() * T * T
    if math.isclose(got, ce.item(), rel_tol=1e-4):
        return ("это cross-entropy с мягкими метками: градиент тот же, но значение больше на энтропию учителя и "
                "не равно 0 при совпадении. Нужна KL = Σ p·(log p − log q).")
    only_t = F.kl_div(F.log_softmax(s, -1), lp, log_target=True, reduction="batchmean") * T * T
    if math.isclose(got, only_t.item(), rel_tol=1e-4):
        return "температура применена только к учителю: логиты ученика тоже делятся на T."
    return f"T²·KL должно быть {(kl * T * T).item():.5f}, получено {got:.5f}."


@pytest.mark.parametrize("T", [1.0, 2.0, 5.0])
def test_distill_kl_value(impl, T):
    """distill_kl = T² · KL(p_учителя ‖ p_ученика) при температуре T, среднее по строкам"""
    s, t = _logits(seed=1)
    got = impl.distill_kl(s, t, T)
    assert got.dim() == 0, "лосс — скаляр"
    want = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True, reduction="batchmean") * T * T
    if not math.isclose(got.item(), want.item(), rel_tol=1e-4, abs_tol=1e-6):
        pytest.fail(_diagnose_kl(got.item(), s, t, T), pytrace=False)
    assert abs(impl.distill_kl(t, t, T).item()) < 1e-5, "ученик = учитель → KL = 0"
    s3, t3 = s.view(2, 3, V), t.view(2, 3, V)
    torch.testing.assert_close(impl.distill_kl(s3, t3, T), got, atol=1e-6, rtol=1e-5,
                               msg="логиты [B, T, V]: среднее по всем B·T строкам, как у [N, V]")


def test_t2_keeps_gradient_scale(impl):
    """Градиент по логитам ученика = T·(q − p)/N; с множителем T² его масштаб почти не зависит от T"""
    s, t = _logits(seed=2, scale=1.0)
    norms = {}
    for T in (1.0, 2.0, 4.0, 8.0):
        z = s.clone().requires_grad_(True)
        teacher = t.clone().requires_grad_(True)
        impl.distill_kl(z, teacher, T).backward()
        want = T * (F.softmax(s / T, -1) - F.softmax(t / T, -1)) / s.shape[0]
        torch.testing.assert_close(z.grad, want, atol=1e-6, rtol=1e-4,
                                   msg=f"T = {T:g}: ∂(T²·KL)/∂z = T·(softmax(z/T) − softmax(t/T)) / N")
        assert teacher.grad is None or teacher.grad.abs().max() == 0, (
            "градиент дошёл до логитов учителя: учитель заморожен, его логиты — константа (detach)")
        norms[T] = z.grad.norm().item()
    ratio = norms[8.0] / norms[2.0]
    assert 0.7 < ratio < 1.4, (
        f"‖∇‖ при T = 8 и T = 2 должны быть одного порядка (отношение {ratio:.2f}); без T² было бы ≈ 1/16")


def test_distillation_loss_mix(impl):
    """alpha · мягкая часть (все позиции) + (1 − alpha) · CE (только labels ≠ −100)"""
    s, t = _logits(n=8, seed=3)
    s, t = s.view(2, 4, V), t.view(2, 4, V)
    labels = torch.tensor([[5, -100, 7, -100], [-100, -100, 9, 1]])
    soft = F.kl_div(F.log_softmax(s.reshape(-1, V) / 2, -1), F.log_softmax(t.reshape(-1, V) / 2, -1),
                    log_target=True, reduction="batchmean") * 4
    hard = F.cross_entropy(s.reshape(-1, V), labels.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(impl.distillation_loss(s, t, labels, temperature=2.0, alpha=1.0), soft,
                               atol=1e-5, rtol=1e-4, msg="alpha = 1: только T²·KL по ВСЕМ позициям (и без labels)")
    got0 = impl.distillation_loss(s, t, labels, temperature=2.0, alpha=0.0)
    torch.testing.assert_close(got0, hard, atol=1e-5, rtol=1e-4,
                               msg="alpha = 0: только cross-entropy ученика с labels (ignore_index = −100)")
    torch.testing.assert_close(impl.distillation_loss(s, t, labels, temperature=2.0, alpha=0.3), 0.3 * soft + 0.7 * hard,
                               atol=1e-5, rtol=1e-4, msg="alpha · soft + (1 − alpha) · hard")
    none = torch.full_like(labels, -100)
    val = impl.distillation_loss(s, t, none, temperature=2.0, alpha=0.5)
    assert torch.isfinite(val) and math.isclose(val.item(), 0.5 * soft.item(), rel_tol=1e-4), (
        "если в батче нет ни одной метки, hard loss = 0 (а не nan от среднего по пустому множеству)")


def test_hidden_cosine_loss(impl):
    """1 − cos между скрытыми состояниями, среднее по настоящим токенам; масштаб не важен, PAD не учитывается"""
    g = torch.Generator().manual_seed(4)
    hs, ht = torch.randn(2, 5, 16, generator=g), torch.randn(2, 5, 16, generator=g)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    got = impl.hidden_cosine_loss(hs, ht, mask)
    m = mask.bool()
    want = (1 - F.cosine_similarity(hs[m], ht[m], dim=-1)).mean()
    if math.isclose(got.item(), (1 - F.cosine_similarity(hs, ht, dim=-1)).mean().item(), rel_tol=1e-5):
        pytest.fail("PAD-позиции попали в среднее: берите только токены с attention_mask = 1.", pytrace=False)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-5, msg="mean(1 − cos(h_ученика, h_учителя)) по mask = 1")
    torch.testing.assert_close(impl.hidden_cosine_loss(3 * hs, ht, mask), got, atol=1e-6, rtol=1e-5,
                               msg="косинус не зависит от длины векторов: ученик ×3 → тот же лосс")
    assert impl.hidden_cosine_loss(ht, ht, mask).abs() < 1e-6, "совпадение → лосс 0"
    hs2 = hs.clone()
    hs2[0, 4] = 100.0
    torch.testing.assert_close(impl.hidden_cosine_loss(hs2, ht, mask), got, atol=1e-6, rtol=1e-5,
                               msg="значения на PAD-позициях не влияют на лосс")


def _teacher(layers=4, seed=0):
    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=V, hidden_size=32, num_hidden_layers=layers, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=16)
    return BertForMaskedLM(cfg).eval()


def test_student_from_teacher(impl):
    """Ученик: эмбеддинги и голова учителя, слои 0, 2, … (или заданные); копии, а не общие тензоры"""
    teacher = _teacher(layers=4)
    student = impl.student_from_teacher(teacher)
    assert isinstance(student, BertForMaskedLM), "ученик — тот же класс BertForMaskedLM (B10), только слоёв меньше"
    assert len(student.bert.encoder.layer) == 2, f"из 4 слоёв учителя берём каждый второй → 2, получено {len(student.bert.encoder.layer)}"
    assert student.bert.cfg.num_hidden_layers == 2 and student.bert.cfg.hidden_size == 32, (
        "конфиг ученика — конфиг учителя с другим num_hidden_layers (dataclasses.replace)")
    sd_s = student.state_dict()
    sd_t = {k: v.clone() for k, v in teacher.state_dict().items()}   # снимок: state_dict отдаёт ссылки
    for name in ["bert.embeddings.word_embeddings.weight", "bert.embeddings.LayerNorm.weight", "cls.predictions.bias",
                 "cls.predictions.transform.dense.weight"]:
        assert torch.equal(sd_s[name], sd_t[name]), f"{name}: эмбеддинги и MLM-голова — как у учителя"
    for i, j in [(0, 0), (1, 2)]:
        a = sd_s[f"bert.encoder.layer.{i}.attention.self.query.weight"]
        b = sd_t[f"bert.encoder.layer.{j}.attention.self.query.weight"]
        assert torch.equal(a, b), f"слой {i} ученика = слой {j} учителя (каждый второй, начиная с 0)"
    assert student.cls.predictions.decoder.weight is student.bert.embeddings.word_embeddings.weight, (
        "у ученика декодер MLM связан с ЕГО эмбеддингами (weight tying), как в B10")
    with torch.no_grad():
        student.bert.encoder.layer[0].attention.self.query.weight.add_(1.0)
        student.bert.embeddings.word_embeddings.weight.add_(1.0)
    assert torch.equal(teacher.state_dict()["bert.encoder.layer.0.attention.self.query.weight"],
                       sd_t["bert.encoder.layer.0.attention.self.query.weight"]), (
        "правка ученика изменила учителя: веса нужно скопировать (load_state_dict), а не взять те же модули")
    custom = impl.student_from_teacher(_teacher(layers=6), layers=[0, 2, 5])
    assert len(custom.bert.encoder.layer) == 3, "layers=[0, 2, 5] → 3 слоя ученика"
    t6 = _teacher(layers=6)
    assert torch.equal(custom.state_dict()["bert.encoder.layer.2.output.dense.weight"],
                       t6.state_dict()["bert.encoder.layer.5.output.dense.weight"]), "слой 2 ученика = слой 5 учителя"


def test_distillation_trains_student(impl):
    """Ученик из слоёв учителя стартует ближе случайного и за 60 шагов дистилляции почти копирует учителя"""
    teacher = _teacher(layers=4)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(5, V, (16, 10), generator=g)
    with torch.no_grad():
        target = teacher(ids)
    student = impl.student_from_teacher(teacher)
    torch.manual_seed(3)
    rnd = BertForMaskedLM(BertConfig(vocab_size=V, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                     intermediate_size=64, max_position_embeddings=16))
    kl = lambda m: F.kl_div(F.log_softmax(m(ids), -1), F.log_softmax(target, -1), log_target=True,
                            reduction="batchmean").item()
    with torch.no_grad():
        k0, k_rnd = kl(student), kl(rnd)
    assert k0 < 0.5 * k_rnd, (
        f"KL ученика из слоёв учителя ({k0:.2f}) должен быть заметно меньше, чем у случайного ({k_rnd:.2f})")
    opt = torch.optim.Adam(student.parameters(), lr=3e-3)
    for _ in range(60):
        x = torch.randint(5, V, (16, 10), generator=g)
        with torch.no_grad():
            tl = teacher(x)
        loss = impl.distill_kl(student(x), tl, 2.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        k1 = kl(student)
        agree = (student(ids).argmax(-1) == target.argmax(-1)).float().mean().item()
    assert k1 < 0.25 * k0, f"за 60 шагов KL к учителю должен упасть хотя бы в 4 раза: {k0:.3f} → {k1:.3f}"
    assert agree >= 0.8, f"после дистилляции top-1 ученика совпадает с учителем в {agree:.0%} позиций (ждём ≥ 80 %)"
