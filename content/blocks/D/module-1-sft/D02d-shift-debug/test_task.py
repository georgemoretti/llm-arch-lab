"""Тесты «Найди баг: сдвиг меток» — те же, что у «Маски лосса SFT», без подсказок, называющих ошибку.

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter).

Оракулы — диалоги, отрендеренные эталоном D01 (llmlab.core.sft) посимвольным токенизатором курса: в нём
один символ = один токен, поэтому позиции ответа видны глазами. Упаковка сверяется с прогоном каждого примера
отдельно через scaled_dot_product_attention.
"""

import pytest
import torch
import torch.nn.functional as F

from llmlab.core import sft as core

IGNORE = -100
TOK = core.CharTokenizer()
HEADER = core.assistant_header(TOK)
U = {"role": "user", "content": "Сколько будет два плюс два?"}
A = {"role": "assistant", "content": "4"}
U2 = {"role": "user", "content": "А три плюс три?"}
A2 = {"role": "assistant", "content": "6"}


def _ids(msgs, **kw):
    return core.encode_chat(core.render_chat(msgs, **kw), TOK)


def _show(ids, mask) -> str:
    """Строка, где токены в маске взяты в [скобки] — видно, что попало в ответ."""
    out, inside = [], False
    for i, m in zip(ids, mask.tolist()):
        if m and not inside:
            out.append("[")
        if not m and inside:
            out.append("]")
        inside = m
        out.append(TOK.itos[i])
    return repr("".join(out) + ("]" if inside else ""))


def _span(text_before: str, text_answer: str, total: int) -> torch.Tensor:
    m = torch.zeros(total, dtype=torch.bool)
    a = len(core.encode_chat(text_before, TOK))
    m[a:a + len(core.encode_chat(text_answer, TOK))] = True
    return m


def test_mask_single_answer(impl):
    """Маска — всё после '<|im_start|>assistant\\n' до <|im_end|> включительно; вопрос и заголовок — нет"""
    ids = _ids([U, A])
    mask = impl.assistant_mask(ids, HEADER, TOK.eos_id)
    assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and mask.shape == (len(ids),), (
        f"маска — bool-тензор [T = {len(ids)}], получено {getattr(mask, 'dtype', type(mask))} "
        f"{tuple(getattr(mask, 'shape', ()))}")
    before = "<|im_start|>user\nСколько будет два плюс два?<|im_end|>\n<|im_start|>assistant\n"
    want = _span(before, "<think>\n\n</think>\n\n4<|im_end|>", len(ids))
    if torch.equal(mask, want):
        return
    end = len(ids) - 2                                   # позиция <|im_end|> ответа (после неё — '\n')
    if torch.equal(mask, want & (torch.arange(len(ids)) != end)):
        pytest.fail("<|im_end|> ответа не в маске. Тогда модель ни разу не учится закрывать ход и на инференсе не "
                    f"остановится. Маска: {_show(ids, mask)}", pytrace=False)
    hdr = _span("<|im_start|>user\nСколько будет два плюс два?<|im_end|>\n", "<|im_start|>assistant\n", len(ids))
    if torch.equal(mask, want | hdr):
        pytest.fail("заголовок '<|im_start|>assistant\\n' попал в маску. Его вставляет шаблон (generation prompt), "
                    f"генерировать его модели не нужно. Маска: {_show(ids, mask)}", pytrace=False)
    pytest.fail(f"маска {_show(ids, mask)}\nждём    {_show(ids, want)}", pytrace=False)


def test_mask_multi_turn(impl):
    """Два ответа в диалоге — оба в маске; system и вопросы — нет"""
    msgs = [{"role": "system", "content": "Кратко."}, U, A, U2, A2]
    ids = _ids(msgs)
    mask = impl.assistant_mask(ids, HEADER, TOK.eos_id)
    text = core.render_chat(msgs)
    first = text.index("<|im_start|>assistant\n4")
    want = _span(text[:first + len("<|im_start|>assistant\n")], "4<|im_end|>", len(ids))
    second = text.rindex("<|im_start|>assistant\n")
    want |= _span(text[:second + len("<|im_start|>assistant\n")], "<think>\n\n</think>\n\n6<|im_end|>", len(ids))
    assert torch.equal(mask, want), (
        f"в маске должны быть ОБА ответа ассистента (ищи каждый заголовок, а не только первый или последний):\n"
        f"маска {_show(ids, mask)}\nждём  {_show(ids, want)}")


def test_mask_open_turn_and_no_answer(impl):
    """Ход без <|im_end|> (обрезан) — в маске до конца; диалог без ответа — пустая маска"""
    ids = _ids([U], add_generation_prompt=True) + TOK.encode("42")
    mask = impl.assistant_mask(ids, HEADER, TOK.eos_id)
    assert mask.tolist() == [False] * (len(ids) - 2) + [True, True], (
        f"после заголовка '42' без <|im_end|>: в маске оба токена до конца строки, получено {_show(ids, mask)}")
    ids = _ids([U])
    assert not impl.assistant_mask(ids, HEADER, TOK.eos_id).any(), "в диалоге без ответа маска пустая"


def test_targets_shift(impl):
    """targets[t] = ids[t + 1], если токен t + 1 — часть ответа; иначе и в последней позиции — −100"""
    ids = _ids([U, A])
    mask = core.assistant_mask(ids, HEADER, TOK.eos_id)
    tgt = impl.sft_targets(ids, mask)
    assert isinstance(tgt, torch.Tensor) and tgt.dtype == torch.long and tgt.shape == (len(ids),), (
        f"targets — long-тензор [T = {len(ids)}], получено {getattr(tgt, 'dtype', None)} {tuple(getattr(tgt, 'shape', ()))}")
    t = torch.tensor(ids)
    want = torch.full_like(t, IGNORE)
    want[:-1] = torch.where(mask[1:], t[1:], IGNORE)
    if torch.equal(tgt, want):
        return
    pytest.fail(f"метки разошлись с ожидаемыми (позиция t предсказывает токен t + 1):\n"
                f"targets {tgt.tolist()}\nждём    {want.tolist()}", pytrace=False)


def test_targets_count_and_first_answer_token(impl):
    """Число меток = число токенов ответа; первый токен ответа предсказывает '\\n' заголовка"""
    msgs = [U, A, U2, A2]
    ids = _ids(msgs)
    mask = core.assistant_mask(ids, HEADER, TOK.eos_id)
    tgt = impl.sft_targets(ids, mask)
    assert int((tgt != IGNORE).sum()) == int(mask.sum()), (
        f"меток {int((tgt != IGNORE).sum())}, токенов ответа {int(mask.sum())}: каждый токен ответа — ровно одна метка")
    first = int(mask.nonzero()[0])
    assert int(tgt[first - 1]) == ids[first], (
        "первый токен ответа предсказывается позицией перед ним — последним '\\n' заголовка '<|im_start|>assistant\\n'")
    assert int(tgt[-1]) == IGNORE, "у последней позиции следующего токена нет: −100"


def test_pack_rows(impl):
    """pack: жадно по порядку, строки ≤ max_len; position_ids с нуля у каждого примера; cu_seqlens int32"""
    exs = [(list(range(1, 6)), [IGNORE, 3, 4, 5, IGNORE]),
           (list(range(10, 14)), [11, 12, 13, IGNORE]),
           (list(range(20, 26)), [IGNORE] * 5 + [IGNORE])]
    rows = impl.pack(exs, max_len=10)
    assert len(rows) == 2, f"5 + 4 ≤ 10, а + 6 уже нет: две строки, получено {len(rows)}"
    r0, r1 = rows
    assert r0["input_ids"].tolist() == list(range(1, 6)) + list(range(10, 14))
    assert r0["targets"].tolist() == [IGNORE, 3, 4, 5, IGNORE, 11, 12, 13, IGNORE], (
        "targets склеиваются как есть: на границе примеров метка −100 уже стоит (последняя позиция примера)")
    pos = r0["position_ids"].tolist()
    if pos == list(range(9)):
        pytest.fail("position_ids идут подряд через всю строку. У второго примера позиции должны начинаться с нуля, "
                    "иначе RoPE видит его как продолжение первого (так же, как в B17).", pytrace=False)
    assert pos == [0, 1, 2, 3, 4, 0, 1, 2, 3], f"position_ids {pos}"
    cu = r0["cu_seqlens"]
    assert cu.dtype == torch.int32 and cu.tolist() == [0, 5, 9], f"cu_seqlens = [0, 5, 9] int32, получено {cu}"
    assert r1["input_ids"].tolist() == list(range(20, 26)) and r1["cu_seqlens"].tolist() == [0, 6]
    with pytest.raises(ValueError):
        impl.pack([(list(range(12)), [IGNORE] * 12)], max_len=10)


def test_packed_causal_mask(impl):
    """Маска упаковки: causal внутри примера, соседи по строке друг друга не видят"""
    cu = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    m = impl.packed_causal_mask(cu)
    want = torch.block_diag(*[torch.ones(n, n, dtype=torch.bool).tril() for n in (3, 2, 4)])
    assert m.dtype == torch.bool and m.shape == (9, 9), f"bool [9, 9], получено {m.dtype} {tuple(m.shape)}"
    if torch.equal(m, torch.ones(9, 9, dtype=torch.bool).tril()):
        pytest.fail("обычная causal-маска на всю строку: второй пример видит первый, и его ответ зависит от "
                    "случайного соседа по упаковке", pytrace=False)
    if torch.equal(m, torch.block_diag(*[torch.ones(n, n, dtype=torch.bool) for n in (3, 2, 4)])):
        pytest.fail("блоки есть, но внутри примера маска двунаправленная: нужен ещё tril (causal)", pytrace=False)
    assert torch.equal(m, want), f"маска\n{m.int()}\nждём\n{want.int()}"


def test_packed_attention_equals_separate(impl):
    """Attention по упакованной строке с этой маской == каждый пример отдельно"""
    g = torch.Generator().manual_seed(0)
    lens = [4, 7, 3]
    cu = torch.tensor([0, 4, 11, 14], dtype=torch.int32)
    q, k, v = (torch.randn(1, 2, 14, 8, generator=g) for _ in range(3))
    packed = F.scaled_dot_product_attention(q, k, v, attn_mask=impl.packed_causal_mask(cu))
    for i, n in enumerate(lens):
        s, e = int(cu[i]), int(cu[i + 1])
        alone = F.scaled_dot_product_attention(q[..., s:e, :], k[..., s:e, :], v[..., s:e, :], is_causal=True)
        torch.testing.assert_close(packed[..., s:e, :], alone, atol=1e-6, rtol=1e-5,
                                   msg=f"пример {i} в упаковке дал другой выход, чем один: он видит соседей")


def test_end_to_end_on_dialogs(impl):
    """Конвейер на трёх диалогах: маска → метки → упаковка; лосс по упаковке == сумма лоссов по отдельности"""
    dialogs = [[U, A], [U2, A2], [{"role": "user", "content": "Напиши заглавными: лес"},
                                  {"role": "assistant", "content": "ЛЕС"}]]
    exs = []
    for d in dialogs:
        ids = _ids(d)
        exs.append((ids, impl.sft_targets(ids, impl.assistant_mask(ids, HEADER, TOK.eos_id))))
    rows = impl.pack(exs, max_len=256)
    assert len(rows) == 1
    torch.manual_seed(0)
    logits = torch.randn(len(rows[0]["input_ids"]), TOK.vocab_size)
    packed = F.cross_entropy(logits, rows[0]["targets"], ignore_index=IGNORE, reduction="sum")
    sep, start = 0.0, 0
    for ids, tgt in exs:
        sep += F.cross_entropy(logits[start:start + len(ids)], tgt, ignore_index=IGNORE, reduction="sum")
        start += len(ids)
    torch.testing.assert_close(packed, sep, atol=1e-4, rtol=1e-5,
                               msg="лосс по упакованной строке должен быть суммой лоссов примеров")
    n = int((rows[0]["targets"] != IGNORE).sum())
    # ответ = <think> + '\n\n' + </think> + '\n\n' + текст + <|im_end|> → 6 + len(текст) + 1 токенов
    assert n == (7 + 1) + (7 + 1) + (7 + 3), f"меток {n}, а токенов в трёх ответах 8 + 8 + 10 = 26"
