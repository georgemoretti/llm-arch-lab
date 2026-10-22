"""Тесты «Проект: SFT».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter).

Оракулы — эталоны D01/D02 (llmlab.core.sft), ручные примеры ответов и «сценарная» модель, которая выдаёт заранее
заданные токены и записывает, какой промпт ей подали. Короткое обучение — 80 шагов SFT игрушечной модели с нуля на
одной задаче (регистр): формат выучивается за секунды, и это проверяет всю цепочку функций проекта.
"""

import json

import pytest
import torch
import torch.nn as nn

from llmlab.core import sft as core
from llmlab.core.model import Model

IGNORE = -100
TOK = core.CharTokenizer()
EX = {"task": "arith", "prompt": "Сколько будет два плюс три?", "answer": "5"}


def test_build_example_response(impl):
    """loss_on='response': ids — диалог «вопрос + ответ» по шаблону D01, метки только у ответа (D02)"""
    out = impl.build_example(EX, TOK, loss_on="response")
    assert isinstance(out, tuple) and len(out) == 2, "build_example возвращает (ids, targets)"
    ids, tgt = out
    text = core.render_chat(core.to_messages(EX))
    want_ids = core.encode_chat(text, TOK)
    if list(ids) == core.encode_chat(core.render_chat(core.to_messages(EX, False), add_generation_prompt=True,
                                                      enable_thinking=False), TOK) + TOK.encode("5"):
        pytest.fail("ids собраны как «generation prompt + ответ» без <|im_end|>: модель не научится закрывать ход. "
                    "Отрендери диалог из двух сообщений (вопрос и ответ) обычным render_chat.", pytrace=False)
    assert list(ids) == want_ids, f"ids должны совпасть с encode_chat(render_chat([user, assistant])):\n{TOK.decode(ids)!r}"
    want = core.sft_targets(want_ids, core.assistant_mask(want_ids, core.assistant_header(TOK), TOK.eos_id))
    if torch.equal(torch.as_tensor(tgt), torch.as_tensor(want_ids)):
        pytest.fail("targets = ids: метки не сдвинуты и не замаскированы — используй sft_targets и assistant_mask",
                    pytrace=False)
    assert torch.equal(torch.as_tensor(tgt), want), "метки — sft_targets(ids, assistant_mask(...)) из D02"
    labeled = [TOK.itos[int(t)] for t in tgt if t != IGNORE]
    assert labeled == ["<think>", "\n", "\n", "</think>", "\n", "\n", "5", "<|im_end|>"], f"в лоссе: {labeled}"


def test_build_example_all_tokens(impl):
    """loss_on='all': метки у всех позиций, кроме последней (как в предобучении); иначе — ValueError"""
    ids, tgt = impl.build_example(EX, TOK, loss_on="all")
    t = torch.as_tensor(ids)
    want = torch.cat([t[1:], torch.tensor([IGNORE])])
    assert torch.equal(torch.as_tensor(tgt), want), (
        "loss_on='all': targets[t] = ids[t + 1] для всех t, последняя позиция −100 (маска из одних True)")
    with pytest.raises(ValueError):
        impl.build_example(EX, TOK, loss_on="prompt")


@pytest.mark.parametrize("task,want,answer,expected", [
    ("arith", "5", "5", (True, True)),
    ("arith", "5", "7", (True, False)),
    ("arith", "5", "пять", (False, False)),
    ("arith", "12", " 12", (False, False)),
    ("arith", "5", "", (False, False)),
    ("upper", "МОКАТУ", "МОКАТУ", (True, True)),
    ("upper", "МОКАТУ", "МОКАТЕ", (True, False)),
    ("upper", "МОКАТУ", "мокату", (False, False)),
    ("upper", "МОКАТУ", "МОКАТУ!", (False, False)),
    ("json", '{"name": "Педу", "age": 76}', '{"name": "Педу", "age": 76}', (True, True)),
    ("json", '{"name": "Педу", "age": 76}', '{"age":76,"name":"Педу"}', (True, True)),
    ("json", '{"name": "Педу", "age": 76}', '{"name": "Педу", "age": 67}', (True, False)),
    ("json", '{"name": "Педу", "age": 76}', '{"name": "Педу", "age": "76"}', (False, False)),
    ("json", '{"name": "Педу", "age": 76}', '{"name": "Педу"}', (False, False)),
    ("json", '{"name": "Педу", "age": 1}', '{"name": "Педу", "age": true}', (False, False)),
    ("json", '{"name": "Педу", "age": 76}', '{"name": "Педу", "age": 76', (False, False)),
    ("json", '{"name": "Педу", "age": 76}', '{"name": "Педу", "age": 76, "x": 1}', (False, False)),
])
def test_check_answer(impl, task, want, answer, expected):
    """check_answer → (формат верен, ответ верен)"""
    got = impl.check_answer({"task": task, "prompt": "…", "answer": want}, answer)
    assert tuple(got) == expected, (
        f"{task}: ответ {answer!r} при эталоне {want!r} → ждём (формат, ответ) = {expected}, получено {tuple(got)}. "
        "Формат: arith — только цифры; upper — только заглавные буквы; json — объект ровно с name (строка) и age "
        "(целое). JSON сравнивай как объекты (json.loads), а не как строки.")


def test_evaluate(impl):
    """evaluate: доли формата и ответа в целом и по задачам; answer_fn получает текст вопроса"""
    exs = [{"task": "arith", "prompt": "a", "answer": "1"}, {"task": "arith", "prompt": "b", "answer": "2"},
           {"task": "upper", "prompt": "c", "answer": "ЛЕС"}, {"task": "upper", "prompt": "d", "answer": "ДОМ"}]
    replies = {"a": "1", "b": "3", "c": "ЛЕС", "d": "дом"}
    seen = []
    res = impl.evaluate(exs, lambda p: seen.append(p) or replies[p])
    assert seen == ["a", "b", "c", "d"], f"answer_fn вызывается с prompt каждого примера по порядку, вызовы: {seen}"
    assert abs(res["format"] - 0.75) < 1e-9 and abs(res["exact"] - 0.5) < 1e-9, (
        f"формат 3/4, ответ 2/4; получено format {res.get('format')}, exact {res.get('exact')}")
    bt = res["by_task"]
    assert bt["arith"] == {"format": 1.0, "exact": 0.5} and bt["upper"] == {"format": 0.5, "exact": 0.5}, f"by_task: {bt}"


class _Scripted(nn.Module):
    """«Модель», которая жадно выдаёт заранее заданные токены и запоминает промпт (первый вызов forward)."""

    def __init__(self, script: list[int]):
        super().__init__()
        self.script, self.prompt, self.calls = script, None, 0
        self.dummy = nn.Parameter(torch.zeros(1))

    def new_caches(self):
        return [None]

    def forward(self, idx, caches=None):
        if self.prompt is None:
            self.prompt = idx[0].tolist()
        logits = torch.zeros(idx.shape[0], idx.shape[1], TOK.vocab_size)
        logits[:, -1, self.script[min(self.calls, len(self.script) - 1)]] = 10.0
        self.calls += 1
        return logits


def test_chat_answer_prompt_and_stop(impl):
    """chat_answer: промпт — generation prompt с enable_thinking=False; ответ — до <|im_end|>, без служебных токенов"""
    model = _Scripted(TOK.encode("12") + [TOK.eos_id] + TOK.encode("мусор"))
    ans = impl.chat_answer(model, TOK, "Сколько будет семь плюс пять?")
    want_prompt = core.encode_chat(core.render_chat([{"role": "user", "content": "Сколько будет семь плюс пять?"}],
                                                    add_generation_prompt=True, enable_thinking=False), TOK)
    if model.prompt == want_prompt[:-6]:
        pytest.fail("в промпте нет пустого блока <think>\\n\\n</think>\\n\\n: модель учили отвечать после него "
                    "(последний ответ в обучающем примере получает этот блок, D01). Нужен enable_thinking=False.",
                    pytrace=False)
    assert model.prompt == want_prompt, f"промпт модели:\n{TOK.decode(model.prompt or [])!r}\nждём:\n{TOK.decode(want_prompt)!r}"
    assert isinstance(ans, str), f"chat_answer возвращает str, получено {type(ans).__name__}"
    if "мусор" in ans:
        pytest.fail("генерация не остановилась на <|im_end|>: передай eos_token_id=tok.eos_id в generate или обрежь "
                    "ответ по первому <|im_end|>", pytrace=False)
    assert ans == "12", f"ответ {ans!r}: ждём '12' без <|im_end|>"
    model = _Scripted(TOK.encode("Б"))
    assert impl.chat_answer(model, TOK, "x", max_new_tokens=5) == "БББББ", "без <|im_end|> — ровно max_new_tokens токенов"


def test_short_sft_learns_format(impl):
    """80 шагов SFT с нуля на задаче «заглавными»: доля верного формата на отложенных словах ≥ 0.8"""
    torch.manual_seed(0)
    model = Model(core.toy_config(TOK.vocab_size, d_model=64, n_layers=2))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    train = core.make_instructions(200, seed=0, tasks=("upper",))
    test = core.make_instructions(20, seed=1, split="test", tasks=("upper",))
    data = [impl.build_example(e, TOK, "response") for e in train]
    g = torch.Generator().manual_seed(0)
    for step in range(80):
        idx = torch.randint(len(data), (16,), generator=g).tolist()
        core.sft_step(model, opt, [core.collate([data[i] for i in idx], TOK.pad_id)], step, max_lr=3e-3,
                      warmup_steps=5, total_steps=80, min_lr=3e-4)
    model.eval()
    res = impl.evaluate(test, lambda p: impl.chat_answer(model, TOK, p, max_new_tokens=12))
    assert res["format"] >= 0.8, (
        f"после 80 шагов доля ответов из одних заглавных букв {res['format']:.2f} (ждём ≥ 0.8). Проверь, что "
        "обучающий пример и промпт генерации совпадают по формату и что метки — только на ответе.")
    json.dumps(res)                                                     # результат сериализуется (для --record)
