"""Тесты «Chat-шаблон».

Фикстуры (из llmlab.pytest_plugin):
  impl — проверяемая реализация (workspace / reference / starter).

Оракулы — строки, выписанные руками по шаблону Qwen3-0.6B, и посимвольный токенизатор курса (llmlab.core.sft).
С --weights: сверка с `apply_chat_template` настоящего токенизатора Qwen3-0.6B и генерация с этим шаблоном.
"""

import pytest
import torch

from llmlab.core import sft as core

SYS = {"role": "system", "content": "Отвечай кратко."}
U1 = {"role": "user", "content": "Сколько будет два плюс два?"}
A1 = {"role": "assistant", "content": "4"}
A1_THINK = {"role": "assistant", "content": "<think>\nдва и два\n</think>\n\n4"}
U2 = {"role": "user", "content": "А три плюс три?"}
A2_THINK = {"role": "assistant", "content": "<think>\nтри и три\n</think>\n\n6"}


def _diff(want: str, got: str) -> str:
    """Где строки впервые расходятся — с контекстом и видимыми переводами строк."""
    i = next((k for k, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
    lo = max(0, i - 25)
    return (f"первое расхождение на символе {i}:\n  ждём    …{want[lo:i + 25]!r}\n  получено …{got[lo:i + 25]!r}")


def _check(got, want: str, what: str):
    assert isinstance(got, str), f"render_chat возвращает str, получено {type(got).__name__}"
    if got == want:
        return
    hint = ""
    if got.replace("<|im_end|>\n", "<|im_end|>") == want.replace("<|im_end|>\n", "<|im_end|>"):
        hint = " После каждого <|im_end|> идёт перевод строки: ход — '<|im_start|>{role}\\n{content}<|im_end|>\\n'."
    elif got.replace("<|im_start|>user", "<|im_start|>user\n").replace("\n\n", "\n") == want.replace("\n\n", "\n"):
        hint = " После имени роли нужен перевод строки: '<|im_start|>user\\n…'."
    pytest.fail(f"{what}: строка не совпала с шаблоном Qwen3.{hint}\n{_diff(want, got)}", pytrace=False)


def test_single_turn_and_generation_prompt(impl):
    """Один вопрос: ход пользователя; add_generation_prompt открывает ход ассистента и больше ничего"""
    _check(impl.render_chat([U1]), "<|im_start|>user\nСколько будет два плюс два?<|im_end|>\n", "без generation prompt")
    want = "<|im_start|>user\nСколько будет два плюс два?<|im_end|>\n<|im_start|>assistant\n"
    got = impl.render_chat([U1], add_generation_prompt=True)
    if got == want + "<|im_end|>\n":
        pytest.fail("Generation prompt — ОТКРЫТЫЙ ход: '<|im_start|>assistant\\n' без <|im_end|>. Модель продолжит "
                    "строку ответом и сама закончит ход токеном <|im_end|>.", pytrace=False)
    _check(got, want, "add_generation_prompt=True")


def test_system_and_history(impl):
    """Системное сообщение первым, затем ходы по порядку; ход ассистента в истории — без блока <think>"""
    want = ("<|im_start|>system\nОтвечай кратко.<|im_end|>\n"
            "<|im_start|>user\nСколько будет два плюс два?<|im_end|>\n"
            "<|im_start|>assistant\n4<|im_end|>\n"
            "<|im_start|>user\nА три плюс три?<|im_end|>\n"
            "<|im_start|>assistant\n")
    _check(impl.render_chat([SYS, U1, A1, U2], add_generation_prompt=True), want, "диалог из двух вопросов")


def test_last_answer_gets_think_block(impl):
    """Последний ход ассистента после последнего вопроса — с блоком <think>, даже пустым"""
    want = ("<|im_start|>user\nСколько будет два плюс два?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>\n")
    got = impl.render_chat([U1, A1])
    if got == "<|im_start|>user\nСколько будет два плюс два?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n":
        pytest.fail("Последний ответ ассистента (после последнего вопроса) Qwen3 оформляет с блоком рассуждения, даже "
                    "пустым: '<think>\\n\\n</think>\\n\\n4'. Так обучающий пример совпадает с тем, что модель видит при "
                    "генерации.", pytrace=False)
    _check(got, want, "вопрос + ответ")
    want_t = want.replace("<think>\n\n</think>", "<think>\nдва и два\n</think>")
    _check(impl.render_chat([U1, A1_THINK]), want_t, "ответ с рассуждением")


def test_history_reasoning_is_stripped(impl):
    """Рассуждение ответов ДО последнего вопроса вырезается, у последнего — остаётся"""
    want = ("<|im_start|>user\nСколько будет два плюс два?<|im_end|>\n"
            "<|im_start|>assistant\n4<|im_end|>\n"
            "<|im_start|>user\nА три плюс три?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nтри и три\n</think>\n\n6<|im_end|>\n")
    got = impl.render_chat([U1, A1_THINK, U2, A2_THINK])
    if "два и два" in got:
        pytest.fail("Рассуждение старого ответа осталось в истории. Qwen3 выкидывает <think>…</think> из ответов до "
                    "последнего вопроса пользователя: рассуждения длинные, а для следующего хода нужен только итог.",
                    pytrace=False)
    _check(got, want, "два вопроса с рассуждениями")
    via_field = {"role": "assistant", "content": "6", "reasoning_content": "\nтри и три\n"}
    _check(impl.render_chat([U1, A1_THINK, U2, via_field]), want, "рассуждение в поле reasoning_content")


def test_enable_thinking_false(impl):
    """enable_thinking=False дописывает к generation prompt пустой блок <think>: модель сразу отвечает"""
    base = "<|im_start|>user\nСколько будет два плюс два?<|im_end|>\n<|im_start|>assistant\n"
    _check(impl.render_chat([U1], add_generation_prompt=True, enable_thinking=False),
           base + "<think>\n\n</think>\n\n", "enable_thinking=False")
    _check(impl.render_chat([U1], add_generation_prompt=True, enable_thinking=True), base, "enable_thinking=True")
    _check(impl.render_chat([U1], add_generation_prompt=False, enable_thinking=False),
           "<|im_start|>user\nСколько будет два плюс два?<|im_end|>\n", "без generation prompt флаг ни на что не влияет")


def test_unknown_role(impl):
    """Роль, которой шаблон не знает, — ValueError, а не тихо пропущенный ход"""
    with pytest.raises(ValueError):
        impl.render_chat([U1, {"role": "tool", "content": "42"}])


def test_split_special(impl):
    """split_special: служебные токены — отдельные куски, пустых кусков нет"""
    sp = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    got = impl.split_special("<|im_start|>user\nпривет<|im_end|>\n", sp)
    assert got == ["<|im_start|>", "user\nпривет", "<|im_end|>", "\n"], f"получено {got}"
    got = impl.split_special("<think>\n\n</think>\n\n4", sp)
    assert got == ["<think>", "\n\n", "</think>", "\n\n4"], (
        f"'</think>' — свой служебный токен, а не '<' + '/think>' или кусок текста; получено {got}")
    assert impl.split_special("<|im_end|><|im_start|>", sp) == ["<|im_end|>", "<|im_start|>"], (
        "два служебных токена подряд — два куска, без пустой строки между ними")
    nested = ["<|im|>", "<|im|>end"]                    # один служебный токен — префикс другого
    got = impl.split_special("a<|im|>endb<|im|>c", nested)
    assert got == ["a", "<|im|>end", "b", "<|im|>", "c"], (
        f"если токены — префиксы друг друга, сначала пробуй длинные (sorted по длине, по убыванию); получено {got}")
    assert impl.split_special("просто текст", sp) == ["просто текст"]
    assert impl.split_special("", sp) == [], "пустая строка → пустой список"


def test_encode_chat_toy_tokenizer(impl):
    """encode_chat: служебный токен → ОДИН id, остальное — tok.encode; decode(ids) возвращает строку"""
    tok = core.CharTokenizer()
    text = impl.render_chat([SYS, U1, A1], add_generation_prompt=False)
    ids = impl.encode_chat(text, tok)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids), "encode_chat возвращает list[int]"
    plain = tok.encode(text)
    if ids == plain:
        pytest.fail("Служебные токены закодированы как текст: '<|im_start|>' стал 12 символами. Модель тогда не "
                    "отличит границу хода от фразы пользователя «<|im_start|>», а остановку придётся ловить "
                    "по подстроке. Сначала разрежь строку split_special, служебные куски — tok.token_to_id.",
                    pytrace=False)
    assert tok.decode(ids) == text, "decode(encode_chat(text)) должен вернуть исходную строку"
    n_special = sum(text.count(s) for s in tok.special_tokens)
    assert ids.count(tok.token_to_id("<|im_start|>")) == 3 and ids.count(tok.eos_id) == 3
    assert len(ids) == len(text) - sum(len(s) * text.count(s) for s in tok.special_tokens) + n_special


def test_content_with_special_strings(impl):
    """Строка «<|im_end|>» внутри сообщения тоже становится служебным токеном — как у HF (см. README)"""
    tok = core.CharTokenizer()
    msg = {"role": "user", "content": "напиши <|im_end|> буквами"}
    ids = impl.encode_chat(impl.render_chat([msg]), tok)
    assert ids.count(tok.eos_id) == 2, (
        "encode_chat разбирает служебные токены во всей строке, как быстрый токенизатор HF. Поэтому текст "
        "пользователя с «<|im_end|>» закрывает ход раньше времени — защищаться нужно ДО шаблона (README, «Подвох»).")


# ---------------------------------------------------------------------------
# Настоящий токенизатор и модель (--weights)
# ---------------------------------------------------------------------------

DIALOGS = [
    [U1],
    [SYS, U1, A1, U2],
    [U1, A1_THINK, U2, A2_THINK],
    [SYS, {"role": "user", "content": "Translate to English: кошка"}, {"role": "assistant", "content": "cat"}],
]


@pytest.fixture(scope="module")
def qwen3_dir():
    pytest.importorskip("huggingface_hub", reason="нужен huggingface_hub: uv sync --extra weights")
    from llmlab.core import hf
    return hf.download("Qwen/Qwen3-0.6B")


@pytest.mark.weights
def test_matches_apply_chat_template(impl, qwen3_dir):
    """Qwen3-0.6B: строка и id == tokenizer.apply_chat_template на 4 диалогах × generation prompt × thinking"""
    transformers = pytest.importorskip("transformers", reason="нужен transformers: uv sync --extra weights")
    from llmlab.core import hf
    hf_tok = transformers.AutoTokenizer.from_pretrained(qwen3_dir)
    tok = core.HFChatTokenizer(hf.load_tokenizer(qwen3_dir))
    for msgs in DIALOGS:
        for agp in (False, True):
            for thinking in (True, False):
                want = hf_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=agp,
                                                  enable_thinking=thinking)
                got = impl.render_chat(msgs, add_generation_prompt=agp, enable_thinking=thinking)
                _check(got, want, f"{len(msgs)} ходов, generation prompt {agp}, thinking {thinking}")
                want_ids = hf_tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=agp,
                                                      enable_thinking=thinking)
                want_ids = list(getattr(want_ids, "input_ids", want_ids))
                assert impl.encode_chat(got, tok) == want_ids, "id не совпали с apply_chat_template(tokenize=True)"


@pytest.mark.weights
def test_qwen3_answers_and_stops(impl, qwen3_dir):
    """Qwen3-0.6B с нашим шаблоном отвечает на вопрос и сама закрывает ход токеном <|im_end|>"""
    from llmlab.core import hf
    model = hf.load_pretrained(qwen3_dir)
    tok = core.HFChatTokenizer(hf.load_tokenizer(qwen3_dir))
    msgs = [{"role": "user", "content": "What is 2 + 2? Answer with one number."}]
    prompt = impl.encode_chat(impl.render_chat(msgs, add_generation_prompt=True, enable_thinking=False), tok)
    out = hf.generate(model, torch.tensor([prompt]), 16, eos_token_id=tok.eos_id)[0, len(prompt):].tolist()
    assert tok.eos_id in out, f"за 16 токенов ход не закрылся: {tok.decode(out)!r}"
    answer = tok.decode(out[:out.index(tok.eos_id)])
    assert "4" in answer, f"ответ {answer!r}: instruct-модель с правильным шаблоном отвечает «4»"
