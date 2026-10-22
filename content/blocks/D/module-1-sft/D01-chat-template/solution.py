"""Chat-шаблон: список ходов диалога → строка со служебными токенами ролей → id.

Эталон. Правила — как у шаблона Qwen3-0.6B (``tokenizer_config.json``, без tools): строка совпадает с
``tokenizer.apply_chat_template(messages, tokenize=False, ...)``, id — с ``tokenize=True``.
"""

from __future__ import annotations

import re


def render_chat(messages: list[dict], add_generation_prompt: bool = False, enable_thinking: bool = True) -> str:
    """Список ходов → строка chat-шаблона Qwen3 (без tools) == ``tokenizer.apply_chat_template(..., tokenize=False)``.

    Каждый ход — ``<|im_start|>{role}\\n{content}<|im_end|>\\n``. У ассистента рассуждение ``<think>…</think>``
    вырезается из ходов ДО последнего вопроса пользователя; последний ход ассистента после этого вопроса получает
    блок ``<think>\\n{рассуждение}\\n</think>\\n\\n`` (пустой, если рассуждения нет).
    """
    last_query = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            last_query = i
            break
    out = []
    for i, msg in enumerate(messages):
        role, content = msg["role"], msg["content"]
        if role in ("system", "user"):
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        elif role == "assistant":
            reasoning = msg.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = ""
                if "</think>" in content:
                    reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    content = content.split("</think>")[-1].lstrip("\n")
            if i > last_query and (i == len(messages) - 1 or reasoning):
                think, answer = reasoning.strip("\n"), content.lstrip("\n")
                out.append(f"<|im_start|>assistant\n<think>\n{think}\n</think>\n\n{answer}")
            else:
                out.append(f"<|im_start|>assistant\n{content}")
            out.append("<|im_end|>\n")
        else:
            raise ValueError(f"роль {role!r} не поддержана: только system, user, assistant")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
        if not enable_thinking:
            out.append("<think>\n\n</think>\n\n")
    return "".join(out)


def split_special(text: str, special_tokens: list[str]) -> list[str]:
    """Разрезать строку по служебным токенам: служебный токен — отдельный кусок, пустых кусков нет."""
    if not special_tokens:
        return [text] if text else []
    pattern = "(" + "|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)) + ")"
    return [piece for piece in re.split(pattern, text) if piece]


def encode_chat(text: str, tok) -> list[int]:
    """Строка шаблона → id: служебный токен — один id (``tok.token_to_id``), текст между ними — ``tok.encode``."""
    ids: list[int] = []
    for piece in split_special(text, tok.special_tokens):
        if piece in tok.special_tokens:
            ids.append(tok.token_to_id(piece))
        else:
            ids.extend(tok.encode(piece))
    return ids
